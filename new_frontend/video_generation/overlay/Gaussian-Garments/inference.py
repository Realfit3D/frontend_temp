from pathlib import Path
import sys
import os
import bpy
import copy
import functools
import json
import glob
import pickle
import torch
from torch.utils.data import Dataset
import time

import trimesh
import open3d as o3d
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from gaussian_renderer import doll_render
from matplotlib import pyplot as plt
from utils.defaults import DEFAULTS
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from arguments import ParamGroup
from scene.cameras import Camera, get_cam_info
from scene import MeshGaussianModel
from scene.avatar_gaussian_model import AvatarSimulationModel
from scene.avatar_net import AvatarNet
from opacity_controller_v3_common import load_v3
from utils.graphics_utils import focal2fov
from utils.io_utils import read_obj, write_obj
from aitviewer.headless import HeadlessRenderer
from aitviewer.scene.camera import PinholeCamera
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.meshes import VariableTopologyMeshes as VTMeshes
from aitviewer.scene.camera import OpenCVCamera
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def get_aitviewer_cam(cam: dict):
    cam_intrinsics = np.array(cam['intrinsics'])
    cam_extrinsics = np.eye(4)
    cam_extrinsics[:3] = np.array(cam['extrinsics'])

    # # The OpenCVCamera class expects extrinsics with Y pointing down, so we flip both Y and Z axis to keep a
    # # positive determinant.
    # cam_extrinsics[1:3, :3] *= -1.0

    # Create an OpenCV camera.
    camera = OpenCVCamera(cam_intrinsics, cam_extrinsics[:3], w, h, viewer=viewer)
    return camera

def from_gs_to_ait(cam):
    extrinsic, intrinsic = np.zeros([3,4]), np.eye(3)
    extrinsic[:, :3] = cam.R.T
    extrinsic[:, 3] = cam.T

    intrinsic[0, 0], intrinsic[1, 1] = cam.fx, cam.fy
    intrinsic[0, 2], intrinsic[1, 2] = cam.cx, cam.cy

    h, w = cam.original_image.shape[1:]

    return OpenCVCamera(intrinsic, extrinsic, w, h, viewer=viewer)


def from_ait_to_gs(cam, w, h):
    cam = cam.to_opencv_camera() if hasattr(cam, 'to_opencv_camera') else cam
    intrinsic = cam.K[0]
    extrinsic = cam.Rt[0]
    
    R, T = np.transpose(extrinsic[:, :3]), extrinsic[:, 3]
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[:2, 2]
    FovY, FovX = focal2fov(fy, h), focal2fov(fx, w)

    return Camera(R=R, T=T, FoVx=FovX, FoVy=FovY, fx=fx, fy=fy, cx=cx, cy=cy, colmap_id=None, image_name=None, uid=None,
                    image=torch.zeros([3,h,w]), gt_alpha_mask=torch.zeros([1,h,w]), data_device='cuda')

def adjust_color(num):
    """
    Adjusts color to look nice.
    :param color: [..., 4] RGBA color
    :return: updated color
    """
    color = np.array(cmap(num))
    color[..., :3] /= color[..., :3].max(axis=-1, keepdims=True)
    color[..., :3] *= 0.3
    color[..., :3] += 0.3
    return tuple(color)

def scan_render(vtmesh, cam=None):
    for l in viewer.scene.lights: l.enabled = False
    viewer.scene.nodes = viewer.scene.nodes[:4]
    viewer.scene.add(vtmesh)

    if cam is None:
        # horizontal view
        target = viewer.scene.nodes[-1].current_center
        position = target + np.array([0,0,3])
        camera = PinholeCamera(position, target, w, h, viewer=viewer)
    else:
        if isinstance(cam, Camera):
            target = viewer.scene.nodes[-1].current_center
            position = cam.T #  + (target - cam.T) * 0.2
            camera = PinholeCamera(position, target, w, h, viewer=viewer)
        else:
            camera = cam

    viewer.set_temp_camera(camera)
    image = np.array(viewer.get_frame())
    depth = np.array(viewer.get_depth())
    return torch.tensor(image), torch.tensor(depth), camera

def ait_render(verts=None, faces=None, vb=None, fb=None, cam=None):
    '''
    v, f : np.array
    '''
    for l in viewer.scene.lights: l.enabled = True
    viewer.scene.nodes = viewer.scene.nodes[:4]

    if verts is not None and faces is not None:
        for v, f in zip(verts, faces):
            mesh = Meshes(v, f, color=adjust_color(len(viewer.scene.nodes)*0.1), draw_edges=False)
            mesh.backface_culling = False
            viewer.scene.add(mesh)

    if vb is not None and fb is not None:
        body = Meshes(vb, fb, draw_edges=False)
        body.backface_culling = False
        viewer.scene.add(body)

    if cam is None:
        # horizontal view
        target = viewer.scene.nodes[-1].current_center
        position = target + np.array([0,0,3])
        camera = PinholeCamera(position, target, w, h, viewer=viewer)
    else:
        if isinstance(cam, Camera):
            # target = viewer.scene.nodes[-1].current_center # + np.array([0,0.3,0])
            # position = cam.T # + (target - cam.T) * 0.5
            # camera = PinholeCamera(position, target, w, h, viewer=viewer)
            camera = from_gs_to_ait(cam).to_pinhole_camera()
        else:
            camera = cam

    viewer.set_temp_camera(camera)
    image = np.array(viewer.get_frame())
    depth = np.array(viewer.get_depth())
    return torch.tensor(image), torch.tensor(depth), camera


def bake_texture(mesh_path, obstacles,  body_path=None, w=512, h=512, texture_margin=5, show=False, save=False, samples=16):
    '''
    Bake normal map and ambient occlusion map for given mesh
    Output:
        ambient: np.array(w, h)
        normal: np.array(w, h, 3)
    '''
    # Remove all elements
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # load mesh
    if os.path.exists(body_path):
        try:
            bpy.ops.wm.ply_import(filepath=body_path)
        except:
            print(body_path)
    for _o in obstacles:
        bpy.ops.wm.obj_import(filepath=_o)
    bpy.ops.wm.obj_import(filepath=mesh_path)
    mesh = bpy.data.objects[-1]

    # init ambient occlusion material && texture
    ao_mat = bpy.data.materials.new(name="AmbientOcclusion")
    ao_mat.use_nodes = True
    ImgTextnode = ao_mat.node_tree.nodes.new('ShaderNodeTexImage')
    ao_img = bpy.data.images.new("AmbientOcclusion", width=w, height=h, alpha=False)
    ImgTextnode.image = ao_img
    # assign material to mesh
    mesh.data.materials.clear()
    mesh.data.materials.append(ao_mat)     
    # Set up context for baking
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.object.mode_set(mode='OBJECT')
    # Set render engine and device
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'

    #壓榨 Cycles 引擎效能
    bpy.context.scene.cycles.samples = samples
    bpy.context.scene.cycles.use_denoising = False

    # Bake ambient occlusion texture
    bpy.ops.object.bake(type='AO', margin=texture_margin)


    # init normal material && texture
    nm_mat = bpy.data.materials.new(name="Normal")
    nm_mat.use_nodes = True
    ImgTextnode = nm_mat.node_tree.nodes.new('ShaderNodeTexImage')
    nm_img = bpy.data.images.new("Normal", width=w, height=h, alpha=False)
    ImgTextnode.image = nm_img
    # assign material to mesh
    mesh.data.materials.clear()
    mesh.data.materials.append(nm_mat)     
    # Set up context for baking
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.mode_set(mode='OBJECT')
    # Set render engine and device
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'

    #壓榨 Cycles 引擎效能
    bpy.context.scene.cycles.samples = samples
    bpy.context.scene.cycles.use_denoising = False

    # Bake ambient occlusion texture
    bpy.ops.object.bake(type='NORMAL', normal_space='OBJECT', margin=texture_margin)

    # output
    ambient = np.array(ao_img.pixels).reshape(w,h,-1)[...,0]
    normal = np.array(nm_img.pixels).reshape(w,h,-1)[...,:3]

    # Convert the NumPy array to PIL image, viualize/save
    if show:
        Image.fromarray(np.uint8(ambient * 255)).show()
        Image.fromarray(np.uint8(normal * 255)).show()
    if save:
        _ambient = mesh_path.replace("meshes/","texture/ambient/")
        _normal = mesh_path.replace("meshes/","texture/normal/")
        os.makedirs(os.path.dirname(_ambient), exist_ok=True)
        os.makedirs(os.path.dirname(_normal), exist_ok=True)

        Image.fromarray(np.uint8(ambient * 255)).save(_ambient.replace(".obj",".png"))
        Image.fromarray(np.uint8(normal * 255)).save(_normal.replace(".obj",".png"))

    return ambient, normal

class Doll(MeshGaussianModel):
    def __init__(self, args, garment_names) -> None:
        super(MeshGaussianModel, self).__init__(args.sh_degree)
        self.active_sh_degree = args.sh_degree
        opacity_scale = float(getattr(args, "opacity_scale", 1.0))
        opacity_mode = getattr(args, "opacity_mode", "direct")
        opacity_sweep = getattr(args, "opacity_sweep", None)
        needs_v3_controller = (
            any(0.0 < float(value) < 1.0 for value in opacity_sweep)
            if opacity_sweep
            else 0.0 < opacity_scale < 1.0
        )
        v3_controller_map = getattr(args, "v3_controller_map", {})
        allow_v3_controller_transfer = bool(
            getattr(args, "allow_v3_controller_transfer", False)
        )
        opacity_garments = set(getattr(args, "opacity_garments", []) or garment_names)
        self.garment_names = list(garment_names)
        unknown_opacity_targets = opacity_garments - set(self.garment_names)
        if unknown_opacity_targets:
            raise ValueError(
                "Opacity targets are not in the trajectory: "
                + ", ".join(sorted(unknown_opacity_targets))
            )

        # get garment gaussian (temporal version --> to include simulator)
        self.garments = []
        self.avatar_nets = []
        self.opacity_controllers = []
        self.opacity_controller_checkpoints = []
        # for _avatar, _tmp in zip(args.avatar_list, args.garment_list):
        for garment_name in garment_names:
            _tmp = Path(DEFAULTS.output_root) / garment_name
            gaussian = AvatarSimulationModel(_tmp, texture_size=args.texture_size)
            self.garments.append(gaussian)
            # V3 was trained with the rest-pose Gaussian coordinates. Keep a
            # frozen copy because get_xyz changes with the simulated mesh.
            gaussian.v3_canonical_xyz = gaussian.get_xyz.detach().clone()
            avatar = AvatarNet(args, gaussian).to('cuda')

            stage3_path = _tmp / DEFAULTS.stage3
            avatar.load_ckpt(stage3_path, load_optm=False)
            self.avatar_nets.append(avatar)

            controller = checkpoint = None
            if (garment_name in opacity_garments and opacity_mode == "v3"
                    and needs_v3_controller):
                controller_path = v3_controller_map.get(garment_name)
                if controller_path is None:
                    candidates = [
                        _tmp / "opacity_controller_v3" / "controller.pt",
                        _tmp / "opacity_controller_v3_valid_cameras" / "controller.pt",
                    ]
                    controller_path = next((path for path in candidates if path.is_file()), None)
                if controller_path is None:
                    raise FileNotFoundError(
                        f"No V3 opacity controller found for {garment_name}. "
                        "Pass --v3-controller GARMENT=/absolute/path/controller.pt."
                    )
                controller, checkpoint = load_v3(controller_path)
                checkpoint_subject = checkpoint.get("config", {}).get("subject")
                if checkpoint_subject and checkpoint_subject != garment_name:
                    message = (
                        f"V3 controller subject mismatch: trajectory requests {garment_name}, "
                        f"but {controller_path} declares {checkpoint_subject}"
                    )
                    if not allow_v3_controller_transfer:
                        raise RuntimeError(message)
                    print(f"WARNING: experimental zero-shot controller transfer enabled: {message}")
                center = checkpoint.get("canonical_center")
                scale = checkpoint.get("canonical_scale")
                if center is None or scale is None or tuple(center.shape) != (3,) or tuple(scale.shape) != (3,):
                    raise RuntimeError(f"Invalid canonical normalization in {controller_path}")
                if not torch.isfinite(center).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
                    raise RuntimeError(f"Non-finite or non-positive canonical normalization in {controller_path}")
                print(f"Loaded V3 opacity controller for {garment_name}: {controller_path}")
            self.opacity_controllers.append(controller)
            self.opacity_controller_checkpoints.append(checkpoint)
        self.body = None
        self.opacity_scale = opacity_scale
        self.opacity_mode = opacity_mode
        self.opacity_garments = opacity_garments

    def update_garments(self, vert_list, ambient_list, normal_list, camera):
        style_list = []
        for i, vert in enumerate(vert_list):
            avatar, garment = self.avatar_nets[i], self.garments[i]
            garment.mesh.v = torch.tensor(vert, dtype=torch.float32).cuda()
            garment.update_face_coor()

            # forward avatar_net
            style_output, _ = avatar(ambient_list[i], normal_list[i], camera)
            style_list.append(style_output)
            
        return style_list

    def get_o3d_mesh(self, verts, faces):
        mesh = o3d.t.geometry.TriangleMesh()
        mesh.vertex.positions = o3d.core.Tensor(verts, o3d.core.float32)
        mesh.triangle.indices = o3d.core.Tensor(faces, o3d.core.int32)
        return mesh
        
    def prepare_gaussian(self, p_cam, style_list):
        # xyz = torch.cat([g.get_xyz for g in self.garments])
        xyz = torch.cat([g.get_barycentric_3d() for g in self.garments])
        bindings = torch.cat([g.binding for g in self.garments])
        vis_mask = self.get_visible_mask(p_cam, xyz, bindings)

        self.xyz = torch.cat([g.get_final_xyz for g in self.garments])[vis_mask]
        self.rotation = torch.cat([g.get_rotation for g in self.garments])[vis_mask]
        self.features = torch.cat([g.shs for g in self.garments])[vis_mask]
        self.raw_features = torch.cat([g.get_features for g in self.garments])[vis_mask]
        self.style_features = torch.cat([g for g in style_list])[vis_mask]
        opacity_chunks = []
        for index, garment in enumerate(self.garments):
            base_opacity = garment.get_opacity
            garment_name = self.garment_names[index]
            if garment_name not in self.opacity_garments:
                controlled_opacity = base_opacity
            elif self.opacity_scale <= 0.0:
                controlled_opacity = torch.zeros_like(base_opacity)
            elif self.opacity_scale >= 1.0:
                controlled_opacity = base_opacity
            elif self.opacity_mode == "direct":
                controlled_opacity = base_opacity * self.opacity_scale
            elif self.opacity_mode == "optical_depth":
                controlled_opacity = -torch.expm1(
                    base_opacity.new_tensor(self.opacity_scale)
                    * torch.log1p(-base_opacity.clamp(0.0, 1.0 - 1e-7))
                )
            elif self.opacity_mode == "v3":
                checkpoint = self.opacity_controller_checkpoints[index]
                center = checkpoint["canonical_center"].to(base_opacity)
                scale = checkpoint["canonical_scale"].to(base_opacity)
                canonical_xyz = (garment.v3_canonical_xyz - center) / scale
                controlled_opacity = self.opacity_controllers[index](
                    base_opacity, canonical_xyz, self.opacity_scale
                )[0]
            else:
                raise RuntimeError(f"Unsupported opacity mode: {self.opacity_mode}")
            opacity_chunks.append(controlled_opacity)
        self.opacity = torch.cat(opacity_chunks)[vis_mask]
        self.scaling = torch.cat([g.get_scaling for g in self.garments])[vis_mask]
        self.covariance = torch.cat([g.get_covariance() for g in self.garments])[vis_mask]

    def get_visible_mask(self, camera, xyz, binding, dot_product_t=-0.7):
        '''
        compute the mask, for filtering gaussians that are visible
        params:
            @camera: torch.tensor(3,)
        output:
            @vis_mask: torch.tensor(num_gs)
        '''
        device = camera.device
        # init scene
        scene = o3d.t.geometry.RaycastingScene()
        for g in self.garments:
            scene.add_triangles(self.get_o3d_mesh(g.mesh.v.cpu().numpy(), g.mesh.f.cpu().numpy()))
        # scene.add_triangles(self.body)
        geometry_ids = torch.cat([i * torch.ones(g._xyz.shape[0], dtype=int) for i, g in enumerate(self.garments)])
        num_gs = len(self.garments) + 1

        # cast ray from camera to GS
        ray_o = camera[None].repeat(len(xyz),1)
        _d = xyz - ray_o
        norm_d = _d.norm(dim=1, keepdim=True)
        ray_d = _d / norm_d
        rays = o3d.core.Tensor(torch.cat([ray_o, ray_d], axis=-1).detach().cpu().numpy())

        # Compute the ray intersections.
        EPSILON = 5e-2 # max distance between intersection point and gaussians
        ans = scene.cast_rays(rays)
        # mesh id based mask
        vis_mask = (ans['geometry_ids'].numpy() == np.array(geometry_ids))
        vis_mask += (ans['geometry_ids'].numpy() >= num_gs)
        vis_mask = torch.tensor(vis_mask)
        return vis_mask.to(device)
        
    def render(self, viewpoint_camera, pc, args, bg, body_image, _body_mask, override_shs=None):
        image, depth, alpha = doll_render(viewpoint_camera, pc, args, bg, override_shs=override_shs)
        image = torch.clip(image, min=0, max=1).permute(1,2,0).detach().cpu()*255
        alpha = alpha.permute(1,2,0).detach().cpu()*255
        # garment image
        garment = torch.cat([image, alpha], dim=-1) * ~_body_mask.unsqueeze(-1)
        garment = Image.fromarray(np.array(garment, dtype=np.uint8))
        # body image
        body = torch.cat([body_image, torch.ones_like(alpha)*255], dim=-1)
        body= Image.fromarray(np.array(body, dtype=np.uint8))
        
        output = Image.alpha_composite(body, garment)
        return np.array(output)[...,:3]
        
class Simulation(Dataset):
    def __init__(self, args):
        # simulation and templates
        self.simu_pkl = pickle.load(open(args.traj_path, 'rb'))

        # MODIFICATION
        if len(self.simu_pkl["garment_names"]) == 0:
            filename = args.traj_path.split("/")[-1]
            filename = filename.replace(".pkl", "")
            only_garment_name = '_'.join(filename.split("_")[:2])
            self.simu_pkl["garment_names"] = [only_garment_name, ]

        self.garment_names = self.simu_pkl['garment_names']

        # self.garment_names = ["00122_outer"]

        models_root = Path(DEFAULTS.output_root)
        uv_tmp_path = [models_root / name / DEFAULTS.stage1 / 'template_uv.obj' for name in self.garment_names]
        self.uv_tmp_list = [read_obj(path) for path in uv_tmp_path]
        # camera info
        cam_params_path = models_root / self.garment_names[0] / DEFAULTS.stage1 / 'cameras.json'
        cam_params = json.load(open(cam_params_path, 'r'))
        self.gs_cams = {k:get_cam_info(v) for k,v in cam_params.items()}
        self.ait_cams = {k:get_aitviewer_cam(v) for k,v in cam_params.items()}

        # Construct Gaussian Doll
        self.doll = Doll(args, self.garment_names)

        # Simulation output path
        self.output = args.output_path
        os.makedirs(self.output, exist_ok=True)

    def __len__(self,):
        return len(self.simu_pkl['pred'])

    def __getitem__(self, index):
        return index
    
    def get_vertices(self, vertices):
        output = []
        for i, uv_mesh in enumerate(self.uv_tmp_list):
            output.append(vertices[:len(uv_mesh['vertices'])])
            try: vertices = vertices[len(uv_mesh['vertices']):]
            except: pass
        return output

    def prepare_frame(self, idx):
            vert_list = self.get_vertices(self.simu_pkl['pred'][idx])

            # store body
            os.makedirs(os.path.join(self.output, 'body'), exist_ok=True)
            _body = os.path.join(self.output, 'body', f'{idx:05d}.ply')
            if not os.path.exists(_body): 
                body = trimesh.Trimesh(vertices=self.simu_pkl['obstacle'][idx], faces=self.simu_pkl['obstacle_faces'], process=False)
                body.export(_body)
                
            # store meshes
            _mesh_list = [os.path.join(self.output, _g, "meshes", f"{idx:05d}.obj") for _g in self.garment_names]
            for i, _mesh in enumerate(_mesh_list):
                if not os.path.exists(_mesh):
                    # store garments
                    os.makedirs(os.path.dirname(_mesh), exist_ok=True)
                    output = copy.deepcopy(self.uv_tmp_list[i])
                    assert len(output['vertices']) == len(vert_list[i]), "Num of Vertices mismatch"
                    output['vertices'] = vert_list[i]
                    write_obj(output, _mesh)

            ambient_list, normal_list = [], []
            for _mesh in _mesh_list:
                # bake textures
                _ambient = _mesh.replace("meshes/","texture/ambient/").replace(".obj",".png")
                _normal = _mesh.replace("meshes/","texture/normal/").replace(".obj",".png")
                if not (os.path.exists(_ambient) and os.path.exists(_normal)):
                    obstacle_meshes = [_m for _m in _mesh_list if _m != _mesh]
                    ambient, normal = bake_texture(_mesh,  obstacle_meshes, _body, save=True)
                else:
                    ambient = np.array(Image.open(_ambient)) / 255.
                    normal = np.array(Image.open(_normal)) / 255.
                ambient = torch.tensor(ambient, dtype=torch.float32).unsqueeze(0).cuda()
                normal = torch.tensor(normal, dtype=torch.float32).permute(2,0,1).cuda()
                ambient_list.append(ambient)
                normal_list.append(normal)
                
            return vert_list, ambient_list, normal_list
    
    def forward(self, index, camera):
        vert_list, ambient_list, normal_list = self.prepare_frame(index)
        # update garments and body
        style_list = self.doll.update_garments(vert_list, ambient_list, normal_list, camera)
        self.doll.body = self.doll.get_o3d_mesh(self.simu_pkl['obstacle'][index], faces=self.simu_pkl['obstacle_faces'])
        # select visible gaussians
        self.doll.prepare_gaussian(camera.camera_center, style_list)



class Config(ParamGroup):
    def __init__(self, parser) -> None:
        self.name = "renders"
        self.texture_size = 512
        self.gender = 'female'
        self.add_position = False
        super().__init__(parser, "Visualize doll Parameters")


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    cp = Config(parser)
    parser.add_argument("--HQ", action="store_true")
    parser.add_argument("--start_from", type=int, default=0)
    parser.add_argument("--bake-samples", type=int, default=16,
        help="Cycles path-tracing sample count for AO/normal texture baking "
             "(previously hardcoded to 16).")
    parser.add_argument("--opacity-scale", type=float, default=1.0,
        help="Requested garment opacity from 0 to 1. Default 1 preserves the old output.")
    parser.add_argument("--opacity-mode", choices=("direct", "optical_depth", "v3"), default="direct",
        help="Per-Gaussian opacity transform.")
    parser.add_argument("--v3-controller", action="append", default=[], metavar="GARMENT=PATH",
        help="V3 checkpoint for one garment; repeat once per garment.")
    parser.add_argument("--opacity-garments", nargs="+", default=[],
        help="Only adjust these trajectory garments; default adjusts all garments.")
    parser.add_argument("--allow-v3-controller-transfer", action="store_true",
        help="EXPERIMENTAL: allow a V3 checkpoint trained for another garment.")
    parser.add_argument('--gt_mesh', type=str, required=False, default='')
    parser.add_argument('--traj_path', type=str, required=True, default='')
    parser.add_argument('--output_path', type=str, required=True, default='')
    parser.add_argument("--avatar_list", "-al", required=False, nargs='+', type=str, default=[], help="path to avatar model folder")
    parser.add_argument("--garment_list", "-gl", required=False, nargs='+', type=str, default=[], help="template for garments")
    parser.add_argument("--texture_list", "-tl", required=False, nargs='+', type=str, help="path to texture pkl")
    parser.add_argument('--registration', type=str, required=False, default='')
    args = parser.parse_args(sys.argv[1:])
    if not 0.0 <= args.opacity_scale <= 1.0:
        parser.error("--opacity-scale must be between 0 and 1")
    args.v3_controller_map = {}
    for item in args.v3_controller:
        if "=" not in item:
            parser.error("--v3-controller must use GARMENT=PATH")
        garment_name, controller_path = item.split("=", 1)
        if not garment_name or not controller_path:
            parser.error("--v3-controller must use GARMENT=PATH")
        if garment_name in args.v3_controller_map:
            parser.error(f"Duplicate --v3-controller entry for {garment_name}")
        resolved = Path(controller_path).expanduser().resolve()
        if not resolved.is_file():
            parser.error(f"--v3-controller file does not exist: {resolved}")
        args.v3_controller_map[garment_name] = resolved
    # path processing

    if args.HQ: w, h = 3004, 4092
    else: w, h = 940, 1280

    args.source_path = DEFAULTS.data_root
    args.store_path = DEFAULTS.output_root
    viewer = HeadlessRenderer(size=(w, h))
    viewer.playback_fps = 30
    viewer.shadows_enabled = False
    viewer.scene.origin.enabled = False
    cmap = plt.get_cmap('gist_rainbow')

    args.output_path = args.output_path
    # args.avatar_list = [os.path.join(args.scratch_path, 'Registration_artur_44', path) for path in args.avatar_list]

    # Simulation.prepare_frame() looks up bake_texture as this module's own
    # global at call time, so rebinding the name here (same trick render_lean.py
    # uses from outside) redirects it to use --bake-samples without touching
    # prepare_frame() itself.
    bake_texture = functools.partial(bake_texture, samples=args.bake_samples)

    # init objects
    simu = Simulation(args)
    # with torch.no_grad():
    with torch.inference_mode():
        bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
        bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Time reocrding
        total_prep_time = 0.0
        total_render_time = 0.0
        total_io_time = 0.0
        total_frames = 0
        global_start_time = time.time()

        for idx in tqdm(simu, desc="Rendering Sequence"):
            if idx < args.start_from: continue
            if idx >= len(simu): break

            # record time
            torch.cuda.synchronize()
            t0 = time.time()

            # body mesh
            _body, _body_depth, ait_cam = ait_render(vb=simu.simu_pkl['obstacle'][idx], fb=simu.simu_pkl['obstacle_faces'])
            gs_cam = from_ait_to_gs(ait_cam, w, h)
            simu.forward(idx, gs_cam)
            # garment mesh
            vert_list = [g.mesh.v.cpu().numpy() for g in simu.doll.garments]
            face_list = [g.mesh.f.cpu().numpy() for g in simu.doll.garments]
            _garment, _body_garm_depth, _ = ait_render(vert_list, face_list, simu.simu_pkl['obstacle'][idx], simu.simu_pkl['obstacle_faces'], ait_cam)
            _, _garm_depth, _ = ait_render(vert_list, face_list, cam=ait_cam)
            _body_mask = _body_garm_depth < _garm_depth

            torch.cuda.synchronize()
            t1 = time.time()


            # final
            _final = simu.doll.render(gs_cam, simu.doll, args, bg, _body, _body_mask)
            # raw gaussian
            _raw = simu.doll.render(gs_cam, simu.doll, args, bg, _body, _body_mask, simu.doll.raw_features)
            # style output
            _style = simu.doll.render(gs_cam, simu.doll, args, bg, _body, _body_mask, simu.doll.style_features)
            
            torch.cuda.synchronize()
            t2 = time.time()

            image = np.concatenate([_raw, _style, _final, _garment], axis=1)
            os.makedirs(os.path.join(simu.output, f"{args.name}"), exist_ok=True)
            Image.fromarray(np.array(image, dtype=np.uint8)).save(os.path.join(simu.output, f"{args.name}/{idx:04d}.png"))
            t3 = time.time()

            # Accumlate time
            total_prep_time += (t1 - t0)
            total_render_time += (t2 - t1)
            total_io_time += (t3 - t2)
            total_frames += 1
    # Showing time statistics
    global_end_time = time.time()
    print("\n" + "="*50)
    print(f"Total frames rendered: {total_frames}")
    print(f"Total time taken: {global_end_time - global_start_time:.2f} seconds")

    total_tracked = total_prep_time + total_render_time + total_io_time
    print(f"1. 前置與光影推論 (GPU): {total_prep_time:.2f} 秒 ({(total_prep_time/total_tracked)*100:.1f}%)")
    print(f"2. 3DGS 光柵化渲染 (GPU): {total_render_time:.2f} 秒 ({(total_render_time/total_tracked)*100:.1f}%)")
    print(f"3. 圖片拼接與 I/O (CPU) : {total_io_time:.2f} 秒 ({(total_io_time/total_tracked)*100:.1f}%)")
    print("-" * 50)
    print("【平均單幀耗時】")
    print(f"平均每幀前置推論: {total_prep_time/total_frames:.4f} 秒")
    print(f"平均每幀光柵渲染: {total_render_time/total_frames:.4f} 秒")
    print(f"平均每幀 I/O 寫入: {total_io_time/total_frames:.4f} 秒")
    print("="*50 + "\n")

    print("Simulation Finished")
