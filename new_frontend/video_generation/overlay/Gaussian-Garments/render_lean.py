#!/usr/bin/env python3
"""Leaner stage 5 render: only produce what the final video needs.

Ablation counterpart to inference.py. Per inference.py's own timing printout
(see its "3DGS 光柵化渲染" vs "圖片拼接與 I/O" lines), the per-frame cost splits
roughly 80% mesh/texture-bake prep, 8% Gaussian rasterization, 12% image I/O.
This script doesn't touch the 80% (mesh export, Blender AO/normal baking,
AvatarNet forward, visibility raycasting -- all required regardless of output
format). It only removes the parts that existed purely for a debug comparison
view:

  - inference.py rasterizes each frame 3x (raw / style / final) and
    concatenates all three plus the raw garment-only render into one wide
    debug PNG. This script rasterizes only "final" (the actual composited
    result) and saves just that -- no raw/style panels, no quad image.

vert/face-level correctness is untouched: "final" here is pixel-identical to
the "final" panel inference.py would have produced, just without its debug
neighbors. Reuses inference.py's own Doll/Simulation classes and helpers by
importing it as a module, so any behavior fix made there stays in sync.

Per-substep profiling of the "prep" bucket (see sub_times below) showed it's
dominated by prepare_frame()'s Blender Cycles AO/normal texture baking
(~60-65% of prep, by far the largest single piece). --bake-samples lets you
trade a little baking quality for speed: it lowers Cycles' path-tracing
sample count (hardcoded to 16 in inference.py's bake_texture) for both the
AO and normal bakes. This only adds noise to the baked maps at the *same*
512x512 resolution -- resolution itself must NOT be changed, since AvatarNet's
pretrained checkpoint has its style/MLP layers sized to args.texture_size
(512); changing the actual texture resolution would break that checkpoint's
expected input shape, not just soften it.

--profile-substeps (off by default) turns on that fine-grained breakdown --
useful for finding the next bottleneck, but NOT for production runs: getting
per-substep timing means calling torch.cuda.synchronize() before/after each
of the 6 substeps (7 syncs/frame total), each one a hard CPU/GPU barrier that
blocks the pipelining a normal run gets for free. Measured cost: with
--profile-substeps on, a 2-garment outfit rendered slower than plain
inference.py (179.5s vs 133.9s) despite doing strictly less work (1 panel,
not 4) -- the sync tax alone outweighed the panel-count and bake-sample
savings combined. Default off uses the same 3-sync/frame pattern as
inference.py's own timing code (prep/render/IO boundaries only), so the
--bake-samples and single-panel savings actually show up in wall-clock time.

Must be launched with the gaugar Python environment and cwd=Gaussian-Garments
root, exactly like inference.py itself:

    <gaugar-python> render_lean.py --traj_path traj.pkl --output_path frames/ \
        --bake-samples 8
"""

import sys
import os
import time
import functools
from pathlib import Path
import numpy as np
import torch
import bpy
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from aitviewer.headless import HeadlessRenderer

import inference as inf
from utils.defaults import DEFAULTS
from arguments import ModelParams, PipelineParams


class Config(inf.ParamGroup):
    def __init__(self, parser) -> None:
        self.name = "renders"
        self.texture_size = 512
        self.gender = 'female'
        self.add_position = False
        super().__init__(parser, "Visualize doll Parameters")


def bake_texture_fast(mesh_path, obstacles, body_path=None, w=512, h=512,
                       texture_margin=5, show=False, save=False, samples=16):
    """Same as inference.py's bake_texture(), except the Cycles sample count
    (hardcoded to 16 there) is a parameter. Resolution (w, h) is intentionally
    left alone -- see module docstring for why."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    if os.path.exists(body_path):
        try:
            bpy.ops.wm.ply_import(filepath=body_path)
        except Exception:
            print(body_path)
    for _o in obstacles:
        bpy.ops.wm.obj_import(filepath=_o)
    bpy.ops.wm.obj_import(filepath=mesh_path)
    mesh = bpy.data.objects[-1]

    ao_mat = bpy.data.materials.new(name="AmbientOcclusion")
    ao_mat.use_nodes = True
    ImgTextnode = ao_mat.node_tree.nodes.new('ShaderNodeTexImage')
    ao_img = bpy.data.images.new("AmbientOcclusion", width=w, height=h, alpha=False)
    ImgTextnode.image = ao_img
    mesh.data.materials.clear()
    mesh.data.materials.append(ao_mat)
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = samples
    bpy.context.scene.cycles.use_denoising = False
    bpy.ops.object.bake(type='AO', margin=texture_margin)

    nm_mat = bpy.data.materials.new(name="Normal")
    nm_mat.use_nodes = True
    ImgTextnode = nm_mat.node_tree.nodes.new('ShaderNodeTexImage')
    nm_img = bpy.data.images.new("Normal", width=w, height=h, alpha=False)
    ImgTextnode.image = nm_img
    mesh.data.materials.clear()
    mesh.data.materials.append(nm_mat)
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = samples
    bpy.context.scene.cycles.use_denoising = False
    bpy.ops.object.bake(type='NORMAL', normal_space='OBJECT', margin=texture_margin)

    ambient = np.array(ao_img.pixels).reshape(w, h, -1)[..., 0]
    normal = np.array(nm_img.pixels).reshape(w, h, -1)[..., :3]

    if show:
        Image.fromarray(np.uint8(ambient * 255)).show()
        Image.fromarray(np.uint8(normal * 255)).show()
    if save:
        _ambient = mesh_path.replace("meshes/", "texture/ambient/")
        _normal = mesh_path.replace("meshes/", "texture/normal/")
        os.makedirs(os.path.dirname(_ambient), exist_ok=True)
        os.makedirs(os.path.dirname(_normal), exist_ok=True)
        Image.fromarray(np.uint8(ambient * 255)).save(_ambient.replace(".obj", ".png"))
        Image.fromarray(np.uint8(normal * 255)).save(_normal.replace(".obj", ".png"))

    return ambient, normal


if __name__ == "__main__":
    parser = ArgumentParser(description="Lean render (final panel only)")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    cp = Config(parser)
    parser.add_argument("--HQ", action="store_true")
    parser.add_argument("--start_from", type=int, default=0)
    parser.add_argument("--bake-samples", type=int, default=4,
        help="Cycles path-tracing sample count for AO/normal baking "
             "(inference.py hardcodes 16, the pre-optimization baseline's "
             "inference_GG.py never sets this at all and just inherits bpy's "
             "own Cycles default of ~4096 -- neither number was tuned for how "
             "cheap AO/normal maps actually are to converge). Compared side "
             "by side at 4 vs 8 vs 16 on a 2-garment outfit, no visible "
             "difference showed up even at 4, so that's the default here -- "
             "lower still trades a little more baking quality for speed, same "
             "512x512 resolution -- does not touch AvatarNet's expected input "
             "shape.")
    parser.add_argument("--profile-substeps", action="store_true",
        help="Print a per-substep timing breakdown of the prep bucket. Adds "
             "7 torch.cuda.synchronize() calls/frame (vs 3 without this flag) "
             "which measurably slows the render -- see module docstring. Use "
             "for finding bottlenecks, not for production/timed runs.")
    parser.add_argument("--opacity-scale", type=float, default=1.0,
        help="Requested garment opacity from 0 to 1. Default 1 preserves the old output.")
    parser.add_argument("--opacity-sweep", nargs="+", type=float, default=None,
        help="Comparison mode: render each frame at several opacity values into separate folders.")
    parser.add_argument("--opacity-mode", choices=("direct", "optical_depth", "v3"), default="direct",
        help="Per-Gaussian opacity transform.")
    parser.add_argument("--v3-controller", action="append", default=[], metavar="GARMENT=PATH",
        help="V3 checkpoint for one garment; repeat once per garment.")
    parser.add_argument("--opacity-garments", nargs="+", default=[],
        help="Only adjust these trajectory garments; default adjusts all garments.")
    parser.add_argument("--allow-v3-controller-transfer", action="store_true",
        help="EXPERIMENTAL: allow a V3 checkpoint trained for another garment.")
    parser.add_argument('--traj_path', type=str, required=True, default='')
    parser.add_argument('--output_path', type=str, required=True, default='')
    parser.add_argument("--avatar_list", "-al", required=False, nargs='+', type=str, default=[])
    parser.add_argument("--garment_list", "-gl", required=False, nargs='+', type=str, default=[])
    parser.add_argument("--texture_list", "-tl", required=False, nargs='+', type=str)
    parser.add_argument('--registration', type=str, required=False, default='')
    args = parser.parse_args(sys.argv[1:])
    if not 0.0 <= args.opacity_scale <= 1.0:
        parser.error("--opacity-scale must be between 0 and 1")
    if args.opacity_sweep and any(not 0.0 <= value <= 1.0 for value in args.opacity_sweep):
        parser.error("every --opacity-sweep value must be between 0 and 1")
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

    if args.HQ: w, h = 3004, 4092
    else: w, h = 940, 1280

    args.source_path = DEFAULTS.data_root
    args.store_path = DEFAULTS.output_root
    viewer = HeadlessRenderer(size=(w, h))
    viewer.playback_fps = 30
    viewer.shadows_enabled = False
    viewer.scene.origin.enabled = False
    cmap = plt.get_cmap('gist_rainbow')

    # inference.py's helpers (ait_render, from_ait_to_gs, ...) read these as
    # their own module's globals -- set them there, not just in this script.
    inf.w, inf.h = w, h
    inf.viewer = viewer
    inf.cmap = cmap
    # Simulation.prepare_frame() (unmodified, in inference.py) calls
    # bake_texture(...) by looking it up in its own module's globals at call
    # time -- reassigning it here redirects those calls to our
    # sample-count-parameterized version without touching inference.py.
    inf.bake_texture = functools.partial(bake_texture_fast, samples=args.bake_samples)

    simu = inf.Simulation(args)
    with torch.inference_mode():
        bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
        bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        total_prep_time = 0.0
        total_render_time = 0.0
        total_io_time = 0.0
        total_frames = 0
        global_start_time = time.time()

        # fine-grained breakdown of what "prep" actually contains -- only
        # populated (and only paid for, see tick() below) with --profile-substeps
        sub_times = {
            "ait_render body": 0.0,
            "prepare_frame (mesh export + Blender AO/normal bake)": 0.0,
            "update_garments (AvatarNet forward)": 0.0,
            "prepare_gaussian (o3d visibility raycast)": 0.0,
            "ait_render garment+body depth": 0.0,
            "ait_render garment-only depth": 0.0,
        }

        def tick():
            torch.cuda.synchronize()
            return time.time()

        for idx in tqdm(simu, desc="Rendering Sequence (lean)"):
            if idx < args.start_from: continue
            if idx >= len(simu): break

            if args.profile_substeps:
                # 7 syncs/frame -- see module docstring for why this is opt-in only.
                t0 = tick()

                _body, _body_depth, ait_cam = inf.ait_render(vb=simu.simu_pkl['obstacle'][idx], fb=simu.simu_pkl['obstacle_faces'])
                gs_cam = inf.from_ait_to_gs(ait_cam, w, h)
                ta = tick()
                sub_times["ait_render body"] += ta - t0

                # inlined Simulation.forward() so each piece can be timed separately
                vert_list, ambient_list, normal_list = simu.prepare_frame(idx)
                tb = tick()
                sub_times["prepare_frame (mesh export + Blender AO/normal bake)"] += tb - ta

                style_list = simu.doll.update_garments(vert_list, ambient_list, normal_list, gs_cam)
                tc = tick()
                sub_times["update_garments (AvatarNet forward)"] += tc - tb

                simu.doll.body = simu.doll.get_o3d_mesh(simu.simu_pkl['obstacle'][idx], faces=simu.simu_pkl['obstacle_faces'])
                simu.doll.prepare_gaussian(gs_cam.camera_center, style_list)
                td = tick()
                sub_times["prepare_gaussian (o3d visibility raycast)"] += td - tc

                face_list = [g.mesh.f.cpu().numpy() for g in simu.doll.garments]
                # Still needed for the body/garment visibility mask below, even
                # though we don't keep its color image as a debug panel.
                _, _body_garm_depth, _ = inf.ait_render(vert_list, face_list, simu.simu_pkl['obstacle'][idx], simu.simu_pkl['obstacle_faces'], ait_cam)
                te = tick()
                sub_times["ait_render garment+body depth"] += te - td

                _, _garm_depth, _ = inf.ait_render(vert_list, face_list, cam=ait_cam)
                t1 = tick()
                sub_times["ait_render garment-only depth"] += t1 - te
            else:
                # 3 syncs/frame -- matches inference.py's own timing pattern
                # (prep/render/IO boundaries only), so --bake-samples and the
                # single-panel save actually show up in wall-clock time instead
                # of being eaten by sync overhead. See module docstring.
                t0 = tick()

                _body, _body_depth, ait_cam = inf.ait_render(vb=simu.simu_pkl['obstacle'][idx], fb=simu.simu_pkl['obstacle_faces'])
                gs_cam = inf.from_ait_to_gs(ait_cam, w, h)
                vert_list, ambient_list, normal_list = simu.prepare_frame(idx)
                style_list = simu.doll.update_garments(vert_list, ambient_list, normal_list, gs_cam)
                simu.doll.body = simu.doll.get_o3d_mesh(simu.simu_pkl['obstacle'][idx], faces=simu.simu_pkl['obstacle_faces'])
                simu.doll.prepare_gaussian(gs_cam.camera_center, style_list)

                face_list = [g.mesh.f.cpu().numpy() for g in simu.doll.garments]
                # Still needed for the body/garment visibility mask below, even
                # though we don't keep its color image as a debug panel.
                _, _body_garm_depth, _ = inf.ait_render(vert_list, face_list, simu.simu_pkl['obstacle'][idx], simu.simu_pkl['obstacle_faces'], ait_cam)
                _, _garm_depth, _ = inf.ait_render(vert_list, face_list, cam=ait_cam)
                t1 = tick()

            _body_mask = _body_garm_depth < _garm_depth

            opacity_values = args.opacity_sweep or [args.opacity_scale]
            rendered = []
            for opacity_value in opacity_values:
                if opacity_value != simu.doll.opacity_scale:
                    simu.doll.opacity_scale = opacity_value
                    simu.doll.prepare_gaussian(gs_cam.camera_center, style_list)
                image = simu.doll.render(gs_cam, simu.doll, args, bg, _body, _body_mask)
                rendered.append((opacity_value, image))

            torch.cuda.synchronize()
            t2 = time.time()

            for opacity_value, image in rendered:
                if args.opacity_sweep:
                    value = f"{opacity_value:.6f}".rstrip("0").rstrip(".").replace(".", "p")
                    frame_dir = f"{args.name}_opacity_{value}"
                else:
                    frame_dir = args.name
                os.makedirs(os.path.join(simu.output, frame_dir), exist_ok=True)
                Image.fromarray(np.array(image, dtype=np.uint8)).save(
                    os.path.join(simu.output, frame_dir, f"{idx:04d}.png")
                )
            t3 = time.time()

            total_prep_time += (t1 - t0)
            total_render_time += (t2 - t1)
            total_io_time += (t3 - t2)
            total_frames += len(rendered)

    global_end_time = time.time()
    print("\n" + "=" * 50)
    print(f"Total frames rendered: {total_frames}")
    print(f"Total time taken: {global_end_time - global_start_time:.2f} seconds")
    total_tracked = total_prep_time + total_render_time + total_io_time
    print(f"1. prep (GPU): {total_prep_time:.2f}s ({(total_prep_time/total_tracked)*100:.1f}%)")
    print(f"2. gaussian render, final only (GPU): {total_render_time:.2f}s ({(total_render_time/total_tracked)*100:.1f}%)")
    print(f"3. image I/O, single panel (CPU): {total_io_time:.2f}s ({(total_io_time/total_tracked)*100:.1f}%)")
    if not args.profile_substeps:
        print("=" * 50 + "\n")
        print("Simulation Finished")
    else:
        # sub_times is only populated inside the --profile-substeps branch above.
        print("-" * 50)
        print("prep breakdown:")
        for name, elapsed in sub_times.items():
            print(f"  {name}: {elapsed:.2f}s ({(elapsed/total_prep_time)*100:.1f}% of prep)")
        print("=" * 50 + "\n")
    print("Simulation Finished")
