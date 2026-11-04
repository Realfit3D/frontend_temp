#!/usr/bin/env python3
"""Reference-guided, anatomical-region garment initialization repair (SMPL-X).

Default CLI operation diagnoses only. An explicit region profile and --apply
are required to write a repaired dict. Does not change rest mesh or simulation.
"""
import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from repair_sleeve_lbs import array, load_dict, validate_garment, preview

# Anatomical membership, not world-coordinate crop boxes. SMPL-X 55-joint order.
REGIONS = {
    "left_arm": ([13, 16, 18, 20] + list(range(25, 40)), (16, 18, 20)),
    "right_arm": ([14, 17, 19, 21] + list(range(40, 55)), (17, 19, 21)),
    "left_leg": ([1, 4, 7, 10], (1, 4, 7)),
    "right_leg": ([2, 5, 8, 11], (2, 5, 8)),
    "torso": ([0, 3, 6, 9, 12, 15], (0, 3, 6, 9, 12)),
}
DEFAULTS = {
    "mode": "restore-rebind", "regions": [],
    "membership_threshold": .45, "body_membership_threshold": .5,
    "damage_fraction": .10, "max_displacement_fraction": .75,
    "max_binding_distance_fraction": .5, "max_edge_strain": .35,
    "clearance_m": 0.0,
}


def validate_profile(config):
    if not isinstance(config, dict) or set(config) - set(DEFAULTS):
        raise ValueError("Profile must contain only: " + ", ".join(DEFAULTS))
    cfg = {**DEFAULTS, **config}
    if cfg["mode"] not in ("restore-rebind", "rebind"):
        raise ValueError("mode must be restore-rebind or rebind")
    if (not isinstance(cfg["regions"], list) or
            any(r not in REGIONS for r in cfg["regions"]) or
            len(set(cfg["regions"])) != len(cfg["regions"])):
        raise ValueError("regions must be unique anatomical region names: " + ", ".join(REGIONS))
    for key in set(DEFAULTS) - {"mode", "regions"}:
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        if not 0 < value <= 1 and key != "clearance_m":
            raise ValueError(f"{key} must be in (0, 1]")
    if not 0 <= cfg["clearance_m"] <= .03:
        raise ValueError("clearance_m must be in [0, .03]")
    if cfg["membership_threshold"] >= 1:
        raise ValueError("membership_threshold must be less than 1")
    return cfg


def nearest(points, candidates):
    """Nearest vertex transfer with a bounded-memory NumPy fallback."""
    if not len(candidates):
        raise ValueError("No anatomically matching target body vertices")
    try:
        from scipy.spatial import cKDTree
        return cKDTree(candidates).query(points)
    except ImportError:
        distances, indices = [], []
        for start in range(0, len(points), 32):
            d2 = np.sum((points[start:start + 32, None] - candidates[None]) ** 2, axis=2)
            ids = d2.argmin(1)
            indices.extend(ids.tolist())
            distances.extend(np.sqrt(d2[np.arange(len(ids)), ids]).tolist())
        return np.asarray(distances), np.asarray(indices, dtype=int)


def edge_strain(vertices, reference, faces, selected):
    edges = np.unique(np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]],
                                             faces[:, [2, 0]])), axis=1), axis=0)
    edges = edges[selected[edges].any(1)]
    lengths = np.linalg.norm(reference[edges[:, 0]] - reference[edges[:, 1]], axis=1)
    valid = lengths > 1e-7
    actual = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    return float(np.max(np.abs(actual[valid] / lengths[valid] - 1))) if valid.any() else 0.


def project_clearance(vertices, selected, body, faces, clearance):
    if not clearance or not selected.any():
        return vertices
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(body.astype(np.float32)),
                        o3d.core.Tensor(faces.astype(np.uint32)))
    v = vertices.copy()
    for _ in range(4):
        sd = scene.compute_signed_distance(o3d.core.Tensor(v.astype(np.float32))).numpy()
        bad = selected & (sd < clearance)
        if not bad.any():
            break
        result = scene.compute_closest_points(o3d.core.Tensor(v[bad].astype(np.float32)))
        direction = v[bad] - result["points"].numpy()
        direction[sd[bad] < 0] *= -1
        length = np.linalg.norm(direction, axis=1)
        valid = length > 1e-7
        direction[valid] /= length[valid, None]
        direction[~valid] = result["primitive_normals"].numpy()[~valid]
        v[bad] += direction * (clearance - sd[bad])[:, None]
    remaining = scene.compute_signed_distance(o3d.core.Tensor(v[selected].astype(np.float32))).numpy()
    if np.any(remaining < clearance - 1e-4):
        raise ValueError("Requested body clearance could not be reached")
    return v


def repair_with_body(input_dict, reference, body, body_faces, joints, body_weights, body_posedirs,
                     config, apply=False):
    """Pure repair core; caller supplies the personalized canonical body arrays.

    Every anatomical mask is derived from the trusted reference, never the
    potentially damaged input binding. A failed gate rejects the whole candidate.
    """
    cfg = validate_profile(config)
    validate_garment(input_dict, reference)
    if apply and not cfg["regions"]:
        raise ValueError("Apply requires explicitly reviewed regions")
    body, joints, body_weights, body_posedirs = map(array, (body, joints, body_weights, body_posedirs))
    if (body.ndim != 2 or body.shape[1] != 3 or body_weights.shape != (len(body), 55)
            or body_posedirs.shape != (486, len(body) * 3) or joints.shape[0] < 55):
        raise ValueError("Expected canonical SMPL-X body geometry and 55-joint skinning")
    if not all(np.isfinite(x).all() for x in (body, joints, body_weights, body_posedirs)):
        raise ValueError("Non-finite target body")
    ref_lbs = reference["lbs"]
    ref = array(ref_lbs["v"]) + np.einsum("vck,k->vc", array(ref_lbs["shapedirs"]),
                                         array(input_dict["fixed_size_metadata"]["source_betas"]))
    ref_weights = array(ref_lbs["lbs_weights"])
    if (not np.isfinite(ref).all() or not np.isfinite(ref_weights).all() or
            np.any(ref_weights < 0) or not np.allclose(ref_weights.sum(1), 1, atol=1e-4)):
        raise ValueError("Invalid canonical reference vertices/weights")
    original = array(input_dict["lbs"]["v"])
    vertices = original.copy()
    faces = array(input_dict["faces"])
    chosen = np.zeros(len(vertices), bool)
    assigned = np.full(len(vertices), -1, dtype=int)
    scale_at = np.ones(len(vertices))
    rows, rejected = [], []
    # Winner-takes-membership prevents contradictory left/right restoration.
    membership = np.stack([ref_weights[:, ids].sum(1) for ids, _ in REGIONS.values()], axis=1)
    winner = membership.argmax(1)
    for region_index, (name, (joint_ids, chain)) in enumerate(REGIONS.items()):
        scale = float(sum(np.linalg.norm(joints[b] - joints[a]) for a, b in zip(chain, chain[1:])))
        if scale < 1e-5:
            raise ValueError(f"Degenerate anatomical chain: {name}")
        mass = membership[:, region_index]
        mask = (winner == region_index) & (mass > cfg["membership_threshold"])
        row = dict(region=name, vertices=int(mask.sum()), chain_length_m=scale,
                   requested=name in cfg["regions"], proposed=False)
        if not mask.any():
            if row["requested"]:
                rejected.append(f"{name}: no matching reference vertices")
            rows.append(row)
            continue
        delta = np.linalg.norm(original[mask] - ref[mask], axis=1) / scale
        row["reference_deviation_p90_fraction"] = float(np.quantile(delta, .9))
        row["reference_deviation_max_fraction"] = float(delta.max())
        candidates = np.flatnonzero(body_weights[:, joint_ids].sum(1) >= cfg["body_membership_threshold"])
        distance, local = nearest(original[mask], body[candidates])
        row["binding_distance_before_p95_fraction"] = float(np.quantile(distance / scale, .95))
        row["binding_region_agreement_before"] = float(np.mean(
            array(input_dict["lbs"]["lbs_weights"])[mask][:, joint_ids].sum(1)))
        if not row["requested"]:
            rows.append(row)
            continue
        if cfg["mode"] == "restore-rebind":
            if row["reference_deviation_p90_fraction"] <= cfg["damage_fraction"]:
                row["reason"] = "below restoration trigger; kept unchanged"
                rows.append(row)
                continue
            # Smoothly blend by trusted skinning membership near region boundaries.
            t = np.clip((mass - cfg["membership_threshold"]) / (1 - cfg["membership_threshold"]), 0, 1)
            blend = t * t * (3 - 2 * t)
            vertices[mask] += (ref[mask] - vertices[mask]) * blend[mask, None]
        chosen |= mask
        scale_at[mask] = scale
        assigned[mask] = candidates[local]
        row["proposed"] = True
        rows.append(row)
    try:
        vertices = project_clearance(vertices, chosen, body, body_faces, cfg["clearance_m"])
    except ValueError as exc:
        rejected.append(str(exc))
    for row, (name, (joint_ids, _)) in zip(rows, REGIONS.items()):
        if not row["proposed"]:
            continue
        mask = chosen & (winner == list(REGIONS).index(name))
        candidates = np.flatnonzero(body_weights[:, joint_ids].sum(1) >= cfg["body_membership_threshold"])
        distance, local = nearest(vertices[mask], body[candidates])
        assigned[mask] = candidates[local]
        row["binding_distance_after_max_fraction"] = float((distance / scale_at[mask]).max())
        row["max_displacement_fraction"] = float((np.linalg.norm(vertices[mask] - original[mask], axis=1) / scale_at[mask]).max())
        if row["binding_distance_after_max_fraction"] > cfg["max_binding_distance_fraction"]:
            rejected.append(f"{name}: target body too far from candidate")
        if row["max_displacement_fraction"] > cfg["max_displacement_fraction"]:
            rejected.append(f"{name}: excessive initialization displacement")
    strain = edge_strain(vertices, ref, faces, chosen)
    if strain > cfg["max_edge_strain"]:
        rejected.append("Excessive edge distortion relative to source reference")
    # Reject flipped local triangles; this is not a self-intersection proof.
    tri_ref, tri_new = ref[faces], vertices[faces]
    nr = np.cross(tri_ref[:, 1] - tri_ref[:, 0], tri_ref[:, 2] - tri_ref[:, 0])
    nn = np.cross(tri_new[:, 1] - tri_new[:, 0], tri_new[:, 2] - tri_new[:, 0])
    flipped = chosen[faces].any(1) & (np.sum(nr * nn, axis=1) <= 0) & (np.linalg.norm(nr, axis=1) > 1e-10)
    if flipped.any():
        rejected.append("Candidate has reversed or collapsed reference triangles")
    report = dict(profile="anatomical_reference_v1_experimental", config=cfg, regions=rows,
                  gates_passed=not rejected, rejection_reasons=rejected,
                  proposed_vertices=int(chosen.sum()), applied=bool(apply and not rejected),
                  max_edge_strain=strain, flipped_triangles=int(flipped.sum()),
                  rest_pos_unchanged=True, topology_unchanged=True)
    result = copy.deepcopy(input_dict)
    if report["applied"]:
        ids = assigned[chosen]
        weights = array(result["lbs"]["lbs_weights"]).copy()
        pd = array(result["lbs"]["posedirs"]).reshape(486, -1, 3).copy()
        weights[chosen] = body_weights[ids]
        pd[:, chosen] = body_posedirs.reshape(486, -1, 3)[:, ids]
        result["lbs"]["v"] = vertices.astype(original.dtype)
        result["lbs"]["lbs_weights"] = weights
        result["lbs"]["posedirs"] = pd.reshape(486, -1)
        result["garment_repair"] = copy.deepcopy(report)
        validate_garment(result)
    geometry = (body, body_faces, original, vertices, faces)
    return result, report, geometry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-garment-dict", type=Path, required=True)
    parser.add_argument("--reference-garment-dict", type=Path, required=True)
    parser.add_argument("--body-model-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--output-garment-dict", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if args.apply != bool(args.output_garment_dict):
        parser.error("Use --apply and --output-garment-dict together; omit both to diagnose")
    paths = [p for p in (args.report, args.preview, args.output_garment_dict) if p]
    inputs = {p.resolve() for p in (args.input_garment_dict, args.reference_garment_dict, args.profile)}
    if len({p.resolve() for p in paths}) != len(paths):
        parser.error("Outputs must have distinct paths")
    if any(p.exists() or p.is_symlink() or p.resolve() in inputs for p in paths):
        parser.error("Never overwrite an input or an existing output; use a new run directory")
    config = validate_profile(json.loads(args.profile.read_text(encoding="utf-8")))
    g, reference = load_dict(args.input_garment_dict), load_dict(args.reference_garment_dict)
    validate_garment(g, reference)
    gender = g.get("gender")
    if gender not in ("male", "female") or g["fixed_size_metadata"].get("target_gender", gender) != gender:
        parser.error("Input gender/target gender mismatch")
    import torch
    import smplx
    model = smplx.create(str(args.body_model_root), model_type="smplx", gender=gender, use_pca=False)
    with torch.no_grad():
        output = model(betas=torch.as_tensor(array(g["fixed_size_metadata"]["target_betas"])[None], dtype=torch.float32))
    result, report, geometry = repair_with_body(g, reference, output.vertices[0].numpy(), model.faces,
        output.joints[0].numpy(), model.lbs_weights.detach().numpy(), model.posedirs.detach().numpy(), config, args.apply)
    report["target_gender"] = gender
    for key, path in (("input", args.input_garment_dict), ("reference", args.reference_garment_dict), ("profile", args.profile)):
        report[key + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    if args.preview:
        preview(args.preview, geometry, label="Candidate: " + ("gates passed" if report["gates_passed"] else "REJECTED"),
                full_body=True)
    if args.apply:
        if not report["gates_passed"]:
            raise SystemExit("Candidate rejected; inspect report. No repaired dictionary written.")
        result["garment_repair"] = copy.deepcopy(report)
        with args.output_garment_dict.open("xb") as handle:
            pickle.dump(result, handle, protocol=4)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
