#!/usr/bin/env python3
"""Report an A08 cuff-region proxy over a saved rollout (no simulation/render).

Distances are to a tracked SMPL-X wrist surface patch, NOT garment ground truth.
A fixed-size oversized garment can have a large offset without a binding error.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from repair_sleeve_lbs import array, load_dict, validate_garment


def main():
    import torch
    import smplx
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trajectory", type=Path, required=True)
    ap.add_argument("--garment-dict", type=Path, required=True)
    ap.add_argument("--reference-garment-dict", type=Path, required=True)
    ap.add_argument("--body-model-root", type=Path, required=True)
    ap.add_argument("--garment-name", required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--max-distance-cm", type=float, help="Optional experiment-specific gate; none by default")
    args = ap.parse_args()
    if args.report.exists() or args.report.is_symlink():
        ap.error("Report already exists; choose a new report")
    if args.max_distance_cm is not None and args.max_distance_cm <= 0:
        ap.error("max-distance-cm must be positive")
    g, ref, t = load_dict(args.garment_dict), load_dict(args.reference_garment_dict), load_dict(args.trajectory)
    validate_garment(g, ref)
    gender = g["gender"]
    if args.garment_name not in t["garment_names"]:
        ap.error("Garment is not present in trajectory.garment_names")
    index = t["garment_names"].index(args.garment_name)
    ids = np.asarray(t["garment_id"]).reshape(-1)
    garment_mask = ids == index
    if garment_mask.sum() != len(g["lbs"]["v"]):
        ap.error("Trajectory vertex count/order differs from this garment dictionary")
    pred = np.asarray(t["pred"])[:, garment_mask]
    body_frames = np.asarray(t["obstacle"])
    if len(pred) != len(body_frames) or not np.isfinite(pred).all() or not np.isfinite(body_frames).all():
        ap.error("Nonfinite or inconsistent trajectory frames")
    model = smplx.create(str(args.body_model_root), model_type="smplx", gender=gender, use_pca=False)
    if not np.array_equal(np.asarray(t["obstacle_faces"]), model.faces):
        ap.error("Obstacle topology/order is not the expected SMPL-X body")
    with torch.no_grad():
        result = model(betas=torch.as_tensor(array(g["fixed_size_metadata"]["target_betas"])[None], dtype=torch.float32))
    body, joints = result.vertices[0].numpy(), result.joints[0].numpy()
    reference_v = array(ref["lbs"]["v"])
    report = {"gender": gender, "garment": args.garment_name, "frames": len(pred),
              "metric": "cuff candidate centroid to posed wrist surface-patch centroid; not GT", "sides": []}
    for side, sign, wrist in (("left", 1, 20), ("right", -1, 21)):
        x = reference_v[:, 0] * sign
        cuff = x > np.quantile(x, .96)
        wrist_ids = np.argsort(np.linalg.norm(body - joints[wrist], axis=1))[:48]
        distance = np.linalg.norm(pred[:, cuff].mean(1) - body_frames[:, wrist_ids].mean(1), axis=1) * 100
        report["sides"].append(dict(side=side, initial_cm=float(distance[0]), final_cm=float(distance[-1]),
            max_cm=float(distance.max()), mean_cm=float(distance.mean()), max_frame=int(distance.argmax())))
    report["gate_cm"] = args.max_distance_cm
    report["gate_passed"] = None if args.max_distance_cm is None else all(
        s["max_cm"] <= args.max_distance_cm for s in report["sides"])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if report["gate_passed"] is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
