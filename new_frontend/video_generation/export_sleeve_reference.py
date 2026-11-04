#!/usr/bin/env python3
"""Export the minimal same-topology sleeve reference; never relax/unpose a mesh.

Use a known pre-relax/pre-untangle dict, NOT the already shortened untangle output.
The reference is a private data asset, separate from the Git source branch.
"""
import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-garment-dict", type=Path, required=True)
    ap.add_argument("--source-fit", type=Path, required=True)
    ap.add_argument("--output-reference", type=Path, required=True)
    args = ap.parse_args()
    if args.output_reference.exists() or args.output_reference.is_symlink():
        ap.error("Output already exists; choose a separate reference filename")
    if args.output_reference.resolve() in (args.input_garment_dict.resolve(), args.source_fit.resolve()):
        ap.error("Reference output must not overwrite an input")
    with args.input_garment_dict.open("rb") as f:
        g = pickle.load(f)
    with args.source_fit.open("rb") as f:
        fit = pickle.load(f)
    betas = np.asarray(fit["betas"]).reshape(-1)
    if betas.shape != (10,) or not np.isfinite(betas).all():
        ap.error("Source fit must contain exactly 10 finite betas")
    ref = {"faces": np.array(g["faces"], copy=True), "lbs": {
        k: np.array(g["lbs"][k], copy=True) for k in ("v", "shapedirs", "lbs_weights")},
        "reference_metadata": {"source_betas": betas, "source_gender": fit.get("gender"),
            "input_sha256": hashlib.sha256(args.input_garment_dict.read_bytes()).hexdigest(),
            "source_fit_sha256": hashlib.sha256(args.source_fit.read_bytes()).hexdigest(),
            "geometry": "provided pre-relax/pre-untangle canonical geometry; not reconstructed by this exporter"}}
    args.output_reference.parent.mkdir(parents=True, exist_ok=True)
    with args.output_reference.open("xb") as f:
        pickle.dump(ref, f, protocol=4)
    print(f"Reference: {args.output_reference} ({args.output_reference.stat().st_size} bytes)")
    print("SHA256:", hashlib.sha256(args.output_reference.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
