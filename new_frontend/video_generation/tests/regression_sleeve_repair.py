#!/usr/bin/env python3
"""Compare portable repair against a saved, validated experiment (CPU only).

No data/model paths are built in. Reports array error; does not simulate/render.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "overlay/ContourCraft"))
from repair_sleeve_lbs import repair, load_dict, array


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--expected", type=Path, required=True)
    ap.add_argument("--body-model-root", type=Path, required=True)
    ap.add_argument("--gender", choices=("male", "female"), required=True)
    ap.add_argument("--mode", choices=("restore-rebind", "radial"), required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    if args.report.exists():
        ap.error("Choose a new report path")
    inputs = [p for p in (args.input, args.reference, args.expected) if p is not None]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    original, expected = load_dict(args.input), load_dict(args.expected)
    actual, report, _ = repair(original, load_dict(args.reference) if args.reference else None,
                              args.body_model_root, args.gender, args.mode)
    errors = {}
    for field in ("v", "lbs_weights", "posedirs", "shapedirs"):
        delta = np.abs(array(actual["lbs"][field]) - array(expected["lbs"][field]))
        errors[field] = float(delta.max())
        if not np.allclose(array(actual["lbs"][field]), array(expected["lbs"][field]), rtol=1e-5, atol=1e-6):
            raise AssertionError(f"Regression mismatch in {field}: max error {errors[field]}")
    for field in ("rest_pos", "faces", "node_type"):
        if field in original:
            assert np.array_equal(array(actual[field]), array(original[field])), field
    for path, digest in hashes.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path
    report.update(regression="passed", max_array_errors=errors, input_files_unchanged=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
