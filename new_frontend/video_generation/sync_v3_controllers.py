#!/usr/bin/env python3
"""Install V3 opacity controllers into a Hugging Face garment package.

The package garment name may differ from the reconstruction subject name. Use
`--subject-map` for those cases, for example `Actor08_shirt=Actor08`.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from toolkit_common import load_catalog, print_err


def parse_map(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--subject-map 必須是 PACKAGE_GARMENT=SUBJECT：{value}")
        garment, subject = value.split("=", 1)
        if not garment or not subject:
            raise ValueError(f"--subject-map 不可有空白名稱：{value}")
        result[garment] = subject
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--controller-root", required=True, type=Path)
    parser.add_argument("--subject-map", action="append", default=[], metavar="GARMENT=SUBJECT")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    catalog = load_catalog(args.data_root)
    subject_map = parse_map(args.subject_map)
    failures = 0
    for garment in sorted(catalog["garments"]):
        subject = subject_map.get(garment, garment)
        source = args.controller_root / subject / "opacity_controller_v3" / "controller.pt"
        destination = args.data_root / "garments" / garment / "opacity_controller_v3" / "controller.pt"
        if not source.is_file():
            print_err(f"✗ {garment}: 找不到 {source}")
            failures += 1
            continue
        if args.dry_run:
            print(f"→ {source} -> {destination}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        metadata = {
            "garment": garment,
            "source_subject": subject,
            "controller": "V3",
            "source_checkpoint": str(source),
        }
        destination.with_name("metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
        )
        print(f"✓ {garment}: {destination}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
