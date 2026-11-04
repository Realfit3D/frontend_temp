#!/usr/bin/env python3
"""Install the toolkit's overlay programs into your official ContourCraft / Gaussian-Garments
checkouts, and hook the downloaded data package up to where those programs expect it.

  python setup_data.py --config paths.json

1. Overlay: copies overlay/ContourCraft/* into contourcraft_root and overlay/Gaussian-Garments/*
   into gaugar_root. A file that already exists with different content is first backed up as
   `<name>.orig` (kept if one already exists). A destination that is a symlink is never written
   through -- it is reported and skipped.
2. Data: for every garment in data_root/garments/ creates symlinks (nothing is copied):
     <gaugar output_root>/<g>/stage1/{template_uv.obj,cameras.json}   -> data package
     <gaugar output_root>/<g>/stage3/ckpt                              -> data package
     <gaugar output_root>/<g>/opacity_controller_v3/controller.pt      -> data package (when present)
     <work_root>/<g>/{garment_dict.pkl,source_body_fit.pkl}            -> data package cache/
   Existing real files/directories are left alone; links pointing elsewhere are reported.

Safe to run repeatedly.
"""

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

from toolkit_common import (OVERLAY_DIRS, TOOLKIT_DIR, ToolkitError, gaugar_output_root, load_paths,
                            print_err)


def install_overlay(paths: dict) -> bool:
    ok = True
    print("== 安裝 overlay 程式")
    for sub, key in OVERLAY_DIRS.items():
        src_root = TOOLKIT_DIR / "overlay" / sub
        dst_root = Path(paths[key])
        if not dst_root.is_dir():
            print_err(f"  ✗ {key} 不是資料夾：{dst_root}")
            ok = False
            continue
        for src in sorted(p for p in src_root.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
            rel = src.relative_to(src_root)
            dst = dst_root / rel
            if dst.is_symlink():
                print_err(f"  ✗ {sub}/{rel}：目標是符號連結，為了不改到連結指向的檔案，已跳過（請手動處理）")
                ok = False
            elif dst.exists() and filecmp.cmp(src, dst, shallow=False):
                print(f"  = {sub}/{rel}（已是最新）")
            else:
                if dst.exists():
                    bak = dst.with_name(dst.name + ".orig")
                    if not bak.exists():
                        shutil.copy2(dst, bak)
                        print(f"  ↻ {sub}/{rel}（原檔已備份為 {bak.name}）")
                    else:
                        print(f"  ↻ {sub}/{rel}（已有 {bak.name}，不重複備份）")
                else:
                    print(f"  + {sub}/{rel}")
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    return ok


def link(src: Path, dst: Path) -> str:
    if dst.is_symlink():
        return "ok" if dst.resolve() == src.resolve() else "elsewhere"
    if dst.exists():
        return "real"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve())
    return "new"


def link_data(paths: dict) -> bool:
    print("\n== 接上資料包")
    garments_dir = Path(paths["data_root"]) / "garments"
    if not garments_dir.is_dir():
        print_err(f"  ✗ 找不到 {garments_dir}。請確認資料包已下載到 data_root。")
        return False
    gg_out = gaugar_output_root(paths)
    work = Path(paths["work_root"])
    print(f"  Gaussian-Garments output_root：{gg_out}")
    ok = True
    for gdir in sorted(p for p in garments_dir.iterdir() if p.is_dir()):
        g = gdir.name
        pairs = [
            (gdir / "stage1" / "template_uv.obj", gg_out / g / "stage1" / "template_uv.obj"),
            (gdir / "stage1" / "cameras.json", gg_out / g / "stage1" / "cameras.json"),
            (gdir / "stage3" / "ckpt", gg_out / g / "stage3" / "ckpt"),
            (gdir / "cache" / "garment_dict.pkl", work / g / "garment_dict.pkl"),
            (gdir / "cache" / "source_body_fit.pkl", work / g / "source_body_fit.pkl"),
        ]
        controller = gdir / "opacity_controller_v3" / "controller.pt"
        if controller.is_file():
            pairs.append((controller, gg_out / g / "opacity_controller_v3" / "controller.pt"))
        results = []
        for src, dst in pairs:
            if not src.exists():
                print_err(f"  ✗ {g}：資料包缺少 {src.relative_to(gdir)}")
                ok = False
                continue
            r = link(src, dst)
            results.append(r)
            if r == "elsewhere":
                print_err(f"  ! {g}：{dst} 已是連結但指向別處，保留不動")
            elif r == "real":
                print_err(f"  ! {g}：{dst} 已存在實體檔案，保留不動")
        new = results.count("new")
        print(f"  {g}：新增 {new} 個連結" if new else f"  {g}：已接好")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path, help="paths.json")
    args = ap.parse_args()
    try:
        paths = load_paths(args.config)
        ok = install_overlay(paths)
        ok = link_data(paths) and ok
    except ToolkitError as e:
        print_err(f"錯誤：{e}")
        sys.exit(2)
    print("\n完成。下一步：python check_setup.py --config paths.json" if ok
          else "\n有項目需要處理（見上面的 ✗ / !）。處理後可重新執行本程式。")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
