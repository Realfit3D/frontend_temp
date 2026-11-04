#!/usr/bin/env python3
"""Check that this machine is ready to render videos.

  python check_setup.py --config paths.json              # all three checks
  python check_setup.py --config paths.json --skip-smoke # skip the 20-frame test render

1. Data:               data package complete, motion files present, official models present.
2. Environment/programs: both Python environments work, GPU visible, overlay programs installed,
                       data hooked up where Gaussian-Garments expects it.
3. Smoke test:         renders a 20-frame video end to end (output: work_root/smoke_test/video.mp4).

Every failed item says what is missing and how to fix it. Exit code 0 only if everything passes.
"""

import argparse
import filecmp
import subprocess
import sys
import time
from pathlib import Path

from render_one_video import format_timings, log_path, stage_timings
from toolkit_common import (GARMENT_FILES, OVERLAY_DIRS, TOOLKIT_DIR, Job, ToolkitError, gaugar_output_root, steps_for_frames,
                            load_catalog, load_paths, motion_file, print_err)

_failures = 0
REF_SEC_PER_STEP = 5.0     # measured on an RTX 5060 laptop GPU (8 GB)
SLOW_SEC_PER_STEP = 15.0   # warn when 3x slower than that


def ok(msg):
    print(f"  ✓ {msg}")


def bad(msg, fix=None):
    global _failures
    _failures += 1
    print(f"  ✗ {msg}")
    if fix:
        print(f"      → {fix}")


def warn(msg, fix=None):
    """A reminder that does not block rendering videos (not counted as a failure)."""
    print(f"  ! {msg}")
    if fix:
        print(f"      → {fix}")


def section(title):
    print(f"\n== {title}")


def python_check(label, python, code, fix):
    python = str(python)
    if not Path(python).is_file():
        bad(f"{label}：找不到 Python（{python}）", f"檢查 paths.json 裡的路徑是否填對")
        return False
    r = subprocess.run([python, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        last = (r.stderr.strip().splitlines() or ["(無訊息)"])[-1]
        bad(f"{label}：{last}", fix)
        return False
    ok(f"{label}")
    return True


def check_data(paths, catalog):
    section("1. 資料")
    data = Path(paths["data_root"])
    for g in catalog["garments"]:
        missing = [f for f in GARMENT_FILES if not (data / "garments" / g / f).is_file()]
        if missing:
            bad(f"衣服 {g} 缺少：{', '.join(missing)}", "確認資料包已完整下載到 data_root")
        else:
            ok(f"衣服 {g}")
        controller = data / "garments" / g / "opacity_controller_v3" / "controller.pt"
        if controller.is_file():
            ok(f"衣服 {g} 的 V3 opacity controller")
        else:
            warn(f"衣服 {g} 沒有 V3 controller：v3 透明度影片不能使用這件衣服",
                 "執行 sync_v3_controllers.py，或補上 data_root/garments/<衣服>/opacity_controller_v3/controller.pt")
    missing_thumbs = [g for g in catalog["garments"] if not (data / "thumbnails" / f"{g}.png").is_file()]
    if missing_thumbs:
        warn(f"資料包缺少這些衣服的縮圖：{', '.join(missing_thumbs)}",
             "不影響跑影片；縮圖由負責統整的人提供，放在 data_root/thumbnails/<衣服名稱>.png（網頁需要）")
    else:
        ok("衣服縮圖齊全（網頁用）")
    for m in catalog["motions"]:
        f = motion_file(paths, catalog, m)
        ok(f"動作 {m}") if f.is_file() else bad(f"動作 {m} 的檔案不存在：{f}", "確認資料包已完整下載")
    cc = Path(paths["contourcraft_root"]) / "ccraft_data"
    ckpt = cc / "trained_models" / "contourcraft.pth"
    ok("官方 contourcraft.pth") if ckpt.is_file() else bad(f"找不到 {ckpt}", "依官方 ContourCraft 說明下載 checkpoint")
    for g in ("MALE", "FEMALE"):
        f = cc / "aux_data" / "body_models" / "smplx" / f"SMPLX_{g}.npz"
        ok(f"SMPL-X {g} 身體模型") if f.is_file() else bad(f"找不到 {f}", "依官方 ContourCraft 說明下載 SMPL-X 模型")


def check_env(paths, catalog):
    section("2. 環境與程式")
    cc_defaults = Path(paths["contourcraft_root"]) / "utils" / "defaults.py"
    if cc_defaults.is_file() and "/path/to" in cc_defaults.read_text():
        bad(f"{cc_defaults} 還有官方的佔位符 /path/to", "依官方 ContourCraft 說明填入 data_root 等路徑")
    elif cc_defaults.is_file():
        ok("ContourCraft utils/defaults.py 已設定")
    else:
        bad(f"找不到 {cc_defaults}", "確認 contourcraft_root 指向 ContourCraft 資料夾")
    python_check("ContourCraft 環境（torch、smplx）", paths["contourcraft_python"],
                 "import torch, smplx, numpy", "確認這個環境已依官方說明安裝 ContourCraft 需要的套件")
    python_check("ContourCraft 環境可以使用 GPU", paths["contourcraft_python"],
                 "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)",
                 "確認 NVIDIA 驅動與 PyTorch 的 CUDA 版本相符")
    python_check("Warp（布料模擬的碰撞偵測）看得到 GPU", paths["contourcraft_python"],
                 "import sys, warp as wp; wp.init(); sys.exit(0 if wp.get_cuda_device_count() > 0 else 1)",
                 "Warp 沒偵測到 GPU 時會默默退回 CPU，模擬會慢非常多（數倍以上）。請確認 warp 的 CUDA 版本與驅動相容，"
                 "並查看 ContourCraft 跑起來的 log 開頭 `Warp x.x initialized` 底下的 Devices 有沒有列出 cuda:0")
    python_check("Gaussian-Garments 環境（torch、bpy、moviepy）", paths["gaugar_python"],
                 "import torch, bpy; from moviepy import ImageSequenceClip",
                 "確認這個環境已依官方說明安裝 Gaussian-Garments 算圖需要的套件")

    stale = []
    for sub, key in OVERLAY_DIRS.items():
        src_root = TOOLKIT_DIR / "overlay" / sub
        for src in sorted(p for p in src_root.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
            dst = Path(paths[key]) / src.relative_to(src_root)
            if not dst.is_file() or not filecmp.cmp(src, dst, shallow=False):
                stale.append(f"{sub}/{src.relative_to(src_root)}")
    ok("overlay 程式已安裝") if not stale else bad(f"overlay 程式未安裝或不是最新：{', '.join(stale)}",
                                                  "執行 python setup_data.py --config paths.json")

    gg_out = gaugar_output_root(paths)
    work = Path(paths["work_root"])
    unlinked = []
    for g in catalog["garments"]:
        need = [gg_out / g / "stage1" / "template_uv.obj", gg_out / g / "stage1" / "cameras.json",
                gg_out / g / "stage3" / "ckpt" / "net.pt",
                work / g / "garment_dict.pkl", work / g / "source_body_fit.pkl"]
        if not all(p.exists() for p in need):
            unlinked.append(g)
    ok("資料已接到官方程式預期的位置") if not unlinked else bad(
        f"這些衣服的資料還沒接上：{', '.join(unlinked)}", "執行 python setup_data.py --config paths.json")


def smoke_test(paths, catalog):
    section("3. 試跑（20 幀）")
    from render_one_video import render_video
    pick = {}
    for g, info in catalog["garments"].items():
        pick.setdefault(info.get("slot"), g)
    if "lower" not in pick or "upper" not in pick:
        bad("catalog.json 裡找不到一件下身（slot=lower）加一件上身（slot=upper）可供試跑")
        return
    garments = (pick["lower"], pick["upper"])
    motion = next(iter(catalog["motions"]))
    gender = catalog["garments"][garments[1]].get("source_gender", "male")
    job = Job(garments, gender, 170, 60, motion, 20, 25)
    out = Path(paths["work_root"]) / "smoke_test" / "video.mp4"
    print(f"  使用 {job.outfit}，{gender}，170cm / 60kg，動作 {motion}。第一次需要幾分鐘…", flush=True)
    t0 = time.perf_counter()
    try:
        render_video(paths, catalog, job, out)
    except ToolkitError as e:
        bad(f"試跑失敗：{e}", f"查看 {Path(paths['work_root']) / 'logs'} 底下最新的 log")
        return
    if out.is_file() and out.stat().st_size > 0:
        ok(f"試跑成功（{time.perf_counter() - t0:.0f} 秒），輸出：{out}")
        stages = stage_timings(log_path(paths, job))
        if stages:
            print("      階段耗時：" + format_timings(stages, steps_for_frames(job.frames), job.frames))
            per_step = stages.get("full_simulating", 0) / steps_for_frames(job.frames)
            if per_step > SLOW_SEC_PER_STEP:
                warn(f"模擬每步約 {per_step:.0f} 秒，比參考值（約 {REF_SEC_PER_STEP:.0f} 秒/步，RTX 5060 筆電）慢很多",
                     "250 幀的影片會需要好幾個小時。請檢查：GPU 是否真的被使用（nvidia-smi）、Warp 是否看得到 GPU、"
                     "是否有其他程式佔用 GPU、work_root 是否放在很慢的磁碟（例如 /mnt/c、/mnt/d）")
    else:
        bad("試跑沒有產生影片")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path, help="paths.json")
    ap.add_argument("--skip-smoke", action="store_true", help="略過 20 幀試跑")
    args = ap.parse_args()
    try:
        paths = load_paths(args.config)
        catalog = load_catalog(paths["data_root"])
        check_data(paths, catalog)
        check_env(paths, catalog)
    except ToolkitError as e:
        print_err(f"錯誤：{e}")
        sys.exit(2)
    if _failures:
        print(f"\n前兩項有 {_failures} 個問題，請先處理再試跑。" + ("（已略過試跑）" if not args.skip_smoke else ""))
        sys.exit(1)
    if not args.skip_smoke:
        try:
            smoke_test(paths, catalog)
        except ToolkitError as e:
            bad(f"{e}")
    print()
    if _failures:
        print(f"共 {_failures} 個問題。")
        sys.exit(1)
    print("全部通過，可以開始跑工作單。")


if __name__ == "__main__":
    main()
