#!/usr/bin/env python3
"""Render ONE video: outfit + gender + height + weight + motion + frames -> mp4 + sidecar json.

Wraps ContourCraft/run_outfit_preview_pipeline.py (fit SMPL-X body to the height/weight ->
rebind garments -> cloth simulation -> Blender render -> mp4), always with --skip-preview
since only the final video is wanted here.

`frames` is the number of frames in the FINAL video. The pipeline drops the first
DISCARD_FRAMES frames (the cloth's unstable settling start) and its trajectory has
steps + 2 frames, so the simulation is run for  frames + DISCARD_FRAMES - 2  steps.

Reuse across videos (kept under work_root, not cleaned up):
  - the per-outfit untangle result  (same garments -> done once)
  - the fitted body betas           (same gender/height/weight -> done once)

Usage:
    python render_one_video.py --config paths.json \
        --garments Actor02_lower Actor02_upper --gender male \
        --height 170 --weight 60 --motion 01_01 --frames 250 [--fps 25] \
        [--opacity 0.5 --opacity-mode v3 --opacity-garments Actor02_upper]
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from toolkit_common import (DISCARD_FRAMES, OPACITY_MODES, Job, ToolkitError, atomic_write_json,
                            ensure_cache_links, find_v3_controller, load_catalog, load_paths,
                            make_sidecar, motion_file, print_err, fail, steps_for_frames,
                            AUTO_FRAMES, resolve_auto_frames)


def _safe(s: str) -> str:
    return s.replace("+", "_")


def run_name(job: Job) -> str:
    return f"{job.gender}_{job.body_dir}_{job.video_stem}_f{job.frames}"


STAGE_NAMES = {
    "recovering_garments": "準備衣服", "untangling": "衣服纏繞修正(untangle)", "fitting_body": "擬合身體",
    "rebinding": "衣服綁定到身體", "full_simulating": "布料模擬", "full_rendering": "算圖與合成影片",
}


def stage_timings(log: Path) -> dict:
    """Seconds spent in each pipeline stage, from the `[status] <phase> (t=12.3s)` lines the pipeline
    prints (t = seconds since the pipeline started). Returns {} if the log has none."""
    import re
    marks = []
    try:
        for line in Path(log).read_text(errors="replace").splitlines():
            m = re.match(r"\[status\] (\w+) \(t=([\d.]+)s\)", line)
            if m:
                marks.append((m.group(1), float(m.group(2))))
    except OSError:
        return {}
    out = {}
    for (name, t0), (_, t1) in zip(marks, marks[1:]):
        if name in STAGE_NAMES:
            out[name] = round(out.get(name, 0.0) + (t1 - t0), 1)
    return out


def format_timings(stages: dict, steps: int, frames: int) -> str:
    parts = [f"{STAGE_NAMES[k]} {v:.0f}秒" for k, v in stages.items()]
    extra = []
    if stages.get("full_simulating") and steps:
        extra.append(f"模擬約 {stages['full_simulating'] / steps:.1f} 秒/步")
    if stages.get("full_rendering") and frames:
        extra.append(f"算圖約 {stages['full_rendering'] / frames:.1f} 秒/幀")
    return "；".join(parts) + (("（" + "，".join(extra) + "）") if extra else "")


def log_path(paths: dict, job: Job) -> Path:
    return Path(paths["work_root"]) / "logs" / _safe(job.outfit) / f"{run_name(job)}.log"


def betas_cache(paths: dict, job: Job) -> Path:
    return Path(paths["work_root"]) / "_betas" / f"{job.gender}_{job.body_dir}" / "betas.pkl"


def _pipeline_cmd(paths, catalog, job, run_dir_name, precomputed_betas):
    script = Path(paths["contourcraft_root"]) / "run_outfit_preview_pipeline.py"
    if not script.is_file():
        fail(f"找不到 {script}。請先執行 setup_data.py 安裝 overlay 程式。")
    source_genders = [catalog["garments"][g]["source_gender"] for g in job.garments]
    cmd = [str(paths["contourcraft_python"]), str(script),
           "--garment-names", *job.garments,
           "--outfit-name", job.work_name,
           "--height", f"{job.height / 100:.2f}",
           "--weight", str(job.weight),
           "--gender", job.gender,
           "--source-genders", *source_genders,
           "--motion", str(motion_file(paths, catalog, job.motion)),
           "--full-steps", str(steps_for_frames(job.frames)),
           "--discard-frames", str(DISCARD_FRAMES),
           "--fps", str(job.fps),
           "--opacity-scale", str(job.opacity),
           "--opacity-mode", job.opacity_mode,
           "--skip-preview",
           "--run-name", run_dir_name,
           "--contourcraft-root", str(paths["contourcraft_root"]),
           "--gaugar-root", str(paths["gaugar_root"]),
           "--ccraft-python", str(paths["contourcraft_python"]),
           "--gaugar-python", str(paths["gaugar_python"]),
           "--output-root", str(paths["work_root"])]
    if job.opacity_garments:
        cmd += ["--opacity-garments", *job.opacity_garments]
    if precomputed_betas is not None:
        cmd += ["--precomputed-betas", str(precomputed_betas)]
    if job.opacity_mode == "v3" and 0.0 < job.opacity < 1.0:
        for garment in job.effective_opacity_garments:
            controller = find_v3_controller(paths, garment)
            if controller is None:
                fail(
                    f"`{garment}` 找不到 V3 controller.pt。請放到 "
                    "data_root/garments/<衣服>/opacity_controller_v3/controller.pt，"
                    "或在 paths.json 的 v3_controllers 指定路徑。"
                )
            cmd += ["--v3-controller", f"{garment}={controller.resolve()}"]
    return cmd


def render_video(paths: dict, catalog: dict, job: Job, dest_video: Path, dest_sidecar: Path = None,
                 keep_run_dir: bool = False) -> dict:
    """Render `job` and put the mp4 at dest_video (and a sidecar json at dest_sidecar, if given).
    Returns the sidecar dict. Raises ToolkitError on failure (after writing a `failed` sidecar)."""
    work = Path(paths["work_root"])
    missing = ensure_cache_links(paths, job.garments)      # re-link the garment caches if work_root was cleaned
    if missing:
        error = ("資料包裡找不到這些檔案：" + "、".join(missing) +
                 "。請確認資料包完整下載到 data_root，然後執行 python setup_data.py --config paths.json")
        raise RenderFailed(error, make_sidecar(job, "failed", error=error))
    rn = run_name(job)
    run_dir = work / "outfits" / job.work_name / "runs_preview" / rn
    if run_dir.exists():                       # leftover from an interrupted attempt: start this video over
        shutil.rmtree(run_dir)
    lp = log_path(paths, job)
    lp.parent.mkdir(parents=True, exist_ok=True)

    cached = betas_cache(paths, job)
    cmd = _pipeline_cmd(paths, catalog, job, rn, cached if cached.is_file() else None)
    t0 = time.perf_counter()
    try:
        with lp.open("w") as log:
            log.write("$ " + " ".join(cmd) + "\n\n")
            log.flush()
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(paths["contourcraft_root"]))
    except OSError as e:                       # e.g. the python path in paths.json can't be executed
        shutil.rmtree(run_dir, ignore_errors=True)
        error = f"無法啟動 pipeline：{e}。請檢查 paths.json 內的 Python 路徑。"
        raise RenderFailed(error, make_sidecar(job, "failed", error=error))
    except BaseException:                      # Ctrl+C etc.: leave nothing half-done behind
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    elapsed = time.perf_counter() - t0

    video = run_dir / "video.mp4"
    n_rendered = len(list((run_dir / "frames" / "renders").glob("*.png")))
    error = None
    if proc.returncode != 0:
        tail = lp.read_text(errors="replace").strip().splitlines()[-12:]
        error = f"pipeline 結束代碼 {proc.returncode}。log 最後幾行：\n" + "\n".join(tail)
    elif not video.is_file():
        error = "pipeline 沒有產生 video.mp4"
    elif n_rendered != job.frames:
        error = f"影片幀數不符：要 {job.frames} 幀，實際渲染 {n_rendered} 幀（steps_for_frames 的換算可能需要調整）"

    if error:
        shutil.rmtree(run_dir, ignore_errors=True)
        sc = make_sidecar(job, "failed", elapsed_seconds=round(elapsed), error=error)
        raise RenderFailed(error, sc)

    betas_src = run_dir / "betas.pkl"
    if not cached.is_file() and betas_src.is_file():
        cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(betas_src, cached)

    dest_video = Path(dest_video)
    dest_video.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_video.with_name(dest_video.name + ".tmp")
    shutil.copyfile(video, tmp)
    tmp.replace(dest_video)
    stages = stage_timings(lp)
    sc = make_sidecar(job, "done", elapsed_seconds=round(elapsed), stage_seconds=stages)
    if dest_sidecar is not None:
        atomic_write_json(dest_sidecar, sc)       # written last: sidecar says "done" only once the mp4 is in place
    if not keep_run_dir:
        shutil.rmtree(run_dir, ignore_errors=True)
    return sc


class RenderFailed(ToolkitError):
    def __init__(self, message, sidecar):
        super().__init__(message)
        self.sidecar = sidecar


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path, help="paths.json")
    ap.add_argument("--garments", nargs="+", required=True, help="由內到外，例：Actor02_lower Actor02_upper")
    ap.add_argument("--gender", required=True, choices=["male", "female"])
    ap.add_argument("--height", required=True, type=int, help="cm，整數")
    ap.add_argument("--weight", required=True, type=int, help="kg，整數")
    ap.add_argument("--motion", required=True, help="動作編號，例：01_01")
    ap.add_argument("--frames", default="auto", help="最後影片的幀數；不寫（auto）= 跑完整個動作檔")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--opacity", type=float, default=1.0, help="透明度 0 到 1；預設 1（不透明）")
    ap.add_argument("--opacity-mode", choices=OPACITY_MODES, default="direct",
                    help="direct/optical_depth 不需權重；v3 使用每件衣服的已訓練 controller")
    ap.add_argument("--opacity-garments", nargs="+", default=[],
                    help="只調整指定衣服；省略表示整套衣服都調整")
    ap.add_argument("--output", type=Path, default=None,
                    help="只輸出到指定的 mp4 路徑（不寫入影片資料庫）。預設寫入 library_root 的標準位置。")
    args = ap.parse_args()
    if not 0.0 <= args.opacity <= 1.0:
        ap.error("--opacity 必須介於 0 到 1")
    unknown_targets = [g for g in args.opacity_garments if g not in args.garments]
    if unknown_targets:
        ap.error("--opacity-garments 只能包含 --garments 內的衣服：" + ", ".join(unknown_targets))
    try:
        paths = load_paths(args.config)
        catalog = load_catalog(paths["data_root"])
        frames = AUTO_FRAMES if args.frames == "auto" else int(args.frames)
        job = resolve_auto_frames([
            Job(tuple(args.garments), args.gender, args.height, args.weight,
                args.motion, frames, args.fps, args.opacity, args.opacity_mode,
                tuple(args.opacity_garments))
        ], paths, catalog)[0]
        if args.output:
            sc = render_video(paths, catalog, job, args.output)
        else:
            sc = render_video(paths, catalog, job, job.video_path(paths["library_root"]),
                              job.sidecar_path(paths["library_root"]))
    except RenderFailed as e:
        print_err(f"失敗：{e}")
        sys.exit(1)
    except ToolkitError as e:
        print_err(f"錯誤：{e}")
        sys.exit(2)
    print(f"完成，耗時 {sc['elapsed_seconds']} 秒")


if __name__ == "__main__":
    main()
