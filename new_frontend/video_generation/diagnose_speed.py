#!/usr/bin/env python3
"""影片跑得很慢時，用這個找原因。約 1 分鐘。

用法（只要這一行，跟跑影片用的是同一個 paths.json）：

    python diagnose_speed.py --config paths.json

跑完會印出報告，最後有一段「結論」，用白話告訴你哪裡有問題；同時存成 diagnose_report.txt。
請把整份報告（或 diagnose_report.txt）貼給負責人。

它會檢查：
  1. 你的程式檔是不是跟推上 GitHub 的 new-frontend 版本完全一致（有沒有漏拿新程式）
  2. GPU：型號、驅動、有沒有其他程式在用、壓力測試時有沒有被降速、顯示記憶體夠不夠
  3. PyTorch / Warp 有沒有真的用到 GPU，速度跟參考值差多少
  4. 磁碟讀寫速度（work_root 放在很慢的磁碟會拖慢）
  5. 會拖慢速度的環境變數
  6. 最近一次影片產生的各階段耗時（跟作者電腦的參考值對照，看哪個階段慢）
參考值來自 RTX 5060 筆電（8GB）。
"""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REF_MATMUL_TFLOPS = 6.0
REF_WARP_GBPS = 260.0
NEED_VRAM_MIB = 7900
REF_DISK_MBPS = 300.0

# 本分支 overlay 程式檔的指紋（已忽略換行格式差異）；包含可選袖口修復。
EXPECTED = [
    ("ContourCraft", "combine_garments_untangle.py", "0d3eb3fe6364148995e2768d1000bf1aa9d06cb96cd45aebe509a22d15ea60bc"),
    ("ContourCraft", "contourcraft_fixed_size_upper.py", "045933553620ede5837acf1e53b358cb67913d778940d87e0628bd28d58325b4"),
    ("ContourCraft", "fit_body_from_measurements_fast.py", "bf19e230582b5c48675d8761b462134c5b0c59ed30126ab4d9b6b3b212460e17"),
    ("ContourCraft", "pin_garment_waistband.py", "09dbb9155fe09af9bfd91bb21a3d05001df85e8ba8f0e78a31d90e3706f86be3"),
    ("ContourCraft", "recover_resize_garment_bootstrap.py", "371baf4b25965ac0df844bbec3f9754823199b892ee358621ada0bee7309858e"),
    ("ContourCraft", "relax_recovered_garment.py", "e6c9dc5acb655cec6d43676781b319eac10f7ec1abbbcbf723b7224fb829f7d4"),
    ("ContourCraft", "run_outfit_preview_pipeline.py", "8f54120b0d948d30a610e9a25c643a4aab611aa4fad1f947268f92dcbd08180a"),
    ("ContourCraft", "repair_sleeve_lbs.py", "c0205d9687ba4387e71715363a9d2441378ac1e22ec0ccaa29a447a33d1937da"),
    ("ContourCraft", "check_sleeve_rollout.py", "cfea89836b0a0be87969cbd0a6749fb17e22984485f7ab0cb514afa14bb13672"),
    ("ContourCraft", "simulate_recovered_outfit.py", "3d9fa5d65a2625b263c12229688ae1fbf8de052d8fee03929e9c3b18baf2a2d9"),
    ("Gaussian-Garments", "inference.py", "563d5fe8dee9723297f39e369ea73a4b0e06fbbb0c9c83ce0828e802861676d8"),
    ("Gaussian-Garments", "render_lean.py", "e0fb1aba693a4bead1ff51d01dafd02ede2b571160df0bfb4b71296bbb0a5ca1"),
    ("Gaussian-Garments", "render_turntable.py", "033ef5cceb5f6266c8ceab2886281e72ece8c74bc3a5d10e13687eda70abf8e8"),
]
# 能讓人一眼看出「這是舊版」的標記
MARKERS = {
    ("ContourCraft", "run_outfit_preview_pipeline.py"): ["--skip-preview", "faststart"],
    ("ContourCraft", "simulate_recovered_outfit.py"): ["skip-safecheck-collision-count"],
    ("Gaussian-Garments", "inference.py"): ["samples=samples"],
    ("Gaussian-Garments", "render_lean.py"): ["bake-samples"],
}
THROTTLE_BITS = {0x2: "應用程式限制時脈", 0x4: "軟體功耗上限(power cap)", 0x8: "硬體降速", 0x20: "軟體過熱降速",
                 0x40: "硬體過熱降速", 0x80: "硬體功耗煞車"}

TORCH_BENCH = r'''
import json, time, sys
out = {}
try:
    import torch
except Exception as e:
    print(json.dumps({"error": "import torch 失敗：%s" % e})); sys.exit(0)
out["torch"] = torch.__version__; out["torch_cuda"] = torch.version.cuda
out["cuda_available"] = bool(torch.cuda.is_available())
import importlib.metadata as md
for pkg in ("torchvision", "torch-geometric", "smplx", "numpy", "warp-lang"):
    try:
        out["v_" + pkg.replace("-", "_")] = md.version(pkg)
    except Exception:
        out["v_" + pkg.replace("-", "_")] = "（沒有安裝）"
try:
    import cccollisions; out["cccollisions"] = "可以載入"
except Exception as e:
    out["cccollisions"] = "無法載入：%s" % e
if out["cuda_available"]:
    cap = torch.cuda.get_device_capability(0)
    out["name"] = torch.cuda.get_device_name(0); out["capability"] = "sm_%d%d" % cap
    out["arch_list"] = torch.cuda.get_arch_list()
    x = torch.randn(4096, 4096, device="cuda")
    for _ in range(3): x @ x
    torch.cuda.synchronize(); t0, n = time.time(), 0
    while time.time() - t0 < __SECONDS__:
        for _ in range(10): x @ x
        torch.cuda.synchronize(); n += 10
    out["matmul_tflops"] = 2 * 4096 ** 3 * n / (time.time() - t0) / 1e12
print(json.dumps(out))
'''

WARP_BENCH = r'''
import json, time
import warp as wp
wp.init()
@wp.kernel
def scale(a: wp.array(dtype=float), b: wp.array(dtype=float)):
    i = wp.tid()
    b[i] = a[i] * 2.0 + 1.0
out = {"warp": wp.__version__, "cuda_devices": wp.get_cuda_device_count(), "device": str(wp.get_device())}
n = 1 << 24
a = wp.zeros(n, dtype=float); b = wp.zeros(n, dtype=float)
wp.launch(scale, dim=n, inputs=[a, b]); wp.synchronize()
t0 = time.time()
for _ in range(100): wp.launch(scale, dim=n, inputs=[a, b])
wp.synchronize()
out["gbps"] = 100 * n * 8 / (time.time() - t0) / 1e9
print("RESULT " + json.dumps(out))
'''

_report = []
problems = []


def say(s=""):
    print(s)
    _report.append(s)


def line(mark, msg):
    say(f"  {mark} {msg}")


def run(cmd, timeout=120):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, "", str(e)


def smi(fields):
    for f in fields:
        code, out, _ = run(["nvidia-smi", f"--query-gpu={f}", "--format=csv,noheader,nounits"], 20)
        if code == 0 and out:
            return out.splitlines()[0].strip()
    return None


def sha(path: Path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def section_code(cc_root, gg_root):
    say("\n== 1. 程式檔是否和推上 GitHub 的版本一致")
    if not cc_root and not gg_root:
        say("  （沒有提供 --config，略過；請用 python diagnose_speed.py --config paths.json）")
        return
    roots = {"ContourCraft": cc_root, "Gaussian-Garments": gg_root}
    for repo, root in roots.items():
        if root and Path(root).is_dir():
            code, out, _ = run(["git", "-C", str(root), "log", "-1", "--format=%h %ad", "--date=short"], 20)
            say(f"  {repo} 官方版本：{out if code == 0 else '（不是 git 資料夾或讀不到）'}（官方最新與作者測試的版本：ContourCraft d606432、Gaussian-Garments ceb66fc）")
    bad = []
    for repo, name, want in EXPECTED:
        root = roots.get(repo)
        if not root:
            continue
        f = Path(root) / name
        if not f.is_file():
            line("✗", f"{repo}/{name}：找不到這個檔案（沒有執行 setup_data.py？）")
            bad.append(f"{repo}/{name} 缺少")
            continue
        if sha(f) == want:
            line("✓", f"{repo}/{name}")
            continue
        text = f.read_text(errors="replace")
        miss = [m for m in MARKERS.get((repo, name), []) if m not in text]
        hint = f"（缺少新版功能：{', '.join(miss)}，這是舊版或自己改過的版本）" if miss else "（內容和推上去的版本不同）"
        line("✗", f"{repo}/{name}：{hint}")
        bad.append(f"{repo}/{name} 與推上去的版本不一致")
    if bad:
        problems.append("程式檔和推上去的版本不一致：" + "；".join(bad) + "。請 `git pull` 最新的 new-frontend 後重新執行 setup_data.py。")


def section_system(work):
    say("== 系統")
    say(f"  作業系統：{platform.platform()}")
    say(f"  CPU：{platform.processor() or 'n/a'}；核心數：{os.cpu_count()}")
    try:
        cpu = re.search(r"model name\s*:\s*(.+)", Path("/proc/cpuinfo").read_text()).group(1).strip()
        say(f"  CPU 型號：{cpu}")
        mem = int(re.search(r"MemTotal:\s+(\d+)", Path("/proc/meminfo").read_text()).group(1)) / 1024 / 1024
        say(f"  記憶體：{mem:.0f} GB")
        if "microsoft" in Path("/proc/version").read_text().lower():
            say("  環境：WSL")
    except (OSError, AttributeError):
        pass
    say(f"  這個診斷程式用的 Python：{sys.version.split()[0]}")


def section_gpu():
    say("\n== 2. GPU")
    if run(["nvidia-smi", "-L"], 20)[0] != 0:
        line("✗", "找不到 nvidia-smi，沒有可用的 NVIDIA 驅動")
        problems.append("沒有 NVIDIA 驅動")
        return
    say(f"  顯卡：{smi(['name'])}；驅動：{smi(['driver_version'])}")
    code, apps, _ = run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], 20)
    apps = [a for a in apps.splitlines() if a.strip()] if code == 0 else []
    if apps:
        line("!", "目前有其他程式在使用 GPU（會跟影片產生搶速度）：")
        for a in apps[:8]:
            say(f"        {a}")
        problems.append("GPU 正在被其他程式使用（可能是前一次沒結束的 batch_render / python / blender）")
    else:
        line("✓", "沒有其他程式在用 GPU")
    used, total = smi(["memory.used"]), smi(["memory.total"])
    say(f"  顯示記憶體：已用 {used} / {total} MiB")
    try:
        free = int(total) - int(used)
        if free < NEED_VRAM_MIB:
            line("!", f"可用顯示記憶體只有約 {free} MiB；布料模擬在作者電腦上會用到約 {NEED_VRAM_MIB} MiB。不夠時 Windows 驅動不會報錯，而是把超出的部分放到系統記憶體，速度會慢好幾倍")
            problems.append("可用顯示記憶體不足：請關閉瀏覽器、遊戲等用到 GPU 的程式（上面列出的那些）")
        else:
            line("✓", f"可用顯示記憶體約 {free} MiB，足夠")
    except (TypeError, ValueError):
        pass


def section_torch(py, seconds):
    say("\n== 3. PyTorch 與 GPU 壓力測試（約 20 秒，期間同時檢查有沒有被降速）")
    bench = subprocess.Popen([py, "-c", TORCH_BENCH.replace("__SECONDS__", str(seconds))],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # 測試進行中連續取樣，取 GPU 使用率最高的那一筆（PyTorch 剛啟動的前幾秒 GPU 還沒開始跑，不能用）
    load, best = {}, -1
    while bench.poll() is None:
        time.sleep(1.0)
        util = smi(["utilization.gpu"])
        try:
            u = int(util)
        except (TypeError, ValueError):
            continue
        if u >= best:
            best = u
            load = dict(clock=smi(["clocks.sm"]), max_clock=smi(["clocks.max.sm"]), power=smi(["power.draw"]),
                        limit=smi(["power.limit"]), temp=smi(["temperature.gpu"]), util=util,
                        throttle=smi(["clocks_throttle_reasons.active", "clocks_event_reasons.active"]))
    stdout, stderr = bench.communicate()
    try:
        t = json.loads(stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        t = {"error": (stderr.strip().splitlines() or ["沒有輸出"])[-1]}
    if "error" in t:
        line("✗", t["error"])
        problems.append("這個 Python 無法使用 PyTorch（請確認 --python 指到跑 ContourCraft 的環境）")
        return
    say(f"  torch {t['torch']}（CUDA {t['torch_cuda']}）；torchvision {t.get('v_torchvision')}；torch_geometric {t.get('v_torch_geometric')}；"
        f"smplx {t.get('v_smplx')}；numpy {t.get('v_numpy')}；warp-lang {t.get('v_warp_lang')}")
    line("✓" if t["cccollisions"] == "可以載入" else "✗", f"cccollisions（碰撞偵測）：{t['cccollisions']}")
    if not t["cuda_available"]:
        line("✗", "PyTorch 看不到 GPU（torch.cuda.is_available() = False）：模擬會跑在 CPU 上，會慢非常多")
        problems.append("PyTorch 看不到 GPU")
        return
    arch = t["capability"]
    if arch in t["arch_list"]:
        line("✓", f"PyTorch 內有這張顯卡（{arch}）的專用程式")
    else:
        line("!", f"PyTorch 沒有 {arch} 的專用程式（它只有：{', '.join(t['arch_list'])}）。CUDA 程式要臨時轉換，可能比正常慢很多倍")
        problems.append(f"PyTorch 不支援這張顯卡（{arch}），請安裝 CUDA 12.8 以上、支援新顯卡的 PyTorch")
    ratio = t["matmul_tflops"] / REF_MATMUL_TFLOPS
    line("✓" if ratio >= 0.5 else "!", f"矩陣運算速度 {t['matmul_tflops']:.1f} TFLOPS（參考值約 {REF_MATMUL_TFLOPS:.0f}，你是參考的 {ratio:.1f} 倍）")
    slow = ratio < 0.5
    if load["clock"]:
        say(f"  壓力測試時：GPU 時脈 {load['clock']} / 最高 {load['max_clock']} MHz；功耗 {load['power']} / 上限 {load['limit']} W；"
            f"溫度 {load['temp']}°C；使用率 {load['util']}%")
        try:
            raw = load["throttle"] or "0"
            bits = int(raw, 16) if "x" in raw.lower() else int(raw)
        except ValueError:
            bits = 0
        reasons = [d for b, d in THROTTLE_BITS.items() if bits & b]
        if reasons and slow:
            line("!", "GPU 速度偏低，而且正在被降速：" + "、".join(reasons))
            problems.append("GPU 被降速導致偏慢：" + "、".join(reasons) + "（筆電請接電源、切到高效能模式、檢查散熱）")
        elif reasons:
            line("✓", "GPU 有被降速（" + "、".join(reasons) + "），筆電滿載時很常見，但速度仍正常")
        elif slow:
            line("!", "GPU 速度偏低但沒有偵測到降速原因，請確認筆電是否接電源、是否為高效能模式")
            problems.append("GPU 速度偏低：筆電請接電源並切到高效能模式")
        else:
            line("✓", "壓力測試時 GPU 沒有被降速")


def section_warp(py, work):
    say("\n== 4. Warp（布料模擬的碰撞偵測）")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=str(work)) as f:
        f.write(WARP_BENCH)
        wfile = f.name
    code, out, err = run([py, wfile], 300)
    os.unlink(wfile)
    res = None
    for ln in out.splitlines():
        if ln.startswith("RESULT "):
            res = json.loads(ln[7:])
    if res is None:
        line("✗", "Warp 無法執行：" + (err.splitlines() or ["沒有訊息"])[-1])
        problems.append("Warp 無法執行")
    elif res["cuda_devices"] == 0:
        line("✗", f"Warp 看不到 GPU（裝置：{res['device']}）：它會退回 CPU 執行，這是布料模擬變很慢的常見原因")
        problems.append("Warp 看不到 GPU")
    else:
        ratio = res["gbps"] / REF_WARP_GBPS
        line("✓" if ratio >= 0.5 else "!", f"warp {res['warp']} 使用 {res['device']}；記憶體頻寬 {res['gbps']:.0f} GB/s（參考值約 {REF_WARP_GBPS:.0f}，你是參考的 {ratio:.1f} 倍）")


def section_disk(work):
    say("\n== 5. 磁碟速度（work-dir 所在的磁碟）")
    work.mkdir(parents=True, exist_ok=True)
    try:
        code, fs, _ = run(["stat", "-f", "-c", "%T", str(work)], 10)
        if code == 0 and fs in ("9p", "drvfs", "fuseblk", "ntfs", "v9fs"):
            line("!", f"work-dir 在 Windows 磁碟（檔案系統 {fs}），讀寫很慢")
            problems.append("work-dir 放在 Windows 磁碟，請改放 Linux 原生路徑（例如 ~/work）")
    except OSError:
        pass
    p = work / "_disk_test.bin"
    data = os.urandom(1 << 20) * 128   # 128 MB
    try:
        t0 = time.time()
        with open(p, "wb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        w = 128 / (time.time() - t0)
        t0 = time.time()
        with open(p, "rb") as f:
            while f.read(1 << 24):
                pass
        r = 128 / (time.time() - t0)
        line("✓" if w >= REF_DISK_MBPS * 0.3 else "!", f"寫入 {w:.0f} MB/s、讀取 {r:.0f} MB/s（讀取可能被快取影響；參考寫入約 {REF_DISK_MBPS:.0f} MB/s）")
        if w < REF_DISK_MBPS * 0.3:
            problems.append("磁碟寫入速度偏低，work-dir 請放到比較快的磁碟")
    finally:
        try:
            p.unlink()
        except OSError:
            pass


def section_env():
    say("\n== 6. 會拖慢速度的環境變數")
    watch = {"CUDA_LAUNCH_BLOCKING": "1 會讓每個 GPU 動作都等待完成，明顯變慢（除錯用）",
             "CUDA_CACHE_DISABLE": "1 會關閉編譯快取，每次都重新編譯",
             "CUDA_VISIBLE_DEVICES": "指定/隱藏顯卡，請確認指到正確的那張",
             "PYTORCH_NO_CUDA_MEMORY_CACHING": "1 會關閉記憶體快取，明顯變慢",
             "OMP_NUM_THREADS": "限制 CPU 執行緒數（算圖會用到 CPU）"}
    found = False
    for k, why in watch.items():
        if os.environ.get(k):
            found = True
            line("!", f"{k}={os.environ[k]}：{why}")
            if k in ("CUDA_LAUNCH_BLOCKING", "CUDA_CACHE_DISABLE", "PYTORCH_NO_CUDA_MEMORY_CACHING") and os.environ[k] not in ("0", ""):
                problems.append(f"請取消設定環境變數 {k}")
    if not found:
        line("✓", "沒有會拖慢速度的環境變數")


def section_log(work, log):
    say("\n== 7. 最近一次影片產生的各階段耗時")
    if log is None and work:
        logs = sorted(Path(work).glob("logs/*/*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        log = logs[0] if logs else None
    if not log:
        say("  找不到影片產生的 log（--work-dir 底下的 logs/，或用 --log 指定）。如果還沒跑過影片，這一段略過")
        return
    txt = Path(log).read_text(errors="replace")
    say(f"  log：{log}")
    marks = [(m.group(1), float(m.group(2))) for m in re.finditer(r"\[status\] (\w+) \(t=([\d.]+)s\)", txt)]
    names = {"untangling": "衣服纏繞修正(untangle)", "fitting_body": "擬合身體", "rebinding": "衣服綁定", "full_simulating": "布料模擬", "full_rendering": "算圖與合成"}
    ref = {"untangling": "約 230~290 秒", "fitting_body": "約 15~25 秒", "rebinding": "約 3~4 秒", "full_simulating": "每步約 3.4~4.6 秒", "full_rendering": "每幀約 2.0~2.6 秒"}
    for (a, s), (b, e) in zip(marks, marks[1:]):
        if a in names:
            say(f"    {names[a]:20s} {e - s:7.0f} 秒    （作者電腦參考：{ref[a]}）")
    steps = re.search(r"Generating full animation \((\d+) steps\)", txt)
    t = dict(marks)
    if steps and "full_simulating" in t and "full_rendering" in t:
        per = (t["full_rendering"] - t["full_simulating"]) / int(steps.group(1))
        line("✓" if per <= 15 else "!", f"布料模擬約 {per:.1f} 秒/步（參考值約 3.4~5 秒/步）")
        if per > 15:
            problems.append(f"布料模擬每步 {per:.0f} 秒，比參考值慢很多")
    if not marks:
        say("  這份 log 裡沒有 [status] 階段記錄（影片可能還沒跑完，或 log 不完整）。最後幾行：")
        for ln in txt.replace("\r", "\n").strip().splitlines()[-4:]:
            say("    " + ln[:150])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None, help="paths.json（會自動帶入下面幾個路徑）")
    ap.add_argument("--python", default=None, help="（進階）跑 ContourCraft 的 python；預設取自 --config")
    ap.add_argument("--contourcraft-root", type=Path, default=None, help="（進階）預設取自 --config")
    ap.add_argument("--gaugar-root", type=Path, default=None, help="（進階）預設取自 --config")
    ap.add_argument("--work-dir", type=Path, default=None, help="（進階）work_root；預設取自 --config")
    ap.add_argument("--log", type=Path, default=None, help="指定要分析的 log 檔")
    ap.add_argument("--seconds", type=int, default=16, help="GPU 壓力測試秒數")
    args = ap.parse_args()
    cfg = {}
    if args.config:
        try:
            cfg = json.loads(args.config.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            sys.exit(f"讀不到設定檔 {args.config}：{e}")
    args.python = args.python or cfg.get("contourcraft_python") or sys.executable
    args.contourcraft_root = args.contourcraft_root or (Path(cfg["contourcraft_root"]) if cfg.get("contourcraft_root") else None)
    args.gaugar_root = args.gaugar_root or (Path(cfg["gaugar_root"]) if cfg.get("gaugar_root") else None)
    args.work_dir = args.work_dir or (Path(cfg["work_root"]) if cfg.get("work_root") else None)
    work = (args.work_dir or Path.cwd()).expanduser()
    py = str(Path(args.python).expanduser())

    section_system(work)
    section_code(args.contourcraft_root and args.contourcraft_root.expanduser(), args.gaugar_root and args.gaugar_root.expanduser())
    section_gpu()
    section_torch(py, args.seconds)
    section_warp(py, work if work.is_dir() else Path.cwd())
    section_disk(work)
    section_env()
    section_log(work if work.is_dir() else None, args.log)

    say("\n== 結論")
    if problems:
        for p in dict.fromkeys(problems):
            say(f"  - {p}")
    else:
        say("  硬體與設定看起來都正常。如果影片還是很慢，請把這份報告和最近的 log 檔一起貼給負責人。")
    out = Path.cwd() / "diagnose_report.txt"
    out.write_text("\n".join(_report), encoding="utf-8")
    print(f"\n（報告已存到 {out}）")


if __name__ == "__main__":
    main()
