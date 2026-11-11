#!/usr/bin/env python3
"""make_site.py — 把網頁與影片打包成可以直接上線的靜態網站資料夾（給 GitHub Pages 用）。

  python3 make_site.py                      # 資料庫位置讀 ../video_generation/paths.json 的 library_root
  python3 make_site.py --out /path/to/site  # 指定輸出資料夾（預設 ~/GG_projects/site）

打包前請先執行 build_index.py，讓 manifest.json 是最新的。

輸出的資料夾內容：
  index.html, library.js, .nojekyll
  video_library/manifest.json   （每支影片多一個 "v" 內容雜湊，網址會變成 ...mp4?v=雜湊，換過的影片就不會讀到手機的舊快取）
  video_library/catalog.json
  video_library/thumbnails/*.png
  video_library/<搭配>/<性別>/<身形>/<動作>.mp4   （只有 .mp4，不含 .json 說明檔）

  - 被 catalog.json 的 fit_rules 排除的影片（前端本來就不會顯示）不會放進網站。
  - 輸出資料夾內的 .git 不會動；其餘不屬於這次打包的舊檔會被刪掉，所以輸出資料夾可以直接當成部署用的 git 儲存庫。
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def short_hash(path, n=10):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def is_unfit(rules, video):
    """和 library.js 的 unfitReason 同一個判斷：任何一件衣服符合任何一條規則就算不適合。"""
    for r in rules:
        if r.get("garment") not in video.get("garments", []):
            continue
        for b in r.get("bodies", []):
            if ((b.get("gender") is None or b["gender"] == video.get("gender")) and
                    (b.get("height") is None or b["height"] == video.get("height")) and
                    (b.get("weight") is None or b["weight"] == video.get("weight"))):
                return True
    return False


def default_library():
    cfg = HERE.parent / "video_generation" / "paths.json"
    if cfg.exists():
        return Path(json.loads(cfg.read_text(encoding="utf-8"))["library_root"])
    return Path.home() / "GG_projects" / "video_library"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", type=Path, default=None, help="影片資料庫資料夾（預設讀 paths.json）")
    ap.add_argument("--out", type=Path, default=Path.home() / "GG_projects" / "site", help="輸出資料夾")
    args = ap.parse_args()

    lib = (args.library or default_library()).resolve()
    out = args.out.resolve()
    manifest_path, catalog_path = lib / "manifest.json", lib / "catalog.json"
    for p in (manifest_path, catalog_path, lib / "thumbnails"):
        if not p.exists():
            sys.exit(f"找不到 {p}。請先執行 build_index.py。")
    if out in lib.parents or out == lib or lib in out.parents:
        sys.exit("輸出資料夾不可以和影片資料庫互相包含。")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    rules = catalog.get("fit_rules") or []

    out.mkdir(parents=True, exist_ok=True)
    keep = set()                                   # 這次打包會存在的檔案（相對 out）

    def put(src, rel):
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.stat().st_size != src.stat().st_size or short_hash(dst, 16) != short_hash(src, 16):
            shutil.copyfile(src, dst)
        keep.add(Path(rel).as_posix())

    # 1. 網頁本體
    for name in ("index.html", "library.js"):
        put(HERE / name, name)
    (out / ".nojekyll").write_text("", encoding="utf-8")
    keep.add(".nojekyll")

    # 2. 影片（跳過不適合的、缺檔的），並寫入內容雜湊
    videos, skipped_unfit, missing = [], 0, []
    for v in manifest.get("videos", []):
        if is_unfit(rules, v):
            skipped_unfit += 1
            continue
        src = lib / v["path"]
        if not src.exists():
            missing.append(v["path"])
            continue
        put(src, f"video_library/{v['path']}")
        nv = dict(v)
        nv["v"] = short_hash(src)
        videos.append(nv)

    site_manifest = dict(manifest)
    site_manifest["videos"] = videos
    site_manifest["count"] = len(videos)
    site_manifest.pop("options", None)             # 網頁沒用到，省流量
    (out / "video_library").mkdir(parents=True, exist_ok=True)
    (out / "video_library" / "manifest.json").write_text(
        json.dumps(site_manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    keep.add("video_library/manifest.json")

    # 3. catalog 與縮圖
    put(catalog_path, "video_library/catalog.json")
    for t in sorted((lib / "thumbnails").glob("*.png")):
        put(t, f"video_library/thumbnails/{t.name}")

    # 4. 清掉輸出資料夾裡不屬於這次打包的舊檔（.git 不動）
    removed = 0
    for p in sorted(out.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        rel = p.relative_to(out)
        if rel.parts and rel.parts[0] == ".git":
            continue
        if p.is_file() and rel.as_posix() not in keep:
            p.unlink(); removed += 1
        elif p.is_dir() and not any(p.iterdir()):
            p.rmdir()

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file() and ".git" not in f.relative_to(out).parts)
    print(f"輸出：{out}")
    print(f"影片 {len(videos)} 支（manifest 原有 {len(manifest.get('videos', []))} 支；因「不適合」規則不收錄 {skipped_unfit} 支）")
    print(f"總大小 {size / 1e6:.1f} MB；清掉舊檔 {removed} 個")
    if missing:
        print(f"警告：manifest 列出但找不到的影片 {len(missing)} 支：", *missing[:5], sep="\n  ")


if __name__ == "__main__":
    main()
