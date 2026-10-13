#!/usr/bin/env python3
"""
TACO Stage-1 official image downloader.

Source of truth: pedropro/TACO official repository
  * data/annotations.json  (official, reviewed annotations)
  * per-image `flickr_url` (original resolution) -- same source used by the
    official download.py.

Differences vs. official download.py (all deliberate, documented in the audit):
  * uses the ORIGINAL flickr_url so that pixel geometry matches the COCO
    width/height fields (the *_640_url variant would break coordinate parity);
  * creates nested output directories (file_name is "batch_N/xxxxxx.jpg");
  * never crashes on a broken link -- every failure is recorded explicitly;
  * verifies the downloaded image dimensions against the COCO metadata;
  * resumable (existing valid files are skipped) and thread-parallel;
  * writes a full machine-readable manifest + a failure CSV.

This script NEVER writes anything outside the taco/ tree and never touches the
indoor `garbage_seg_merged_full1444` dataset.
"""

import argparse
import concurrent.futures as futures
import hashlib
import io
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone

import requests
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # TACO contains a few very large panoramas

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根目录
TACO_ROOT = os.path.join(REPO_ROOT, "datasets", "taco")
DEFAULT_ANN = os.path.join(TACO_ROOT, "raw", "taco_official_repo", "data", "annotations.json")
DEFAULT_OUT = os.path.join(TACO_ROOT, "raw", "images")

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

_print_lock = threading.Lock()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def head_available(session, url, timeout=20):
    """Cheap availability probe used only to describe fallback options."""
    if not url:
        return "no_url"
    try:
        r = session.head(url, timeout=timeout, allow_redirects=True, headers={"User-Agent": UA})
        if r.status_code == 405:  # some CDNs reject HEAD
            r = session.get(url, timeout=timeout, stream=True, headers={"User-Agent": UA})
            r.close()
        return str(r.status_code)
    except Exception as e:  # noqa: BLE001
        return f"ERR:{type(e).__name__}"


def fetch(session, url, retries=3, timeout=60):
    """Return (bytes, http_status, error_string)."""
    last_err = ""
    last_status = ""
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=timeout, headers={"User-Agent": UA})
            last_status = str(r.status_code)
            if r.status_code == 200 and r.content:
                return r.content, last_status, ""
            last_err = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(1.0 * (attempt + 1))
    return None, last_status, last_err


def process_one(session, img, out_dir, check_fallback):
    image_id = img["id"]
    file_name = img["file_name"]
    url = img["flickr_url"]
    url640 = img.get("flickr_640_url")

    rec = {
        "image_id": image_id,
        "file_name": file_name,
        "flickr_url": url,
        "flickr_640_url": url640,
        "coco_width": img.get("width"),
        "coco_height": img.get("height"),
        "status": "UNKNOWN",
        "http_status": "",
        "n_bytes": 0,
        "sha256": "",
        "actual_width": None,
        "actual_height": None,
        "dim_match": None,
        "error": "",
        "fallback_640_http": "",
    }

    dst = os.path.join(out_dir, file_name)
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    # ---- resume: reuse an already-valid file -------------------------------
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        try:
            with Image.open(dst) as im:
                im.verify()
            with Image.open(dst) as im:
                w, h = im.size
            rec.update(
                status="OK_RESUMED",
                n_bytes=os.path.getsize(dst),
                sha256=sha256_file(dst),
                actual_width=w,
                actual_height=h,
            )
            rec["dim_match"] = (w == rec["coco_width"] and h == rec["coco_height"])
            return rec
        except Exception:
            os.remove(dst)  # corrupt leftover -> re-download

    payload, http_status, err = fetch(session, url)
    rec["http_status"] = http_status
    if payload is None:
        rec["status"] = "FAILED"
        rec["error"] = err
        if check_fallback:
            rec["fallback_640_http"] = head_available(session, url640)
        return rec

    # ---- validate it is really an image before committing to disk ----------
    try:
        with Image.open(io.BytesIO(payload)) as im:
            im.load()
            w, h = im.size
        tmp = dst + ".part"
        with open(tmp, "wb") as f:
            f.write(payload)
        os.replace(tmp, dst)
    except Exception as e:  # noqa: BLE001
        rec["status"] = "FAILED"
        rec["error"] = f"decode/save error: {type(e).__name__}: {e}"
        if check_fallback:
            rec["fallback_640_http"] = head_available(session, url640)
        return rec

    rec.update(
        status="OK",
        n_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        actual_width=w,
        actual_height=h,
    )
    rec["dim_match"] = (w == rec["coco_width"] and h == rec["coco_height"])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", default=DEFAULT_ANN)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N images")
    ap.add_argument("--no-fallback-check", action="store_true")
    args = ap.parse_args()

    with open(args.annotations) as f:
        coco = json.load(f)
    images = coco["images"]
    if args.limit:
        images = images[: args.limit]

    os.makedirs(args.out, exist_ok=True)
    ann_sha = sha256_file(args.annotations)

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=args.workers * 2,
                                           pool_maxsize=args.workers * 2)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    results = []
    done = 0
    t0 = time.time()
    with futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(process_one, session, im, args.out, not args.no_fallback_check): im
            for im in images
        }
        for fut in futures.as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % 25 == 0 or done == len(images):
                with _print_lock:
                    ok = sum(1 for r in results if r["status"].startswith("OK"))
                    sys.stdout.write(
                        f"\r[{done}/{len(images)}] ok={ok} fail={done - ok} "
                        f"elapsed={time.time() - t0:.0f}s"
                    )
                    sys.stdout.flush()
    print()

    results.sort(key=lambda r: r["image_id"])
    ok = [r for r in results if r["status"].startswith("OK")]
    failed = [r for r in results if r["status"] == "FAILED"]
    dim_mismatch = [r for r in ok if r["dim_match"] is False]

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repo": "https://github.com/pedropro/TACO",
            "annotations_file": "data/annotations.json",
            "annotations_sha256": ann_sha,
            "image_url_field": "flickr_url (original resolution)",
            "note": "official reviewed annotations; annotations_unofficial.json NOT used",
        },
        "counts": {
            "annotated_images_in_coco": len(coco["images"]),
            "attempted": len(results),
            "downloaded_ok": len(ok),
            "failed": len(failed),
            "dim_mismatch": len(dim_mismatch),
        },
        "results": results,
    }
    man_path = os.path.join(os.path.dirname(args.out), "download_manifest.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=1)

    fail_csv = os.path.join(os.path.dirname(args.out), "download_failures.csv")
    with open(fail_csv, "w") as f:
        f.write("image_id,file_name,flickr_url,http_status,error,fallback_640_http\n")
        for r in failed:
            f.write(
                f'{r["image_id"]},"{r["file_name"]}",{r["flickr_url"]},'
                f'{r["http_status"]},"{r["error"]}",{r["fallback_640_http"]}\n'
            )

    print(f"annotated_in_coco = {len(coco['images'])}")
    print(f"downloaded_ok     = {len(ok)}")
    print(f"failed            = {len(failed)}")
    print(f"dim_mismatch      = {len(dim_mismatch)}")
    print(f"manifest          -> {man_path}")
    print(f"failures          -> {fail_csv}")


if __name__ == "__main__":
    main()
