#!/usr/bin/env python3
"""
TACO (official 60-class COCO) -> project 9-class YOLOv8-Seg
Stage-1 SAFE CANDIDATE conversion (v1).

Design goals (see datasets/taco/TACO_STAGE1_AUDIT.md section 5):
  * EXIF orientation is baked into the pixels at conversion time, so the stored
    pixels always equal the COCO width/height the polygons were drawn in.
  * polygons are normalised by COCO width/height, no resampling, 6 decimals.
  * out-of-bounds vertices (6 annotations, all -1 px) are clipped and logged.
  * `annotation.id` is NEVER used as a dict key (2 duplicated ids exist).
  * multi-part polygons are handled by an explicit, parameterised policy.
  * unmapped instances are never silently turned into background: any image
    holding an unmapped / unresolved instance is excluded whole.
  * full per-annotation reconciliation: every one of the 4784 input
    annotations is accounted for exactly once.

Writes ONLY under datasets/taco/yolo_seg/ and refuses to touch the raw archive
or the indoor merged dataset.
"""

import argparse
import collections
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = "/home/xmv/garbage_yolov8_seg"
TACO_ROOT = os.path.join(REPO_ROOT, "datasets", "taco")
DEFAULT_ANN = os.path.join(TACO_ROOT, "raw", "taco_official_repo", "data", "annotations.json")
DEFAULT_IMAGES = os.path.join(TACO_ROOT, "raw", "images")
DEFAULT_MAP = os.path.join(REPO_ROOT, "tools", "taco", "taco_class_mapping_v1.json")
DEFAULT_OUT = os.path.join(TACO_ROOT, "yolo_seg")
INDOOR_YAML = os.path.join(REPO_ROOT, "datasets", "garbage_seg_merged_full1444", "data.yaml")

FORBIDDEN_OUT_PREFIXES = (
    os.path.join(TACO_ROOT, "raw") + os.sep,
    os.path.join(REPO_ROOT, "datasets", "garbage_seg_merged_full1444") + os.sep,
    os.path.join(REPO_ROOT, "datasets", "half_dataset_yolo") + os.sep,
    os.path.join(REPO_ROOT, "datasets", "waste_ds_c3_neg_yolo") + os.sep,
)

RARE_CLASSES = ["food_waste", "can", "bottle", "cigarette_butt"]


# --------------------------------------------------------------------------- #
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def dhash_file(path, size=8, apply_exif=True):
    """dHash in the FINAL (EXIF-baked) space.

    Must match the space the converted dataset lives in, otherwise two copies of
    the same photo stored with different EXIF orientations hash differently and
    can straddle splits (caught by the validator on the first v2 run).
    """
    with Image.open(path) as im:
        if apply_exif:
            im = ImageOps.exif_transpose(im)
        g = im.convert("L").resize((size + 1, size), Image.LANCZOS)
        px = list(g.getdata())
    bits = 0
    for r in range(size):
        for c in range(size):
            bits = (bits << 1) | (1 if px[r * (size + 1) + c] > px[r * (size + 1) + c + 1] else 0)
    return bits


def hamming(a, b):
    return bin(a ^ b).count("1")


def shoelace(flat_xy):
    n = len(flat_xy) // 2
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = flat_xy[2 * i], flat_xy[2 * i + 1]
        j = (i + 1) % n
        x2, y2 = flat_xy[2 * j], flat_xy[2 * j + 1]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def out_name(file_name):
    base, ext = os.path.splitext(file_name.replace("/", "__"))
    return base + ext.lower()


def read_indoor_names():
    names = {}
    with open(INDOOR_YAML) as f:
        inside = False
        for line in f:
            if line.strip().startswith("names:"):
                inside = True
                continue
            if inside:
                m = re.match(r"\s*(\d+):\s*(\S+)", line)
                if m:
                    names[int(m.group(1))] = m.group(2)
                elif line.strip():
                    break
    return names


def union_find_units(images, sha_by_id, dh_by_id, hamming_max=5):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for im in images:
        find(im["id"])
    by_sha = collections.defaultdict(list)
    for i, s in sha_by_id.items():
        by_sha[s].append(i)
    for g in by_sha.values():
        for x in g[1:]:
            union(g[0], x)
    # dh_by_id maps image_id -> dHash bits: compare the HASH (index 1),
    # and union the IMAGE IDS (index 0).
    dh_items = list(dh_by_id.items())
    for i in range(len(dh_items)):
        for j in range(i + 1, len(dh_items)):
            if hamming(dh_items[i][1], dh_items[j][1]) <= hamming_max:
                union(dh_items[i][0], dh_items[j][0])
    groups = collections.defaultdict(list)
    for im in images:
        groups[find(im["id"])].append(im["id"])
    return dict(groups)


def make_split(units, image_tags, ratios, seed):
    """Deterministic greedy, leak-free split; rare classes seeded into val/test."""
    N = sum(len(v) for v in units.values())
    n_val = int(round(ratios[1] * N))
    n_test = int(round(ratios[2] * N))

    def units_tags(u):
        s = set()
        for i in units[u]:
            s |= image_tags.get(i, set())
        return s

    rare_total = collections.Counter()
    for i, ts in image_tags.items():
        for t in ts:
            if t in RARE_CLASSES:
                rare_total[t] += 1
    need = {t: max(1, int(round(ratios[1] * rare_total[t]))) for t in RARE_CLASSES}

    ids = sorted(units.keys())
    random.Random(seed).shuffle(ids)

    assigned = {}
    got = collections.Counter()

    def fill(split_name, target_n):
        taken = 0
        while taken < target_n:
            best, best_key = None, None
            for u in ids:
                if u in assigned:
                    continue
                tags = units_tags(u)
                score = sum(1 for t in tags if t in need and got[t] < need[t])
                key = (score, -len(units[u]), -u)
                if best_key is None or key > best_key:
                    best, best_key = u, key
            if best is None:
                return taken
            assigned[best] = split_name
            taken += len(units[best])
            for i in units[best]:
                for t in image_tags.get(i, set()):
                    got[t] += 1
        return taken

    fill("val", n_val)
    fill("test", n_test)
    for u in ids:
        assigned.setdefault(u, "train")
    return {i: assigned[u] for u, ids_ in units.items() for i in ids_}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", default=DEFAULT_ANN)
    ap.add_argument("--images", default=DEFAULT_IMAGES)
    ap.add_argument("--mapping", default=DEFAULT_MAP)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--multipart-policy", default="exclude",
                    choices=["exclude", "largest", "split", "union-contour"])
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="explicitly delete out_dir/images and out_dir/labels and regenerate")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    for p in FORBIDDEN_OUT_PREFIXES:
        if (out_dir + os.sep).startswith(p):
            sys.exit(f"refusing to write into protected tree: {out_dir}")

    t0 = time.time()
    coco = json.load(open(args.annotations))
    mapping = json.load(open(args.mapping))

    target_classes = {int(k): v for k, v in mapping["target_classes"].items()}
    indoor = read_indoor_names()
    if target_classes != indoor:
        sys.exit(f"class scheme mismatch with indoor dataset:\n target={target_classes}\n indoor={indoor}")
    name_to_id = {v: k for k, v in target_classes.items()}

    cat_by_id = {c["id"]: c for c in coco["categories"]}
    images = coco["images"]
    anns = coco["annotations"]
    img_by_id = {im["id"]: im for im in images}

    dec, map_path = {}, {}
    for k, v in mapping["category_action"].items():
        cid = int(k)
        assert cat_by_id[cid]["name"] == v["name"], (cid, cat_by_id[cid]["name"], v["name"])
        dec[cid] = v
        if v["action"] == "MAP":
            map_path[cid] = name_to_id[v["to"]]

    anns_by_image = collections.defaultdict(list)   # list order, never keyed by annotation id
    for idx, a in enumerate(anns):
        anns_by_image[a["image_id"]].append((idx, a))

    # ---------------------------------------------------------------- exclusion
    image_reasons = {}
    reason_cat_counts = collections.defaultdict(collections.Counter)
    multipart_ann_ids, oob_ann_ids = [], []

    for im in images:
        iid = im["id"]
        reasons = set()
        for idx, a in anns_by_image.get(iid, []):
            v = dec[a["category_id"]]
            if v["action"] == "EXCLUDE_STRUCTURAL":
                reasons.add("unlabeled_litter")
                reason_cat_counts["unlabeled_litter"][v["name"]] += 1
            elif v["action"] == "EXCLUDE_UNRESOLVED":
                reasons.add("unresolved_category")
                reason_cat_counts["unresolved_category"][v["name"]] += 1
            elif v["action"] == "DEFER":
                reasons.add("deferred_semantics")
                reason_cat_counts["deferred_semantics"][v["name"]] += 1
            seg = a["segmentation"]
            if isinstance(seg, list) and len(seg) > 1:
                multipart_ann_ids.append(a["id"])
                if args.multipart_policy == "exclude":
                    reasons.add("multipart_polygon")
                    reason_cat_counts["multipart_polygon"][v["name"]] += 1
            if isinstance(seg, list):
                flat = [c for poly in seg for c in poly]
                if flat:
                    xs, ys = flat[0::2], flat[1::2]
                    if min(xs) < 0 or min(ys) < 0 or max(xs) > im["width"] or max(ys) > im["height"]:
                        oob_ann_ids.append(a["id"])
        image_reasons[iid] = reasons

    excluded_ids = {i for i, r in image_reasons.items() if r}
    kept_ids = [im["id"] for im in images if im["id"] not in excluded_ids]
    kept_set = set(kept_ids)

    reason_img_counts = collections.Counter()
    for r in image_reasons.values():
        for x in r:
            reason_img_counts[x] += 1
    overlap_ul_mp = sum(1 for r in image_reasons.values()
                        if "unlabeled_litter" in r and "multipart_polygon" in r)
    union_d6_d7 = sum(1 for r in image_reasons.values()
                      if "unlabeled_litter" in r or "multipart_polygon" in r)

    id_counts = collections.Counter(a["id"] for a in anns)
    dup_ids = sorted(k for k, n in id_counts.items() if n > 1)
    dup_audit = []
    for did in dup_ids:
        for idx, a in enumerate(anns):
            if a["id"] == did:
                dup_audit.append({
                    "annotation_id": did, "annotation_index": idx,
                    "image_id": a["image_id"],
                    "file_name": img_by_id[a["image_id"]]["file_name"],
                    "category_id": a["category_id"],
                    "category_name": cat_by_id[a["category_id"]]["name"],
                    "image_kept": a["image_id"] in kept_set,
                })

    analysis = {
        "n_images": len(images),
        "n_annotations": len(anns),
        "kept_images": len(kept_ids),
        "excluded_images": len(excluded_ids),
        "exclusion_reason_image_counts_overlapping": dict(sorted(reason_img_counts.items())),
        "exclusion_reason_instance_counts_overlapping": {
            k: sum(v.values()) for k, v in sorted(reason_cat_counts.items())},
        "exclusion_reason_detail_by_category": {k: dict(v) for k, v in sorted(reason_cat_counts.items())},
        "union_excluded": len(excluded_ids),
        "naive_sum_D6_plus_D7": reason_img_counts["unlabeled_litter"] + reason_img_counts["multipart_polygon"],
        "actual_union_of_D6_and_D7": union_d6_d7,
        "D6_D7_overlap": overlap_ul_mp,
        "multipart_annotations": len(multipart_ann_ids),
        "out_of_bounds_annotations": len(oob_ann_ids),
        "out_of_bounds_annotation_ids": sorted(set(oob_ann_ids)),
        "duplicate_annotation_ids": dup_ids,
        "mapped_categories": len(map_path),
        "mapped_instances": sum(1 for a in anns if a["category_id"] in map_path),
        "multipart_policy": args.multipart_policy,
    }
    print(json.dumps(analysis, indent=1, ensure_ascii=False))
    if args.analyze_only:
        return

    # ------------------------------------------------------------------ hashes
    print(f"[{time.time()-t0:6.1f}s] hashing + dhash on {len(kept_ids)} kept images ...", flush=True)
    sha_by_id, dh_by_id, exif_by_id = {}, {}, {}
    missing = []
    for iid in kept_ids:
        im = img_by_id[iid]
        p = os.path.join(args.images, im["file_name"])
        if not os.path.isfile(p):
            missing.append(im["file_name"])
            continue
        sha_by_id[iid] = sha256_file(p)
        dh_by_id[iid] = dhash_file(p)
        with Image.open(p) as ii:
            exif_by_id[iid] = ii.getexif().get(274)
    if missing:
        sys.exit(f"missing image files: {missing[:5]}")

    kept_images = [img_by_id[i] for i in kept_ids]
    units = union_find_units(kept_images, sha_by_id, dh_by_id)

    image_tags = {}
    for iid in kept_ids:
        image_tags[iid] = {target_classes[map_path[a["category_id"]]]
                           for _, a in anns_by_image.get(iid, []) if a["category_id"] in map_path}

    split_of = make_split(units, image_tags, args.ratios, args.seed)

    split_lists = collections.defaultdict(list)
    for iid, s in split_of.items():
        split_lists[s].append(iid)
    rnd = random.Random(args.seed + 1)
    for s in split_lists:
        split_lists[s].sort(key=lambda i: img_by_id[i]["file_name"])
        rnd.shuffle(split_lists[s])
    if args.limit:
        for s in split_lists:
            split_lists[s] = split_lists[s][: args.limit]

    # --------------------------------------------------------------- write out
    if os.path.isdir(os.path.join(out_dir, "images")):
        if not args.overwrite:
            sys.exit(f"output already exists: {out_dir}/images -- pass --overwrite to regenerate")
        shutil.rmtree(os.path.join(out_dir, "images"))
        shutil.rmtree(os.path.join(out_dir, "labels"), ignore_errors=True)
    images_out, labels_out = {}, {}
    for s in ("train", "val", "test"):
        images_out[s] = os.path.join(out_dir, "images", s)
        labels_out[s] = os.path.join(out_dir, "labels", s)
        os.makedirs(images_out[s], exist_ok=True)
        os.makedirs(labels_out[s], exist_ok=True)

    jobs = [(s, iid) for s in ("train", "val", "test") for iid in split_lists[s]]
    counters = collections.Counter()
    image_rows = []

    def process(job):
        s, iid = job
        im = img_by_id[iid]
        src = os.path.join(args.images, im["file_name"])
        new = out_name(im["file_name"])
        dst = os.path.join(images_out[s], new)
        cw, ch = im["width"], im["height"]
        orientation = exif_by_id.get(iid)
        res = {"image_id": iid, "file_name": im["file_name"], "new_name": new, "split": s,
               "width": cw, "height": ch, "orientation": orientation,
               "sha_before": sha_by_id.get(iid)}
        with Image.open(src) as ii:
            if orientation in (None, 1) and ii.size == (cw, ch):
                shutil.copyfile(src, dst)
                res["reencoded"] = False
            else:
                im2 = ImageOps.exif_transpose(ii)
                if im2.size != (cw, ch):
                    return {"error": f"EXIF bake size mismatch {im2.size} != {(cw, ch)} on {im['file_name']}"}
                im2.convert("RGB").save(dst, "JPEG", quality=95)
                res["reencoded"] = True

        lines, written, degenerate, clipped = [], 0, 0, 0
        for _, a in anns_by_image.get(iid, []):
            cid = a["category_id"]
            if cid not in map_path:
                return {"error": f"unmapped annotation survived exclusion on {im['file_name']}"}
            cls = map_path[cid]
            parts = a["segmentation"] if isinstance(a["segmentation"], list) else []
            if len(parts) > 1:
                if args.multipart_policy == "largest":
                    parts = [max(parts, key=shoelace)]
                elif args.multipart_policy == "union-contour":
                    import cv2
                    mask = np.zeros((ch, cw), np.uint8)
                    for poly in parts:
                        pts = np.array(poly, np.float32).reshape(-1, 2)
                        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
                    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    if cnts:
                        parts = [max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float).flatten().tolist()]
            for poly in parts:
                xs, ys = list(poly[0::2]), list(poly[1::2])
                if not xs:
                    continue
                cx = [min(max(v, 0.0), cw - 1.0) for v in xs]
                cy = [min(max(v, 0.0), ch - 1.0) for v in ys]
                if any(abs(p - q) > 1e-9 for p, q in zip(cx + cy, xs + ys)):
                    clipped += 1
                flat = [v for xy in zip(cx, cy) for v in xy]
                if len(flat) < 6 or len(set(zip(cx, cy))) < 3 or shoelace(flat) < 1.0:
                    degenerate += 1
                    continue
                norm = [v for xy in zip(cx, cy) for v in (xy[0] / cw, xy[1] / ch)]
                if any(v < 0 or v > 1 for v in norm):
                    return {"error": f"normalised out of range on {im['file_name']}"}
                lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in norm))
                written += 1

        with open(os.path.join(labels_out[s], os.path.splitext(new)[0] + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        res.update({"sha_after": sha256_file(dst), "written": written,
                    "degenerate": degenerate, "clipped": clipped})
        return res

    print(f"[{time.time()-t0:6.1f}s] converting {len(jobs)} images with {args.workers} workers ...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(process, jobs):
            if res is None:
                continue
            if "error" in res:
                sys.exit(f"ERROR: {res['error']}")
            image_rows.append({k: res[k] for k in ("image_id", "file_name", "new_name", "split",
                                                   "width", "height", "orientation", "reencoded",
                                                   "sha_before", "sha_after")})
            counters["written"] += res["written"]
            counters["degenerate"] += res["degenerate"]
            counters["clipped_annotations"] += res["clipped"]
            counters["reencoded"] += int(bool(res["reencoded"]))
            counters["copied"] += int(not res["reencoded"])

    # ------------------------------------------------- per annotation manifest
    manifest_rows = []
    for idx, a in enumerate(anns):
        iid = a["image_id"]
        v = dec[a["category_id"]]
        kept = iid in kept_set
        reasons = "|".join(sorted(image_reasons[iid])) if not kept else ""
        written = bool(kept and a["category_id"] in map_path)
        target = target_classes[map_path[a["category_id"]]] if a["category_id"] in map_path else ""
        if not kept:
            drop = "image_excluded:" + reasons
        elif not written:
            drop = "degenerate_or_clipped"
        else:
            drop = ""
        manifest_rows.append({
            "annotation_index": idx, "annotation_id": a["id"], "image_id": iid,
            "file_name": img_by_id[iid]["file_name"], "category_id": a["category_id"],
            "category_name": v["name"], "tier": v["tier"], "action": v["action"],
            "target_class": target, "image_kept": int(kept),
            "image_exclusion_reasons": reasons, "written": int(written), "drop_reason": drop,
        })

    written_total = sum(r["written"] for r in manifest_rows)
    dropped_image_excluded = sum(1 for r in manifest_rows if not r["image_kept"])
    dropped_degenerate = sum(1 for r in manifest_rows
                             if r["image_kept"] and r["action"] == "MAP" and not r["written"])
    total = len(anns)
    reconciled = (written_total + dropped_image_excluded + dropped_degenerate == total)

    # ------------------------------------------------------------------ stats
    dist = collections.defaultdict(collections.Counter)
    img_class = collections.defaultdict(lambda: collections.defaultdict(set))
    for s in ("train", "val", "test"):
        for iid in split_lists[s]:
            for _, a in anns_by_image.get(iid, []):
                if a["category_id"] in map_path:
                    t = target_classes[map_path[a["category_id"]]]
                    dist[s][t] += 1
                    img_class[s][t].add(iid)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        f.write(f"path: {out_dir}\n\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\n\nnames:\n")
        for k in sorted(target_classes):
            f.write(f"  {k}: {target_classes[k]}\n")

    with open(os.path.join(out_dir, "conversion_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)

    with open(os.path.join(out_dir, "images_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(image_rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(image_rows, key=lambda r: (r["split"], r["new_name"])))

    with open(os.path.join(out_dir, "split_assignments.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "file_name", "new_name", "split"])
        for s in ("train", "val", "test"):
            for iid in sorted(split_lists[s], key=lambda i: img_by_id[i]["file_name"]):
                w.writerow([iid, img_by_id[iid]["file_name"], out_name(img_by_id[iid]["file_name"]), s])

    exclusion_report = {
        "policy": {
            "D6_unlabeled_litter": "whole image exclusion",
            "D7_multipart": f"policy={args.multipart_policy} (v1 default = whole image exclusion)",
            "deferred_semantics": "PROVISIONAL tier left unmapped -> image excluded, never implicit background",
            "unresolved_category": "Six pack rings -> image excluded",
        },
        **analysis,
        "union_excluded_detail": {
            "unlabeled_litter_images": reason_img_counts["unlabeled_litter"],
            "multipart_images": reason_img_counts["multipart_polygon"],
            "naive_sum_D6_plus_D7": reason_img_counts["unlabeled_litter"] + reason_img_counts["multipart_polygon"],
            "actual_union_of_D6_and_D7": union_d6_d7,
            "overlap_D6_and_D7": overlap_ul_mp,
            "total_excluded_union": len(excluded_ids),
            "kept_images": len(kept_ids),
        },
        "deferred_categories": mapping["deferred_categories"],
        "excluded_categories": mapping["excluded_categories"],
        "unresolved_categories": mapping["unresolved_categories"],
        "pending_human_adjudication": mapping["pending_human_adjudication"],
    }
    with open(os.path.join(out_dir, "exclusion_report.json"), "w") as f:
        json.dump(exclusion_report, f, indent=1, ensure_ascii=False)

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "annotations_file": args.annotations,
        "annotations_sha256": sha256_file(args.annotations),
        "mapping_file": args.mapping,
        "status": "CANDIDATE_V1_SAFE_SUBSET",
        "target_classes": {str(k): v for k, v in target_classes.items()},
        "split_ratios": args.ratios,
        "seed": args.seed,
        "multipart_policy": args.multipart_policy,
        "images": {
            "input": len(images), "kept": len(kept_ids), "excluded": len(excluded_ids),
            "train": len(split_lists["train"]), "val": len(split_lists["val"]),
            "test": len(split_lists["test"]), "units": len(units),
            "reencoded": counters["reencoded"], "byte_copied": counters["copied"],
            "annotations_clipped": counters["clipped_annotations"],
            "annotations_degenerate_dropped": counters["degenerate"],
        },
        "instances_per_class_per_split": {
            s: {c: dist[s].get(c, 0) for c in target_classes.values()} for s in ("train", "val", "test")},
        "instances_per_class_total": {
            c: sum(dist[s].get(c, 0) for s in ("train", "val", "test")) for c in target_classes.values()},
        "images_per_class_per_split": {
            s: {c: len(img_class[s].get(c, set())) for c in target_classes.values()} for s in ("train", "val", "test")},
        "reconciliation": {
            "input_annotations": total, "written": written_total,
            "dropped_image_excluded": dropped_image_excluded,
            "dropped_degenerate_or_clipped_away": dropped_degenerate,
            "sum_equals_input": reconciled,
        },
        "duplicate_annotation_id_audit": dup_audit,
        "exclusion": exclusion_report["union_excluded_detail"],
    }
    with open(os.path.join(out_dir, "conversion_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)

    print(f"[{time.time()-t0:6.1f}s] done -> {out_dir}")
    print(json.dumps({
        "kept_images": len(kept_ids), "excluded_images": len(excluded_ids),
        "splits": {s: len(split_lists[s]) for s in ("train", "val", "test")},
        "instances_total_per_class": summary["instances_per_class_total"],
        "reconciliation": summary["reconciliation"],
        "duplicate_id_audit": dup_audit,
    }, indent=1, ensure_ascii=False))

    if not reconciled:
        sys.exit("RECONCILIATION FAILED")


if __name__ == "__main__":
    main()
