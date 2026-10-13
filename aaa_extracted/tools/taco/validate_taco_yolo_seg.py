#!/usr/bin/env python3
"""
Independent no-GPU acceptance for datasets/taco/yolo_seg (TACO stage-1 safe
candidate v1).

It does NOT trust the converter: labels are re-derived from the official COCO
annotations with a separate implementation and compared line by line.

Checks
  1. class scheme identical to the indoor 9-class dataset
  2. image/label pairing per split
  3. Ultralytics `verify_image_label` parses every pair (the real loader path)
  4. polygon legality: class id range, even coords, >=3 points, [0,1], area>0
  5. stored pixels carry no residual EXIF orientation (2..8) and match the COCO
     width/height the polygons were drawn in
  6. no identical / near-duplicate (dHash<=5) images across splits
  7. excluded images really absent (Unlabeled litter, multipart, deferred, unresolved)
  8. duplicate annotation ids 309 / 4040 both preserved
  9. instance reconciliation against conversion_summary.json
 10. round-trip re-derivation of every label file
 11. representative visual overlays

Exit code 0 only when every hard check passes.
"""

import argparse
import collections
import csv
import hashlib
import json
import os
import random
import re
import sys

from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根目录
TACO_ROOT = os.path.join(REPO_ROOT, "datasets", "taco")
OUT = os.path.join(TACO_ROOT, "yolo_seg")
ANN = os.path.join(TACO_ROOT, "raw", "taco_official_repo", "data", "annotations.json")
IMAGES = os.path.join(TACO_ROOT, "raw", "images")
MAP = os.path.join(REPO_ROOT, "tools", "taco", "taco_class_mapping_v1.json")
INDOOR_YAML = os.path.join(REPO_ROOT, "datasets", "garbage_seg_merged_full1444", "data.yaml")

SEGS = ("train", "val", "test")
COLORS = {0: (230, 25, 75), 1: (60, 180, 75), 2: (255, 225, 25), 3: (0, 130, 200),
          4: (245, 130, 48), 5: (145, 30, 180), 6: (70, 240, 240), 7: (240, 50, 230),
          8: (0, 128, 128)}


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def dhash_file(p, size=8):
    with Image.open(p) as im:
        g = im.convert("L").resize((size + 1, size), Image.LANCZOS)
        px = list(g.getdata())
    bits = 0
    for r in range(size):
        for c in range(size):
            bits = (bits << 1) | (1 if px[r * (size + 1) + c] > px[r * (size + 1) + c + 1] else 0)
    return bits


def shoelace(xy):
    n = len(xy)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        s += xy[i][0] * xy[j][1] - xy[j][0] * xy[i][1]
    return abs(s) / 2.0


def read_yaml_names(path):
    names = {}
    inside = False
    for line in open(path):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--viz", type=int, default=12, help="samples per montage")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    R = {"out_dir": args.out}
    hard_fail = []

    names = read_yaml_names(os.path.join(args.out, "data.yaml"))
    indoor = read_yaml_names(INDOOR_YAML)
    R["class_scheme_matches_indoor"] = (names == indoor)
    R["class_names"] = names
    if names != indoor:
        hard_fail.append(f"class scheme differs from indoor: {names} != {indoor}")

    summary = json.load(open(os.path.join(args.out, "conversion_summary.json")))
    excl = json.load(open(os.path.join(args.out, "exclusion_report.json")))
    mapping = json.load(open(summary.get("mapping_file", MAP)))
    coco = json.load(open(ANN))
    cat_by_id = {c["id"]: c for c in coco["categories"]}
    dec = {int(k): v for k, v in mapping["category_action"].items()}
    target = {int(k): v for k, v in mapping["target_classes"].items()}
    nid = {v: k for k, v in target.items()}

    anns_by_image = collections.defaultdict(list)
    for idx, a in enumerate(coco["annotations"]):
        anns_by_image[a["image_id"]].append((idx, a))

    imgs_manifest = {}
    with open(os.path.join(args.out, "images_manifest.csv")) as f:
        for row in csv.DictReader(f):
            imgs_manifest[row["new_name"]] = row

    # -------------------------------------------------- 2. pairing per split
    pair_report = {}
    files_by_split = {}
    for s in SEGS:
        idir = os.path.join(args.out, "images", s)
        ldir = os.path.join(args.out, "labels", s)
        imgs = sorted(os.listdir(idir))
        labs = sorted(os.listdir(ldir))
        files_by_split[s] = imgs
        im_stems = {os.path.splitext(x)[0] for x in imgs}
        lab_stems = {os.path.splitext(x)[0] for x in labs}
        pair_report[s] = {
            "images": len(imgs), "labels": len(labs),
            "images_without_label": sorted(im_stems - lab_stems),
            "labels_without_image": sorted(lab_stems - im_stems),
        }
        if im_stems != lab_stems:
            hard_fail.append(f"{s}: image/label mismatch")
    R["pairing"] = pair_report

    total_imgs = sum(p["images"] for p in pair_report.values())
    R["total_images"] = total_imgs
    if total_imgs != summary["images"]["kept"]:
        hard_fail.append("total images != conversion summary kept count")

    # ---------------------------------- 3/4/5. per-image label + pixel checks
    from ultralytics.data.utils import verify_image_label

    inst_hist = collections.defaultdict(collections.Counter)
    img_hist = collections.defaultdict(lambda: collections.defaultdict(set))
    ultralytics_bad = []
    illegal = []
    exif_residual = []
    size_mismatch = []
    empty_labels = []
    point_counts = []
    clip_stats = collections.Counter()
    n_rows = 0

    for s in SEGS:
        for fn in files_by_split[s]:
            stem = os.path.splitext(fn)[0]
            ip = os.path.join(args.out, "images", s, fn)
            lp = os.path.join(args.out, "labels", s, stem + ".txt")
            man = imgs_manifest.get(fn)
            if man is None:
                hard_fail.append(f"{fn}: not in images_manifest.csv")
                continue
            cw, ch = int(man["width"]), int(man["height"])

            # --- Ultralytics real loader path
            try:
                res = verify_image_label((ip, lp, "", False, len(target), 0, 2, False))
                if res[0] is None:  # corrupt => (None, None, ..., nc=1, msg)
                    ultralytics_bad.append((fn, res[9]))
            except Exception as e:  # noqa: BLE001
                ultralytics_bad.append((fn, repr(e)))

            # --- independent pixel / EXIF check
            with Image.open(ip) as ii:
                orient = ii.getexif().get(274)
                size = ii.size
            if orient in (2, 3, 4, 5, 6, 7, 8):
                exif_residual.append((fn, orient))
            if size != (cw, ch):
                size_mismatch.append((fn, size, (cw, ch), man["reencoded"]))

            # --- independent label parse
            rows = [ln for ln in open(lp).read().strip().splitlines() if ln.strip()]
            n_rows += len(rows)
            if not rows:
                empty_labels.append(fn)
            for r_i, ln in enumerate(rows):
                t = ln.split()
                try:
                    c = int(t[0])
                except ValueError:
                    illegal.append((fn, r_i, "class not int")); continue
                vals = t[1:]
                if c not in target:
                    illegal.append((fn, r_i, f"class id {c} out of range"))
                    continue
                if len(vals) % 2 != 0 or len(vals) < 6:
                    illegal.append((fn, r_i, f"odd/short coord list len={len(vals)}")); continue
                try:
                    f = list(map(float, vals))
                except ValueError:
                    illegal.append((fn, r_i, "non numeric coord")); continue
                if any(v < 0 or v > 1 for v in f):
                    illegal.append((fn, r_i, "coord outside [0,1]")); continue
                pts = list(zip(f[0::2], f[1::2]))
                if len(set(pts)) < 3:
                    illegal.append((fn, r_i, "fewer than 3 distinct points")); continue
                if shoelace(pts) <= 0:
                    illegal.append((fn, r_i, "non-positive polygon area")); continue
                point_counts.append(len(pts))
                inst_hist[s][target[c]] += 1
                img_hist[s][target[c]].add(fn)

    R["ultralytics_parse_failures"] = ultralytics_bad[:20]
    R["n_ultralytics_parse_failures"] = len(ultralytics_bad)
    R["illegal_labels"] = illegal[:20]
    R["n_illegal_labels"] = len(illegal)
    R["residual_exif_orientation"] = exif_residual[:20]
    R["n_residual_exif"] = len(exif_residual)
    R["stored_size_mismatch"] = size_mismatch[:20]
    R["n_stored_size_mismatch"] = len(size_mismatch)
    R["empty_label_files"] = len(empty_labels)
    R["polygon_points_min"] = min(point_counts) if point_counts else None
    R["polygon_points_max"] = max(point_counts) if point_counts else None
    if ultralytics_bad:
        hard_fail.append(f"{len(ultralytics_bad)} image/label pairs rejected by Ultralytics")
    if illegal:
        hard_fail.append(f"{len(illegal)} illegal label rows")
    if exif_residual:
        hard_fail.append(f"{len(exif_residual)} images keep EXIF orientation 2..8")
    if size_mismatch:
        hard_fail.append(f"{len(size_mismatch)} stored sizes != COCO annotation space")

    R["instances_per_class_per_split"] = {s: dict(inst_hist[s]) for s in SEGS}
    R["instances_per_class_total"] = {
        c: sum(inst_hist[s].get(c, 0) for s in SEGS) for c in target.values()}
    R["images_per_class_per_split"] = {
        s: {c: len(img_hist[s].get(c, set())) for c in target.values()} for s in SEGS}
    R["total_label_rows"] = n_rows

    # ------------------------------------------------ 6. cross-split leakage
    sha_by = {}
    dh_by = {}
    split_of = {}
    for s in SEGS:
        for fn in files_by_split[s]:
            p = os.path.join(args.out, "images", s, fn)
            sha_by.setdefault(sha256_file(p), []).append((s, fn))
            dh_by.setdefault(dhash_file(p), []).append((s, fn))
            split_of[fn] = s
    cross_sha = [v for v in sha_by.values() if len({x[0] for x in v}) > 1]
    dh_items = list(dh_by.items())
    cross_near = []
    for i in range(len(dh_items)):
        for j in range(i + 1, len(dh_items)):
            if bin(dh_items[i][0] ^ dh_items[j][0]).count("1") <= 5:
                sp_i = {x[0] for x in dh_items[i][1]}
                sp_j = {x[0] for x in dh_items[j][1]}
                if len(sp_i | sp_j) > 1:
                    cross_near.append({"a": dh_items[i][1], "b": dh_items[j][1]})
    R["cross_split_identical_images"] = cross_sha
    R["cross_split_near_duplicate_pairs"] = cross_near
    if cross_sha:
        hard_fail.append(f"{len(cross_sha)} identical image groups straddle splits")
    if cross_near:
        hard_fail.append(f"{len(cross_near)} near-duplicate pairs straddle splits")

    # ---------------------------------------- 7. excluded content really gone
    kept_files = {fn for s in SEGS for fn in files_by_split[s]}
    kept_new = set(kept_files)
    new_to_orig = {row["new_name"]: row["file_name"] for row in imgs_manifest.values()}
    orig_kept = {new_to_orig[f] for f in kept_new if f in new_to_orig}
    img_by_orig = {im["file_name"]: im for im in coco["images"]}
    leaked = collections.Counter()
    for f in orig_kept:
        im = img_by_orig[f]
        for _, a in anns_by_image[im["id"]]:
            v = dec[a["category_id"]]
            if v["action"] in ("DEFER", "EXCLUDE_STRUCTURAL", "EXCLUDE_UNRESOLVED"):
                leaked[v["name"]] += 1
            if len(a["segmentation"]) > 1:
                leaked["MULTIPART"] += 1
    R["excluded_content_leaked_into_kept"] = dict(leaked)
    if leaked:
        hard_fail.append(f"excluded content leaked: {dict(leaked)}")

    # ------------------------------------------ 8. duplicate annotation ids
    dup_ok = True
    dup_ids = summary.get("duplicate_annotation_id_audit", [])
    for did in (309, 4040):
        got = [d for d in dup_ids if d["annotation_id"] == did]
        if len(got) != 2:
            dup_ok = False
    R["duplicate_annotation_ids_preserved"] = dup_ok
    R["duplicate_annotation_id_detail"] = dup_ids
    if not dup_ok:
        hard_fail.append("duplicate annotation ids 309/4040 not both preserved")

    # -------------------------------------------- 9. reconciliation vs summary
    rec = summary["reconciliation"]
    R["reconciliation"] = rec
    R["reconciliation_ok"] = bool(rec["sum_equals_input"])
    if not rec["sum_equals_input"]:
        hard_fail.append("annotation reconciliation failed")
    if R["instances_per_class_total"] != summary["instances_per_class_total"]:
        hard_fail.append("instance histogram != conversion summary")
    R["instance_histogram_matches_summary"] = (
        R["instances_per_class_total"] == summary["instances_per_class_total"])

    # ------------------------------- 10. round-trip label re-derivation
    rt_mismatch = []
    clip_stats = collections.Counter()
    for s in SEGS:
        for fn in files_by_split[s]:
            man = imgs_manifest[fn]
            im = img_by_orig[man["file_name"]]
            cw, ch = im["width"], im["height"]
            exp = []
            for _, a in anns_by_image[im["id"]]:
                v = dec[a["category_id"]]
                if v["action"] != "MAP":
                    rt_mismatch.append((fn, "unexpected unmapped")); continue
                if len(a["segmentation"]) > 1:
                    rt_mismatch.append((fn, "multipart present")); continue
                poly = a["segmentation"][0]
                xs, ys = list(poly[0::2]), list(poly[1::2])
                if min(xs) < 0 or min(ys) < 0:
                    clip_stats["negative_vertex"] += 1
                if max(xs) > cw or max(ys) > ch:
                    clip_stats["beyond_w_or_h"] += 1
                if max(xs) == cw or max(ys) == ch:
                    clip_stats["exactly_on_boundary"] += 1
                cx = [min(max(v2, 0.0), cw - 1.0) for v2 in xs]
                cy = [min(max(v2, 0.0), ch - 1.0) for v2 in ys]
                pts = list(zip([v2 / cw for v2 in cx], [v2 / ch for v2 in cy]))
                if len(set(pts)) < 3 or shoelace(pts) < 1.0 / (cw * ch):
                    continue
                norm = [v2 for xy in zip(cx, cy) for v2 in (xy[0] / cw, xy[1] / ch)]
                exp.append(f"{nid[v['to']]} " + " ".join(f"{v2:.6f}" for v2 in norm))
            got = [ln for ln in open(os.path.join(args.out, "labels", s,
                                                   os.path.splitext(fn)[0] + ".txt")
                                     ).read().strip().splitlines() if ln.strip()]
            if got != exp:
                rt_mismatch.append((fn, f"label mismatch got={len(got)} exp={len(exp)}"))
    R["roundtrip_mismatches"] = rt_mismatch[:20]
    R["n_roundtrip_mismatches"] = len(rt_mismatch)
    R["clipping_breakdown"] = dict(clip_stats)
    if rt_mismatch:
        hard_fail.append(f"{len(rt_mismatch)} label files fail round-trip re-derivation")

    # ------------------------------------------------------------- verdict
    R["hard_failures"] = hard_fail
    R["DATA_QUALITY_ACCEPTANCE"] = "PASS" if not hard_fail else "FAIL"

    # ------------------------------------------------------ 11. visualization
    viz_dir = os.path.join(args.out, "viz")
    os.makedirs(viz_dir, exist_ok=True)
    rnd = random.Random(args.seed)

    def load_rows(s, fn):
        lp = os.path.join(args.out, "labels", s, os.path.splitext(fn)[0] + ".txt")
        rows = []
        for ln in open(lp).read().strip().splitlines():
            t = ln.split()
            f = list(map(float, t[1:]))
            rows.append((int(t[0]), list(zip(f[0::2], f[1::2]))))
        return rows

    def overlay(im, rows, cell):
        sc = cell / max(im.size)
        disp = im.resize((max(1, int(im.size[0] * sc)), max(1, int(im.size[1] * sc))), Image.BILINEAR)
        ov = Image.new("RGBA", disp.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        for c, pts in rows:
            q = [(p[0] * disp.size[0], p[1] * disp.size[1]) for p in pts]
            d.polygon(q, outline=COLORS[c], fill=COLORS[c] + (70,))
            d.line(q + [q[0]], fill=COLORS[c], width=3)
        out = Image.alpha_composite(disp.convert("RGBA"), ov).convert("RGB")
        sq = Image.new("RGB", (cell, cell), (255, 255, 255))
        sq.paste(out, ((cell - out.width) // 2, (cell - out.height) // 2))
        return sq

    def render_full(fn, s, cell=420):
        im = Image.open(os.path.join(args.out, "images", s, fn)).convert("RGB")
        return overlay(im, load_rows(s, fn), cell)

    def render_focus(fn, s, cls, cell=420, pad=0.5):
        im = Image.open(os.path.join(args.out, "images", s, fn)).convert("RGB")
        W, H = im.size
        rows = load_rows(s, fn)
        xs, ys = [], []
        for c, pts in rows:
            if c == cls:
                for nx, ny in pts:
                    xs.append(nx * W); ys.append(ny * H)
        if not xs:
            return None
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        w, h = max(x1 - x0, 6), max(y1 - y0, 6)
        box = (max(0, int(x0 - pad * w)), max(0, int(y0 - pad * h)),
               min(W, int(x1 + (1 + pad) * w)), min(H, int(y1 + (1 + pad) * h)))
        crop = im.crop(box)
        if crop.size[0] < 2 or crop.size[1] < 2:
            return None
        rows2 = [(c, [((px * W - box[0]) / crop.size[0], (py * H - box[1]) / crop.size[1])
                      for px, py in pts]) for c, pts in rows]
        return overlay(crop, rows2, cell)

    def render_box(fn, s, box, cell=420):
        im = Image.open(os.path.join(args.out, "images", s, fn)).convert("RGB")
        W, H = im.size
        box = (max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3]))
        crop = im.crop(box)
        if crop.size[0] < 2 or crop.size[1] < 2:
            return None
        rows = load_rows(s, fn)
        rows2 = [(c, [((px * W - box[0]) / crop.size[0], (py * H - box[1]) / crop.size[1])
                      for px, py in pts]) for c, pts in rows]
        return overlay(crop, rows2, cell)

    def montage(cells, path, cols=4, cell=420, title="", captions=None):
        keep = [k for k, c in enumerate(cells) if c is not None]
        cells = [cells[k] for k in keep]
        if captions:
            captions = [captions[k] for k in keep]
        if not cells:
            return None
        rows = (len(cells) + cols - 1) // cols
        board = Image.new("RGB", (cols * cell, rows * cell + 26), (245, 245, 245))
        dr = ImageDraw.Draw(board)
        dr.text((8, 6), title, fill=(0, 0, 0))
        for k, c in enumerate(cells):
            r, col = divmod(k, cols)
            board.paste(c, (col * cell, 26 + r * cell))
            if captions:
                dr.text((col * cell + 8, 26 + r * cell + 6), captions[k], fill=(255, 0, 0))
        board.save(path)
        return path

    # per-class zoom samples (largest instances => alignment is unambiguous)
    per_class = collections.defaultdict(list)
    for s in SEGS:
        for fn in files_by_split[s]:
            for c, pts in load_rows(s, fn):
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                per_class[target[c]].append(((max(xs) - min(xs)) * (max(ys) - min(ys)), fn, s))
    viz_files = []
    for cls in target.values():
        seen, pick = set(), []
        for _a, fn, s in sorted(per_class[cls], reverse=True):
            if fn in seen:
                continue
            seen.add(fn); pick.append((fn, s))
            if len(pick) == 4:
                break
        if not pick:
            continue
        p = montage([render_focus(fn, s, nid[cls]) for fn, s in pick],
                    os.path.join(viz_dir, f"class_{nid[cls]}_{cls}.png"),
                    cols=4, title=f"class {nid[cls]} = {cls} (zoom to largest instances)")
        if p:
            viz_files.append(p)

    # ratified-category spot check: one largest instance per human-ratified
    # native TACO category, shown in the colour of its TARGET class
    ratified = mapping.get("ratified_2026_10_13", {})
    if ratified:
        catname = {c["id"]: c["name"] for c in coco["categories"]}
        best = {}
        for s in SEGS:
            for fn in files_by_split[s]:
                im = img_by_orig[imgs_manifest[fn]["file_name"]]
                W, H = im["width"], im["height"]
                for _, a in anns_by_image[im["id"]]:
                    v = dec[a["category_id"]]
                    if v["action"] != "MAP":
                        continue
                    key = f'{catname[a["category_id"]]}({a["category_id"]})'
                    if key not in ratified:
                        continue
                    poly = a["segmentation"][0]
                    xs, ys = poly[0::2], poly[1::2]
                    area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                    if key not in best or area > best[key][0]:
                        best[key] = (area, s, fn, min(xs), min(ys), max(xs), max(ys), W, H)
        cells, caps = [], []
        for key in ratified:
            if key not in best:
                continue
            _a, s, fn, x0, y0, x1, y1, W, H = best[key]
            pad = max(0.6 * max(x1 - x0, y1 - y0, 8.0), 12)
            cells.append(render_box(fn, s, (int(x0 - pad), int(y0 - pad), int(x1 + pad), int(y1 + pad))))
            caps.append(f"{key} -> {ratified[key]}")
        p = montage(cells, os.path.join(viz_dir, "ratified_mapping_check.png"),
                    cols=4, title="human-ratified categories (mask colour = TARGET class)",
                    captions=caps)
        if p:
            viz_files.append(p)

    # EXIF-6/8 samples (transpose correctness)
    exif_pick = [(fn, s) for s in SEGS for fn in files_by_split[s]
                 if imgs_manifest[fn]["orientation"] in ("3", "6", "8")]
    rnd.shuffle(exif_pick)
    if exif_pick:
        p = montage([render_full(fn, s) for fn, s in exif_pick[:8]],
                    os.path.join(viz_dir, "exif_transposed_samples.png"),
                    cols=4, title="EXIF orientation 3/6/8 baked into pixels (stored == COCO space)")
        if p:
            viz_files.append(p)

    # dense multi-instance images
    crowded = sorted(((len(load_rows(s, fn)), fn, s) for s in SEGS for fn in files_by_split[s]),
                     reverse=True)[:8]
    if crowded:
        p = montage([render_full(fn, s) for _, fn, s in crowded],
                    os.path.join(viz_dir, "crowded_samples.png"),
                    cols=4, title="dense multi-instance samples")
        if p:
            viz_files.append(p)
    R["visualizations"] = viz_files

    with open(os.path.join(args.out, "validation_report.json"), "w") as f:
        json.dump(R, f, indent=1, ensure_ascii=False)

    print(json.dumps({k: R[k] for k in (
        "class_scheme_matches_indoor", "total_images", "pairing",
        "n_ultralytics_parse_failures", "n_illegal_labels", "n_residual_exif",
        "n_stored_size_mismatch", "empty_label_files",
        "cross_split_identical_images", "cross_split_near_duplicate_pairs",
        "excluded_content_leaked_into_kept", "duplicate_annotation_ids_preserved",
        "reconciliation_ok", "instance_histogram_matches_summary",
        "n_roundtrip_mismatches", "clipping_breakdown",
        "instances_per_class_total", "DATA_QUALITY_ACCEPTANCE", "hard_failures")}, indent=1,
        ensure_ascii=False))

    sys.exit(0 if not hard_fail else 1)


if __name__ == "__main__":
    main()
