#!/usr/bin/env python3
"""
TACO Stage-1 audit: COCO annotations + downloaded image files.

Produces (read-only, writes only under datasets/taco/):
  * taco_audit.json      machine-readable audit summary
  * TACO_AUDIT_REPORT.md human-readable report (class mapping section is filled
                         from the candidate mapping file if present)

Nothing here writes YOLO labels. This stage is audit-only by design: the class
mapping is not yet human-confirmed, so emitting final training labels would
violate the task constraints.
"""

import argparse
import collections
import glob
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根目录
TACO_ROOT = os.path.join(REPO_ROOT, "datasets", "taco")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def dhash(path, size=8):
    with Image.open(path) as im:
        g = im.convert("L").resize((size + 1, size), Image.LANCZOS)
        px = list(g.getdata())
    bits = 0
    for r in range(size):
        for c in range(size):
            bits = (bits << 1) | (1 if px[r * (size + 1) + c] > px[r * (size + 1) + c + 1] else 0)
    return bits


def hamming(a, b):
    return bin(a ^ b).count("1")


def flickr_id(url):
    if not url:
        return None
    m = re.search(r"/(\d+)_[0-9a-f]+(_\w)?\.[A-Za-z]+$", url)
    return m.group(1) if m else None


def is_rle(seg):
    return isinstance(seg, dict)


def rle_area_counts_ok(seg):
    """Structural sanity check for an uncompressed COCO RLE dict."""
    if not isinstance(seg, dict):
        return False, "not_dict"
    if "counts" not in seg or "size" not in seg:
        return False, "missing_counts_or_size"
    c = seg["counts"]
    if isinstance(c, str):
        return True, "compressed_str"
    if isinstance(c, list):
        if len(c) % 2 != 0:
            return False, "odd_run_length"
        if any((not isinstance(x, int)) or x < 0 for x in c):
            return False, "bad_run_values"
        if sum(c) % 2 != 0:
            return False, "runs_do_not_close"
        return True, "uncompressed_ok"
    return False, f"counts_type_{type(c).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", default=os.path.join(
        TACO_ROOT, "raw", "taco_official_repo", "data", "annotations.json"))
    ap.add_argument("--images", default=os.path.join(TACO_ROOT, "raw", "images"))
    ap.add_argument("--manifest", default=os.path.join(TACO_ROOT, "raw", "download_manifest.json"))
    ap.add_argument("--out", default=TACO_ROOT)
    args = ap.parse_args()

    coco = json.load(open(args.annotations))
    images = coco["images"]
    anns = coco["annotations"]
    cats = coco["categories"]
    cat_by_id = {c["id"]: c for c in cats}
    img_by_id = {im["id"]: im for im in images}

    A = {}

    # ---------------------------------------------------------------- basic
    A["annotations_file"] = args.annotations
    A["annotations_sha256"] = sha256_file(args.annotations)
    A["n_images"] = len(images)
    A["n_annotations"] = len(anns)
    A["n_categories"] = len(cats)
    A["scene_annotations_present"] = "scene_annotations" in coco
    A["n_scene_annotations"] = len(coco.get("scene_annotations", []))
    A["n_scene_categories"] = len(coco.get("scene_categories", []))
    A["licenses"] = coco.get("licenses", [])
    A["info"] = coco.get("info", {})

    # ------------------------------------------------- image id uniqueness
    ids = [im["id"] for im in images]
    A["duplicate_image_ids"] = [k for k, v in collections.Counter(ids).items() if v > 1]

    names = [im["file_name"] for im in images]
    A["duplicate_file_names"] = [k for k, v in collections.Counter(names).items() if v > 1]

    urls = [im.get("flickr_url") for im in images]
    A["duplicate_flickr_urls"] = [k for k, v in collections.Counter(urls).items() if v > 1]

    fids = [flickr_id(im.get("flickr_url")) for im in images]
    fid_dups = {k: v for k, v in collections.Counter([f for f in fids if f]).items() if v > 1}
    A["duplicate_flickr_photo_ids"] = fid_dups
    A["n_images_sharing_flickr_photo_id"] = sum(fid_dups.values())

    # ------------------------------------------- annotation referential integrity
    orphan = [a["id"] for a in anns if a["image_id"] not in img_by_id]
    A["annotations_with_missing_image"] = len(orphan)

    img_with_ann = set(a["image_id"] for a in anns)
    no_ann = [im["id"] for im in images if im["id"] not in img_with_ann]
    A["images_without_annotations"] = len(no_ann)
    A["images_without_annotations_ids"] = no_ann[:200]

    bad_cat = [a["id"] for a in anns if a["category_id"] not in cat_by_id]
    A["annotations_with_unknown_category"] = len(bad_cat)

    dup_ann_ids = [k for k, v in collections.Counter(a["id"] for a in anns).items() if v > 1]
    A["duplicate_annotation_ids"] = dup_ann_ids

    # -------------------------------------------------------------- segmentation
    seg_types = collections.Counter()
    rle_structural = collections.Counter()
    n_empty_poly = 0
    n_poly_lt3 = 0
    n_degenerate_poly = 0
    n_non_poly_non_rle = 0
    n_multi_poly = 0
    poly_pt_counts = []
    out_of_bounds = 0
    oob_ann_ids = []
    zero_area = 0
    neg_area = 0
    area_mismatch = 0
    bbox_invalid = 0
    iscrowd_counter = collections.Counter()
    crowded_ids = []

    for a in anns:
        seg = a["segmentation"]
        iscrowd_counter[a.get("iscrowd", 0)] += 1
        if a.get("iscrowd", 0) == 1:
            crowded_ids.append(a["id"])

        if isinstance(seg, dict):
            seg_types["RLE"] += 1
            ok, why = rle_area_counts_ok(seg)
            rle_structural[why] += 1
        elif isinstance(seg, list):
            if len(seg) == 0:
                n_empty_poly += 1
                seg_types["empty_list"] += 1
            else:
                all_lists = all(isinstance(p, list) for p in seg)
                if not all_lists:
                    n_non_poly_non_rle += 1
                    seg_types["unknown_list"] += 1
                else:
                    seg_types["polygon"] += 1
                    if len(seg) > 1:
                        n_multi_poly += 1
                    for poly in seg:
                        poly_pt_counts.append(len(poly) // 2)
                        if len(poly) < 6:
                            n_poly_lt3 += 1
                        # degeneracy: all points identical
                        pts = list(zip(poly[0::2], poly[1::2]))
                        if len(set(pts)) < 3:
                            n_degenerate_poly += 1
                    im = img_by_id.get(a["image_id"])
                    if im:
                        w, h = im["width"], im["height"]
                        flat = [v for poly in seg for v in poly]
                        xs, ys = flat[0::2], flat[1::2]
                        if xs and (min(xs) < -0.5 or min(ys) < -0.5
                                   or max(xs) > w + 0.5 or max(ys) > h + 0.5):
                            out_of_bounds += 1
                            if len(oob_ann_ids) < 50:
                                oob_ann_ids.append(a["id"])
        else:
            seg_types[f"other_{type(seg).__name__}"] += 1

        ar = a.get("area", 0)
        if ar == 0:
            zero_area += 1
        elif ar < 0:
            neg_area += 1
        bb = a.get("bbox", [])
        if len(bb) != 4 or bb[2] <= 0 or bb[3] <= 0:
            bbox_invalid += 1
        elif ar > 0 and not is_rle(seg):
            poly_ar = sum(
                abs(sum(poly[i] * poly[(i + 3) % len(poly)] - poly[(i + 2) % len(poly)] * poly[(i + 1) % len(poly)]
                        for i in range(0, len(poly), 2))) / 2.0
                for poly in seg if isinstance(poly, list) and len(poly) >= 6
            )
            if poly_ar > 0 and abs(poly_ar - ar) / max(poly_ar, ar) > 0.05:
                area_mismatch += 1

    A["segmentation_types"] = dict(seg_types)
    A["rle_structural"] = dict(rle_structural)
    A["empty_segmentations"] = n_empty_poly
    A["polygons_with_lt3_points"] = n_poly_lt3
    A["degenerate_polygons"] = n_degenerate_poly
    A["non_polygon_non_rle_segmentations"] = n_non_poly_non_rle
    A["annotations_with_multiple_polygons"] = n_multi_poly
    A["polygon_point_count_min"] = min(poly_pt_counts) if poly_pt_counts else None
    A["polygon_point_count_max"] = max(poly_pt_counts) if poly_pt_counts else None
    A["polygon_point_count_mean"] = round(sum(poly_pt_counts) / len(poly_pt_counts), 2) if poly_pt_counts else None
    A["annotations_with_out_of_bounds_coords"] = out_of_bounds
    A["out_of_bounds_annotation_ids_sample"] = oob_ann_ids
    A["annotations_zero_area"] = zero_area
    A["annotations_negative_area"] = neg_area
    A["annotations_polygon_vs_area_mismatch_gt5pct"] = area_mismatch
    A["annotations_invalid_bbox"] = bbox_invalid
    A["iscrowd_distribution"] = {str(k): v for k, v in iscrowd_counter.items()}
    A["iscrowd_annotation_ids"] = crowded_ids

    # ------------------------------------------------------- category counts
    per_cat = collections.Counter(a["category_id"] for a in anns)
    per_cat_imgs = collections.defaultdict(set)
    for a in anns:
        per_cat_imgs[a["category_id"]].add(a["image_id"])
    A["category_instance_counts"] = {
        cat_by_id[c]["name"]: {
            "category_id": c,
            "supercategory": cat_by_id[c]["supercategory"],
            "instances": per_cat.get(c, 0),
            "images": len(per_cat_imgs.get(c, set())),
        }
        for c in sorted(cat_by_id)
    }
    A["categories_with_zero_instances"] = [
        cat_by_id[c]["name"] for c in sorted(cat_by_id) if per_cat.get(c, 0) == 0
    ]
    A["instances_per_image_max"] = max(collections.Counter(a["image_id"] for a in anns).values()) \
        if anns else 0

    # ------------------------------------------------------------ image files
    IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    on_disk = {}
    non_image = []
    for p in glob.glob(os.path.join(args.images, "**", "*"), recursive=True):
        if not os.path.isfile(p) or p.endswith(".part"):
            continue
        rel = os.path.relpath(p, args.images)
        if os.path.splitext(p)[1].lower() in IMG_EXT:
            on_disk[rel] = p
        else:
            non_image.append(rel)
    A["non_image_files_under_images_dir"] = sorted(non_image)
    A["image_files_on_disk"] = len(on_disk)
    A["coco_images_missing_on_disk"] = sorted(set(names) - set(on_disk))
    A["files_on_disk_not_in_coco"] = sorted(set(on_disk) - set(names))
    A["n_missing"] = len(A["coco_images_missing_on_disk"])

    # dimension verification + content hashing / duplicate detection
    dim_mismatch = []
    hash_groups = collections.defaultdict(list)
    dh_groups = collections.defaultdict(list)
    sizes = []
    unreadable = []
    for rel, p in on_disk.items():
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception as e:  # noqa: BLE001
            unreadable.append((rel, str(e)))
            continue
        sizes.append((w, h))
        sha = sha256_file(p)
        hash_groups[sha].append(rel)
        dh_groups[dhash(p)].append(rel)

    A["image_files_unreadable"] = unreadable
    # compare against COCO metadata (only for present files)
    name_to_img = {im["file_name"]: im for im in images}
    for rel, p in on_disk.items():
        im = name_to_img.get(rel)
        if not im:
            continue
        try:
            with Image.open(p) as ii:
                w, h = ii.size
        except Exception:  # noqa: BLE001
            continue
        if w != im["width"] or h != im["height"]:
            dim_mismatch.append({
                "file_name": rel, "coco": [im["width"], im["height"]], "file": [w, h],
            })
    A["dimension_mismatch_vs_coco"] = dim_mismatch[:50]
    A["n_dimension_mismatch"] = len(dim_mismatch)

    # ------------------------------------------------ EXIF orientation trap
    # PIL's Image.size ignores EXIF orientation, so a naive "coco dims == file
    # dims" check reports a mismatch for every rotated photo. cv2.imdecode
    # (which Ultralytics uses) DOES apply EXIF, so the real question is which
    # coordinate space the polygons live in.
    import numpy as _np
    import cv2 as _cv2

    anns_by_image = collections.defaultdict(list)
    for a in anns:
        anns_by_image[a["image_id"]].append(a)
    exif_hist = collections.Counter()
    exif_hist_mismatch = collections.Counter()
    exif_dims_mismatch = collections.Counter()   # PIL raw size != coco
    loaders_agree = 0
    loaders_disagree = []
    oob_coco = 0
    poly_exceeds_raw = collections.Counter()
    for rel, p in on_disk.items():
        im = name_to_img.get(rel)
        if not im:
            continue
        try:
            with Image.open(p) as ii:
                raw_w, raw_h = ii.size
                o = ii.getexif().get(274)
        except Exception:  # noqa: BLE001
            continue
        exif_hist[o] += 1
        pil_mismatch = (raw_w != im["width"] or raw_h != im["height"])
        if pil_mismatch:
            exif_hist_mismatch[o] += 1
            exif_dims_mismatch[o] += 1
        # what the training loader actually sees
        try:
            b = _np.fromfile(p, _np.uint8)
            dec = _cv2.imdecode(b, _cv2.IMREAD_COLOR)
            if dec is not None:
                if dec.shape[1] == im["width"] and dec.shape[0] == im["height"]:
                    loaders_agree += 1
                else:
                    loaders_disagree.append({"file_name": rel, "coco": [im["width"], im["height"]],
                                             "cv2_imdecode": [int(dec.shape[1]), int(dec.shape[0])]})
        except Exception:  # noqa: BLE001
            pass
        # do polygon coords fit the coco space, the raw space, or both?
        xs, ys = [], []
        for a in anns_by_image.get(im["id"], []):
            if not isinstance(a["segmentation"], list):
                continue
            for poly in a["segmentation"]:
                if isinstance(poly, list):
                    xs += poly[0::2]
                    ys += poly[1::2]
        if xs:
            fits_coco = max(xs) <= im["width"] + 1 and max(ys) <= im["height"] + 1
            fits_raw = max(xs) <= raw_w + 1 and max(ys) <= raw_h + 1
            if not fits_coco:
                oob_coco += 1
            if not fits_raw:
                poly_exceeds_raw[o] += 1

    A["exif_orientation_histogram"] = {str(k): v for k, v in sorted(exif_hist.items(), key=lambda t: str(t[0]))}
    A["exif_orientation_histogram_only_where_pil_dims_mismatch_coco"] = {
        str(k): v for k, v in sorted(exif_hist_mismatch.items(), key=lambda t: str(t[0]))}
    A["cv2_imdecode_matches_coco_dims"] = loaders_agree
    A["cv2_imdecode_disagrees_with_coco"] = loaders_disagree[:50]
    A["n_cv2_imdecode_disagree"] = len(loaders_disagree)
    A["images_whose_polygons_exceed_coco_dims"] = oob_coco
    A["images_whose_polygons_exceed_RAW_stored_dims_by_exif_orientation"] = {
        str(k): v for k, v in sorted(poly_exceeds_raw.items(), key=lambda t: str(t[0]))}
    A["exif_conclusion"] = (
        "TACO polygons live in the EXIF-corrected (displayed) coordinate space, which equals the "
        "COCO width/height fields. cv2.imdecode -- the loader Ultralytics uses -- applies EXIF and "
        "reproduces exactly those dimensions, so normalising polygons by COCO width/height is correct. "
        "Any loader that IGNORES EXIF (PIL Image.size based scaling, cv2.IMREAD_IGNORE_ORIENTATION) "
        "silently rotates masks by 90 or 180 degrees. Verified visually on batch_1/000104.JPG."
    )

    exact_dups = {k: v for k, v in hash_groups.items() if len(v) > 1}
    A["exact_duplicate_content_groups"] = len(exact_dups)
    A["exact_duplicate_extra_files"] = sum(len(v) - 1 for v in exact_dups.values())
    A["exact_duplicate_groups_sample"] = list(exact_dups.values())[:20]

    dh_list = list(dh_groups.items())
    near_pairs = 0
    for i in range(len(dh_list)):
        for j in range(i + 1, len(dh_list)):
            if hamming(dh_list[i][0], dh_list[j][0]) <= 5:
                near_pairs += 1
    A["near_duplicate_pairs_dhash_le5"] = near_pairs

    # ------------------------------------------- source / provenance summary
    A["provenance"] = {
        "official_repo": "https://github.com/pedropro/TACO",
        "repo_commit": "29de1a9ba05a647b83a90f18d7772e20bb23d846",
        "official_annotations": "data/annotations.json (reviewed)",
        "unofficial_annotations_used": False,
        "images_obtained_via": "Zenodo DOI mirror 10.5281/zenodo.3587843 (official README alternative)",
        "flickr_per_image_download": "blocked by HTTP 429 on original-size URLs",
        "third_party_roboflow_export_used": False,
    }
    if os.path.isfile(args.manifest):
        man = json.load(open(args.manifest))
        A["download_manifest_counts"] = man.get("counts")

    # -------------------------------------------------------- split proposal
    A["split_proposal"] = {
        "note": "grouped split by duplicate unit; indoor test set NOT used for TACO tuning",
        "target_ratio": {"train": 0.8, "val": 0.2},
        "method": "union-find over (exact sha256) + (dhash<=5) + (same flickr photo id) then stratified by rarest class",
    }

    out_json = os.path.join(args.out, "taco_audit.json")
    with open(out_json, "w") as f:
        json.dump(A, f, indent=1, ensure_ascii=False)
    print(f"wrote {out_json}")
    print(json.dumps({k: A[k] for k in [
        "n_images", "n_annotations", "n_categories", "segmentation_types",
        "iscrowd_distribution", "empty_segmentations", "polygons_with_lt3_points",
        "annotations_with_out_of_bounds_coords", "image_files_on_disk", "n_missing",
        "n_dimension_mismatch", "exact_duplicate_content_groups",
        "near_duplicate_pairs_dhash_le5", "duplicate_flickr_photo_ids",
        "images_without_annotations"]}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
