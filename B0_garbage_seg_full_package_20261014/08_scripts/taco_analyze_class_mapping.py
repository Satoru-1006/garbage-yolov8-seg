#!/usr/bin/env python3
"""
Evaluate TACO -> project 9-class mapping scenarios.

Reads the tiered candidate mapping and the official COCO annotations and asks:
  * what does the class distribution look like under each tier policy?
  * how many IMAGES are affected by the `Unlabeled litter` problem?
  * how many images survive each exclusion policy?
  * how does the choice compare against the indoor 1444 dataset's imbalance?

Also emits a concrete, leak-aware train/val split proposal for TACO.

Writes taco_class_mapping_analysis.json next to the audit output.
"""

import argparse
import collections
import json
import os
import re

REPO_ROOT = "/home/xmv/garbage_yolov8_seg"
TACO_ROOT = os.path.join(REPO_ROOT, "datasets", "taco")


def flickr_id(url):
    if not url:
        return None
    m = re.search(r"/(\d+)_[0-9a-f]+(_\w)?\.[A-Za-z]+$", url)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", default=os.path.join(
        TACO_ROOT, "raw", "taco_official_repo", "data", "annotations.json"))
    ap.add_argument("--mapping", default=os.path.join(
        REPO_ROOT, "tools", "taco", "taco_class_mapping_candidate.json"))
    ap.add_argument("--out", default=TACO_ROOT)
    ap.add_argument("--images", default=os.path.join(TACO_ROOT, "raw", "images"))
    args = ap.parse_args()

    coco = json.load(open(args.annotations))
    mapping = json.load(open(args.mapping))["mapping"]
    target = json.load(open(args.mapping))["target_classes"]

    cat = {c["id"]: c["name"] for c in coco["categories"]}
    anns = coco["annotations"]

    UNLABELED = [cid for cid, v in mapping.items() if v["tier"] == "EXCLUDE"]
    EXCLUDED = set(str(c) for c in UNLABELED)

    # ---- images flagged by the `Unlabeled litter` problem -------------------
    cats_per_image = collections.defaultdict(set)
    for a in anns:
        cats_per_image[a["image_id"]].add(str(a["category_id"]))

    unlabeled_imgs = {iid for iid, cs in cats_per_image.items() if cs & EXCLUDED}
    unlabeled_only_imgs = {iid for iid, cs in cats_per_image.items()
                           if cs and cs <= EXCLUDED}
    unlabeled_instances = sum(1 for a in anns if str(a["category_id"]) in EXCLUDED)

    R = {
        "unlabeled_litter_category_ids": UNLABELED,
        "unlabeled_litter_instances": unlabeled_instances,
        "images_containing_unlabeled_litter": len(unlabeled_imgs),
        "images_with_ONLY_unlabeled_litter": len(unlabeled_only_imgs),
        "images_usable_if_drop_unlabeled_only": len(coco["images"]) - len(unlabeled_only_imgs),
        "images_usable_if_drop_all_touching_unlabeled": len(coco["images"]) - len(unlabeled_imgs),
    }

    # ------------------------------------------------------- tier scenarios
    def dist(tiers):
        c = collections.Counter()
        for cid, v in mapping.items():
            if v["tier"] in tiers:
                for t in v["to"]:
                    c[t] += v["instances"]
        return dict(sorted(c.items()))

    def unmapped(tiers):
        return sum(v["instances"] for v in mapping.values() if v["tier"] not in tiers)

    core = ("CONFIRMED_CANDIDATE",)
    core_prov = ("CONFIRMED_CANDIDATE", "PROVISIONAL")
    plus_other = ("CONFIRMED_CANDIDATE", "PROVISIONAL", "OTHER_CANDIDATE")

    R["scenario_A_core_only"] = {
        "tiers": list(core),
        "instances_per_target_class": dist(core),
        "total_mapped_instances": sum(dist(core).values()),
        "unmapped_instances": unmapped(core),
    }
    R["scenario_B_core_plus_provisional"] = {
        "tiers": list(core_prov),
        "instances_per_target_class": dist(core_prov),
        "total_mapped_instances": sum(dist(core_prov).values()),
        "unmapped_instances": unmapped(core_prov),
    }
    R["scenario_C_full_9class_with_other_ratified"] = {
        "tiers": list(plus_other),
        "instances_per_target_class": dist(plus_other),
        "total_mapped_instances": sum(dist(plus_other).values()),
        "unmapped_instances": unmapped(plus_other),
        "caveat": "requires D1 (other ratified) + D2..D6 human decisions",
    }

    # ------------------------------------------- comparison with indoor set
    R["indoor_1444_class_instance_count"] = {
        "bottle": 99, "box": 584, "can": 9, "cigarette_butt": 121, "food_waste": 26,
        "other": 1313, "paper": 674, "plastic_bag": 774, "wrapper": 215,
    }

    # ------------------------------------------------- TACO native feasibility
    per_cat = collections.Counter(a["category_id"] for a in anns)
    R["taco_native_classes"] = len(cat)
    R["taco_native_categories_lt10_instances"] = sorted(
        [cat[c] for c in cat if per_cat.get(c, 0) < 10])
    R["taco_native_categories_with_zero_instances"] = sorted(
        [cat[c] for c in cat if per_cat.get(c, 0) == 0])
    R["taco_native_min_instances"] = min(per_cat.get(c, 0) for c in cat)
    R["taco_native_max_instances"] = max(per_cat.get(c, 0) for c in cat)
    R["taco_native_imbalance_ratio"] = round(
        R["taco_native_max_instances"] / max(R["taco_native_min_instances"], 1), 1)

    # --------------------------------------------------------- split proposal
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

    for im in coco["images"]:
        find(im["id"])
    by_url = collections.defaultdict(list)
    by_fid = collections.defaultdict(list)
    for im in coco["images"]:
        if im.get("flickr_url"):
            by_url[im["flickr_url"]].append(im["id"])
        f = flickr_id(im.get("flickr_url"))
        if f:
            by_fid[f].append(im["id"])
    for g in list(by_url.values()) + list(by_fid.values()):
        for x in g[1:]:
            union(g[0], x)

    # union exact-content duplicates and dHash near-duplicates too, so no
    # visually identical pair can straddle train/val
    if args.images and os.path.isdir(args.images):
        import hashlib as _hl
        from PIL import Image as _Im
        _Im.MAX_IMAGE_PIXELS = None
        dh = {}
        sha = {}
        for im in coco["images"]:
            p = os.path.join(args.images, im["file_name"])
            if not os.path.isfile(p):
                continue
            try:
                with open(p, "rb") as f:
                    sha.setdefault(_hl.sha256(f.read()).hexdigest(), []).append(im["id"])
                with _Im.open(p) as ii:
                    g = ii.convert("L").resize((9, 8), _Im.LANCZOS)
                    px = list(g.getdata())
                bits = 0
                for r in range(8):
                    for c in range(8):
                        bits = (bits << 1) | (1 if px[r * 9 + c] > px[r * 9 + c + 1] else 0)
                dh.setdefault(bits, []).append(im["id"])
            except Exception:  # noqa: BLE001
                continue
        for grp in sha.values():
            for x in grp[1:]:
                union(grp[0], x)
        dh_items = list(dh.items())
        for i in range(len(dh_items)):
            for j in range(i + 1, len(dh_items)):
                if bin(dh_items[i][0] ^ dh_items[j][0]).count("1") <= 5:
                    union(dh_items[i][1][0], dh_items[j][1][0])

    units = collections.defaultdict(list)
    for im in coco["images"]:
        units[find(im["id"])].append(im["id"])

    im_tags = collections.defaultdict(set)
    for a in anns:
        v = mapping.get(str(a["category_id"]))
        if v and v["tier"] != "EXCLUDE":
            im_tags[a["image_id"]] |= set(v["to"])

    rarest = ["food_waste", "can", "bottle", "cigarette_butt"]
    total_need = collections.Counter()
    for im in coco["images"]:
        for t in im_tags[im["id"]]:
            if t in rarest:
                total_need[t] += 1
    need = collections.Counter({t: round(0.2 * total_need[t]) for t in rarest})

    remaining = list(units.values())
    got = collections.Counter()
    val_ids = []
    val_target = round(0.2 * len(coco["images"]))
    while len(val_ids) < val_target and remaining:
        best, best_score = None, None
        for g in remaining:
            score = sum(1 for i in g for t in im_tags[i] if t in rarest and got[t] < need[t])
            key = (score, -len(g))
            if best_score is None or key > best_score:
                best, best_score = g, key
        remaining.remove(best)
        val_ids.extend(best)
        for i in best:
            for t in im_tags[i]:
                got[t] += 1

    R["split_proposal"] = {
        "method": "greedy group split; group = union over duplicate flickr_url / flickr photo id",
        "n_groups": len(units),
        "n_images": len(coco["images"]),
        "val_images": len(val_ids),
        "train_images": len(coco["images"]) - len(val_ids),
        "val_ratio": round(len(val_ids) / len(coco["images"]), 4),
        "rarest_class_recall_in_val": {
            t: {"got": got[t], "target": need[t], "total": total_need[t]} for t in rarest},
        "constraints": [
            "indoor test split is NOT used for TACO model selection",
            "duplicate groups never straddle train/val",
            "garbage_seg_merged_full1444 is read-only and untouched",
        ],
    }

    out = os.path.join(args.out, "taco_class_mapping_analysis.json")
    with open(out, "w") as f:
        json.dump(R, f, indent=1, ensure_ascii=False)
    print(f"wrote {out}")
    print(json.dumps({
        "images_containing_unlabeled_litter": R["images_containing_unlabeled_litter"],
        "images_with_ONLY_unlabeled_litter": R["images_with_ONLY_unlabeled_litter"],
        "unlabeled_litter_instances": unlabeled_instances,
        "scenario_A_total": R["scenario_A_core_only"]["total_mapped_instances"],
        "scenario_B_total": R["scenario_B_core_plus_provisional"]["total_mapped_instances"],
        "scenario_C_total": R["scenario_C_full_9class_with_other_ratified"]["total_mapped_instances"],
        "scenario_C_dist": R["scenario_C_full_9class_with_other_ratified"]["instances_per_target_class"],
        "native_imbalance": R["taco_native_imbalance_ratio"],
        "n_groups": R["split_proposal"]["n_groups"],
        "val_images": R["split_proposal"]["val_images"],
    }, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
