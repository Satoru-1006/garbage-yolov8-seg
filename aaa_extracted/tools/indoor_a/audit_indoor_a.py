#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 A: COCO -> 室内垃圾数据集 训练前强制数据审计 (独立复核)

对 datasets/garbage_seg_merged_full1444 做独立复核:
  1. 1444 张图片是否全部存在并可解码
  2. 标签是否为 YOLO 实例分割多边形格式 (非检测框)
  3. 源 A(5类)/B(9类) 映射是否统一正确
  4. 非法类别 ID / 无效多边形 / 越界坐标 / 过小目标
  5. 重复图片 (SHA256 / dHash 近重复)
  6. train/val/test 泄漏检查
  7. 类别图片数 / 实例数 / 目标尺寸分布
  8. 小目标 (烟蒂/碎纸片/小包装袋) 标签质量
  9. 空标签图片统计
 10. 逐类统计与稀缺类别识别

输出: JSON + Markdown 报告
"""
import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict, Counter

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPERIMENT_DIR = os.path.join(
    REPO_ROOT, "experiments", "indoor_a", "A_coco_direct_800_2stage"
)
MERGED = os.path.join(REPO_ROOT, "datasets", "garbage_seg_merged_full1444")
SRC_A = os.path.join(REPO_ROOT, "datasets", "waste_ds_c3_neg_yolo")
SRC_B = os.path.join(REPO_ROOT, "datasets", "half_dataset_yolo")

CLASS_NAMES = ["bottle", "box", "can", "cigarette_butt", "food_waste",
               "other", "paper", "plastic_bag", "wrapper"]
NC = 9

# A 源: 5类 -> 9类 期望映射
A_EXPECTED_MAP = {0: 0, 1: 1, 2: 6, 3: 7, 4: 8}
B_EXPECTED_MAP = {i: i for i in range(9)}

SPLITS = ["train", "val", "test"]

# 小目标阈值 (相对多边形面积占整图比例)
TINY_AREA = 0.0005      # <0.05% 面积 => 极小
SMALL_AREA = 0.005      # <0.5% 面积 => 小目标


def sha256_file(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dhash(path, size=8):
    """8x8 difference hash, 返回 64bit int"""
    try:
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("L").resize((size + 1, size), Image.BILINEAR)
            px = np.asarray(im, dtype=np.int16)
        diff = px[:, 1:] > px[:, :-1]
        bits = 0
        for v in diff.flatten():
            bits = (bits << 1) | int(v)
        return bits
    except Exception:
        return None


def hamming(a, b):
    return bin(a ^ b).count("1")


def poly_area(pts):
    """鞋带公式, pts = [(x,y), ...] 归一化坐标"""
    n = len(pts)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def poly_bbox(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def load_label(txt_path):
    """返回 (instances, errors); instance = dict(cls, pts, ntok)"""
    instances = []
    errors = []
    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    for ln_no, ln in enumerate(lines, 1):
        toks = ln.split()
        if len(toks) < 2:
            errors.append(f"line{ln_no}:too_few_tokens")
            continue
        try:
            cls = float(toks[0])
        except ValueError:
            errors.append(f"line{ln_no}:bad_class_token")
            continue
        if cls != int(cls):
            errors.append(f"line{ln_no}:non_integer_class")
            continue
        cls = int(cls)
        rest = toks[1:]
        if len(rest) % 2 != 0:
            errors.append(f"line{ln_no}:odd_coord_count")
        if len(rest) < 6:
            errors.append(f"line{ln_no}:bbox_or_degenerate(n={len(rest)//2})")
            continue
        try:
            vals = [float(v) for v in rest]
        except ValueError:
            errors.append(f"line{ln_no}:bad_coord_token")
            continue
        pts = [(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)]
        instances.append({
            "cls": cls,
            "pts": pts,
            "ntok": len(toks),
            "npt": len(pts),
            "line": ln_no,
        })
    return instances, errors, len(lines)


def audit(args):
    report = {}
    print("=" * 78)
    print("实验 A 训练前强制数据审计 (独立复核)")
    print("=" * 78)

    # ---------- 0. 结构 ----------
    struct = {}
    for sp in SPLITS:
        idir = os.path.join(MERGED, "images", sp)
        ldir = os.path.join(MERGED, "labels", sp)
        imgs = sorted(os.listdir(idir)) if os.path.isdir(idir) else []
        lbls = sorted(os.listdir(ldir)) if os.path.isdir(ldir) else []
        struct[sp] = {"images": len(imgs), "labels": len(lbls)}
    total_imgs = sum(v["images"] for v in struct.values())
    report["structure"] = struct
    report["total_images"] = total_imgs
    print(f"\n[0] 目录结构: {struct}  合计={total_imgs}")

    # ---------- 1. 解码 + SHA256 + dHash + 尺寸 ----------
    print("[1] 校验图片存在性/可解码性 + 计算 SHA256/dHash ...")
    img_meta = {}    # filename -> dict
    sha_groups = defaultdict(list)
    dhash_map = {}
    n_bad = 0
    for sp in SPLITS:
        idir = os.path.join(MERGED, "images", sp)
        for fn in sorted(os.listdir(idir)):
            p = os.path.join(idir, fn)
            rec = {"split": sp, "path": p, "filename": fn}
            try:
                from PIL import Image
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    im.load()
                    rec["w"], rec["h"] = im.size
                    rec["mode"] = im.mode
                rec["decode"] = "OK"
            except Exception as e:
                rec["decode"] = f"FAIL:{type(e).__name__}"
                n_bad += 1
            rec["sha256"] = sha256_file(p)
            dh = dhash(p)
            rec["dhash"] = dh
            img_meta[fn] = rec
            sha_groups[rec["sha256"]].append(fn)
            if dh is not None:
                dhash_map.setdefault(dh, []).append(fn)

    report["decode_fail"] = n_bad
    report["unique_sha256"] = len(sha_groups)
    print(f"    解码失败={n_bad}  唯一SHA256内容={len(sha_groups)}  (图片总数={total_imgs})")

    # ---------- 2. 标签格式 ----------
    print("[2] 校验标签格式 (YOLO 实例分割多边形 vs 检测框) ...")
    lbl_stats = defaultdict(lambda: {"files": 0, "empty": 0, "instances": 0})
    per_class_inst = Counter()
    per_class_imgs = defaultdict(set)
    per_class_area = defaultdict(list)
    per_class_imgsize = defaultdict(list)
    per_split_class_inst = {sp: Counter() for sp in SPLITS}
    err_counter = Counter()
    bad_classids = Counter()
    oob_coords = 0
    degenerate_poly = 0
    tiny_inst = 0
    small_inst = 0
    total_inst = 0
    empty_files = []
    file_inst_count = {}

    for sp in SPLITS:
        ldir = os.path.join(MERGED, "labels", sp)
        for fn in sorted(os.listdir(ldir)):
            assert fn.endswith(".txt")
            base = fn[:-4]
            imeta = img_meta.get(base + ".jpg") or img_meta.get(base + ".png") or img_meta.get(base + ".jpeg")
            if imeta is None:
                err_counter["label_without_image"] += 1
                continue
            insts, errs, nlines = load_label(os.path.join(ldir, fn))
            lbl_stats[sp]["files"] += 1
            if nlines == 0:
                lbl_stats[sp]["empty"] += 1
                empty_files.append((sp, base))
            for e in errs:
                err_counter[e.split(":")[-1]] += 1
                if "bbox_or_degenerate" in e:
                    degenerate_poly += 1
            file_inst_count[(sp, base)] = len(insts)
            for it in insts:
                cid = it["cls"]
                if cid < 0 or cid >= NC:
                    bad_classids[cid] += 1
                    continue
                pts = it["pts"]
                # 越界检查
                if any((x < -1e-6 or x > 1 + 1e-6 or y < -1e-6 or y > 1 + 1e-6) for x, y in pts):
                    oob_coords += 1
                a = poly_area(pts)
                if a <= 0:
                    degenerate_poly += 1
                total_inst += 1
                lbl_stats[sp]["instances"] += 1
                per_class_inst[CLASS_NAMES[cid]] += 1
                per_split_class_inst[sp][CLASS_NAMES[cid]] += 1
                per_class_imgs[CLASS_NAMES[cid]].add((sp, base))
                per_class_area[CLASS_NAMES[cid]].append(a)
                if imeta:
                    x1, y1, x2, y2 = poly_bbox(pts)
                    per_class_imgsize[CLASS_NAMES[cid]].append(
                        (x2 - x1) * imeta["w"] * (y2 - y1) * imeta["h"])
                if a < TINY_AREA:
                    tiny_inst += 1
                elif a < SMALL_AREA:
                    small_inst += 1

    report["label_format"] = {
        "empty_label_files": sum(v["empty"] for v in lbl_stats.values()),
        "total_instances": total_inst,
        "bad_class_ids": dict(bad_classids),
        "out_of_bound_coords": oob_coords,
        "degenerate_polygons": degenerate_poly,
        "tiny_instances(<0.05%area)": tiny_inst,
        "small_instances(0.05-0.5%area)": small_inst,
        "parse_errors": dict(err_counter),
    }
    print(f"    实例总数={total_inst}  非法类别ID={dict(bad_classids)}  越界坐标={oob_coords}")
    print(f"    退化多边形={degenerate_poly}  极小目标={tiny_inst}  小目标={small_inst}")
    print(f"    空标签文件={sum(v['empty'] for v in lbl_stats.values())}")

    # Detection format check
    det_fmt = err_counter.get("bbox_or_degenerate(n=2)", 0)
    report["is_segmentation_format"] = (det_fmt == 0)
    print(f"    >>> 检测框格式行数(应为0)={det_fmt}  => {'多边形分割格式 OK' if det_fmt==0 else '存在检测框格式!'}")

    # ---------- 3. 源映射复核 ----------
    print("[3] 复核源 A(5类)/B(9类) 映射一致性 ...")
    map_res = {"a_files": 0, "b_files": 0, "a_classids": Counter(), "b_classids": Counter(),
               "a_bad": 0, "b_bad": 0}
    for src, d, expmap in ((SRC_A, "A", A_EXPECTED_MAP), (SRC_B, "B", B_EXPECTED_MAP)):
        ldir = os.path.join(src, "labels")
        for sp in SPLITS:
            p = os.path.join(ldir, sp)
            if not os.path.isdir(p):
                continue
            for fn in os.listdir(p):
                map_res[d.lower() + "_files"] += 1
                insts, errs, _ = load_label(os.path.join(p, fn))
                for it in insts:
                    c = it["cls"]
                    map_res[d.lower() + "_classids"][c] += 1
                    if c not in expmap:
                        map_res[d.lower() + "_bad"] += 1
    report["source_mapping"] = {
        "A_files": map_res["a_files"], "B_files": map_res["b_files"],
        "A_classids_observed": {int(k): v for k, v in sorted(map_res["a_classids"].items())},
        "B_classids_observed": {int(k): v for k, v in sorted(map_res["b_classids"].items())},
        "A_classids_out_of_5cls": map_res["a_bad"],
        "B_classids_out_of_9cls": map_res["b_bad"],
    }
    print(f"    A 文件={map_res['a_files']} 观测类别ID={sorted(map_res['a_classids'])} (期望0-4, 越界={map_res['a_bad']})")
    print(f"    B 文件={map_res['b_files']} 观测类别ID={sorted(map_res['b_classids'])} (期望0-8, 越界={map_res['b_bad']})")

    # 逐条复核最终合并标签类别ID == 源ID经map后
    print("    逐条复核合并标签类别ID与源重映射期望 ...")
    mism = 0
    checked = 0
    for sp in SPLITS:
        ldir = os.path.join(MERGED, "labels", sp)
        for fn in sorted(os.listdir(ldir)):
            base = fn[:-4]
            if base.startswith("A_"):
                src = SRC_A; expmap = A_EXPECTED_MAP
            elif base.startswith("B_"):
                src = SRC_B; expmap = B_EXPECTED_MAP
            else:
                continue
            # 合并后 split 可能与源 split 不同, 因此在源中查找同名文件
            found = None
            for s2 in SPLITS:
                p2 = os.path.join(src, "labels", s2, base + ".txt")
                if os.path.exists(p2):
                    found = p2
                    break
            if found is None:
                continue
            mi, _, _ = load_label(os.path.join(ldir, fn))
            si, _, _ = load_label(found)
            checked += 1
            mc = sorted(x["cls"] for x in mi)
            sc = sorted(expmap.get(x["cls"], -999) for x in si)
            if mc != sc:
                mism += 1
    report["remap_verify"] = {"checked_files": checked, "mismatch": mism}
    print(f"    复核文件={checked}  类别ID不一致={mism}")

    # ---------- 4. 重复与泄漏 ----------
    print("[4] 重复图片与跨 split 泄漏检查 ...")
    dup_groups = {h: fns for h, fns in sha_groups.items() if len(fns) > 1}
    cross_split_sha = {h: fns for h, fns in sha_groups.items()
                       if len({img_meta[f]["split"] for f in fns}) > 1}
    # dHash 近重复 (Hamming<=5)
    dh_keys = list(dhash_map.keys())
    near_pairs = 0
    near_cross = 0
    near_cross_examples = []
    for i in range(len(dh_keys)):
        for j in range(i + 1, len(dh_keys)):
            if hamming(dh_keys[i], dh_keys[j]) <= 5:
                g1 = dhash_map[dh_keys[i]]
                g2 = dhash_map[dh_keys[j]]
                near_pairs += 1
                s1 = {img_meta[f]["split"] for f in g1}
                s2 = {img_meta[f]["split"] for f in g2}
                if not (s1 & s2):
                    near_cross += 1
                    if len(near_cross_examples) < 10:
                        near_cross_examples.append((g1[:2], s1, g2[:2], s2))
    # 同名 A/B 重复 (同一内容)
    ab_same = 0
    for h, fns in sha_groups.items():
        pref = {f.split("_")[0] for f in fns if f[:2] in ("A_", "B_")}
        if len(pref) > 1:
            ab_same += 1
    report["duplicates_leakage"] = {
        "sha_dup_groups": len(dup_groups),
        "sha_cross_split_groups": len(cross_split_sha),
        "near_dup_pairs(hamming<=5)": near_pairs,
        "near_dup_cross_split_pairs": near_cross,
        "same_content_in_both_A_and_B": ab_same,
    }
    print(f"    SHA 重复组={len(dup_groups)}  跨split SHA组={len(cross_split_sha)}")
    print(f"    近重复对={near_pairs}  跨split近重复对={near_cross}")
    print(f"    A/B 同内容重复组={ab_same}")

    # ---------- 5. 类别统计 ----------
    print("[5] 逐类统计 ...")
    cls_stats = {}
    for i, name in enumerate(CLASS_NAMES):
        areas = per_class_area[name]
        sizes = per_class_imgsize[name]
        n_img = len(per_class_imgs[name])
        cls_stats[name] = {
            "id": i,
            "instances": per_class_inst[name],
            "images": n_img,
            "split_instances": {sp: per_split_class_inst[sp][name] for sp in SPLITS},
            "area_pct_median": round(float(np.median(areas)) * 100, 4) if areas else None,
            "area_pct_mean": round(float(np.mean(areas)) * 100, 4) if areas else None,
            "sqrt_area_px_median": round(float(np.median(np.sqrt(sizes))), 1) if sizes else None,
            "small_share(<0.5%)": round(float(np.mean([a < SMALL_AREA for a in areas])), 4) if areas else None,
            "tiny_instances(<0.05%)": int(sum(1 for a in areas if a < TINY_AREA)),
        }
    report["per_class"] = cls_stats
    print(f"\n    {'类别':<16}{'实例':>6}{'图片':>7}{'train':>7}{'val':>5}{'test':>6}{'中位面积%':>11}{'小目标占比':>11}")
    for name in CLASS_NAMES:
        s = cls_stats[name]
        print(f"    {name:<16}{s['instances']:>6}{s['images']:>7}"
              f"{s['split_instances']['train']:>7}{s['split_instances']['val']:>5}"
              f"{s['split_instances']['test']:>6}{str(s['area_pct_median']):>11}"
              f"{str(s['small_share(<0.5%)']):>11}")

    # ---------- 6. 空标签/负样本 ----------
    print("[6] 空标签图片 (负样本候选) ...")
    empty_per_split = Counter(sp for sp, _ in empty_files)
    report["empty_labels"] = {
        "total": len(empty_files),
        "per_split": dict(empty_per_split),
        "ratio": round(len(empty_files) / total_imgs, 4),
    }
    print(f"    空标签总数={len(empty_files)} ({len(empty_files)/total_imgs:.2%})  分布={dict(empty_per_split)}")

    # ---------- 7. 图片尺寸 ----------
    sizes = Counter((m["w"], m["h"]) for m in img_meta.values() if m.get("w"))
    report["image_sizes"] = {f"{k[0]}x{k[1]}": v for k, v in sizes.most_common(10)}
    print(f"[7] 图片尺寸分布 top: {report['image_sizes']}")

    # ---------- 8. 稀缺类别 ----------
    scarcity = sorted(
        [(n, cls_stats[n]["instances"], cls_stats[n]["split_instances"]) for n in CLASS_NAMES],
        key=lambda x: x[1])
    report["scarcity_ranking"] = [{"class": n, "instances": c, "splits": s} for n, c, s in scarcity]
    print("\n[8] 稀缺类别排序 (按实例数):")
    for n, c, s in scarcity:
        flag = "  <== 严重稀缺" if c < 50 else ("  <== 稀少" if c < 200 else "")
        print(f"    {n:<16}{c:>6}  train/val/test={s['train']}/{s['val']}/{s['test']}{flag}")

    noclass_val_test = [n for n in CLASS_NAMES
                        if cls_stats[n]["split_instances"]["val"] == 0
                        or cls_stats[n]["split_instances"]["test"] == 0]
    report["classes_unevaluable_in_val_or_test"] = noclass_val_test

    print("\n" + "=" * 78)
    print("审计摘要")
    print("=" * 78)
    print(f"  图片总数          : {total_imgs}")
    print(f"  解码失败          : {n_bad}")
    print(f"  标签-图片配对     : {'PASS' if total_imgs == sum(v['files'] for v in lbl_stats.values()) else 'FAIL'}")
    print(f"  分割格式          : {'PASS' if report['is_segmentation_format'] else 'FAIL'}")
    print(f"  非法类别ID        : {'PASS' if not bad_classids else 'FAIL'}")
    print(f"  类别重映射        : {'PASS' if mism == 0 else 'FAIL(' + str(mism) + ')'}")
    print(f"  跨split泄漏(SHA)  : {'PASS' if not cross_split_sha else 'FAIL'}")
    print(f"  跨split近重复     : {'PASS' if near_cross == 0 else 'WARN(' + str(near_cross) + ')'}")
    print(f"  越界坐标          : {oob_coords}")
    print(f"  退化多边形        : {degenerate_poly}")
    print(f"  空标签(负样本)    : {len(empty_files)}")
    print(f"  val/test 不可评估类: {noclass_val_test}")

    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 报告已写入: {args.json_out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default=os.path.join(EXPERIMENT_DIR, "data_audit.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    audit(args)
