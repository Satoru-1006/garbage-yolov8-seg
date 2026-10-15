#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验 B0 最终评估 (仅使用 val 集; test 集保持锁定)

数据集: datasets/garbage_seg_merged_dedup_v2 (845/145/145, 9 类)

1. 逐类 Mask / Box 的 P, R, mAP50, mAP50-95
2. 按目标尺寸分层的掩码召回率 (小目标识别效果)
3. 推理速度
"""
import argparse
import json
import os
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

ROOT = "/home/xmv/garbage_yolov8_seg"
DATA = os.path.join(ROOT, "datasets/garbage_seg_merged_dedup_v2/data.yaml")
VAL_IMG = os.path.join(ROOT, "datasets/garbage_seg_merged_dedup_v2/images/val")
VAL_LBL = os.path.join(ROOT, "datasets/garbage_seg_merged_dedup_v2/labels/val")
CLASS_NAMES = ["bottle", "box", "can", "cigarette_butt", "food_waste",
               "other", "paper", "plastic_bag", "wrapper"]

# 等效边长(px, 800 输入尺度) 分桶
BUCKETS = [(0, 16), (16, 32), (32, 64), (64, 128), (128, 1e9)]


def load_gt(lbl_path):
    out = []
    if not os.path.exists(lbl_path):
        return out
    for ln in open(lbl_path):
        t = ln.split()
        if len(t) < 7:
            continue
        c = int(float(t[0]))
        v = [float(x) for x in t[1:]]
        pts = np.array(v).reshape(-1, 2)
        out.append({"cls": c, "pts": pts})
    return out


def rasterize(pts, W, H):
    from PIL import Image, ImageDraw
    im = Image.new("1", (W, H), 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in pts * np.array([W, H])], fill=1)
    return np.asarray(im, dtype=bool)


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--split", default="val")
    ap.add_argument("--rect", action="store_true",
                    help="rect 推理 (与训练期验证器一致)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--size-analysis", action="store_true")
    a = ap.parse_args()

    from ultralytics import YOLO
    res = {"weights": a.weights, "imgsz": a.imgsz, "split": a.split,
           "data": DATA, "data_version": "garbage_seg_merged_dedup_v2",
           "data_quality_status": "PROVISIONAL",
           "protocol": {"conf": 0.001, "iou": 0.7, "batch": a.batch,
                        "rect": bool(a.rect), "max_det": 300}}
    model = YOLO(a.weights)

    # ---------- 标准验证 ----------
    m = model.val(data=DATA, imgsz=a.imgsz, split=a.split, batch=a.batch, device=0,
                  rect=bool(a.rect), plots=True,
                  project=os.path.dirname(os.path.abspath(a.out)),
                  name=os.path.basename(a.out).replace(".json", "_val"),
                  exist_ok=True, verbose=True, conf=0.001, iou=0.7)

    res["overall"] = {
        "mask": {"precision": float(m.seg.mp), "recall": float(m.seg.mr),
                 "mAP50": float(m.seg.map50), "mAP50_95": float(m.seg.map)},
        "box": {"precision": float(m.box.mp), "recall": float(m.box.mr),
                "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map)},
        "fitness": float(getattr(m, "fitness", float("nan"))),
        "speed_ms_per_img": {k: float(v) for k, v in m.speed.items()},
    }

    per_class = {}
    for name, idxs in (("mask", m.seg.ap_class_index), ("box", m.box.ap_class_index)):
        src = m.seg if name == "mask" else m.box
        d = {}
        for j, ci in enumerate(idxs):
            cname = CLASS_NAMES[int(ci)]
            d[cname] = {
                "class_id": int(ci),
                "precision": float(src.p[j]), "recall": float(src.r[j]),
                "AP50": float(src.ap50[j]), "AP50_95": float(src.ap[j]),
            }
        per_class[name] = d
    res["per_class"] = per_class

    # 补全 val 中无实例的类别
    res["classes_with_zero_val_instances"] = [
        n for n in CLASS_NAMES if n not in per_class["mask"]]
    # 排除不可评估类别后的宏平均 (参考指标)
    valid = [d for n, d in per_class["mask"].items()
             if n not in res["classes_with_zero_val_instances"]]
    if valid:
        res["mask_macro_excluding_unscorable"] = {
            "n_classes": len(valid),
            "mAP50": round(float(np.mean([d["AP50"] for d in valid])), 4),
            "mAP50_95": round(float(np.mean([d["AP50_95"] for d in valid])), 4),
        }

    # ---------- 尺寸分层掩码召回 ----------
    if a.size_analysis:
        import cv2
        print("\n[尺寸分层掩码召回率分析] 仅 val 集", flush=True)
        files = sorted(f for f in os.listdir(VAL_IMG)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        stats = {f"{int(lo)}-{int(hi) if hi<1e9 else 'inf'}px": {"gt": 0, "hit": 0}
                 for lo, hi in BUCKETS}
        per_cls_stats = {n: {f"{int(lo)}-{int(hi) if hi<1e9 else 'inf'}px": {"gt": 0, "hit": 0}
                             for lo, hi in BUCKETS} for n in CLASS_NAMES}
        iou_hist = []
        for fn in files:
            p = os.path.join(VAL_IMG, fn)
            gts = load_gt(os.path.join(VAL_LBL, fn[:-4] + ".txt"))
            with __import__("PIL.Image").Image.open(p) as im:
                W, H = im.size
            r = model.predict(source=p, imgsz=a.imgsz, conf=0.001, iou=0.7,
                              retina_masks=True, verbose=False, device=0)[0]
            pred_masks, pred_cls = [], []
            if r.masks is not None and len(r.masks):
                pm = r.masks.data.cpu().numpy().astype(bool)
                pc = r.boxes.cls.cpu().numpy().astype(int)
                pred_masks, pred_cls = pm, pc
            n_pred = len(pred_masks)
            for g in gts:
                x1, y1 = g["pts"].min(0)
                x2, y2 = g["pts"].max(0)
                side = float(np.sqrt(max((x2 - x1) * W, 1) * max((y2 - y1) * H, 1)))
                side800 = side * (a.imgsz / max(W, H))
                best = 0.0
                if n_pred:
                    gmask = cv2.resize(
                        rasterize(g["pts"], W, H).astype(np.uint8),
                        (pred_masks[0].shape[1], pred_masks[0].shape[0]),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
                    for k in range(n_pred):
                        if pred_cls[k] != g["cls"]:
                            continue
                        best = max(best, mask_iou(gmask, pred_masks[k]))
                    iou_hist.append(best)
                cname = CLASS_NAMES[g["cls"]]
                for lo, hi in BUCKETS:
                    key = f"{int(lo)}-{int(hi) if hi < 1e9 else 'inf'}px"
                    if lo <= side800 < hi:
                        stats[key]["gt"] += 1
                        per_cls_stats[cname][key]["gt"] += 1
                        if best >= 0.5:
                            stats[key]["hit"] += 1
                            per_cls_stats[cname][key]["hit"] += 1
                        break
        out = {}
        for k, v in stats.items():
            out[k] = {"gt_instances": v["gt"], "matched@maskIoU0.5": v["hit"],
                      "recall": round(v["hit"] / v["gt"], 4) if v["gt"] else None}
        res["recall_by_size"] = out
        res["recall_by_size_per_class"] = {
            n: {k: {"gt": v[k]["gt"], "hit": v[k]["hit"],
                    "recall": round(v[k]["hit"] / v[k]["gt"], 4) if v[k]["gt"] else None}
                for k in v if v[k]["gt"] > 0}
            for n, v in per_cls_stats.items()
            if any(v[k]["gt"] > 0 for k in v)}
        if iou_hist:
            res["mask_iou_all_gt"] = {
                "n": len(iou_hist), "mean_best_iou": round(float(np.mean(iou_hist)), 4),
                "recall@0.5": round(float(np.mean(np.array(iou_hist) >= 0.5)), 4),
                "recall@0.75": round(float(np.mean(np.array(iou_hist) >= 0.75)), 4)}

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=2)
    print(f"\n评估结果已写入 {a.out}")

    print("\n===== 总体 (val) =====")
    for k, v in res["overall"].items():
        print(f"  {k}: {v}")
    print("\n===== 逐类 Mask =====")
    print(f"  {'class':<16}{'P':>8}{'R':>8}{'AP50':>9}{'AP50-95':>10}")
    for n in CLASS_NAMES:
        d = per_class["mask"].get(n)
        print(f"  {n:<16}" + (f"{d['precision']:>8.4f}{d['recall']:>8.4f}"
                             f"{d['AP50']:>9.4f}{d['AP50_95']:>10.4f}" if d else "   (val 无实例)"))


if __name__ == "__main__":
    main()
