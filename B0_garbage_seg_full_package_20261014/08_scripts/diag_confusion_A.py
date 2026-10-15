#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GT -> 预测 类别错配分析 (定位为何某类 AP=0, 以及误检来自哪里)

对 val 集每个 GT 实例, 找掩码 IoU>=0.5 的最佳匹配预测 (不限类别),
统计 GT 类别 -> 预测类别的错配矩阵; 并统计无任何匹配(漏检)的比例。
"""
import argparse
import os
import warnings
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore")

ROOT = "/home/xmv/garbage_yolov8_seg"
VAL_IMG = os.path.join(ROOT, "datasets/garbage_seg_merged_full1444/images/val")
VAL_LBL = os.path.join(ROOT, "datasets/garbage_seg_merged_full1444/labels/val")
CLASS_NAMES = ["bottle", "box", "can", "cigarette_butt", "food_waste",
               "other", "paper", "plastic_bag", "wrapper"]


def load_gt(p):
    out = []
    if not os.path.exists(p):
        return out
    for ln in open(p):
        t = ln.split()
        if len(t) < 7:
            continue
        v = [float(x) for x in t[1:]]
        out.append((int(float(t[0])), np.array(v).reshape(-1, 2)))
    return out


def rasterize(pts, W, H):
    from PIL import Image, ImageDraw
    im = Image.new("1", (W, H), 0)
    ImageDraw.Draw(im).polygon([tuple(q) for q in pts * np.array([W, H])], fill=1)
    return np.asarray(im, dtype=bool)


def iou(a, b):
    i = np.logical_and(a, b).sum()
    u = np.logical_or(a, b).sum()
    return float(i / u) if u else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    import cv2
    from ultralytics import YOLO
    model = YOLO(a.weights)

    files = sorted(f for f in os.listdir(VAL_IMG) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    conf_mat = Counter()      # (gt_cls, pred_cls) -> n
    missed = Counter()        # gt_cls -> n
    total = Counter()
    pred_used = Counter()
    # 额外: 训练集上同口径统计, 用于对比"记忆"强度
    for split, idir, ldir in (("val", VAL_IMG, VAL_LBL),
                              ("train", os.path.join(ROOT, "datasets/garbage_seg_merged_full1444/images/train"),
                               os.path.join(ROOT, "datasets/garbage_seg_merged_full1444/labels/train"))):
        fl = sorted(f for f in os.listdir(idir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if split == "train":
            fl = fl[:300]      # 抽样 300 张, 够统计
        cm, ms, tt = Counter(), Counter(), Counter()
        for fn in fl:
            p = os.path.join(idir, fn)
            gts = load_gt(os.path.join(ldir, fn[:-4] + ".txt"))
            if not gts:
                continue
            with __import__("PIL.Image").Image.open(p) as im:
                W, H = im.size
            r = model.predict(source=p, imgsz=800, conf=a.conf, iou=0.7,
                              retina_masks=True, verbose=False, device=0)[0]
            pm, pc = [], []
            if r.masks is not None and len(r.masks):
                pm = r.masks.data.cpu().numpy().astype(bool)
                pc = r.boxes.cls.cpu().numpy().astype(int)
            for gc, pts in gts:
                tt[CLASS_NAMES[gc]] += 1
                gm = cv2.resize(rasterize(pts, W, H).astype(np.uint8),
                                (pm[0].shape[1], pm[0].shape[0]),
                                interpolation=cv2.INTER_NEAREST).astype(bool) if len(pm) else None
                best, bcls = 0.0, -1
                for k in range(len(pm)):
                    v = iou(gm, pm[k])
                    if v > best:
                        best, bcls = v, int(pc[k])
                if best >= 0.5:
                    cm[(CLASS_NAMES[gc], CLASS_NAMES[bcls])] += 1
                else:
                    ms[CLASS_NAMES[gc]] += 1
        print(f"\n{'='*80}\n[{split}] conf>={a.conf}  统计样本数={len(fl)}  GT实例={sum(tt.values())}")
        print(f"{'='*80}")
        print(f"  {'GT类别':<16}{'实例':>6}{'命中(任意类)':>13}{'漏检':>7}{'同类命中':>10}{'跨类混淆':>10}")
        for n in CLASS_NAMES:
            if tt[n] == 0:
                continue
            hit = sum(v for k, v in cm.items() if k[0] == n)
            same = cm.get((n, n), 0)
            print(f"  {n:<16}{tt[n]:>6}{hit:>13}{ms[n]:>7}{same:>10}{hit-same:>10}")
        print(f"\n  错配明细 (GT -> 预测, 仅跨类, 计数>=1):")
        cross = [(k, v) for k, v in cm.items() if k[0] != k[1] and v > 0]
        for (g, p_), v in sorted(cross, key=lambda x: -x[1]):
            print(f"      {g:<16} -> {p_:<16} {v}")
        print(f"\n  误检(预测了但没对应GT)按类统计: 见下方 val 汇总")
        if split == "val":
            pred_used.update(Counter(CLASS_NAMES[int(c)] for c in pc))
            conf_mat = cm
            missed = ms
            total = tt

    # val 的误检: 对 val 统计预测类别总数
    pc_all = Counter()
    for fn in files:
        p = os.path.join(VAL_IMG, fn)
        r = model.predict(source=p, imgsz=800, conf=a.conf, iou=0.7,
                          retina_masks=True, verbose=False, device=0)[0]
        if r.masks is not None and len(r.masks):
            for c in r.boxes.cls.cpu().numpy().astype(int):
                pc_all[CLASS_NAMES[int(c)]] += 1
    print(f"\n{'='*80}\n[val] 预测总数 vs GT 总数 (conf>={a.conf})\n{'='*80}")
    print(f"  {'类别':<16}{'GT':>6}{'预测':>7}{'差':>8}")
    for n in CLASS_NAMES:
        print(f"  {n:<16}{total[n]:>6}{pc_all[n]:>7}{pc_all[n]-total[n]:>8}")


if __name__ == "__main__":
    main()
