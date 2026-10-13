#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""审计补充 C: 严格复验 27 对跨 split 近重复是否为真实泄漏

采用更强判据:
  - 保持宽高比的 128x128 灰度归一化互相关 (NCC)
  - 16x16 dHash Hamming
  - 双向核查: 同一 split 内的近重复对作为 dHash 假阳性基线
"""
import os
import sys
import json
from collections import defaultdict, Counter

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_indoor_a import MERGED, EXPERIMENT_DIR, SPLITS

SPLITS_ = SPLITS


def load_gray(path, size=128):
    with Image.open(path) as im:
        im = im.convert("L")
        w, h = im.size
        s = size / max(w, h)
        im = im.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32)
    # letterbox 到 size x size
    canvas = np.zeros((size, size), dtype=np.float32)
    canvas[:a.shape[0], :a.shape[1]] = a
    return canvas


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-6 else 0.0


def dhash16(path, size=16):
    with Image.open(path) as im:
        im = im.convert("L").resize((size + 1, size), Image.BILINEAR)
        px = np.asarray(im, dtype=np.int16)
    bits = 0
    for v in (px[:, 1:] > px[:, :-1]).flatten():
        bits = (bits << 1) | int(v)
    return bits


def ham(a, b):
    return bin(a ^ b).count("1")


# 收集全部图片
files = []
meta = {}
for sp in SPLITS_:
    idir = os.path.join(MERGED, "images", sp)
    for fn in sorted(os.listdir(idir)):
        p = os.path.join(idir, fn)
        files.append(p)
        meta[p] = {"split": sp, "fn": fn}

print(f"图片总数 {len(files)}")

# 16x16 dHash 分组 (Hamming<=10, 更接近真实近重复)
dh = {}
for p in files:
    dh[p] = dhash16(p)
keys = list(dh.keys())
groups = defaultdict(list)
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        if ham(dh[keys[i]], dh[keys[j]]) <= 10:
            groups[keys[i]].append(keys[j])

# 并查集合并
parent = {p: p for p in files}


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb


for a, bs in groups.items():
    for b in bs:
        union(a, b)

comp = defaultdict(list)
for p in files:
    comp[find(p)].append(p)

multi = [v for v in comp.values() if len(v) > 1]
print(f"16x16 dHash(H<=10) 近重复组数 = {len(multi)}")

cross_real = []
same = 0
for g in multi:
    sps = {meta[p]["split"] for p in g}
    if len(sps) > 1:
        # 严格复验: 组内任意两两 NCC
        gray = {p: load_gray(p) for p in g}
        worst = 1.0
        best = -1.0
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                v = ncc(gray[g[i]], gray[g[j]])
                best = max(best, v)
                worst = min(worst, v)
        dims = {p: Image.open(p).size for p in g}
        cross_real.append({
            "files": [meta[p]["fn"] for p in g],
            "splits": sorted(sps),
            "ncc_max": round(best, 4),
            "ncc_min": round(worst, 4),
            "dims": {meta[p]["fn"]: dims[p] for p in g},
            "verdict": "REAL_LEAK" if best >= 0.90 else "FALSE_POSITIVE",
        })
    else:
        same += 1

print(f"跨 split 组数 = {len(cross_real)}   组内同 split 组数 = {same}")
real = [c for c in cross_real if c["verdict"] == "REAL_LEAK"]
print(f"判定为真实泄漏(组内最大NCC>=0.90) = {len(real)}")
for c in sorted(cross_real, key=lambda x: -x["ncc_max"])[:40]:
    print(f"  ncc_max={c['ncc_max']:.3f} splits={c['splits']} n={len(c['files'])} {c['verdict']}")
    for f in c["files"][:4]:
        print(f"        {f}  dims={c['dims'][f]}")

json.dump({"n_groups_multi": len(multi),
           "n_cross_split_groups": len(cross_real),
           "n_real_leak": len(real),
           "details": cross_real},
          open(os.path.join(EXPERIMENT_DIR, "data_audit_leakcheck.json"), "w"),
          ensure_ascii=False, indent=2)
print("\n严格泄漏复验完成")
