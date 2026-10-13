#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""审计补充: (a) 修正后的逐条重映射复核; (b) 跨 split 近重复对详情"""
import json
import os
import sys
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_indoor_a import (MERGED, SRC_A, SRC_B, EXPERIMENT_DIR, CLASS_NAMES, SPLITS,
                            A_EXPECTED_MAP, B_EXPECTED_MAP, load_label,
                            dhash, hamming, sha256_file, poly_area)

# (a) 逐条重映射复核
mism = 0
checked = 0
mism_examples = []
map_bad = 0
for sp in SPLITS:
    ldir = os.path.join(MERGED, "labels", sp)
    for fn in sorted(os.listdir(ldir)):
        base = fn[:-4]                     # A_xxx / B_xxx
        if base[:2] == "A_":
            src, expmap, raw = SRC_A, A_EXPECTED_MAP, base[2:]
        elif base[:2] == "B_":
            src, expmap, raw = SRC_B, B_EXPECTED_MAP, base[2:]
        else:
            map_bad += 1
            continue
        found = None
        for s2 in SPLITS:
            p2 = os.path.join(src, "labels", s2, raw + ".txt")
            if os.path.exists(p2):
                found = p2
                break
        if found is None:
            map_bad += 1
            continue
        mi, _, _ = load_label(os.path.join(ldir, fn))
        si, _, _ = load_label(found)
        checked += 1
        got = sorted(x["cls"] for x in mi)
        exp = sorted(expmap.get(x["cls"], -999) for x in si)
        if got != exp:
            mism += 1
            if len(mism_examples) < 5:
                mism_examples.append({"merged": fn, "src": found, "got": got, "exp": exp})

print("=" * 78)
print("[A] 逐条类别重映射复核 (修正版)")
print(f"    复核文件数 = {checked}")
print(f"    类别ID不一致 = {mism}")
print(f"    无法在源定位/前缀异常 = {map_bad}")
for e in mism_examples:
    print(f"      MISMATCH {e['merged']} <- {e['src']}  got={e['got']} exp={e['exp']}")

# (b) 跨 split 近重复对详情
print("=" * 78)
print("[B] 跨 split 近重复对详情 (dHash Hamming<=5)")
img_meta = {}
for sp in SPLITS:
    idir = os.path.join(MERGED, "images", sp)
    for fn in sorted(os.listdir(idir)):
        p = os.path.join(idir, fn)
        img_meta[fn] = {"split": sp, "path": p, "dhash": dhash(p),
                        "sha": sha256_file(p)}

dhash_map = defaultdict(list)
for fn, m in img_meta.items():
    if m["dhash"] is not None:
        dhash_map[m["dhash"]].append(fn)
keys = list(dhash_map.keys())
cross = []
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        d = hamming(keys[i], keys[j])
        if d <= 5:
            g1, g2 = dhash_map[keys[i]], dhash_map[keys[j]]
            s1 = {img_meta[f]["split"] for f in g1}
            s2 = {img_meta[f]["split"] for f in g2}
            if not (s1 & s2):
                cross.append({"dist": d, "grp1": g1, "s1": sorted(s1),
                              "grp2": g2, "s2": sorted(s2),
                              "same_sha": keys[i] == keys[j] or
                                          any(img_meta[a]["sha"] == img_meta[b]["sha"]
                                              for a in g1 for b in g2)})
print(f"    跨 split 近重复组对数 = {len(cross)}")
dist_hist = Counter(c["dist"] for c in cross)
print(f"    汉明距离分布 = {dict(sorted(dist_hist.items()))}")
# 检查是否为纯色/低纹理图片导致的 dHash 碰撞
from PIL import Image
import numpy as np
for c in cross[:30]:
    f1 = c["grp1"][0]
    f2 = c["grp2"][0]
    with Image.open(img_meta[f1]["path"]) as im:
        a1 = np.asarray(im.convert("L").resize((64, 64)), dtype=np.float32)
    with Image.open(img_meta[f2]["path"]) as im:
        a2 = np.asarray(im.convert("L").resize((64, 64)), dtype=np.float32)
    std1, std2 = a1.std(), a2.std()
    # 归一化互相关
    ncc = float(((a1 - a1.mean()) * (a2 - a2.mean())).mean() / (std1 * std2 + 1e-6))
    print(f"    d={c['dist']} {c['s1'][0]:>5}:{f1[:34]:<34} std={std1:6.2f} | "
          f"{c['s2'][0]:>5}:{f2[:34]:<34} std={std2:6.2f} NCC={ncc:6.3f} sameSHA={c['same_sha']}")

out = {"remap_checked": checked, "remap_mismatch": mism, "remap_unresolved": map_bad,
       "cross_split_near_dup_pairs": len(cross)}
json.dump(out, open(os.path.join(EXPERIMENT_DIR, "data_audit_supplement.json"), "w"),
          ensure_ascii=False, indent=2)
print("\n补充审计完成")
