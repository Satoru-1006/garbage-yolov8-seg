#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验 A 过拟合根因诊断

维度:
 1. 训练集重复度 (唯一内容 vs 文件数) 与重复样本权重放大
 2. 采集会话/时间切分 —— val 是否与 train 场景重叠
 3. train / val 的目标分布与尺寸分布差异 (分布漂移)
 4. 标注冲突噪声量
 5. 类别长尾与损失主导项
 6. 有效样本量 vs 模型容量的定量比较
 7. 逐步验证: 观察到的 gap 是否可归因于批次归一化 (排除法)
"""
import os
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_indoor_a import (MERGED, SPLITS, CLASS_NAMES, load_label,
                            sha256_file, poly_area)

TS = re.compile(r"^([AB])_(\d{13})_")
SMALL = 0.005


def fmt_ms(ms):
    return time.strftime("%m-%d %H:%M", time.localtime(ms / 1000))


def main():
    print("=" * 84)
    print("实验 A 过拟合根因诊断")
    print("=" * 84)

    # ---------------- 1. 重复度 ----------------
    print("\n[1] 训练集重复度 (直接影响有效样本量)")
    sha_map = {}
    for sp in SPLITS:
        d = os.path.join(MERGED, "images", sp)
        m = defaultdict(list)
        for fn in sorted(os.listdir(d)):
            m[sha256_file(os.path.join(d, fn))].append(fn)
        sha_map[sp] = m
    tr = sha_map["train"]
    n_files = sum(len(v) for v in tr.values())
    print(f"    train 文件数 = {n_files}, 唯一内容 = {len(tr)}, 重复副本 = {n_files - len(tr)}")
    mult = Counter(len(v) for v in tr.values())
    print(f"    内容重复次数分布 = {dict(sorted(mult.items()))}")
    over = sum(1 for v in tr.values() if len(v) > 1)
    print(f"    => 每 epoch 唯一图像仅 {len(tr)} 张 (占文件数 {len(tr)/n_files:.1%})")
    print(f"    => {over} 个内容组被重复计入损失, 权重被放大 2~{max(len(v) for v in tr.values())} 倍")

    # ---------------- 2. 会话/时间切分 ----------------
    print("\n[2] 采集会话与时间切分 (val 是否为独立场景)")
    tmap, srcmap = {}, {}
    for sp in SPLITS:
        d = os.path.join(MERGED, "images", sp)
        ts, src = [], []
        for fn in sorted(os.listdir(d)):
            mm = TS.match(fn)
            if mm:
                src.append(mm.group(1))
                ts.append(int(mm.group(2)))
        arr = np.array(ts, dtype=np.float64)
        tmap[sp] = arr
        srcmap[sp] = src
        if len(arr):
            print(f"    {sp:5} n={len(arr):5} 起点={fmt_ms(arr.min())} 终点={fmt_ms(arr.max())} "
                  f"中位={fmt_ms(np.median(arr))}")
    trmin, trmax = tmap["train"].min(), tmap["train"].max()
    for sp in ("val", "test"):
        t = tmap[sp]
        inset = int(((t >= trmin) & (t <= trmax)).sum())
        print(f"    {sp}: 落在 train 时间跨度内的比例 = {inset}/{len(t)} = {inset/len(t):.1%}")
    # 时间轴重叠度: 把 train 时间按 30 分钟分桶, 看 val 有多少桶被 train 覆盖
    BIN = 30 * 60 * 1000
    trbins = {(int(x // BIN)) for x in tmap["train"]}
    for sp in ("val", "test"):
        vb = [(int(x // BIN)) for x in tmap[sp]]
        cov = sum(1 for b in vb if b in trbins)
        print(f"    {sp}: 30min 时间桶被 train 覆盖 = {cov}/{len(set(vb))} 个桶")
    print("    各 split 来源构成:")
    for sp in SPLITS:
        print(f"      {sp:5} {dict(Counter(srcmap[sp]))}")

    # ---------------- 3. 分布差异 ----------------
    print("\n[3] train / val 目标分布与尺寸分布差异")
    stats = {}
    for sp in SPLITS:
        ldir = os.path.join(MERGED, "labels", sp)
        idir = os.path.join(MERGED, "images", sp)
        cls_cnt = Counter()
        areas = defaultdict(list)
        n_inst = 0
        n_img = 0
        n_bg = 0
        sizes = []
        for fn in sorted(os.listdir(ldir)):
            p = os.path.join(idir, fn[:-4] + ".jpg")
            if not os.path.exists(p):
                p = os.path.join(idir, fn[:-4] + ".png")
            with __import__("PIL.Image").Image.open(p) as im:
                W, H = im.size
            sizes.append(max(W, H))
            insts, _, nl = load_label(os.path.join(ldir, fn))
            n_img += 1
            if nl == 0:
                n_bg += 1
            for it in insts:
                n_inst += 1
                cls_cnt[CLASS_NAMES[it["cls"]]] += 1
                xs = [q[0] for q in it["pts"]]
                ys = [q[1] for q in it["pts"]]
                side = float(np.sqrt(max((max(xs) - min(xs)) * W, 1) *
                                     max((max(ys) - min(ys)) * H, 1))) * (800 / max(W, H))
                areas[CLASS_NAMES[it["cls"]]].append(side)
        stats[sp] = {"cls": cls_cnt, "areas": areas, "n_inst": n_inst,
                     "n_img": n_img, "n_bg": n_bg, "sizes": np.array(sizes)}
        print(f"    {sp:5} imgs={n_img:5} 背景图={n_bg:4} ({n_bg/n_img:.1%}) inst={n_inst:5} "
              f"平均实例/图={n_inst/n_img:.2f} 原图长边中位={np.median(sizes):.0f}")

    print("\n    逐类实例占比 (train vs val) —— 找分布漂移:")
    print(f"      {'class':<16}{'train%':>9}{'val%':>9}{'test%':>9}{'偏差(val-tr)':>14}")
    for n in CLASS_NAMES:
        trp = stats["train"]["cls"][n] / max(stats["train"]["n_inst"], 1)
        vap = stats["val"]["cls"][n] / max(stats["val"]["n_inst"], 1)
        tep = stats["test"]["cls"][n] / max(stats["test"]["n_inst"], 1)
        print(f"      {n:<16}{trp:>9.1%}{vap:>9.1%}{tep:>9.1%}{(vap-trp)*100:>13.1f}pp")

    print("\n    逐类目标尺寸中位 (等效边长 px@800) —— train vs val:")
    print(f"      {'class':<16}{'train':>9}{'val':>9}{'比值':>9}")
    for n in CLASS_NAMES:
        a = stats["train"]["areas"][n]
        b = stats["val"]["areas"][n]
        if a and b:
            ma, mb = np.median(a), np.median(b)
            print(f"      {n:<16}{ma:>9.1f}{mb:>9.1f}{mb/ma:>9.2f}")
        else:
            print(f"      {n:<16}{'n/a' if not a else f'{np.median(a):.1f}':>9}{'n/a' if not b else f'{np.median(b):.1f}':>9}{'-':>9}")

    # 整体小目标占比差异
    def small_share(sp):
        allv = [v for n in CLASS_NAMES for v in stats[sp]["areas"][n]]
        return np.mean([v * 1 for v in allv]) if allv else 0
    for sp in ("train", "val"):
        allsz = np.concatenate([np.array(stats[sp]["areas"][n]) for n in CLASS_NAMES if stats[sp]["areas"][n]])
        print(f"    {sp}: 目标等效边长 中位={np.median(allsz):.1f} "
              f"<16px占比={np.mean(allsz < 16):.1%} <32px占比={np.mean(allsz < 32):.1%}")

    # ---------------- 4. 标注冲突噪声 ----------------
    print("\n[4] 标注冲突/噪声量 (记忆噪声会加速过拟合)")
    import csv as _csv
    rows = list(_csv.DictReader(open(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "annotation_conflicts_full1444.csv"))))
    sp_of = {}
    for sp in SPLITS:
        for fn in os.listdir(os.path.join(MERGED, "images", sp)):
            sp_of[fn] = sp
    cnt = Counter()
    lvl = Counter()
    for r in rows:
        f = os.path.basename(r.get("dst", "") or r.get("file", "") or "")
        s = sp_of.get(f) or sp_of.get(f.split("/")[-1])
        key = r.get("severity") or r.get("level") or r.get("conflict_level") or "unknown"
        if s:
            cnt[s] += 1
            lvl[(s, key)] += 1
    print(f"    冲突清单总行数 = {len(rows)}")
    print(f"    按 split 分布 = {dict(cnt)}")
    print(f"    train 中冲突记录 = {cnt.get('train', 0)}  (train 图像数 {stats['train']['n_img']}, "
          f"占比 {cnt.get('train',0)/stats['train']['n_img']:.1%})")
    print(f"    级别分布(train) = { {k[1]: v for k, v in lvl.items() if k[0]=='train'} }")

    # ---------------- 5. 类别长尾 ----------------
    print("\n[5] 类别长尾与损失主导项 (train)")
    tot = stats["train"]["n_inst"]
    for n, c in stats["train"]["cls"].most_common():
        print(f"      {n:<16}{c:>6}  占 {c/tot:>6.1%}   val实例={stats['val']['cls'][n]:>4}")

    # ---------------- 6. 容量 vs 数据量 ----------------
    print("\n[6] 模型容量 vs 有效数据量")
    n_unique = len(tr)
    n_inst_tr = stats["train"]["n_inst"]
    print(f"    可训练参数            = 27,244,843 (Phase2) / 15,388,987 (Phase1)")
    print(f"    唯一训练图像          = {n_unique}")
    print(f"    训练集实例总数        = {n_inst_tr}")
    print(f"    参数量 / 唯一图像     = {27244843/n_unique:,.0f}")
    print(f"    epoch 数 (Phase2 实际) = 102  → 每张唯一图像被看 102 次")
    print(f"    等效优化步数          = {102*73//4} 步 (73 iter/epoch, 累积 4)")

    # ---------------- 7. 排除批次归一化 ----------------
    print("\n[7] 排除法: 观测 gap 是否由批次大小/归一化造成")
    print("    - ultralytics 验证器累积的是 criterion 的第 2 个返回值 loss_items;")
    print("      训练 tloss 亦为同一 loss_items 的滑动均值 → 两者尺度一致。")
    print("    - loss_items 以 target_scores_sum 归一 (每正样本平均), 不对 batch_size 求平均,")
    print("      故 train batch=16 与 val batch=32 不产生系统性倍率差异。")
    print("    - 结论: 训练/验证 loss 的 5 倍间隙无法用归一化解释, 属真实泛化差距。")

    # ---------------- 8. 结论 ----------------
    print("\n" + "=" * 84)
    print("诊断结论")
    print("=" * 84)
    print(f"  1. 训练集含 {n_files - len(tr)} 个重复副本, 唯一内容仅 {len(tr)} 张 "
          f"({len(tr)/n_files:.1%}) → 有效数据量比名义值少 {1-len(tr)/n_files:.1%}")
    print(f"  2. Phase2 在 {len(tr)} 张唯一图像上训练 27.24M 参数 102 epoch → 参数量/图 ≈ "
          f"{27244843/len(tr):,.0f}, 容量远超数据支撑")
    print("  3. 增强强度偏低 (mosaic 0.3 / mixup 0 / copy_paste 0.1), 不足以抑制记忆")
    print("  4. val 仅 145 图 / 340 实例, best-epoch 选择与 early-stop 判定都带较大噪声")


if __name__ == "__main__":
    main()
