#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8-Seg 数据集全量合并（1444 张，保留重复样本）

  A: waste_ds_c3_neg_yolo  (5 类)  --remap-->  9 类体系
  B: half_dataset_yolo     (9 类, 基准，类别 ID 不变)

输出: datasets/garbage_seg_merged_full1444

核心约定
  * 源数据集只读：不修改、不删除、不移动源文件
  * 图片用独立文件复制(非硬链接/软链接)，输出侧改动不会回写源数据
  * 新文件统一加 A_ / B_ 前缀，避免同名覆盖
  * 重复图片全部保留（不删样本）
  * 重复组标注冲突时以 B 的 9 类标注为候选基准，并生成待复核记录

split 协调原则
  先构造"独立单位"：同一 SHA256 内容组 ∪ dHash 近重复组 ∪ 时间连拍组。
  同一独立单位的全部图片必须进入同一个 split，从而消除跨 split 泄漏。
  单位内部原始 split 一致者保持原样；存在冲突者按 80/10/10 目标贪心补位。
  独立性优先于比例，绝不为了凑比例拆散独立单位。

用法:
  python tools/merge_full1444.py --analyze-only   # 只分析，不写数据集
  python tools/merge_full1444.py --merge          # 执行合并
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]  # 仓库根目录
TOOLS = ROOT / "tools"
CFG_PATH = TOOLS / "class_map_full1444.json"
CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))

SRC = CFG["sources"]
A_KEY, B_KEY = "dataset_a", "dataset_b"
A_NAME = SRC[A_KEY]["name"]
B_NAME = SRC[B_KEY]["name"]
OUT = Path(CFG["output"]["path"])

TARGET_CLASSES = {int(k): v for k, v in CFG["target_classes"].items()}
SPLITS = ["train", "val", "test"]
SPLIT_PRIORITY = {s: i for i, s in enumerate(CFG["split_policy"]["priority"])}
SPLIT_TARGET = {"train": 0.80, "val": 0.10, "test": 0.10}
REMAP = {A_NAME: {int(k): v for k, v in SRC[A_KEY]["remap"].items()},
         B_NAME: {int(k): v for k, v in SRC[B_KEY]["remap"].items()}}
PREFIX = {A_NAME: SRC[A_KEY]["file_prefix"], B_NAME: SRC[B_KEY]["file_prefix"]}

NEAR_DUP_HAMMING = 5   # 8x8 dHash 近重复阈值
BURST_SECONDS = 3.0    # 连续拍摄判定阈值(秒)


# ------------------------------------------------------------------ 基础
def scan(key):
    ds = SRC[key]
    base = Path(ds["path"])
    recs = []
    for split in SPLITS:
        idir, ldir = base / "images" / split, base / "labels" / split
        if not idir.is_dir():
            continue
        for img in sorted(idir.glob("*.jpg")):
            lp = ldir / (img.stem + ".txt")
            lines = []
            if lp.exists():
                for ln in lp.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if ln.strip():
                        lines.append(ln.split())
            recs.append(dict(ds=ds["name"], split=split, img=img, label=lp,
                             lines=lines,
                             sha256=hashlib.sha256(img.read_bytes()).hexdigest(),
                             size=img.stat().st_size))
    return recs


def norm(line, remap):
    c = int(line[0])
    if c not in remap:
        raise ValueError(f"类别 ID {c} 不在映射表中: {line[:3]}")
    return (remap[c],) + tuple(round(float(x), 6) for x in line[1:])


def dhash_int(path, size=8):
    with Image.open(path) as im:
        im = im.convert("L").resize((size + 1, size), Image.LANCZOS)
        px = list(im.tobytes())
    v, w = 0, size + 1
    for r in range(size):
        for c in range(size):
            v = (v << 1) | (1 if px[r * w + c] < px[r * w + c + 1] else 0)
    return v


def hamming(a, b):
    return bin(a ^ b).count("1")


def burst_ts(name):
    head = name.split("_")[0]
    return int(head) / 1000.0 if head.isdigit() else None


def valid_polygon(line):
    if len(line) < 7 or (len(line) - 1) % 2 != 0:
        return False
    try:
        xs = [float(x) for x in line[1:]]
    except ValueError:
        return False
    return all(0.0 <= x <= 1.0 for x in xs)


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra

    def groups(self):
        g = defaultdict(list)
        for x in self.p:
            g[self.find(x)].append(x)
        return list(g.values())


# ------------------------------------------------------------------ 分组
def build_groups(recs):
    """按 SHA256 分组，并对 A/B 标注做冲突分析。"""
    by_sha = defaultdict(list)
    for r in recs:
        by_sha[r["sha256"]].append(r)

    groups = []
    for sha, rs in by_sha.items():
        a_rs = [r for r in rs if r["ds"] == A_NAME]
        b_rs = [r for r in rs if r["ds"] == B_NAME]

        a_set = {norm(l, REMAP[A_NAME]) for r in a_rs for l in r["lines"]} if a_rs else None
        b_set = {norm(l, REMAP[B_NAME]) for r in b_rs for l in r["lines"]} if b_rs else None

        if a_set is None or b_set is None:
            status = "single_source"
        elif a_set == b_set:
            status = "identical"
        else:
            status = "conflict"

        a_classes = sorted({x[0] for x in a_set}) if a_set is not None else []
        b_classes = sorted({x[0] for x in b_set}) if b_set is not None else []
        only_in_b = sorted(set(b_classes) - set(a_classes))
        only_in_a = sorted(set(a_classes) - set(b_classes))
        n_a = len(a_set) if a_set is not None else 0
        n_b = len(b_set) if b_set is not None else 0

        if status == "conflict":
            sev = abs(n_a - n_b)
            if sev >= 2 or len(only_in_b) >= 2 or len(only_in_a) >= 2:
                review = "HIGH"
            elif sev == 1 or only_in_b or only_in_a:
                review = "MEDIUM"
            else:
                review = "LOW"
        else:
            review = "-"

        # 同一数据集内部、同内容副本之间标注是否一致
        inner_inconsistent = any(
            len(sub) > 1
            and len({tuple(tuple(l) for l in x["lines"]) for x in sub}) > 1
            for sub in (a_rs, b_rs))

        groups.append(dict(
            sha256=sha, n_files=len(rs),
            splits=sorted({r["split"] for r in rs}),
            datasets=sorted({r["ds"] for r in rs}),
            a_files=len(a_rs), b_files=len(b_rs),
            status=status, review=review,
            inner_inconsistent=inner_inconsistent,
            n_inst_a=n_a, n_inst_b=n_b,
            only_in_b=only_in_b, only_in_a=only_in_a,
            a_classes=a_classes, b_classes=b_classes,
            sample=rs[0]["img"].name, sample_ds=rs[0]["ds"],
        ))
    return groups


def near_dup_components(groups, recs):
    """对每个内容组取代表图算 dHash，返回近重复连通分量。"""
    index = {}
    for r in recs:
        index[(r["ds"], r["img"].name)] = r["img"]

    hashed = []
    for g in groups:
        hashed.append((g, dhash_int(index[(g["sample_ds"], g["sample"])])))

    dsu = DSU()
    for g in groups:
        dsu.find(g["sha256"])
    pairs = 0
    for i in range(len(hashed)):
        gi, hi = hashed[i]
        for j in range(i + 1, len(hashed)):
            gj, hj = hashed[j]
            if hamming(hi, hj) <= NEAR_DUP_HAMMING:
                dsu.union(gi["sha256"], gj["sha256"])
                pairs += 1
    return pairs, dsu.groups()


def build_units(groups, recs, nd_components):
    """独立单位 = SHA256 组 ∪ dHash 近重复 ∪ 时间连拍。"""
    dsu = DSU()
    for g in groups:
        dsu.find(g["sha256"])
    for comp in nd_components:                     # 近重复
        first = comp[0]
        for s in comp[1:]:
            dsu.union(first, s)

    timed = [r for r in recs if burst_ts(r["img"].name) is not None]
    timed.sort(key=lambda r: burst_ts(r["img"].name))
    burst_edges = 0
    for i in range(len(timed) - 1):
        t0 = burst_ts(timed[i]["img"].name)
        t1 = burst_ts(timed[i + 1]["img"].name)
        if t1 - t0 <= BURST_SECONDS:               # 时间连拍
            dsu.union(timed[i]["sha256"], timed[i + 1]["sha256"])
            burst_edges += 1
    return dsu.groups(), burst_edges


def assign_splits(units, sha2recs):
    """保留单 split 单位，冲突单位按 80/10/10 贪心补位。"""
    total = sum(len(v) for v in sha2recs.values())
    target = {s: total * SPLIT_TARGET[s] for s in SPLITS}
    counts = Counter()
    sha2split = {}

    kept, pending = 0, []
    for unit in units:
        files = [r for s in unit for r in sha2recs[s]]
        splits = sorted({r["split"] for r in files})
        if len(splits) == 1:
            for s in unit:
                sha2split[s] = splits[0]
            counts[splits[0]] += len(files)
            kept += len(files)
        else:
            pending.append((unit, len(files)))

    for unit, n in sorted(pending, key=lambda x: -x[1]):
        best = max(SPLITS, key=lambda s: target[s] - counts[s])
        for s in unit:
            sha2split[s] = best
        counts[best] += n

    moved = sum(n for _, n in pending)
    return sha2split, dict(counts), kept, moved, len(pending)


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    print("扫描源数据集 ...")
    recs = scan(A_KEY) + scan(B_KEY)
    n_a = sum(1 for r in recs if r["ds"] == A_NAME)
    n_b = sum(1 for r in recs if r["ds"] == B_NAME)
    sha2recs = defaultdict(list)
    for r in recs:
        sha2recs[r["sha256"]].append(r)
    print(f"  A={n_a}  B={n_b}  合计={len(recs)}  唯一内容(SHA256)={len(sha2recs)}")

    print("按 SHA256 分组并做标注冲突分析 ...")
    groups = build_groups(recs)
    conflict_groups = [g for g in groups if g["status"] == "conflict"]
    print(f"  内容组={len(groups)}  标注冲突组={len(conflict_groups)}")

    print("dHash 近重复检测 ...")
    nd_pairs, nd_comp = near_dup_components(groups, recs)
    print(f"  近重复对={nd_pairs}  近重复分量={len(nd_comp)}")

    print("构造独立单位(SHA ∪ 近重复 ∪ 连拍) ...")
    units, burst_edges = build_units(groups, recs, nd_comp)
    multi = [u for u in units if len(u) > 1]
    print(f"  单位数={len(units)}  含多内容组的单位={len(multi)}  连拍边={burst_edges}")

    print("协调 split ...")
    sha2split, split_counts, kept, moved, n_pending = assign_splits(units, sha2recs)
    total = len(recs)
    print(f"  保持原划分的图片={kept}  重新补位的图片={moved} (涉及 {n_pending} 个单位)")
    print(f"  split 分布 = {split_counts}  "
          f"占比 = {{{', '.join(f'{s}:{split_counts[s]/total:.1%}' for s in SPLITS)}}}")

    # ---------------------------------------------------------- 审计
    # 1) 跨 split 泄漏再检查
    leak_sha = sum(1 for sha, rs in sha2recs.items()
                   if len({sha2split[sha] for _ in rs}) > 1)
    leak_unit = sum(1 for u in units if len({sha2split[s] for s in u}) > 1)
    unit_split_mix = sum(1 for u in units
                         if len({r["split"] for s in u for r in sha2recs[s]}) > 1)
    print(f"  跨 split 的 SHA 组={leak_sha}  跨 split 的独立单位={leak_unit}")

    # 2) 标注完整性与合法性（同一内容组全部副本写入同一份协调后的标签）
    empty_labels = 0
    invalid_poly = 0
    cls_count = Counter()
    label_src_used = Counter()
    plan = []
    for sha, rs in sha2recs.items():
        target = sha2split[sha]
        srcs = {x["ds"] for x in rs}
        best_src = B_NAME if B_NAME in srcs else A_NAME        # B 优先
        # 基准副本确定性选取：先 split 优先级(train>val>test)，再按文件名
        cands = sorted([x for x in rs if x["ds"] == best_src],
                       key=lambda x: (SPLIT_PRIORITY[x["split"]], x["img"].name))
        base_lines = cands[0]["lines"]
        remap = REMAP[best_src]
        n_copy = len(rs)
        for l in base_lines:
            cls_count[remap[int(l[0])]] += n_copy
            if not valid_polygon(l):
                invalid_poly += n_copy
        if not base_lines:
            empty_labels += n_copy
        for r in sorted(rs, key=lambda x: (0 if x["ds"] == B_NAME else 1, str(x["img"]))):
            label_src_used[best_src] += 1
            plan.append(dict(rec=r, target_split=target, label_src=best_src,
                             base_lines=base_lines, remap=remap))

    stats = dict(
        source_a_files=n_a, source_b_files=n_b, total_images=total,
        unique_content=len(sha2recs), units=len(units),
        burst_edges=burst_edges, near_dup_pairs=nd_pairs,
        near_dup_components=len(nd_comp),
        conflict_groups=len(conflict_groups),
        conflict_high=sum(1 for g in conflict_groups if g["review"] == "HIGH"),
        conflict_medium=sum(1 for g in conflict_groups if g["review"] == "MEDIUM"),
        conflict_low=sum(1 for g in conflict_groups if g["review"] == "LOW"),
        inner_inconsistent_groups=sum(1 for g in groups if g["inner_inconsistent"]),
        unresolved_review_items=(sum(1 for g in groups if g["inner_inconsistent"])
                                 + sum(1 for g in conflict_groups
                                       if g["review"] in ("HIGH", "MEDIUM"))),
        duplicate_groups=sum(1 for g in groups if g["n_files"] > 1),
        split_kept=kept, split_moved=moved, split_units_moved=n_pending,
        split_distribution=split_counts,
        split_ratio={s: round(split_counts[s] / total, 4) for s in SPLITS},
        leak_sha_groups=leak_sha, leak_units=leak_unit,
        units_with_mixed_source_split=unit_split_mix,
        empty_labels=empty_labels, invalid_polygon=invalid_poly,
        label_source_used=dict(label_src_used),
        class_instance_count={TARGET_CLASSES[k]: v for k, v in sorted(cls_count.items())},
    )

    print()
    print(f"计划写出 = {len(plan)}")
    print(f"空标签(负样本) = {empty_labels}  非法 polygon = {invalid_poly}")
    print(f"同数据集内部标注不一致的组 = {stats['inner_inconsistent_groups']}")
    print(f"实例类别分布 = {stats['class_instance_count']}")

    if args.analyze_only and not args.merge:
        (TOOLS / "merge_analysis_full1444.json").write_text(
            json.dumps({"stats": stats, "conflicts": conflict_groups,
                        "units": [{"size": len(u), "shas": u} for u in units]},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        print("\n[analyze-only] 已写出 merge_analysis_full1444.json，未生成数据集。")
        return

    # ---------------------------------------------------------- 写出
    if OUT.exists():
        print(f"ERROR: 输出目录已存在，拒绝覆盖: {OUT}", file=sys.stderr)
        sys.exit(1)
    for split in SPLITS:
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest = []
    written = Counter()
    for p in plan:
        r = p["rec"]
        split = p["target_split"]
        newname = f"{PREFIX[r['ds']]}_{r['img'].name}"
        dst_img = OUT / "images" / split / newname
        dst_lbl = OUT / "labels" / split / (newname.rsplit(".", 1)[0] + ".txt")
        shutil.copy2(r["img"], dst_img)

        remap = p["remap"]
        out_lines = []
        for l in p["base_lines"]:
            out_lines.append(" ".join([str(remap[int(l[0])])] + l[1:]))
        dst_lbl.write_text(("\n".join(out_lines) + "\n") if out_lines else "",
                           encoding="utf-8")
        written[split] += 1
        manifest.append(dict(src=str(r["img"]), src_dataset=r["ds"],
                             src_split=r["split"], sha256=r["sha256"],
                             dst=str(dst_img), dst_split=split,
                             label_source=p["label_src"],
                             label_lines=len(out_lines)))

    yaml_text = (f"path: {OUT}\n\ntrain: images/train\nval: images/val\n"
                 f"test: images/test\n\nnames:\n")
    for i in sorted(TARGET_CLASSES):
        yaml_text += f"  {i}: {TARGET_CLASSES[i]}\n"
    (OUT / "data.yaml").write_text(yaml_text, encoding="utf-8")

    with open(TOOLS / "merge_manifest_full1444.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)

    cols = ["sha256", "sample", "sample_ds", "n_files", "datasets", "splits",
            "n_inst_a", "n_inst_b", "only_in_b", "only_in_a", "review",
            "inner_inconsistent"]
    to_review = [g for g in groups
                 if g["status"] == "conflict" or g["inner_inconsistent"]]
    with open(TOOLS / "annotation_conflicts_full1444.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for g in sorted(to_review, key=lambda x: str(x["review"])):
            w.writerow({k: (json.dumps(g[k], ensure_ascii=False)
                            if isinstance(g[k], list) else g[k]) for k in cols})

    stats["written"] = dict(written)
    (TOOLS / "merge_stats_full1444.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"合并完成 -> {OUT}")
    print(f"  写出图片 = {dict(written)}  合计={sum(written.values())}")
    print(f"  tools/merge_manifest_full1444.csv")
    print(f"  tools/annotation_conflicts_full1444.csv")


if __name__ == "__main__":
    main()
