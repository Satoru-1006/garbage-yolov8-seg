#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
garbage_seg_merged_dedup_v2 构建脚本（精确去重 + 标签复核）

上游: datasets/garbage_seg_merged_full1444  (1444 张, 含重复副本, 状态 PROVISIONAL)
下游: datasets/garbage_seg_merged_dedup_v2  (1135 张)

目标
----
  train = 845  (对 full1444 的 train 按图片内容 SHA256 精确去重, 丢弃 309 个重复副本)
  val   = 145  (完全冻结: 图片与标签逐字节复制自 full1444)
  test  = 145  (完全冻结)
  合计  = 1135

原则
----
  * 源数据集(A/B)与 full1444 全程只读: 不修改、不删除、不移动
  * 仅对 train 做 SHA256 精确去重; 近重复(dHash)与连拍组只审查, 不自动删除
  * val/test 完全冻结, 不做任何标签变动, 不做去重
  * 每个内容组只保留 1 张, 该张使用"复核后最完整"的标签
  * 同一内容组内所有副本原本共享同一标签(full1444 已统一); 本脚本按新规则重新判定
    标签完整性, 以"保留正确、完整的实例分割标签"

内容口径说明
----
  full1444 中 719 张图片在合并后被近乎无损地重编码(PSNR≈60, 逐字节不同),
  因此去重口径统一采用 **full1444 自身文件内容的 SHA256**(即训练实际见到的内容),
  而各副本的"原始源标签"通过 tools/merge_manifest_full1444.csv 回取源文件后重映射。

标签解析策略(全部确定性可复现)
----
  0. 空/非空冲突优先: 若同一内容组内既有"空标签"(0 实例)副本、又有"有标注"副本,
     则一律采信有标注版本 (rule=prefer_nonempty)。
     依据: 内容完全相同的重复副本不可能一侧真有目标、另一侧真无目标; 空标签在
     结构上就是"最不完整"的标注, 且把正样本降级为负样本(假阴性)对分割训练危害
     大于假阳性。此类组全部登记为待人工复核 (EMPTY_VS_ANNOTATED)。
  1. 候选池: 若内容组含 B(half_dataset_yolo, 9 类完整体系)副本 → 池 = B 副本;
     否则池 = A(waste_ds_c3_neg_yolo, 5 类)副本。
     依据: A 仅具 5 类标注能力, 对 can/cigarette_butt/food_waste/other 结构性漏标。
  2. 池内多样性: 计算每个候选的实例签名集合。
     - 若存在某候选是其余全部候选签名的严格超集 → 选之 (rule=strict_superset)
     - 否则取 (类别数, 实例数, 面积) 最大者 (rule=max_completeness)
     - 仍并列 → split 优先级 train>val>test, 再按文件名升序 (rule=deterministic_tie)
  3. 跨源冲突(A/B 标注不一致)、A 独有内容组、源内不一致组、空/非空冲突组一律登记为
     待人工复核, 不得视为已通过质量验收。

用法
----
  python tools/dedup_v2_full1444.py --analyze-only   # 只分析, 不写数据集
  python tools/dedup_v2_full1444.py --build          # 构建数据集
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/xmv/garbage_yolov8_seg")
TOOLS = ROOT / "tools"
CFG_PATH = TOOLS / "class_map_full1444.json"
CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))
MANIFEST = TOOLS / "merge_manifest_full1444.csv"

SRC = CFG["sources"]
A_KEY, B_KEY = "dataset_a", "dataset_b"
A_NAME = SRC[A_KEY]["name"]
B_NAME = SRC[B_KEY]["name"]
FULL = Path(CFG["output"]["path"])
OUT = ROOT / "datasets" / "garbage_seg_merged_dedup_v2"

TARGET_CLASSES = {int(k): v for k, v in CFG["target_classes"].items()}
SPLITS = ["train", "val", "test"]
SPLIT_PRIORITY = {s: i for i, s in enumerate(CFG["split_policy"]["priority"])}
REMAP = {A_NAME: {int(k): v for k, v in SRC[A_KEY]["remap"].items()},
         B_NAME: {int(k): v for k, v in SRC[B_KEY]["remap"].items()}}
PREFIX = {A_NAME: SRC[A_KEY]["file_prefix"], B_NAME: SRC[B_KEY]["file_prefix"]}

EXPECT = {"train": 845, "val": 145, "test": 145}


# ------------------------------------------------------------------ 基础
def polygon_area(coords):
    xs, ys = coords[0::2], coords[1::2]
    n = len(xs)
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(a) / 2.0


def read_lines(lp):
    if not Path(lp).exists():
        return []
    return [ln.split() for ln in Path(lp).read_text(encoding="utf-8", errors="ignore").splitlines()
            if ln.strip()]


def norm(lines, ds):
    out = []
    for l in lines:
        out.append((REMAP[ds][int(l[0])],) + tuple(round(float(x), 6) for x in l[1:]))
    return out


def sig(lines):
    return {tuple([l[0]] + [round(float(x), 4) for x in l[1:]]) for l in lines}


def completeness(lines):
    return (len({l[0] for l in lines}), len(lines),
            round(sum(polygon_area(list(l[1:])) for l in lines), 6))


def valid_polygon(line):
    if len(line) < 7 or (len(line) - 1) % 2 != 0:
        return False
    try:
        xs = [float(x) for x in line[1:]]
    except ValueError:
        return False
    return all(0.0 <= x <= 1.0 for x in xs)


def src_label_path(src_img):
    """源图片路径 -> 对应源标签文件路径 (images/<split>/x.jpg -> labels/<split>/x.txt)"""
    return (src_img.parent.parent.parent / "labels" / src_img.parent.name
            / (src_img.stem + ".txt"))


# ------------------------------------------------------------------ 扫描
def load_manifest():
    m = {}
    for r in csv.DictReader(open(MANIFEST, encoding="utf-8")):
        m[Path(r["dst"]).name] = dict(src=Path(r["src"]), ds=r["src_dataset"],
                                      src_split=r["src_split"])
    return m


def scan_full(manifest):
    """以 full1444 实际文件内容为内容口径; 标签回取原始源标签并重映射。"""
    recs = []
    for sp in SPLITS:
        for img in sorted((FULL / "images" / sp).glob("*.jpg")):
            if img.name not in manifest:
                raise RuntimeError(f"full1444 文件不在 manifest 中: {img.name}")
            m = manifest[img.name]
            recs.append(dict(
                name=img.name, dst_split=sp, ds=m["ds"], src=m["src"],
                src_split=m["src_split"],
                sha256=hashlib.sha256(img.read_bytes()).hexdigest(),
                lines=norm(read_lines(src_label_path(m["src"])), m["ds"]),
            ))
    return recs


# ------------------------------------------------------------------ 解析
def resolve_label(pool):
    """pool: list of dict(ds, src_split, name, lines)。返回 (chosen, rule)。"""
    uniq = {}
    for c in sorted(pool, key=lambda x: (SPLIT_PRIORITY[x["src_split"]], x["name"])):
        uniq.setdefault(frozenset(sig(c["lines"])), c)
    cands = list(uniq.values())
    if len(cands) == 1:
        return cands[0], "single_candidate"

    sets = [(c, sig(c["lines"])) for c in cands]
    supers = [c for c, s in sets if all(s > o for _, o in sets if _ is not c)]
    if len(supers) == 1:
        return supers[0], "strict_superset"
    if supers:
        supers.sort(key=lambda x: (completeness(x["lines"]),
                                   -SPLIT_PRIORITY[x["src_split"]], x["name"]), reverse=True)
        return supers[0], "strict_superset_multi"

    def rank(c):
        cls, n, area = completeness(c["lines"])
        return (cls, n, area, -SPLIT_PRIORITY[c["src_split"]], c["name"])

    best = max(rank(c) for c in cands)
    top = [c for c in cands if rank(c) == best]
    if len(top) == 1:
        return top[0], "max_completeness"
    return sorted(top, key=lambda x: x["name"])[0], "deterministic_tie"


def build_groups(recs):
    by_sha = defaultdict(list)
    for r in recs:
        by_sha[r["sha256"]].append(r)

    groups = []
    for sha, rs in sorted(by_sha.items()):
        splits = {r["dst_split"] for r in rs}
        if len(splits) != 1:
            raise RuntimeError(f"内容组 {sha[:12]} 跨 split: {splits}")
        target_split = next(iter(splits))

        a_rs = [r for r in rs if r["ds"] == A_NAME]
        b_rs = [r for r in rs if r["ds"] == B_NAME]

        inner = [ds for ds, sub in ((A_NAME, a_rs), (B_NAME, b_rs))
                 if len({frozenset(sig(x["lines"])) for x in sub}) > 1]

        a_sig = frozenset(x for r in a_rs for x in sig(r["lines"])) if a_rs else None
        b_sig = frozenset(x for r in b_rs for x in sig(r["lines"])) if b_rs else None
        if a_sig is None or b_sig is None:
            status = "single_source"
        elif a_sig == b_sig:
            status = "identical"
        else:
            status = "conflict"

        a_classes = sorted({x[0] for x in a_sig}) if a_sig is not None else []
        b_classes = sorted({x[0] for x in b_sig}) if b_sig is not None else []
        only_in_b = sorted(set(b_classes) - set(a_classes))
        only_in_a = sorted(set(a_classes) - set(b_classes))
        n_a = len(a_sig) if a_sig is not None else 0
        n_b = len(b_sig) if b_sig is not None else 0

        sev = "-"
        if status == "conflict":
            d = abs(n_a - n_b)
            sev = "HIGH" if (d >= 2 or len(only_in_b) >= 2 or len(only_in_a) >= 2) \
                else ("MEDIUM" if (d == 1 or only_in_b or only_in_a) else "LOW")

        # 空/非空冲突: 优先采信有标注版本
        empty_vs_annotated = bool([r for r in rs if r["lines"]]) and \
            bool([r for r in rs if not r["lines"]])
        if empty_vs_annotated:
            pool = [r for r in (b_rs or a_rs) if r["lines"]] or [r for r in rs if r["lines"]]
        else:
            pool = b_rs if b_rs else a_rs
        chosen, rule = resolve_label(pool)
        if empty_vs_annotated:
            rule = "prefer_nonempty" if rule == "single_candidate" else f"prefer_nonempty+{rule}"

        groups.append(dict(
            sha256=sha, target_split=target_split, n_copies=len(rs),
            copies=[r["name"] for r in sorted(rs, key=lambda x: x["name"])],
            datasets=sorted({r["ds"] for r in rs}), n_a=len(a_rs), n_b=len(b_rs),
            status=status, severity=sev, inner_inconsistent=bool(inner),
            inner_sources=inner, n_inst_a=n_a, n_inst_b=n_b,
            only_in_b=only_in_b, only_in_a=only_in_a,
            pool_source=B_NAME if b_rs else A_NAME, rule=rule,
            empty_vs_annotated=empty_vs_annotated,
            chosen=chosen, chosen_lines=chosen["lines"],
            chosen_name=chosen["name"], chosen_ds=chosen["ds"],
        ))
    return groups


def flags(g):
    f = []
    if g["status"] == "conflict":
        f.append("AB_CONFLICT")
    if g["inner_inconsistent"]:
        f.append("WITHIN_SOURCE_INCONSISTENT")
    if g["empty_vs_annotated"]:
        f.append("EMPTY_VS_ANNOTATED")
    if not g["n_b"]:
        f.append("A_ONLY_UNDERLABEL")
    if g["only_in_a"]:
        f.append("A_HAS_CLASS_NOT_IN_B")
    return f


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()

    print("读取 manifest (源->full1444 映射, 只读) ...")
    manifest = load_manifest()
    print(f"  manifest 条目 = {len(manifest)}")

    print("扫描 full1444 (内容 SHA256) 并回取源标签 ...")
    recs = scan_full(manifest)
    print(f"  文件 = {len(recs)}")

    print("按 SHA256 分组并解析标签 ...")
    groups = build_groups(recs)
    dup_groups = [g for g in groups if g["n_copies"] > 1]
    conflict_groups = [g for g in groups if g["status"] == "conflict"]
    inner_groups = [g for g in groups if g["inner_inconsistent"]]
    print(f"  唯一内容 = {len(groups)}  重复组 = {len(dup_groups)}  "
          f"A/B冲突组 = {len(conflict_groups)}  源内不一致组 = {len(inner_groups)}")

    # 与 full1444 已写出标签比较
    changed = []
    for g in groups:
        old = frozenset(tuple([int(float(l[0]))] + [round(float(x), 4) for x in l[1:]])
                        for l in read_lines(FULL / "labels" / g["target_split"] /
                                            (g["chosen_name"].rsplit(".", 1)[0] + ".txt")))
        g["_old"] = old
        if frozenset(sig(g["chosen_lines"])) != old:
            changed.append(g)

    train_groups = [g for g in groups if g["target_split"] == "train"]
    val_groups = [g for g in groups if g["target_split"] == "val"]
    test_groups = [g for g in groups if g["target_split"] == "test"]
    plan_train = len(train_groups)
    plan_val = sum(g["n_copies"] for g in val_groups)
    plan_test = sum(g["n_copies"] for g in test_groups)
    orig_train = sum(g["n_copies"] for g in train_groups)

    print()
    print("=== 计划 ===")
    print(f"  train 去重后 = {plan_train} (原 {orig_train}, 丢弃重复副本 {orig_train - plan_train})")
    print(f"  val 冻结 = {plan_val}   test 冻结 = {plan_test}")
    print(f"  合计 = {plan_train + plan_val + plan_test}")
    print(f"  标签相对 full1444 变更组 = {len(changed)} "
          f"(其中 train = {sum(1 for g in changed if g['target_split'] == 'train')})")
    print(f"  解析规则分布 = {dict(Counter(g['rule'] for g in groups))}")

    cls_train = Counter(l[0] for g in train_groups for l in g["chosen_lines"])
    cls_val = Counter(int(l[0]) for lp in (FULL / "labels" / "val").glob("*.txt")
                      for l in read_lines(lp))
    cls_test = Counter(int(l[0]) for lp in (FULL / "labels" / "test").glob("*.txt")
                       for l in read_lines(lp))
    fmt = lambda c: {TARGET_CLASSES[k]: v for k, v in sorted(c.items())}
    print(f"  train 实例分布 = {fmt(cls_train)}")
    print(f"  val   实例分布 = {fmt(cls_val)}")
    print(f"  test  实例分布 = {fmt(cls_test)}")

    if (plan_train, plan_val, plan_test) != (EXPECT["train"], EXPECT["val"], EXPECT["test"]):
        print(f"ERROR: 计划 {plan_train}/{plan_val}/{plan_test} 与目标 {EXPECT} 不符", file=sys.stderr)
        sys.exit(1)

    if args.analyze_only and not args.build:
        print("\n[analyze-only] 未构建数据集。")
        return

    if OUT.exists():
        print(f"ERROR: 输出目录已存在，拒绝覆盖: {OUT}", file=sys.stderr)
        sys.exit(1)
    for sp in SPLITS:
        (OUT / "images" / sp).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / sp).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for g in sorted(train_groups, key=lambda x: x["chosen_name"]):
        name = g["chosen_name"]
        src_img = FULL / "images" / "train" / name
        if not src_img.exists():
            raise RuntimeError(f"missing full1444 image: {src_img}")
        shutil.copy2(src_img, OUT / "images" / "train" / name)
        out_lines = [" ".join([str(l[0])] + [f"{x:.6g}" for x in l[1:]]) for l in g["chosen_lines"]]
        (OUT / "labels" / "train" / (name.rsplit(".", 1)[0] + ".txt")).write_text(
            ("\n".join(out_lines) + "\n") if out_lines else "", encoding="utf-8")
        manifest_rows.append(dict(split="train", file=name, sha256=g["sha256"],
                                  label_source=g["chosen_ds"], resolution_rule=g["rule"],
                                  n_copies_in_full1444=g["n_copies"],
                                  n_instances=len(out_lines),
                                  label_changed_vs_full1444=frozenset(sig(g["chosen_lines"])) != g["_old"]))

    for sp in ("val", "test"):
        for img in sorted((FULL / "images" / sp).glob("*.jpg")):
            shutil.copy2(img, OUT / "images" / sp / img.name)
            lp = FULL / "labels" / sp / (img.stem + ".txt")
            shutil.copy2(lp, OUT / "labels" / sp / lp.name)
            manifest_rows.append(dict(split=sp, file=img.name,
                                      sha256=hashlib.sha256(img.read_bytes()).hexdigest(),
                                      label_source="frozen_full1444", resolution_rule="frozen",
                                      n_copies_in_full1444="",
                                      n_instances=len(read_lines(lp)),
                                      label_changed_vs_full1444=False))

    yaml_text = (f"path: {OUT}\n\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n")
    for i in sorted(TARGET_CLASSES):
        yaml_text += f"  {i}: {TARGET_CLASSES[i]}\n"
    (OUT / "data.yaml").write_text(yaml_text, encoding="utf-8")

    with open(TOOLS / "dedup_v2_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)

    def unresolved(g):
        return bool(flags(g))

    dup_cols = ["sha256", "sample", "target_split", "n_copies", "datasets", "n_a", "n_b",
                "status", "severity", "n_inst_a", "n_inst_b", "only_in_b", "only_in_a",
                "inner_inconsistent", "inner_sources", "pool_source", "rule",
                "chosen_label_source", "chosen_file", "label_changed_vs_full1444",
                "unresolved", "unresolved_reasons", "copies"]
    dup_rows = []
    for g in sorted(dup_groups, key=lambda x: (x["target_split"], x["chosen_name"])):
        dup_rows.append({
            "sha256": g["sha256"], "sample": g["chosen_name"],
            "target_split": g["target_split"], "n_copies": g["n_copies"],
            "datasets": json.dumps(g["datasets"], ensure_ascii=False),
            "n_a": g["n_a"], "n_b": g["n_b"], "status": g["status"], "severity": g["severity"],
            "n_inst_a": g["n_inst_a"], "n_inst_b": g["n_inst_b"],
            "only_in_b": json.dumps(g["only_in_b"]), "only_in_a": json.dumps(g["only_in_a"]),
            "inner_inconsistent": g["inner_inconsistent"],
            "inner_sources": json.dumps(g["inner_sources"]),
            "pool_source": g["pool_source"], "rule": g["rule"],
            "chosen_label_source": g["chosen_ds"], "chosen_file": g["chosen_name"],
            "label_changed_vs_full1444": (g["target_split"] == "train"
                                          and frozenset(sig(g["chosen_lines"])) != g["_old"]),
            "unresolved": unresolved(g),
            "unresolved_reasons": json.dumps(flags(g)),
            "copies": json.dumps(g["copies"], ensure_ascii=False),
        })
    with open(TOOLS / "dedup_v2_review_duplicate_groups.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=dup_cols)
        w.writeheader()
        w.writerows(dup_rows)

    conf_cols = ["sha256", "sample", "target_split", "n_copies", "status", "severity",
                 "n_inst_a", "n_inst_b", "only_in_b", "only_in_a",
                 "inner_inconsistent", "inner_sources", "pool_source", "rule",
                 "chosen_label_source", "chosen_file", "unresolved", "unresolved_reasons"]
    conf_rows = []
    for g in sorted([g for g in groups if g["status"] == "conflict"
                     or g["inner_inconsistent"] or g["empty_vs_annotated"]],
                    key=lambda x: (x["target_split"], x["chosen_name"])):
        conf_rows.append({
            "sha256": g["sha256"], "sample": g["chosen_name"],
            "target_split": g["target_split"], "n_copies": g["n_copies"],
            "status": g["status"], "severity": g["severity"],
            "n_inst_a": g["n_inst_a"], "n_inst_b": g["n_inst_b"],
            "only_in_b": json.dumps(g["only_in_b"]), "only_in_a": json.dumps(g["only_in_a"]),
            "inner_inconsistent": g["inner_inconsistent"],
            "inner_sources": json.dumps(g["inner_sources"]),
            "pool_source": g["pool_source"], "rule": g["rule"],
            "chosen_label_source": g["chosen_ds"], "chosen_file": g["chosen_name"],
            "unresolved": unresolved(g), "unresolved_reasons": json.dumps(flags(g)),
        })
    with open(TOOLS / "dedup_v2_annotation_conflicts.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=conf_cols)
        w.writeheader()
        w.writerows(conf_rows)

    stats = dict(
        unique_content=len(groups), duplicate_groups=len(dup_groups),
        conflict_groups=len(conflict_groups),
        conflict_high=sum(1 for g in conflict_groups if g["severity"] == "HIGH"),
        conflict_medium=sum(1 for g in conflict_groups if g["severity"] == "MEDIUM"),
        conflict_low=sum(1 for g in conflict_groups if g["severity"] == "LOW"),
        inner_inconsistent_groups=len(inner_groups),
        empty_vs_annotated_groups=sum(1 for g in groups if g["empty_vs_annotated"]),
        empty_vs_annotated_train=sum(1 for g in groups
                                     if g["empty_vs_annotated"] and g["target_split"] == "train"),
        empty_vs_annotated_val=sum(1 for g in groups
                                   if g["empty_vs_annotated"] and g["target_split"] == "val"),
        empty_vs_annotated_test=sum(1 for g in groups
                                    if g["empty_vs_annotated"] and g["target_split"] == "test"),
        a_only_groups=sum(1 for g in groups if not g["n_b"]),
        a_only_groups_train=sum(1 for g in groups
                                if not g["n_b"] and g["target_split"] == "train"),
        duplicate_and_conflict_overlap=sum(1 for g in groups
                                           if g["n_copies"] > 1 and g["status"] == "conflict"),
        unresolved_groups=sum(1 for g in groups if unresolved(g)),
        unresolved_groups_in_train=sum(1 for g in groups if unresolved(g)
                                       and g["target_split"] == "train"),
        label_changed_vs_full1444=len(changed),
        label_changed_in_train=sum(1 for g in changed if g["target_split"] == "train"),
        train_images=plan_train, val_images=plan_val, test_images=plan_test,
        total_images=plan_train + plan_val + plan_test,
        train_dropped_duplicates=orig_train - plan_train,
        rule_histogram=dict(Counter(g["rule"] for g in groups)),
        class_instance_count_train=fmt(cls_train),
        class_instance_count_val=fmt(cls_val),
        class_instance_count_test=fmt(cls_test),
        empty_labels_train=sum(1 for g in train_groups if not g["chosen_lines"]),
        empty_labels_val=sum(1 for lp in (FULL / "labels" / "val").glob("*.txt")
                             if not read_lines(lp)),
        empty_labels_test=sum(1 for lp in (FULL / "labels" / "test").glob("*.txt")
                              if not read_lines(lp)),
        invalid_polygon_train=sum(1 for g in train_groups for l in g["chosen_lines"]
                                  if not valid_polygon(l)),
    )
    (TOOLS / "dedup_v2_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"构建完成 -> {OUT}")
    print(f"  train={plan_train} val={plan_val} test={plan_test} "
          f"合计={plan_train + plan_val + plan_test}")
    print(f"  tools/dedup_v2_manifest.csv")
    print(f"  tools/dedup_v2_review_duplicate_groups.csv ({len(dup_rows)} 组)")
    print(f"  tools/dedup_v2_annotation_conflicts.csv ({len(conf_rows)} 组)")
    print(f"  tools/dedup_v2_stats.json")


if __name__ == "__main__":
    main()
