#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only acceptance audit for garbage_seg_merged_v3.

This verifier performs no training and never edits images or labels.  It
checks structure, label syntax, exact-content uniqueness, cross-split
dHash/burst/session separation, class coverage, and the human-admission gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


SPLITS = ("train", "val", "test")
NAMES = {0: "bottle", 1: "box", 2: "can", 3: "cigarette_butt",
         4: "food_waste", 5: "other", 6: "paper", 7: "plastic_bag",
         8: "wrapper"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_label(path: Path):
    if not path.exists():
        return None
    rows = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.split()
        if not parts:
            continue
        try:
            cls = int(parts[0])
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            return ["PARSE_ERROR"]
        rows.append((cls, coords))
    return rows


def dhash(path: Path) -> int:
    with Image.open(path) as im:
        im = im.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        px = list(im.tobytes())
    value = 0
    for r in range(8):
        for c in range(8):
            value = (value << 1) | int(px[r * 9 + c] < px[r * 9 + c + 1])
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def ts(name: str):
    try:
        return int(name.split("_", 1)[0]) / 1000.0
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2] /
                    "datasets" / "garbage_seg_merged_v3")
    ap.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[3])
    args = ap.parse_args()

    results = {}
    counts = {}
    rows = list(csv.DictReader((args.root / "tools" / "v3_manifest.csv").open(encoding="utf-8")))
    all_sha = defaultdict(list)
    all_times = defaultdict(list)
    distribution = {sp: Counter() for sp in SPLITS}
    invalid = []
    missing_pairs = []
    decode_fail = []
    empty = Counter()
    d_hashes = {sp: {} for sp in SPLITS}
    for sp in SPLITS:
        images = sorted((args.root / "images" / sp).glob("*.jpg"))
        labels = sorted((args.root / "labels" / sp).glob("*.txt"))
        counts[sp] = len(images)
        results[f"{sp}_image_count"] = len(images) == {"train": 850, "val": 106, "test": 107}[sp]
        results[f"{sp}_label_count"] = len(labels) == len(images)
        for image in images:
            label = args.root / "labels" / sp / (image.stem + ".txt")
            if not label.exists():
                missing_pairs.append(str(image)); continue
            try:
                with Image.open(image) as im:
                    im.verify()
            except Exception as exc:
                decode_fail.append((str(image), str(exc)))
            lines = read_label(label)
            if lines is None or lines == ["PARSE_ERROR"]:
                invalid.append((str(label), "parse")); continue
            if not lines:
                empty[sp] += 1
            for cls, coords in lines:
                distribution[sp][cls] += 1
                if cls not in NAMES:
                    invalid.append((str(label), f"class={cls}"))
                if len(coords) < 6 or len(coords) % 2:
                    invalid.append((str(label), "polygon_points"))
                elif any(x < 0 or x > 1 for x in coords):
                    invalid.append((str(label), "coordinate_bounds"))
            image_sha = sha(image)
            all_sha[image_sha].append((sp, image.name))
            t = ts(image.name)
            if t is not None:
                all_times[sp].append((t, image.name))
            d_hashes[sp][image.name] = dhash(image)

    results["total_images"] = sum(counts.values()) == 1063
    results["image_label_pairing"] = not missing_pairs and all(results[f"{sp}_label_count"] for sp in SPLITS)
    results["image_decode"] = not decode_fail
    results["label_syntax"] = not invalid
    duplicate_groups = {h: files for h, files in all_sha.items() if len(files) > 1}
    results["exact_content_unique_across_all_splits"] = not duplicate_groups

    near_cross = []
    burst_cross = []
    session_cross = []
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            for na, ha in d_hashes[a].items():
                for nb, hb in d_hashes[b].items():
                    if hamming(ha, hb) <= 5:
                        near_cross.append((a, na, b, nb))
            for ta, na in all_times[a]:
                for tb, nb in all_times[b]:
                    gap = abs(ta - tb)
                    if gap <= 3.0:
                        burst_cross.append((a, na, b, nb, gap))
                    if gap <= 60.0:
                        session_cross.append((a, na, b, nb, gap))
    results["near_duplicate_cross_split"] = not near_cross
    results["burst_cross_split"] = not burst_cross
    results["session_60s_cross_split"] = not session_cross

    pending = [r for r in rows if r.get("human_required", "").lower() == "true"]
    high = [r for r in pending if r.get("severity") == "HIGH"]
    ai = Counter(r.get("resolution") for r in rows if r.get("resolution") in
                 {"KEEP_EMPTY", "USE_NONEMPTY", "USE_A", "USE_B"})
    results["human_admission_gate"] = not pending
    results["training_readiness"] = not pending

    summary = {
        "checks_total": len(results),
        "checks_passed": sum(bool(x) for x in results.values()),
        "checks_failed": sum(not bool(x) for x in results.values()),
        "checks": results,
        "counts": counts,
        "total_images": sum(counts.values()),
        "empty_labels": dict(empty),
        "invalid_label_records": len(invalid),
        "decode_failures": len(decode_fail),
        "exact_duplicate_groups": len(duplicate_groups),
        "near_duplicate_cross_split_pairs": len(near_cross),
        "burst_cross_split_pairs": len(burst_cross),
        "session_60s_cross_split_pairs": len(session_cross),
        "class_distribution": {sp: {NAMES[k]: v for k, v in sorted(distribution[sp].items())}
                                for sp in SPLITS},
        "pending_human_groups": len(pending),
        "pending_high_groups": len(high),
        "ai_decisions": dict(ai),
        "training": "NOT_RUN",
        "training_readiness": "BLOCKED_UNRESOLVED_ANNOTATIONS" if pending else "CANDIDATE_ONLY",
    }
    out = args.root / "tools" / "v3_verification.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# garbage_seg_merged_v3 全面质量验收",
        "",
        f"生成日期：{__import__('datetime').date.today().isoformat()}",
        "",
        "本轮未训练模型。v1/v2 与源 A/B 均为只读输入。",
        "",
        "## 结构与格式",
        "",
        f"- 图片：train/val/test = {counts['train']}/{counts['val']}/{counts['test']}；总计 {sum(counts.values())}。",
        f"- 图片-标签配对：{'PASS' if results['image_label_pairing'] else 'FAIL'}。",
        f"- 图片解码：{'PASS' if results['image_decode'] else 'FAIL'}；非法标签记录：{len(invalid)}。",
        f"- 全集精确内容重复：{len(duplicate_groups)} 组；跨 split dHash 近重复：{len(near_cross)} 对。",
        "",
        "## 类别覆盖",
        "",
        "| split | bottle | box | can | cigarette_butt | food_waste | other | paper | plastic_bag | wrapper |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sp in SPLITS:
        lines.append("| " + sp + " | " + " | ".join(str(distribution[sp][i]) for i in range(9)) + " |")
    lines += [
        "",
        f"- `can`：val={distribution['val'][2]}、test={distribution['test'][2]}；已具备最低可评估入口，但样本仍极少。",
        f"- `food_waste`：val={distribution['val'][4]}、test={distribution['test'][4]}；仍属于低样本类别。",
        "",
        "## 标注质量门禁",
        "",
        f"- v2 637 组待审问题在 v3 中全部保留审计记录；当前 `human_required=true`：{len(pending)} 组。",
        f"- HIGH 优先清单：{len(high)} 组（本地源 SHA 分组重算；与 v2 的 141 组存在口径差异，不能静默合并）。",
        f"- 本轮明确记录的 AI 处理：{dict(ai)}；均保留人工确认边界。",
        f"- 14 组 val/test 空/非空问题：4 组清空明显海报/标识假阳性，1 组保留明显垃圾桶场景，9 组待人工确认。",
        "",
        "## 会话划分",
        "",
        "v3 使用 exact SHA、dHash≤5、3 秒连拍、以及 60 秒时间相邻启发式构造不可拆分单位。",
        f"跨 split 60 秒时间邻近对：{len(session_cross)}；该结果仅支持有限的时间相关性检查，不是场景独立性证明。",
        "",
        "## 最终判定",
        "",
        "```text",
        f"STRUCTURE_AND_SYNTAX = {'PASS' if all(results[k] for k in ('total_images','image_label_pairing','image_decode','label_syntax') ) else 'FAIL'}",
        f"CROSS_SPLIT_CONTENT  = {'PASS' if results['exact_content_unique_across_all_splits'] else 'FAIL'}",
        f"CROSS_SPLIT_NEAR_DUP = {'PASS' if results['near_duplicate_cross_split'] else 'FAIL'}",
        f"SESSION_60S_CHECK    = {'PASS' if results['session_60s_cross_split'] else 'FAIL'}",
        "ANNOTATION_ADMISSION = BLOCKED_UNRESOLVED_HUMAN_REVIEW",
        "TRAINING_THIS_ROUND  = NOT_RUN",
        "V3_TRAINING_READY    = NO",
        "```",
    ]
    (args.root / "tools" / "v3_acceptance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["checks_failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
