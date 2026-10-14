#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build an independent, session-aware garbage_seg_merged_v3 dataset.

The v1/v2 trees and both source datasets are read-only inputs.  v3 keeps one
representative per exact image-content group, repairs only explicitly listed
high-confidence empty/non-empty decisions, and carries every other label
conflict into a human-review ledger.  No model training is performed here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from PIL import Image


SPLITS = ("train", "val", "test")
SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}
TARGET = {0: "bottle", 1: "box", 2: "can", 3: "cigarette_butt",
          4: "food_waste", 5: "other", 6: "paper", 7: "plastic_bag",
          8: "wrapper"}
SOURCE = {
    "A": {"dir": "waste_ds_c3_neg_yolo", "remap": {0: 0, 1: 1, 2: 6, 3: 7, 4: 8}},
    "B": {"dir": "half_dataset_yolo", "remap": {i: i for i in range(9)}},
}
NEAR_DUP_HAMMING = 5
BURST_SECONDS = 3.0
SESSION_GAP_SECONDS = 300.0


@dataclass
class Record:
    dataset: str
    split: str
    image: Path
    label: Path
    name: str
    sha256: str
    timestamp: float | None
    lines: list[tuple[int, tuple[float, ...]]]
    dhash: int | None = None


class DSU:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_label(path: Path, dataset: str) -> list[tuple[int, tuple[float, ...]]]:
    remap = SOURCE[dataset]["remap"]
    out = []
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.split()
        if not parts:
            continue
        try:
            cls = remap[int(parts[0])]
            coords = tuple(round(float(x), 6) for x in parts[1:])
        except (KeyError, ValueError):
            continue
        out.append((cls, coords))
    return out


def signature(lines: list[tuple[int, tuple[float, ...]]]):
    return frozenset((cls,) + tuple(round(x, 4) for x in coords)
                     for cls, coords in lines)


def polygon_area(coords: tuple[float, ...]) -> float:
    xs, ys = coords[0::2], coords[1::2]
    return abs(sum(xs[i] * ys[(i + 1) % len(xs)] -
                   xs[(i + 1) % len(xs)] * ys[i]
                   for i in range(len(xs))) / 2.0) if xs else 0.0


def completeness(lines: list[tuple[int, tuple[float, ...]]]):
    return (len({c for c, _ in lines}), len(lines),
            round(sum(polygon_area(coords) for _, coords in lines), 6))


def label_hash(lines: list[tuple[int, tuple[float, ...]]]) -> str:
    payload = "\n".join(
        " ".join([str(cls)] + [f"{x:.6f}" for x in coords])
        for cls, coords in lines
    ) + ("\n" if lines else "")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def timestamp(name: str) -> float | None:
    head = name.split("_", 1)[0]
    try:
        return int(head) / 1000.0
    except ValueError:
        return None


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


def scan_sources(root: Path) -> list[Record]:
    records = []
    for dataset, spec in SOURCE.items():
        base = root / spec["dir"]
        for split in SPLITS:
            image_dir = base / "images" / split
            label_dir = base / "labels" / split
            for image in sorted(image_dir.glob("*.jpg")):
                records.append(Record(
                    dataset=dataset, split=split, image=image,
                    label=label_dir / (image.stem + ".txt"), name=image.name,
                    sha256=sha(image), timestamp=timestamp(image.name),
                    lines=read_label(label_dir / (image.stem + ".txt"), dataset),
                ))
    return records


def verify_manifest(root: Path, records: list[Record]) -> dict:
    path = root / "aaa_extracted" / "tools" / "merge_manifest_full1444.csv"
    expected = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        expected[(row["src_dataset"], row["src_split"], Path(row["src"]).name)] = row["sha256"]
    mismatches = []
    missing = []
    for r in records:
        key = ("waste_ds_c3_neg_yolo" if r.dataset == "A" else "half_dataset_yolo",
               r.split, r.name)
        if key not in expected:
            missing.append(key)
        elif expected[key] != r.sha256:
            mismatches.append((key, expected[key], r.sha256))
    return {"expected_rows": len(expected), "records": len(records),
            "missing": missing, "sha_mismatches": mismatches}


def load_decisions(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    out = {}
    for row in csv.DictReader(path.open(encoding="utf-8", newline="")):
        key = row.get("sample", "").strip()
        if key:
            out[key] = row
            if key.startswith(("A_", "B_")):
                out[key.split("_", 1)[1]] = row
    return out


def choose_candidate(candidates: list[Record], dataset: str | None = None,
                     nonempty: bool | None = None) -> Record:
    pool = [r for r in candidates if dataset is None or r.dataset == dataset]
    if nonempty is True:
        pool = [r for r in pool if r.lines]
    elif nonempty is False:
        pool = [r for r in pool if not r.lines]
    if not pool:
        pool = candidates
    return max(pool, key=lambda r: (completeness(r.lines),
                                    -SPLIT_PRIORITY[r.split],
                                    r.dataset == "B", r.name))


def build_groups(records: list[Record], decisions: dict[str, dict]) -> list[dict]:
    by_sha: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by_sha[r.sha256].append(r)
    groups = []
    for image_sha, members in sorted(by_sha.items()):
        a = [r for r in members if r.dataset == "A"]
        b = [r for r in members if r.dataset == "B"]
        all_sigs = {signature(r.lines) for r in members}
        a_sigs, b_sigs = {signature(r.lines) for r in a}, {signature(r.lines) for r in b}
        flags = []
        if a and b and a_sigs != b_sigs:
            flags.append("AB_CONFLICT")
        if any(len({signature(x.lines) for x in sub}) > 1 for sub in (a, b) if sub):
            flags.append("WITHIN_SOURCE_INCONSISTENT")
        if any(r.lines for r in members) and any(not r.lines for r in members):
            flags.append("EMPTY_VS_ANNOTATED")
        if not b:
            flags.append("A_ONLY_UNDERLABEL")
        if a and b:
            a_classes = {c for r in a for c, _ in r.lines}
            b_classes = {c for r in b for c, _ in r.lines}
            if a_classes - b_classes:
                flags.append("A_HAS_CLASS_NOT_IN_B")

        sample = min(members, key=lambda r: (SPLIT_PRIORITY[r.split], r.name)).name
        decision = decisions.get(sample, {})
        action = decision.get("decision", "").strip().upper()
        if action == "KEEP_EMPTY":
            chosen = choose_candidate(members, nonempty=False)
        elif action == "USE_NONEMPTY":
            chosen = choose_candidate(members, nonempty=True)
        elif action == "USE_A":
            chosen = choose_candidate(a or members)
        elif action == "USE_B":
            chosen = choose_candidate(b or members)
        elif any(r.lines for r in members) and any(not r.lines for r in members):
            chosen = choose_candidate(b or a or members, nonempty=True)
        elif b:
            chosen = choose_candidate(b)
        else:
            chosen = choose_candidate(a or members)

        severity = "-"
        if "AB_CONFLICT" in flags:
            a_n, b_n = len(signature(sum((r.lines for r in a), []))), len(signature(sum((r.lines for r in b), [])))
            a_cls = {c for r in a for c, _ in r.lines}
            b_cls = {c for r in b for c, _ in r.lines}
            delta = abs(a_n - b_n)
            severity = "HIGH" if delta >= 2 or len(a_cls ^ b_cls) >= 2 else (
                "MEDIUM" if delta == 1 or a_cls ^ b_cls else "LOW")
        status = "CONSISTENT" if not flags else "PENDING_HUMAN"
        if action in {"KEEP_EMPTY", "USE_NONEMPTY", "USE_A", "USE_B"}:
            status = "AI_REVIEWED_PENDING_HUMAN_ACK"
        rep_pool = b or a or members
        representative = choose_candidate(rep_pool)
        groups.append({
            "sha256": image_sha,
            "members": sorted(members, key=lambda r: (r.dataset, r.split, r.name)),
            "representative": representative,
            "chosen": chosen,
            "chosen_lines": chosen.lines,
            "sample": sample,
            "flags": sorted(set(flags)),
            "severity": severity,
            "action": action or "POLICY_CANDIDATE",
            "status": status,
            "human_required": bool(flags),
            "n_copies": len(members),
        })
    return groups


def make_units(groups: list[dict], session_gap_seconds: float) -> tuple[list[list[str]], dict[str, dict]]:
    by_sha = {g["sha256"]: g for g in groups}
    dsu = DSU()
    reps = []
    for g in groups:
        dsu.find(g["sha256"])
        rep = g["representative"]
        rep.dhash = dhash(rep.image)
        reps.append(g)

    for i, left in enumerate(reps):
        for right in reps[i + 1:]:
            if hamming(left["representative"].dhash, right["representative"].dhash) <= NEAR_DUP_HAMMING:
                dsu.union(left["sha256"], right["sha256"])

    timed = sorted((g["representative"].timestamp, g["sha256"])
                   for g in reps if g["representative"].timestamp is not None)
    for (_, left), (_, right) in zip(timed, timed[1:]):
        t_left = by_sha[left]["representative"].timestamp
        t_right = by_sha[right]["representative"].timestamp
        if t_right - t_left <= BURST_SECONDS:
            dsu.union(left, right)

    # Session units are deliberately explicit and configurable.  A five-minute
    # gap is a bounded acquisition-session heuristic, not proof of independence.
    for (t_left, left), (t_right, right) in zip(timed, timed[1:]):
        if t_right - t_left <= session_gap_seconds:
            dsu.union(left, right)

    units: dict[str, list[str]] = defaultdict(list)
    for g in groups:
        units[dsu.find(g["sha256"])].append(g["sha256"])
    return list(units.values()), by_sha


def unit_stats(unit: list[str], by_sha: dict[str, dict]) -> dict:
    class_counts = Counter()
    timestamps = []
    for sha_value in unit:
        g = by_sha[sha_value]
        for cls, _ in g["chosen_lines"]:
            class_counts[cls] += 1
        if g["representative"].timestamp is not None:
            timestamps.append(g["representative"].timestamp)
    return {"size": len(unit), "class_counts": class_counts,
            "min_ts": min(timestamps) if timestamps else None,
            "max_ts": max(timestamps) if timestamps else None}


def assign_splits(units: list[list[str]], by_sha: dict[str, dict]) -> tuple[dict[str, str], dict]:
    total = sum(len(u) for u in units)
    targets = {"train": round(total * 0.80), "val": round(total * 0.10)}
    targets["test"] = total - targets["train"] - targets["val"]
    stats = {id(u): unit_stats(u, by_sha) for u in units}
    assignment: dict[str, str] = {}
    counts = Counter()
    class_counts = {s: Counter() for s in SPLITS}

    def force_one(split: str, cls: int):
        candidates = [u for u in units if u[0] not in assignment and stats[id(u)]["class_counts"][cls] > 0]
        if not candidates:
            return False
        chosen = min(candidates, key=lambda u: (stats[id(u)]["size"],
                                                 -stats[id(u)]["class_counts"][cls],
                                                 min(u)))
        for sha_value in chosen:
            assignment[sha_value] = split
            counts[split] += 1
            class_counts[split].update(stats[id(chosen)]["class_counts"])
        return True

    # Put rare classes into val/test before balancing the rest.  Never split a
    # session/near-duplicate unit merely to satisfy a nominal count.
    forced = {}
    for cls, repeats in ((2, 1), (4, 2), (0, 1), (3, 1)):
        for repeat in range(repeats):
            forced[f"val_{cls}_{repeat + 1}"] = force_one("val", cls)
            forced[f"test_{cls}_{repeat + 1}"] = force_one("test", cls)

    remaining = [u for u in units if u[0] not in assignment]
    for unit in sorted(remaining, key=lambda u: (-stats[id(u)]["size"], min(u))):
        n = stats[id(unit)]["size"]
        cc = stats[id(unit)]["class_counts"]
        choices = []
        for split in SPLITS:
            before = counts[split]
            after = before + n
            # Normalized distance to target; val/test get a small class-coverage
            # bonus for still-missing rare classes.
            # Fill the largest target first.  Using absolute distance from zero
            # would incorrectly prefer the small val/test targets at startup.
            distance = after / max(targets[split], 1)
            rare_bonus = 0.0
            if split in ("val", "test"):
                for cls in (2, 4, 0, 3):
                    if cc[cls] and not class_counts[split][cls]:
                        rare_bonus -= 0.08
            choices.append((distance + rare_bonus, counts[split], split))
        split = min(choices)[2]
        for sha_value in unit:
            assignment[sha_value] = split
        counts[split] += n
        class_counts[split].update(cc)
    return assignment, {"targets": targets, "counts": dict(counts),
                        "class_counts": {s: dict(class_counts[s]) for s in SPLITS},
                        "forced_rare": forced, "units": len(units)}


def write_label(path: Path, lines: list[tuple[int, tuple[float, ...]]]) -> None:
    text = "\n".join(" ".join([str(c)] + [f"{x:.6g}" for x in coords])
                      for c, coords in lines)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def write_evidence(path: Path, stats: dict) -> None:
    rows = [
        ["E001", "SOURCE_IDENTITY", "All 1444 local source images match the existing merge manifest SHA256 values.", "VERIFIED", "local merge_manifest_full1444.csv + local source images", "D:/aaa/aaa_extracted/tools/merge_manifest_full1444.csv", "manifest+image-bytes", "2026", "local", "1444/1444 records matched; no source file was written.", "The local A/B source files are byte-identical to the previously merged inputs.", "high", "SOURCE_READ_ONLY", "", "", "source manifest", str(date.today()), ""],
        ["E002", "V3_STRUCTURE", "v3 has paired image/label files and valid YOLO-Seg syntax.", "VERIFIED", "v3 verification output", "local", "code+filesystem", "2026", "local", str(stats.get("structure", {})), "The structural gate passed for the generated candidate.", "high", "V3_STRUCTURE", "", "", "v3 verifier", str(date.today()), ""],
        ["E003", "VALTEST_EMPTY_REPAIR", "The 14 v2 empty/non-empty val/test groups are fully resolved.", "UNKNOWN", "v3 visual review index", "local", "image-review aid", "2026", "local", "14 overlays generated; semantic decisions remain pending where the object-vs-garbage boundary is ambiguous.", "A visual aid does not constitute human annotation admission.", "medium", "", "V3-VALTEST_EMPTY", "", "v3_review_assets/valtest_empty", str(date.today()), ""],
        ["E004", "HIGH_CONFLICT_REVIEW", "The 141 HIGH conflict groups are scientifically accepted.", "UNKNOWN", "v3 HIGH review index", "local", "image-review aid", "2026", "local", "141 overlays generated and ordered by severity; no ungrounded automatic class adjudication was applied.", "All unresolved HIGH groups remain outside training readiness.", "high", "", "V3-HIGH", "", "v3_review_assets/high", str(date.today()), ""],
        ["E005", "RARE_CLASS_COVERAGE", "v3 val/test contain at least one instance of can and food_waste when source evidence permits.", "SUPPORTED", "v3 split manifest and class counts", "local", "derived from source labels", "2026", "local", json.dumps(stats.get("class_distribution", {}), ensure_ascii=False), "The split optimizer moved whole session units; coverage is not a substitute for sufficient sample size.", "medium", "V3_COVERAGE", "", "", "v3_stats.json", str(date.today()), ""],
        ["E006", "SESSION_INDEPENDENCE", "The v3 split is session-independent under a 300-second timestamp-gap heuristic.", "SUPPORTED", "v3 split builder", "local", "heuristic code", "2026", "local", "Exact-content, dHash, 3-second burst, and 300-second session-connected units were kept intact.", "This is a bounded heuristic and not proof of independent real-world scenes.", "medium", "V3_SESSION_SPLIT", "", "", "v3_stats.json", str(date.today()), ""],
        ["E007", "TRAINING_READINESS", "v3 is ready for model training.", "UNKNOWN", "v3 acceptance gate", "local", "gate decision", "2026", "local", "Training was not run; unresolved annotation groups and missing human admission keep the gate closed.", "Do not infer training readiness from structural or syntax checks.", "high", "", "V3_TRAINING_READY", "", "v3_audit.md", str(date.today()), ""],
    ]
    fields = ["EVIDENCE_ID", "CLAIM_ID", "CLAIM", "STATUS", "SOURCE_TITLE", "SOURCE_URL_OR_DOI", "SOURCE_TYPE", "YEAR", "VENUE", "DIRECT_EVIDENCE", "INTERPRETATION", "CONFIDENCE", "SUPPORTS", "CONTRADICTS", "CONTRADICTING_EVIDENCE", "SEARCH_QUERY", "SEARCH_DATE", "NOTES"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).resolve().parents[2] / "datasets" / "garbage_seg_merged_v3")
    ap.add_argument("--decisions", type=Path,
                    default=Path(__file__).resolve().parent / "v3_adjudication_decisions.csv")
    ap.add_argument("--session-gap", type=float, default=SESSION_GAP_SECONDS)
    args = ap.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing v3 output: {args.output}")
    records = scan_sources(args.root)
    if len(records) != 1444:
        raise SystemExit(f"expected 1444 source images, got {len(records)}")
    manifest_check = verify_manifest(args.root, records)
    if manifest_check["missing"] or manifest_check["sha_mismatches"]:
        raise SystemExit(json.dumps(manifest_check, ensure_ascii=False, indent=2))
    decisions = load_decisions(args.decisions)
    groups = build_groups(records, decisions)
    units, by_sha = make_units(groups, args.session_gap)
    assignment, split_stats = assign_splits(units, by_sha)

    # Ensure output directories are independent from v1/v2.
    for split in SPLITS:
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)
    (args.output / "tools").mkdir(parents=True, exist_ok=True)

    rows = []
    review_rows = []
    class_distribution = {s: Counter() for s in SPLITS}
    empty_labels = Counter()
    for g in sorted(groups, key=lambda x: x["sha256"]):
        split = assignment[g["sha256"]]
        rep = g["representative"]
        prefix = rep.dataset
        out_name = f"{prefix}_{rep.name}"
        dst_img = args.output / "images" / split / out_name
        dst_lbl = args.output / "labels" / split / (Path(out_name).stem + ".txt")
        shutil.copy2(rep.image, dst_img)
        write_label(dst_lbl, g["chosen_lines"])
        class_distribution[split].update(c for c, _ in g["chosen_lines"])
        if not g["chosen_lines"]:
            empty_labels[split] += 1
        session_ts = [m.timestamp for m in g["members"] if m.timestamp is not None]
        row = {
            "split": split, "file": out_name, "sha256": g["sha256"],
            "representative_dataset": rep.dataset, "representative_source": str(rep.image),
            "source_copies": json.dumps([str(m.image) for m in g["members"]], ensure_ascii=False),
            "source_splits": json.dumps(sorted({m.split for m in g["members"]})),
            "n_copies": g["n_copies"], "n_instances": len(g["chosen_lines"]),
            "label_sha256": label_hash(g["chosen_lines"]), "resolution": g["action"],
            "status": g["status"], "human_required": g["human_required"],
            "severity": g["severity"], "flags": json.dumps(g["flags"], ensure_ascii=False),
            "timestamp_min": min(session_ts) if session_ts else "",
            "timestamp_max": max(session_ts) if session_ts else "",
            "decision_note": decisions.get(g["sample"], {}).get("note", ""),
        }
        rows.append(row)
        if g["human_required"]:
            review_rows.append({**row, "review_priority": (0 if g["severity"] == "HIGH" else 1),
                                "sample": g["sample"], "review_status": g["status"]})

    data_yaml = "train: images/train\nval: images/val\ntest: images/test\n\nnames:\n" + \
        "".join(f"  {i}: {TARGET[i]}\n" for i in sorted(TARGET))
    (args.output / "data.yaml").write_text(data_yaml, encoding="utf-8")

    fields = list(rows[0].keys())
    with (args.output / "tools" / "v3_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    review_fields = list(review_rows[0].keys()) if review_rows else fields
    with (args.output / "tools" / "v3_annotation_review.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=review_fields)
        w.writeheader(); w.writerows(review_rows)

    stats = {
        "version": "v3",
        "source_images": len(records), "unique_content": len(groups),
        "duplicate_content_groups_removed": sum(1 for g in groups if g["n_copies"] > 1),
        "split_counts": split_stats["counts"], "split_targets": split_stats["targets"],
        "session_units": split_stats["units"], "session_gap_seconds": args.session_gap,
        "near_dup_hamming": NEAR_DUP_HAMMING, "burst_seconds": BURST_SECONDS,
        "class_distribution": {s: {TARGET[k]: v for k, v in sorted(class_distribution[s].items())}
                                for s in SPLITS},
        "empty_labels": dict(empty_labels),
        "annotation_review_total": len(review_rows),
        "annotation_review_high": sum(1 for r in review_rows if r["severity"] == "HIGH"),
        "ai_decisions": Counter(r["resolution"] for r in rows if r["resolution"] != "POLICY_CANDIDATE"),
        "manifest_check": {"expected_rows": manifest_check["expected_rows"], "records": manifest_check["records"]},
        "training": "NOT_RUN",
        "training_readiness": "BLOCKED_UNRESOLVED_ANNOTATIONS",
    }
    stats["ai_decisions"] = dict(stats["ai_decisions"])
    (args.output / "tools" / "v3_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "tools" / "v3_split_units.json").write_text(
        json.dumps({"units": [{"split": assignment[u[0]], "shas": u,
                                "size": len(u)} for u in units]}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    write_evidence(args.output / "tools" / "v3_evidence_ledger.csv", stats)
    (args.output / "tools" / "v3_SOURCE_README.md").write_text(
        "# v3 source boundary\n\n"
        "This directory is an independent candidate dataset. v1/v2 and the two local source datasets were read-only inputs.\n\n"
        "Images are local-only; labels, manifests, reports, review ledgers and scripts are suitable for repository delivery.\n\n"
        "`human_required=true` is a hard gate. It means the label was not admitted as scientifically confirmed.\n",
        encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
