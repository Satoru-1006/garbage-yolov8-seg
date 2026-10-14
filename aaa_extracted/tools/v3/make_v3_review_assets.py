#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create image-grounded review assets for the independent v3 annotation pass.

This tool never edits source images or labels.  It reads the two local source
datasets and the v2 conflict ledger, then writes per-group overlays and
contact sheets.  A/B polygons are deliberately shown in separate colours;
the output is a review aid, not an automatic adjudicator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CLASSES = {
    0: "bottle",
    1: "box",
    2: "can",
    3: "cigarette_butt",
    4: "food_waste",
    5: "other",
    6: "paper",
    7: "plastic_bag",
    8: "wrapper",
}
REMAP_A = {0: 0, 1: 1, 2: 6, 3: 7, 4: 8}


def parse_label(path: Path, dataset: str) -> list[tuple[int, list[float]]]:
    if not path.exists():
        return []
    remap = REMAP_A if dataset == "A" else {i: i for i in range(9)}
    rows = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.split()
        if not parts:
            continue
        try:
            cls = remap[int(parts[0])]
            coords = [float(x) for x in parts[1:]]
        except (KeyError, ValueError):
            continue
        rows.append((cls, coords))
    return rows


def source_image(root: Path, dataset: str, basename: str) -> Path | None:
    folder = "waste_ds_c3_neg_yolo" if dataset == "A" else "half_dataset_yolo"
    for split in ("train", "val", "test"):
        p = root / folder / "images" / split / basename
        if p.exists():
            return p
    return None


def source_label(image: Path) -> Path:
    return image.parent.parent.parent / "labels" / image.parent.name / (image.stem + ".txt")


def polygon_points(coords: list[float], width: int, height: int) -> list[tuple[int, int]]:
    return [(round(coords[i] * width), round(coords[i + 1] * height))
            for i in range(0, len(coords) - 1, 2)]


def draw_labels(im: Image.Image, labels, colour, label_prefix: str,
                image_offset_y: int = 0, image_height: int | None = None) -> None:
    draw = ImageDraw.Draw(im, "RGBA")
    width, height = im.size
    if image_height is None:
        image_height = height - image_offset_y
    for idx, (cls, coords) in enumerate(labels, start=1):
        points = polygon_points(coords, width, image_height)
        points = [(x, y + image_offset_y) for x, y in points]
        if len(points) < 3:
            continue
        draw.polygon(points, fill=colour + (65,), outline=colour + (235,))
        x, y = min(p[0] for p in points), min(p[1] for p in points)
        draw.text((x + 2, y + 2), f"{label_prefix}{idx}:{CLASSES.get(cls, cls)}",
                  fill=colour + (255,))


def load_font(size: int):
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyh.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def make_overlay(root: Path, row: dict, out_path: Path) -> tuple[str, str]:
    sample = row["sample"]
    dataset = "A" if sample.startswith("A_") else "B"
    basename = sample[2:] if sample[:2] in ("A_", "B_") else sample
    image_path = source_image(root, dataset, basename)
    if image_path is None:
        return "MISSING", ""

    with Image.open(image_path) as raw:
        image = raw.convert("RGB")
    image.thumbnail((1280, 900), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (image.width, image.height + 88), "black")
    canvas.paste(image.convert("RGBA"), (0, 88))
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))

    # The displayed image has been resized, so coordinates remain normalized.
    a_image = source_image(root, "A", basename)
    b_image = source_image(root, "B", basename)
    if a_image is not None:
        draw_labels(overlay, parse_label(source_label(a_image), "A"), (255, 80, 80), "A",
                    image_offset_y=88, image_height=image.height)
    if b_image is not None:
        draw_labels(overlay, parse_label(source_label(b_image), "B"), (70, 220, 120), "B",
                    image_offset_y=88, image_height=image.height)
    canvas = Image.alpha_composite(canvas, overlay)

    font = load_font(22)
    small = load_font(16)
    draw = ImageDraw.Draw(canvas)
    flags = row.get("unresolved_reasons", "")
    title = f"{row['target_split']} | {row['severity']} | {sample}"
    draw.text((10, 8), title, font=font, fill="white")
    draw.text((10, 38), f"A={row['n_inst_a']}  B={row['n_inst_b']}  {flags}",
              font=small, fill=(255, 220, 120))
    draw.text((10, 62), "RED=A source polygons; GREEN=B source polygons; visual judgement required",
              font=small, fill=(200, 200, 200))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_path, quality=94)
    return "OK", str(image_path)


def make_sheets(entries: list[dict], overlay_dir: Path, sheet_dir: Path,
                per_sheet: int = 12) -> None:
    sheet_dir.mkdir(parents=True, exist_ok=True)
    font = load_font(16)
    for start in range(0, len(entries), per_sheet):
        chunk = entries[start:start + per_sheet]
        thumbs = []
        for e in chunk:
            p = overlay_dir / e["overlay"]
            if not p.exists():
                continue
            with Image.open(p) as im:
                th = im.copy()
            th.thumbnail((420, 330), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (420, 360), "#111111")
            tile.paste(th, ((420 - th.width) // 2, 4))
            ImageDraw.Draw(tile).text((6, 338), e["sample"], font=font,
                                      fill="white")
            thumbs.append(tile)
        if not thumbs:
            continue
        cols = 3
        rows = math.ceil(len(thumbs) / cols)
        sheet = Image.new("RGB", (cols * 420, rows * 360), "#222222")
        for i, tile in enumerate(thumbs):
            sheet.paste(tile, ((i % cols) * 420, (i // cols) * 360))
        sheet.save(sheet_dir / f"sheet_{start // per_sheet + 1:03d}.jpg", quality=90)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    ap.add_argument("--conflicts", type=Path,
                    default=Path(__file__).resolve().parents[3] / "dedup_v2_reports" /
                    "tools" / "dedup_v2_annotation_conflicts.csv")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[3] / "v3_review_assets")
    ap.add_argument("--scope", choices=("all", "high", "valtest_empty"), default="all")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.conflicts.open(encoding="utf-8", newline="")))
    # Accept the native v3 review ledger as well as the legacy v2 conflict CSV.
    # v3 stores the representative source path and flags instead of v2's
    # prefixed sample/n_inst_a/n_inst_b columns.
    if rows and "representative_source" in rows[0]:
        normalized = []
        for raw in rows:
            rep = Path(raw["representative_source"])
            dataset = raw["representative_dataset"]
            basename = rep.name
            a_image = source_image(args.root, "A", basename)
            b_image = source_image(args.root, "B", basename)
            normalized.append({
                "sample": f"{dataset}_{basename}",
                "target_split": raw["split"],
                "severity": raw["severity"],
                "n_inst_a": len(parse_label(source_label(a_image), "A")) if a_image else 0,
                "n_inst_b": len(parse_label(source_label(b_image), "B")) if b_image else 0,
                "unresolved_reasons": raw["flags"],
            })
        rows = normalized
    if args.scope == "high":
        rows = [r for r in rows if r["severity"] == "HIGH"]
    elif args.scope == "valtest_empty":
        rows = [r for r in rows if r["target_split"] in ("val", "test") and
                "EMPTY_VS_ANNOTATED" in r["unresolved_reasons"]]
    rows.sort(key=lambda r: (0 if r["severity"] == "HIGH" else 1,
                             {"val": 0, "test": 1, "train": 2}.get(r["target_split"], 9),
                             r["sample"]))

    overlay_dir = args.out / args.scope / "overlays"
    sheet_dir = args.out / args.scope / "sheets"
    index_rows = []
    for i, row in enumerate(rows, start=1):
        overlay = f"{i:04d}_{row['sample'].replace('.jpg', '')}.jpg"
        status, source = make_overlay(args.root, row, overlay_dir / overlay)
        item = dict(row)
        item.update({"review_id": f"V3-{args.scope.upper()}-{i:04d}",
                     "review_status": "PENDING_HUMAN" if status == "OK" else "MISSING_SOURCE",
                     "overlay": overlay, "source_image": source})
        index_rows.append(item)
    make_sheets(index_rows, overlay_dir, sheet_dir)

    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.out / args.scope / "review_index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if index_rows:
        with index_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
            w.writeheader()
            w.writerows(index_rows)
    summary = {
        "scope": args.scope,
        "rows": len(index_rows),
        "overlays_ok": sum(1 for r in index_rows if r["review_status"] == "PENDING_HUMAN"),
        "missing_source": sum(1 for r in index_rows if r["review_status"] == "MISSING_SOURCE"),
        "action": "visual_review_only_no_automatic_adjudication",
    }
    (args.out / args.scope / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not summary["missing_source"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
