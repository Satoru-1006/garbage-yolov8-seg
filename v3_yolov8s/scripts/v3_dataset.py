#!/usr/bin/env python3
"""V3 YOLOv8s-seg safe dataset building. Originals are never overwritten.

Inputs: train/val YOLO instance-seg exports with a data.yaml per source.
Output: images/{train,val,test}, labels/{train,val,test}, data.yaml and manifest.

Safety: unknown class names / malformed or missing polygons / duplicate split
leakage / unclear license / unapproved indoor labels cause hard errors.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageOps

NAMES = ('bottle', 'box', 'can', 'cigarette_butt', 'food_waste',
         'other', 'paper', 'plastic_bag', 'wrapper')
NAME2ID = {name: n for n, name in enumerate(NAMES)}
EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def read_yaml(path: Path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def dhash(path: Path) -> int:
    with Image.open(path) as img:
        gray = ImageOps.grayscale(img).resize((9, 8), Image.Resampling.LANCZOS)
        pix = list(gray.get_flattened_data() if hasattr(gray, 'get_flattened_data') else gray.getdata())
    value = 0
    for row in range(8):
        for col in range(8):
            value = (value << 1) | (pix[row * 9 + col] > pix[row * 9 + col + 1])
    return value


def names_from_yaml(data: dict) -> list[str]:
    val = data.get('names')
    if isinstance(val, list):
        names = [str(x) for x in val]
    elif isinstance(val, dict):
        keys = sorted(int(k) for k in val)
        if keys != list(range(len(keys))):
            raise ValueError(f'Class IDs not consecutive: {keys}')
        names = [str(val.get(k, val.get(str(k)))) for k in keys]
    else:
        raise ValueError('data.yaml must include list/dict names')
    if len(set(names)) != len(names):
        raise ValueError(f'Duplicate source class names: {names}')
    return names


def split_dirs(root: Path, data: dict, split: str) -> tuple[Path, Path]:
    spec = data.get(split)
    if not isinstance(spec, str) or spec.endswith('.txt'):
        raise ValueError(f'{root}: {split} must be an image DIRECTORY in data.yaml')
    image_dir = (root / spec).resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f'{split} images missing: {image_dir}')
    segments = list(image_dir.parts)
    image_positions = [i for i, x in enumerate(segments) if x == 'images']
    if not image_positions:
        raise ValueError(f'Cannot derive labels dir (requires images segment): {image_dir}')
    segments[image_positions[-1]] = 'labels'
    label_dir = Path(*segments)
    if not label_dir.is_dir():
        raise FileNotFoundError(f'{split} labels directory missing: {label_dir}')
    return image_dir, label_dir


def parse_segmentation(path: Path, src_names: list[str], mapping: dict) -> tuple[str, Counter]:
    lines = path.read_text(encoding='utf-8').splitlines()
    rendered = []
    counts = Counter()
    for lineno, line in enumerate(lines, 1):
        fields = line.split()
        if not fields:
            continue
        try:
            index = int(fields[0])
        except ValueError as e:
            raise ValueError(f'{path}:{lineno} invalid integer class ID') from e
        if not 0 <= index < len(src_names):
            raise ValueError(f'{path}:{lineno}: class ID {index} out of range')
        source_class = src_names[index]
        target = mapping.get(source_class, source_class)
        if target not in NAME2ID:
            raise ValueError(f'{path}:{lineno}: unmapped class {source_class!r}; explicit mapping required')
        # 1 class column + at least 3 xy points; a detection box has only 5 columns.
        if len(fields) < 7 or len(fields) % 2 == 0:
            raise ValueError(f'{path}:{lineno}: NOT a valid YOLO-Seg polygon ({len(fields)} fields)')
        try:
            points = [float(v) for v in fields[1:]]
        except ValueError as e:
            raise ValueError(f'{path}:{lineno}: non-numeric polygon') from e
        if any(not 0 <= v <= 1 for v in points):
            raise ValueError(f'{path}:{lineno}: polygon outside 0..1')
        # Reject degenerate polygons with no measurable extent.
        if max(points[0::2]) - min(points[0::2]) < 1e-5 or max(points[1::2]) - min(points[1::2]) < 1e-5:
            raise ValueError(f'{path}:{lineno}: degenerate polygon')
        counts[target] += 1
        rendered.append(f'{NAME2ID[target]} ' + ' '.join(fields[1:]))
    return ('\n'.join(rendered) + ('\n' if rendered else '')), counts


@dataclass
class Entry:
    source: str
    split: str
    image: Path
    label: Path
    label_text: str
    counts: Counter
    sha: str

    @property
    def other_only(self):
        return bool(self.counts) and set(self.counts) == {'other'}


def load_entries(source_id: str, root: Path, split: str, mapping: dict) -> list[Entry]:
    data = read_yaml(root / 'data.yaml')
    src_names = names_from_yaml(data)
    if mapping is None or not isinstance(mapping, dict):
        raise ValueError(f'{source_id}: mapping must be a dictionary')
    if any(v not in NAME2ID for v in mapping.values()):
        raise ValueError(f'{source_id}: bad target class in mapping')
    image_dir, label_dir = split_dirs(root, data, split)
    images = sorted(p for p in image_dir.rglob('*') if p.is_file() and p.suffix.lower() in EXT)
    if not images:
        raise ValueError(f'{source_id}/{split}: no images in {image_dir}')
    rows = []
    for img in images:
        rel = img.relative_to(image_dir)
        lab = (label_dir / rel).with_suffix('.txt')
        if not lab.is_file():
            raise FileNotFoundError(f'{source_id}: missing label {lab} (use empty .txt only if checked negative)')
        label_text, counts = parse_segmentation(lab, src_names, mapping)
        with Image.open(img) as im:
            im.verify()  # Check decoding metadata; catches many corrupt files.
        rows.append(Entry(source_id, split, img, lab, label_text, counts, sha256(img)))
    return rows


def assert_no_exact_leaks(splits: dict[str, list[Entry]]) -> None:
    seen = {}
    for split, records in splits.items():
        for rec in records:
            old = seen.get(rec.sha)
            if old and old[0] != split:
                raise ValueError(f'EXACT CROSS-SPLIT LEAK: {old} <> {(split, str(rec.image))}')
            seen.setdefault(rec.sha, (split, str(rec.image)))


def near_duplicate_review(splits: dict[str, list[Entry]], out: Path, cutoff=3, cap=250) -> int:
    """Cross-split dHash candidates are NOT proof of duplicates; human review."""
    train = splits.get('train', [])
    refs = [(split, e) for split in ('val', 'test') for e in splits.get(split, [])]
    pairs = []
    if not train or not refs:
        return 0
    t_hashes = [(e, dhash(e.image)) for e in train]
    val_hashes = [(e, dhash(e.image)) for e in splits.get('val', [])]
    for split, entry in refs:
        h = dhash(entry.image)
        reference = t_hashes if split == 'val' else (t_hashes + val_hashes)
        for other, h2 in reference:
            dist = (h ^ h2).bit_count()
            if dist <= cutoff:
                pairs.append((split, str(entry.image), str(other.image), dist))
                if len(pairs) >= cap:
                    break
        if len(pairs) >= cap:
            break
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['heldout_split', 'heldout_image', 'train_image', 'dhash_hamming_distance'])
        wr.writerows(pairs)
    return len(pairs)


def select_train(rows: list[Entry], quota: int, keep_other: float, seed: int) -> list[Entry]:
    if not 0 <= keep_other <= 1:
        raise ValueError('other_only_keep_rate must be between 0 and 1')
    rng = random.Random(seed)
    remaining = [e for e in rows if not (e.other_only and rng.random() > keep_other)]
    # Greedy class-balanced whole-image sampling. No instance labels are deleted.
    chosen = []
    coverage = Counter()
    rng.shuffle(remaining)
    while remaining and len(chosen) < quota:
        # low-count classes receive priority, but do not chase instance-count parity
        def score(e):
            rare = sum(1.0 / (1 + coverage[c]) for c in e.counts if c != 'other')
            return rare + (0.025 if e.counts else 0.0)
        best = max(range(len(remaining)), key=lambda i: score(remaining[i]))
        e = remaining.pop(best)
        chosen.append(e)
        for c in e.counts:
            coverage[c] += 1
    return chosen


def dedup_within(records: list[Entry]) -> tuple[list[Entry], int]:
    kept = []
    seen = {}
    for r in records:
        if r.sha in seen:
            if seen[r.sha].label_text != r.label_text:
                raise ValueError(f'Image has CONFLICTING labels: {seen[r.sha].image} <> {r.image}')
            continue
        seen[r.sha] = r
        kept.append(r)
    return kept, len(records) - len(kept)


def manifest_row(e: Entry, newstem: str) -> dict:
    return {'newstem': newstem, 'source': e.source, 'split': e.split,
            'src_image': str(e.image), 'src_label': str(e.label), 'sha256': e.sha,
            'instances': sum(e.counts.values()),
            **{f'n_{c}': e.counts[c] for c in NAMES}}


def write_dataset(out: Path, groups: dict[str, list[Entry]], *, note: str):
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite: {out}')
    all_entries = [e for records in groups.values() for e in records]
    srcs = [e.image.resolve() for e in all_entries]
    if any(out.resolve() in p.parents for p in srcs):
        raise ValueError('Output must not contain source files')
    if any(p == out.resolve() or p in out.resolve().parents for p in srcs):
        raise ValueError('Output must not be within or overwrite source input tree')
    assert_no_exact_leaks(groups)
    out.mkdir(parents=True)  # never overwrite an existing output
    rows = []
    report = {'note': note, 'splits': {}, 'names': list(NAMES),
              'IMPORTANT': 'Near-duplicate candidates and polygon semantics require HUMAN review.'}
    for split, entries in groups.items():
        (out / 'images' / split).mkdir(parents=True)
        (out / 'labels' / split).mkdir(parents=True)
        stats = Counter()
        for idx, e in enumerate(entries):
            safe_source = ''.join(ch if ch.isalnum() or ch in '_-' else '_' for ch in e.source)
            stem = f'{safe_source}_{idx:06d}_{e.sha[:12]}'
            target_img = out / 'images' / split / (stem + e.image.suffix.lower())
            shutil.copy2(e.image, target_img)
            (out / 'labels' / split / (stem + '.txt')).write_text(e.label_text, encoding='utf-8')
            stats.update(e.counts)
            rows.append(manifest_row(e, stem))
        report['splits'][split] = {'images': len(entries), 'instances': sum(stats.values()),
                                   'by_class': {c: stats[c] for c in NAMES}}
    config = {'path': str(out.resolve()), 'train': 'images/train',
              'val': 'images/val', 'names': dict(enumerate(NAMES))}
    if 'test' in groups:
        config['test'] = 'images/test'
    (out / 'data.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    with open(out / 'manifest.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    report['near_duplicate_candidate_pairs'] = near_duplicate_review(groups, out / 'near_duplicate_review.csv')
    (out / 'audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report


def inspect(cfg_path: Path):
    cfg = read_yaml(cfg_path)
    result = {}
    for source in cfg.get('sources', []):
        root = Path(source.get('root', '')).expanduser().resolve()
        if not root.is_dir() or not (root / 'data.yaml').is_file():
            result[source['id']] = {'status': 'MISSING', 'root': str(root)}
            continue
        data = read_yaml(root / 'data.yaml')
        info = {'status': 'FOUND', 'enabled': source.get('enabled', False),
                'license_ok': source.get('license_ok', False),
                'root': str(root), 'classes': names_from_yaml(data)}
        for split in ('train', 'val', 'test'):
            if not data.get(split):
                info[split] = 'NOT DECLARED'
                continue
            try:
                img_dir, lab_dir = split_dirs(root, data, split)
                info[split] = {'images': sum(p.suffix.lower() in EXT for p in img_dir.rglob('*') if p.is_file()),
                               'labels': sum(p.suffix == '.txt' for p in lab_dir.rglob('*') if p.is_file())}
            except Exception as exc:
                info[split] = f'ERROR: {exc}'
        result[source['id']] = info
    print(json.dumps(result, indent=2, ensure_ascii=False))


def public_build(cfg_path: Path, out: Path):
    cfg = read_yaml(cfg_path)
    seed = int(cfg.get('seed', 42))
    all_train, all_val = [], []
    enabled = [s for s in cfg.get('sources', []) if s.get('enabled')]
    if not enabled:
        raise ValueError('No public sources enabled. Edit sources.local.yaml first.')
    if len(set(s['id'] for s in enabled)) != len(enabled):
        raise ValueError('Duplicate source IDs')
    for j, source in enumerate(enabled):
        name = source['id']
        if not source.get('license_ok'):
            raise ValueError(f'{name}: confirm dataset license and set license_ok=true')
        root = Path(source['root']).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        tr = load_entries(name, root, 'train', source.get('mapping', {}))
        va = load_entries(name, root, 'val', source.get('mapping', {}))
        selected = select_train(tr, int(source['quota_train']),
                                float(source.get('other_only_keep_rate', 1)), seed + j)
        print(f'[source={name}] train available {len(tr)}, selected {len(selected)}, val {len(va)}')
        all_train.extend(selected)
        all_val.extend(va)
    all_train, dtrain = dedup_within(all_train)
    all_val, dval = dedup_within(all_val)
    print(f'Dropped identical within-split images: train={dtrain}, val={dval}')
    report = write_dataset(out, {'train': all_train, 'val': all_val}, note='Phase1 public; no indoor images')
    print(json.dumps(report, ensure_ascii=False, indent=2))


def indoor_build(cfg_path: Path, out: Path):
    cfg = read_yaml(cfg_path)['indoor']
    if cfg.get('review_approved') is not True:
        raise ValueError('Indoor SAM3 results NOT approved: review_approved must be true')
    root = Path(cfg['root']).expanduser().resolve()
    data = read_yaml(root / 'data.yaml')
    if names_from_yaml(data) != list(NAMES):
        raise ValueError('Indoor classes/order differ from canonical nine; remap before building')
    groups = {split: load_entries('indoor_reviewed', root, split, {})
              for split in ('train', 'val', 'test')}
    for split in ('val', 'test'):
        if not groups[split]:
            raise ValueError(f'No held-out {split} images')
    report = write_dataset(out, groups, note='SAM3-assisted plus human reviewed indoor; session-disjoint split required')
    print(json.dumps(report, ensure_ascii=False, indent=2))


def load_built(root: Path, split: str, source: str) -> list[Entry]:
    data = read_yaml(root / 'data.yaml')
    if names_from_yaml(data) != list(NAMES):
        raise ValueError(f'{root}: expected canonical nine classes')
    return load_entries(source, root, split, {})


def guard(public: Path, indoor: Path, *, out: Path | None = None):
    pub = {k: load_built(public, k, 'public') for k in ('train', 'val')}
    ind = {k: load_built(indoor, k, 'indoor') for k in ('train', 'val', 'test')}
    assert_no_exact_leaks(pub)
    assert_no_exact_leaks(ind)
    overlap = set(e.sha for x in pub.values() for e in x) & set(e.sha for x in ind.values() for e in x)
    if overlap:
        raise ValueError(f'Cross-public/indoor exact duplicate leakage: {len(overlap)} images')
    groups = {'train': pub['train'] + ind['train'], 'val': pub['val'] + ind['val'], 'test': ind['test']}
    review_path = out or (public / 'cross_domain_near_duplicate_review.csv')
    near = near_duplicate_review(groups, review_path)
    summary = {'public': {k: len(v) for k, v in pub.items()},
               'indoor': {k: len(v) for k, v in ind.items()},
               'exact_overlap': 0, 'near_duplicate_candidates': near,
               'near_duplicate_review_csv': str(review_path)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('IMPORTANT: manually review flagged near duplicates + indoor scene/session splits.')


def mixed_build(public: Path, indoor: Path, out: Path, public_fraction: float, seed: int):
    if not (0 < public_fraction < 0.5):
        raise ValueError('Expect 0 < public_fraction < 0.5 (indoor-majority)')
    public_rows = load_built(public, 'train', 'public')
    train_rows = load_built(indoor, 'train', 'indoor')
    val_rows = load_built(indoor, 'val', 'indoor')
    test_rows = load_built(indoor, 'test', 'indoor')
    # Do not allow public images that were part of either held-out indoor split.
    heldout = {r.sha for r in val_rows + test_rows}
    candidates = [r for r in public_rows if r.sha not in heldout]
    target = round(len(train_rows) * public_fraction / (1 - public_fraction))
    selected = select_train(candidates, min(target, len(candidates)), 1.0, seed)
    mixed, dup = dedup_within(train_rows + selected)
    if dup:
        print(f'[warning] removed {dup} identical train images')
    report = write_dataset(out, {'train': mixed, 'val': val_rows, 'test': test_rows},
                           note=f'Phase2 mixed indoor-majority public target fraction={public_fraction:.3f}')
    print(json.dumps(report, ensure_ascii=False, indent=2))


def preview(root: Path, split: str, out: Path, count: int, seed: int):
    entries = load_built(root, split, split)
    rng = random.Random(seed)
    for index, e in enumerate(rng.sample(entries, min(count, len(entries)))):
        with Image.open(e.image) as img:
            rgb = img.convert('RGB')
        draw = ImageDraw.Draw(rgb)
        w, h = rgb.size
        for line in e.label_text.splitlines():
            items = line.split()
            pts = [float(v) for v in items[1:]]
            xy = [(int(pts[i] * w), int(pts[i+1] * h)) for i in range(0, len(pts), 2)]
            draw.line(xy + [xy[0]], fill=(255, 0, 0), width=max(2, w // 500))
            draw.text(xy[0], NAMES[int(items[0])], fill=(255, 255, 0))
        out.mkdir(parents=True, exist_ok=True)
        rgb.save(out / f'{index:03d}_{e.image.stem}.jpg', quality=90)
    print(f'Saved {min(count, len(entries))} labeled previews in {out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('inspect')
    s.add_argument('--config', type=Path, required=True)
    s = sub.add_parser('build-public')
    s.add_argument('--config', type=Path, required=True)
    s.add_argument('--out', type=Path, required=True)
    s = sub.add_parser('build-indoor')
    s.add_argument('--config', type=Path, required=True)
    s.add_argument('--out', type=Path, required=True)
    s = sub.add_parser('guard')
    s.add_argument('--public', type=Path, required=True)
    s.add_argument('--indoor', type=Path, required=True)
    s.add_argument('--out', type=Path)
    s = sub.add_parser('build-mixed')
    s.add_argument('--public', type=Path, required=True)
    s.add_argument('--indoor', type=Path, required=True)
    s.add_argument('--out', type=Path, required=True)
    s.add_argument('--public-fraction', type=float, default=0.25)
    s.add_argument('--seed', type=int, default=42)
    s = sub.add_parser('preview')
    s.add_argument('--dataset', type=Path, required=True)
    s.add_argument('--split', choices=('train', 'val', 'test'), default='train')
    s.add_argument('--out', type=Path, required=True)
    s.add_argument('--count', type=int, default=40)
    s.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.cmd == 'inspect':
        inspect(args.config)
    elif args.cmd == 'build-public':
        public_build(args.config, args.out)
    elif args.cmd == 'build-indoor':
        indoor_build(args.config, args.out)
    elif args.cmd == 'guard':
        guard(args.public, args.indoor, out=args.out)
    elif args.cmd == 'build-mixed':
        mixed_build(args.public, args.indoor, args.out, args.public_fraction, args.seed)
    elif args.cmd == 'preview':
        preview(args.dataset, args.split, args.out, args.count, args.seed)


if __name__ == '__main__':
    main()