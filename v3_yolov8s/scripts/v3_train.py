#!/usr/bin/env python3
"""Reproducible V3 YOLOv8s-seg training: smoke, p1, p2a, p2b.
Do not use --resume or overwrite prior B0/TACO model artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import yaml
from v3_dataset import NAMES, guard, names_from_yaml


def filehash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def check_dataset(root: Path, heldout_required=False) -> Path:
    p = root / 'data.yaml'
    if not p.is_file():
        raise FileNotFoundError(p)
    data = yaml.safe_load(p.read_text(encoding='utf-8'))
    if names_from_yaml(data) != list(NAMES):
        raise ValueError(f'{root}: 9 classes mismatch')
    if Path(data['path']).resolve() != root.resolve():
        raise ValueError(f'YAML path mismatch: {data["path"]} != {root}')
    for split in ('train', 'val') + (('test',) if heldout_required else ()):
        where = root / data[split]
        if not where.is_dir() or not any(where.iterdir()):
            raise ValueError(f'{split} absent/empty: {where}')
    if not (root / 'manifest.csv').is_file() or not (root / 'audit.json').is_file():
        raise ValueError(f'{root}: pipeline manifest/audit missing. Use v3_dataset.py builder.')
    return p


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True, choices=['smoke', 'p1', 'p2a', 'p2b'])
    p.add_argument('--public', type=Path, required=True)
    p.add_argument('--indoor', type=Path, help='Required for p2a/p2b; recommended guard for p1')
    p.add_argument('--mixed', type=Path, help='Required for p2b')
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--p1-weights', type=Path, help='Actual p1 best.pt for p2a/p2b')
    p.add_argument('--init', default='yolov8s-seg.pt', help='COCO pretrained s-seg .pt for p1')
    p.add_argument('--device', default='0')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.batch <= 0:
        raise ValueError('batch must be positive')
    if not args.public.is_dir():
        raise FileNotFoundError(args.public)
    public_yaml = check_dataset(args.public)
    if args.indoor is not None:
        check_dataset(args.indoor, heldout_required=True)
        guard(args.public, args.indoor, out=args.project / 'preflight_near_duplicate_review.csv')
    if args.stage in ('p2a', 'p2b') and args.indoor is None:
        raise ValueError('Need --indoor for p2a/p2b')
    if args.stage == 'p2b' and args.mixed is None:
        raise ValueError('Need --mixed for p2b')

    if args.stage in ('smoke', 'p1'):
        data_yaml = public_yaml
        weights = args.init
    else:
        if args.p1_weights is None or not args.p1_weights.is_file():
            raise ValueError('--p1-weights must point to existing p1 best.pt')
        weights = str(args.p1_weights)
        data_yaml = check_dataset(args.indoor if args.stage == 'p2a' else args.mixed, heldout_required=True)

    configs = {
        'smoke': dict(epochs=3, patience=3, lr0=.001, lrf=.01, mosaic=.3, close_mosaic=0),
        'p1': dict(epochs=100, patience=20, lr0=.001, lrf=.01, mosaic=.3, close_mosaic=15),
        'p2a': dict(epochs=40, patience=12, lr0=.0001, lrf=.05, mosaic=.1, close_mosaic=10),
        'p2b': dict(epochs=40, patience=12, lr0=.0001, lrf=.05, mosaic=.1, close_mosaic=10),
    }
    stage = args.stage
    jobname = {'smoke':'p1_smoke', 'p1':'p1_public', 'p2a':'p2a_indoor', 'p2b':'p2b_mixed'}[stage]
    run_dir = args.project / jobname
    if run_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing training run: {run_dir}')
    import torch
    import ultralytics
    from ultralytics import YOLO
    if not torch.cuda.is_available() and args.device not in ('cpu', 'mps'):
        raise RuntimeError('CUDA unavailable, aborting instead of silently switching device')
    model = YOLO(str(weights))
    if stage in ('p2a', 'p2b') and list(model.names.values()) != list(NAMES):
        raise ValueError('Phase1 checkpoint class ontology is not canonical nine')

    params = dict(
        data=str(data_yaml), project=str(args.project), name=jobname, exist_ok=False,
        imgsz=800, batch=args.batch, workers=args.workers, device=args.device,
        optimizer='AdamW', freeze=0, cos_lr=True, weight_decay=0.0005,
        seed=args.seed, deterministic=True, amp=True, plots=True, save=True,
        hsv_h=.015, hsv_s=.35, hsv_v=.3, degrees=8.0, translate=.1, scale=.25,
        fliplr=.5, flipud=0.0, mixup=0.0, copy_paste=0.0,
        overlap_mask=True, mask_ratio=2, max_det=300,
        **configs[stage]
    )
    print('RUN CONFIGURATION:\n' + json.dumps(params, ensure_ascii=False, indent=2))
    model.train(**params)
    best = run_dir / 'weights' / 'best.pt'
    if not best.is_file():
        raise RuntimeError(f'Training returned but best checkpoint missing: {best}')
    meta = {
        'stage': stage, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'os': platform.platform(), 'python': platform.python_version(),
        'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
        'cuda': str(torch.version.cuda), 'device': args.device,
        'weights_source': str(weights), 'weights_sha256': filehash(Path(weights)) if Path(weights).is_file() else 'official-download-by-ultralytics',
        'best_sha256': filehash(best), 'dataset_yaml': str(data_yaml),
        'dataset_manifest_sha256': filehash(data_yaml.parent / 'manifest.csv'),
        'params': params,
    }
    (run_dir / 'v3_run_manifest.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'\nDONE: {best}\nManifest: {run_dir / "v3_run_manifest.json"}')


if __name__ == '__main__':
    main()