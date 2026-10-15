#!/usr/bin/env python3
"""Evaluate any 9-class segment checkpoint on ONE fixed indoor val dataset.
Test split requires --unlock-test and should be used once after selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v3_dataset import NAMES, load_built, names_from_yaml, read_yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--indoor', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--split', choices=('val', 'test'), default='val')
    p.add_argument('--unlock-test', action='store_true')
    p.add_argument('--device', default='0')
    p.add_argument('--batch', type=int, default=16)
    args = p.parse_args()
    if args.split == 'test' and not args.unlock_test:
        raise ValueError('Final test is locked during development. Use --unlock-test ONLY after model selection.')
    if args.out.exists():
        raise FileExistsError(f'Refusing to overwrite previous evaluation: {args.out}')
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    data_path = args.indoor / 'data.yaml'
    data = read_yaml(data_path)
    if names_from_yaml(data) != list(NAMES):
        raise ValueError('Indoor class mapping wrong')
    if (args.indoor / 'manifest.csv').is_file() is False:
        raise ValueError('Missing indoor audit manifest')
    eval_rows = load_built(args.indoor, args.split, 'heldout')
    # Seal held-out image identity and exact labels for a fair comparison.
    fingerprint = hashlib.sha256()
    for entry in sorted(eval_rows, key=lambda x: str(x.image)):
        fingerprint.update(entry.sha.encode())
        fingerprint.update(entry.label.read_bytes())
    cls_counts = {n: sum(e.counts[n] for e in eval_rows) for n in NAMES}
    import torch
    from ultralytics import YOLO
    if not torch.cuda.is_available() and args.device not in ('cpu', 'mps'):
        raise RuntimeError('CUDA unavailable')
    model = YOLO(str(args.weights))
    if list(model.names.values()) != list(NAMES):
        raise ValueError('Checkpoint 9-class names/order differ')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    metrics = model.val(data=str(data_path), split=args.split, imgsz=800, batch=args.batch,
                        device=args.device, conf=0.001, iou=0.7, rect=True,
                        max_det=300, plots=True, project=str(args.out.parent),
                        name=args.out.name + '_plots', exist_ok=False)
    def scalar(v):
        return round(float(v), 6)
    report = {
        'weights': str(args.weights), 'data_yaml': str(data_path), 'split': args.split,
        'images': len(eval_rows), 'instances_per_class': cls_counts,
        'validation_fingerprint_sha256': fingerprint.hexdigest(),
        'protocol': {'imgsz':800, 'conf':0.001, 'iou':0.7, 'rect':True,
                     'max_det':300, 'batch':args.batch},
        'mask': {'precision':scalar(metrics.seg.mp), 'recall':scalar(metrics.seg.mr),
                 'mAP50':scalar(metrics.seg.map50), 'mAP50_95':scalar(metrics.seg.map)},
        'box': {'precision':scalar(metrics.box.mp), 'recall':scalar(metrics.box.mr),
                'mAP50':scalar(metrics.box.map50), 'mAP50_95':scalar(metrics.box.map)},
        'per_class_mask_mAP50_95': {},
        'WARNING': 'Do not infer test/generalization from val. Metrics for zero-GT classes are null.',
    }
    # metrics.seg.maps is per-class mAP50-95 across ALL model classes, in model.names order.
    mask_maps = list(metrics.seg.maps)
    for idx, n in enumerate(NAMES):
        report['per_class_mask_mAP50_95'][n] = (scalar(mask_maps[idx])
            if cls_counts[n] and idx < len(mask_maps) else None)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()