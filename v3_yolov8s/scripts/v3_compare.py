#!/usr/bin/env python3
"""Compare saved evaluations; all must have identical indoor validation protocol."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('reports', nargs='+', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if len(args.reports) < 2:
        raise ValueError('Supply 2+ metric JSON files')
    data = [json.loads(x.read_text(encoding='utf-8')) for x in args.reports]
    first = data[0]
    for cur in data[1:]:
        if (cur['data_yaml'], cur['split'], cur['protocol']['imgsz'], cur['protocol']['conf'],
            cur['protocol']['iou'], cur['protocol']['rect'], cur['protocol']['max_det']) != (
            first['data_yaml'], first['split'], first['protocol']['imgsz'], first['protocol']['conf'],
            first['protocol']['iou'], first['protocol']['rect'], first['protocol']['max_det']):
            raise ValueError('Evaluation configs differ; comparison not fair')
        if cur.get('validation_fingerprint_sha256') != first.get('validation_fingerprint_sha256'):
            raise ValueError('Held-out images/labels changed between evaluations')
        if cur['instances_per_class'] != first['instances_per_class']:
            raise ValueError('GT distribution differs, comparison blocked')
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        fields = ['file', 'weights', 'mask_mAP50', 'mask_mAP50_95', 'mask_precision',
                  'mask_recall', 'box_mAP50', 'box_mAP50_95']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for path, r in zip(args.reports, data):
            writer.writerow({'file':str(path), 'weights':r['weights'],
                             'mask_mAP50':r['mask']['mAP50'],
                             'mask_mAP50_95':r['mask']['mAP50_95'],
                             'mask_precision':r['mask']['precision'],
                             'mask_recall':r['mask']['recall'],
                             'box_mAP50':r['box']['mAP50'],
                             'box_mAP50_95':r['box']['mAP50_95']})
    print(f'Written {args.out}; also inspect classwise results and real robot latency.')


if __name__ == '__main__':
    main()