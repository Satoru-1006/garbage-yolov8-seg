import importlib.util
from pathlib import Path
import sys

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from v3_dataset import (NAMES, parse_segmentation, load_entries, assert_no_exact_leaks,
                        public_build, indoor_build, guard, mixed_build, read_yaml, load_built)

CANONICAL = 'names:\n' + ''.join(f'  {i}: {n}\n' for i, n in enumerate(NAMES))


def make_src(root, *, label='0 .1 .1 .5 .1 .5 .5', class_name='can', leakage=False):
    for split in ['train', 'val']:
        (root / split / 'images').mkdir(parents=True, exist_ok=True)
        (root / split / 'labels').mkdir(parents=True, exist_ok=True)
        pic = root / split / 'images' / 'a.png'
        img = Image.new('RGB', (32, 32), (0, 0, 0) if (leakage or split == 'train') else (255, 0, 0))
        img.putpixel((7, 9), (128, 12, 24))
        img.save(pic)
        (root / split / 'labels' / 'a.txt').write_text(label, encoding='utf-8')
    (root / 'data.yaml').write_text('train: train/images\nval: val/images\nnames: ['+class_name+']\n')
    return root


def test_seg_invalid_bbox(tmp_path):
    label = tmp_path / 'a.txt'
    label.write_text('0 .1 .2 .3 .4')
    with pytest.raises(ValueError, match='NOT a valid'):
        parse_segmentation(label, ['can'], {})


def test_mapping_requires_explicit(tmp_path):
    make_src(tmp_path / 'src', class_name='mystery')
    with pytest.raises(ValueError, match='unmapped class'):
        load_entries('a', tmp_path / 'src', 'train', {})


def test_public_build_and_remap(tmp_path):
    src = make_src(tmp_path / 'src')
    cfg = tmp_path / 'sources.yaml'
    cfg.write_text(f'sources:\n  - id: my_can\n    enabled: true\n    license_ok: true\n    root: {src}\n    quota_train: 2\n    mapping: {{}}\n')
    out = tmp_path / 'public'
    public_build(cfg, out)
    assert len(list((out / 'images/train').glob('*'))) == 1
    assert (out / 'data.yaml').exists()
    assert list((out / 'labels/train').glob('*.txt'))[0].read_text().startswith('2 ')
    assert read_yaml(out / 'data.yaml')['names'][2] == 'can'
    with pytest.raises(FileExistsError):
        public_build(cfg, out)


def test_exact_cross_split_leak(tmp_path):
    src = make_src(tmp_path / 'src', leakage=True)
    cfg = tmp_path / 'sources.yaml'
    cfg.write_text(f'sources:\n  - id: can\n    enabled: true\n    license_ok: true\n    root: {src}\n    quota_train: 2\n    mapping: {{}}\n')
    with pytest.raises(ValueError, match='EXACT CROSS-SPLIT LEAK'):
        public_build(cfg, tmp_path / 'out')


def test_license_gate(tmp_path):
    src = make_src(tmp_path / 'src')
    cfg = tmp_path / 'sources.yaml'
    cfg.write_text(f'sources:\n  - id: can\n    enabled: true\n    license_ok: false\n    root: {src}\n    quota_train: 2\n    mapping: {{}}\n')
    with pytest.raises(ValueError, match='license'):
        public_build(cfg, tmp_path / 'out')


def test_same_image_different_label_block(tmp_path):
    # Source only is valid; duplicate with conflicting labels must fail in dedup.
    from v3_dataset import dedup_within
    src = make_src(tmp_path / 'src')
    a = load_entries('x', src, 'train', {})[0]
    b = load_entries('y', src, 'train', {})[0]
    b.label_text = '1 0.1 0.1 0.5 0.1 0.5 0.5\n'
    with pytest.raises(ValueError, match='CONFLICTING'):
        dedup_within([a, b])


def test_indoor_approval_and_mixed(tmp_path):
    src = make_src(tmp_path / 'src')
    p = tmp_path / 'cfg.yaml'
    p.write_text(f'sources:\n  - id: can\n    enabled: true\n    license_ok: true\n    root: {src}\n    quota_train: 1\n    mapping: {{}}\n')
    public = tmp_path / 'public'
    public_build(p, public)

    indoor_original = tmp_path / 'indoor_original'
    colors = {'train':(0,255,0), 'val':(0,0,255), 'test':(255,255,0)}
    for split, color in colors.items():
        (indoor_original / 'images' / split).mkdir(parents=True)
        (indoor_original / 'labels' / split).mkdir(parents=True)
        Image.new('RGB', (32, 32), color).save(indoor_original / 'images' / split / '1.png')
        (indoor_original / 'labels' / split / '1.txt').write_text('2 .1 .1 .5 .1 .5 .5\n')
    (indoor_original / 'data.yaml').write_text('train: images/train\nval: images/val\ntest: images/test\n' + CANONICAL)
    icfg = tmp_path / 'indoor_cfg.yaml'
    icfg.write_text(f'indoor:\n  root: {indoor_original}\n  review_approved: false\n')
    with pytest.raises(ValueError, match='NOT approved'):
        indoor_build(icfg, tmp_path / 'fail')
    icfg.write_text(f'indoor:\n  root: {indoor_original}\n  review_approved: true\n')
    indoor = tmp_path / 'indoor'
    indoor_build(icfg, indoor)
    guard(public, indoor, out=tmp_path / 'cross.csv')
    mixed = tmp_path / 'mixed'
    mixed_build(public, indoor, mixed, public_fraction=0.25, seed=42)
    assert (mixed / 'data.yaml').is_file()
    assert len(load_built(mixed, 'val', 'mixed')) == len(load_built(indoor, 'val', 'indoor'))
    assert len(load_built(mixed, 'test', 'mixed')) == len(load_built(indoor, 'test', 'indoor'))