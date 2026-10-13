#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验 A 训练前 GPU 冒烟测试 (基于训练器真实 dataloader)

验证:
  1. yolov8m-seg.pt 加载并替换为 9 类分割头
  2. freeze=10 实际冻结层名称 / 可训练参数量 / 实际优化器配置
  3. 800 分辨率前向 + 反向传播
  4. batch 显存探测 (16 -> 8 -> 4)
  5. 实例分割损失有限性 (无 NaN/Inf)
  6. 所有类别 ID 合法
  7. 增强后图片与掩码对齐 (实例索引图 唯一值数 == 实例数; 掩码bbox vs 标签bbox IoU)
  8. 增强可视化样张
  9. label_smoothing 支持性
"""
import json
import os
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "experiments", "indoor_a", "A_coco_direct_800_2stage", "smoke")
DATA = os.path.join(ROOT, "datasets/garbage_seg_merged_full1444/data.yaml")
WEIGHTS = os.environ.get("COCO_WEIGHTS", os.path.join(ROOT, "yolov8m-seg.pt"))
NC = 9
CLASS_NAMES = ["bottle", "box", "can", "cigarette_butt", "food_waste",
               "other", "paper", "plastic_bag", "wrapper"]

AUG = dict(
    hsv_h=0.015, hsv_s=0.35, hsv_v=0.30, degrees=8.0, translate=0.10,
    scale=0.25, fliplr=0.5, flipud=0.0, mosaic=0.30, mixup=0.0,
    copy_paste=0.10, multi_scale=0.10, perspective=0.0,
    mask_ratio=2, overlap_mask=True,
)
result = {}
os.makedirs(OUT, exist_ok=True)


def sec(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


from ultralytics import YOLO, __version__ as ULVER
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.data.utils import check_det_dataset

# ---------------------------------------------------------------- 1. 数据与类别
sec("[1] 加载权重 / 数据集类别核对")
model = YOLO(WEIGHTS)
m = model.model
print(f"  权重            : {WEIGHTS}")
print(f"  原始 nc         : {m.yaml.get('nc')}  (COCO)")
print(f"  总参数量        : {sum(p.numel() for p in m.parameters()):,}")
d = check_det_dataset(DATA)
names_map = {i: n for i, n in enumerate(d["names"])} if isinstance(d["names"], (list, tuple)) else d["names"]
print(f"  数据集 nc       : {d.get('nc')}   names={names_map}")
assert len(names_map) == NC
result["orig_nc"] = int(m.yaml.get("nc"))
result["coco_params"] = int(sum(p.numel() for p in m.parameters()))
result["dataset_nc"] = int(d.get("nc"))
result["class_names"] = [names_map[i] for i in range(NC)]

# ---------------------------------------------------------------- 2. 构建训练器
sec("[2] 使用 Phase 1 真实参数构建训练器 (freeze=10, imgsz=800)")


def build_trainer(batch, name, epochs=20, freeze=10, phase1=True):
    ov = dict(
        model=WEIGHTS, data=DATA, task="segment", imgsz=800, epochs=epochs,
        batch=batch, freeze=freeze, device=0, workers=2, seed=42,
        deterministic=True, amp=True, optimizer="AdamW", lr0=1e-3,
        weight_decay=5e-4, project=OUT, name=name, exist_ok=True,
        verbose=False, plots=False, save=False, val=False, cache=False,
        close_mosaic=0 if phase1 else 20,
    )
    ov.update(AUG)
    return SegmentationTrainer(overrides=ov)


t16 = build_trainer(16, "_probe_bs16")
t16._setup_train()
print(f"  实际 args.imgsz        : {t16.args.imgsz}")
print(f"  实际 args.batch        : {t16.args.batch}")
print(f"  实际 args.freeze       : {t16.args.freeze}")
print(f"  实际 args.mask_ratio   : {t16.args.mask_ratio}")
print(f"  实际 args.copy_paste   : {t16.args.copy_paste}")
print(f"  实际 args.multi_scale  : {t16.args.multi_scale}")
print(f"  实际 args.overlap_mask : {t16.args.overlap_mask}")
print(f"  实际优化器             : {type(t16.optimizer).__name__}  "
      f"lr={t16.optimizer.param_groups[0]['lr']:.6g}  wd={t16.optimizer.param_groups[0]['weight_decay']:.6g}")
print(f"  AMP 自检结果           : {bool(t16.amp)}")
result["actual_args"] = {k: getattr(t16.args, k)
                         for k in ["imgsz", "batch", "freeze", "mask_ratio", "copy_paste",
                                   "multi_scale", "overlap_mask", "mosaic", "mixup",
                                   "degrees", "translate", "scale", "fliplr", "flipud",
                                   "hsv_h", "hsv_s", "hsv_v", "optimizer", "lr0",
                                   "weight_decay", "close_mosaic", "deterministic",
                                   "amp", "seed", "workers"]}
result["optimizer_actual"] = {
    "class": type(t16.optimizer).__name__,
    "lr": t16.optimizer.param_groups[0]["lr"],
    "weight_decay": t16.optimizer.param_groups[0]["weight_decay"],
    "n_param_groups": len(t16.optimizer.param_groups),
}
result["amp_check"] = bool(t16.amp)
# 实际设备/模型
print(f"  模型设备               : {next(t16.model.parameters()).device}")
print(f"  训练集/验证集样本数    : {len(t16.train_loader.dataset)} / {len(t16.test_loader.dataset)}")

# 冻结层详情
fl = t16.freeze_layer_names
frozen_p = sum(p.numel() for n, p in t16.model.named_parameters() if any(x in n for x in fl))
train_p = sum(p.numel() for n, p in t16.model.named_parameters() if not any(x in n for x in fl))
print(f"\n  冻结匹配前缀     : {fl}")
print(f"  冻结参数         : {frozen_p:,}")
print(f"  可训练参数       : {train_p:,}  ({train_p/(frozen_p+train_p):.2%})")
frozen_idx = sorted(int(x.split(".")[1]) for x in fl if x.startswith("model."))
children = dict(t16.model.model.named_children())
ftypes = {i: type(children[str(i)]).__name__ for i in frozen_idx}
print(f"  被冻结层索引     : {frozen_idx}")
print(f"  被冻结层类型     : {ftypes}")
result["freeze_layer_names"] = fl
frozen_all = [n for n, p in t16.model.named_parameters() if any(x in n for x in fl)]
open(os.path.join(OUT, "frozen_parameters.txt"), "w").write(
    f"# freeze = {t16.args.freeze}\n# freeze_layer_names = {fl}\n"
    f"# frozen params = {frozen_p:,}\n# trainable params = {train_p:,}\n\n"
    + "\n".join(frozen_all) + "\n")
print(f"  完整冻结参数清单已写入 {os.path.join(OUT,'frozen_parameters.txt')} (共 {len(frozen_all)} 个张量)")
result["n_frozen_tensors"] = len(frozen_all)
result["frozen_layer_indices"] = frozen_idx
result["frozen_layer_types"] = {str(k): v for k, v in ftypes.items()}
result["frozen_params"] = int(frozen_p)
result["trainable_params"] = int(train_p)
result["trainable_ratio"] = round(train_p / (frozen_p + train_p), 4)

# ---------------------------------------------------------------- 3. 增强对齐检查
sec("[3] 增强后图片与掩码对齐检查 (训练器真实 dataset)")
ds = t16.train_loader.dataset
print(f"  dataset imgsz={ds.imgsz}  augment={ds.augment}  "
      f"mask_ratio={t16.args.mask_ratio}  overlap_mask={t16.args.overlap_mask}")
assert ds.imgsz == 800, "数据集 imgsz 未生效"

# 实际生效的增强链 (确认 copy_paste / multi_scale / mosaic 是否真正装配)
print("  实际装配的增强变换链 (递归展开嵌套 pre_transform):")
tr_names = []


def walk(tr, depth=0):
    name = type(tr).__name__
    tr_names.append(name)
    p = getattr(tr, "p", None)
    extra = ""
    if name == "Format":
        extra = f"mask_ratio={getattr(tr,'mask_ratio',None)} overlap={getattr(tr,'mask_overlap',None)}"
    if name == "RandomPerspective":
        extra = (f"degrees={tr.degrees} translate={tr.translate} scale={tr.scale} "
                 f"shear={tr.shear} perspective={tr.perspective}")
    print(f"    {'  '*depth}- {name:22} p={p}  {extra}")
    for sub in (getattr(tr, "transforms", None) or []):
        walk(sub, depth + 1)
    pt = getattr(tr, "pre_transform", None)
    if pt is not None and not isinstance(pt, list):
        print(f"    {'  '*depth}  [pre_transform of {name}]")
        walk(pt, depth + 2)
    elif pt:
        for s in pt:
            walk(s, depth + 2)


for tr in ds.transforms.transforms:
    walk(tr)
result["transform_chain"] = tr_names
print(f"  >>> CopyPaste 已装配 : {'CopyPaste' in tr_names}")
print(f"  >>> Mosaic 已装配    : {'Mosaic' in tr_names}")
print(f"  >>> RandomPerspective: {'RandomPerspective' in tr_names} (几何增强, 同步变换掩码)")
result["copy_paste_assembled"] = "CopyPaste" in tr_names
result["mosaic_assembled"] = "Mosaic" in tr_names

rng = np.random.default_rng(42)
sample_idx = rng.choice(len(ds), size=16, replace=False)
align, viz, ious_all = [], [], []
max_cls = -1
for si in sample_idx:
    s = ds[int(si)]
    img, masks, cls, bboxes = s["img"], s["masks"], s["cls"], s["bboxes"]
    C, H, W = img.shape
    ninst = int(cls.shape[0])
    cl = cls.squeeze(-1) if cls.ndim > 1 else cls
    max_cls = max(max_cls, int(cl.max()) if ninst else -1)
    uniq = np.unique(masks.numpy()) if masks.numel() else np.array([])
    n_idx = int((uniq > 0).sum())
    # 掩码 bbox vs 标签 bbox IoU (按实例索引 k 对应第 k-1 个标注)
    ious = []
    mh, mw = masks.shape[1], masks.shape[2]
    sx, sy = mw / W, mh / H          # 掩码坐标 -> 图像归一化坐标的比例系数
    for k in range(1, n_idx + 1):
        if k - 1 >= ninst:
            break
        mk = (masks[0] == k).numpy()
        ys, xs = np.nonzero(mk)
        if len(xs) == 0:
            continue
        # 掩码 bbox 转到图像像素坐标
        mb = np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float64) / np.array([sx, sy, sx, sy])
        # 标签 bbox 由归一化 xywh -> 像素 xyxy
        cx, cy, bw, bh = bboxes[k - 1, :4].numpy().astype(np.float64)
        bb = np.array([(cx - bw / 2) * W, (cy - bh / 2) * H,
                       (cx + bw / 2) * W, (cy + bh / 2) * H])
        ix1, iy1 = max(mb[0], bb[0]), max(mb[1], bb[1])
        ix2, iy2 = min(mb[2], bb[2]), min(mb[3], bb[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        ua = (mb[2] - mb[0]) * (mb[3] - mb[1]) + (bb[2] - bb[0]) * (bb[3] - bb[1]) - inter
        ious.append(float(inter / ua) if ua > 0 else 0.0)
    if ious:
        ious_all.extend(ious)
    align.append({
        "index": int(si), "img_shape": [int(H), int(W)],
        "mask_shape": [int(x) for x in masks.shape],
        "n_instances": ninst, "mask_unique_nonzero": n_idx,
        "match": n_idx == ninst, "img_dtype": str(img.dtype),
        "img_finite": bool(torch.isfinite(img.float()).all()),
        "cls_ids": sorted(set(int(x) for x in cl.tolist())),
        "bboxes_in_range": bool(((bboxes[:, :4] >= -1e-6) & (bboxes[:, :4] <= 1 + 1e-6)).all()) if ninst else True,
        "mask_ratio_effective": round(H / masks.shape[1], 3),
    })
    viz.append((img, masks, cl, bboxes, int(si)))

bad = [a for a in align if not a["match"]]
print(f"  抽样 {len(align)} 个增强样本:")
print(f"    imgsz 固定 800 : {all(a['img_shape'] == [800, 800] for a in align)}")
print(f"    掩码唯一值数 != 实例数 : {len(bad)}")
print(f"    图像 dtype / 有限值    : {align[0]['img_dtype']} / {all(a['img_finite'] for a in align)}")
print(f"    掩码下采样比           : {align[0]['mask_ratio_effective']} "
      f"(masks {align[0]['mask_shape']} -> mask_ratio=2 生效)")
print(f"    最大类别 ID            : {max_cls} (合法 0-{NC-1})")
print(f"    掩码bbox vs 标注bbox 平均 IoU: {np.mean(ious_all):.4f} (min={np.min(ious_all):.4f}, n={len(ious_all)})")
result["aug_align"] = align
result["aug_all_match"] = len(bad) == 0
result["aug_max_class_id"] = max_cls
result["mask_bbox_iou_mean"] = round(float(np.mean(ious_all)), 4)
result["mask_bbox_iou_min"] = round(float(np.min(ious_all)), 4)
result["mask_ratio_effective"] = align[0]["mask_ratio_effective"]
assert max_cls < NC, "训练标签存在非法类别ID"
if bad:
    print(f"    [注意] {len(bad)} 个样本掩码索引数 < 实例数 (mask_ratio=2 下采样可能吞掉极小实例)")
    for b_ in bad[:3]:
        print(f"        idx={b_['index']} instances={b_['n_instances']} mask_idx={b_['mask_unique_nonzero']} "
              f"cls={b_['cls_ids']}")
result["aug_mismatch_samples"] = len(bad)

# ---------------------------------------------------------------- 4. 可视化
sec("[4] 增强可视化样张")
import cv2

tiles = []
for img, masks, cl, bboxes, si in viz[:6]:
    im = (img.permute(1, 2, 0).numpy()[:, :, ::-1]).astype(np.uint8).copy()
    mk = masks.numpy()
    for k in range(1, int(mk.max()) + 1):
        mi = (mk[0] == k).astype(np.uint8)
        if mi.sum() == 0:
            continue
        mfull = cv2.resize(mi, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(mfull, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cnts, -1, (0, 255, 255), 2)
        ys, xs = np.nonzero(mfull)
        name = CLASS_NAMES[int(cl[k - 1])] if k - 1 < len(cl) else "?"
        cv2.putText(im, name, (max(2, int(xs.min())), max(14, int(ys.min()) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    im = cv2.resize(im, (420, 420))
    cv2.putText(im, f"#{si}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    tiles.append(im)
rows = [np.hstack(tiles[i:i + 3]) for i in range(0, 6, 3)]
montage = np.vstack(rows)
vp = os.path.join(OUT, "augmented_samples.jpg")
cv2.imwrite(vp, montage)
print(f"  已保存: {vp}  shape={montage.shape}")
result["aug_viz"] = vp

# ---------------------------------------------------------------- 5. 显存探测
sec("[5] 800 分辨率前向+反向 与 batch 显存探测")


def probe(batch):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        t = build_trainer(batch, f"_probe_bs{batch}")
        t._setup_train()
        t.model.train()
        it = iter(t.train_loader)
        b = t.preprocess_batch(next(it))
        loss, items = t.model(b)
        loss = loss.sum() if loss.ndim else loss
        loss.backward()
        t.optimizer.step()
        t.optimizer.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        if isinstance(items, dict):
            items_d = {k: float(v.detach().cpu()) for k, v in items.items()}
        else:
            keys = ["box_loss", "seg_loss", "cls_loss", "dfl_loss"]
            vals = [float(x) for x in items.detach().cpu().reshape(-1)]
            items_d = dict(zip(keys[:len(vals)], vals))
        r = {"batch": batch, "status": "OK", "loss_total": float(loss.detach()),
             "loss_items": items_d,
             "peak_mem_MiB": round(peak, 1),
             "n_trainable_actual": int(sum(p.numel() for p in t.model.parameters() if p.requires_grad)),
             "n_instances_in_batch": int(b["cls"].numel()) if "cls" in b else None}
        del t, b, loss, items
        torch.cuda.empty_cache()
        return r
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {"batch": batch, "status": "OOM", "error": str(e)[:160]}
    except Exception as e:
        torch.cuda.empty_cache()
        import traceback
        return {"batch": batch, "status": f"ERR:{type(e).__name__}",
                "error": str(e)[:300], "tb": traceback.format_exc()[-800:]}


del t16
torch.cuda.empty_cache()
probes = []
for bs in (16, 8, 4):
    print(f"  --- batch={bs} ---", flush=True)
    t0 = time.time()
    r = probe(bs)
    r["seconds"] = round(time.time() - t0, 1)
    if r["status"] == "OK":
        print(f"      OK  {r['seconds']}s  峰值显存={r['peak_mem_MiB']} MiB  loss={r['loss_total']:.4f}", flush=True)
        print(f"      loss_items = { {k: round(v, 4) for k, v in r['loss_items'].items()} }", flush=True)
    else:
        print(f"      {r['status']}  用时 {r['seconds']}s  {r.get('error','')[:200]}", flush=True)
        if r.get("tb"):
            print(r["tb"], flush=True)
    probes.append(r)
    if r["status"] == "OK":
        break

ok = [p for p in probes if p["status"] == "OK"]
result["batch_probes"] = [{k: v for k, v in p.items() if k not in ("error", "tb")} for p in probes]
if ok:
    sel = ok[0]
    result["selected_batch"] = sel["batch"]
    result["peak_mem_MiB"] = sel["peak_mem_MiB"]
    fin = bool(np.isfinite(sel["loss_total"]) and all(np.isfinite(v) for v in sel["loss_items"].values()))
    result["loss_finite"] = fin
    print(f"\n  >>> 选定 batch={sel['batch']}  峰值显存={sel['peak_mem_MiB']} MiB")
    print(f"  >>> 可训练参数(实际)={sel['n_trainable_actual']:,}")
    print(f"  >>> loss 有限性(无NaN/Inf)={fin}")
else:
    result["selected_batch"] = None
    print("\n  >>> 全部 batch 探测失败, 需人工介入")

# ---------------------------------------------------------------- 6. 环境
sec("[6] 环境与 GPU")
p = torch.cuda.get_device_properties(0)
result["env"] = {
    "gpu": p.name, "gpu_mem_GiB": round(p.total_memory / 2 ** 30, 2),
    "cc": f"{p.major}.{p.minor}", "torch": torch.__version__,
    "cuda": torch.version.cuda, "ultralytics": ULVER,
    "python": __import__("sys").version.split()[0],
}
for k, v in result["env"].items():
    print(f"  {k:14}: {v}")

# ---------------------------------------------------------------- 7. label_smoothing
sec("[7] label_smoothing 支持性")
from ultralytics.cfg import DEFAULT_CFG_DICT
ls = "label_smoothing" in DEFAULT_CFG_DICT
result["label_smoothing_supported"] = bool(ls)
print(f"  DEFAULT_CFG 含 label_smoothing: {ls}")
print(f"  => 状态: {'APPLIED' if ls else 'NOT_APPLIED (该版本已移除该参数, 不使用)'}")

json.dump(result, open(os.path.join(OUT, "smoke_result.json"), "w"), ensure_ascii=False, indent=2)
print(f"\n结果已写入: {os.path.join(OUT, 'smoke_result.json')}")
