#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验 A: COCO -> 室内垃圾 两阶段正式训练

A_COCO_DIRECT_INDOOR_800

Phase 1: 官方 yolov8m-seg.pt (COCO) -> freeze=10 骨干冻结, 20 epoch
Phase 2: 从 Phase 1 best.pt 续训, freeze=0, 最多 180 epoch

严禁:
  - 使用 TACO best.pt 初始化
  - 使用 yolov8m-seg.yaml 随机初始化
  - 使用 test 集参与训练/早停/调参
"""
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROJ = os.path.join(ROOT, "experiments", "indoor_a", "A_coco_direct_800_2stage")
DATA = os.path.join(ROOT, "datasets/garbage_seg_merged_full1444/data.yaml")
COCO_W = os.environ.get("COCO_WEIGHTS", os.path.join(ROOT, "yolov8m-seg.pt"))
P1_DIR = os.path.join(PROJ, "phase1")
P2_DIR = os.path.join(PROJ, "phase2")
VAL_SPLIT = "val"

AUG = dict(
    hsv_h=0.015, hsv_s=0.35, hsv_v=0.30,
    degrees=8.0, translate=0.10, scale=0.25,
    fliplr=0.5, flipud=0.0, perspective=0.0,
    mosaic=0.30, mixup=0.0, copy_paste=0.10,
    multi_scale=0.10, mask_ratio=2, overlap_mask=True,
    auto_augment=None, erasing=0.0, shear=0.0, cutmix=0.0,
    bgr=0.0,
)
COMMON = dict(imgsz=800, batch=16, seed=42, deterministic=True, amp=True,
              device=0, workers=8, data=DATA, project=PROJ,
              val=True, plots=True, save=True, save_period=-1, cache=False,
              split=VAL_SPLIT, task="segment", verbose=True, pretrained=True,
              optimizer="AdamW", weight_decay=5e-4, exist_ok=False,
              single_cls=False, rect=False)
COMMON.update(AUG)

SUMMARY = {"experiment": "A_COCO_DIRECT_INDOOR_800", "stages": {}}
_state = {"peak_alloc": 0, "peak_reserved": 0, "gpu_total_used": 0, "stop": False}


def log(msg):
    print(msg, flush=True)


def sha256_file(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def monitor_thread():
    import torch
    while not _state["stop"]:
        try:
            a = torch.cuda.max_memory_allocated() / 2 ** 20
            r = torch.cuda.max_memory_reserved() / 2 ** 20
            _state["peak_alloc"] = max(_state["peak_alloc"], a)
            _state["peak_reserved"] = max(_state["peak_reserved"], r)
        except Exception:
            pass
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
            if out:
                _state["gpu_total_used"] = max(_state["gpu_total_used"], int(out[0]))
        except Exception:
            pass
        time.sleep(5)


def read_csv_metrics(csv_path):
    """返回 (rows, best_row_by_seg_map50_95)"""
    if not os.path.exists(csv_path):
        return [], None
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]

    def fv(r, k):
        try:
            return float(r.get(k, "nan"))
        except (TypeError, ValueError):
            return float("nan")

    best = None
    for r in rows:
        v = fv(r, "metrics/mAP50-95(M)")
        if best is None or (v == v and v > best[1]):
            best = (r, v)
    return rows, best


def run_stage(tag, name, model_path, epochs, freeze, lr0, lrf, cos_lr,
              warmup_epochs, patience, close_mosaic, out_dir):
    from ultralytics import YOLO

    log("\n" + "#" * 88)
    log(f"#  {tag}   起点权重: {model_path}")
    log(f"#  epochs={epochs} freeze={freeze} lr0={lr0} lrf={lrf} cos_lr={cos_lr} "
        f"warmup={warmup_epochs} patience={patience} close_mosaic={close_mosaic}")
    log("#" * 88)

    if os.path.exists(out_dir):
        raise SystemExit(f"[BLOCKER] 输出目录已存在, 拒绝自动编号: {out_dir}")

    src_sha = sha256_file(model_path)
    log(f"[{tag}] 起始权重 SHA256 = {src_sha}")

    args = dict(COMMON)
    args.update(model=model_path, epochs=epochs, freeze=freeze, lr0=lr0,
                lrf=lrf, cos_lr=cos_lr, warmup_epochs=warmup_epochs,
                patience=patience, close_mosaic=close_mosaic, name=name,
                project=PROJ)

    t0 = time.time()
    model = YOLO(model_path)
    results = model.train(**args)
    wall = time.time() - t0

    log(f"[{tag}] 训练结束, 墙钟耗时 {wall/60:.1f} min")

    # 真实冻结信息
    try:
        m = model.trainer.model
        fl = model.trainer.freeze_layer_names
        frozen_p = sum(p.numel() for n, p in m.named_parameters() if any(x in n for x in fl))
        train_p = sum(p.numel() for n, p in m.named_parameters() if not any(x in n for x in fl))
        opt = model.trainer.optimizer
        opt_info = {
            "class": type(opt).__name__,
            "n_param_groups": len(opt.param_groups),
            "lr_group0": opt.param_groups[0]["lr"],
            "weight_decay_group_used": model.trainer.args.weight_decay,
            "momentum": model.trainer.args.momentum,
            "nbs": model.trainer.args.nbs,
            "accumulate": int(model.trainer.accumulate),
            "effective_batch": int(model.trainer.args.batch * max(int(model.trainer.accumulate), 1)),
        }
        freeze_info = {
            "freeze_arg": int(model.trainer.args.freeze) if model.trainer.args.freeze is not None else None,
            "freeze_layer_names": list(fl),
            "frozen_layer_indices": sorted(int(x.split(".")[1]) for x in fl if x.startswith("model.")),
            "frozen_params": int(frozen_p),
            "trainable_params": int(train_p),
        }
    except Exception as e:
        log(f"[{tag}] 冻结信息提取失败: {e}")
        freeze_info, opt_info = {}, {}

    rows, best = read_csv_metrics(os.path.join(out_dir, "results.csv"))
    epochs_done = len(rows)
    best_epoch = int(float(best[0]["epoch"])) if best else None

    stage = {
        "name": name, "tag": tag, "start_weights": model_path,
        "start_weights_sha256": src_sha,
        "epochs_requested": epochs, "epochs_completed": epochs_done,
        "best_epoch_by_map50_95_mask": best_epoch,
        "wall_seconds": round(wall, 1), "wall_minutes": round(wall / 60, 2),
        "freeze": freeze_info, "optimizer": opt_info,
        "best_path": os.path.join(out_dir, "weights", "best.pt"),
        "last_path": os.path.join(out_dir, "weights", "last.pt"),
        "results_csv": os.path.join(out_dir, "results.csv"),
    }
    for p in (stage["best_path"], stage["last_path"]):
        stage[f"sha256:{os.path.basename(p)}"] = sha256_file(p) if os.path.exists(p) else None
    if epochs_done:
        stage["images_per_sec"] = round(1154 * epochs_done / wall, 1)
    SUMMARY["stages"][tag] = stage
    json.dump(SUMMARY, open(os.path.join(PROJ, "training_summary.json"), "w"),
              ensure_ascii=False, indent=2)
    return stage


def main():
    log("=" * 88)
    log("实验 A_COCO_DIRECT_INDOOR_800 — 两阶段正式训练")
    log("=" * 88)

    # ---------- 预检 ----------
    for p in (COCO_W, DATA):
        assert os.path.exists(p), f"缺失: {p}"
    for d in (P1_DIR, P2_DIR):
        if os.path.exists(d):
            raise SystemExit(f"[BLOCKER] 目标目录已存在, 拒绝覆盖: {d}")
    assert "taco" not in COCO_W.lower(), "禁止使用 TACO 权重"
    import torch
    from ultralytics import YOLO, __version__ as ULVER
    log(f"起点权重 = {COCO_W}  (COCO 官方, 非 TACO)")
    log(f"COCO 权重 SHA256 = {sha256_file(COCO_W)}")
    log(f"ultralytics = {ULVER}   torch = {torch.__version__}   "
        f"cuda = {torch.version.cuda}   GPU = {torch.cuda.get_device_name(0)}")
    SUMMARY["coco_weights_sha256"] = sha256_file(COCO_W)
    SUMMARY["ultralytics"] = ULVER
    SUMMARY["torch"] = torch.__version__
    SUMMARY["cuda"] = torch.version.cuda
    SUMMARY["gpu"] = torch.cuda.get_device_name(0)
    SUMMARY["data_yaml"] = DATA
    SUMMARY["common_args"] = {k: (v if not isinstance(v, (bytes,)) else str(v)) for k, v in COMMON.items()}
    SUMMARY["augmentation"] = AUG

    # 数据快照哈希 (labels 清单)
    man = []
    for sp in ("train", "val", "test"):
        d = os.path.join(ROOT, "datasets/garbage_seg_merged_full1444/labels", sp)
        man += sorted(f"{sp}/{x}" for x in os.listdir(d))
    SUMMARY["label_manifest_sha256"] = hashlib.sha256(
        "\n".join(man).encode()).hexdigest()
    SUMMARY["n_label_files"] = len(man)
    log(f"标签清单 SHA256 = {SUMMARY['label_manifest_sha256']}  (共 {len(man)} 个)")

    th = threading.Thread(target=monitor_thread, daemon=True)
    th.start()

    # ---------- Phase 1 ----------
    s1 = run_stage(
        tag="PHASE1", name="phase1", model_path=COCO_W,
        epochs=20, freeze=10, lr0=1e-3, lrf=0.01, cos_lr=False,
        warmup_epochs=3.0, patience=20, close_mosaic=0, out_dir=P1_DIR)

    p1_best = s1["best_path"]
    assert os.path.exists(p1_best), f"[BLOCKER] Phase1 best.pt 缺失: {p1_best}"
    log(f"\nPhase 1 完成. best.pt = {p1_best}")
    log(f"Phase 1 best.pt SHA256 = {s1.get('sha256:best.pt')}")

    # ---------- Phase 2 ----------
    s2 = run_stage(
        tag="PHASE2", name="phase2", model_path=p1_best,
        epochs=180, freeze=0, lr0=5e-4, lrf=0.05, cos_lr=True,
        warmup_epochs=2.0, patience=50, close_mosaic=20, out_dir=P2_DIR)

    _state["stop"] = True
    time.sleep(1)
    SUMMARY["resource"] = {
        "peak_gpu_alloc_MiB": round(_state["peak_alloc"], 1),
        "peak_gpu_reserved_MiB": round(_state["peak_reserved"], 1),
        "peak_nvidia_smi_used_MiB": _state["gpu_total_used"],
        "total_wall_seconds": round(s1["wall_seconds"] + s2["wall_seconds"], 1),
        "total_wall_minutes": round((s1["wall_seconds"] + s2["wall_seconds"]) / 60, 2),
        "total_epochs_completed": s1["epochs_completed"] + s2["epochs_completed"],
    }

    json.dump(SUMMARY, open(os.path.join(PROJ, "training_summary.json"), "w"),
              ensure_ascii=False, indent=2)
    log("\n" + "=" * 88)
    log("两阶段训练完成")
    log(f"  Phase1: {s1['epochs_completed']} epoch, best_epoch={s1['best_epoch_by_map50_95_mask']}, "
        f"{s1['wall_minutes']} min")
    log(f"  Phase2: {s2['epochs_completed']} epoch, best_epoch={s2['best_epoch_by_map50_95_mask']}, "
        f"{s2['wall_minutes']} min")
    log(f"  峰值显存(alloc)={SUMMARY['resource']['peak_gpu_alloc_MiB']} MiB")
    log(f"  最终权重: {s2['best_path']}")
    log("=" * 88)


if __name__ == "__main__":
    main()
