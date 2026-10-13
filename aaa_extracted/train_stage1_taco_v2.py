#!/usr/bin/env python
"""YOLOv8m-Seg Stage 1 正式训练入口（TACO v2, 9 类实例分割）。

本脚本实现「other 类别分类损失权重 = 0.7、其余 8 类 = 1.0」的精确类别加权：

* 从 ``data.yaml`` 自动解析 ``other`` 的真实类别 ID（不假定它是最后一类）；
* 覆盖 ``SegmentationTrainer.set_class_weights()``，把权重张量写到真正参与损失计算的
  模型（``unwrap_model`` 后的 SegmentationModel）上；
* 权重只作用于 ``v8SegmentationLoss`` 中分类分支（cls）的 BCE，
  box / seg / DFL 损失与其权重完全不变；
* ``cls_pw = 0.0``，不启用 Ultralytics 自动逆频率类别加权；
* 不修改 conda 环境内的 Ultralytics 源码。

用法::

    # 启动前验证（不训练，不产生训练产物）
    python train_stage1_taco_v2.py --verify-only

    # 验证通过后正式训练
    python train_stage1_taco_v2.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

from ultralytics import YOLO
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import unwrap_model

ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "datasets" / "taco" / "yolo_seg" / "data.yaml"
WEIGHTS = ROOT / "yolov8m-seg.pt"
PROJECT = ROOT / "taco_baseline_v2"
RUN_NAME = "train"
SAVE_DIR = PROJECT / RUN_NAME

OTHER_WEIGHT = 0.7
DEFAULT_WEIGHT = 1.0

TRAIN_ARGS: dict = {
    "task": "segment",
    "data": str(DATA_YAML),
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "device": 0,
    "workers": 8,
    "patience": 15,
    "optimizer": "auto",
    "amp": True,
    "seed": 42,
    "plots": True,
    "save": True,
    "cls_pw": 0.0,
    "project": str(PROJECT),
    "name": RUN_NAME,
    "pretrained": True,
    "val": True,
    "exist_ok": False,
}


# --------------------------------------------------------------------------------------
# 类别权重
# --------------------------------------------------------------------------------------
def resolve_class_weights(data_yaml: Path = DATA_YAML) -> tuple[np.ndarray, int, dict[int, str]]:
    """从 data.yaml 解析类别并生成权重向量（other=DEFAULT, 其余=1.0）。"""
    cfg = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    if "names" not in cfg:
        raise ValueError(f"{data_yaml} 缺少 names 字段")
    raw = cfg["names"]
    names = (
        {int(k): str(v) for k, v in raw.items()}
        if isinstance(raw, dict)
        else {i: str(v) for i, v in enumerate(raw)}
    )
    ids = sorted(names)
    if ids != list(range(len(ids))):
        raise ValueError(f"data.yaml 类别 ID 不是 0..nc-1 连续: {ids}")
    nc = len(ids)
    other_ids = [i for i in ids if names[i] == "other"]
    if len(other_ids) != 1:
        raise ValueError(f"data.yaml 中名为 'other' 的类别数量 = {len(other_ids)}，期望恰好 1 个")
    other_id = other_ids[0]
    weights = np.full(nc, DEFAULT_WEIGHT, dtype=np.float32)
    weights[other_id] = OTHER_WEIGHT
    return weights, other_id, names


def _loss_instance(model: torch.nn.Module):
    """取出真正计算损失的 criterion（兼容 E2ELoss 之类的包装）。"""
    crit = getattr(model, "criterion", None)
    if crit is None:
        return None
    inner = getattr(crit, "inner", None)
    return inner if inner is not None and hasattr(inner, "class_weights") else crit


def _sync_criterion(model: torch.nn.Module, weights: torch.Tensor) -> bool:
    """若 criterion 已构建，则同步 class_weights（保证不会被 __init__ 的 None 覆盖）。"""
    crit = _loss_instance(model)
    if crit is None or not hasattr(crit, "class_weights"):
        return False
    crit.class_weights = weights.detach().reshape(1, 1, -1).clone()
    return True


def _set_weights_on_model(model: torch.nn.Module, weights: np.ndarray) -> torch.Tensor:
    """把权重张量设置到实际参与损失计算的模型上。"""
    model = unwrap_model(model)
    if hasattr(model, "student_model"):  # 蒸馏：student 负责构建 criterion
        model = model.student_model
    device = next(model.parameters()).device
    w = torch.from_numpy(np.ascontiguousarray(weights, dtype=np.float32)).to(device)
    model.class_weights = w
    _sync_criterion(model, w)
    return w


# --------------------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------------------
class CustomWeightedSegTrainer(SegmentationTrainer):
    """仅覆盖类别权重设置，其余训练逻辑完全沿用 Ultralytics ``SegmentationTrainer``。"""

    def set_class_weights(self) -> None:
        if float(self.args.cls_pw) != 0.0:
            raise ValueError(
                f"cls_pw 必须为 0.0（避免启用自动逆频率类别加权），当前为 {self.args.cls_pw}"
            )
        weights, other_id, names = resolve_class_weights(Path(self.args.data))
        w = _set_weights_on_model(self.model, weights)
        synced = _sync_criterion(unwrap_model(self.model), w) or _sync_criterion(
            getattr(unwrap_model(self.model), "student_model", unwrap_model(self.model)), w
        )
        LOGGER.info(
            "[class-weights] other(id=%d)=%.3f, 其余 %d 类=%.3f, 向量=%s, criterion 当时已同步=%s"
            "（False 表示 criterion 由首次前向惰性构建，将直接读取 model.class_weights）",
            other_id,
            weights[other_id],
            len(weights) - 1,
            DEFAULT_WEIGHT,
            np.array2string(w.detach().cpu().numpy(), precision=3),
            synced,
        )


# --------------------------------------------------------------------------------------
# 验证：权重张量 / 数值传播 / 真实梯度
# --------------------------------------------------------------------------------------
def verify_weights(trainer, expected: np.ndarray, other_id: int, run_grad_check: bool = True) -> dict:
    """启动前验证：证明自定义权重被 v8SegmentationLoss 真正读取并参与计算。"""
    report: dict = {}
    model = unwrap_model(trainer.model)
    nc = len(expected)
    exp = torch.as_tensor(expected, dtype=torch.float32)
    dev = next(model.parameters()).device

    # --- 1. 模型上的权重张量 ---
    mw = getattr(model, "class_weights", None)
    if mw is None:
        raise RuntimeError("model.class_weights 未设置")
    if mw.numel() != nc:
        raise RuntimeError(f"权重张量长度 {mw.numel()} != 类别总数 {nc}")
    if not torch.allclose(mw.detach().cpu().float(), exp, atol=1e-6):
        raise RuntimeError(f"model.class_weights = {mw.detach().cpu().tolist()} 与期望 {exp.tolist()} 不符")
    report["model_class_weights"] = mw.detach().cpu().tolist()
    report["length_ok"] = True
    report["other_is_07"] = bool(np.isclose(float(mw[other_id]), OTHER_WEIGHT, atol=1e-6))
    report["others_are_1"] = bool(
        all(np.isclose(float(mw[i]), DEFAULT_WEIGHT, atol=1e-6) for i in range(nc) if i != other_id)
    )
    if not (report["other_is_07"] and report["others_are_1"]):
        raise RuntimeError(f"权重向量不符合预期: {report['model_class_weights']}")

    # --- 2. 触发 criterion 构建（与训练路径一致），确认它读取了权重 ---
    # 注意：DataLoader 每次 iter() 的 shuffle 首批不同，故全程只物化一次并复用，
    # 保证后续所有对照使用的是**同一个 batch**。
    batch = _materialize_batch(trainer)
    with torch.no_grad():
        model.train()
        loss, items = model(batch)

    crit = _loss_instance(model)
    if crit is None or getattr(crit, "class_weights", None) is None:
        raise RuntimeError("v8SegmentationLoss.class_weights 为 None —— 权重未被损失读取")
    cw = crit.class_weights
    if cw.numel() != nc:
        raise RuntimeError(f"criterion.class_weights 长度 {cw.numel()} != {nc}")
    if not torch.allclose(cw.reshape(-1).detach().cpu().float(), exp, atol=1e-6):
        raise RuntimeError(f"criterion.class_weights = {cw.reshape(-1).detach().cpu().tolist()} 与期望不符")
    if type(crit).__name__ != "v8SegmentationLoss":
        raise RuntimeError(f"损失类型为 {type(crit).__name__}，期望 v8SegmentationLoss")
    report["criterion"] = type(crit).__name__
    report["criterion_class_weights"] = cw.reshape(-1).detach().cpu().tolist()
    report["criterion_reads_weights"] = True
    report["first_batch_loss"] = {k: float(v) for k, v in items.items()}

    # --- 3. 数值传播：复刻损失源码中的 bce_loss *= class_weights ---
    gen = torch.Generator(device=dev).manual_seed(0)
    pred_scores = torch.randn(2, 16, nc, device=dev, generator=gen)
    target_scores = torch.rand(2, 16, nc, device=dev, generator=gen)
    base = crit.bce(pred_scores, target_scores)
    per_channel = (base * crit.class_weights).sum(dim=(0, 1)) / base.sum(dim=(0, 1))
    if not torch.allclose(per_channel, exp.to(dev), atol=1e-5):
        raise RuntimeError(
            f"BCE 逐通道加权比例 {per_channel.detach().cpu().tolist()} != 期望 {exp.tolist()}"
        )
    report["bce_per_channel_ratio"] = per_channel.detach().cpu().tolist()
    report["bce_weighting_applied"] = True

    # --- 4. 真实数据闭式检查：cls 分支对 other 通道权重严格线性；seg/box/dfl 完全不受影响 ---
    lin = _weight_linearity_check(model, crit, batch, nc, other_id)
    report["linearity_check"] = lin

    # --- 5. 真实梯度检查：cls 分支梯度在有/无权重时必须不同（同一个 batch）---
    if run_grad_check:
        gc = _gradient_check(model, crit, batch)
        report["grad_check"] = gc
        # 交叉自洽：乘 1.0 与置 None 在浮点下必须逐位相同；两个比值必须一致
        if gc["cls_loss_without_weights"] != lin["cls_loss_w_other_1.0"]:
            raise RuntimeError(
                "同一 batch 下 class_weights=None 与 class_weights=全 1 得到的 cls_loss 不一致："
                f"{gc['cls_loss_without_weights']} vs {lin['cls_loss_w_other_1.0']}"
            )
        if abs(gc["ratio_with_over_without"] - lin["ratio_0.7_over_1.0"]) > 1e-9:
            raise RuntimeError(
                "两个独立检查得到的比值不一致："
                f"{gc['ratio_with_over_without']} vs {lin['ratio_0.7_over_1.0']}"
            )
        report["cross_check"] = {
            "class_weights_None_equals_ones": True,
            "ratios_agree": True,
        }
    return report


def _materialize_batch(trainer) -> dict:
    """取一个真实训练 batch 并预处理；三次对照共用同一个 tensor，保证输入完全一致。"""
    torch.manual_seed(0)
    return trainer.preprocess_batch(next(iter(trainer.train_loader)))


_LOSS_KEYS = ("box_loss", "seg_loss", "cls_loss", "dfl_loss")


def _loss_items_with_weights(model, crit, batch, weights, keys=_LOSS_KEYS) -> dict:
    """固定输入与模型状态，仅切换 crit.class_weights，返回各项损失值。"""
    saved = crit.class_weights
    crit.class_weights = None if weights is None else weights.reshape(1, 1, -1).clone()
    try:
        model.train()
        with torch.no_grad():
            _, items = model(batch)
    finally:
        crit.class_weights = saved
    return {k: float(items[k]) for k in keys}


def _weight_linearity_check(model: torch.nn.Module, crit, batch: dict, nc: int, other_id: int) -> dict:
    """闭式验证 ``class_weights`` 的真实作用方式（真实数据 / 同一 batch / 同一模型状态）。

    源码为 ``bce_loss *= class_weights`` 后求和，故当**只有 other 通道**权重 α 变化时，
    cls_loss 关于 α 严格仿射：

        L(0.7) == 0.7 * L(1.0) + 0.3 * L(0.0)

    同时 box / seg / dfl 三项与 class_weights 无关，必须**逐位相同**（这同时证明了三次前向的
    输入与归一化条件完全一致，排除"加权前后用了不同计算条件"）。反向即可解出 other 通道在
    BCE 中占的质量比 s = 1 − L(0)/L(1)，用于核对 L(0.7)/L(1) = 1 − 0.3s。
    """
    dev = crit.class_weights.device
    ones = torch.ones(nc, device=dev)
    w_other_07 = ones.clone()
    w_other_07[other_id] = OTHER_WEIGHT
    w_other_00 = ones.clone()
    w_other_00[other_id] = 0.0

    r1 = _loss_items_with_weights(model, crit, batch, ones)
    r07 = _loss_items_with_weights(model, crit, batch, w_other_07)
    r00 = _loss_items_with_weights(model, crit, batch, w_other_00)

    # (a) 非 cls 损失必须完全不变 —— 证明 other=0.7 没有同步削弱 Mask / Box / DFL 损失
    unaffected = {}
    for k in ("box_loss", "seg_loss", "dfl_loss"):
        d = max(abs(r1[k] - r07[k]), abs(r1[k] - r00[k]))
        unaffected[k] = d
        if d != 0.0:
            raise RuntimeError(
                f"{k} 随 class_weights 发生变化（最大偏差 {d:.3e}），权重影响范围超出 cls 通道"
            )

    # (b) cls_loss 的严格仿射关系
    predicted_07 = OTHER_WEIGHT * r1["cls_loss"] + (1.0 - OTHER_WEIGHT) * r00["cls_loss"]
    err = abs(predicted_07 - r07["cls_loss"])
    if err > 1e-6 * max(abs(predicted_07), 1.0):
        raise RuntimeError(
            f"cls_loss 对 other 权重不是仿射关系：实测 {r07['cls_loss']:.6f} vs "
            f"0.7*L(1)+0.3*L(0)={predicted_07:.6f}（偏差 {err:.3e}）"
        )

    # (c) 方向：加权必须使 cls_loss 下降
    if not r07["cls_loss"] < r1["cls_loss"]:
        raise RuntimeError(
            f"加权后 cls_loss({r07['cls_loss']:.6f}) 未低于未加权值({r1['cls_loss']:.6f})，权重方向异常"
        )

    share_other = 1.0 - r00["cls_loss"] / r1["cls_loss"]
    return {
        "cls_loss_w_other_1.0": r1["cls_loss"],
        "cls_loss_w_other_0.7": r07["cls_loss"],
        "cls_loss_w_other_0.0": r00["cls_loss"],
        "ratio_0.7_over_1.0": r07["cls_loss"] / r1["cls_loss"],
        "affine_prediction_0.7": predicted_07,
        "affine_abs_error": err,
        "affine_holds": True,
        "others_weight_share_of_bce": share_other,
        "non_cls_loss_max_abs_change": unaffected,
        "non_cls_losses_unaffected": True,
    }


def _cls_grad_vector(model: torch.nn.Module) -> tuple[torch.Tensor, int]:
    head = model.model[-1]
    grads = []
    n_params = 0
    for name, p in head.named_parameters():
        if name.startswith("cv3"):
            n_params += 1
            if p.grad is not None:
                grads.append(p.grad.detach().reshape(-1).float().cpu())
    if not grads:
        raise RuntimeError("未找到分类分支（cv3）梯度")
    return torch.cat(grads), n_params


def _gradient_check(model: torch.nn.Module, crit, batch: dict) -> dict:
    """同一真实 batch 下对比 cls 分支梯度。

    命名约定（务必按字面读）：
      ``cls_loss_with_weights``    = 应用 other=0.7 权重后的 cls_loss
      ``cls_loss_without_weights`` = 完全不加权（class_weights=None）的 cls_loss
    """
    params = [p for p in model.model[-1].cv3.parameters() if p.requires_grad]

    def run_once() -> tuple[torch.Tensor, float]:
        for p in params:
            p.grad = None
        model.train()
        loss, items = model(batch)
        loss.sum().backward()
        g, _ = _cls_grad_vector(model)
        return g, float(items["cls_loss"])

    saved = crit.class_weights
    g_with, cls_with = run_once()
    crit.class_weights = None
    g_without, cls_without = run_once()
    crit.class_weights = saved

    rel = float((g_with - g_without).norm() / (g_without.norm() + 1e-12))
    for p in params:
        p.grad = None
    if rel < 1e-6:
        raise RuntimeError(f"cls 分支梯度在加权前后几乎相同（相对差 {rel:.3e}），加权未参与计算")
    if abs(cls_with - cls_without) < 1e-9:
        raise RuntimeError("cls_loss 在加权前后相同，加权未参与计算")
    if not cls_with < cls_without:
        raise RuntimeError(
            f"加权后 cls_loss({cls_with:.6f}) 应小于未加权({cls_without:.6f})，方向异常"
        )
    return {
        "cls_loss_with_weights(other=0.7)": cls_with,
        "cls_loss_without_weights": cls_without,
        "ratio_with_over_without": cls_with / cls_without if cls_without else None,
        "cls_grad_rel_diff": rel,
        "same_batch_and_conditions": True,
        "applied": True,
    }


def make_weight_guards(expected: np.ndarray, batch_checks: int = 3, epoch_checks: int = 2):
    """训练过程中复核权重未被覆盖。

    注意：``criterion`` 由首次前向惰性构建，故 ``on_train_epoch_start`` 阶段可能尚不存在；
    此时只强制校验 ``model.class_weights``，一旦 criterion 出现则必须一致。
    """
    exp = torch.as_tensor(expected, dtype=torch.float32)
    state = {"batch": 0, "epoch": 0}

    def _check(trainer, tag: str) -> None:
        model = unwrap_model(trainer.model)
        mw = getattr(model, "class_weights", None)
        ok_model = mw is not None and torch.allclose(
            mw.detach().cpu().float().reshape(-1), exp, atol=1e-6
        )
        crit = _loss_instance(model)
        cw = None if crit is None else getattr(crit, "class_weights", None)
        ok_crit = cw is not None and torch.allclose(
            cw.detach().cpu().float().reshape(-1), exp, atol=1e-6
        )
        LOGGER.info(
            "[weight-guard] %s: model.class_weights 生效=%s, criterion.class_weights=%s",
            tag,
            ok_model,
            "未构建" if crit is None else f"生效={ok_crit}",
        )
        if not ok_model:
            raise RuntimeError("model.class_weights 在训练过程中被覆盖或丢失，训练终止")
        if crit is not None and not ok_crit:
            raise RuntimeError(f"{type(crit).__name__}.class_weights 在训练过程中被覆盖或丢失，训练终止")

    def on_batch(trainer):
        if state["batch"] >= batch_checks:
            return
        state["batch"] += 1
        _check(trainer, f"epoch {trainer.epoch} batch {state['batch']} end")

    def on_epoch_start(trainer):
        _check(trainer, f"epoch {trainer.epoch} start")

    def on_epoch_end(trainer):
        if state["epoch"] >= epoch_checks:
            return
        state["epoch"] += 1
        _check(trainer, f"epoch {trainer.epoch} end")

    return on_batch, on_epoch_start, on_epoch_end


# --------------------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------------------
def _build_overrides(project: str | None = None) -> dict:
    args = dict(TRAIN_ARGS)
    args["model"] = str(WEIGHTS)
    if project is not None:
        args["project"] = project
    return args


def run_verify_only() -> int:
    """启动前验证：使用真实 trainer 完整 setup，但不进入训练循环，产物落在临时目录。"""
    if not DATA_YAML.exists():
        raise SystemExit(f"数据配置不存在: {DATA_YAML}")
    if not WEIGHTS.exists():
        raise SystemExit(f"预训练权重不存在: {WEIGHTS}")

    weights, other_id, names = resolve_class_weights(DATA_YAML)
    LOGGER.info("[preflight] data.yaml = %s", DATA_YAML)
    LOGGER.info("[preflight] 类别 = %s", {k: names[k] for k in sorted(names)})
    LOGGER.info("[preflight] other 真实类别 ID = %d", other_id)
    LOGGER.info("[preflight] 期望权重向量 = %s", np.array2string(weights, precision=3))

    tmp_project = Path(tempfile.mkdtemp(prefix="stage1_preflight_"))
    trainer = None
    try:
        overrides = _build_overrides(project=str(tmp_project))
        overrides["epochs"] = 100
        trainer = CustomWeightedSegTrainer(overrides=overrides)
        trainer._setup_train()
        report = verify_weights(trainer, weights, other_id, run_grad_check=True)
    finally:
        # 释放 dataloader workers / GPU 内存，再清理临时目录
        if trainer is not None and getattr(trainer, "train_loader", None) is not None:
            close = getattr(trainer.train_loader, "close", None)
            if callable(close):
                close()
        shutil.rmtree(tmp_project, ignore_errors=True)

    print("\n================ PREFLIGHT RESULT ================")
    print(f"其它类别权重 (nc={len(weights)}): {report['model_class_weights']}")
    print(f"权重张量长度 == 类别总数      : {report['length_ok']}")
    print(f"other({other_id}) == 0.7        : {report['other_is_07']}")
    print(f"其余类别均 == 1.0             : {report['others_are_1']}")
    print(f"损失类                        : {report['criterion']}")
    print(f"v8SegmentationLoss 读取权重   : {report['criterion_reads_weights']}")
    print(f"criterion.class_weights       : {report['criterion_class_weights']}")
    print(f"BCE 逐通道加权比例            : {report['bce_per_channel_ratio']}")
    lin = report["linearity_check"]
    print("---- 真实数据闭式检查（同一 batch，仅切换 other 通道权重）----")
    print(f"  cls_loss  when other=1.0    : {lin['cls_loss_w_other_1.0']:.6f}")
    print(f"  cls_loss  when other=0.7    : {lin['cls_loss_w_other_0.7']:.6f}")
    print(f"  cls_loss  when other=0.0    : {lin['cls_loss_w_other_0.0']:.6f}")
    print(f"  比值 0.7/1.0                : {lin['ratio_0.7_over_1.0']:.6f}")
    print(f"  仿射式 0.7*L(1)+0.3*L(0)    : {lin['affine_prediction_0.7']:.6f}  (绝对偏差 {lin['affine_abs_error']:.3e})")
    print(f"  other 在 BCE 中的质量占比 s  : {lin['others_weight_share_of_bce']:.6f}")
    print(f"  1 - 0.3*s 预测比值           : {1 - (1 - OTHER_WEIGHT) * lin['others_weight_share_of_bce']:.6f}")
    print(f"  box/seg/dfl 绝对变化         : {lin['non_cls_loss_max_abs_change']}  (必须全为 0.0)")
    print(f"  非 cls 损失完全不受影响      : {lin['non_cls_losses_unaffected']}")
    print(f"  仿射关系成立                : {lin['affine_holds']}")
    gc = report["grad_check"]
    print("---- 真实梯度检查（cls 分支 cv3）----")
    print(f"  cls_loss 加权后 other=0.7   : {gc['cls_loss_with_weights(other=0.7)']:.6f}")
    print(f"  cls_loss 未加权             : {gc['cls_loss_without_weights']:.6f}")
    print(f"  比值 加权后/未加权           : {gc['ratio_with_over_without']:.6f}")
    print(f"  cls 梯度相对差              : {gc['cls_grad_rel_diff']:.6f}")
    print(f"  同一 batch 同一条件          : {gc['same_batch_and_conditions']}")
    print(f"  交叉自洽（None==全1 / 比值一致）: {report['cross_check']}")
    print(f"首个 batch loss               : {report['first_batch_loss']}")
    print("=================================================")
    print("PREFLIGHT: PASS —— other=0.7 / 其余=1.0 自定义分类损失权重确认生效")
    return 0


def run_training() -> int:
    if SAVE_DIR.exists():
        raise SystemExit(
            f"目标目录已存在，拒绝覆盖: {SAVE_DIR}\n"
            "请先确认是否存在同一任务在运行，或已有完整结果；新实验请使用独立 name。"
        )

    weights, other_id, names = resolve_class_weights(DATA_YAML)
    LOGGER.info("[train] other 真实类别 ID = %d，权重向量 = %s", other_id, np.array2string(weights, precision=3))
    LOGGER.info("[train] 类别 = %s", {k: names[k] for k in sorted(names)})

    yolo = YOLO(str(WEIGHTS))
    on_batch, on_epoch_start, on_epoch_end = make_weight_guards(weights)
    yolo.add_callback("on_train_batch_end", on_batch)
    yolo.add_callback("on_train_epoch_start", on_epoch_start)
    yolo.add_callback("on_train_epoch_end", on_epoch_end)
    yolo.train(trainer=CustomWeightedSegTrainer, **_build_overrides())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLOv8m-Seg Stage 1 (TACO v2) 训练入口")
    parser.add_argument("--verify-only", action="store_true", help="仅做启动前类别权重验证，不训练")
    args = parser.parse_args()
    print("训练参数:", {k: v for k, v in TRAIN_ARGS.items()})
    print("模型权重:", WEIGHTS)
    return run_verify_only() if args.verify_only else run_training()


if __name__ == "__main__":
    sys.exit(main())
