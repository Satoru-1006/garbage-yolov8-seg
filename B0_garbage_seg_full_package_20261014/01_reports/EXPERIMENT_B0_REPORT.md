# 实验 B0 报告：TACO → 室内垃圾 v2 微调（两阶段，YOLOv8m-Seg）

**实验名称**：`B0_TACO_INDOOR_V2_800_2STAGE`
**路线**：TACO 已训练底座 `taco_baseline_v2/train/weights/best.pt`（nc=9）→ 室内垃圾 `garbage_seg_merged_dedup_v2`
**训练日期**：2026-10-14
**输出目录**：`/home/xmv/garbage_yolov8_seg/indoor_ab/B0_taco_indoor_v2_800_2stage/`
**状态**：`TRAIN_STATUS = SUCCESS`（两阶段完成，产物完整，test 集全程未读取）
**数据质量**：**PROVISIONAL**（v2 未通过质量验收，见 §2.2）

---

## 0. 执行摘要（结论先行）

| 项目 | Phase 2 best（**正式交付**） | Phase 1 best（对照） |
|---|---|---|
| **Mask mAP50** | **0.2402** | 0.1852 |
| **Mask mAP50-95** | **0.1356** | 0.1124 |
| Mask Precision / Recall | 0.5450 / 0.2045 | 0.3459 / 0.2382 |
| Box mAP50 / mAP50-95 | 0.2446 / 0.1646 | 0.2021 / 0.1437 |
| fitness | 0.3002 | 0.2561 |
| 掩码召回@IoU0.5（340 GT） | **0.5588** | 0.6471 |
| 最优 epoch | Phase 1 **ep20**；Phase 2 **ep26** | — |
| 实际 epoch（请求） | 20 / **20** | 76 / 180（patience=50 早停） |
| 是否过拟合 | **轻度**（val/seg 由 2.46 升至 2.56，远轻于 A） | 否 |
| 峰值显存（本进程 alloc） | **16014.7 MiB** | — |
| 总耗时 | **15.0 min**（P1 2.9 + P2 12.1） | — |

**最终交付权重**：`indoor_ab/B0_taco_indoor_v2_800_2stage/phase2/weights/best.pt`
SHA256 `1e102051acdd840983240923e09438068d7ffa2f8e38a706cde1131c6e8e76d6`

**一句话结论**：B0 训练链路完全合规（唯一预训练来源为 TACO 底座，未触碰 COCO 权重、未随机初始化、未改架构、未改数据集、未覆盖模型 1/2），
两阶段均收敛，**Phase 2 全解冻带来明确增益**（Mask mAP50 0.1852→0.2402，mAP50-95 0.1124→0.1356），
且在**与实验 A 完全相同的 val 集**上，Mask mAP50（0.2402 vs 0.2126）与实例掩码召回（0.5588 vs 0.4912）均优于 A 对照。
但因 v2 数据为 **PROVISIONAL**、且 A/B0 **同时改变了起点权重与数据版本**，该优势**不能直接解读为"TACO 迁移优于 COCO 迁移"**（见 §6.3）。

---

## 1. 训练环境

| 项目 | 值 |
|---|---|
| Conda 环境 | `zxj-py312`（`/home/xmv/miniconda3/envs/zxj-py312`） |
| Python | 3.12.15 |
| PyTorch | 2.14.1+cu130 |
| torch.version.cuda | 13.0 |
| Ultralytics | **8.4.176** |
| GPU | NVIDIA GeForce RTX 5090，31.34 GiB，Compute Capability 12.0 |
| CUDA 可用 | True |
| AMP 自检 | PASS |
| 干扰进程 | 训练前已有 2 个无关 `python3` 进程占用约 9.5 GiB（峰值 nvidia-smi 29825 MiB 含此占用） |

> 复用既有环境，未重新安装任何依赖。

---

## 2. 数据

### 2.1 规模与类别（已独立核对）

| split | 图片 | 标签 | 实例数 |
|---|---|---|---|
| train | 845 | 845 | 2259 |
| val | 145 | 145 | 340 |
| test | 145 | 145 | 314（**本轮未读取**） |
| 合计 | 1135 | 1135 | 2913 |

类别（9 类，与 TACO 底座完全一致）：`bottle, box, can, cigarette_butt, food_waste, other, paper, plastic_bag, wrapper`

逐类实例（train/val/test）：
`bottle 57/9/15`、`box 376/41/63`、`can 6/0/0`、`cigarette_butt 71/11/13`、`food_waste 11/1/6`、
`other 700/118/87`、`paper 436/61/49`、`plastic_bag 463/87/68`、`wrapper 139/12/13`。

标签格式经抽样确认为 **YOLO 实例分割多边形**（字段数均为奇数），类别 ID ∈ [0,8]，无越界。

### 2.2 v2 数据质量 = **PROVISIONAL**（必须随结果一并报告）

依据 `tools/dedup_v2_audit.md`：结构层面完全达标（零 train 内 SHA 重复、零跨 split 内容泄漏、val/test 逐字节冻结、
零非法标注、35/36 项自检通过），但：

1. **仍有 637 / 1063 组标注未获人工确认**（train 514 组），构成系统性标签噪声。
2. **val/test 冻结集中含 14 组已知假阴性**（U2：空标签 vs 有标注，因冻结未修正）→ 指标偏保守。
3. **`can` 在 val/test 实例数为 0 → 该类不可评估**（val/test 各 0；train 仅 6）。
4. **会话级划分不独立**（弱形式泄漏）：val/test 距最近 train 图像采集时间中位 20.9 s / 24.7 s，
   <60 s 占比约 72% → val/test 指标为**乐观上界**，不可外推为新场景性能。
5. `food_waste` val 仅 1 个实例 → 该类 AP 极不稳定，不作决策依据。

> **结论**：本轮所有指标应视为 **PROVISIONAL 口径**，与 A 的对照结论需在 v2 数据通过人工验收后复核。

### 2.3 与实验 A 的数据差异（影响可解释性）

| | 实验 A | B0 |
|---|---|---|
| 数据集 | `garbage_seg_merged_full1444` | `garbage_seg_merged_dedup_v2` |
| train | 1154 张（含 309 个重复副本） | **845 张（精确去重后）** |
| val / test | 145 / 145（与 B0 **逐字节相同**） | 145 / 145 |
| 起点权重 | COCO `yolov8m-seg.pt` | **TACO `taco_baseline_v2/.../best.pt`** |

> val/test 完全一致 → **val 上的对比公平**；但 train 集合不同，A/B0 差异是「起点权重 × 数据版本」的混合效应。

---

## 3. 训练配置（与实验 A 超参数完全一致，仅替换起点权重与数据集）

| 参数 | Phase 1 | Phase 2 |
|---|---|---|
| 起点权重 | `taco_baseline_v2/train/weights/best.pt`（SHA256 `bfcf3d35…03e2`） | `phase1/weights/best.pt`（SHA256 `f88f420a…82e1`） |
| epochs（请求/实际） | 20 / **20** | 180 / **76**（patience=50 早停，best=ep26） |
| freeze | **10**（`model.0`–`model.9` + `.dfl`） | **0** |
| optimizer | AdamW | AdamW |
| lr0 / lrf | 1e-3 / 0.01 | 5e-4 / 0.05 |
| cos_lr | False | **True** |
| warmup_epochs | 3.0 | 2.0 |
| patience / close_mosaic | 20 / 0 | 50 / 20 |
| imgsz / batch | 800 / 16（nbs=64 → 等效批 64） | 800 / 16（等效 64） |
| weight_decay / momentum | 5e-4 / 0.937 | 5e-4 / 0.937 |
| seed / deterministic / amp | 42 / True / True | 42 / True / True |
| workers / device | 8 / 0 | 8 / 0 |

**增强（两阶段一致）**：`hsv_h=0.015, hsv_s=0.35, hsv_v=0.30, degrees=8, translate=0.10, scale=0.25,
fliplr=0.5, flipud=0.0, mosaic=0.30, mixup=0.0, copy_paste=0.10, multi_scale=0.10,
perspective=0.0, mask_ratio=2, overlap_mask=True`

**冻结核验（实测）**：Phase 1 冻结层索引 `[0..9]`（含 SPPF）+ `.dfl`，冻结参数 **11,855,872**，
可训练 **15,388,987（56.48%）**；Phase 2 全解冻。完整冻结张量清单见 `smoke/frozen_parameters.txt`（118 个张量）。

**合规性声明**：未使用 COCO 权重、未使用 `yolov8m-seg.yaml` 随机初始化、未更换架构、
**未修改数据集**（训练前后数据集快照哈希一致）、未覆盖模型 1（`A_coco_direct_800_2stage`）与模型 2（`taco_baseline_v2`）
—— 训练前后受保护目录快照哈希**逐字节一致**（`protected_dirs_unchanged = true`）。
`label_smoothing` 在该版本已移除 → `NOT_APPLIED`（未使用无效参数）。

---

## 4. 冒烟测试（训练前强制，产物 `smoke/smoke_result.json`）

| 检查项 | 结果 |
|---|---|
| TACO 底座加载与类别核对 | PASS（`nc=9`，9 类名与 v2 顺序完全一致，**无需替换分割头**） |
| 数据集规模核对 | PASS（845 / 145 / 145） |
| 800 分辨率前向 + 反向 | PASS |
| **batch=16 显存** | **PASS，峰值 9757.3 MiB**（未 OOM，无需降级到 8/4） |
| 实例分割损失有限性 | PASS，`{box 1.5303, seg 4.1250, cls 5.4511, dfl 1.4831}`，无 NaN/Inf |
| 所有类别 ID 合法 | PASS（最大类别 ID = 8） |
| 增强后掩码与实例对齐 | PASS（16 抽样中 15 完全一致；1 例因 `mask_ratio=2` 下采样吞掉 1 个极小实例，22→21，属已知管线现象） |
| 掩码 bbox vs 标注 bbox 平均 IoU | 0.8325 |
| `mask_ratio=2` 生效 | PASS（masks `(1,400,400)`，下采样比 2.0） |
| `copy_paste` / `mosaic` 已装配 | PASS |
| label_smoothing | NOT_APPLIED（该版本已移除） |

---

## 5. 训练结果

### 5.1 Phase 1（冻结骨干，20 epoch，lr0=1e-3）

| epoch | train/box | train/seg | val/box | val/seg | Mask mAP50 | Mask mAP50-95 |
|---|---|---|---|---|---|---|
| 1 | 1.2256 | 2.4192 | 1.3452 | 4.5277 | 0.1044 | 0.0538 |
| 5 | 1.0431 | 1.9409 | 1.2608 | 2.4644 | 0.1674 | 0.1017 |
| 10 | 0.9157 | 1.6301 | 1.2165 | 2.5120 | 0.1830 | 0.1128 |
| **20（best）** | **0.7499** | **1.3106** | 1.1752 | 2.4547 | **0.1882** | **0.1234** |

→ 训练损失单调下降、验证指标同步提升；val/seg 波动很小（最低 2.4021@ep7 → 2.4547@ep20），**未过拟合**，
ep20 仍在改善（best = ep20，与 A 一致）。磁盘复评：mask mAP50 0.1852 / mAP50-95 0.1124。

### 5.2 Phase 2（全解冻，实际 76 epoch，best=ep26）

| epoch | train/box | train/seg | val/box | val/seg | 泛化间隙(seg) | Mask mAP50 | Mask mAP50-95 |
|---|---|---|---|---|---|---|---|
| 1 | 0.9685 | 1.7353 | 1.4590 | 4.4109 | — | 0.0930 | 0.0517 |
| 5 | 0.8272 | 1.4462 | 1.2072 | 2.5375 | 1.09 | 0.1555 | 0.0945 |
| 10 | 0.8067 | 1.3188 | 1.1991 | 2.8721 | 1.55 | 0.1626 | 0.1025 |
| **26（best）** | 0.7027 | 1.1334 | 1.3295 | 2.8697 | 1.74 | **0.2389** | **0.1394** |
| 50 | 0.5852 | 0.9356 | 1.3132 | 2.9417 | 2.01 | 0.1873 | 0.1051 |
| 76（last） | **0.5219** | **0.8097** | 1.1154 | 2.5648 | 1.76 | 0.1787 | 0.1108 |

> 表中 mAP 为训练期验证值；**正式交付指标以磁盘 `best.pt` 的独立复评为准**（见 §5.4，与训练日志存在 strip 前后差异，与 A 同因）。

### 5.3 过拟合判断 —— **轻度，远轻于实验 A**

- train/seg 从 1.7353 降至 0.8097（−53%），val/seg 在 **2.4576（ep37 最低）** 附近波动，末期 2.5648（+0.107）。
- 泛化间隙（val_seg − train_seg）从 1.09 扩大到 1.74–2.01，但**未出现 A 那种"val 损失单调恶化"**。
- 对比 A：A 的 val/seg 由 ep5 最低 2.4825 单调升至 **3.0402**，间隙扩大到 **2.43**；
  B0 的 val/seg 最低 2.4576、末期 2.5648，间隙 1.76。→ **B0 的过拟合显著更轻**。
- 仍存在的证据：train 损失持续下降而 val mAP 在 ep26 后不再提升（早停于 76，best=ep26）。
- 推测原因：v2 精确去重后**重复副本放大效应消失**（A 的 train 含 309 个重复副本），等效数据多样性提高。

### 5.4 正式评估（**仅验证集**，test 集锁定未使用）

**评估协议（Phase 1 / Phase 2 完全一致）**：`imgsz=800, split=val, conf=0.001, iou=0.7, batch=32, rect=True, max_det=300`
val 集：145 张图 / 340 个实例（与实验 A 逐字节相同）。

| 指标 | **Phase 2 best（交付）** | Phase 1 best（对照） |
|---|---|---|
| Mask Precision | **0.5450** | 0.3459 |
| Mask Recall | 0.2045 | 0.2382 |
| **Mask mAP50** | **0.2402** | 0.1852 |
| **Mask mAP50-95** | **0.1356** | 0.1124 |
| Box Precision | 0.4804 | 0.3654 |
| Box Recall | 0.2263 | 0.2673 |
| Box mAP50 | 0.2446 | 0.2021 |
| Box mAP50-95 | 0.1646 | 0.1437 |
| fitness | 0.3002 | 0.2561 |
| 推理速度 | 3.71 ms/img | 3.66 ms/img |
| 掩码宏平均（排除 `can`，8 类） | 0.2402 / 0.1356 | 0.1852 / 0.1124 |

> 因 val 中 `can` 实例数为 0，上表总体值为 **8 类有效宏平均**（`can` 天然不贡献）。

### 5.5 逐类 Mask AP（Phase 2 best）

| 类别 | val 实例 | Mask P | Mask R | Mask AP50 | Mask AP50-95 |
|---|---|---|---|---|---|
| bottle | 9 | 0.9909 | 0.2222 | **0.2432** | 0.1088 |
| box | 41 | 0.0676 | 0.0244 | **0.0273** | 0.0083 |
| **can** | **0** | — | — | **不可评估** | **不可评估** |
| cigarette_butt | 11 | 0.4460 | 0.1818 | 0.2562 | 0.0957 |
| food_waste | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| other | 118 | 0.6177 | 0.1017 | 0.2368 | 0.0930 |
| paper | 61 | 0.4560 | 0.3934 | **0.4086** | **0.2915** |
| plastic_bag | 87 | 0.6111 | 0.3793 | **0.4277** | **0.2806** |
| wrapper | 12 | 0.1710 | 0.3333 | 0.3218 | 0.2069 |

**Phase 1 → Phase 2 逐类 AP50 变化**：
`bottle 0.0125→0.2432`、`box 0.0396→0.0273`、`cigarette_butt 0.2827→0.2562`、`food_waste 0→0`、
`other 0.2089→0.2368`、`paper 0.3607→0.4086`、`plastic_bag 0.3471→0.4277`、`wrapper 0.2302→0.3218`。

→ **Phase 2 全解冻整体增益为正**（8 类中 5 类改善），最显著的是 `bottle` 从"完全失效"恢复到 AP50 0.2432；
退化主要发生在 `box`（0.0396→0.0273）与 `cigarette_butt`。

### 5.6 小目标识别效果（尺寸分层，掩码 IoU ≥ 0.5 匹配）

**全部 340 个 val GT 实例的最佳掩码 IoU 统计**：

| | Phase 2 best（交付） | Phase 1 best（对照） |
|---|---|---|
| mean best mask IoU | 0.4816 | **0.5544** |
| Recall@IoU0.5 | 0.5588 | **0.6471** |
| Recall@IoU0.75 | **0.3882** | 0.4765 |

**按目标尺寸分桶召回率**：

| 等效边长(@800) | GT 数 | Phase 2 召回 | Phase 1 召回 |
|---|---|---|---|
| **0–16 px** | 98 | 0.5510 | **0.6633** |
| **16–32 px** | 69 | 0.5942 | 0.6522 |
| 32–64 px | 77 | 0.5844 | 0.6623 |
| 64–128 px | 65 | 0.5692 | 0.6154 |
| 128+ px | 31 | 0.4194 | 0.6129 |

**解读**：
1. 模型确实学到极小目标：0–16 px 桶召回 0.5510（98 个 GT 命中 54 个），`imgsz=800` 必要性得到验证。
2. 与 A 的 Phase 2 相比，B0 Phase 2 在**所有尺寸桶召回均更高**（0–16 px：0.5510 vs 0.4898；整体 0.5588 vs 0.4912）。
3. 但 B0 的 Phase 2 仍**削弱了实例召回**（0.6471 → 0.5588），与 A 的结论（全解冻牺牲检出能力）**方向一致**；
   差别在于 B0 的削弱幅度更小（−8.8 pt）且 mAP 净收益为正（A 为净收益≈0）。

### 5.7 资源占用与吞吐

| 项目 | 值 |
|---|---|
| 峰值显存（本进程 allocated） | **16,014.7 MiB** |
| 峰值显存（本进程 reserved） | 19,370.0 MiB |
| 系统级 nvidia-smi 峰值 | 29,825 MiB（含训练前已存在的无关进程占用） |
| Phase 1 | 174.2 s / 20 epoch（8.7 s/epoch，≈97 img/s） |
| Phase 2 | 726.0 s / 76 epoch（9.6 s/epoch，≈88 img/s） |
| **总训练耗时** | **900.2 s = 15.0 min**（96 epoch） |

> batch=16 在 800 分辨率下**未出现 OOM**，无需降级到 8 或 4；未通过降分辨率或修改数据回避显存问题。

---

## 6. 与实验 A 的对照（仅供参考，非纯消融）

### 6.1 同 val 集（145 图 / 340 实例，逐字节相同）Mask 指标对比

| 指标 | A（COCO→full1444）Phase 2 best | **B0（TACO→dedup_v2）Phase 2 best** | 差值 |
|---|---|---|---|
| Mask mAP50 | 0.2126 | **0.2402** | **+0.0276** |
| Mask mAP50-95 | 0.1378 | 0.1356 | −0.0022 |
| Mask Precision | 0.3575 | **0.5450** | +0.1875 |
| Mask Recall | **0.2594** | 0.2045 | −0.0549 |
| Box mAP50 / mAP50-95 | 0.2302 / 0.1713 | 0.2446 / 0.1646 | +0.0144 / −0.0067 |
| 掩码召回@IoU0.5 | 0.4912 | **0.5588** | +0.0676 |
| mean best mask IoU | 0.4271 | **0.4816** | +0.0545 |
| 掩码召回@IoU0.75 | 0.3735 | **0.3882** | +0.0147 |
| 0–16 px 桶召回 | 0.4898 | **0.5510** | +0.0612 |
| 总耗时 | 25.25 min（122 ep） | **15.0 min（96 ep）** | −10.25 min |

**逐类 Mask AP50（A → B0）**：
`bottle 0.0000→0.2432`、`box 0.0495→0.0273`、`cigarette_butt 0.2232→0.2562`、`food_waste 0.0711→0.0000`、
`other 0.2041→0.2368`、`paper 0.4499→0.4086`、`plastic_bag 0.3985→0.4277`、`wrapper 0.3047→0.3218`。

### 6.2 结论

- B0 在 **Mask mAP50（+0.028）与实例级掩码召回（+0.068）上明确优于 A**，mAP50-95 基本打平（−0.002）。
- B0 的 Precision 大幅提升（0.358→0.545）而 Recall 下降（0.259→0.205）→ **B0 倾向于"更少但更准"的预测**；
  但按实例匹配口径（掩码 IoU≥0.5）B0 召回反而更高，说明其**排序质量更好**。
- B0 **训练更快（15.0 vs 25.25 min）且过拟合更轻**。

### 6.3 ⚠️ 必须随对照结论报告的限制

1. A 与 B0 同时改变了两项变量（**起点权重** COCO→TACO、**数据版本** full1444→dedup_v2），
   因此上述差异是**混合效应**，**不能**归因为"TACO 底座优于 COCO 底座"或"去重数据优于含重复数据"中的任一项。
   若需分离，须补充 **B1（TACO + full1444）** 与 **B2（COCO + dedup_v2）** 两组消融（本轮按任务要求未执行）。
2. v2 数据为 **PROVISIONAL**（§2.2），val/test 含已知假阴性且会话级不独立。
3. val 仅 145 图 / 340 实例，**mAP50-95 差值 < 0.01 不应判为显著**；mAP50 差值 0.028 亦偏小，建议多 seed 复核。
4. `can` 在 A/B0 中均不可评估；`food_waste` val 仅 1 实例，不作决策依据。

---

## 7. 产物清单与可复现信息

```
indoor_ab/B0_taco_indoor_v2_800_2stage/
├── EXPERIMENT_B0_REPORT.md              # 本报告
├── training_summary.json                # 两阶段完整配置/哈希/耗时/显存/受保护目录校验
├── train_full.log                       # 完整训练日志
├── eval_full.log                        # 两次评估日志
├── eval_phase2_best.json                # 正式评估（含尺寸分层、逐类 AP）
├── eval_phase1_best.json                # 对照评估
├── eval_phase2_best_val/ , eval_phase1_best_val/   # 评估绘图产物
├── smoke/
│   ├── smoke_result.json                # 冒烟测试结果
│   ├── frozen_parameters.txt            # freeze=10 完整冻结张量清单（118 个）
│   └── augmented_samples.jpg            # 增强可视化样张
├── phase1/                              # args.yaml, results.csv, results.png, 混淆矩阵,
│   └── weights/{best.pt,last.pt}        # 各类曲线, val_batch 预测
└── phase2/                              # 同上（**正式交付**）
    └── weights/{best.pt,last.pt}
```

### 版本与哈希（可复现性锚点）

| 项目 | 值 |
|---|---|
| **训练代码** | `tools/train_B0_2stage.py` SHA256 `3336197685f694da802f6b748b742d8715a42d6864301aa5080fa37972ed320a` |
| 冒烟代码 | `tools/smoke_B0.py` SHA256 `82f0b10405c2d46106e049f86fbc170df038c9f169a59410b916b2c27b976081` |
| 评估代码 | `tools/eval_B0.py` SHA256 `51a580c151ffe84be09e87d140fba78372d610fbb13e033c97bb66411750f650` |
| **输入数据（data.yaml）** | SHA256 `97205a9080ad0fa1d8f9373bab2956fca9122fa4e0c79b6f6d7d0b4e6cccf61f` |
| **输入数据（标签清单）** | 1135 个文件清单 SHA256 `e89fa7f35bae893876e454cde5ed738dea2117cc58147ae3845d93ca5e0cb8a2` |
| **起点权重（TACO 底座）** | SHA256 `bfcf3d357346e82e75bb554d16964e7dd786c2886fd8d064d382dd3a4ee703e2` |
| **Phase 1 best.pt** | SHA256 `f88f420a54f82023513159ef0402391f5a451c9ec208390efb7489895fde82e1` |
| **Phase 2 best.pt（最终交付）** | SHA256 `1e102051acdd840983240923e09438068d7ffa2f8e38a706cde1131c6e8e76d6` |
| Phase 2 last.pt | SHA256 `8a01dca60a082eef4806a9e766678c5e71f3a05868afc57cc8056f7e8931ae33` |
| 受保护目录（未改动） | `protected_dirs_unchanged = true`（A 与 TACO 基线目录快照哈希前后一致） |

### 复现命令

```bash
cd /home/xmv/garbage_yolov8_seg
source /home/xmv/miniconda3/etc/profile.d/conda.sh && conda activate zxj-py312

python tools/smoke_B0.py          # GPU/增强/显存冒烟
python tools/train_B0_2stage.py   # 两阶段训练（约 15 min）
python tools/eval_B0.py --weights indoor_ab/B0_taco_indoor_v2_800_2stage/phase2/weights/best.pt \
  --out indoor_ab/B0_taco_indoor_v2_800_2stage/eval_phase2_best.json --imgsz 800 --rect --batch 32 --size-analysis
python tools/eval_B0.py --weights indoor_ab/B0_taco_indoor_v2_800_2stage/phase1/weights/best.pt \
  --out indoor_ab/B0_taco_indoor_v2_800_2stage/eval_phase1_best.json --imgsz 800 --rect --batch 32 --size-analysis
```

> 训练脚本在 `phase1/`、`phase2/` 已存在时会直接报错退出（拒绝自动编号），保证不会覆盖既有产物。

---

## 8. 最终判定与下一步建议

### 8.1 判定

- **训练链路合规**：唯一预训练来源为 TACO 底座，未使用 COCO 权重、未随机初始化、未改架构、未改数据集；
  test 集全程未读取；模型 1 / 模型 2 产物未被覆盖（哈希与 mtime 一致）。
- **模型可用**：Phase 2 best 在 val 上 Mask mAP50 **0.2402**、mAP50-95 **0.1356**，掩码召回@0.5 **0.5588**。
- **Phase 2 增益为正**：全程全解冻相比冻结骨干在 mAP50（+0.055）、mAP50-95（+0.023）上明确改善，且过拟合轻度可控。
- **数据质量为唯一主要风险**：v2 仍为 **PROVISIONAL**，所有指标应标注该口径。

### 8.2 下一步建议（按优先级）

1. **补做 B1 / B2 消融以分离变量**（本轮未执行）：
   `B1 = TACO + full1444`、`B2 = COCO + dedup_v2`。只有这样才能回答"TACO 底座 vs COCO 底座"与"去重 vs 含重复"各自的贡献。
   （注意：B1 需使用 `full1444`，与 A 完全同数据、仅起点不同 → 这是**最干净的迁移增益消融**。）
2. **v2 数据人工验收**：优先处理 U2 的 val/test 14 组已知假阴性（冻结集中）、U1 的 HIGH 141 组、U4 的 344 组 A 独有内容。
   验收完成后再复核本报告结论。
3. **缓解 Phase 2 召回下降**：B0 与 A 均显示全解冻会削弱实例召回。
   建议消融 `patience=15–20`、`mosaic 0.3→0.5`、`copy_paste 0.1→0.3`，或采用"Phase 1 权重 + 更低 lr 的短程解冻"策略。
4. **多 seed 稳健性**：val 仅 340 实例，建议对 A / B0（及后续 B1/B2）各跑 3 个 seed，报告均值±标准差，
   避免把 <0.01 的 mAP50-95 差异当作迁移收益。
5. **`can` 类可评估性**：需补采 `can` 样本并重建划分（属新数据版本 v3），会破坏当前可比性，应作为独立议题。
6. **权重选型建议**：若以 mAP 为唯一指标 → 采用 **Phase 2 best**（协议规定交付）；
   若更看重实际检出能力（掩码召回 0.6471 vs 0.5588），则 **Phase 1 best** 更优。
   该取舍**必须与 A/B1/B2 采用完全相同的标准**，否则对照不公平。
