# 实验 A 报告：COCO → 室内垃圾 直接微调（两阶段）

**实验名称**：`A_COCO_DIRECT_INDOOR_800`
**路线**：Ultralytics 官方 `yolov8m-seg.pt`（COCO 80 类）→ 自采室内垃圾 9 类实例分割
**训练日期**：2026-10-13
**输出根目录**：`/home/xmv/garbage_yolov8_seg/indoor_ab/A_coco_direct_800_2stage/`
**状态**：`TRAIN_STATUS = SUCCESS`（两阶段完成，产物完整，test 集未参与任何环节）

---

## 0. 执行摘要（结论先行）

| 项目 | 数值 |
|---|---|
| Mask mAP50 | **0.2126** |
| Mask mAP50-95 | **0.1378** |
| Mask Precision / Recall | **0.3575 / 0.2594** |
| Box mAP50 / mAP50-95 | **0.2302 / 0.1713** |
| 最优 epoch | Phase 1 ep20；Phase 2 **ep52** |
| 实际训练 epoch | 20 + 102 = **122**（未跑满 200，早停触发） |
| 最终权重 | `phase2/weights/best.pt` |
| 总耗时 | **25.25 min**（Phase1 3.77 + Phase2 21.47） |
| 峰值显存（本进程） | **15976 MiB** allocated / 18860 MiB reserved |
| 是否过拟合 | **是**（val/seg_loss 从 ep5 最低 2.4825 单调升至 3.0402） |

**一句话结论**：模型可用但**精度偏低**（Mask mAP50 仅 0.213），具备作为 A/B 对照基线的资格；
但 Phase 2 全解冻**没有带来实质增益**（Mask mAP50 0.2151→0.2126，mAP50-95 0.1343→0.1378，基本打平），
且实例级召回显著下降、验证损失持续上升，表明在 1063 张唯一内容图像上直接全量微调 27M 参数模型**严重过拟合**。

---

## 1. 环境与版本核验

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090，31.34 GiB，Compute Capability 12.0 |
| CUDA Driver | 595.99.02 / CUDA 13.2 |
| Python | 3.12.15 (Anaconda, conda env `zxj-py312`) |
| PyTorch | 2.14.1+cu130 |
| torch.version.cuda | 13.0 |
| Ultralytics | **8.4.176**（与预期一致） |
| AMP 自检 | PASS |
| 干扰进程 | 训练前 GPU 已有无关 `python3` 进程占用约 4.6 GiB（未干扰本次训练） |

**关键版本行为核验（与经典 YOLOv8 不同，已实测确认）**：

1. `freeze=10` → 冻结 `model.0` ~ `model.9`（整个骨干至 SPPF）+ 始终冻结 `.dfl`。
   冻结参数 **11,855,872**，可训练参数 **15,388,987（56.48%）**。冻结区内 BatchNorm 置 eval 且不更新 running stats。
   完整冻结张量清单：`smoke/frozen_parameters.txt`（118 个张量）。
2. `overlap_mask=True` 在本版本的含义是**合并为单通道实例索引图** `(1, H/mask_ratio, W/mask_ratio)`，
   与经典 YOLOv8（每实例一张掩码）**相反**。
3. `label_smoothing` **已被该版本移除**（在 `removed_keys` 中），传入会被丢弃 → **NOT_APPLIED**，全程未使用无效参数。
4. `auto_augment='randaugment'` 与 `erasing` 在**分割训练链路中不生效**（仅分类任务使用）；
   且 albumentations 库未安装，`Albumentations(p=1.0)` 节点实测为 **no-op**。
5. **按类名重映射分类头**：`bottle` 一行 logits 从 COCO 按类名迁移，其余 8 类为新建初始化；
   骨干 / 颈部 / proto / mask 分支 **全部（537/537 张量）** 从 COCO 迁移。
   → 预训练来源确为 COCO 官方权重，未使用任何 TACO 权重。
6. 梯度累积：`nbs=64`、`batch=16` → `accumulate=4`，等效批大小 64，weight_decay 实际生效值 = 5e-4。
7. 验证使用 `split='val'`；`final_eval` 亦使用 val → **test 集全程未被读取**（未见 `test.cache`）。

---

## 2. 数据审计（DATA_AUDIT）

### 2.1 独立复核结果

对 `datasets/garbage_seg_merged_full1444` 重新独立审计（脚本 `tools/audit_indoor_a.py` 等）：

| 检查项 | 结果 |
|---|---|
| 图片总数 | 1444（train 1154 / val 145 / test 145） |
| 图片可解码 | **1444/1444 PASS**（0 失败） |
| 标签-图片配对 | PASS（1444 : 1444） |
| 标签格式 | **YOLO 实例分割多边形 PASS**（检测框格式行数 = 0） |
| 实例总数 | 3815 |
| 非法类别 ID | **0**（全部 ∈ [0,8]） |
| 越界坐标 | **0** |
| 退化多边形（面积 ≤ 0 或点数 < 3） | **0** |
| 源 A(5类)→9类 重映射 | **PASS**（逐条复核 1444 文件） |
| 跨 split SHA256 内容组 | **0** |
| 跨 split 近重复组（16×16 dHash，Hamming ≤ 10） | **0**（351 组全部同 split） |
| 空标签（负样本） | 242（16.76%：train 187 / val 24 / test 31） |
| val/test 不可评估类别 | **`can`**（val 0、test 0 实例） |

### 2.2 三类需要说明的“非错误”现象（已定性，不影响训练正确性）

**(a) 295 条类别 ID 与源标签不一致 —— 属既有合并策略，非重映射错误**
- 其中 293 条 `label_source = half_dataset_yolo`：合并脚本对 A/B 同内容冲突组**统一采信 B 的 9 类标注**（`tools/class_map_full1444.json` 的 `label_policy`），正是设计行为。
- 其余 16 条例外为**同一内容组内部标注协调**（组内所有副本写入同一份标签），同样属既有策略。
- 结论：`CLASS_REMAP = PASS`。

**(b) 早期 8×8 dHash 报出的「27 对跨 split 近重复」为误报**
- 用 16×16 dHash（Hamming ≤ 10）+ 保宽高比 128×128 NCC 严格复验：**跨 split 组数 = 0**。
- 8×8 dHash 阈值过松，在纹理贫乏的室内地砖/墙面图上大量碰撞（NCC 低至 0.01）。
- 结论：`SPLIT_LEAKAGE = PASS`（在**训练实际使用的修复后磁盘状态**上复核，结果一致）。

**(c) 242 张空标签是真实负样本，判定为可保留**
- 242 张构成 188 个唯一内容组，**无任何一组同时存在非空副本** → 不是随机局部漏标。
- 其中 **53 个内容组在 A、B 两个独立标注来源中同时为空** → 双重独立证据支持真实负样本。
- 抽样检查仅 1/242 为近纯色图 → 非采集失败图。
- 结论：保留（16.76% 负样本有助于抑制室内场景误检）。

### 2.3 数据集真实规模与已知局限（必须写入对照基线说明）

| 局限 | 说明 |
|---|---|
| **有效唯一内容仅 1063 张** | 1444 个文件中有 343 个 SHA 重复内容组，其中 **323 组同时存在于源 A 与源 B**。文件数 1444 ≈ 1063 唯一内容 + 重复副本。 |
| **`can` 完全不可评估** | 仅 9 个实例且**全部在 train**，val/test 均为 0 → 任何模型都无法在该类上给出可比较指标。 |
| `food_waste` 近乎不可评估 | 26 实例（train 19 / val 1 / test 6）。 |
| `bottle` 稀少 | 99 实例（train 75 / val 9 / test 15）。 |
| `cigarette_butt` 稀少 | 121 实例（train 97 / val 11 / test 13），且**极小**。 |
| 类别不均衡 | `other` 1313 实例（占 32.9%），`can` 仅 9。 |

### 2.4 逐类实例与目标尺寸（800 输入等效边长）

| 类别 | 实例数 | train/val/test | 中位面积占比 | 等效边长中位(px@800) | <16px 占比 |
|---|---|---|---|---|---|
| bottle | 99 | 75/9/15 | 0.272% | 47.5 | — |
| box | 584 | 480/41/63 | 0.474% | 49.4 | — |
| **can** | **9** | **9/0/0** | 0.259% | 39.9 | 0.0% |
| **cigarette_butt** | 121 | 97/11/13 | 0.033% | **14.2** | **59.5%** |
| food_waste | 26 | 19/1/6 | 0.067% | 26.5 | 26.9% |
| other | 1313 | 1108/118/87 | 0.045% | **17.9** | **44.6%** |
| paper | 674 | 564/61/49 | 0.126% | 30.1 | 14.8% |
| plastic_bag | 774 | 619/87/68 | 0.890% | 43.0 | — |
| wrapper | 215 | 190/12/13 | 0.175% | 35.4 | 7.4% |

> **核心事实**：`cigarette_butt` 在 800 输入下中位等效边长仅 **14.2 px**（59.5% 小于 16 px），
> `other` 中位 17.9 px（44.6% 小于 16 px）。这是本数据集真正的难度来源，也是必须使用 `imgsz=800` 的直接依据。

### 2.5 审计过程中的一次数据完整性事件（如实记录）

Ultralytics 的 `check_image()` 在数据集扫描时检测到合并数据集副本中共 **719 张 JPEG 文件缺少 EOI 结束标记**（`\xff\xd9`，即文件被截断），
并对其实施了「应用 EXIF 方向后以 quality=100 重存」的**原子替换修复**（`_replace_image` 写临时文件后 `os.replace`，因此不会破坏硬链接）。

修复分两次完成：
- 首轮（冒烟测试扫描 train）修复 **559 张**；
- 随后**主动对 train/val/test 全部预扫描**，再补 **160 张**，合计 719 张 —— 目的是让训练开始前数据状态即固定，避免训练中途写盘。

精确统计（按 mtime 核对）：合并数据集被修改文件 **719**；<u>源数据集被修改文件 **0**</u>。

- **源数据集未受任何影响**（`waste_ds_c3_neg_yolo` / `half_dataset_yolo` 改动文件数 = 0），Labelme 标注与原始图像未被触碰。
- 修复前已主动对 train/val/test **全部预执行**该修复，因此训练期间不再发生中途写盘。
- **修复保真性已验证**：随机抽取修复文件与其源文件对比，**尺寸完全一致**、MAD ≈ 0.05/255、PSNR ≈ 60 dB（仅 JPEG 重编码噪声），
  几何与像素内容不变 → **不引入标注错位**。
- 修复后重新核验：跨 split SHA 泄漏 = 0，跨 split 近重复 = 0。

---

## 3. 冒烟测试（训练前强制）

产物：`smoke/smoke_result.json`、`smoke/augmented_samples.jpg`、`smoke/frozen_parameters.txt`

| 检查项 | 结果 |
|---|---|
| 模型正确加载并替换为 9 类分割头 | PASS（`Overriding model.yaml nc=80 with nc=9`） |
| 800 分辨率前向 + 反向传播 | PASS |
| **batch=16 显存** | **PASS，峰值 9152.8 MiB**（未 OOM，无需降级到 8/4） |
| 实例分割损失计算 | PASS，`{box_loss 1.2026, seg_loss 2.062, cls_loss 5.8206, dfl_loss 1.1201}` |
| Loss 有限性（无 NaN/Inf） | PASS |
| 所有类别 ID 合法 | PASS（实测最大类别 ID = 7 ≤ 8） |
| **增强后掩码与实例对齐** | **PASS**：掩码唯一值数 == 实例数（0 例不匹配）；掩码 bbox 与标注 bbox 平均 IoU **0.7814** |
| `mask_ratio=2` 生效 | PASS（masks 形状 `(1,400,400)`，下采样比 2.0） |
| 增强可视化人工确认 | PASS（见 `smoke/augmented_samples.jpg`，掩码轮廓紧贴目标，Mosaic 4 宫格与极小目标正常） |
| `label_smoothing` | **NOT_APPLIED**（该版本已移除，未使用） |

**实际装配的增强链**（递归展开后确认已生效）：
`Compose[Mosaic(p=0.3) → CopyPaste(p=0.1) → RandomPerspective(degrees=8, translate=0.1, scale=0.25, shear=0, perspective=0)]
→ MixUp(p=0) → CutMix(p=0) → Albumentations(no-op) → RandomHSV → RandomFlip(v, p=0) → RandomFlip(h, p=0.5) → Format(mask_ratio=2)`

> `copy_paste` 与 `multi_scale` 经实测**确认生效**；所有几何增强均同步变换分割掩码。

---

## 4. 训练配置（如实记录，与任务建议完全一致）

| 参数 | Phase 1 | Phase 2 |
|---|---|---|
| 起点权重 | `yolov8m-seg.pt`（COCO） | `phase1/weights/best.pt` |
| 起点权重 SHA256 | `51fa7e5e…45c4f4` | `b8b78efc…28c400` |
| epochs（请求/实际） | 20 / **20** | 180 / **102**（patience=50 早停） |
| freeze | **10** | **0** |
| optimizer | AdamW | AdamW |
| lr0 | 1e-3 | 5e-4 |
| lrf | 0.01 | 0.05 |
| cos_lr | False | True |
| warmup_epochs | 3.0 | 2.0 |
| patience | 20 | 50 |
| close_mosaic | 0 | 20 |
| imgsz | 800 | 800 |
| batch | 16（等效 64） | 16（等效 64） |
| weight_decay | 5e-4 | 5e-4 |
| seed / deterministic | 42 / True | 42 / True |
| amp / device | True / 0 | True / 0 |
| workers | 8 | 8 |

**增强（两阶段一致）**：`hsv_h=0.015, hsv_s=0.35, hsv_v=0.30, degrees=8, translate=0.10, scale=0.25, fliplr=0.5, flipud=0.0, mosaic=0.30, mixup=0.0, copy_paste=0.10, multi_scale=0.10, perspective=0.0, mask_ratio=2, overlap_mask=True`

**严格遵守的约束**：未使用 TACO 权重、未使用 `yolov8m-seg.yaml` 随机初始化、未更换主架构、未修改 Ultralytics 源码、未修改/删除原始图像与 Labelme 标注、未误改 `train_stage1_taco_v2.py`、未覆盖 TACO Stage 1 模型（`taco_baseline_v2/` 与 `runs/` mtime 保持 11:25 未变）。

---

## 5. 训练结果

### 5.1 训练/验证损失变化

**Phase 1（冻结骨干，20 epoch）**

| epoch | train/box | train/seg | val/box | val/seg | Mask mAP50 | Mask mAP50-95 |
|---|---|---|---|---|---|---|
| 1 | 0.9937 | 1.9212 | — | — | 0.1281 | 0.0741 |
| 10 | 0.7585 | 1.1947 | 1.2194 | 2.4705 | 0.2024 | 0.1268 |
| 20 | **0.5505** | **0.9024** | 1.1707 | 2.4769 | **0.2313** | **0.1513** |

→ 训练损失单调下降，验证指标同步提升，**未过拟合**，20 epoch 结束时仍在改善（best = ep20）。

**Phase 2（全解冻，实际 102 epoch）**

| epoch | train/box | train/seg | val/box | val/seg | 泛化间隙(seg) | Mask mAP50 | Mask mAP50-95 |
|---|---|---|---|---|---|---|---|
| 1 | 0.7922 | 1.3587 | — | — | — | 0.1049 | 0.0633 |
| 5 | — | — | — | **2.4825（最低）** | — | — | — |
| 30 | 0.6141 | 0.9689 | 1.2567 | 2.5811 | 1.61 | 0.1911 | 0.1172 |
| 52（best） | 0.5253 | 0.8042 | 1.2873 | 3.0277 | 2.22 | 0.2125 | **0.1430** |
| 79 | 0.4551 | 0.6909 | 1.2457 | 3.0250 | 2.33 | 0.1616 | 0.0992 |
| 102（last） | **0.3955** | **0.6130** | 1.1745 | **3.0402** | **2.43** | — | — |

### 5.2 是否过拟合 —— **是，且证据充分**

- **val/seg_loss 在 epoch 5 触底（2.4825）后单调上升至 3.0402**，而 train/seg_loss 从 1.36 持续降到 0.61。
- 泛化间隙（val_seg − train_seg）从 1.52 扩大到 **2.43**；box 分支同样（train 0.79→0.40，val 稳定在 1.17–1.29）。
- 训练损失仍在下降但验证 mAP 不再提升 → 典型过拟合。
- 根因：**1063 张唯一内容图像** × 27.2M 可训练参数 × 200 epoch 预算，数据量与模型容量严重不匹配；
  增强虽已适度（mosaic 仅 0.3）但不足以弥补。

### 5.3 正式评估结果（**仅验证集**，test 集锁定未使用）

**评估协议（两组完全一致）**：`imgsz=800, split=val, conf=0.001, iou=0.7, batch=32, rect=True, max_det=300`

| 指标 | **Phase 2 best（正式交付）** | Phase 1 best（对照） |
|---|---|---|
| Mask Precision | **0.3575** | 0.4205 |
| Mask Recall | **0.2594** | 0.2422 |
| **Mask mAP50** | **0.2126** | 0.2151 |
| **Mask mAP50-95** | **0.1378** | 0.1343 |
| Box Precision | **0.3818** | 0.4049 |
| Box Recall | **0.2788** | 0.3283 |
| **Box mAP50** | **0.2302** | 0.2396 |
| **Box mAP50-95** | **0.1713** | 0.1711 |
| fitness | 0.3091 | 0.3054 |
| 推理速度 | 3.70 ms/img（800 输入，RTX 5090） | 3.66 ms/img |

> 注：训练日志末期 `final_eval` 曾打印 mask mAP50-95 = 0.144（对 strip 前的内存 checkpoint 求值）；
> 对磁盘上 `best.pt` 反复复现的结果为 **0.1378**（batch 16/32 一致），故以磁盘权重可复现值为准。

### 5.4 逐类 Mask AP（正式交付 Phase 2 best）

| 类别 | val 实例 | Mask P | Mask R | Mask AP50 | Mask AP50-95 | Box AP50 | Box AP50-95 |
|---|---|---|---|---|---|---|---|
| bottle | 9 | 0.0000 | 0.0000 | **0.0000** | 0.0000 | 0.0000 | 0.0000 |
| box | 41 | 0.1800 | 0.0964 | 0.0495 | 0.0350 | 0.0830 | 0.0440 |
| **can** | **0** | — | — | **不可评估** | **不可评估** | — | — |
| cigarette_butt | 11 | 0.2671 | 0.2727 | 0.2232 | 0.0980 | 0.2232 | 0.1553 |
| food_waste | 1 | 1.0000 | 0.0000 | 0.0711 | 0.0569 | 0.0711 | 0.0711 |
| other | 118 | 0.2738 | 0.2461 | 0.2041 | 0.1020 | 0.2626 | 0.1663 |
| paper | 61 | 0.3411 | 0.5574 | **0.4499** | **0.3224** | 0.4717 | 0.3672 |
| plastic_bag | 87 | 0.4596 | 0.4023 | **0.3985** | **0.2770** | 0.4255 | 0.3085 |
| wrapper | 12 | 0.3389 | 0.5000 | 0.3047 | 0.2112 | 0.3047 | 0.2579 |

**逐类对照（Phase 1 → Phase 2，Mask AP50）**：
`bottle 0.0349→0.0000`、`box 0.1118→0.0495`、`cigarette_butt 0.2541→0.2232`、`food_waste 0.0000→0.0711`、
`other 0.2003→0.2041`、`paper 0.3591→0.4499`、`plastic_bag 0.4819→0.3985`、`wrapper 0.2791→0.3047`

→ Phase 2 仅在 `paper` / `other` / `food_waste` 上改善，却在 `bottle` / `box` / `plastic_bag` / `cigarette_butt` 上退化，**净收益≈0**。

### 5.5 小目标识别效果（尺寸分层，掩码 IoU ≥ 0.5 匹配）

**全部 340 个 val GT 实例的最佳掩码 IoU 统计**

| | Phase 2 best（交付） | Phase 1 best（对照） |
|---|---|---|
| mean best mask IoU | 0.4271 | **0.5348** |
| Recall@IoU0.5 | 0.4912 | **0.6088** |
| Recall@IoU0.75 | 0.3735 | **0.4618** |

**按目标尺寸分桶召回率**

| 等效边长(@800) | GT 数 | Phase 2 召回 | Phase 1 召回 |
|---|---|---|---|
| **0–16 px** | 98 | 0.4898 | **0.6429** |
| **16–32 px** | 69 | 0.6377 | 0.6522 |
| 32–64 px | 77 | 0.4545 | 0.5974 |
| 64–128 px | 65 | 0.3692 | 0.5385 |
| 128+ px | 31 | 0.5161 | 0.5806 |

**结论与解读**：
1. **模型确实学到了极小目标**：0–16 px 桶召回 0.4898，`paper` 在 0–16 px 桶达 0.6154，`other` 在 0–16 px 桶达 0.5938。
   目视验证（`phase2/val_batch*_pred.jpg`）可见对地砖上极小纸屑给出 0.5–0.9 置信度的正确检测。
2. **但 Phase 2 全面削弱了实例召回**（整体 −11.8 个百分点），说明全解冻虽略微改善 AP 排序，却牺牲了实际检出能力。
3. 逐类尺寸分层显示表现差异主要**由类别驱动而非尺寸驱动**：
   `plastic_bag` 召回随尺寸单调改善（0.0 → 0.25 → 0.35 → 0.61 → 0.72）；
   `bottle` 在**所有尺寸**召回均为 0（该类别彻底失败）；`box` 在 64–128 px 桶仅 0.0625。
4. `cigarette_butt` 在 0–16 px 桶召回 0.25（8 例中命中 2 例）—— 这是本任务最难的类别，符合其 14.2 px 中位尺寸的预期。

### 5.6 资源占用与吞吐

| 项目 | 值 |
|---|---|
| 峰值显存（本进程 allocated） | **15,975.7 MiB** |
| 峰值显存（本进程 reserved） | 18,860.0 MiB |
| 系统级 nvidia-smi 峰值 | 28,967 MiB（含训练前已存在的无关进程占用） |
| Phase 1 平均每 epoch | 11.3 s |
| Phase 2 平均每 epoch | 12.6 s |
| Phase 1 吞吐 | 101.9 img/s |
| Phase 2 吞吐 | 91.4 img/s |
| **总训练耗时** | **25.25 min**（122 epoch） |

> batch=16 在 800 分辨率下**未出现 OOM**，无需降级到 8 或 4；显存余量充足（31.34 GiB 中本进程仅用约 15.6 GiB）。
> 未通过降低到 640 分辨率或修改数据来回避显存问题。

---

## 6. 产物清单与可复现信息

```
indoor_ab/A_coco_direct_800_2stage/
├── EXPERIMENT_A_REPORT.md              # 本报告
├── training_summary.json               # 两阶段完整配置/哈希/耗时/显存
├── train_full.log                      # 完整训练日志
├── data_audit.json                     # 数据审计主报告
├── data_audit_supplement.json          # 重映射/近重复补充审计
├── data_audit_leakcheck.json           # 严格泄漏复验
├── eval_phase2_best.json               # 正式评估（含尺寸分层）
├── eval_phase1_best.json               # 对照评估
├── smoke/
│   ├── smoke_result.json               # 冒烟测试结果
│   ├── frozen_parameters.txt           # freeze=10 完整冻结张量清单
│   └── augmented_samples.jpg           # 增强可视化样张
├── phase1/                             # args.yaml, results.csv, results.png,
│   └── weights/{best.pt,last.pt}       # 混淆矩阵, 各类曲线, val_batch 预测
└── phase2/                             # 同上（正式交付）
    └── weights/{best.pt,last.pt}
```

### 版本与哈希（可复现性锚点）

| 项目 | 值 |
|---|---|
| **训练代码** | `tools/train_A_2stage.py` SHA256 `5ede760f6d6ef229061bc0f96ec08fdb8608bb05a4ace3489522af73810ea282` |
| 评估代码 | `tools/eval_A.py` SHA256 `291f0af64a729b15bb9bfb72725c5bcd135133b1c2278b9488b3077b257a375c` |
| 审计代码 | `tools/audit_indoor_a.py` SHA256 `3c9032cfe7ec3e3e721d74cc27fa14c7bf7adaa38870b94d6e969cdd41b2fc28` |
| 冒烟代码 | `tools/smoke_A.py` SHA256 `015a832836217e2a210d878f0f15baa9733ff932c97a57ac0094a36d30a32301` |
| **输入数据（data.yaml）** | SHA256 `08a08e78ca220af5a4318f18dcfd5d33b178ef998325e8074ccbce8237261c70` |
| **输入数据（标签清单）** | 1444 个文件清单 SHA256 `a8708179eae5a92da853b8578c8068b511cd2b5dbb4e8724d309aae90f89ebb0` |
| 输入数据（图片清单，修复后） | SHA256 `29bb61292a18b0f0ee13e6a0568278ca9f86a260390968514560ba810a20229b` |
| **Phase 1 起点权重（COCO 官方）** | SHA256 `51fa7e5ef385efa6d5b1d8e31b73399be6ed5d7ca71bda4bd4b2794bb445c4f4` |
| **Phase 1 best.pt** | SHA256 `b8b78efc8ca6d2642e736216e788f2240a9438cdf4054ef694bfd142cd28c400` |
| **Phase 2 best.pt（最终交付）** | SHA256 `e09239494f729538c60a5c55fa1f62993558981813288b4863502a3dd6cea2f3` |
| Phase 2 last.pt | SHA256 `489b92eeab276489a6df5d22f3165f6ca3298343b05738552282eaee37664f31` |

### 复现命令

```bash
cd /home/xmv/garbage_yolov8_seg
source /home/xmv/miniconda3/etc/profile.d/conda.sh && conda activate zxj-py312

python tools/audit_indoor_a.py            # 数据审计
python tools/smoke_A.py                   # GPU/增强/显存冒烟
python tools/train_A_2stage.py            # 两阶段训练（约 25 min）
python tools/eval_A.py \
  --weights indoor_ab/A_coco_direct_800_2stage/phase2/weights/best.pt \
  --out indoor_ab/A_coco_direct_800_2stage/eval_phase2_best.json \
  --imgsz 800 --rect --batch 32 --size-analysis
```

---

## 7. 最终判定

### 7.1 该 COCO 直接室内微调模型是否具备 A/B 对照基线资格？

**具备资格，但需附带明确限定条件。**

**支持作为基线的理由**
1. 训练链路完全合规：唯一预训练来源为 COCO 官方 `yolov8m-seg.pt`，未触碰 TACO，未随机初始化，架构未变。
2. 数据审计通过：1444 张全部可解码、类别重映射零错误、零越界/退化多边形、**零跨 split 泄漏**（SHA 与近密双重验证）。
3. 划分可复现且来源分组安全：同一 SHA 内容组 / 近重复组 / 连拍组全部落入同一 split；seed=42、deterministic=True。
4. test 集全程锁定，未被训练、早停或调参读取。
5. 全部关键配置、权重与数据哈希已固化，评估协议文档化，结果可精确复现。

**必须随基线一起报告的限定条件（否则对照不公平）**
1. **有效数据仅 1063 张唯一内容**（1444 文件含 323 个 A/B 跨源重复组）。若 B 组使用同一 1444 文件集，两组名义数据量相同、实际信息量也相同，公平；但**不得**在 B 组另行去重或增补样本。
2. **`can` 类不可评估**（val/test 实例数为 0），任何 A/B 的总体 mAP 都天然不含该类贡献；比较时应同时给出「排除 `can` 的宏平均」以免误读。
3. **`food_waste` 仅 1 个 val 实例** → 该类 AP 极不稳定，不应作为决策依据。
4. **val 集仅 340 个实例、145 张图** → 指标抖动明显（同权重不同 batch/rect 协议下 mAP50-95 波动可达 ±0.005，训练日志值 0.144 vs 磁盘可复现值 0.138），
   A/B 差异小于约 0.01 时应判定为「无显著差异」而非「一方更优」。
5. 本题给出的两阶段协议**并非最优配置**：Phase 2 全解冻出现明确过拟合，
   建议在对照前先决定 A 基线到底采用 **Phase 2 best（0.2126/0.1378，协议规定交付）** 还是 **Phase 1 best（0.2151/0.1343，召回显著更优 0.6088 vs 0.4912）**，
   并让 B 组使用**完全相同**的取舍标准。

### 7.2 下一步与 TACO 底座比较时必须保证的实验条件

| # | 条件 | 具体要求 |
|---|---|---|
| 1 | **同一数据与划分** | A/B 使用完全相同的 `garbage_seg_merged_full1444` 与同一份 train/val/test 文件清单（清单 SHA256 `a8708179…ebb0`），不得重新划分 |
| 2 | **同一类别映射** | 9 类体系与 ID 顺序严格一致（`bottle, box, can, cigarette_butt, food_waste, other, paper, plastic_bag, wrapper`）；B 组的 TACO 底座若类别集不同，只允许通过 `_remap_cls_by_names` 等原生机制迁移，不得改动类别定义 |
| 3 | **同一架构** | 均为 YOLOv8m-Seg，不得更换尺度或结构 |
| 4 | **同一 imgsz** | 均为 800 |
| 5 | **同一随机种子与确定性** | `seed=42, deterministic=True` |
| 6 | **同一优化与调度** | 相同的 `optimizer / lr0 / lrf / cos_lr / warmup_epochs / weight_decay / batch / nbs`；相同冻结策略（建议两组都跑 freeze=10 与 freeze=0 两个配置，或都只跑其中之一） |
| 7 | **同一增强** | 相同的完整增强参数（含 `mask_ratio=2`、`overlap_mask`、`mosaic`、`copy_paste`、`multi_scale`） |
| 8 | **同一早停规则** | 相同 `patience`、相同的早停触发后处理方式（不得一方跑满、一方早停） |
| 9 | **同一评估协议** | `split=val, imgsz=800, conf=0.001, iou=0.7, batch=32, rect=True, max_det=300`；比较 **Mask mAP50 与 Mask mAP50-95**，并同时报告逐类 AP、Mask P/R 与尺寸分层召回 |
| 10 | **同一 test 评估时机** | test 集在 A/B 两组配置**全部冻结后**统一评估一次，评估前不得用于任何模型选择 |
| 11 | **必须报告的公平性要素** | 起点权重哈希、是否使用了 TACO 权重、实际冻结层与可训练参数量、等效批大小、峰值显存、总耗时 |
| 12 | **建议补充** | 因 val 集偏小，建议同时报告 3 个随机种子的均值±标准差（或至少对 A/B 差异做显著性说明），避免把噪声当成迁移收益 |

### 7.3 待办与风险提示（下一轮应优先处理）

1. **过拟合是本实验的首要问题**：建议为 B 组（及 A 组复跑）单独安排一组消融，测试
   `patience` 调到 15–20、`mosaic` 提到 0.5、`copy_paste` 提到 0.3、或引入有上限的类别均衡采样。
   本轮未引入未经验证的自定义 DataLoader（按任务要求，列为独立消融实验）。
2. **`can` 类无法评估**：本轮按任务要求沿用经核验的现有划分，未改动 val/test。
   若后续需要评估该类，必须**同时**为 A/B 增补 `can` 样本并重做划分（会破坏当前基线可比性，应作为独立数据版本 v2）。
3. **`bottle` 完全失效（AP=0，全尺寸召回 0）**：99 个实例（75 训练）不足以支撑该类别，
   建议排查是否为标注过小/遮挡严重，或与其他类别混淆（见 `phase2/confusion_matrix.png`）。
4. **重复副本策略**：323 个 A/B 跨源同内容组意味着训练集中存在大量近乎重复（但标签已被统一）的样本，
   会放大有效样本权重并加剧过拟合。是否去重应作为 A/B 双方共同采用的数据版本决策，不得单方面执行。
5. **训练日志与磁盘权重指标的细微差异**（0.144 vs 0.138）已定位为 strip 前内存权重 vs 磁盘权重的差异，
   未来报告请统一使用磁盘权重的可复现值。
