# TACO 第一阶段：人工裁决版 v2 报告

**项目**：`/home/xmv/garbage_yolov8_seg`
**环境**：conda `zxj-py312`（Python 3.12.15 / Pillow 12.3.0 / ultralytics 8.4.176）
**日期**：2026-10-13
**本轮范围**：在 v1 安全候选子集基础上，按**人工裁决 7 项**更新映射、重跑转换与无 GPU 验收。
**同一日复议 1 项**：`Drink carton` 由 `can` 改判 `box`（见 §1.1），已重跑转换与验收，本文数据均为复议后结果。
**未启动训练**，未修改 `datasets/taco/raw/` 与 `datasets/garbage_seg_merged_full1444/`。

---

## 0. 交付标志

```
TACO_CLASS_MAPPING      = V2_HUMAN_RATIFIED_PARTIAL
YOLO_SEG_CONVERSION     = DONE           datasets/taco/yolo_seg
INPUT_IMAGES            = 1500
KEPT_IMAGES             = 1145           train/val/test = 917/114/114
EXCLUDED_IMAGES         = 355            (整图排除并集)
WRITTEN_INSTANCES       = 2625
RECONCILIATION          = PASS           4784 = 2625 + 2159 + 0
DATA_QUALITY_ACCEPTANCE = PASS
READY_FOR_STAGE1        = YES            (含 §9 非阻塞条件)
```

---

## 1. v1 → v2 变更（2026-10-13 人工裁决）

| TACO 类别 | 实例 | 图 | v1 | **v2** |
|---|---|---|---|---|
| Plastic film | 451 | 310 | DEFER | **→ `wrapper`** |
| Glass bottle | 104 | 74 | DEFER | **→ `bottle`** |
| Paper cup | 67 | 62 | DEFER | **→ `paper`** |
| Aluminium foil | 62 | 43 | DEFER | **→ `wrapper`** |
| Drink carton | 45 | 41 | DEFER | **→ `box`**（复议自 `can`，见 §1.1） |
| Toilet tube | 5 | 3 | DEFER | **→ `other`** |
| Paper straw | 4 | 4 | DEFER | **→ `other`** |
| Aerosol | 10 | 10 | DEFER | DEFER（未裁决） |
| Paper bag | 27 | 23 | DEFER | DEFER（未裁决） |
| Wrapping paper | 12 | 11 | DEFER | DEFER（未裁决） |
| 多部件 Polygon（D7） | 114 条 | 81 | exclude | exclude（未裁决） |

**影响**：MAP 类别 47 → **54**；映射实例 3475 → **4213**；保留图像 798 → **1145**。

### 1.1 `Drink carton` 复议（2026-10-13）

原判 `Drink carton → can` 把纸板/复合材质的方柱饮料盒（利乐包）并入金属罐类，与 `can`
的几何定义（**圆柱易拉罐**）冲突，会污染该类的「圆柱罐」特征。经复议改判：

| 项 | 复议前 | 复议后 |
|---|---|---|
| `Drink carton`(16) | `MAP -> can` | **`MAP -> box`** |
| 理由 | — | 纸板/复合材质、方柱盒形 → 归 `box`；保持 `can` 几何特征纯净 |

- 该类**原本已 `MAP`**，图像的保留/排除集合不变（仍 1145 / 355）；
- 仅落盘标签类别由 `2`(can) 变为 `1`(box)：45 条标注中 **29 条实写**，
  其余 16 条所在图像被 D6/D7 整图排除；
- 实例分布变化：`can` 230 → **201**，`box` 161 → **190**（合计 2625 不变）；
- 因 `can` 属稀缺类（`RARE_CLASSES`），复议后划分的稀缺类配额随之变化，
  **train/val/test 的图像归属与复议前不同**（图像数仍 917/114/114）；
- 可视化 `yolo_seg/viz/ratified_mapping_check.png` 中该格掩码颜色由 `can`（黄）变为 `box`（绿）。

---

## 2. D1–D7 最终状态

| 编号 | 裁决 | 状态 |
|---|---|---|
| **D1** | `other` 为兜底类，27 类 OTHER_CANDIDATE 逐类审核通过 | ✅ 已执行 |
| **D2** | Plastic film 不强制映射 | ✅ 人工裁决 → `wrapper` |
| **D3** | Carton 系列 | ✅ 盒形 → `box`（含 `Drink carton`，2026-10-13 复议）；`Toilet tube` → `other` |
| **D4** | Glass bottle | ✅ 人工裁决 → `bottle` |
| **D5** | paper 形态边界 | ✅ `Paper cup` → `paper`；`Paper straw` → `other`；`Paper bag` / `Wrapping paper` 仍待裁决 |
| **D6** | Unlabeled litter 269 图 | ✅ 整图排除 |
| **D7** | 多部件 Polygon 策略 | ⏳ 仍为 `exclude`（整图隔离）；转换器内置 `largest / split / union-contour` |

**类别动作汇总（60 类 / 4784 实例）**

| 动作 | 类别数 | 实例数 |
|---|---|---|
| `MAP` | 54 | 4213 |
| `DEFER`（整图排除） | 3 | 49 |
| `EXCLUDE_STRUCTURAL` | 1 | 517 |
| `EXCLUDE_UNRESOLVED` | 1 | 5 |
| `EMPTY` | 1 | 0 |

**MAP 层到 9 类的对应**

| 目标类 | TACO 来源 |
|---|---|
| `bottle` | Other plastic bottle, Clear plastic bottle, **Glass bottle** |
| `box` | Other carton, Egg carton, Corrugated carton, Meal carton, Pizza box, **Drink carton** |
| `can` | Food Can, Drink can |
| `cigarette_butt` | Cigarette |
| `food_waste` | Food waste |
| `other` | Other plastic + 27 类杂项 + **Toilet tube** + **Paper straw** |
| `paper` | Magazine paper, Tissues, Normal paper, **Paper cup** |
| `plastic_bag` | Garbage bag, Single-use carrier bag, Polypropylene bag |
| `wrapper` | Other plastic wrapper, Crisp packet, **Plastic film**, **Aluminium foil** |

---

## 3. 排除规则（防"未映射实例变背景"）

YOLOv8-Seg 无 ignore 区域机制 ⇒ 规则：**只保留"每个实例都能确证映射"的图像**。

| 排除原因 | 触发条件 | 图像数 |
|---|---|---|
| `unlabeled_litter` | 含 `Unlabeled litter`（D6） | 269 |
| `multipart_polygon` | 含多部件 Polygon（D7=exclude） | 81 |
| `deferred_semantics` | 含 3 个未裁决类别任一 | 43 |
| `unresolved_category` | 含 `Six pack rings` | 4 |

---

## 4. 排除统计：**并集**（不得直接相加）

### 4.1 D6 与 D7 的并集

| 项 | 图像数 | 实例数 |
|---|---|---|
| D6 `Unlabeled litter` | 269 | 517 |
| D7 多部件 Polygon | 81 | 114 |
| 朴素相加（**未采用**） | 350 | 631 |
| **实际并集** | **321** | — |
| 重叠 | **29** | — |

### 4.2 全部原因（含重叠）与最终并集

| 原因 | 图像数（含重叠） | 实例数（含重叠） |
|---|---|---|
| `unlabeled_litter` | 269 | 517 |
| `multipart_polygon` | 81 | 114 |
| `deferred_semantics` | 43 | 49 |
| `unresolved_category` | 4 | 5 |
| 各项之和（**未采用**） | 397 | 685 |
| **去重后的整图排除并集** | **355** | **2159** |
| **实际保留** | **1145** | **2625** |

### 4.3 剩余 DEFER 明细

| TACO 类别 | 实例数 | 图数 | 待裁决去向 |
|---|---|---|---|
| Paper bag | 27 | 23 | paper / plastic_bag / other |
| Wrapping paper | 12 | 11 | paper / wrapper |
| Aerosol | 10 | 10 | can / other |

---

## 5. 保留样本与九类实例分布

**保留 1145 张，train/val/test = 917 / 114 / 114**（按独立单位分组，防泄漏）。

### 5.1 实例分布

| 类 | train | val | test | **合计** | 室内 1444 |
|---|---|---|---|---|---|
| `bottle` | 210 | 59 | 9 | **278** | 99 |
| `box` | 120 | 25 | 45 | **190** | 584 |
| `can` | 130 | 68 | 3 | **201** | **9** |
| `cigarette_butt` | 283 | 58 | 27 | **368** | 121 |
| `food_waste` | 2 | 3 | **0** | **5** | 26 |
| `other` | 753 | 102 | 72 | **927** | 1313 |
| `paper` | 97 | 17 | 7 | **121** | 674 |
| `plastic_bag` | 44 | 7 | 12 | **63** | 774 |
| `wrapper` | 375 | 40 | 57 | **472** | 215 |
| **合计** | 2014 | 379 | 232 | **2625** | 3815 |

### 5.2 图像分布

| 类 | train | val | test | 合计 |
|---|---|---|---|---|
| `bottle` | 172 | 45 | 9 | 226 |
| `box` | 100 | 13 | 27 | 140 |
| `can` | 86 | 42 | 3 | 131 |
| `cigarette_butt` | 114 | 24 | 10 | 148 |
| `food_waste` | 2 | 3 | 0 | 5 |
| `other` | 492 | 58 | 55 | 605 |
| `paper` | 81 | 12 | 7 | 100 |
| `plastic_bag` | 32 | 5 | 6 | 43 |
| `wrapper` | 305 | 25 | 40 | 370 |

### 5.3 划分与泄漏

- 独立单位 = 同一 SHA256 ∪ dHash≤5 近重复（dHash 在 **EXIF 烘焙后**空间计算）；
- 1145 张 → **1140** 个独立单位（5 对近重复合并）；
- 划分结果落盘 `split_assignments.csv`；稀缺类贪心配额优先进入 val/test；
- 跨 split 相同内容 **0**、跨 split 近重复 **0**；
- 未使用室内 test 划分；室内数据只读。

### 5.4 与室内的互补价值

`can` 9 → **+201**、`cigarette_butt` 121 → **+368**、`bottle` 99 → **+278**、`wrapper` 215 → **+472**。
`food_waste` 仍**无法由 TACO 补充**（仅 5 实例）。

---

## 6. 转换实现与验证

| 项 | 结果 |
|---|---|
| EXIF 逐字节复制 / 烘焙重编码 | 685 / 460 |
| 残留 EXIF 方向 2..8 | **0** |
| 存储尺寸 ≠ COCO 标注坐标系 | **0** |
| 坐标裁剪事件 | **25**（4 条负顶点 + 21 条恰在 w/h 边界） |
| 裁剪后退化丢弃 | **0** |
| 重复 annotation id（309 / 4040） | 各 2 条完整保留 |
| 非法标签行（类 ID / 偶数坐标 / ≥3 点 / [0,1] / 面积>0） | **0** |
| 顶点数范围 | 3 – 543 |
| 空标签文件 | 0 |
| 用矩形框冒充轮廓 | **未使用**（100% polygon） |

**实例对账**

```
输入标注                 = 4784
  写入                   = 2625
  因整图排除而丢弃        = 2159
  因退化/裁剪消失          = 0
  合计                   = 4784   ✔
```

逐条动作见 `conversion_manifest.csv`（4784 行）。

**独立复核**：从官方 COCO 独立重推导每行标签并与写出文件逐行比对 → **不一致 0**；
Ultralytics `verify_image_label` 全量解析 → **失败 0**。

---

## 7. 可视化核对

`datasets/taco/yolo_seg/viz/`

| 文件 | 用途 |
|---|---|
| `class_0_bottle.png` … `class_8_wrapper.png` | 逐类放大裁剪，核对掩码与物体轮廓 |
| `ratified_mapping_check.png` | **本次 7 项人工裁决**的逐类实例（掩码颜色 = 目标类） |
| `exif_transposed_samples.png` | EXIF 3/6/8 样本方向正确性 |
| `crowded_samples.png` | 密集多实例场景 |

**结论**：9 类掩码紧密贴合轮廓；已裁决类别（铝箔→wrapper、玻璃瓶→bottle、纸杯→paper、
塑料薄膜→wrapper、纸吸管→other、**复议后的饮料纸盒→box**）掩码均正确。
`Drink carton` 复议前被标为 `can` 的几何冲突已消除：`ratified_mapping_check.png` 中该格
掩码颜色由 `can`（黄）变为 `box`（绿），与方柱盒形一致。

---

## 8. 转换缺陷清单 / 已知限制

| 编号 | 缺陷 / 限制 | 影响 | 处置 |
|---|---|---|---|
| **B1** | **近重复分组比较了错误字段**（`dh_by_id` 的键当哈希比），近重复合并自 v1 起一直是**空操作** | v1 未暴露（6 对近重复无一同时被保留）；v2 首轮出现 1 对近重复跨 train/val | **已修复**（改为比较哈希、合并 image_id）；验收复测通过 |
| **B2** | dHash 必须在 **EXIF 烘焙后**空间计算 | 原始空间对 EXIF-6 图会与其它副本失配 | **已修复**（`dhash_file(..., apply_exif=True)`） |
| C1 | `food_waste` 仅 **5** 实例，且 **test = 0** | 该类在 test 不可评估 | 需另寻数据源 |
| C2 | `can` 在 test 仅 **3** 实例 | test 上 `can` 指标不稳定 | 按类分别汇报 |
| ~~C3~~ | ~~`Drink carton → can` 语义不一致~~ | **已复议**：2026-10-13 改判 `box`，`can` 几何定义不再被纸盒污染 | §1.1（已关闭） |
| C4 | `other` 占 **35.3%**（927/2625） | 主导性偏差 | 训练时加权/重采样评估 |
| C5 | 排除率 **23.7%**（355/1500） | 剩余为待裁决/D6/D7 | 裁决后可再回收 |
| C6 | 多部件 81 图仍隔离 | 缺失 114 条标注 | D7 待裁决 |
| C7 | 重复 annotation id 各有一半落在被排除图像 | 实际各写入 1 条 | 两个不同物体，无信息丢失 |
| C8 | `food_waste` 无法由 TACO 补充 | 该类短板仍在 | 阶段二另寻数据源 |
| C9 | TACO 与室内影像域差异未评估 | 迁移效果未知 | 阶段二议题 |

---

## 9. 剩余需人工裁决的类别

| 项目 | 规模 | 可选项 |
|---|---|---|
| Paper bag | 27 实例 / 23 图 | paper / plastic_bag / other |
| Wrapping paper | 12 实例 / 11 图 | paper / wrapper |
| Aerosol | 10 实例 / 10 图 | can / other |
| 多部件 Polygon 策略（D7） | 114 条 / 81 图 | largest / split / union-contour / exclude |

~~`Drink carton → can` 复核（可选）~~ → **已复议为 `box`（2026-10-13，见 §1.1）**。

裁决后增量重跑：`python tools/taco/convert_coco_to_yolo_seg.py --mapping <新映射> --overwrite`

---

## 10. 无 GPU 数据质量验收

命令：`python tools/taco/validate_taco_yolo_seg.py`（CPU/内存即可）

| 检查项 | 结果 |
|---|---|
| 类别编号与室内 9 类完全一致 | **PASS** |
| 图像/标签配对（917 / 114 / 114） | **PASS** |
| Ultralytics `verify_image_label` 全量解析 | **PASS**（失败 0） |
| 多边形合法性 | **PASS**（非法 0） |
| EXIF 残留 / 存储尺寸与标注坐标系一致 | **PASS**（0 / 0） |
| 跨 split 相同内容 / 近重复 | **PASS**（0 / 0） |
| 被排除内容泄漏进保留集 | **PASS**（0） |
| 重复 annotation id 保留 | **PASS** |
| 实例数量对账 | **PASS**（4784 = 2625 + 2159 + 0） |
| 标签往返重推导一致 | **PASS**（不一致 0） |

```
DATA_QUALITY_ACCEPTANCE = PASS
READY_FOR_STAGE1        = YES
```

**非阻塞条件**

1. `food_waste`（5 实例，test=0）与 `can`（test=3）在 test 上不可评估 —— 须按类分别汇报；
2. 3 个 DEFER 类别（49 实例 / 43 图）与多部件策略（D7）仍待裁决；
3. `Drink carton → can` 的语义冲突已通过复议改判 `box` 关闭（§1.1）；
4. 候选子集排除 355 图，域覆盖不完全；室内 1444 自身的 295 项标注冲突独立存在。

> 若"STAGE1"要求**全量 60 类覆盖**：`NO`（3 类 + D7 待裁决）；
> 若要求"可安全用于第一阶段训练的候选子集"：**YES**。

---

## 11. 交付物与约束核对

```
tools/taco/taco_class_mapping_v2.json          v2 人工裁决映射（权威，含 2026-10-13 Drink carton 复议）
tools/taco/taco_class_mapping_v1.json          v1 安全候选映射（历史）
tools/taco/convert_coco_to_yolo_seg.py        转换器（含 --overwrite 与四种多部件策略）
tools/taco/validate_taco_yolo_seg.py          无 GPU 验收 + 可视化
datasets/taco/yolo_seg/                       数据集（见其 README.md）
datasets/taco/TACO_STAGE1_YOLO_SEG_V2_REPORT.md  本报告
```

| 约束 | 状态 |
|---|---|
| 数据保存在 `datasets/taco/` 内 | ✅ |
| 未覆盖原始数据 / 室内 1444 | ✅（raw 只读，标注 SHA256 未变） |
| 允许 `other` 但逐类审核 | ✅ |
| D6/D7 按并集计算未相加 | ✅ 321（非 350） |
| 未确定类别不得静默变背景 | ✅ 整图排除，泄漏检查 0 |
| EXIF / 越界 / 重复 ID / 多边形 / 对账 已实现并验证 | ✅ |
| 可视化核对、防泄漏分组、类别编号一致 | ✅ |
| 未用矩形框冒充分割轮廓 | ✅ |
| 无 GPU 质量验收 + READY_FOR_STAGE1 | ✅ YES |
| Minimal-artifact | ✅（单份数据集；v1 报告已被本版取代） |
| 未启动训练 | ✅ |
