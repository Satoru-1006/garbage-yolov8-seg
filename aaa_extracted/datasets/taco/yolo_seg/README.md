# datasets/taco/yolo_seg — TACO 第一阶段【人工裁决版 v2】

**状态：`V2_HUMAN_RATIFIED_PARTIAL`（候选子集，仍有 3 类待裁决）**

TACO 官方 COCO 标注（1500 图 / 4784 实例）转换为项目统一 **9 类** YOLOv8-Seg
数据集。只包含**语义可确证**的样本；任何未映射实例一律**整图排除**，
绝不静默变成背景负样本。

| 项 | 值 |
|---|---|
| 输入图像 | 1500 |
| **保留图像** | **1145** |
| 整图排除 | 355 |
| train / val / test | 917 / 114 / 114 |
| 写入实例 | 2625 |
| 类别 | 9（编号与 `garbage_seg_merged_full1444` **完全一致**） |
| 数据质量验收 | **PASS** |
| `READY_FOR_STAGE1` | **YES**（含 §9 非阻塞条件） |

## 2026-10-13 人工裁决（v1 → v2）

| TACO 类别 | 实例 | → 目标类 |
|---|---|---|
| Plastic film | 451 | `wrapper` |
| Glass bottle | 104 | `bottle` |
| Paper cup | 67 | `paper` |
| Aluminium foil | 62 | `wrapper` |
| Drink carton | 45 | `box`（同日复议，原判 `can`） |
| Toilet tube | 5 | `other` |
| Paper straw | 4 | `other` |

**2026-10-13 复议**：`Drink carton` 由 `can` 改判 `box`。利乐包为纸板/复合材质、
方柱盒形，与 `can` 的「圆柱金属罐」几何特征冲突；该类原本已 `MAP`，图像保留/排除
集合不变，仅标签类别由 2 变为 1（实写 29 条；其余 16 条所在图像被 D6/D7 整图排除）。
`can` 实例 230 → **201**，`box` 实例 161 → **190**。

仍 DEFER（整图排除）：`Aerosol`(10)、`Paper bag`(27)、`Wrapping paper`(12)；
多部件 Polygon 策略仍为 `exclude`。

## 目录

```
yolo_seg/
├── data.yaml                  # 9 类，绝对路径，train/val/test
├── images/{train,val,test}/   # 已烘焙 EXIF 方向
├── labels/{train,val,test}/   # <cls> x1 y1 ... xn yn（归一化，6 位小数）
├── conversion_manifest.csv    # 4784 条逐条处理动作（可追溯）
├── images_manifest.csv        # 1145 条图像级信息（尺寸/EXIF/哈希）
├── split_assignments.csv      # 图像 → split 清单（划分结果落盘）
├── exclusion_report.json      # 排除原因统计与并集计算
├── conversion_summary.json    # 九类分布、对账、重复 ID 审计
├── validation_report.json     # 无 GPU 验收结果
└── viz/                       # 对齐核对图（每类 / 已裁决类别 / EXIF / 密集场景）
```

## 复现

```bash
source $(conda info --base)/etc/profile.d/conda.sh && conda activate zxj-py312
cd <repo-root>
python tools/taco/convert_coco_to_yolo_seg.py --mapping tools/taco/taco_class_mapping_v2.json --analyze-only
python tools/taco/convert_coco_to_yolo_seg.py --mapping tools/taco/taco_class_mapping_v2.json
python tools/taco/validate_taco_yolo_seg.py
```

完整说明见 `../TACO_STAGE1_YOLO_SEG_V2_REPORT.md`；
裁决依据见 `tools/taco/taco_class_mapping_v2.json`（权威）与 `..._v1.json`（历史）。

**未做**：未修改 `datasets/taco/raw/`；未修改 `datasets/garbage_seg_merged_full1444/`；未启动任何训练。
