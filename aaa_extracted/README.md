# garbage-yolov8-seg — 垃圾目标检测/分割数据集构建与 YOLOv8-Seg 训练

面向垃圾图像的统一 **9 类** YOLOv8-Seg 数据集构建流水线，包含多源数据集的
类别对齐、合并审计、COCO→YOLO 分割标注转换，以及可追溯的完整性校验清单。

> **状态：进行中（PROVISIONAL）。** TACO 候选子集已完成数据质量验收并记录了
> 一个 YOLOv8m-Seg 基线；合并数据集尚未通过数据质量验收，也尚未训练。
> 详见文末「状态」和 `taco_baseline_v2/README.md`。

---

## ⚠️ 本仓库不包含图片和权重

出于**体积**与**第三方图片版权**两方面考虑，仓库只入库代码、标注（labels）、
清单（manifests）、配置与审计报告，**不含任何图像文件与 `.pt` 权重**。

| 内容 | 体积 | 是否入库 | 原因 |
|---|---|---|---|
| `datasets/**/labels/` 标注 | ≈18 MB | ✅ | 体积小，是复现的关键 |
| manifests / 报告 / 配置 | ≈4 MB | ✅ | 可追溯性 |
| `datasets/**/images/` 图片 | ≈3.4 GB | ❌ | 体积 + 版权 |
| `datasets/taco/raw/`（含 2.5 GB 官方归档） | ≈5.1 GB | ❌ | 可由 DOI 重新下载 |
| `*.pt` 权重 | 3 × 53 MB | ❌ | 建议走 Release / Git LFS |

---

## 目录结构

```
.
├── tools/
│   ├── merge_full1444.py            # 合并两个源数据集（类别重映射 + 校验）
│   ├── merge_manifest_full1444.csv  # 1444 条逐图记录（含 SHA256，可校验）
│   ├── merge_audit_full1444.md      # 合并审计报告
│   ├── class_map_full1444.json      # 类别映射表
│   └── taco/
│       ├── fetch_zenodo_taco.sh         # 从官方 Zenodo DOI 拉取 TACO 原始数据
│       ├── download_taco_official.py    # 备选：按 Flickr URL 逐图下载
│       ├── audit_taco.py                # TACO 官方标注审计
│       ├── analyze_class_mapping.py     # 类别映射候选分析
│       ├── convert_coco_to_yolo_seg.py  # COCO → YOLO-Seg 转换
│       ├── validate_taco_yolo_seg.py    # 无 GPU 验收
│       └── taco_class_mapping_v2.json   # 类别裁决（权威版本）
├── datasets/
│   ├── taco/yolo_seg/               # TACO 转换结果（labels + 清单）
│   ├── garbage_seg_merged_full1444/ # A+B 合并数据集（labels + data.yaml）
│   ├── half_dataset_yolo/           # 源数据集 B（labels + data.yaml）
│   └── waste_ds_c3_neg_yolo/        # 源数据集 A（labels + data.yaml）
└── runs/segment/runs/smoke/         # 流程自检产物（曲线图、results.csv）

另有：

```text
├── train_stage1_taco_v2.py          # TACO 9 类 YOLOv8m-Seg 基线训练入口
└── taco_baseline_v2/train/           # 参数与 results.csv（不含权重和图像）
```

实验 A（COCO 直接微调两阶段）的核心脚本与记录位于：

```text
├── tools/indoor_a/                   # 独立审计、冒烟、训练、评估脚本
└── experiments/indoor_a/
    └── A_coco_direct_800_2stage/     # 报告、审计 JSON、参数与 results.csv
```

该实验同样不入库图片、权重、曲线图和完整训练日志；详见该目录的
`README.md`。原始训练记录显示 test split 未参与训练或验证调参，但该实验结果
仍属于基线记录，不等同于部署或生产性能认证。
```

---

## 数据来源与可复现性

**关键区分：本仓库的两条数据链路，只有一条可以完整复现。**

### 链路 A — TACO（✅ 完整可复现）

TACO 官方数据（1500 图 / 4784 实例）→ 统一 9 类 YOLO-Seg 标注。
图片不在仓库里，但可按下述步骤重建：

```bash
# 1. 拉取官方归档（DOI 10.5281/zenodo.3587843，含 MD5 校验，约 2.72 GB）
bash tools/taco/fetch_zenodo_taco.sh

# 2. 解压出 images/ 与 annotations.json 放进 datasets/taco/raw/
# 3. COCO -> YOLO-Seg 转换（先 analyze-only 预览，再实际写出）
python tools/taco/convert_coco_to_yolo_seg.py --mapping tools/taco/taco_class_mapping_v2.json --analyze-only
python tools/taco/convert_coco_to_yolo_seg.py --mapping tools/taco/taco_class_mapping_v2.json

# 4. 无 GPU 验收（标签合法性、对账、重复 ID 审计）
python tools/taco/validate_taco_yolo_seg.py
```

转换产出为 `datasets/taco/yolo_seg/`：保留 1145 / 排除 355，
train / val / test = 917 / 114 / 114，写入 2625 个实例。

> TACO 官方原图来自 Flickr，**版权归原作者**。本仓库因此不重新分发图片，
> 只提供下载脚本与 DOI。逐图处理动作见 `datasets/taco/yolo_seg/conversion_manifest.csv`
> （4784 条），数据来源与校验值详见 `datasets/taco/raw/SOURCE.md`。

### 链路 B — 自采数据集（❌ 不可复现）

| 代号 | 数据集 | 图片数 | 类别数 |
|---|---|---|---|
| A | `waste_ds_c3_neg_yolo` | 694 | 5 |
| B | `half_dataset_yolo` | 750 | 9 |

这两个数据集**没有公开下载源**，图片不在仓库中，因此 `garbage_seg_merged_full1444`
（1444 图）**无法由外部重建**。仓库提供的是：

- 全部标注（`labels/`）与划分结果，即"拿到原图即可复现"的完整定义；
- `tools/merge_manifest_full1444.csv`：1444 条逐图记录，含**每张源图的 SHA256**，
  可用于验证你手上的原图与本次合并所用的是否为同一批；
- 类别重映射表与合并审计报告。

如需图片，请联系仓库所有者确认授权与隐私条件。

---

## 统一类别体系（9 类）

```
0 bottle   1 box      2 can        3 cigarette_butt   4 food_waste
5 other    6 paper    7 plastic_bag   8 wrapper
```

`datasets/taco/yolo_seg`、`garbage_seg_merged_full1444`、`half_dataset_yolo`
三者的类别编号**完全一致**；源数据集 A（5 类）的映射关系为
`paper→6`、`plastic_bag→7`、`wrapper→8`。

---

## 环境

```bash
conda create -n zxj-py312 python=3.12
conda activate zxj-py312
pip install -r requirements.txt
```

实测环境：Python 3.12.15 / CUDA 13.0 / NVIDIA RTX 5090 (32 GB) / ultralytics 8.4.176。

---

## 关于路径

仓库内所有路径均为**相对路径**，可直接 clone 后使用：

- 4 个 `data.yaml` **未设置 `path` 字段**，ultralytics 会自动回退到
  「yaml 文件所在目录」解析 `train/val/test`，因此无需任何额外配置。
  若你要把图片放在别处，再自行补 `path:` 即可。
- 各脚本通过 `__file__` 自行推导仓库根目录，无硬编码绝对路径。

---

## 状态

| 环节 | 状态 |
|---|---|
| TACO → YOLO-Seg 转换 | 数据质量验收 **PASS**，`READY_FOR_STAGE1 = YES` |
| 合并数据集 `garbage_seg_merged_full1444` | **PROVISIONAL**，存在待人工复核项，**未通过数据质量验收** |
| TACO 9 类基线训练 | **已完成**（74 epochs，best epoch 59；仅验证集） |
| 合并数据集正式训练 | **尚未开始** |

`runs/segment/runs/smoke/yolov8m_seg/` 是**流程自检（smoke test）**：1 epoch、
batch 2、跑在 ultralytics 自带的 `coco8-seg` 示例集上，仅用于确认训练链路可跑通。
**其中的 mAP 等指标无任何实际意义，不构成项目成果。**

TACO 基线的实际验证结果记录在 `taco_baseline_v2/README.md`；该结果使用 114 张
TACO val 图像，未使用 test split，不代表跨域泛化或部署性能。

---

## 版权与许可

- **TACO**：官方仓库（[pedropro/TACO](https://github.com/pedropro/TACO)，MIT License）
  的代码与标注；原图来自 Flickr，版权归各原作者。完整许可见
  `datasets/taco/raw/release_metadata/LICENSE`。
- **本项目代码**：尚未指定开源许可证（待补 `LICENSE`）。
- **自采数据集**：版权归数据采集方，未在仓库中分发。
