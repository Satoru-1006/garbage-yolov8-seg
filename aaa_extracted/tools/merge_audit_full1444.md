# YOLOv8-Seg 全量合并数据集审计报告

**数据集名称**：`garbage_seg_merged_full1444`
**输出路径**：`datasets/garbage_seg_merged_full1444`
**生成时间**：2026-10-13
**状态标记**：**PROVISIONAL（临时交付，存在待人工复核项，未通过数据质量验收）**

---

## 1. 源数据集

| 数据集 | 路径 | 图片数 | 类别数 |
|---|---|---|---|
| A | `datasets/waste_ds_c3_neg_yolo` | 694 | 5 |
| B | `datasets/half_dataset_yolo` | 750 | 9 |

两数据集**只读**使用：合并全程未修改、未删除、未移动任何源文件。
验证方式：对 manifest 中全部 1444 条源路径重新计算 SHA256 与记录比对，**缺失或被改动 = 0**。

---

## 2. 类别重映射（CLASS_REMAP）

目标体系 = 数据集 B 的 9 类，**B 的类别 ID 全部保持不变**。

A 的映射：

| A 原 ID | 类别名 | → 新 ID |
|---|---|---|
| 0 | bottle | 0 |
| 1 | box | 1 |
| 2 | paper | **6** |
| 3 | plastic_bag | **7** |
| 4 | wrapper | **8** |

合并后 9 类：

```
0 bottle  1 box  2 can  3 cigarette_butt  4 food_waste
5 other   6 paper  7 plastic_bag  8 wrapper
```

**验证结果**：全量 1444 条逐条比对最终写出的标签与期望重映射结果，**不一致 = 0**；
标签内非法类别 ID（<0 或 >8）= **0**。

---

## 3. 合并结果

```
garbage_seg_merged_full1444/         814 MB
├── data.yaml                        (绝对路径 + 9 类)
├── images/train/  1154
├── images/val/     145
├── images/test/    145
├── labels/train/  1154
├── labels/val/     145
└── labels/test/    145
```

| 项目 | 数值 |
|---|---|
| 图片总数 | **1444** |
| 标签总数 | **1444**（其中合法空标签 242，作负样本） |
| 重复图片 | **全部保留**（未去重、未删除任何样本） |
| 文件命名 | 统一加前缀 `A_` / `B_`，避免同名覆盖 |
| 复制方式 | 独立文件复制（`shutil.copy2`），非硬链接／软链接 |

来源构成：A 694 张 + B 750 张 = 1444 张。

---

## 4. split 协调

### 协调依据

先构造「独立单位」：同一 **SHA256 内容组** ∪ **dHash 近重复组**（8×8 dHash，Hamming ≤ 5）∪ **时间连拍组**（文件名时间戳间隔 ≤ 3 秒）。

同一独立单位内的全部图片强制进入同一个 split，从根源消除跨 split 泄漏。

- 独立单位数：**423**（含多内容组的单位 251）
- 连拍合并边：**969**
- 近重复对：**33**

### 分配规则

1. 单位内部原始 split 一致 → **保留原划分**（582 张）
2. 单位内部 split 冲突 → 按 80/10/10 目标**贪心补位**（862 张，涉及 150 个单位）

### 实际划分（独立性优先，未为凑比例拆散任何单位）

| split | 图片数 | 占比 | 目标 |
|---|---|---|---|
| train | 1154 | **79.9%** | 80% |
| val | 145 | **10.0%** | 10% |
| test | 145 | **10.0%** | 10% |

### 泄漏检查

| 检查项 | 结果 |
|---|---|
| 跨 split 的 SHA256 内容组 | **0** |
| 跨 split 的独立单位（含近重复、连拍） | **0** |
| 合并后跨 split 的唯一内容 | **0** |

> 注：协调前，源数据中存在 151 个连拍组跨 split（涉及 723 张图），已全部纠正。

---

## 5. 重复样本与标注冲突

### 重复情况

| 项目 | 数值 |
|---|---|
| 唯一内容（SHA256） | 1063 |
| 含重复副本的内容组 | 343 |
| 同一内容的最大副本数 | — |
| 保留的副本总数 | 1444（= 全部图片） |

### 标注冲突

同一内容组在 A / B 中标注不一致者：**282 组**（占共有内容组 323 的 87%）。

严重度分级：

| 级别 | 组数 | 判定条件 |
|---|---|---|
| HIGH | 141 | 实例数差 ≥ 2，或类别集合差异 ≥ 2 |
| MEDIUM | 124 | 实例数差 = 1，或存在单类别差异 |
| LOW | 17 | 实例数相同但多边形坐标不同 |

另有**同一数据集内部同内容副本标注不一致**：**24 组**（源数据自身问题）。

### 协调策略

1. 以 **B 的 9 类标注为候选基准**（B 为 9 类体系、类别更完整）
2. 基准副本确定性选取：`B 优先 → split 优先级 train>val>test → 文件名排序`
3. **同一内容组的全部副本写入同一份协调后标签**，杜绝同一图片两套矛盾标签进入训练
4. 全部 295 个问题组写入待复核清单，**不视为已通过质量验收**

标签来源最终分布：

| 来源 | 图片数 |
|---|---|
| B 的 9 类原始标签 | 1096 |
| A 标签重映射后 | 348 |

---

## 6. 标注完整性审计

### 6.1 A 疑似漏标（A 仅 5 类导致）

323 个共有内容组中，**178 组**存在「B 标注了而 A 未标注」的类别：

| 疑似漏标类别 | 涉及组数 |
|---|---|
| other | 110 |
| plastic_bag | 56 |
| paper | 21 |
| cigarette_butt | 19 |
| wrapper | 17 |
| box | 9 |
| bottle | 5 |
| food_waste | 5 |
| can | 3 |

> 由于 B 已是类别更完整的版本且被选为基准，这些位置在合并结果中**采信 B 的标注**。但A侧漏标事实已记录在案，需人工确认。

### 6.2 各 split 类别实例分布

| split | imgs | bottle | box | can | cigarette_butt | food_waste | other | paper | plastic_bag | wrapper |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 1154 | 75 | 480 | 9 | 97 | 19 | 1108 | 564 | 619 | 190 |
| val | 145 | 9 | 41 | **0** | 11 | 1 | 118 | 61 | 87 | 12 |
| test | 145 | 15 | 63 | **0** | 13 | 6 | 87 | 49 | 68 | 13 |

### 6.3 稀缺类别与空标签

| 类别 | 实例数 | 备注 |
|---|---|---|
| can | **9** | 极度稀缺；**val / test 中均为 0，无法验证该类性能** |
| food_waste | **26** | 严重稀缺；val 仅 1、test 仅 6 |
| bottle | 99 | 偏少 |
| other | **1313** | 占全部实例 32.9%，存在主导性偏差 |

空标签（负样本）：train 187 / val 24 / test 31，合计 **242**。

### 6.4 格式合法性

| 检查项 | 结果 |
|---|---|
| 非法类别 ID | 0 |
| 非法 polygon（点数不足/奇数/坐标越界） | **0** |
| 损坏图片（全量 PIL 校验） | **0** |
| 图片与标签配对 | 1444 : 1444，无缺失 |

---

## 7. Ultralytics 解析验证

```
RESOLVE = PASS
nc = 9
train : .../images/train  exists=True
val   : .../images/val    exists=True
test  : .../images/test   exists=True
```

`data.yaml` 使用服务器绝对路径，可直接用于 `yolo segment train`。

---

## 8. 交付物清单

| 文件 | 说明 |
|---|---|
| `tools/merge_full1444.py` | 合并脚本（可重复执行，`--analyze-only` / `--merge`） |
| `tools/class_map_full1444.json` | 类别映射与策略配置 |
| `tools/merge_manifest_full1444.csv` | 1444 条源→目标映射清单（含 SHA256、split 变更、标签来源） |
| `tools/annotation_conflicts_full1444.csv` | 295 条待人工复核记录 |
| `tools/merge_stats_full1444.json` | 合并统计 |
| `tools/merge_audit_full1444.md` | 本报告 |

---

## 9. 验收结论

```
TOTAL_IMAGES               = 1444
TOTAL_LABELS               = 1444
CLASSES                    = 9
DUPLICATE_IMAGES_RETAINED  = YES
CLASS_REMAP                = PASS
SPLIT_LEAKAGE              = PASS
ANNOTATION_CONFLICTS       = UNRESOLVED   (295 项待人工复核)
MERGE_STATUS               = PASS
READY_FOR_STAGE2           = YES
DATA_QUALITY_ACCEPTANCE    = PROVISIONAL  (未通过质量验收)
```

### 说明

- **合并本身完全成功**：1444 张全部保留、类别映射零错误、零跨 split 泄漏、零损坏图像、零非法 polygon。
- **但数据质量未通过验收**，必须标记为 PROVISIONAL：
  - 282 组 A/B 标注冲突，其中 265 组为 HIGH / MEDIUM，需人工判定
  - 24 组源数据内部同内容副本标注不一致
  - 178 组存在 A 侧疑似漏标
  - `can` 类别在 val / test 中实例数为 0，该类别不可评估

### 进入第二阶段（TACO 迁移训练）前建议处理

1. 人工复核 `annotation_conflicts_full1444.csv` 中的 HIGH 级别 141 组
2. 补充 `can` 与 `food_waste` 样本，或在类别定义上合并
3. 评估 `other` 类占比 32.9% 带来的偏差，必要时重采样
4. 决定是否保留 A 侧 348 张独有图片（均为 5 类体系，缺少 can / cigarette_butt / food_waste / other 标注）

---

## 10. 复现方式

```bash
cd <repo-root>
source $(conda info --base)/etc/profile.d/conda.sh && conda activate zxj-py312
python tools/merge_full1444.py --analyze-only   # 只分析
python tools/merge_full1444.py --merge          # 执行合并（输出目录已存在时拒绝覆盖）
```

源数据集在合并过程中始终保持只读。
