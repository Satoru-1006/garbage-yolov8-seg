# garbage_seg_merged_dedup_v2 精确去重与标注质量审计报告

**数据集名称**：`garbage_seg_merged_dedup_v2`
**输出路径**：`/home/xmv/garbage_yolov8_seg/datasets/garbage_seg_merged_dedup_v2`
**上游数据集**：`datasets/garbage_seg_merged_full1444`（1444 张，状态 PROVISIONAL）
**生成时间**：2026-10-14
**状态标记**：**PROVISIONAL（去重与结构已通过验证；标签疑点已定位并显著减少，仍有 637 组待人工确认，未通过质量验收）**
**本轮范围**：仅数据处理与审计；**未启动任何模型训练**

---

## 1. 目标与达成

| 指标 | 目标 | 实际 | 结果 |
|---|---|---|---|
| train | 845 | **845** | 达成 |
| val | 145 | **145** | 达成（完全冻结） |
| test | 145 | **145** | 达成（完全冻结） |
| 合计 | 1135 | **1135** | 达成 |

```
datasets/garbage_seg_merged_dedup_v2/
├── data.yaml                (绝对路径 + 9 类)
├── images/train/   845      labels/train/   845
├── images/val/     145      labels/val/     145
└── images/test/    145      labels/test/    145
```

来源：`full1444` 的 train 1154 张按内容 SHA256 去重为 845 张（丢弃 309 个重复副本）；
val / test 从 `full1444` **逐字节复制**，未做任何去重与标签改动。

---

## 2. 精确去重

| 项目 | 数值 |
|---|---|
| full1444 train 图片 | 1154 |
| 唯一内容（SHA256） | **845** |
| 丢弃的重复副本 | **309** |
| train 内剩余重复组 | **0** |
| 内容组副本数分布（原始 train） | 1×574、2×246、3×14、4×10、6×1 |
| 内容组总数（全量） | 1063 = 845(train) + 111(val) + 107(test) |

去重口径：以 **full1444 自身文件内容的 SHA256** 判定（即训练实际见到的内容）。
去重只发生在 train；每个内容组保留 1 张，保留张即该组的“标签来源副本”。

---

## 3. 标注复核：343 / 282 / 24 组

`full1444` 审计标记的三类问题全部复核，数量与上游报告完全一致、可复现：

| 复核项 | 上游报告 | 本轮复核 | 一致性 |
|---|---|---|---|
| 重复内容组 | 343 | **343** | 一致 |
| A/B 标注冲突组 | 282 | **282** | 一致 |
| 源内同内容标注不一致组 | 24 | **24** | 一致 |
| 待复核清单（并集） | 295 | **295** | 一致 |

冲突严重度（A/B）：HIGH **141** / MEDIUM **124** / LOW **17**。

### 3.1 新增发现的系统性缺陷：空标签 vs 有标注（52 组）

复核中发现上游未单独分级的一类冲突：**同一内容组一侧为空标签（0 实例）、另一侧有标注**，共 **52 组**（train 38 / val 3 / test 11）。

- 上游策略“B 优先”把这 52 组中的 51 组（B 空、A 有标注）统一写成了**空标签**，即把**正样本降级为负样本（假阴性）**。
- 经抽样目视核查（叠加 A 侧多边形并查看原图）：这些画面中确实存在垃圾物体（垃圾桶内物品、地面纸片、包装袋、地面袋状物等），**B 侧为空属漏标**。
- 反方向（A 空、B 有标注）仅 **1 组**，方向高度单向，进一步支持“空标签侧为漏标”。

**本轮处置**：新增最高优先级规则“有标注优先于空标签”，把这 52 组改为采信有标注版本；全部登记为待人工复核（`EMPTY_VS_ANNOTATED`）。
**注意**：val / test 按指令完全冻结，**其中 14 组（val 3 + test 11）未被修正**，其假阴性标签仍留在冻结集中（见 §7）。

### 3.2 标签解析策略（确定性、可复现）

| 优先级 | 规则 | 触发组数 | 说明 |
|---|---|---|---|
| 0 | `prefer_nonempty` | **52** | 组内空/非空并存时采信有标注版本（§3.1） |
| 1 | `single_candidate` | 992 | 候选池内标签一致，直接采信 |
| 2 | `strict_superset` | **3** | 某候选实例集合为其余候选的严格超集 → 更完整 |
| 2 | `max_completeness` | **16** | 按 (类别数, 实例数, 面积) 取最大者 |
| — | `single_source` / 源偏好 | — | A 有 B 无或 B 有 A 无时按 0→2 在唯一池内解析 |

其余约定：
- 候选池默认取 **B（`half_dataset_yolo`，9 类完整体系）**；无 B 副本时取 A（`waste_ds_c3_neg_yolo`，5 类）。
  依据：A 仅具 5 类标注能力，对 `can`/`cigarette_butt`/`food_waste`/`other` **结构性漏标**。
- 并列时按 split 优先级 `train > val > test`，再按文件名升序，保证确定性。
- **同一内容组的全部副本共享同一份标签**（去重后仅剩 1 张，不存在同图两套标签）。

### 3.3 相对 full1444 的标签修正

| 项目 | 组数 |
|---|---|
| 解析标签与 full1444 已写标签不同的组 | **67** |
| 其中**实际写入** train 的修正 | **53** |
| 其中 val/test 组（按冻结要求**未改动**，仅登记） | 14（val 4 / test 10） |
| 修正类型：`prefer_nonempty`（空 → 有标注） | 51 |
| 修正类型：`max_completeness`（更完整标签替换任意选取的旧标签） | 13 |
| 修正类型：`strict_superset` | 3 |

train 的 53 组 = `prefer_nonempty` 38 + `max_completeness` 12 + `strict_superset` 3。

> 上游对源内不一致组采用“文件名排序取第一个副本”的任意规则；本轮改为“最完整”规则，
> 并新增空/非空规则，因此产生了这些标签修正。所有修正方向均为**增加标注完整性**。
> val/test 的 14 组差异**未写入**（冻结优先），作为 U2 问题登记。

---

## 4. 独立验证（35 / 36 项通过）

验证脚本：`tools/verify_dedup_v2.py`（独立于构建脚本重写，仅读取，不修改数据）。

| 维度 | 结果 |
|---|---|
| A 结构完整性：845/145/145=1135，图片-标签 1:1 配对 | PASS |
| B 合法性：图片全部可解码、类别 ID ∈ 0..8、polygon 点数/奇偶/坐标合法 | PASS（0 异常） |
| C 精确去重：train 内 SHA256 重复组 = 0，唯一内容 = 845 | PASS |
| D 泄漏：train/val、train/test、val/test 精确内容重叠 = 0；跨 split 内容组 = 0 | PASS |
| D 泄漏：dHash 近重复跨 split = 0 对；时间连拍跨 split = 0 对 | PASS |
| E 冻结：val / test 图片与标签与 full1444 **逐字节一致**（290/290） | PASS |
| F 只读：源 A/B 全部 1444 个文件 SHA256 与 manifest 一致；full1444 全程未被写入 | PASS |
| G 分布：9 类在 train 中均有实例 | PASS |
| G 分布：`can` 在 val 与 test 中实例数均为 0 → **不可评估** | **FAIL（1 项）** |
| H 标签完整性：保留标签 (类别数, 实例数) 均不劣于被丢弃副本 | PASS（0 组劣于） |
| I 会话级相关性（弱形式泄漏，仅报告） | 见 §6 |
| J data.yaml 路径与 9 类声明 | PASS |

**唯一失败项**为上游遗留的数据覆盖问题（`can` 在 val/test 无实例），非本次去重引入，
且因 val/test 冻结而**无法在本轮修复**（见 §7）。

---

## 5. 9 类实例分布

| split | imgs | bottle | box | can | cigarette_butt | food_waste | other | paper | plastic_bag | wrapper | 实例合计 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **train** | 845 | 57 | 376 | **6** | 71 | **11** | 700 | 436 | 463 | 139 | 2259 |
| val | 145 | 9 | 41 | **0** | 11 | 1 | 118 | 61 | 87 | 12 | 340 |
| test | 145 | 15 | 63 | **0** | 13 | 6 | 87 | 49 | 68 | 13 | 314 |
| 合计 | 1135 | 81 | 480 | 6 | 95 | 18 | 905 | 546 | 618 | 164 | 2913 |

对比 full1444（实例 3815 → 2913，去重后 −23.6%，与丢弃 26.8% 副本同量级）：

| 项目 | full1444 train | dedup_v2 train | 变化 |
|---|---|---|---|
| 图片 | 1154 | 845 | −309 |
| 实例 | 3161 | 2259 | −902 |
| 空标签（负样本） | 187 | **108** | −79（含 38 组空标签被修正为有标注） |
| `other` 占比 | 35.1% | **31.0%** | 主导性略有下降 |
| `can` | 9 | 6 | 仍极稀缺 |

**长尾问题依旧严重**：`other` 仍占 31.0%，前 4 类（other/plastic_bag/paper/box）占 **87.4%**；
`can`/`food_waste`/`bottle` 合计仅 **3.3%**（train 74 个实例）。

---

## 6. 数据泄漏评估

| 检查项 | 结果 | 判定 |
|---|---|---|
| 跨 split 的精确内容组（SHA256） | **0** | 无泄漏 |
| 跨 split 的 dHash 近重复对（Hamming ≤ 5） | **0** | 无泄漏 |
| 跨 split 的时间连拍对（间隔 ≤ 3 s） | **0** | 无泄漏 |
| val 到最近 train 图像的采集时间差 | 中位 **20.9 s**，<60 s 占 **71.7%** | **弱形式相关性仍在** |
| test 到最近 train 图像的采集时间差 | 中位 **24.7 s**，<60 s 占 **72.4%** | **弱形式相关性仍在** |

**结论**：内容级泄漏已彻底消除（三种判据均为 0）。
但 val/test 与 train 仍来自**同一批连续拍摄序列**（多数画面相隔数十秒），
即划分做到的是“内容不重复”，**不是“场景/会话独立”**。
含义：val/test 指标仍是**乐观上界**，不可外推为“新室内场景的性能”。
若需要无偏估计，需按采集会话重新划分——这会改变 val/test 内容，与本轮“val/test 完全冻结”要求冲突，故**本轮不做**，留作后续数据版本（v3）议题。

---

## 7. 未解决的标注问题（本轮无法定论，需人工）

**总计 637 / 1063 组未获人工确认**（train 514 组），构成系统性标签噪声。按类别分：

| 编号 | 问题 | 组数 | train | val | test | 处置 | 风险 |
|---|---|---|---|---|---|---|---|
| U1 | **A/B 标注冲突**：同一内容 A 与 B 标注不一致 | **282**（H141/M124/L17） | — | — | — | 按策略采信 B（9 类更完整），**未经人工确认** | 中~高 |
| U2 | **空标签 vs 有标注**：一侧 0 实例、一侧有标注 | **52** | 38 | **3** | **11** | train 已改为采信有标注；**val 3 + test 11 因冻结未修正** | **高（假阴性）** |
| U3 | **源内同内容副本标注不一致**：上游按文件名任取 | **24** | — | — | — | 改为“最完整”规则；**未经人工确认** | 中 |
| U4 | **A 独有内容组**（无 B 副本） | **344** | 284 | — | — | 采信 A 重映射标签；A 无 `can`/`cigarette_butt`/`food_waste`/`other` 标注能力 → **结构性漏标无法自动判定** | 中~高 |
| U5 | **A 含 B 不含的类别**（相同内容 A 标了 B 没标） | **150** | — | — | — | 采信 B → 这些类别实例**被丢弃**，需人工确认是否为 B 漏标 | 中 |
| U6 | **`can` 在 val/test 实例数 = 0** | 1 类 | — | 0 | 0 | **冻结不可修复**；该类别本轮及后续均不可评估 | 高（可评估性） |
| U7 | **会话级划分不独立**（弱形式泄漏） | — | — | 145 | 145 | 冻结不可修复；val/test 指标为乐观上界 | 中（评测可信度） |

### 优先人工复核顺序

1. **U2 的 val/test 14 组**（`dedup_v2_annotation_conflicts.csv` 中 `unresolved_reasons` 含 `EMPTY_VS_ANNOTATED` 且 `target_split` 为 val/test）：冻结集中已知的假阴性标签。
2. **U1 的 HIGH 141 组**：其中 `only_in_a` 非空者（A 标了 B 没标的类别）最需判定，共 150 组受 U5 影响。
3. **U4 的 344 组 A 独有内容**：需确认是否真的不存在 `can`/`cigarette_butt`/`food_waste`/`other`。
4. **U3 的 24 组源内不一致**：确认“最完整”规则选出的标签是否正确。
5. **U6/U7**：补采 `can` 样本 / 重建会话独立 test 集（属新数据版本，非本轮范围）。

---

## 8. 已知偏差与说明

1. **full1444 存在合并后重编码**：full1444 中有 **719 张图片**的修改时间晚于合并时刻，
   与源文件**逐字节不同**（尺寸一致，PSNR ≈ 60，属近似无损重编码，内容实质未变）。
   因此本轮去重口径统一采用 **full1444 自身内容的 SHA256**，而非源文件 SHA256；
   各副本的**原始源标签**通过 `tools/merge_manifest_full1444.csv` 回取源文件后重映射。
   该重编码不影响内容分组（去重后唯一内容数 1063 与上游记录一致）。
2. **源数据集与 full1444 全程只读**：未修改、未删除、未移动任何源文件；
   源 A/B 的 1444 个文件 SHA256 与 manifest 记录 **0 处不一致**。
3. **近重复与连拍只审查、不删除**：清单见 `tools/dedup_v2_near_dup_review.csv` /
   `tools/dedup_v2_burst_review.csv`（train 内 31 近重复对、219 个连拍组，跨 split 均为 0）。
   这些图片**全部保留**在数据集中。
4. **val/test 完全冻结**：290 个文件（图片+标签）与 full1444 逐字节一致，零改动。

---

## 9. 交付物

| 文件 | 说明 |
|---|---|
| `tools/dedup_v2_full1444.py` | 去重 + 标签复核构建脚本（`--analyze-only` / `--build`） |
| `tools/verify_dedup_v2.py` | 独立验证脚本（35/36 项，只读） |
| `tools/review_near_dup_v2.py` | 近重复/连拍审查清单生成（只审查不删除） |
| `tools/dedup_v2_stats.json` | 去重与标签解析统计 |
| `tools/dedup_v2_verification.json` | 独立验证结果（含 9 类分布、泄漏、会话相关性） |
| `tools/dedup_v2_manifest.csv` | 1135 条目标清单（split、SHA256、标签来源、解析规则、是否变更） |
| `tools/dedup_v2_review_duplicate_groups.csv` | **343** 条重复内容组复核记录 |
| `tools/dedup_v2_annotation_conflicts.csv` | **295** 条标注冲突/源内不一致复核记录 |
| `tools/dedup_v2_near_dup_review.csv` / `.json` | 近重复审查（106 条） |
| `tools/dedup_v2_burst_review.csv` | 连拍审查（959 条） |
| `tools/dedup_v2_audit.md` | 本报告 |

---

## 10. 验收结论

```
TRAIN_IMAGES                = 845    (目标 845)   PASS
VAL_IMAGES                  = 145    (目标 145, 冻结) PASS
TEST_IMAGES                 = 145    (目标 145, 冻结) PASS
TOTAL_IMAGES                = 1135                PASS
CLASSES                     = 9                   PASS
TRAIN_SHA256_DUPLICATES     = 0                   PASS
CROSS_SPLIT_CONTENT_LEAK    = 0                   PASS
CROSS_SPLIT_NEAR_DUP        = 0                   PASS
CROSS_SPLIT_BURST           = 0                   PASS
VAL_TEST_FROZEN_BYTE_EXACT  = PASS (290/290)
SOURCE_DATASETS_UNCHANGED   = PASS (1444/1444 SHA256 一致)
IMAGE_LABEL_PAIRING         = 1135 : 1135         PASS
INVALID_CLASS_ID            = 0                   PASS
INVALID_POLYGON             = 0                   PASS
CORRUPT_IMAGE               = 0                   PASS
LABEL_COMPLETENESS_NON_REGRESSIVE = PASS
SELF_CHECK                   = 35/36              (1 项失败: can 在 val/test 无实例)
UNRESOLVED_ANNOTATION_GROUPS = 637 / 1063         (train 514)
DATA_QUALITY_ACCEPTANCE      = PROVISIONAL         (未通过质量验收)
TRAINING_THIS_ROUND          = NO
```

### 一句话结论

去重与结构层面**完全达标**（1135 张、零重复、零跨 split 泄漏、val/test 逐字节冻结、
零非法标注），并把上游 295 项标签问题压缩到 **67 组已按“完整性”规则修正**
（其中 52 组是把被误降级为负样本的正样本恢复）；但**仍有 637 组标注问题未经人工确认**，
其中 val/test 冻结集中 14 组已知假阴性、`can` 类不可评估，故整体状态仍为 **PROVISIONAL**。

---

## 11. 复现方式

```bash
cd /home/xmv/garbage_yolov8_seg
source /home/xmv/miniconda3/etc/profile.d/conda.sh && conda activate zxj-py312

python tools/dedup_v2_full1444.py --analyze-only   # 只分析, 不写数据
python tools/dedup_v2_full1444.py --build          # 构建（输出目录已存在则拒绝覆盖）
python tools/verify_dedup_v2.py                    # 独立验证
python tools/review_near_dup_v2.py                 # 近重复/连拍审查清单
```

源数据集（`waste_ds_c3_neg_yolo`、`half_dataset_yolo`）与 `garbage_seg_merged_full1444`
在全部流程中保持只读。
