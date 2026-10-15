# 垃圾识别 V3 · YOLOv8s-Seg 两阶段训练与审计工具包

**项目：** 清洁机器人室内地面垃圾实例分割（9 类）。  
**状态：** 可执行的训练/数据处理工程模板，**尚未在公司 RTX 5090、用户新增公开数据和 SAM 3 清洗后的实际数据上运行**。  
**约束：** 不覆盖 B0、TACO 基线、原始图片、原始标签；不在本仓库上传数据集、原始图片、SAM 3 模型权重、训练权重或内部路径中的敏感文件。

## 1. 方案一览

| 项目 | Phase 1 公共域基础模型 | Phase 2A 室内微调 | Phase 2B 室内+公开回放 |
|---|---|---|---|
| 架构 | **YOLOv8s-Seg** | YOLOv8s-Seg | YOLOv8s-Seg |
| 起点 | 官方 `yolov8s-seg.pt` | P1 `best.pt` | **同一个** P1 `best.pt` |
| 训练数据 | TACO、Can Instance Segmentation、Plastic SEG、已清洗 Waste-Segmentation、Cardboard Box、Cigarette Butts | SAM 3 辅助且人工复核的**室内 train** | 室内 train + 约 25% 高质量公开 train |
| 验证集 | **公开 val**（模型选权重） | **相同的真实室内 val** | **相同的真实室内 val** |
| 训练上限 | 100 epochs | 40 epochs | 40 epochs |
| 学习率 `lr0` | 0.001 | 0.0001 | 0.0001 |
| `freeze` | 0 | 0 | 0 |
| 分辨率 / batch | 800 / 16 | 800 / 16 | 800 / 16 |
| 输出目录 | `p1_public/` | `p2a_indoor/` | `p2b_mixed/` |

**训练作业：3 次**（P1、P2A、P2B）；另有一次 3-epoch 冒烟测试、数据审核、固定协议评估。可选 Phase 3（针对薄弱类、主动学习）**暂不执行**，由 P2 的错误分析决定。

> 关键概念：P1 不使用待清洗的真实室内**训练**图像；但 P1 结束后须在锁定的室内 val 上测一次跨域表现。P2A/B 从同一个 P1 best 开始，不能把 P2B 接在 P2A 后面。

## 2. 为什么这样设计？

旧 B0 为 YOLOv8**m**-Seg，TACO 预训练底座 → 在同一份去重后室内 train 上 freeze=10 微调 → 全解冻。报告中的旧 val Mask mAP50=0.2402，Mask mAP50-95=0.1356；训练集 845 图 / 验证集 145 图，`can` 验证 0 实例，且有未人工确认/漏标/相似采集会话问题。因此 **不能直接将 24.02% 与新数据/新验证协议的成绩相减**。

新 V3 固定结构为 YOLOv8**s**-Seg，与旧 B0 **模型大小也不同**，对比能展示最终方案差异，但**不能单独归因**为某个数据集或 SAM 3 的效果。必要时再做固定起点、固定 val 的消融实验。

### 9 类（永久固定顺序）

```
0 bottle       1 box           2 can
3 cigarette_butt 4 food_waste  5 other
6 paper        7 plastic_bag   8 wrapper
```

`other` 不通过删除图片中的真实标签来“压制”，而通过**整图筛选**和定向补齐其它类。`other` 不能作为未知类别的默认兜底（否则产生语义污染）。bottle 是否包含玻璃瓶、box 是否包含折叠包装盒等都需统一写入人工标注规范。

## 3. 下载、安装和操作

详见 **[docs/RUNBOOK_ZH.md](docs/RUNBOOK_ZH.md)**（命令可以直接复制，完成路径/许可/映射/数据审计后运行）。

```
configs/sources.example.yaml         # 数据来源模板：默认禁止导入
scripts/v3_dataset.py                # 来源检查、类别重映射、审计、整图筛选、去重、构建数据集、人工复核预览
scripts/v3_train.py                  # smoke / p1 / p2a / p2b 四种作业；拒绝覆盖
scripts/v3_evaluate.py               # 固定协议公开模型/室内模型/B0 的 val 或最终锁定 test 评估
scripts/v3_compare.py                # 固定 val 身份与评价协议的多模型对比 CSV
docs/PLAN_ZH.md                     # 完整项目计划、阶段门槛、分布策略、验收指标
docs/SAM3_QA_ZH.md                  # SAM 3 人工复核、漏标、空标签、验证/测试集守则
docs/RUNBOOK_ZH.md                  # RTX5090 终端逐条命令、问题排查
tests/test_pipeline.py              # 本地无 GPU 数据逻辑单元测试
```

## 4. 严格的执行顺序

1. 备份旧模型/标签，记录 B0 权重 SHA256 和 Git 版本；**不要训练到旧 B0 的输出目录**。
2. 检查六个公开数据来源的真实 `data.yaml`、许可、图片、各类别映射及标注完整性；你已清洗的 Waste-Segmentation 应使用**清洗后版本**，不使用原始版本的图片/实例统计制定配比。
3. 运行 `inspect`，填好 `configs/sources.local.yaml`，确认每个来源的许可和类别；仅将准备好的来源设为 `enabled: true`。
4. `build-public` 创建 P1 公开 train/val，运行 `preview` 人工检查，并核对 `audit.json`、`manifest.csv`、`near_duplicate_review.csv`。若有跨 split 精确重复/类别未映射/错误标签，脚本会阻止构建。
5. SAM 3 处理真实室内**原始 train**，人工逐张复核相应候选；人工确认 val/test 的**全部标签与场景独立性**，形成 `v3_indoor_sam3_reviewed`。**先按拍摄会话/房间划分，再审核并冻结。**
6. 标记 `review_approved: true` 后 `build-indoor` 复制成新的只读审核版；运行 `guard`，排查跨公开/室内精确重复与近似重复候选。
7. 先跑 `smoke` 3 epochs，检查训练 batch 可视化、GPU、路径、YOLO-Seg 多边形；再跑 P1 100 上限，保存 P1 `best.pt`。
8. 在**相同室内 val**上评价旧 B0 和 P1（仅对比，不用 test）；记录两者的逐类评价和小目标漏检案例。
9. 跑 P2A；用 `build-mixed --public-fraction 0.25` 生成 P2B 数据；跑 P2B。注意 P2B 的每个 epoch 图像量略多，并非完全严格的计算预算消融；若要严格比较，可固定总优化步数进一步复试。
10. 用 `v3_evaluate.py` 在相同室内 val 评价 P2A/P2B，`v3_compare.py` 输出对比；检查 `can`/`food_waste` 是否有足够代表性的 GT。
11. 只在**选定最终模型、固定后处理阈值**后，使用 `--unlock-test` 对**锁定 test**评估一次；考虑实际机器人摄像头误检/漏检、推理延迟与可靠性。
12. 根据薄弱类别决定是否执行可选 P3；若引入新标注或新类别，版本升为 V3.1，重新冻结数据/验证协议。

## 5. 数据质量与许可闸门

- **人工把关不可跳过。** SAM 3 输出是候选，不是 ground truth；分割 IoU 变大不证明类别正确。
- **分组划分。** 真实室内同一次连续拍摄/同一房间近邻图片不得简单随机分到 train/val/test。dHash 仅能列出可疑近重复，不能证明所有泄漏都被排除。
- **禁止静默忽略标签。** 类别名未映射、label 丢失、出现 bbox-only 检测行、无效多边形、跨 split 精确重复时中止。
- **不盲目删 `other` 行。** `other_only_keep_rate` 只决定是否保留**纯 other 的整张图片**；多类图片中的所有合法标签完整保留。
- **许可**由使用者确认：在 `sources.local.yaml` 手工改 `license_ok: true`；Roboflow 上不同项目、不同版本的许可可能不同。
- **负样本。** 空的 `.txt` 必须人工确认场景中确实无9类目标；不能把错删标签的图片当成背景。
- **测试集禁用于调参、早停和 P1/P2 模型选择。** 不要向 GitHub 提交真实图片/标注/模型权重/内部日志。

## 6. 代码实现边界

工具只支持标准 YOLO-Seg 分割格式以及 `data.yaml` 指向的**图片目录**；可识别常见的 `train/images,valid/images` 和 `images/train,images/val` 布局。对于 COCO JSON、检测框、非标准 RLE、多边形 mask PNG 等格式，**先转换再运行**。新下载的四份 Roboflow 数据是否真是实例分割格式，必须本机审计确认。

`v3_dataset.py` 检测精确 SHA 去重、dHash 候选、类别映射、坏图（有限解码检查）、多边形点数/取值范围，**无法替代语义审核**。执行结果放在服务器、不会自动回写 GitHub。训练所需权重由 Ultralytics 自动取得或用户自行放置；本包不包括任何训练好的 V3 模型。

### 官方文档

- [Ultralytics：YOLO 实例分割数据格式](https://docs.ultralytics.com/datasets/segment/)
- [Ultralytics：训练参数](https://docs.ultralytics.com/modes/train/)
- [Ultralytics：实例分割](https://docs.ultralytics.com/tasks/segment/)
- [Ultralytics：数据增强](https://docs.ultralytics.com/guides/yolo-data-augmentation/)
- [Meta：SAM3](https://github.com/facebookresearch/sam3)