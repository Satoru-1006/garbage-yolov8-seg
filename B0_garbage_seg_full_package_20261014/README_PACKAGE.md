# B0 垃圾识别模型 · 完整审计包

实验: `B0_TACO_INDOOR_V2_800_2STAGE` (TACO 底座 -> 室内垃圾 dedup_v2, YOLOv8m-Seg, 两阶段)
打包时间: 2026-10-14
最终交付权重: `09_weights/phase2_best_DELIVERED.pt`
  SHA256 = 1e102051acdd840983240923e09438068d7ffa2f8e38a706cde1131c6e8e76d6

> 注意: 需求里写的 `runs/segment/Phase2/` 在本工作区**不存在**;
> B0 的真实产物目录是 `indoor_ab/B0_taco_indoor_v2_800_2stage/`,
> 本包已按其内容重组为下列可读结构。

## 目录索引

| 目录 | 内容 | 对应你索要的东西 |
|---|---|---|
| `01_reports/` | B0/A 实验报告、过拟合诊断、TACO 一期报告与审计、v2 去重审计 | 完整实验记录文档(Markdown 原文) |
| `02_eval_phase2_best/` | **正式交付**评估绘图: 归一化混淆矩阵、Mask/Box 的 PR/F1/P/R 曲线、val_batch0/1/2 的 labels-vs-pred | 核心: 验证集可视化与评估图表 |
| `03_eval_phase1_best/` | Phase 1 对照的同套图 | 对照 |
| `04_train_phase1/` | phase1 训练目录(不含 .pt): args.yaml, results.csv, results.png, labels.jpg, train/val batch 图, 混淆矩阵与曲线 | 训练配置与损失曲线(Phase 1) |
| `05_train_phase2/` | phase2 训练目录(不含 .pt), 同上 | 训练配置与损失曲线(Phase 2) |
| `06_metrics_smoke/` | eval_phase1/2_best.json, training_summary.json, smoke/ (冒烟结果+冻结张量清单+增强样张) | 逐类 AP / 尺寸分层 / 冻结核验 |
| `07_dataset_config/` | 三个 data.yaml(dedup_v2 / full1444 / taco_yolo_seg)、class_map_full1444.json、TACO 映射 v1/v2/candidate、TACO 转换/排除/校验 JSON、类别映射分析 | 数据集 yaml + 类别映射脚本产出的映射表 |
| `08_scripts/` | train_B0_2stage.py, eval_B0.py, smoke_B0.py, taco 转换/校验脚本, 合并与去重脚本, 混淆诊断脚本 | TACO/新数据集转 9 类的 Python 映射脚本 |
| `09_weights/` | phase1/phase2 的 best.pt + last.pt (共 4 个) | 模型本体 |
| `10_logs/` | B0 训练/评估完整日志 + TACO 底座与实验 A 的参考日志 | 训练日志 |
| `11_legacy_smoke_run_runs_segment/` | 早期 smoke 跑批 (`runs/segment/runs/smoke/yolov8m_seg`), 不含 .pt | 你提到的 runs/segment 路径实际指向此处 |

## 关键指标速查 (来自 01_reports/EXPERIMENT_B0_REPORT.md §5.4/§5.5/§5.6)

- Mask mAP50 = 0.2402 / mAP50-95 = 0.1356 (Phase 2 best, val 145 图 / 340 实例)
- 掩码召回@IoU0.5 = 0.5588; 0-16px 桶召回 = 0.5510
- 逐类 Mask AP50: bottle 0.2432 / **box 0.0273** / can 不可评估(val 0 实例) /
  cigarette_butt 0.2562 / food_waste 0.0000(val 仅 1 实例) / other 0.2368 /
  paper 0.4086 / plastic_bag 0.4277 / **wrapper P=0.1710, AP50 0.3218**

## box 类的映射真相 (来自 07_dataset_config/taco_conversion_summary.json)

初始化 B0 的 TACO 底座一期内 `box` 仅 **190 个实例**: train 120 / val 25 / test 45,
全部来自 Carton 系列 (Other carton 93 + Corrugated carton 64 + Meal carton 30 +
Egg carton 11 + Drink carton 45 + Pizza box 3, 经整图排除后实写 190)。
`box` 是 COCO 多边形 -> YOLO polygon 的正常转换, **不是 bbox 冒充 mask**。
室内 dedup_v2 中 `box` = 480 实例 (train 376 / val 41 / test 63)。

## 本包无法证实的数字

需求中提到 "新数据集中 box 有 123,790 个实例、占比 77.26%" ——
在 `/home/xmv/garbage_yolov8_seg` 全库检索**没有**任何数据集或脚本包含该数量级;
现有全部数据集的 `box` 实例数上限为 480(dedup_v2)。该数字应为上一轮对话中的
**未落地提案**或误记, 请确认其来源数据集后再纳入分析。
