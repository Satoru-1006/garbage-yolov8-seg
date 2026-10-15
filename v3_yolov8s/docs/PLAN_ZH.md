# V3 技术实施计划与验收清单（YOLOv8s-Seg）

## A. 已知事实 vs 待确认

**GitHub 上 B0 旧版已核实：** YOLOv8m-Seg、TACO 已训练底座、室内 `garbage_seg_merged_dedup_v2`（train 845 / val 145 / test 145）；B0 两个 Phase **都**使用室内训练集，P1 冻结 10 层 20 epochs，P2 全解冻实际 76 epochs；P2 Mask mAP50=0.2402 / mAP50-95=0.1356；纸箱 AP50=0.0273，`can` 在 val 中无实例。旧模型使用数据存在标注未充分人工确认和会话级相关性，不能视为可靠全新场景表现。

**用户已经告知：** 公共数据来源是 TACO、Can Instance Segmentation、Plastic SEG、已删减映射的 Waste-Segmentation、Cardboard Box 和 Cigarette Butts；室内照片正在接受 SAM3 二次核对。**不能把四个下载数据集截图里的“原始”图片数当作当前保留规模。**

**待确认（阻塞正式数据构建）：** 六个数据根目录、各自 `data.yaml` 实际类名、实际有效 train/val、是否 bbox-only、源图片精确/近似重复、映射后的语义、每份的使用许可、SAM3 人工审核签收、室内会话独立 val/test。

## B. 项目阶段

| 序号 | 工作包 | 必需交付物 | 通过后才能进入 |
|---|---|---|---|
| 0.1 | 公开来源盘点 | 每份原始路径/版本/许可证、`inspect` 输出、类目对照 | 公开构建 |
| 0.2 | 标签清洗/整图筛选 | `manifest.csv`, `audit.json`, 40+ 张多边形预览、近重复复核 | P1 smoke |
| 0.3 | SAM3 + 人工复核 | 修复记录、负样本/漏标复核、场景分组 train/val/test | P2 |
| 0.4 | 评价协议冻结 | 9 类定义、固定室内 val/test、B0 在新 val 重新评估 | 可比较模型 |
| 1 | Phase 1 公开训练 | `p1_public/weights/best.pt`、训练图、`v3_run_manifest.json` | P2 初始权重 |
| 2A | Phase 2A 纯室内训练 | `p2a_indoor/weights/best.pt` + 室内 val 评估 | 模型选择 |
| 2B | Phase 2B 混合回放训练 | `p2b_mixed/weights/best.pt` + 室内 val 评估 | 模型选择 |
| 3 | 室内真实推理/误差复核 | AP、Recall、混淆、薄弱类、部署速度 | 是否需要 P3 |
| 4 | 最终测试 | 已定权重 + test 一次性评估报告 | 发布候选 |

## C. 公共数据整图筛选的建议起点

| 来源 | 训练图片目标上限 | 理由 |
|---|---:|---|
| TACO | 800 | 丰富类别和垃圾形态；9类映射已曾实现，但须复核本次版本 |
| Can Instance Segmentation | 700 | 补强极少的 can |
| Plastic SEG | 700 | 重点 bottle、plastic_bag；模糊类别不得乱归 other |
| Waste-Segmentation（**已删减版本**） | 900 | 保留有用的 bottle/can/wrapper；控制纯 other 图像重复 |
| Cardboard Box | 650 | 有针对性地补强 box |
| Cigarette Butts | 250 | 小目标烟头 |
| 合计 | **约4000** | 起始实验预算，不是强行凑数 |

`quota_train` 是来源整图数量上限；不足就以实际为准，质量差就少用。不预设硬性“other 20%”合格线；输出中同时关注实例分布、包含该类图片占比、真实室内验证表现。根据分析改变配比时，保存新的配置版本/commit，不修改已完成的实验清单。

**数据核验细节：**
- 每个源的 `data.yaml names` 是源类别 ID 的唯一依据；需要实际读取原始 `.txt` 中的 class ID 完成映射。`v3_dataset.py` 自动重写目标 ID、保留多边形点；无法确认的源类立即报错。
- `other` 不代表“未知/没标注”，必须是明确且可辨的其他垃圾。仅为实现数量配比丢掉标签会产生负样本污染。
- 按 SHA256 保证精确去重、相同图片不同标签拒绝；dHash 列潜在跨 split 近重复，必须人工核对，同时检查连续拍摄会话关联。
- 重复不同数据源会造成“重复学习”以及 train/val 泄漏；**尤其 TACO 与 Waste 的交集**要检查。
- 样本内容可以是多类别，一张图同时有 can 与 other 时应完整保留。

## D. Phase 1 配置与判断

- 架构 YOLOv8s-Seg，初始化 `yolov8s-seg.pt`（COCO 预训练）；与 B0 的 m-seg 权重不同、不能直接沿用。
- P1 数据只含公开 train/val，无室内图片；保留旧 B0，不覆盖旧数据。
- epochs ≤100、patience=20、imgsz=800、batch=16、AdamW、lr0=1e-3、weight_decay=5e-4、cos_lr、freeze=0、mosaic=.3、copy_paste=0、mask_ratio=2；先 smoke 3 epochs。
- P1 `best.pt` 由**公开 val**选择；额外使用室内 val 测“未适配室内”的跨域性能，**不得按室内 val 反复挑公开图片做 P1**。
- P1 若数据含难检物很少，如 food_waste、can 等，检验按类 AP 和样本数量而非只看 mAP 宏平均。

## E. Phase 2 配置与判断

- 从**同一 P1 best.pt**分别启动两次独立作业。
- P2A：SAM3 + 人工确认的真实室内 train，约 40 epochs，patience=12，lr0=1e-4，mosaic=.1，copy_paste=0。
- P2B：同上参数，train = **全部室内 train + 从公开 train 中抽约 25% 图像份额**，强调 can/bottle/box。由 `build-mixed` 构建新目录，val/test 复制**相同室内**集合。例：若室内 train=845，目标额外抽 `round(845*.25/.75)=282` 张公开训练图。
- P2B 由于增加训练图片，每轮迭代数可能比 P2A 多；两者属于**可执行的初步对照**，若差异接近则按相同总梯度更新次数进一步复验，不要把不公平比较解释为因果结论。
- 两组均在同一室内 val 用统一推理协议比较，禁止触碰 test。

## F. 评价协议与选型

固定：`imgsz=800`, `conf=0.001`（计算 PR/mAP 时使用低阈值）, `iou=0.7`, `rect=True`, `max_det=300`, `batch=16`，`split=val`。`v3_evaluate.py` 会写 JSON 和分割评价图；缺少真值的类按 null 标识，不得声称 AP=0。

最低决策看板：Mask mAP50 / Mask mAP50-95 / Mask Recall / Box mAP50 / per-class Mask AP / 极小目标召回 / 同场景实际机器人视频上的 FP/FN / 目标边缘质量 / 推理延迟（在实际部署设备测，不用 RTX 5090 训练卡代替）。

**B0 必须先在同一份 V3 室内 val 复评。** 若新 val 含标注问题或与 train 高度相关，先修复/改划分再正式比较；不同验证集指标不可比较。

**不承诺精度门槛**。第一阶段输出可训练可复现实验，第二阶段判断 V3 是否真正优于 B0；若所有 V3 的 mAP 仍低，优先诊断标签/对象尺度/类别定义/负样本与真实场景，不要盲目增加 epoch。

## G. Phase 3（暂不执行）

只有真实室内复评后，才能决定针对 box（旧 AP50=2.73%）、bottle、can、烟头、反光/透明物的补样策略。若新增数据，应单独建 V3.1，保留固定 test 不参与挑参。支持主动学习：现场误检/漏检 → SAM3 候选 → 人工签收 → 新版 train → 小学习率重训 → 冻结 val 对照。

## H. 工程记录建议

存档：Git commit、9类映射说明、源下载版本、许可审核、SAM3 人工审核签收、训练环境（CUDA/PyTorch/Ultralytics）、数据 manifest SHA、起点权重 SHA、训练参数、每一阶段 best/last、Mask/Box 评价 JSON、按类与小目标统计、真实机器人录像中的典型 FP/FN。不将内部图片/公司数据提交到 GitHub。