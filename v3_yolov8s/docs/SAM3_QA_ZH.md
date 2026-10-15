# SAM 3 二次核对 · 真实室内垃圾标注质量 SOP

## 核心原则

SAM 3 是辅助生成“可能更好的掩码”，**不是直接替换的真值标签**。要复核三个维度：**类别是否正确、该标的实例是否全部标出、多边形是否贴合垃圾**。高 IoU 不表示正确类别，自动匹配 bbox 可能截断薄塑料袋和透明瓶轮廓。

## 1. 先冻结来源和划分

1. 导出原始室内图片+标签文件清单及 SHA256，保持原件只读。
2. 根据房间、连续拍摄会话/机器人工作任务、日期划分 train/val/test，禁止类似帧随机拆分。否则容易被同场景相关性误导。
3. 使用原 B0 清单和去重历史核对：不要把 full1444 的309个训练重复副本又加回 V3。
4. 让 `can`、`bottle`、`box` 在真实 val/test 上有可解释的真值数量；若无法做到，不要报告其可靠的 AP。

## 2. 对每张图片应完成的人工复核

| 核对点 | 标准 | 错误示例 |
|---|---|---|
| 实例完整性 | 地面上每件属于目标9类的垃圾均有标签 | 两个塑料瓶只标一个 |
| 类别定义 | class ID 固定 0-8，`other` 不吞并明确类别 | 易拉罐标 other |
| 轮廓质量 | 边缘基本贴合；不包含过大地面、影子/高光 | 透明瓶整片地面被分割 |
| 小目标 | 烟头、包装碎片不能被自动过滤 | 极小物体漏标 |
| 空标签 | 场景确实无目标，才能用空 txt | 空标签图片有纸箱 |
| 特殊物理状态 | 横放瓶、压扁罐、撕裂纸、褶皱包装袋 | 一团塑料袋只标一部分 |

建议每张修订保留审核 CSV：image_id、original_label_sha、SAM3_suggestion_sha、class_changed、missing_added、polygon_repaired、approved_by、approved_at、comment。即使使用自动工具批处理，也必须对高差异图像全部审核，并对低差异图像抽查；验证/测试集应逐张人工确认。

## 3. 数据验收闸门

- 新 train 质量：抽样审核 + 所有冲突/漏标候选处置；不能仅凭脚本格式 PASS。
- val/test：100%人工核对、场景/会话独立性核验。旧 B0 的 val/test 已有14组已知空标注冲突和相似场景相关性，不能不审就照搬。
- 输出路径必须含：`images/train`, `images/val`, `images/test`, `labels/{train,val,test}`, `data.yaml`；所有标签为 YOLO-Seg 多边形。
- 不使用 SAM3 候选的临时路径作为 `indoor.root`，只有人工签收后才能 `review_approved: true`。
- `v3_dataset.py build-indoor` 复制出独立训练版。原始数据不修改、不覆盖。
- 将审核说明、场景 split 签收文件保存在内部受控位置，不将客户/公司图片上传公开 GitHub。

## 4. SAM3工具包关系

本 GitHub 仓库此前已有 `tools/sam3_indoor_trash_toolkit.zip`。该包生成候选轮廓与疑似漏标，不自动将文本提示识别当成正确标签。建议输出建议区与签收数据区分开：

```
original_indoor/           # 只读原件
sam3_candidates/           # 提议，不参加训练
human_reviewed/           # 已确认的 train/val/test
v3_indoor_clean/           # v3_dataset.py build-indoor 的版本化复制
```

另请核对 SAM3/Ultralytics 模型授权条款是否覆盖计划中的商业部署用途。