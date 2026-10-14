# garbage_seg_merged_v3 全面质量验收

生成日期：2026-10-14

本轮未训练模型。v1/v2 与源 A/B 均为只读输入。

## 结构与格式

- 图片：train/val/test = 850/106/107；总计 1063。
- 图片-标签配对：PASS。
- 图片解码：PASS；非法标签记录：0。
- 全集精确内容重复：0 组；跨 split dHash 近重复：0 对。

## 类别覆盖

| split | bottle | box | can | cigarette_butt | food_waste | other | paper | plastic_bag | wrapper |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 55 | 275 | 4 | 70 | 12 | 706 | 412 | 433 | 128 |
| val | 5 | 132 | 1 | 9 | 2 | 69 | 37 | 56 | 9 |
| test | 9 | 15 | 1 | 8 | 2 | 72 | 60 | 67 | 17 |

- `can`：val=1、test=1；已具备最低可评估入口，但样本仍极少。
- `food_waste`：val=2、test=2；仍属于低样本类别。

## 标注质量门禁

- v2 637 组待审问题在 v3 中全部保留审计记录；当前 `human_required=true`：637 组。
- HIGH 优先清单：193 组（本地源 SHA 分组重算；与 v2 的 141 组存在口径差异，不能静默合并）。
- 本轮明确记录的 AI 处理：{'USE_NONEMPTY': 1, 'KEEP_EMPTY': 4}；均保留人工确认边界。
- 14 组 val/test 空/非空问题：4 组清空明显海报/标识假阳性，1 组保留明显垃圾桶场景，9 组待人工确认。

## 会话划分

v3 使用 exact SHA、dHash≤5、3 秒连拍、以及 60 秒时间相邻启发式构造不可拆分单位。
跨 split 60 秒时间邻近对：0；该结果仅支持有限的时间相关性检查，不是场景独立性证明。

## 最终判定

```text
STRUCTURE_AND_SYNTAX = PASS
CROSS_SPLIT_CONTENT  = PASS
CROSS_SPLIT_NEAR_DUP = PASS
SESSION_60S_CHECK    = PASS
ANNOTATION_ADMISSION = BLOCKED_UNRESOLVED_HUMAN_REVIEW
TRAINING_THIS_ROUND  = NOT_RUN
V3_TRAINING_READY    = NO
```
