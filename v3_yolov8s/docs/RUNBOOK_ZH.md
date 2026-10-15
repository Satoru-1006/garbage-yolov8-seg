# RTX 5090 / Linux 逐步运行手册

以下命令为**待在公司 GPU 服务器运行**，不是已经执行的日志。首次必须核对实际数据路径、权重、许可证和类别映射。不要覆盖 `/home/xmv/garbage_yolov8_seg` 旧版目录。

## 0. 获取新版源码与准备环境

```bash
# GitHub 私有仓库要求已配置 SSH 密钥（或使用自己的合法认证方式）。
mkdir -p /home/xmv/projects
cd /home/xmv/projects
git clone git@github.com:Satoru-1006/garbage-yolov8-seg.git
cd garbage-yolov8-seg/v3_yolov8s

# 优先复用已配置正确 PyTorch/CUDA 的环境，不要盲目覆盖公司 PyTorch。
conda activate zxj-py312
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
nvidia-smi
python -c 'import ultralytics; print(ultralytics.__version__)'
python -m pip install -r requirements.txt
pytest -q tests/
```

旧仓库可能已经在 `/home/xmv/garbage_yolov8_seg` 的 Git 工作树中；如已 clone，**无需在同一目录重复 clone**。可按项目管理实际需要调整项目源码所在位置。

## 1. 填写来源配置、盘点原始数据

```bash
cp configs/sources.example.yaml configs/sources.local.yaml
nano configs/sources.local.yaml
python scripts/v3_dataset.py inspect --config configs/sources.local.yaml
```

在 `sources.local.yaml` 手动填写**实际绝对路径**、每份数据集的**真实原始类名映射**、有效版本和许可。确认之后才设置 `enabled: true`, `license_ok: true`。比如 source `data.yaml` 为 `names: ['Cardboard Boxes']` 时，映射必须写 `Cardboard Boxes: box`，**不能仅假设原始 ID=1**。对已经映射9类的 Waste 数据，仅当 `names` 与目标语义完全对应时可设置 `mapping: {}`。如果某公开源没有 val 子集，应先按原场景/来源手动建立 val，不得把 train 随机照片混入来假装独立验证集。

## 2. 建公开 P1 数据集

```bash
ROOT=/home/xmv/garbage_yolov8_seg
python scripts/v3_dataset.py build-public \
  --config configs/sources.local.yaml \
  --out "$ROOT/datasets/v3_public"

python scripts/v3_dataset.py preview \
  --dataset "$ROOT/datasets/v3_public" --split train \
  --out "$ROOT/experiments/v3_yolov8s/previews_public" --count 50
```

人工检查 `datasets/v3_public/audit.json`、`manifest.csv`、`near_duplicate_review.csv` 及预览图。有 `other` 语义混乱、漏标、误映射、近似重复就修复**来源或配置**，并以**新的数据版本目录**重建。不要在既有版本目录就地覆写。

## 3. 室内 SAM 3 人工签收（可与步骤2并行）

依照 `docs/SAM3_QA_ZH.md` 完成真实室内图像、labels/train/val/test 及 `data.yaml`。此时的数据目录应该是**人工审核后的结果**，而不是 SAM3 `pending_labels` 或其他候选区。

在 `configs/sources.local.yaml` 中设置 `indoor.root` 为已签收数据目录，完成确认后设置 `indoor.review_approved: true`。

```bash
ROOT=/home/xmv/garbage_yolov8_seg
python scripts/v3_dataset.py build-indoor \
  --config configs/sources.local.yaml \
  --out "$ROOT/datasets/v3_indoor_clean"

python scripts/v3_dataset.py preview \
  --dataset "$ROOT/datasets/v3_indoor_clean" --split val \
  --out "$ROOT/experiments/v3_yolov8s/previews_indoor_val" --count 145

python scripts/v3_dataset.py guard \
  --public "$ROOT/datasets/v3_public" \
  --indoor "$ROOT/datasets/v3_indoor_clean" \
  --out "$ROOT/experiments/v3_yolov8s/cross_domain_near_duplicate_review.csv"
```

**必须人工复核 near_duplicate_review.csv** 和同场景连续拍摄序列。SHA256 检查只能排除完全相同的文件，dHash 只能提出候选。旧数据的某些 val/test 标注及会话划分存在问题，不能直接在原始 B0 val 上宣布改进。

## 4. Phase 1 冒烟测试 → 正式训练（纯公开数据）

这一步无需等待 SAM3 全部完成，但若已经有独立室内集，建议先执行跨源 `guard`。

```bash
ROOT=/home/xmv/garbage_yolov8_seg
CODE=/home/xmv/projects/garbage-yolov8-seg/v3_yolov8s
cd "$CODE"

python scripts/v3_train.py --stage smoke \
  --public "$ROOT/datasets/v3_public" \
  --project "$ROOT/experiments/v3_yolov8s" --device 0 --batch 16

# 查看 p1_smoke/train_batch*.jpg、loss、有无 CUDA OOM/NaN/错标。
# 通过后执行：
python scripts/v3_train.py --stage p1 \
  --public "$ROOT/datasets/v3_public" \
  --project "$ROOT/experiments/v3_yolov8s" --device 0 --batch 16
```

参数：`yolov8s-seg.pt`、imgsz800、epoch上限100、patience20、freeze0、lr0=1e-3、AdamW、mosaic.3、copy_paste0、mask_ratio2。GPU 不够时只改 batch，并记录实际值。第一次建议不要盲目提升 imgsz，先用固定800建立基线。

P1 正式输出：`$ROOT/experiments/v3_yolov8s/p1_public/weights/best.pt`。

## 5. 旧 B0/P1 在新室内 val 上统一评估

```bash
ROOT=/home/xmv/garbage_yolov8_seg

# 确保这是 LFS 下载后的真实 ~54.8MB 权重，不是133字节的LFS指针！
B0="$ROOT/indoor_ab/B0_taco_indoor_v2_800_2stage/phase2/weights/best.pt"
P1="$ROOT/experiments/v3_yolov8s/p1_public/weights/best.pt"

python scripts/v3_evaluate.py --weights "$B0" \
  --indoor "$ROOT/datasets/v3_indoor_clean" \
  --out "$ROOT/experiments/v3_yolov8s/eval_b0_newval.json"
python scripts/v3_evaluate.py --weights "$P1" \
  --indoor "$ROOT/datasets/v3_indoor_clean" \
  --out "$ROOT/experiments/v3_yolov8s/eval_p1_newval.json"
```

两者在新 val 上的结果才是可对照基线；旧 B0 的 24.02% 是另一个未完成质量验收的 val 协议，不可直接并列比较。

## 6. Phase 2A：100% 室内微调

```bash
ROOT=/home/xmv/garbage_yolov8_seg
P1="$ROOT/experiments/v3_yolov8s/p1_public/weights/best.pt"
python scripts/v3_train.py --stage p2a \
  --public "$ROOT/datasets/v3_public" \
  --indoor "$ROOT/datasets/v3_indoor_clean" \
  --p1-weights "$P1" \
  --project "$ROOT/experiments/v3_yolov8s" --device 0 --batch 16
```

数据仅 `v3_indoor_clean/images/train`；验证 `v3_indoor_clean/images/val`。40 epoch 上限、patience12、lr0=1e-4、freeze0、mosaic0.1；输出 `p2a_indoor/weights/best.pt`。

## 7. Phase 2B：室内为主 + 25% 公开样本

```bash
ROOT=/home/xmv/garbage_yolov8_seg
P1="$ROOT/experiments/v3_yolov8s/p1_public/weights/best.pt"

python scripts/v3_dataset.py build-mixed \
  --public "$ROOT/datasets/v3_public" \
  --indoor "$ROOT/datasets/v3_indoor_clean" \
  --out "$ROOT/datasets/v3_indoor_rehearsal" --public-fraction 0.25

python scripts/v3_train.py --stage p2b \
  --public "$ROOT/datasets/v3_public" \
  --indoor "$ROOT/datasets/v3_indoor_clean" \
  --mixed "$ROOT/datasets/v3_indoor_rehearsal" \
  --p1-weights "$P1" \
  --project "$ROOT/experiments/v3_yolov8s" --device 0 --batch 16
```

P2B 仍由 **P1 best.pt 重新开始**，不是由 P2A 继续训。两组的 lr、imgsz、epoch 上限相同，只改数据组成。由于 mixed 多加公开图片，建议记录每个 epoch 的 step 数；如最终差别很小，需要固定总优化步数复验。

## 8. 统一评价四个模型与对比

```bash
ROOT=/home/xmv/garbage_yolov8_seg
EXP="$ROOT/experiments/v3_yolov8s"
for phase in p2a_indoor p2b_mixed; do
  python scripts/v3_evaluate.py \
    --weights "$EXP/$phase/weights/best.pt" \
    --indoor "$ROOT/datasets/v3_indoor_clean" \
    --out "$EXP/eval_${phase}_newval.json"
done

python scripts/v3_compare.py \
  "$EXP/eval_b0_newval.json" "$EXP/eval_p1_newval.json" \
  "$EXP/eval_p2a_indoor_newval.json" "$EXP/eval_p2b_mixed_newval.json" \
  --out "$EXP/v3_val_comparison.csv"
```

额外人工评估真实机器人拍摄的视频帧，记录典型误检/漏检、纸箱/透明瓶/易拉罐等类别表现以及**部署终端上的延迟**（训练卡 RTX5090 的推理耗时不能代替机器人部署端测试）。

## 9. 最终锁定 test（只在模型定版之后）

```bash
ROOT=/home/xmv/garbage_yolov8_seg
BEST=/ABSOLUTE/PATH/TO/SELECTED/best.pt
python scripts/v3_evaluate.py --weights "$BEST" \
  --indoor "$ROOT/datasets/v3_indoor_clean" --split test --unlock-test \
  --out "$ROOT/experiments/v3_yolov8s/FINAL_LOCKED_TEST.json"
```

## 常见故障与排查

| 提示 | 原因 | 处理 |
|---|---|---|
| `No public sources enabled` | 模板默认全部禁用 | 指向实际数据，并逐个完成检查后启用 |
| `unmapped class` | 源 `names` 与目标 9 类不一致 | 查看 `data.yaml` 后明确 mapping；不要全映到 other |
| `NOT a valid YOLO-Seg polygon` | 检测 bbox 或坏坐标 | 先转换成真正多边形；不要伪造掩码 |
| `EXACT CROSS-SPLIT LEAK` | train 与 val/test 图片重复 | 调整来源划分并重建，不要用此指标 |
| `Near duplicate candidates` | dHash 找到疑似相似图片 | 人工复核，配合拍摄会话信息 |
| `review_approved` | SAM3 尚未人工签收 | 完成检查并记录审核人/日期后再启用 |
| `FileExistsError` | 代码拒绝覆盖已完成实验 | 换新的数据/实验版本名，保留旧版 |
| `CUDA unavailable` / `OOM` | PyTorch CUDA 环境错误或显存不足 | 核对 nvidia-smi、torch.cuda、batch；必要时降 batch |
| `133 bytes best.pt` | Git LFS 指针，不是真实权重 | `git lfs pull` 或读取公司服务器训练产物 |
| `Results differing val` | 评估时 val 文件/标签改过 | 固定并审核新 val 后统一全部重新评估 |

**源码之外的步骤必须由真实文件/模型完成。** 此仓库包只提供训练计划和可运行工具，不包含已经训练完成的新模型或清洗完成的数据。