# Experiment A: COCO direct fine-tuning (two stages)

This directory records the core, non-binary artifacts for
`A_COCO_DIRECT_INDOOR_800`:

- `EXPERIMENT_A_REPORT.md`: run protocol, environment, audit interpretation and results;
- `data_audit*.json`: independent dataset/leakage audit outputs;
- `eval_phase*_best.json`: validation-only evaluation outputs;
- `phase1/` and `phase2/`: Ultralytics arguments and `results.csv` histories;
- `smoke/`: frozen-parameter and preflight smoke-test evidence.

The recorded run used COCO-pretrained `yolov8m-seg.pt`, with a frozen-backbone
Phase 1 followed by a fully unfrozen Phase 2. The test split was kept out of
training, early stopping, and the recorded validation evaluations.

## Reproduction entry points

Run these commands from the `aaa_extracted/` directory after installing the
dependencies in `requirements.txt` and making the source images available:

```bash
python tools/indoor_a/audit_indoor_a.py
python tools/indoor_a/audit_indoor_a2.py
python tools/indoor_a/audit_indoor_a3.py
```

The training and smoke scripts expect a local COCO segmentation checkpoint.
Set `COCO_WEIGHTS` to its path; weights are intentionally not stored in Git:

```bash
COCO_WEIGHTS=/path/to/yolov8m-seg.pt python tools/indoor_a/smoke_A.py
COCO_WEIGHTS=/path/to/yolov8m-seg.pt python tools/indoor_a/train_A_2stage.py
```

For evaluation, provide a local phase checkpoint and an output JSON path:

```bash
python tools/indoor_a/eval_A.py \
  --weights /path/to/best.pt \
  --out experiments/indoor_a/A_coco_direct_800_2stage/eval_local.json \
  --split val --rect
```

The scripts resolve the repository root from `__file__`; the absolute paths
inside the archived YAML/JSON/report files are provenance from the original
Linux training host and are not rewritten.

## Deliberately excluded

Model weights (`*.pt`), images, visualization JPG/PNG files, and the full
training log are local or release-level artifacts. They are excluded to keep
the source repository small and to avoid redistributing dataset images.
