# TACO YOLOv8m-Seg baseline

This directory records the TACO v2 9-class baseline run. The training entry point is
`../train_stage1_taco_v2.py`; compact arguments and epoch metrics are kept in `train/`.
Images, logs, plots, and `.pt` weights are intentionally excluded from GitHub.

## Run configuration

- Ultralytics 8.4.176, Python 3.12.15, PyTorch 2.14.1+cu130
- NVIDIA RTX 5090, 640 image size, batch 16, seed 42, AMP enabled
- Maximum 100 epochs, patience 15; early stopped after 74 epochs
- TACO subset: 917 train / 114 val; the test split was not evaluated
- Custom classification weight: `other` (class 5) = 0.7; other classes = 1.0
- Preflight verified the weight tensor, loss propagation, non-classification losses, and
  classification gradients before training

## Best validation result

The best checkpoint was selected at epoch 59 by mask mAP50-95:

| metric | value |
|---|---:|
| mask precision | 0.413 |
| mask recall | 0.291 |
| mask mAP50 | 0.216 |
| mask mAP50-95 | 0.147 |
| box mAP50-95 | 0.165 |

Per-class mask mAP50-95 was: `bottle 0.363`, `box 0.112`, `can 0.386`,
`cigarette_butt 0.014`, `food_waste 0`, `other 0.118`, `paper 0.017`,
`plastic_bag 0.110`, `wrapper 0.201`.

These numbers are a TACO-subset validation baseline, not a test-set or deployment result.
`food_waste` has only 5 total instances in the subset and is not a reliable model-quality
gate here.

## Local checkpoint hashes

The local, non-uploaded files were:

- `best.pt`: SHA256 `BFCF3D357346E82E75BB554D16964E7DD786C2886FD8D064D382DD3A4EE703E2`
- `last.pt`: SHA256 `F41ABECA9A5FE46B71DF02437EB13742024821257221C5DD26CCDF5A5E0E5036`
