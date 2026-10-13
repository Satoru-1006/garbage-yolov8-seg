# Garbage YOLOv8-Seg Project

Indoor garbage instance-segmentation data preparation for YOLOv8-Seg.

## Scope

Planned sources include TACO, GSA2D, DWSD, Cigarette Butt Dataset, and subsequent LabelMe annotations.

## Repository contents

- `convert_labelme_to_yolo_seg.py`: deterministic LabelMe polygon to YOLO segmentation conversion script.
- `GARBAGE_YOLOV8_SEG_PROJECT.md`: current audit summary, evidence boundaries, and next stages.

Local datasets, raw archives, generated labels/images, model weights, and experiment outputs are intentionally excluded from this repository.

## Current status

`DATA_PREPARATION_PENDING; TRAINING_NOT_RUN`

The current conversion evidence establishes file-format conversion only. It does not establish annotation quality, duplicate/leakage status, cross-dataset ontology compatibility, remote runtime readiness, or training success.
