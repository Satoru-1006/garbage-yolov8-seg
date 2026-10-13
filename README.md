# Garbage YOLOv8-Seg Project

Indoor garbage instance-segmentation data preparation for YOLOv8-Seg.

## Scope

Planned sources include TACO, GSA2D, DWSD, Cigarette Butt Dataset, and subsequent LabelMe annotations.

## Repository contents

- `convert_labelme_to_yolo_seg.py`: deterministic LabelMe polygon to YOLO segmentation conversion script.
- `GARBAGE_YOLOV8_SEG_PROJECT.md`: current audit summary, evidence boundaries, and next stages.
- `aaa_extracted/`: packaged TACO audit, 9-class YOLO-Seg labels/manifests, conversion tools, and the recorded TACO baseline training code/metrics.

Local datasets, raw archives, generated labels/images, model weights, and experiment outputs are intentionally excluded from this repository.

## Current status

`TACO_BASELINE_RECORDED; MERGED_DATASET_TRAINING_NOT_RUN`

The TACO baseline record is limited to its 114-image validation split. It does not establish
full merged-dataset readiness, test-set performance, annotation semantics, cross-domain
generalisation, deployment safety, or production performance.
