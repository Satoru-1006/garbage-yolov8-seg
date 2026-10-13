# Garbage YOLOv8-Seg Project

## Scope

Indoor garbage instance segmentation with YOLOv8-Seg. Planned sources: TACO, GSA2D, DWSD, Cigarette Butt Dataset, and subsequent LabelMe annotations.

## Current workspace

- Local workspace: `D:\aaa`
- Remote target: `/home/xmv/garbage_yolov8_seg`
- Candidate pretrained weight: `/home/xmv/waste_seg_train/yolov8s-seg.pt`
- Remote SSH verification: pending; the current read-only attempt was rejected by SSH authentication.

## Local data audit

- `waste_ds_c3_neg`: 694 images, 613 LabelMe JSON files, 1,416 polygons, 5 classes, 81 images without JSON.
- `half_dataset`: 750 images, 689 JSON files, 2,067 polygons, 9 classes, 61 images without JSON, 50 empty `shapes` JSON files.
- `half_dataset_yolo`: YOLO Segmentation output with 600/75/75 train/val/test images, 750 label files, 2,067 polygon rows, class IDs 0-8, and 111 empty labels.
- TACO v2 candidate subset: 1,145 kept images, 2,625 polygon rows, 917/114/114 train/val/test, and data-quality acceptance PASS.

Current local class map:

`0 bottle, 1 box, 2 can, 3 cigarette_butt, 4 food_waste, 5 other, 6 paper, 7 plastic_bag, 8 wrapper`

## Evidence boundaries

The conversion output and TACO validator establish separate data-format and leakage checks only. Annotation semantics, cross-dataset ontology compatibility, remote runtime readiness, test-set performance, and deployment success remain separate gates. A TACO-only baseline was recorded; the merged project has not been trained.

## Next stages

1. Complete remote read-only verification of GPU, CUDA, Conda, Ultralytics, disk, datasets, and weight SHA256.
2. Freeze a shared class ontology and source manifests before merging datasets.
3. Validate polygons, empty annotations, image-label pairing, class IDs, and source/scene-disjoint splits.
4. Run an approved YOLOv8-Seg smoke test, then a fixed single-GPU baseline.
5. Optimize one variable at a time and keep the final test set frozen.

Status: TACO_BASELINE_RECORDED; MERGED_DATASET_TRAINING_NOT_RUN.
