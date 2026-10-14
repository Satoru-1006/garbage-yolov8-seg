# YOLOv8-Seg indoor garbage dataset v3 delivery

This is an independent v3 candidate built on 2026-10-14 from the local A/B
source image directories under `D:\aaa`. The source image SHA256 values match
`aaa_extracted/tools/merge_manifest_full1444.csv` for all 1,444 records.

## Result

- 1,063 exact-content representatives; 343 duplicate-content groups removed.
- Split counts: train 850, val 106, test 107.
- Split units keep exact content, dHash Hamming <= 5, <=3 s bursts, and a
  60-second timestamp-session heuristic together.
- `can`: train 4, val 1, test 1; `food_waste`: train 12, val 2, test 2.
- Structural, decode, polygon, pairing, exact-content, near-duplicate,
  burst, and 60-second cross-split checks pass.
- No model training was run.

## Annotation gate

The v2 ledger's 637 unresolved groups are carried into v3. A fresh grouping on
the byte-identified local source images yields 193 HIGH-priority groups (v2's
141 HIGH count used the full1444 content grouping; both ledgers are preserved).
All unresolved groups remain `human_required=true`.

Four val/test groups with clear permanent-poster/sign false positives were
provisionally set empty, and one clear outdoor-bin scene was provisionally kept
non-empty. Nine of the 14 val/test empty/non-empty groups remain pending. These
AI decisions do not constitute human annotation admission.

Therefore `V3_TRAINING_READY = NO`; the v3 tree is an audit/review candidate,
not a training authorization.

Review assets are visual aids only. Red polygons are A-source labels and green
polygons are B-source labels; unresolved semantic decisions require human
review.
