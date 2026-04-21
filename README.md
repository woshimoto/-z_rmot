# T-ZeroRMOT Baseline

Minimal baseline scaffold for zero-shot / open-vocabulary referring multi-object
tracking experiments on Refer-KITTI, plus a lightweight RefCOCO grounding sanity
check.

The first baseline intentionally avoids training and heavy dependencies:

- Refer-KITTI: use `labels_with_ids` or a detector JSON as high-recall proposals,
  build/keep tracklets, parse the referring expression with rules, score/filter
  tracklets, and export MOT-style prediction files.
- RefCOCO: use COCO annotations or detector JSON as proposals, rank candidate
  boxes with the same query parser, and report Acc@0.5.

This is a runnable skeleton. You can later replace the proposal source with
GroundingDINO / YOLO-World and replace rule scoring with vLLM-hosted VLM/LLM
tracklet memory.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

No GPU package is required for the baseline.

## Smoke Test

```bash
python scripts/make_toy_data.py --out toy_data
python -m zerormot.run_refer_kitti \
  --data-root toy_data \
  --output-dir outputs/toy_refer_kitti \
  --proposal-source labels \
  --config configs/baseline.json
python -m zerormot.evaluate_simple \
  --pred-dir outputs/toy_refer_kitti/pred \
  --gt-dir outputs/toy_refer_kitti/gt
```

## Expected Refer-KITTI Layout

The runner is flexible, but it expects the official-style files somewhere under
`--data-root`:

```text
<data-root>/
  KITTI/
    training/image_02/<seq>/*.png
    labels_with_ids/image_02/<seq>/*.txt
  expression/<seq>/*.json
```

Each expression JSON should contain at least:

```json
{
  "sentence": "the white car behind the bus",
  "label": {
    "0": [3],
    "1": [3]
  }
}
```

Each `labels_with_ids` txt line is read as:

```text
class_id track_id norm_x norm_y norm_w norm_h
```

## Run Refer-KITTI Baseline

First check that the server-side dataset tree is readable:

```bash
python -m zerormot.check_refer_kitti \
  --data-root /path/to/refer-kitti \
  --limit 20
```

Then run an oracle sanity pass. This copies the expression-referred GT boxes to
the prediction folder, so it should be close to perfect under the simple IoU
check. Use this only to verify paths and output format:

```bash
python -m zerormot.run_refer_kitti \
  --data-root /path/to/refer-kitti \
  --output-dir outputs/refer_kitti_oracle \
  --proposal-source oracle \
  --config configs/baseline.json
```

Run the actual rule-based baseline:

```bash
python -m zerormot.run_refer_kitti \
  --data-root /path/to/Refer-KITTI \
  --output-dir outputs/refer_kitti_rule \
  --proposal-source labels \
  --config configs/baseline.json
```

Outputs:

```text
outputs/refer_kitti_oracle/
  pred/<seq>/<expr>.txt
  gt/<seq>/<expr>.txt
  summary.json
```

Prediction files use MOT format:

```text
frame,track_id,x,y,w,h,score,-1,-1,-1
```

Frames are written as 1-based indices to match common MOT tooling.

## Simple Refer-KITTI Sanity Evaluation

This is not a replacement for TrackEval/HOTA. It is a quick check that your
pipeline is producing reasonable boxes.

```bash
python -m zerormot.evaluate_simple \
  --pred-dir outputs/refer_kitti_oracle/pred \
  --gt-dir outputs/refer_kitti_oracle/gt \
  --iou-thr 0.5
```

## Detector JSON Format

Both Refer-KITTI and RefCOCO runners can read detector proposals from JSON:

```json
{
  "0000/000001": [
    {"bbox": [100, 80, 50, 40], "score": 0.9, "class_text": "car"},
    {"bbox": [20, 60, 30, 70], "score": 0.8, "class_text": "person"}
  ]
}
```

For Refer-KITTI, keys can be any of:

```text
<seq>/<frame_stem>
<seq>/<frame_number>
<seq>/<filename>
```

For RefCOCO, keys can be image id strings or image file names.

## Run RefCOCO Sanity Check

This supports the common UNC `refs(unc).p` pickle or a JSON dump of the same
list, plus a COCO instances JSON.

```bash
python -m zerormot.run_refcoco \
  --instances-json /path/to/instances_train2014.json \
  --refs /path/to/refs(unc).p \
  --split val \
  --output outputs/refcoco_val_predictions.json
```

The oracle-proposal baseline ranks all COCO boxes in the image by query match
and reports Acc@0.5 against the target `ann_id`.

## Where To Plug In VLM/vLLM

Recommended next extension points:

- `zerormot/query.py`: replace/augment rule parsing with LLM JSON output.
- `zerormot/scoring.py`: add VLM tracklet memory and self-verification scores.
- `zerormot/proposals.py`: add GroundingDINO / YOLO-World proposal generation.
- `zerormot/run_refer_kitti.py`: cache per-tracklet VLM outputs before scoring.
