# TALM-MPR-NSCLC

Temporal Aligned Longitudinal Multimodal model for MPR prediction in NSCLC.

This repository contains the training and evaluation code for a longitudinal
multimodal model that combines CT image tensors, radiomics features, and lab
features. The public `dataset/` directory contains only five anonymized sample
records to document the expected JSON schema.

## Repository Layout

```text
dataset/
  main_pat_mpr.json        # anonymized sample labels
  m1_m2.json               # anonymized sample CT/radiomics time points
  lab.json                 # anonymized sample lab features
  check.json               # redacted report schema example
models/
  attention.py
  network.py
  shufflenetv2.py
preprocessing/
  preprocess_isbi_radiomics_reference.py
  preprocess_lab_reference.py
train/
  config.json
  train.py
  evaluate_metrics_from_csv.py
requirements.txt
```

## Data Policy

Do not commit real patient data, raw DICOM/NIfTI files, tensorized CT volumes,
radiomics tensors, model checkpoints, or clinical text. The `.gitignore` file
already excludes `data/`, model weights, tensor files, logs, and training
outputs.

The included JSON files are small anonymized examples. They are useful for
checking field names and longitudinal structure, but they are not enough to
reproduce the paper experiments.

## Expected Tensor Layout

Training expects tensor files under the configured `paths.tensor_root`
directory. With the default config and sample JSON, the layout is:

```text
data/tensors/
  m2_norm_params.pt
  example/PAT_001/REG_001/timepoint_01_roi_128.pt
  example/PAT_001/REG_001/timepoint_01_context_256.pt
  example/PAT_001/REG_001/timepoint_01_radiomics.pt
```

Each time point referenced by `dataset/m1_m2.json` may have:

- `_roi_128.pt`: ROI CT tensor, default shape `[1, 128, 128, 128]`
- `_context_256.pt`: context CT tensor, default shape `[1, 256, 256, 256]`
- `_radiomics.pt`: radiomics tensor, default length `293`

`m2_norm_params.pt` should be a PyTorch object with:

```python
{"mean": mean_tensor, "std": std_tensor}
```

The sample config sets `data.allow_missing_m2_norm=true` so schema-only checks
can initialize without this file. For real training, set it to `false` and
provide the normalization file.

## Lab Feature Preprocessing

`preprocessing/preprocess_lab_reference.py` documents the reference pipeline
used to produce `dataset/lab.json` from a raw SQL-exported inspection table.
It is included for transparency and may need local adaptation to match your
hospital information system fields.

Expected raw columns:

- `reg_no`
- `report_no`
- `item_name`
- `report_time`
- `result_data`
- `spec_type`

The reference pipeline:

1. Parses numeric, qualitative, and semi-quantitative lab values.
2. Deduplicates repeated report rows.
3. Keeps selected specimen categories.
4. Converts the long table into a `reg_no` and `date` indexed wide table.
5. Filters features by missing rate.
6. Drops zero-variance features.
7. Applies within-registration forward fill.
8. Fills remaining missing values with `0.0`.
9. Optionally applies global Z-score standardization.
10. Exports the model-ready time-series JSON and feature schema.

Example:

```bash
python preprocessing/preprocess_lab_reference.py \
  --input inspection_parse.json \
  --output dataset/lab.json \
  --schema-output outputs/lab_feature_schema.json \
  --missing-threshold 0.7
```

Do not commit `inspection_parse.json` or other raw inspection exports.

## iSBI / Radiomics Feature Preprocessing

`preprocessing/preprocess_isbi_radiomics_reference.py` documents how local
nodule-level radiomics outputs can be converted into the model's M2 feature
tensors. In this repository, these are image-based lesion/radiomics features,
not whole-slide pathology features.

The full local service used to create the per-scan radiomics JSON may depend
on private or hospital-specific modules such as DICOM-to-NIfTI conversion,
nodule detection, and fixed-mask radiomics extraction. Those model files and
private service modules are not included here. The open-source side expects one
radiomics JSON per scan time point.

Expected local service payload:

```json
{
  "job_id": "example-job",
  "results": [
    {
      "software_version": "v0.1",
      "space": "HU",
      "nodule": {
        "score": 0.91,
        "box_order_assumed": "(z0, y0, z1, y1, x0, x1)"
      },
      "radiomics_3d": {
        "binWidth": 25,
        "features": {
          "original_firstorder_Mean": -520.4,
          "original_shape_MeshVolume": 1420.0
        }
      }
    }
  ]
}
```

The JSON files should mirror the `m1_m2.json` scan paths relative to
`paths.input_root`. For example:

```text
outputs/radiomics_json/example/PAT_001/REG_001/timepoint_01.json
```

Then convert them into tensors:

```bash
python preprocessing/preprocess_isbi_radiomics_reference.py \
  --index dataset/m1_m2.json \
  --input-root /data/share/hospital_2 \
  --radiomics-json-root outputs/radiomics_json \
  --tensor-root data/tensors \
  --schema-output outputs/radiomics_feature_schema.json \
  --write-contract
```

This creates `_radiomics.pt` files under `data/tensors/`, writes
`data/tensors/m2_norm_params.pt`, and saves the final radiomics feature order.

The dependencies needed by the reference radiomics scripts are included in
`requirements.txt`. You will still need to provide your own detector weights
and local modules for DICOM conversion and nodule detection.

```bash
pip install -r requirements.txt
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA builds of PyTorch, install the wheel that matches your local CUDA
runtime from the official PyTorch instructions, then install the remaining
requirements.

## Configuration

Edit `train/config.json` before training. The main sections are:

- `paths`: dataset paths, tensor root, output directory
- `data`: time windowing, tensor suffixes, missing tensor shapes
- `model`: dropout, temporal fusion temperature, modality prior bias
- `training`: batch size, learning rates, CNN freeze schedule, workers
- `cross_validation`: fold count and shuffle behavior
- `runtime`: seed, device, mixed precision
- `parallel`: optional fold-to-device mapping for multi-GPU runs

Relative paths are resolved from the repository root.

## Training

Single process, all folds:

```bash
python train/train.py --config train/config.json
```

One fold only:

```bash
python train/train.py --config train/config.json --fold 1
```

Parallel fold training:

```bash
python train/train.py --config train/config.json --parallel
```

Outputs are written to `paths.output_dir`, which defaults to `outputs/`.

If you train on the five anonymized sample records, reduce
`cross_validation.n_splits` to `2` or provide more examples. The default `5`
fold setup is intended for the full private cohort.

## Evaluation

After training creates an OOF prediction CSV:

```bash
python train/evaluate_metrics_from_csv.py \
  --input outputs/pooled_oof_aligned.csv \
  --output-dir outputs/figures
```

The evaluation script prints AUC, accuracy, sensitivity, specificity, F1,
Brier score, and decision-curve net benefit, and saves ROC/PR/calibration/DCA
figures as PDF and SVG.

## Notes

- The public sample data is anonymized and redacted.
- The repository does not include preprocessing code for raw DICOM/NIfTI.
- The repository does not include trained checkpoints or pretrained BERT
  weights.
- Large generated files should remain outside version control.

## Closing Note

The journey may be long and arduous, but with perseverance, we will reach our
destination; with relentless efforts, a bright future is worth expecting.
