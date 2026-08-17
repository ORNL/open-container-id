# Data Preparation Guide

This guide describes how to register, extract, audit, and build the datasets required for training the Open Container ID models.

## 1. Prerequisites and Downloads

You must manually download the required datasets. The exact expected placement of the ZIP files is in the `data/raw/downloads` directory.

```bash
mkdir -p data/raw/downloads
# Place the downloaded datasets here
cp /path/to/pranw-v7-coco.zip data/raw/downloads/
cp /path/to/dasad-v1-coco.zip data/raw/downloads/
```

- [PranW Container Number Detection v7](https://universe.roboflow.com/pranw/container-number-detection-wcunq/dataset/7)
- [dasad Container number v1](https://universe.roboflow.com/dasad/container-number-pmov4-tvflz/dataset/1)

## 2. Source Registration

Register the downloaded datasets. This ensures the configuration knows where to find them and their expected format.

```bash
uv run container-id data register --config configs/data/sources.local.yaml
```

## 3. Extraction

Extract the registered dataset archives safely.

```bash
uv run container-id data extract --config configs/data/sources.local.yaml
```

## 4. Audit and Acknowledgment

Run the automated audit. This step generates a report checking for class semantics, duplicates, missing labels, and other dataset health metrics.

```bash
uv run container-id data audit --config configs/data/sources.local.yaml
```

This will produce a report in `artifacts/audit/<run-id>/report.md` and contact sheets. Open them and visually inspect the results.

Once you have visually confirmed the semantics, run the acknowledgment step:

```bash
uv run container-id data acknowledge-audit \
  --audit-dir artifacts/audit/<run-id> \
  --confirm-class pranw_container_number_v7:objects=container_number
```

## 5. Manual Review

Review labels to resolve ambiguous samples, invalid-check-digit samples, and multiple-box samples.

```bash
uv run container-id data review --audit-dir artifacts/audit/<run-id>
```

## 6. Build Canonical and OCR Datasets

Finally, build the split canonical dataset for detector training, and the OCR crop dataset for text recognizer training.

```bash
uv run container-id data build-canonical --config configs/data/canonical.yaml
uv run container-id data build-ocr --config configs/data/ocr.yaml
```

You can summarize the built datasets:

```bash
uv run container-id data summarize --dataset data/processed/detection-v1
uv run container-id data summarize --dataset data/processed/ocr-v1
```

## Generated Artifacts

- **Detection Dataset:** `data/processed/detection-v1`
- **OCR Dataset:** `data/processed/ocr-v1`

These paths will be referenced in your training configurations.

## See Also
- [Dataset Audit Guide](dataset-audit.md)
- [Detector Training Guide](detector-training.md)
- [OCR Training Guide](ocr-training.md)