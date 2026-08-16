# Data Preparation

Preparing the dataset properly is crucial for achieving high accuracy and ensuring no data leaks occur between the training and test sets.

## 1. Archives & Registration

Begin by obtaining the raw dataset ZIP files (e.g., the PranW dataset). Place these ZIP files in a safe location, but **do not commit them to Git**.

Create a configuration YAML file defining your sources:

```yaml
# sources.yaml
archives:
  - name: "pranw_dataset"
    path: "data/raw_archives/pranw_dataset.zip"
    license: "CC BY 4.0"
```

Extract them securely:
```bash
uv run container-id data extract --archives sources.yaml --dest data/raw/
```

## 2. Auditing & Contact Sheets

The dataset must be audited for invalid labels. The system expects source labels to be encoded in the filenames.

```bash
uv run container-id data audit --sources data/raw/ --output artifacts/audit/latest/
```

This command generates:
- Contact sheets containing collages of bounding boxes.
- Check-digit failure logs for manual review.
- A `acknowledgment.json` file which you must review and commit to verify you accept the audit results.

## 3. Canonical Dataset & OCR Generation

Once you have manually reviewed and overridden any errors, build the deduplicated canonical dataset. This step applies exact and perceptual (pHash) deduplication, ensuring groups of similar images are deterministic split (Train 80%, Valid 10%, Test 10%) together to prevent leakage.

```bash
uv run container-id data build-canonical --config canonical_config.yaml
```

Finally, slice the canonical dataset into character crops for the OCR stage:
```bash
uv run container-id data build-ocr --canonical data/processed/detection-v1 --output data/processed/ocr-v1
```
