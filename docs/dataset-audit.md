# Dataset Audit & Manual Review

Because dataset labels in Open Container ID are derived primarily from image filenames (e.g., `MSKU1234567_door.jpg`), incorrect filenames result directly in training errors.

## The Audit Command

```bash
uv run container-id data audit --sources data/raw/ --output artifacts/audit/latest/
```

This creates several critical artifacts:
1. **Invalid Check Digits Log**: Lists all files where the filename parses to an ISO 6346 container number, but the mathematical check digit calculation fails.
2. **Missing Labels Log**: Files that could not be parsed at all.
3. **Contact Sheets**: Grid images representing every bounding box grouped by class.

## The Semantic Class Gate

We apply a strict "PranW class-semantic confirmation gate". If an image contains the class label `objects`, but the model is looking for text labels, the contact sheet helps a human quickly scan the dataset visually to ensure `objects` actually correlates with the region containing the ID.

## Review Store

If the audit discovers errors, you do not modify the raw images. Instead, you create a CSV file resolving the incorrect labels.

1. Open `artifacts/audit/latest/invalid_check_digits.csv`.
2. Inspect the associated images.
3. Create a corrections CSV file (e.g., `data/review/corrections.csv`) with columns: `original_filename,corrected_label`.
4. Run the review import:
```bash
uv run container-id data import-review --csv data/review/corrections.csv --store data/manifests/review_store.local.json
```
The canonical builder will respect the `review_store.local.json` over the raw filenames.
