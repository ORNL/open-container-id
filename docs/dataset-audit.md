# Dataset Audit Guide

This document details the automated dataset audit process. You must run this process to verify the integrity and semantics of the raw datasets before training.

## The Problem

Community datasets like PranW and dasad often suffer from inconsistencies:
- The class name `objects` in PranW might represent a container number, a seal, or an entire container depending on the image.
- Images may be duplicated or augmented in ways that leak into the validation set, ruining metrics.
- Filename labels used for OCR might contain invalid check digits or be completely wrong for the visible text.

## Automated Audit Command

To run the audit:

```bash
uv run container-id data audit --config configs/data/sources.local.yaml
```

This command performs several checks and outputs a report in `artifacts/audit/<run-id>/report.md` along with contact sheet images.

### What the Audit Checks

1. **Class Semantics:** Generates contact sheets for each class. You must visually inspect these to confirm what the class actually bounds.
2. **Duplication & Leakage:** Uses perceptual hashing (pHash) to group identical or near-identical images to ensure they are assigned to the same split.
3. **Check-Digit Validity:** Validates filename labels against the ISO 6346 check digit algorithm.
4. **Multiple Boxes:** Identifies images with multiple bounding boxes but only one filename label.

## Mandatory Actions

### 1. Visual Confirmation

You *must* open the generated contact sheets and visually confirm the semantics. For example, if the contact sheet shows that `objects` cleanly bounds container numbers, you can proceed.

### 2. Acknowledgment

Run the following command to explicitly acknowledge the audit and map the class to its semantic meaning:

```bash
uv run container-id data acknowledge-audit \
  --audit-dir artifacts/audit/<run-id> \
  --confirm-class pranw_container_number_v7:objects=container_number
```
*Note: Do not run this until you have reviewed the contact sheets.*

### 3. Manual Review

Some issues require manual intervention. Run the review tool to step through ambiguous cases (e.g., multiple boxes, invalid check digits):

```bash
uv run container-id data review --audit-dir artifacts/audit/<run-id>
```

You can choose to exclude problematic samples or correct their labels.

## Next Steps

Once the audit is acknowledged and reviewed, proceed to build the datasets as outlined in the [Data Preparation Guide](data-prep.md).