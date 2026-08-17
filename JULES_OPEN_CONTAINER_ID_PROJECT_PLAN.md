# Jules Build Plan: Open, Offline Container Identification Reader

> **Audience:** Jules or another autonomous coding agent creating a new GitHub repository from scratch  
> **Working repository name:** `open-container-id`  
> **Working Python package name:** `container_id`  
> **Document status:** Implementation specification  
> **Primary development machine:** Apple Mac with M4 Max and 128 GB unified memory  
> **Primary training inputs:** Two local Roboflow COCO exports supplied by the repository owner  
> **Primary deployment requirement:** Completely local/offline inference from images, video files, and RTSP cameras  
> **Primary distribution requirement:** Open-source application with redistributable model bundles and no Roboflow runtime dependency

---

## 0. Instructions to Jules

Create a new GitHub repository and implement this project in small, reviewable stages. Treat this document as the authoritative product and engineering specification unless the repository owner explicitly changes a requirement.

### 0.1 Non-negotiable rules

1. **Do not call Roboflow APIs at runtime.** Roboflow is only the source of the two manually downloaded datasets.
2. **Do not commit the downloaded datasets to Git.** The repository must contain import scripts, manifests, notices, and instructions, not the raw image archives.
3. **Do not commit API keys, RTSP credentials, local paths, checkpoints, model caches, or private camera frames.**
4. **Do not silently trust dataset filenames or class semantics.** Build automated audits and manual-review outputs before using samples for training.
5. **Do not silently modify source exports.** Preserve the original archives and extracted source directories as read-only inputs. Write all normalized data to a separate generated directory.
6. **Do not use Ultralytics YOLO as the default implementation.** The intended default stack is permissively licensed RF-DETR for detection and docTR for text recognition.
7. **Do not use RF-DETR XLarge or 2XLarge.** Use only the Apache-2.0 core variants from Nano through Large. The initial model must be RF-DETR Small.
8. **Do not train from random initialization unless an experiment explicitly requires it.** Fine-tune permissively licensed pretrained checkpoints and record their origin and checksum.
9. **Do not require a network connection after dependencies and model bundles are installed.** The final runtime must pass an offline test.
10. **Do not fabricate accuracy, speed, dataset counts, or completed training results.** When real datasets or the M4 Max are unavailable to Jules, implement and test the pipeline using generated fixtures and leave clearly documented commands for the owner to run.
11. **Keep training dependencies separate from runtime dependencies.** A production installation must not require PyTorch, RF-DETR, docTR, Roboflow, notebooks, or training utilities when using exported ONNX models.
12. **Prefer deterministic, inspectable pipelines over opaque convenience wrappers.** Every generated split, crop, model bundle, and evaluation report must have a manifest and reproducible command.
13. **Use native macOS Python for MPS training.** Do not claim Docker can expose Apple MPS acceleration.
14. **Default to no telemetry, no cloud logging, and no frame retention.** Any optional persistence must be explicit and disabled by default.
15. **Use secure handling for archives and untrusted images.** Protect against ZIP path traversal, decompression bombs, malformed JSON, oversized images, and credential leakage.

### 0.2 Required agent working style

- Create an initial scaffold commit before implementing features.
- Make logically separated commits for data ingestion, model training, inference, RTSP support, API, packaging, and documentation.
- Maintain `IMPLEMENTATION_STATUS.md` with completed, in-progress, blocked, and owner-run tasks.
- Create GitHub issues for deferred work rather than burying TODOs in code.
- Add tests with every functional module.
- Run formatting, linting, type checking, and unit tests before every milestone commit.
- When an ambiguity affects correctness, expose it in an audit report or configuration option rather than inventing an undocumented behavior.
- At completion, provide a report containing:
  - repository tree;
  - implemented commands;
  - test results;
  - model-training commands the owner must run;
  - known limitations;
  - remaining issues;
  - exact release steps.

---

## 1. Project objective

Build an open-source system that can:

1. Train a model to locate an intermodal shipping-container identification number in an image.
2. Train a second model to recognize the characters inside the detected number region.
3. Normalize and validate candidate identifiers using ISO 6346 structure and check-digit logic.
4. Process still images, video files, webcams, and RTSP streams entirely on local hardware.
5. Combine detections and OCR results across multiple frames to produce a stable container-number event.
6. Export and package the trained detector and recognizer for offline, cross-platform inference.
7. Distribute the application and model bundle without requiring a Roboflow account, Roboflow API key, or internet connection.
8. Preserve dataset attribution, model provenance, reproducibility, and license notices.

The project is not merely an OCR script. It is a complete, auditable pipeline covering data preparation, training, evaluation, export, runtime inference, RTSP reliability, temporal consensus, licensing, testing, documentation, and release engineering.

---

## 2. Supplied datasets

The owner will manually download the following COCO exports and place the resulting ZIP files in a local, Git-ignored directory.

### 2.1 Dataset A: PranW, version 7

Source page:

`https://universe.roboflow.com/pranw/container-number-detection-wcunq/dataset/7/download/coco`

Known public metadata at the time this plan was written:

- Task: object detection.
- License displayed by Roboflow Universe: CC BY 4.0.
- Version 7 name: `Yolo 11 initial`.
- Version 7 export: 3,664 image records.
- Published source split: 3,070 train, 231 validation, and 363 test images.
- Published preprocessing: auto-orient and stretch resize to 640 × 640.
- Published augmentation: two outputs per training example with rotation from -15° through +15°.
- Published model/project class: `objects`.
- The project overview reports 2,129 project images, while the version 7 export reports 3,664 image records. Treat the difference as evidence of generated/augmented derivatives and audit source grouping carefully.

**Required hard gate:** Visually verify that the `objects` boxes surround container identification-number regions rather than entire containers, logos, isolated characters, or unrelated labels. Do not map `objects` to `container_number` until this check passes.

### 2.2 Dataset B: dasad, version 1

Source page:

`https://universe.roboflow.com/dasad/container-number-pmov4-tvflz/dataset/1/download/coco`

Known public metadata at the time this plan was written:

- Task: object detection.
- License displayed by Roboflow Universe: CC BY 4.0.
- Version 1 export: 3,407 image records.
- Published source split: 2,746 train, 661 validation, and no test images.
- Published preprocessing: none.
- Published augmentation: none.
- Published classes:
  - `container`
  - `ISO`
  - `container_number`

For the initial one-class detector, retain only `container_number` annotations. Preserve the other annotations in source metadata so a later multi-class experiment remains possible.

### 2.3 Filename-label declaration

The repository owner has stated that the source filename represents the text visible in the image. Implement the pipeline on that basis, but verify it rather than blindly trusting it.

The COCO image records may contain both:

```json
{
  "file_name": "BMOU4445146-1-_jpg.rf.28ea7b9f1f561335e364c5fc5d8a9a37.jpg",
  "extra": {
    "name": "BMOU4445146-1-.jpg"
  }
}
```

Use:

- `file_name` to locate the actual exported image;
- `extra.name`, when present, as the preferred original filename from which to derive the OCR label;
- a safe fallback parser for `file_name` when `extra.name` is absent.

The filename parser must reject ambiguous and nonconforming names rather than producing speculative labels.

---

## 3. Product requirements

### 3.1 Functional requirements

The repository must provide:

- Dataset registration and safe extraction.
- COCO schema and image-integrity validation.
- Automated dataset profiling and visual contact sheets.
- Class remapping into a canonical one-class detection dataset.
- Filename-to-container-number label extraction.
- ISO 6346 format and check-digit utilities.
- Exact and near-duplicate detection.
- Leakage-resistant train/validation/test splitting.
- OCR crop generation from COCO bounding boxes.
- A local review workflow for uncertain labels and boxes.
- RF-DETR detector training, evaluation, checkpointing, and export.
- docTR recognizer training, evaluation, checkpointing, and export.
- ONNX detector and recognizer runtime adapters.
- Still-image inference.
- Batch-directory inference.
- Video-file inference.
- RTSP inference with reconnect and frame dropping.
- Tracking and multi-frame consensus.
- Event output in JSON and JSON Lines.
- Optional local REST API.
- Optional annotated preview/video output.
- Offline model bundle validation.
- Docker CPU deployment.
- Native macOS deployment guidance.
- Model card, dataset notices, third-party notices, and release checksums.

### 3.2 Quality requirements

- Reproducible splits and preprocessing.
- Strong typing for public interfaces.
- Clear error messages and fail-fast validation.
- No cloud dependency in the inference path.
- No runtime loading of untrusted Python pickle checkpoints.
- Models loaded from ONNX or an explicitly trusted local development checkpoint.
- No source-image mutation.
- No hidden download behavior.
- No logging of RTSP passwords.
- No frame retention by default.
- Deterministic sample IDs and manifests.
- Cross-platform path handling.
- Unit, integration, data-contract, and offline tests.

### 3.3 Initial performance objectives

These are release targets, not fabricated promises. Record actual results and revise thresholds based on measured data.

#### Detector objectives

- Primary selection metric: recall at an operational precision floor, not mAP alone.
- Target clean-public-test recall: at least 0.95 at the selected threshold.
- Target small-box recall: at least 0.90 where the number region is small but still human-readable.
- Report AP50, AP50:95, precision, recall, F1, F2, and metrics by dataset source, orientation, and box-size stratum.

#### OCR objectives

- Primary selection metric: exact 11-character normalized match.
- Target clean-public-test exact match: at least 0.90.
- Report character error rate, character accuracy, exact match, valid-check-digit rate, and confusion pairs.
- Report results separately for horizontal, vertical/tall, low-resolution, blurred, and oblique crops.

#### End-to-end objectives

- Exact correct number is the success condition; ten correct characters out of eleven is still a failure.
- Target clean-public still-image exact match: at least 0.85.
- Default event policy must not emit a single-frame unvalidated guess.
- Record false accepted identifiers, missed identifiers, time to confirmation, and throughput.

#### Runtime objectives

- Maintain real-time stream freshness by dropping stale frames rather than building unbounded latency.
- Initial single-camera target: process at least 5 selected frames per second on the M4 Max or document the measured sustainable rate.
- Confirm a stable number within approximately one second when at least five usable frames per second are available.
- Run after network disconnection and host reboot when the model bundle and dependencies are already installed.

---

## 4. Deliberate technical decisions

### 4.1 Detector

Use **RF-DETR Small** as the default detector.

Rationale:

- COCO input is directly supported.
- Core RF-DETR variants Nano through Large and their code are published under Apache 2.0.
- The model can be fine-tuned from pretrained weights.
- The model can be exported to ONNX and, optionally, native Core ML or other formats.
- Small provides a practical first balance between accuracy and runtime cost.

Required experiments after the baseline:

1. RF-DETR Nano for lower-latency/edge deployment.
2. RF-DETR Small at a larger valid input resolution if small-number recall is inadequate.
3. RF-DETR Medium only when Small clearly underperforms and runtime remains acceptable.

Do not make XLarge or 2XLarge dependencies part of this project.

### 4.2 Detector label space

The default detector has exactly one canonical category:

```text
category_id: 0
category_name: container_number
```

Mapping:

```text
PranW v7: objects          -> container_number, only after visual validation
Dasad v1: container_number -> container_number
Dasad v1: container        -> excluded from v1 detector
Dasad v1: ISO              -> excluded from v1 detector
```

Support future class-map configuration, but do not make the initial detector multi-class.

### 4.3 OCR recognizer

Use **docTR `crnn_mobilenet_v3_small`** as the baseline recognizer, fine-tuned with a restricted vocabulary.

Default vocabulary:

```text
0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ
```

Default normalized target length:

```text
11 characters
```

Run a challenger experiment with docTR PARSeq after the CRNN baseline. Promote PARSeq only when its exact-match improvement justifies its runtime and export complexity.

### 4.4 OCR architecture boundary

The detector locates the number region. The recognizer reads the cropped region. Do not use docTR’s general page-text detector in the production path because the project already has domain-specific number-region boxes.

### 4.5 Runtime format

Use ONNX as the required cross-platform model format.

- ONNX Runtime CPU is the minimum supported backend.
- A native/Core ML path may be added for Apple acceleration after parity testing.
- TensorRT may be added for NVIDIA deployments after the generic ONNX runtime is stable.
- Runtime code must use model-bundle metadata rather than hardcoded tensor sizes, normalization constants, thresholds, or label maps.

### 4.6 Application language and tooling

- Python 3.11 is the reference development and training version.
- Python 3.12 should be supported when all pinned dependencies pass CI.
- Use `uv` for virtual environments, dependency resolution, and lockfile management.
- Use a `src/` package layout.
- Use Typer for the CLI.
- Use Pydantic for configuration and public data models.
- Use PyYAML for configuration files.
- Use OpenCV for image transformations and annotation rendering.
- Use PyAV as the preferred RTSP/video decoder, with a documented OpenCV fallback.
- Use FastAPI only as an optional extra for the local REST service.
- Use Ruff, MyPy, Pytest, and pre-commit for quality checks.

### 4.7 Repository and artifact licenses

Recommended starting structure:

- Repository source code: Apache License 2.0.
- Core RF-DETR dependency/checkpoint: Apache 2.0, restricted to core variants.
- docTR dependency/checkpoint: Apache 2.0, subject to exact checkpoint provenance verification.
- Dataset notices: CC BY 4.0 attribution for both source datasets.
- Trained model bundle: add an explicit `MODEL_LICENSE` after reviewing all upstream checkpoint and dataset obligations; Apache 2.0 is the intended choice, with dataset attribution retained.

This plan is engineering guidance, not a substitute for legal review. Include a release checklist that blocks publication of weights until the model-license and attribution files are complete.

---

## 5. Intended user workflows

### 5.1 Repository developer

```bash
git clone <repository>
cd open-container-id
uv sync --extra dev
uv run pre-commit install
uv run pytest
```

### 5.2 Dataset owner

```bash
mkdir -p data/raw/downloads
cp /path/to/pranw-v7-coco.zip data/raw/downloads/
cp /path/to/dasad-v1-coco.zip data/raw/downloads/

uv run container-id data register --config configs/data/sources.local.yaml
uv run container-id data extract --config configs/data/sources.local.yaml
uv run container-id data audit --config configs/data/sources.local.yaml
uv run container-id data build-canonical --config configs/data/canonical.yaml
uv run container-id data build-ocr --config configs/data/ocr.yaml
```

### 5.3 Detector trainer

```bash
uv sync --extra train
uv run container-id train detector --config configs/train/detector-rfdetr-small.yaml
uv run container-id evaluate detector --run-dir runs/detector/<run-id>
uv run container-id export detector --run-dir runs/detector/<run-id>
```

### 5.4 OCR trainer

```bash
uv run container-id train ocr --config configs/train/ocr-crnn-mobilenet-v3-small.yaml
uv run container-id evaluate ocr --run-dir runs/ocr/<run-id>
uv run container-id export ocr --run-dir runs/ocr/<run-id>
```

### 5.5 Model-bundle builder

```bash
uv run container-id bundle build \
  --detector runs/detector/<run-id>/exports/detector.onnx \
  --recognizer runs/ocr/<run-id>/exports/recognizer.onnx \
  --config configs/runtime/default.yaml \
  --output dist/models/container-id-models-0.1.0

uv run container-id bundle verify dist/models/container-id-models-0.1.0
```

### 5.6 Offline end user

```bash
uv sync --extra runtime
uv run container-id infer image \
  --models /opt/container-id/models \
  --input sample.jpg \
  --output result.json
```

### 5.7 RTSP operator

```bash
export CONTAINER_ID_RTSP_URL='rtsp://user:password@camera.example/stream'
uv run container-id rtsp run \
  --models /opt/container-id/models \
  --config configs/runtime/rtsp.example.yaml
```

---

## 6. Scope and non-scope

### 6.1 Version 0.1 scope

- One container-number-region class.
- One camera process per runtime instance.
- Images, directory batches, video files, and one RTSP source.
- RF-DETR Small detector.
- CRNN MobileNet V3 Small recognizer.
- ISO 6346 normalization and validation.
- Simple IoU tracking.
- Multi-frame candidate aggregation.
- ONNX Runtime CPU inference.
- Native MPS training scripts.
- Docker CPU runtime.
- CLI-first interface.
- Optional local FastAPI service.

### 6.2 Deferred or optional scope

- Multi-camera shared batching.
- Native desktop GUI.
- Mobile app.
- Cloud dashboard.
- BIC owner-code API lookup.
- Automatic gate-control integration.
- Vehicle or trailer tracking.
- Container size/type-code recognition.
- CSC plate reading.
- Full-container detection as a required first stage.
- Character-level bounding-box annotation.
- TensorRT and specialized edge-device packages.
- Automatic quadrilateral rectification learned by a dedicated model.
- Human identity or vehicle license-plate recognition.

Do not let deferred features delay the reproducible training and offline runtime baseline.

---

## 7. High-level architecture

```text
                               TRAINING PIPELINE

  Manually downloaded COCO ZIP files
                    |
                    v
         Safe extract + provenance
                    |
                    v
       COCO validation + visual audit
                    |
                    v
     Class mapping + exact/pHash dedupe
                    |
                    v
 Grouped deterministic train/valid/test split
              /                     \
             v                       v
 Canonical one-class COCO       OCR crop dataset
             |                 + filename labels
             v                       |
    RF-DETR fine-tuning              v
             |             docTR recognizer fine-tuning
             v                       |
      Detector evaluation            v
             |                OCR evaluation
             \                       /
              v                     v
                 ONNX export + parity
                         |
                         v
                 Signed model bundle


                               RUNTIME PIPELINE

 RTSP / image / video
          |
          v
 Decode + sample + latest-frame bounded queue
          |
          v
 ONNX detector: container_number boxes
          |
          v
 Crop from original-resolution frame + quality scoring
          |
          v
 Orientation/layout candidate generation
          |
          v
 ONNX recognizer + candidate normalization
          |
          v
 ISO 6346 pattern + check-digit validation
          |
          v
 IoU tracking + weighted multi-frame consensus
          |
          v
 JSON event / local API / optional annotated preview
```

---

## 8. Required repository structure

Create the following structure. Minor changes are acceptable only when they preserve the same separation of concerns.

```text
open-container-id/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml
│   │   ├── feature_request.yml
│   │   ├── data_quality.yml
│   │   └── model_regression.yml
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── package.yml
│   │   ├── docker.yml
│   │   ├── dependency-audit.yml
│   │   └── release.yml
│   ├── dependabot.yml
│   ├── labels.yml
│   └── PULL_REQUEST_TEMPLATE.md
├── assets/
│   ├── fonts/
│   │   └── README.md
│   └── examples/
│       └── README.md
├── configs/
│   ├── data/
│   │   ├── sources.example.yaml
│   │   ├── canonical.yaml
│   │   ├── ocr.yaml
│   │   └── synthetic.yaml
│   ├── train/
│   │   ├── detector-rfdetr-small.yaml
│   │   ├── detector-rfdetr-nano.yaml
│   │   ├── ocr-crnn-mobilenet-v3-small.yaml
│   │   └── ocr-parseq.yaml
│   ├── evaluate/
│   │   ├── detector.yaml
│   │   ├── ocr.yaml
│   │   └── end_to_end.yaml
│   └── runtime/
│       ├── default.yaml
│       ├── rtsp.example.yaml
│       └── service.example.yaml
├── data/
│   ├── README.md
│   ├── raw/
│   │   ├── downloads/.gitkeep
│   │   └── extracted/.gitkeep
│   ├── interim/.gitkeep
│   ├── processed/.gitkeep
│   ├── reviews/.gitkeep
│   └── manifests/.gitkeep
├── docker/
│   ├── Dockerfile.runtime
│   ├── Dockerfile.dev
│   ├── compose.example.yaml
│   └── entrypoint.sh
├── docs/
│   ├── architecture.md
│   ├── data-preparation.md
│   ├── dataset-audit.md
│   ├── training-macos-mps.md
│   ├── detector-training.md
│   ├── ocr-training.md
│   ├── evaluation.md
│   ├── model-bundle.md
│   ├── offline-deployment.md
│   ├── rtsp-deployment.md
│   ├── api.md
│   ├── security-and-privacy.md
│   ├── troubleshooting.md
│   └── release-process.md
├── licenses/
│   ├── DATASET_ATTRIBUTION.md
│   ├── THIRD_PARTY_NOTICES.md
│   ├── MODEL_LICENSE_TEMPLATE.md
│   └── SPDX-DEPENDENCIES.md
├── models/
│   └── README.md
├── notebooks/
│   └── README.md
├── scripts/
│   ├── bootstrap_macos.sh
│   ├── bootstrap_linux.sh
│   ├── verify_offline.sh
│   ├── benchmark_runtime.sh
│   └── build_release_bundle.sh
├── src/
│   └── container_id/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── version.py
│       ├── config/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   └── models.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── archives.py
│       │   ├── coco.py
│       │   ├── registry.py
│       │   ├── audit.py
│       │   ├── contact_sheets.py
│       │   ├── canonical.py
│       │   ├── dedupe.py
│       │   ├── splitting.py
│       │   ├── filename_labels.py
│       │   ├── ocr_crops.py
│       │   ├── review_store.py
│       │   ├── synthetic.py
│       │   └── schemas.py
│       ├── iso6346/
│       │   ├── __init__.py
│       │   ├── check_digit.py
│       │   ├── normalize.py
│       │   ├── candidates.py
│       │   └── types.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── device.py
│       │   ├── runs.py
│       │   ├── detector_rfdetr.py
│       │   ├── ocr_doctr.py
│       │   └── callbacks.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── detector.py
│       │   ├── ocr.py
│       │   ├── end_to_end.py
│       │   ├── strata.py
│       │   └── reports.py
│       ├── export/
│       │   ├── __init__.py
│       │   ├── detector.py
│       │   ├── recognizer.py
│       │   ├── parity.py
│       │   ├── bundle.py
│       │   └── manifest.py
│       ├── runtime/
│       │   ├── __init__.py
│       │   ├── interfaces.py
│       │   ├── detector_onnx.py
│       │   ├── recognizer_onnx.py
│       │   ├── preprocess.py
│       │   ├── crop_quality.py
│       │   ├── orientation.py
│       │   ├── vertical_unstack.py
│       │   ├── candidate_scoring.py
│       │   ├── pipeline.py
│       │   ├── tracking.py
│       │   ├── consensus.py
│       │   └── events.py
│       ├── streams/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── images.py
│       │   ├── video.py
│       │   ├── rtsp.py
│       │   ├── queue.py
│       │   └── reconnect.py
│       ├── service/
│       │   ├── __init__.py
│       │   ├── app.py
│       │   ├── dependencies.py
│       │   ├── routes.py
│       │   ├── schemas.py
│       │   └── metrics.py
│       └── util/
│           ├── __init__.py
│           ├── hashing.py
│           ├── images.py
│           ├── jsonl.py
│           ├── logging.py
│           ├── paths.py
│           ├── redaction.py
│           └── timing.py
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   ├── README.md
│   │   └── generate_fixtures.py
│   ├── unit/
│   ├── integration/
│   ├── data_contract/
│   ├── model_contract/
│   └── offline/
├── .dockerignore
├── .editorconfig
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── CITATION.cff
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── IMPLEMENTATION_STATUS.md
├── LICENSE
├── Makefile
├── MODEL_CARD_TEMPLATE.md
├── README.md
├── ROADMAP.md
├── SECURITY.md
├── pyproject.toml
└── uv.lock
```

### 8.1 Files that must remain Git-ignored

At minimum:

```gitignore
.env
.env.*
!.env.example
.DS_Store
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
venv/

# Raw and generated data
data/raw/downloads/*
!data/raw/downloads/.gitkeep
data/raw/extracted/*
!data/raw/extracted/.gitkeep
data/interim/*
!data/interim/.gitkeep
data/processed/*
!data/processed/.gitkeep
data/reviews/*
!data/reviews/.gitkeep
data/manifests/*.local.*

# Training outputs and checkpoints
runs/
checkpoints/
*.pth
*.pt
*.ckpt
*.safetensors

# Exported models and release bundles
models/*.onnx
models/*.mlpackage/
models/*.pte
models/*.trt
models/*.engine
models/*.tflite
dist/models/

# Local camera output
captures/
recordings/
annotated/
events.local.jsonl
```

Do not ignore license templates, model-card templates, sample configs, or generated audit summaries intentionally selected for publication.

---

## 9. Packaging and dependency design

### 9.1 `pyproject.toml` groups

Define optional dependency groups so runtime installations remain lean.

Suggested organization:

```toml
[project]
name = "open-container-id"
requires-python = ">=3.11,<3.13"
license = { text = "Apache-2.0" }

[project.scripts]
container-id = "container_id.cli:app"

[project.optional-dependencies]
runtime = [
  "numpy",
  "opencv-python-headless",
  "onnxruntime",
  "pydantic>=2",
  "pydantic-settings",
  "pyyaml",
  "typer",
  "rich",
]
rtsp = [
  "av",
]
api = [
  "fastapi",
  "uvicorn[standard]",
  "python-multipart",
  "prometheus-client",
]
train = [
  "torch",
  "torchvision",
  "rfdetr",
  "python-doctr[torch]",
  "pycocotools",
  "imagehash",
  "albumentations",
  "tensorboard",
]
export = [
  "onnx",
  "onnxruntime",
]
dev = [
  "pytest",
  "pytest-cov",
  "pytest-timeout",
  "mypy",
  "ruff",
  "pre-commit",
  "pip-audit",
  "types-pyyaml",
  "httpx",
]
```

Resolve actual compatible versions and commit `uv.lock`. Do not copy version numbers from this document without testing them together.

### 9.2 Import boundaries

- `container_id.runtime` must never import RF-DETR, docTR, TensorBoard, or training-only packages.
- `container_id.training` may import heavy training packages lazily inside commands.
- `container_id.service` must remain optional.
- Importing `container_id` must not initialize models, open cameras, create network connections, or download files.

### 9.3 Reproducibility metadata

Every training run directory must contain:

```text
config.resolved.yaml
run_manifest.json
source_git_commit.txt
python_environment.txt
pip_freeze.txt or uv_export.txt
dataset_manifest_sha256.txt
split_manifest_sha256.txt
stdout.log
metrics.jsonl
checkpoints/
evaluation/
exports/
```

`run_manifest.json` must record:

- run ID;
- UTC start and end times;
- command line;
- Git commit and dirty-tree status;
- operating system and architecture;
- Python, PyTorch, RF-DETR/docTR, ONNX, and ONNX Runtime versions;
- detected device;
- seed;
- dataset and split hashes;
- pretrained checkpoint URL/name and SHA-256;
- full resolved hyperparameters;
- best checkpoint and selection metric;
- failures or fallbacks encountered.

---

## 10. Configuration system

Use YAML configuration parsed into strict Pydantic models. Unknown keys should fail by default to catch typos.

### 10.1 Source configuration example

```yaml
schema_version: 1
sources:
  - id: pranw_container_number_v7
    archive_path: data/raw/downloads/pranw-v7-coco.zip
    extract_path: data/raw/extracted/pranw-v7
    source_url: https://universe.roboflow.com/pranw/container-number-detection-wcunq/dataset/7/download/coco
    license: CC-BY-4.0
    attribution_name: Container Number Detection Dataset
    attribution_creator: PranW
    expected_task: object_detection
    class_map:
      objects: container_number
    allowed_source_classes:
      - objects
    requires_manual_class_confirmation: true

  - id: dasad_container_number_v1
    archive_path: data/raw/downloads/dasad-v1-coco.zip
    extract_path: data/raw/extracted/dasad-v1
    source_url: https://universe.roboflow.com/dasad/container-number-pmov4-tvflz/dataset/1/download/coco
    license: CC-BY-4.0
    attribution_name: Container number Dataset
    attribution_creator: dasad
    expected_task: object_detection
    class_map:
      container_number: container_number
    excluded_source_classes:
      - container
      - ISO
    requires_manual_class_confirmation: false
```

Permit environment substitution such as `${PRANW_DATASET_ZIP}`, but never print secret values.

### 10.2 Runtime configuration example

```yaml
schema_version: 1
models:
  bundle_path: models/container-id-models
  verify_hashes: true

runtime:
  execution_provider_order:
    - CoreMLExecutionProvider
    - CPUExecutionProvider
  detector_threshold: 0.25
  max_detections: 20
  crop_padding_fraction: 0.08
  min_crop_width: 48
  min_crop_height: 16
  selected_frame_rate: 5.0
  detector_batch_size: 1
  recognizer_batch_size: 16

ocr:
  charset: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  require_iso_structure: true
  require_valid_check_digit_for_event: true
  max_confusion_corrections: 1
  transforms:
    - identity
    - rotate_180
    - rotate_90
    - rotate_270
    - vertical_unstack

tracking:
  iou_threshold: 0.30
  max_missed_frames: 10
  track_ttl_seconds: 3.0

consensus:
  minimum_supporting_frames: 3
  window_frames: 7
  minimum_weighted_score: 2.2
  duplicate_suppression_seconds: 30.0

privacy:
  save_frames: false
  save_crops: false
  save_annotated_video: false
  redact_stream_uri: true

logging:
  level: INFO
  json: false
```

All defaults must be documented and included in model-bundle metadata where they influence correctness.

---

## 11. Data registration and safe extraction

### 11.1 Registration command

Implement:

```bash
container-id data register --config <sources.yaml>
```

For each archive, calculate and record:

- absolute path, but do not commit it;
- archive filename;
- archive byte size;
- SHA-256;
- registration UTC timestamp;
- declared source URL;
- declared dataset version;
- license;
- attribution fields.

Write a local registry such as:

```text
data/manifests/source_registry.local.json
```

Write a sanitized, portable summary without machine-specific paths to the audit output.

### 11.2 Safe ZIP extraction

Implement extraction without trusting member paths.

Requirements:

- Reject absolute paths.
- Reject `..` path traversal.
- Reject symlink entries by default.
- Reject files that resolve outside the destination.
- Set a configurable maximum uncompressed total size.
- Set a configurable maximum file count.
- Refuse overwrite unless `--force` is supplied.
- Write an extraction manifest containing every member path, size, CRC when available, and SHA-256 after extraction.
- Do not alter timestamps or metadata in a way that changes source bytes unnecessarily.
- Mark the extracted directory read-only where practical, or at least include a prominent `DO_NOT_EDIT_SOURCE_DATA.txt` file.

### 11.3 Source layout discovery

Roboflow COCO exports commonly contain `train`, `valid`, and `test` directories with `_annotations.coco.json`, but scripts must discover and validate rather than assume blindly.

Support:

- `valid` and `val` aliases;
- absent `test` directories;
- empty test directories;
- an extra top-level folder inside a ZIP;
- image extensions `.jpg`, `.jpeg`, `.png`, and `.webp`;
- category IDs that do not begin at zero;
- optional `extra` fields.

Do not treat missing test data as an error for the source dataset. The canonical pipeline will create a new test split.

---

## 12. COCO validation

Implement a strict validator under `container_id.data.coco` and expose:

```bash
container-id data validate-coco --source <source-id>
```

### 12.1 Required validations

For each split:

- JSON parses successfully as UTF-8.
- Top-level `images`, `annotations`, and `categories` are lists.
- Image IDs are unique.
- Annotation IDs are unique where supplied.
- Category IDs are unique.
- Every annotation references an existing image ID.
- Every annotation references an existing category ID.
- Every `file_name` resolves inside the split directory.
- Every referenced image exists and is readable.
- Decoded image dimensions match COCO metadata.
- Bounding boxes have four finite numeric values.
- COCO box format is interpreted as `[x, y, width, height]`.
- Width and height are positive.
- Boxes are not entirely outside the image.
- Out-of-bounds boxes are reported.
- `area` is positive when present and approximately consistent with the bounding box.
- `iscrowd` is supported or explicitly rejected for unsupported cases.
- Segmentation fields are ignored safely because the task is box detection.
- Duplicate image filenames are identified.
- Images without annotations are counted and retained as possible negatives.
- Annotations with unsupported classes are counted but not discarded until canonicalization.

### 12.2 Repair policy

The source validator must not repair in place.

Canonicalization may perform explicit repairs only when configured, and every repair must be logged in a change manifest. Examples:

- clipping a slightly out-of-bounds box;
- converting a non-integer image dimension to integer metadata after confirming the decoded image;
- assigning a generated annotation ID when absent.

Never repair:

- a missing image;
- a zero-area box;
- an unknown class mapping;
- a filename-derived transcription;
- an annotation that appears to cover the wrong object.

Those cases require exclusion or review.

---

## 13. Dataset audit

Implement:

```bash
container-id data audit --config configs/data/sources.local.yaml
```

The command must produce both machine-readable JSON and a readable Markdown report under:

```text
artifacts/audit/<UTC-run-id>/
```

### 13.1 Audit report contents

Include at minimum:

#### Source-level summary

- archive hash and size;
- source version and license;
- discovered splits;
- image and annotation counts;
- category mapping;
- missing/unreadable files;
- invalid boxes;
- images without boxes;
- images with multiple number boxes.

#### Image profile

- width and height distributions;
- aspect-ratio distribution;
- file format distribution;
- estimated image byte-size distribution;
- grayscale/color distribution;
- blur score distribution;
- brightness and contrast distribution.

#### Box profile

- count by source class;
- normalized width, height, and area distributions;
- pixel width and height distributions;
- aspect ratio;
- horizontal/tall/near-square classification;
- boxes near image borders;
- overlapping boxes;
- number of boxes per image.

#### Filename-label profile

- presence of `extra.name`;
- successful exact ISO candidate extraction;
- successful structural candidate extraction with invalid check digit;
- no candidate;
- multiple candidates;
- rejected names by reason;
- label frequency;
- repeated container numbers;
- owner-prefix distribution;
- equipment-category character distribution;
- check-digit distribution.

#### Duplicate profile

- exact duplicate image hashes;
- duplicate original filenames;
- near-duplicate perceptual-hash clusters;
- samples appearing in both source datasets;
- likely augmentation families;
- likely sequential video-frame families.

#### Leakage risks

- same normalized container number in different source splits;
- same exact image in different source splits;
- near-duplicate images in different source splits;
- same original filename with different annotations;
- same image bytes with different filename labels.

### 13.2 Required contact sheets

Generate annotated contact sheets for:

- 100 random PranW `objects` boxes;
- 100 random dasad `container_number` boxes;
- all source classes from dasad;
- the smallest 100 number boxes;
- the largest 50 number boxes;
- 100 tall/vertical boxes;
- 100 horizontal boxes;
- 100 filename-label successes;
- 100 filename-label failures;
- invalid check-digit candidates;
- multiple-box images;
- exact-duplicate clusters;
- pHash near-duplicate clusters;
- any proposed box repairs.

Each thumbnail must show:

- source dataset;
- source split;
- original filename;
- exported filename;
- source class;
- box coordinates;
- parsed label and validation status;
- sample ID.

### 13.3 Hard audit gates

Canonical build must fail unless:

- every mapped class has been manually confirmed;
- no referenced image is missing;
- no accepted box has nonpositive area;
- no accepted annotation references an unknown category;
- a source-specific label extraction policy is configured;
- an audit acknowledgment file exists.

Implement an acknowledgment mechanism such as:

```bash
container-id data acknowledge-audit \
  --audit-dir artifacts/audit/<run-id> \
  --confirm-class pranw_container_number_v7:objects=container_number
```

The acknowledgment file must contain the audit hash and the exact confirmations. Do not use an untracked interactive-only state.

---

## 14. Canonical sample model

Create a versioned canonical manifest in JSON Lines. Every image should have one record.

Example:

```json
{
  "schema_version": 1,
  "sample_id": "pranw_v7_8f4c...",
  "source_dataset_id": "pranw_container_number_v7",
  "source_split": "train",
  "source_image_id": 104,
  "exported_file_name": "BMOU4445146-1-_jpg.rf....jpg",
  "original_file_name": "BMOU4445146-1-.jpg",
  "source_image_path": "data/raw/extracted/pranw-v7/train/...jpg",
  "image_sha256": "...",
  "perceptual_hash": "...",
  "width": 640,
  "height": 640,
  "raw_filename_label": "BMOU4445146-1-",
  "normalized_label": "BMOU4445146",
  "label_status": "accepted_check_digit_valid",
  "iso_structure_valid": true,
  "check_digit_valid": true,
  "owner_prefix": "BMO",
  "equipment_category": "U",
  "serial_number": "444514",
  "check_digit": "6",
  "annotations": [
    {
      "source_annotation_id": 37,
      "source_category": "objects",
      "canonical_category": "container_number",
      "bbox_xywh": [101.0, 203.0, 320.0, 73.0],
      "bbox_area": 23360.0,
      "orientation_class": "horizontal",
      "review_status": "unreviewed"
    }
  ],
  "exact_duplicate_group": "sha256:...",
  "near_duplicate_group": "phash-cluster-0021",
  "source_family_group": "original:BMOU4445146-1-.jpg",
  "split_group": "container:BMOU4445146",
  "canonical_split": "train",
  "license": "CC-BY-4.0"
}
```

### 14.1 Stable IDs

Generate stable IDs from immutable source identity, not filesystem order.

Suggested formula:

```text
sample_id = sha256(
  source_dataset_id + "\0" +
  source_split + "\0" +
  exported_file_name + "\0" +
  image_sha256
)[:24]
```

Annotation IDs can derive from `sample_id`, source annotation ID, and exact box coordinates.

---

## 15. Filename-to-label extraction

Implement a conservative parser under `container_id.data.filename_labels`.

### 15.1 Input selection

Preferred label source order:

1. `image.extra.name` when it is a nonempty string.
2. `image.extra.original_name` or equivalent known keys when present.
3. COCO `file_name` after removing Roboflow export suffixes.

Retain the full raw value in provenance.

### 15.2 Normalization steps

1. Extract the basename safely.
2. Remove the final image extension only.
3. Uppercase using locale-independent rules.
4. Remove known Roboflow export patterns such as `_jpg.rf.<hexhash>` only when they match exactly.
5. Do not globally delete arbitrary words before extraction.
6. Search for candidate sequences that allow separators between characters.
7. Normalize candidates to uppercase ASCII letters and digits.
8. Require exactly 11 normalized characters for an ISO container identifier.
9. Require the structural pattern:

```regex
^[A-Z]{3}[UJZ][0-9]{7}$
```

10. Compute and compare the check digit.

### 15.3 Candidate statuses

Return one of:

- `accepted_check_digit_valid`
- `candidate_invalid_check_digit`
- `candidate_nonstandard_equipment_category`
- `ambiguous_multiple_candidates`
- `no_candidate`
- `source_name_missing`
- `manual_override_valid`
- `manual_override_invalid_but_confirmed`
- `rejected`

### 15.4 Parsing examples

```text
BMOU4445146-1-.jpg                    -> BMOU4445146
BMOU4460053-1-Copy.jpg                -> BMOU4460053
BMOU 444514 6.jpg                     -> BMOU4445146
BMOU_444514_6_rotated.png             -> BMOU4445146
IMG_CON_CRANE_DOOR_A_20210829.jpg     -> no_candidate
IMG129-Copy.jpg                       -> no_candidate
1-132746001-OCR-AS-B01-Copy.jpg       -> no_candidate unless an explicit source mapping proves otherwise
```

Do not assume every number-like filename contains the visible container identifier.

### 15.5 Manual overrides

Support a versioned CSV or JSONL override file:

```csv
source_dataset_id,exported_file_name,normalized_label,status,reviewer,reviewed_at_utc,notes
```

Rules:

- Overrides must be explicit and reviewable.
- Never mutate the source COCO JSON.
- Overrides must be included in dataset-manifest hashing.
- An override must not bypass image/box validation.
- Invalid-check-digit labels may be retained only with manual confirmation and a reason.

---

## 16. ISO 6346 module

Implement a standalone, dependency-light `container_id.iso6346` package with thorough tests.

### 16.1 Data model

Expose a typed representation:

```python
@dataclass(frozen=True)
class ContainerId:
    owner_prefix: str
    equipment_category: str
    serial_number: str
    check_digit: str
```

Provide:

- `parse_container_id(text: str) -> ContainerId`
- `normalize_container_id(text: str) -> str`
- `calculate_check_digit(prefix_and_serial: str) -> int`
- `validate_check_digit(container_id: str) -> bool`
- `is_structurally_valid(container_id: str) -> bool`
- `generate_valid_container_id(...) -> str`
- `format_container_id(container_id: str, style=...) -> str`

### 16.2 Check-digit algorithm

Implement the standard letter-value mapping that begins at 10 and skips multiples of 11. Multiply each of the first ten character values by powers of two by position, sum, take modulo 11, and convert remainder 10 to check digit 0.

Do not hardcode a lookup only for observed prefixes.

### 16.3 Unit-test vectors

Include valid vectors such as:

```text
CSQU3054383
BMOU4445146
BMOU4460053
TGHU7599330
OOLU7215245
```

Include invalid variants with one changed character and one changed check digit.

### 16.4 Candidate correction

Implement a bounded candidate-correction utility for inference only.

Confusion pairs may include:

```text
O <-> 0
I <-> 1
Z <-> 2
S <-> 5
B <-> 8
G <-> 6
Q <-> 0
```

Rules:

- Never use correction to rewrite training ground truth automatically.
- At inference, apply at most the configured number of substitutions.
- Respect position constraints: the first four positions are letters; the last seven are digits.
- Prefer per-character model probabilities when available.
- Accept a correction only when it is uniquely best and produces a structurally valid number with a valid check digit.
- Record the raw OCR output and corrected output in event diagnostics.

---

## 17. Duplicate detection and leakage prevention

The original source splits must not be reused as the final canonical split because the two datasets may overlap, version 7 contains generated derivatives, and the dasad dataset has no test set.

### 17.1 Exact duplicates

Calculate SHA-256 for every decoded source image byte file. Group identical hashes.

Policy:

- Keep one canonical file reference where annotations and labels agree.
- Preserve all source records in provenance.
- When identical bytes have different boxes or labels, flag the entire group for review.
- Do not place members of the same exact-duplicate group in different splits.

### 17.2 Near duplicates

Calculate a perceptual hash, initially pHash with 64 bits. Build candidate clusters using a configurable Hamming distance, initially 4.

Also consider:

- resized variants;
- compressed variants;
- small rotations;
- brightness changes;
- Roboflow augmentations;
- adjacent video frames.

A pHash threshold is a candidate-generation mechanism, not an infallible identity test. Generate contact sheets and record review decisions for borderline clusters.

### 17.3 Source-family grouping

Use original filenames and Roboflow metadata to group derivatives from the same source image.

PranW version 7 has rotations and two outputs per training example. All derivatives of the same original must share a split.

### 17.4 Container-number grouping

All accepted samples with the same normalized container number must share a split. This prevents the recognizer from seeing the same identifier in training and test and avoids memorizing paint, rust, and container-specific background patterns.

### 17.5 Union-find split groups

Build a union-find graph connecting samples that share any of:

- exact image hash;
- accepted near-duplicate cluster;
- source-family identity;
- normalized container number;
- manually linked recording/sequence group.

The resulting connected component is the indivisible split group.

### 17.6 Canonical split

Create a new deterministic split with seed `6346`.

Initial target:

```text
train: 80%
valid: 10%
test:  10%
```

Balance approximately by:

- source dataset;
- horizontal/tall orientation;
- small/medium/large box strata;
- check-digit validity/manual-review status;
- negative-image presence.

Allow deviation when large groups prevent exact percentages. Record actual counts and group counts.

The test split must be locked. Hyperparameter selection uses only training and validation data.

### 17.7 Field test split

Reserve a future `field_test` split for images from the actual RTSP cameras. It must not be mixed into public-data validation. The repository should support this split from the beginning even when empty.

---

## 18. Canonical detector dataset

Implement:

```bash
container-id data build-canonical --config configs/data/canonical.yaml
```

Output:

```text
data/processed/detection-v1/
├── train/
│   ├── _annotations.coco.json
│   └── <stable-image-files>
├── valid/
│   ├── _annotations.coco.json
│   └── <stable-image-files>
├── test/
│   ├── _annotations.coco.json
│   └── <stable-image-files>
├── manifest.jsonl
├── split_manifest.jsonl
├── changes.jsonl
├── dataset_summary.json
└── DATASET_NOTICE.md
```

### 18.1 Canonical class IDs

```json
{
  "categories": [
    {
      "id": 0,
      "name": "container_number",
      "supercategory": "container"
    }
  ]
}
```

### 18.2 Image handling

- Preserve original source resolution whenever possible.
- Do not resize all images during canonicalization.
- The PranW export is already stretched to 640 × 640; retain it as supplied and record this limitation.
- Use stable filenames based on sample ID rather than potentially sensitive or inconsistent source filenames.
- Prefer hardlinks when source and destination are on the same filesystem and `link_mode: hardlink`; default to copy for cross-platform reliability.
- Never re-encode an image merely to rename it.

### 18.3 Bounding-box handling

- Retain float coordinates in COCO JSON.
- Clip only under an explicit, logged repair policy.
- Exclude zero-area and entirely out-of-bounds boxes.
- Include negative images where the source image legitimately has no mapped number annotation.
- Do not include dasad `container` or `ISO` boxes in the canonical detector annotations.

### 18.4 Dataset fingerprint

Calculate a fingerprint from:

- sorted source archive hashes;
- audit acknowledgment hash;
- class-map config;
- override file hash;
- canonicalization config;
- sorted image and annotation records;
- split manifest.

Write the resulting SHA-256 in `dataset_summary.json` and every training run manifest.

---

## 19. OCR dataset construction

Implement:

```bash
container-id data build-ocr --config configs/data/ocr.yaml
```

### 19.1 Eligibility

A source annotation is eligible when:

- its canonical class is `container_number`;
- its image and box pass validation;
- the filename parser yields a valid label or a manual override exists;
- the label has exactly 11 normalized characters;
- the label and annotation are not in an unresolved duplicate-conflict group;
- the image does not contain multiple different containers that make the filename-to-box relationship ambiguous.

Default training inclusion:

- include `accepted_check_digit_valid`;
- include `manual_override_valid`;
- include `manual_override_invalid_but_confirmed` only under an explicit config flag;
- exclude unresolved invalid-check-digit candidates;
- exclude ambiguous/no-candidate records.

### 19.2 Ground-truth crops

Use the ground-truth COCO box, not a detector prediction, for base OCR crop creation.

Crop procedure:

1. Read the image without changing color order accidentally.
2. Expand the box by a configurable fraction, initial default 8% on each side.
3. Clamp to image bounds.
4. Preserve the crop at native pixel resolution.
5. Save losslessly as PNG unless storage becomes prohibitive.
6. Record source box, padded box, crop dimensions, and padding.
7. Do not upscale during dataset generation.
8. Compute crop SHA-256 and pHash.

### 19.3 Crop quality metadata

For each crop calculate:

- width and height;
- aspect ratio;
- Laplacian blur score;
- brightness;
- contrast;
- estimated character pixels per character;
- clipping flags;
- border proximity;
- JPEG-blocking proxy when feasible;
- orientation class.

Classify, but do not automatically discard solely from one heuristic:

```text
good
small
very_small
blurred
low_contrast
overexposed
underexposed
tall_vertical
near_square
possibly_clipped
```

### 19.4 Multiple boxes in one image

When an image contains multiple number boxes:

- assign the filename label to all boxes only when audit confirms they depict the same physical container identifier;
- otherwise place the sample in manual review;
- record a `box_label_relationship` status.

### 19.5 docTR layout

Export each split in docTR recognition format:

```text
data/processed/ocr-v1/
├── train/
│   ├── images/
│   └── labels.json
├── valid/
│   ├── images/
│   └── labels.json
├── test/
│   ├── images/
│   └── labels.json
├── field_test/
│   ├── images/
│   └── labels.json
├── manifest.jsonl
├── rejected.jsonl
├── summary.json
└── DATASET_NOTICE.md
```

`labels.json` example:

```json
{
  "ocr_00000001.png": "BMOU4445146",
  "ocr_00000002.png": "CSQU3054383"
}
```

### 19.6 Split consistency

OCR crops inherit the canonical split of the source sample. Never re-split OCR crops independently.

### 19.7 Detector-noise robustness

Do not create permanent random jittered copies in the test set.

For training, apply bounded box/crop jitter on the fly:

- position shift up to approximately 5% of box width/height;
- padding variation;
- occasional minor clipping;
- scale variation;
- perspective distortion.

This approximates detector error without multiplying static files or creating split leakage.

---

## 20. Manual review workflow

Quality of filename labels is likely the main determinant of OCR quality. Implement a local review mechanism.

### 20.1 Required review queues

- PranW class-semantic confirmation.
- Filename with no candidate.
- Multiple candidate numbers.
- Structurally valid but invalid check digit.
- Multiple number boxes in one image.
- Exact duplicate with conflicting labels.
- Near duplicate with conflicting labels.
- Very small or clipped OCR crops.
- Tall/vertical layouts.
- Random accepted-label quality-control sample.

### 20.2 Review storage

Use SQLite or a versioned JSONL/CSV store with fields:

```text
review_id
sample_id
annotation_id
queue
current_status
proposed_label
reviewed_label
decision
reviewer
reviewed_at_utc
notes
audit_run_id
source_image_sha256
```

The review system must be local and require no external service.

### 20.3 Review UI

Implement one of the following, in order of preference:

1. A minimal local FastAPI/Jinja or static HTML review interface.
2. A CLI that opens/generated contact sheets and accepts decisions from a CSV.

The review interface should show:

- full image with box;
- enlarged crop;
- original and exported filenames;
- parsed candidate;
- check-digit result;
- source dataset and split;
- duplicate-cluster context;
- text entry and accept/reject controls.

### 20.4 Review reproducibility

- Export decisions to a text-based file suitable for Git.
- Do not commit private camera frames.
- Public dataset review overrides may be committed when they contain no image bytes and satisfy dataset terms.
- Every canonical rebuild must consume the same review decisions deterministically.

---

## 21. Orientation and vertical text strategy

Container identifiers may appear horizontally or vertically. Build explicit support rather than assuming all crops are wide.

### 21.1 Orientation classification

Use box/crop aspect ratio as an initial heuristic:

```text
horizontal: width / height >= 1.5
tall:       height / width >= 1.2
near_square: otherwise
```

Make thresholds configurable and report their distribution.

### 21.2 Runtime transform candidates

For each crop, generate only configured candidates:

- original;
- 180° rotation;
- 90° clockwise;
- 90° counterclockwise;
- contrast-enhanced variant;
- vertical-unstack variant for tall crops.

Avoid combinatorial explosion. Batch transformed crops through the recognizer.

### 21.3 Vertical unstacking

Implement an experimental deterministic transform for identifiers whose characters are upright but stacked vertically.

Suggested algorithm:

1. Convert crop to grayscale.
2. Estimate foreground polarity from border/background statistics.
3. Normalize contrast.
4. Calculate horizontal projection profile.
5. Identify candidate character bands using connected components or projection valleys.
6. Expect approximately 11 character positions, allowing merged or missing components.
7. Extract bands in top-to-bottom reading order.
8. Normalize each character tile to a common height while preserving aspect ratio.
9. Concatenate tiles left-to-right with padding.
10. Return a horizontal strip for the OCR recognizer.
11. Attach a confidence/quality score and diagnostics.

Do not use this transform when segmentation confidence is low. Retain direct rotations as alternate candidates.

### 21.4 Training support

- Keep real tall crops in their original form in the manifest.
- Apply transform selection in the dataset loader.
- Generate synthetic vertical stacked samples in the optional synthetic pipeline.
- Evaluate all layout strategies separately.
- Do not claim vertical support until a dedicated vertical/tall test subset passes.

---

## 22. Optional synthetic OCR data

Implement synthetic data as a separate, optional phase. The real public data remains the foundation.

### 22.1 Purpose

Synthetic examples can increase coverage of:

- rare letters and owner prefixes;
- check digits;
- clean and degraded fonts;
- horizontal and vertical layout;
- blur, glare, rust, shadows, oblique angles, and partial occlusion.

### 22.2 License requirements

- Use only fonts with redistribution and training rights documented in `assets/fonts/README.md`.
- Prefer SIL Open Font License or similarly permissive fonts.
- Store each font’s license alongside or reference its source.
- Do not use proprietary fonts from the host operating system in a published reproducibility pipeline.
- Use procedurally generated backgrounds or properly licensed textures.

### 22.3 Identifier generation

- Generate structurally valid identifiers.
- Use `U` predominantly, with configurable smaller proportions of `J` and `Z`.
- Calculate correct check digits.
- Sample owner prefixes broadly; do not imply generated prefixes correspond to actual registered owners.
- Mark synthetic records clearly.

### 22.4 Rendering variations

Include configurable:

- font families and weights;
- letter spacing;
- character size and stroke thickness;
- white-on-dark and dark-on-light text;
- painted, stenciled, and label-like appearance;
- perspective warp;
- affine rotation;
- motion blur;
- defocus blur;
- JPEG compression;
- rain streaks;
- shadows;
- glare;
- noise;
- corrosion masks;
- partial occlusion;
- horizontal layout;
- vertical stacked layout;
- nearby distractor text.

### 22.5 Mixing policy

- Never put synthetic samples in validation or test.
- Begin with a maximum 1:1 synthetic-to-real sampling ratio.
- Compare real-only and mixed experiments.
- Promote synthetic training only when real test accuracy improves.

---

## 23. Training-device abstraction

Implement `container_id.training.device` with:

```text
auto -> MPS when available on Apple Silicon, otherwise CUDA, otherwise CPU
mps
cuda
cpu
```

### 23.1 MPS checks

At startup record:

- `platform.machine()`;
- macOS version;
- PyTorch version;
- `torch.backends.mps.is_built()`;
- `torch.backends.mps.is_available()`;
- selected device.

Provide a command:

```bash
container-id doctor
```

It should verify:

- Python architecture is `arm64` on Apple Silicon;
- MPS availability;
- required dataset paths;
- optional FFmpeg/PyAV availability;
- ONNX Runtime providers;
- writable output directories;
- model-bundle hashes when supplied.

### 23.2 MPS fallbacks

- Run a one-batch and one-epoch smoke test before a long training run.
- Keep automatic mixed precision disabled on MPS unless verified for the selected model and PyTorch version.
- Do not set `PYTORCH_ENABLE_MPS_FALLBACK=1` silently.
- Provide a documented `--allow-mps-cpu-fallback` option that sets the environment variable before importing PyTorch.
- Record every fallback warning in the run manifest.
- Fail rather than silently switching the whole run to CPU unless the user passes an explicit fallback flag.

### 23.3 Memory handling

- Catch MPS/CUDA out-of-memory errors.
- Print a suggested smaller batch size and larger gradient accumulation value.
- Save resumable checkpoints.
- Do not assume 128 GB unified memory means every operation can allocate the full amount.

---

## 24. Detector training

Implement:

```bash
container-id train detector --config configs/train/detector-rfdetr-small.yaml
```

### 24.1 Baseline configuration

Use a resolved configuration approximately like:

```yaml
schema_version: 1
model:
  family: rfdetr
  variant: small
  pretrained: true
  class_names:
    - container_number

data:
  dataset_dir: data/processed/detection-v1
  dataset_manifest: data/processed/detection-v1/manifest.jsonl
  split_manifest: data/processed/detection-v1/split_manifest.jsonl
training:
  device: auto
  seed: 6346
  epochs: 100
  batch_size: 4
  grad_accum_steps: 4
  learning_rate: 0.0001
  use_ema: true
  early_stopping: true
  early_stopping_patience: 15
  checkpoint_interval: 5
  tensorboard: true
  wandb: false
  gradient_checkpointing: false
  resolution: default
output:
  root: runs/detector
```

Resolve actual parameter names against the pinned RF-DETR version. Do not silently ignore unsupported configuration keys.

### 24.2 Training phases

#### Smoke test

- 1–2 epochs.
- Small subset or full dataset with minimal epochs.
- Confirm MPS forward/backward, class mapping, checkpoint writing, validation, and resume.

#### Baseline

- RF-DETR Small, default resolution.
- 100 maximum epochs with early stopping.
- Effective batch size near 16 using gradient accumulation.
- No cloud logger.

#### Controlled experiments

Run one change at a time:

1. Threshold calibration only.
2. RF-DETR Nano.
3. Higher valid input resolution.
4. Increased crop/box augmentation.
5. Source balancing.
6. Hard-negative inclusion.
7. RF-DETR Medium when justified.

### 24.3 Augmentation policy

The PranW data is already stretched and augmented. Avoid applying excessive additional geometric distortion blindly.

Use moderate training-time augmentation and compare:

- horizontal flip only when semantically safe; container numbers remain readable but mirrored text is not realistic, so default must be **off**;
- small rotation;
- brightness/contrast;
- blur;
- noise;
- mild perspective;
- random resize/crop that does not cut the target excessively.

Do not use text-mirroring augmentation.

### 24.4 Checkpoints

Retain:

- latest checkpoint;
- best regular checkpoint;
- best EMA checkpoint;
- periodic checkpoints according to config.

Select the release candidate using validation recall/F2 and end-to-end results, not merely lowest loss.

### 24.5 Detector evaluation threshold

Sweep confidence thresholds over a reasonable range, such as 0.05 through 0.95. Produce a table and plots for:

- precision;
- recall;
- F1;
- F2;
- false positives per image;
- downstream OCR exact match.

Choose a deployment threshold that protects recall while keeping OCR workload manageable. Store the selected threshold in model-bundle metadata.

---

## 25. OCR training

Implement:

```bash
container-id train ocr --config configs/train/ocr-crnn-mobilenet-v3-small.yaml
```

### 25.1 Baseline configuration

```yaml
schema_version: 1
model:
  family: doctr
  architecture: crnn_mobilenet_v3_small
  pretrained: true
  vocabulary: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  max_length: 11
  input_height: 32
  input_width: 128
data:
  dataset_dir: data/processed/ocr-v1
  include_statuses:
    - accepted_check_digit_valid
    - manual_override_valid
training:
  device: auto
  seed: 6346
  epochs: 50
  batch_size: 64
  learning_rate: 0.001
  optimizer: adamw
  weight_decay: 0.0001
  scheduler: cosine
  early_stopping: true
  early_stopping_patience: 10
  freeze_backbone_epochs: 3
  amp: false
  tensorboard: true
augmentation:
  grayscale_probability: 0.10
  color_jitter_probability: 0.20
  shadow_probability: 0.30
  gaussian_noise_probability: 0.15
  gaussian_blur_probability: 0.25
  motion_blur_probability: 0.15
  perspective_probability: 0.25
  jpeg_compression_probability: 0.20
  crop_jitter_probability: 0.50
output:
  root: runs/ocr
```

Resolve the exact docTR API and model-supported arguments against the pinned version.

### 25.2 Vocabulary handling

- The model must not output lowercase letters, punctuation, or spaces.
- The recognizer may use internal special tokens required by its architecture.
- Export the exact output-index-to-character mapping.
- Include vocabulary and decoder behavior in the model manifest.
- Unit test encoding and decoding round trips.

### 25.3 Pretrained classifier mismatch

A custom vocabulary may require reinitializing the final classification layer while retaining pretrained feature-extractor weights. Verify and document how the pinned docTR version handles this. Record which layers load and which are newly initialized.

### 25.4 Training phases

#### Smoke test

- One epoch on generated fixture data.
- One epoch on a small real subset when available.
- Confirm MPS support, target encoding, loss, checkpoint save/load, and inference decoding.

#### Warm-up

- Freeze feature extractor for approximately three epochs.
- Train the recognition head.

#### Fine-tuning

- Unfreeze the network.
- Lower learning rate if needed.
- Continue with early stopping.

#### Challenger

- Train PARSeq on the same immutable split.
- Compare exact match, vertical/tall performance, runtime, model size, and ONNX export parity.

### 25.5 Input-size experiment

Compare at least:

```text
32 × 128
48 × 192 or another architecture-valid proportional size
```

The choice must be based on exact-match improvement and runtime. Do not upscale low-resolution source information and assume it creates detail; larger model input may still help preserve available pixels and spacing.

### 25.6 OCR metrics

Calculate:

- exact normalized match;
- character error rate;
- normalized edit distance;
- per-character accuracy;
- check-digit-valid output rate;
- structure-valid output rate;
- exact-match by source;
- exact-match by orientation;
- exact-match by crop-size stratum;
- confusion matrix;
- confidence calibration.

Preserve raw predictions for error analysis without exposing private field data in public artifacts.

---

## 26. Evaluation framework

Implement commands:

```bash
container-id evaluate detector --run-dir <run>
container-id evaluate ocr --run-dir <run>
container-id evaluate end-to-end --config <config>
```

### 26.1 Detector report

Generate:

```text
evaluation/detector/
├── metrics.json
├── metrics_by_source.csv
├── metrics_by_box_size.csv
├── metrics_by_orientation.csv
├── threshold_sweep.csv
├── errors_false_negative/
├── errors_false_positive/
├── contact_sheets/
└── report.md
```

Use test data only for final evaluation after model selection.

### 26.2 OCR report

Generate:

```text
evaluation/ocr/
├── metrics.json
├── predictions.jsonl
├── confusion_matrix.csv
├── metrics_by_source.csv
├── metrics_by_orientation.csv
├── metrics_by_quality.csv
├── errors_exact_match/
├── errors_invalid_check_digit/
├── contact_sheets/
└── report.md
```

### 26.3 End-to-end still-image report

Run detector predictions rather than ground-truth boxes, then OCR predicted crops. Report:

- detection success;
- OCR exact match conditional on successful detection;
- overall exact match;
- valid but wrong accepted number;
- invalid output rejected;
- no-read rate;
- extra false events;
- latency distribution.

### 26.4 Event/video evaluation

Define a video event ground-truth format:

```json
{
  "video_id": "camera1_2026-08-01_001",
  "events": [
    {
      "container_number": "BMOU4445146",
      "start_time_seconds": 12.5,
      "end_time_seconds": 21.0
    }
  ]
}
```

Measure:

- event exact match;
- missed event;
- duplicate event;
- false event per hour;
- time from first visible readable frame to confirmation;
- number of supporting frames;
- stream lag.

### 26.5 Error taxonomy

Every end-to-end error should be assignable to:

```text
detector_miss
detector_wrong_region
detector_crop_too_tight
low_resolution
motion_blur
glare_or_shadow
occlusion
vertical_layout_failure
ocr_character_confusion
filename_ground_truth_error
invalid_check_digit_ground_truth
multiple_container_ambiguity
tracking_fragmentation
consensus_failure
runtime_decode_failure
```

Use this taxonomy in reports and issue templates.

---

## 27. Model export

### 27.1 Detector export

Use RF-DETR’s supported export path to ONNX. The wrapper must:

- load the selected checkpoint and matching architecture;
- export a fixed, documented input shape initially;
- use a supported ONNX opset, initially the package default unless testing requires another;
- embed notes/metadata when supported;
- record preprocessing and output-decoding requirements;
- save SHA-256.

### 27.2 Recognizer export

Use docTR’s ONNX export utility or a tested `torch.onnx.export` wrapper.

Export metadata must include:

- input tensor layout;
- input height and width;
- color channel order;
- normalization mean and standard deviation;
- vocabulary;
- decoder type;
- maximum output length;
- special-token handling.

### 27.3 Export parity tests

Run each exported model on a fixed evaluation subset and compare eager PyTorch and ONNX outputs.

#### Detector parity

Compare:

- number of detections;
- class IDs;
- confidence values;
- box coordinates;
- matching by IoU;
- dataset-level metric difference.

Gate suggestion:

- no material recall loss;
- AP difference within 0.5 percentage points unless documented;
- matched boxes within a small coordinate tolerance.

#### OCR parity

Compare:

- decoded exact string;
- confidence;
- character logits/probabilities when exposed.

Gate suggestion:

- at least 99.5% prediction-string agreement on the parity set;
- exact-match metric reduction no greater than 0.5 percentage points.

### 27.4 ONNX model validation

- Run `onnx.checker`.
- Open an ONNX Runtime session.
- Execute representative samples.
- Test CPU provider.
- Test Core ML provider where available.
- Fail export when unsupported operators force unusable runtime behavior.

### 27.5 Optional Core ML

Add only after ONNX is stable.

- Export RF-DETR native Core ML or use a validated conversion path.
- Export recognizer to Core ML when practical.
- Compare exact metrics and performance.
- Keep ONNX as the portable reference bundle.
- Mark experimental exports clearly.

---

## 28. Model bundle specification

A distributable model bundle must be self-contained and verifiable.

```text
container-id-models-0.1.0/
├── detector.onnx
├── recognizer.onnx
├── manifest.json
├── detector_labels.json
├── recognizer_charset.txt
├── runtime_defaults.yaml
├── MODEL_CARD.md
├── MODEL_LICENSE
├── DATASET_ATTRIBUTION.md
├── THIRD_PARTY_NOTICES.md
└── SHA256SUMS
```

### 28.1 Manifest example

```json
{
  "bundle_schema_version": 1,
  "bundle_version": "0.1.0",
  "created_at_utc": "2026-08-01T00:00:00Z",
  "source_git_commit": "<commit>",
  "detector": {
    "file": "detector.onnx",
    "sha256": "<sha256>",
    "family": "rfdetr",
    "variant": "small",
    "input_shape": [1, 3, 512, 512],
    "color_order": "RGB",
    "class_names": ["container_number"],
    "default_threshold": 0.25,
    "preprocess": {
      "resize": "letterbox_or_model_specific",
      "normalization": "document_exact_values"
    }
  },
  "recognizer": {
    "file": "recognizer.onnx",
    "sha256": "<sha256>",
    "family": "doctr",
    "architecture": "crnn_mobilenet_v3_small",
    "input_shape": [1, 3, 32, 128],
    "color_order": "RGB",
    "charset": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "max_length": 11,
    "decoder": "ctc"
  },
  "iso6346": {
    "require_structure": true,
    "require_check_digit_for_events": true,
    "max_confusion_corrections": 1
  },
  "training_data": {
    "canonical_dataset_fingerprint": "<sha256>",
    "sources": [
      "pranw_container_number_v7",
      "dasad_container_number_v1"
    ]
  },
  "tested_runtime": {
    "onnxruntime_versions": ["<version>"],
    "platforms": ["macos-arm64", "linux-amd64", "linux-arm64"]
  }
}
```

### 28.2 Bundle verification

Implement:

```bash
container-id bundle verify <bundle-dir>
```

Checks:

- required files;
- schema version;
- SHA-256 sums;
- ONNX validity;
- runtime session creation;
- label and tensor compatibility;
- a deterministic smoke inference using a generated fixture;
- license and attribution files present.

Runtime must refuse hash mismatch by default.

---

## 29. Runtime interfaces

Create clean protocols independent of training frameworks.

```python
class Detector(Protocol):
    def detect(self, images: Sequence[np.ndarray]) -> list[list[Detection]]: ...


class Recognizer(Protocol):
    def recognize(self, crops: Sequence[np.ndarray]) -> list[list[OCRCandidate]]: ...
```

Typed models should include:

```python
class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


class OCRCandidate(BaseModel):
    raw_text: str
    normalized_text: str | None
    confidence: float
    transform: str
    structure_valid: bool
    check_digit_valid: bool
    corrections: list[str]


class ContainerEvent(BaseModel):
    event_id: str
    timestamp_utc: datetime
    camera_id: str
    track_id: str
    container_number: str
    check_digit_valid: bool
    confidence: float
    detector_confidence: float
    ocr_confidence: float
    supporting_frames: int
    first_seen_utc: datetime
    confirmed_at_utc: datetime
    bbox_xyxy: tuple[float, float, float, float]
    model_bundle_version: str
```

Do not leak ONNX Runtime objects into the rest of the application.

---

## 30. Runtime preprocessing and detection

### 30.1 Frame handling

- Preserve an original BGR or RGB frame for crop extraction.
- Create a detector input according to bundle metadata.
- Map model output coordinates back to the original frame exactly.
- Unit test coordinate transforms across portrait, landscape, square, and odd dimensions.

### 30.2 Crop extraction

- Expand detection box by configured padding.
- Clamp to frame bounds.
- Reject or down-rank very small crops.
- Keep original-resolution pixels.
- Never OCR the detector’s heavily resized tensor when the original frame is available.

### 30.3 Crop-quality score

Combine:

- crop pixel dimensions;
- blur score;
- contrast;
- detector confidence;
- clipping;
- perspective/extreme aspect ratio.

Use quality to weight OCR candidates and select the best frames for consensus.

### 30.4 Multiple detections

- Handle multiple container-number regions per frame.
- Cap detections by confidence and configurable maximum.
- Track independently.
- Avoid merging boxes from different containers merely because OCR strings are similar.

---

## 31. OCR candidate processing

### 31.1 Candidate transforms

Generate transforms based on layout, not all transforms for every crop without limit.

Suggested policy:

```text
horizontal crop:
  identity
  rotate_180 when upside-down is plausible
  optional contrast-enhanced identity

tall crop:
  rotate_90
  rotate_270
  vertical_unstack
  optional identity for PARSeq/experimental model

near-square:
  identity
  rotate_90
  rotate_270
  rotate_180
```

### 31.2 Normalization

For each OCR string:

1. Unicode normalize.
2. Uppercase.
3. Retain only ASCII `A-Z0-9` for ISO candidate extraction.
4. Remove spaces/hyphens only for normalized comparison.
5. Keep raw text for diagnostics.
6. Enforce position-aware structure.
7. Validate check digit.
8. Generate bounded confusion corrections when enabled.

### 31.3 Candidate score

A candidate score may combine:

```text
recognizer confidence
crop quality
transform prior
structure validity bonus
check-digit validity bonus
detector confidence
correction penalty
```

Do not allow check-digit validity to turn a very low-confidence arbitrary string into an accepted event. Require a minimum OCR confidence and multi-frame support.

### 31.4 No-read behavior

Return an explicit no-read result when no candidate meets minimum quality. Do not emit the closest-looking identifier merely to always return something.

---

## 32. Tracking

Implement a lightweight IoU tracker without an AGPL dependency.

### 32.1 Track state

Each track stores:

- track ID;
- last box;
- first and last timestamp;
- missed-frame count;
- detector confidences;
- crop-quality scores;
- OCR candidates by frame;
- current consensus;
- emitted-event status.

### 32.2 Association

For each frame:

1. Calculate IoU between active tracks and detections.
2. Match highest IoU pairs above threshold.
3. Prefer one-to-one greedy matching for MVP; add Hungarian assignment only when needed.
4. Create tracks for unmatched detections.
5. Age unmatched tracks.
6. Finalize tracks after TTL or missed-frame limit.

### 32.3 Future-proofing

Keep the tracker behind an interface so ByteTrack or another permissively licensed tracker can be added later. Do not couple consensus logic directly to the IoU implementation.

---

## 33. Multi-frame consensus

Single-frame OCR is too fragile for gate or logistics use. Implement weighted temporal aggregation.

### 33.1 Candidate accumulation

For each track, retain recent candidates with:

- normalized number;
- raw OCR;
- confidence;
- check-digit validity;
- detector confidence;
- crop quality;
- transform;
- frame timestamp.

### 33.2 Default event rule

An event may be emitted when:

- the same normalized number has at least three supporting frames within the configured window;
- the number is structurally valid;
- the check digit is valid by default;
- weighted score exceeds threshold;
- no competing number is within a configurable margin;
- the track is not already emitted;
- duplicate suppression does not block it.

### 33.3 Weighted voting

Suggested frame weight:

```text
weight = detector_confidence
       * ocr_confidence
       * crop_quality
       * transform_factor
       * correction_factor
```

Where:

- valid uncorrected candidates have factor 1.0;
- one-substitution corrected candidates have a penalty;
- invalid-check-digit candidates normally have zero event weight but remain diagnostic.

### 33.4 Character-level aggregation

Implement as a later enhancement behind the same interface:

- align 11-character candidates;
- sum character-position probabilities;
- search for the highest-probability ISO-valid sequence;
- require unique margin.

Do not block MVP on this enhancement.

### 33.5 Duplicate suppression

Suppress repeated events with the same camera and number for a configurable period. Allow a new event after the track disappears and suppression expires.

Record suppressed duplicates in debug metrics, not normal output.

---

## 34. Input sources

### 34.1 Still image

```bash
container-id infer image --input image.jpg --models <bundle> --output result.json
```

Options:

- annotated image output;
- JSON pretty print;
- raw detection diagnostics;
- threshold override;
- no correction;
- retain crops for debugging, explicit only.

### 34.2 Directory batch

```bash
container-id infer directory --input-dir images --recursive --output results.jsonl
```

Requirements:

- deterministic sorted order;
- continue-on-error option;
- per-file error records;
- bounded memory;
- optional batched inference.

### 34.3 Video file

```bash
container-id infer video --input video.mp4 --output events.jsonl
```

Requirements:

- preserve timestamps;
- selected-frame-rate option;
- optional annotated output;
- progress reporting;
- no real-time dropping for file mode unless configured.

### 34.4 Webcam

Support integer device index with clear documentation. Treat as development convenience, not the primary deployment path.

---

## 35. RTSP implementation

Implement:

```bash
container-id rtsp run --models <bundle> --config <rtsp.yaml>
```

### 35.1 Decoder design

Use a producer/consumer architecture:

```text
RTSP decoder thread/process
        |
        v
bounded latest-frame queue, size 1–3
        |
        v
inference worker
        |
        v
event sink
```

The queue must drop the oldest pending frame when full. Stream freshness is more important than processing every frame.

### 35.2 RTSP configuration

Support:

```yaml
camera:
  id: gate-1
  url_env: CONTAINER_ID_RTSP_URL
  transport: tcp
  selected_frame_rate: 5.0
  connect_timeout_seconds: 10
  read_timeout_seconds: 10
  reconnect:
    initial_delay_seconds: 1
    maximum_delay_seconds: 30
    multiplier: 2
    jitter_fraction: 0.20
```

Prefer an environment-variable reference over embedding credentials in YAML.

### 35.3 Credential redaction

All logs must render:

```text
rtsp://user:***@host/path
```

Do not log query tokens, passwords, or full environment variable values.

### 35.4 Reconnection

- Detect end-of-stream, timeout, decode error, and connection reset.
- Close decoder resources before reconnecting.
- Use exponential backoff with jitter.
- Reset backoff after a stable connection period.
- Preserve application health but mark camera readiness false while disconnected.
- Do not crash permanently on a transient camera outage.

### 35.5 Timestamp policy

Retain:

- source presentation timestamp when available;
- host receive monotonic time;
- UTC wall-clock timestamp.

Use monotonic time for TTL and latency calculations.

### 35.6 Stream metrics

Record:

- decoded frames;
- selected frames;
- dropped frames;
- decode errors;
- reconnect count;
- current stream lag;
- inference FPS;
- detector latency;
- OCR latency;
- active tracks;
- emitted events.

### 35.7 Privacy defaults

- Do not save frames.
- Do not save crops.
- Do not save annotated video.
- Do not send images or metadata to external services.
- Provide explicit flags and retention controls for debugging capture.

---

## 36. CLI design

Use a consistent Typer command tree.

```text
container-id doctor
container-id version

container-id data register
container-id data extract
container-id data validate-coco
container-id data audit
container-id data acknowledge-audit
container-id data build-canonical
container-id data build-ocr
container-id data review
container-id data summarize

container-id train detector
container-id train ocr
container-id train resume

container-id evaluate detector
container-id evaluate ocr
container-id evaluate end-to-end

container-id export detector
container-id export ocr

container-id bundle build
container-id bundle verify
container-id bundle inspect

container-id infer image
container-id infer directory
container-id infer video
container-id infer webcam

container-id rtsp run

container-id serve
```

### 36.1 CLI behavior

- `--help` must be informative.
- Use nonzero exit codes for validation failures.
- Print paths to generated reports.
- Support `--json` status output for automation.
- Support `--log-level` and `--log-json`.
- Do not catch all exceptions and hide stack traces in debug mode.
- Redact secrets in errors.
- Require explicit `--force` for destructive regeneration.

---

## 37. Local REST API

The API is optional and must use the same runtime pipeline as the CLI.

### 37.1 Endpoints

```text
GET  /healthz
GET  /readyz
GET  /version
GET  /v1/models
POST /v1/infer/image
GET  /metrics                  optional Prometheus format
```

Optional later:

```text
POST /v1/streams
DELETE /v1/streams/{id}
GET /v1/events
GET /v1/events/stream          server-sent events
```

### 37.2 API defaults

- Bind to `127.0.0.1` by default.
- Require an explicit flag to bind to all interfaces.
- Provide token authentication when exposed beyond loopback.
- Limit upload size and decoded pixel count.
- Accept only documented image types.
- Use timeouts.
- Do not persist uploads.
- Do not return raw crops unless requested and authorized.

### 37.3 Response example

```json
{
  "model_bundle_version": "0.1.0",
  "detections": [
    {
      "bbox_xyxy": [100.0, 200.0, 420.0, 275.0],
      "detector_confidence": 0.97,
      "best_candidate": {
        "raw_text": "BMOU4445146",
        "normalized_text": "BMOU4445146",
        "ocr_confidence": 0.96,
        "structure_valid": true,
        "check_digit_valid": true,
        "transform": "identity"
      }
    }
  ],
  "elapsed_ms": 82.4
}
```

---

## 38. Logging and observability

### 38.1 Logging

Use standard logging with optional JSON formatting.

Every log record should support:

- timestamp UTC;
- level;
- component;
- run ID or camera ID;
- frame/track/event identifier when relevant;
- message;
- elapsed time;
- redacted source URI.

Never log image bytes, model tensors, passwords, API tokens, or complete private file paths at normal log levels.

### 38.2 Metrics

Use an internal metrics interface. Prometheus export is optional.

Metrics should include:

```text
container_id_frames_decoded_total
container_id_frames_dropped_total
container_id_frames_processed_total
container_id_stream_reconnects_total
container_id_detector_latency_seconds
container_id_ocr_latency_seconds
container_id_end_to_end_latency_seconds
container_id_active_tracks
container_id_events_total
container_id_no_reads_total
container_id_invalid_check_digit_total
```

Use low-cardinality labels only, such as camera ID and model version. Do not label metrics by container number.

---

## 39. Docker and offline deployment

### 39.1 Runtime Dockerfile

Create a multi-stage, CPU-only image.

Requirements:

- Python slim base or similarly maintained minimal image.
- Install only runtime dependencies.
- Run as a non-root user.
- Read model bundle from `/opt/container-id/models`.
- Read configuration from `/etc/container-id/config.yaml` or mounted path.
- Write optional output to `/var/lib/container-id`.
- Use a read-only root filesystem where deployment permits.
- Add a health check.
- Do not download models during image start.
- Do not include datasets or training checkpoints.

### 39.2 Model inclusion options

Support two patterns:

1. Base runtime image with model directory mounted at runtime.
2. Release image built with a verified model bundle copied in.

The second pattern must only be used after model redistribution rights and notices are confirmed.

### 39.3 Multi-architecture

Build and test:

- `linux/amd64`
- `linux/arm64`

Do not claim GPU acceleration in the generic image.

### 39.4 Offline test

Provide:

```bash
scripts/verify_offline.sh
```

On Linux/Docker, test with:

```bash
docker run --rm --network none \
  -v "$PWD/models:/opt/container-id/models:ro" \
  <image> \
  container-id bundle verify /opt/container-id/models
```

Then run an image inference using a generated local fixture. The test must fail if the runtime attempts a network download.

### 39.5 Native macOS runtime

Document:

- ONNX Runtime CPU installation.
- Available execution providers.
- Optional Core ML execution provider or native Core ML bundle.
- Expected difference between native macOS and Linux Docker performance.
- Why training should run natively for MPS.

---

## 40. Security and privacy

### 40.1 Threat model

Document threats including:

- malicious ZIP archives;
- malformed COCO JSON;
- decompression bombs;
- oversized images;
- arbitrary file writes;
- path traversal;
- untrusted PyTorch pickle checkpoints;
- model tampering;
- RTSP credential leakage;
- unauthenticated network API exposure;
- denial of service through high-rate streams or uploads;
- accidental retention of camera imagery;
- dependency vulnerabilities.

### 40.2 Controls

- Safe archive extraction.
- Pillow/OpenCV decoded-size limits.
- Configurable maximum upload bytes and pixels.
- Model SHA-256 verification.
- ONNX runtime in production rather than arbitrary `.pth` loading.
- Trusted-checkpoint documentation for training.
- Redacted URIs.
- Loopback-only API default.
- Optional token auth.
- Bounded queues and batch sizes.
- No telemetry.
- No persistence by default.
- Non-root container.
- Read-only model mounts.
- Dependabot.
- `pip-audit` workflow.
- Security policy and private vulnerability-reporting instructions.

### 40.3 Model safety behavior

The system must expose confidence and validation state. It must not present OCR output as guaranteed truth. Default operational behavior should reject uncertain or invalid reads rather than silently accepting them.

---

## 41. Testing strategy

### 41.1 Unit tests

Cover at minimum:

- ISO letter-value table.
- Check-digit calculation.
- Structural validation.
- Candidate correction boundaries.
- Filename parsing.
- Roboflow suffix stripping.
- COCO box conversion.
- Box clipping.
- Stable ID generation.
- Safe ZIP extraction.
- Exact hashing.
- pHash grouping.
- Union-find split groups.
- Deterministic split.
- No group leakage.
- OCR crop padding.
- Coordinate mapping.
- Crop-quality scoring.
- Orientation selection.
- Vertical-unstack failure safety.
- IoU calculation and matching.
- Track aging.
- Consensus voting.
- Duplicate suppression.
- URI redaction.
- Bundle hash verification.
- Config unknown-key rejection.

### 41.2 Generated fixtures

Do not require public dataset downloads in normal CI.

`tests/fixtures/generate_fixtures.py` should create:

- tiny valid COCO train/valid/test directories;
- synthetic images with known boxes;
- filenames containing valid and invalid labels;
- exact and near duplicates;
- horizontal and vertical rendered identifiers;
- malformed COCO cases;
- path-traversal ZIP cases;
- fake model-bundle manifest.

Use fonts or rendering methods whose license is safe for test fixtures, or create simple geometric glyph placeholders.

### 41.3 Integration tests

- Register, extract, audit, canonicalize, and build OCR dataset from a generated ZIP.
- Train a tiny/mock model or use fake adapters rather than full RF-DETR in normal CI.
- Run end-to-end inference with fake detector and recognizer.
- Run FastAPI endpoint tests.
- Run video/queue logic with generated frames.
- Simulate RTSP disconnect/reconnect through a mock source.
- Build and verify a test bundle.

### 41.4 Optional model-contract tests

Mark tests requiring heavy dependencies:

```text
pytest -m model
pytest -m mps
pytest -m onnx
pytest -m data
```

They should verify:

- RF-DETR model instantiation;
- one inference;
- docTR recognizer instantiation;
- ONNX session load;
- exported tensor contract;
- MPS one-batch forward/backward.

### 41.5 Offline tests

- Block network access.
- Load model bundle.
- Infer a fixture.
- Confirm no DNS or HTTP attempt.
- Confirm no model-cache write outside configured directories.

### 41.6 Coverage

Set a practical initial target such as 85% line coverage for core data, ISO, runtime, and consensus modules. Do not chase superficial coverage in framework adapters at the expense of meaningful tests.

---

## 42. Continuous integration

### 42.1 `ci.yml`

Run on pull requests and pushes to `main`:

- Ubuntu latest, Python 3.11 and 3.12.
- macOS latest, Python 3.11 for platform-sensitive path and configuration tests.
- Windows latest, Python 3.11 for basic package/CLI tests when feasible.
- Ruff format check.
- Ruff lint.
- MyPy.
- Pytest unit/integration without heavy training extras.
- Package build.
- CLI help smoke test.

### 42.2 Dependency audit

Run:

- `pip-audit` or equivalent against locked dependencies;
- license inventory generation;
- Dependabot updates.

Do not auto-merge breaking ML dependency updates.

### 42.3 Docker workflow

- Build runtime image.
- Run as non-root.
- Run health/CLI smoke test.
- Run offline verification with `--network none`.
- Build architecture-specific images on release; multi-arch may use buildx.

### 42.4 Release workflow

Use a manually approved release workflow that:

- verifies clean Git tag;
- runs tests;
- builds Python wheel and source archive;
- validates model bundle supplied as an artifact;
- checks all license/notice files;
- calculates SHA256SUMS;
- builds optional Docker image;
- creates SBOM where practical;
- publishes GitHub release assets.

Never automatically train a release model in public CI using unavailable datasets.

---

## 43. Documentation requirements

### 43.1 `README.md`

Include:

- what the project does;
- two-stage detector/OCR explanation;
- offline and open-source goals;
- installation variants;
- five-minute inference quick start with a model bundle;
- data preparation quick start;
- M4 Max training path;
- RTSP example with credential handling;
- licensing summary;
- privacy defaults;
- limitations;
- links to detailed docs.

### 43.2 `docs/data-preparation.md`

Document exact expected placement of both ZIP files, source registration, extraction, audit acknowledgment, manual review, canonical build, OCR build, and generated artifacts.

### 43.3 `docs/training-macos-mps.md`

Document:

- native ARM64 Python check;
- Xcode command-line tools if needed;
- `uv` setup;
- MPS doctor output;
- smoke tests;
- batch-size tuning;
- fallback behavior;
- resume commands;
- TensorBoard;
- common MPS errors;
- why Docker is not used for MPS training.

### 43.4 `docs/offline-deployment.md`

Document:

- model-bundle installation;
- hash verification;
- ONNX Runtime providers;
- offline test;
- Docker and native paths;
- upgrading models without changing application code;
- rollback.

### 43.5 `MODEL_CARD_TEMPLATE.md`

Require:

- model version;
- intended use;
- out-of-scope use;
- architectures;
- training-data sources and licenses;
- preprocessing;
- split methodology;
- metrics by stratum;
- thresholds;
- known failure modes;
- privacy considerations;
- ethical/operational cautions;
- runtime requirements;
- provenance and checksums.

### 43.6 Dataset attribution

`licenses/DATASET_ATTRIBUTION.md` must identify both datasets, creators, source URLs, versions, CC BY 4.0, access date, and modifications such as class remapping, deduplication, resplitting, cropping, and augmentation.

### 43.7 Citation

Add `CITATION.cff` for the repository. Include citations for RF-DETR and docTR where their projects request them, and cite the two datasets in the model card.

---

## 44. GitHub repository settings

After repository creation:

- Default branch: `main`.
- Enable issues and discussions only when the owner wants discussions.
- Enable vulnerability reporting/private security advisories.
- Enable Dependabot alerts.
- Require pull requests for `main` after the initial scaffold.
- Require CI checks.
- Prefer squash merge.
- Add labels:
  - `area:data`
  - `area:detector`
  - `area:ocr`
  - `area:runtime`
  - `area:rtsp`
  - `area:api`
  - `area:docs`
  - `area:licensing`
  - `type:bug`
  - `type:feature`
  - `type:research`
  - `priority:high`
  - `good first issue`
  - `needs-data`
  - `needs-m4-test`
  - `model-regression`
- Add milestones matching the implementation phases below.

---

## 45. Implementation milestones

## Milestone 0: Repository scaffold

### Deliverables

- New GitHub repository.
- Apache-2.0 `LICENSE`.
- `pyproject.toml`, `uv.lock`, source layout, CLI entrypoint.
- Ruff, MyPy, Pytest, pre-commit.
- README skeleton and documentation tree.
- CI workflow.
- Gitignore and data README.
- `IMPLEMENTATION_STATUS.md`.

### Acceptance criteria

```bash
uv sync --extra dev
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
uv run container-id --help
```

All succeed without datasets or model files.

---

## Milestone 1: ISO 6346 and core schemas

### Deliverables

- ISO parser and check-digit implementation.
- Candidate normalization and correction utility.
- Typed configuration and event schemas.
- Test vectors and full unit tests.

### Acceptance criteria

- Known valid identifiers validate.
- Single-character invalid variants fail.
- Structure and check-digit statuses are distinct.
- Correction never exceeds configured edits.
- No external network or heavy ML dependency is imported.

---

## Milestone 2: Dataset registration, extraction, and COCO validation

### Deliverables

- Safe ZIP extraction.
- Source registry.
- COCO discovery and validation.
- Generated fixture dataset.
- CLI commands.

### Acceptance criteria

- Path-traversal ZIP test fails safely.
- A valid fixture extracts and validates.
- Missing images and invalid boxes are reported.
- Source directories are not modified.
- Registration records archive hashes.

---

## Milestone 3: Audit and filename labels

### Deliverables

- Filename parser.
- Label status taxonomy.
- Dataset audit JSON/Markdown.
- Contact sheets.
- Class-semantic acknowledgment mechanism.
- Duplicate analysis baseline.

### Acceptance criteria

- Parser handles documented examples.
- Ambiguous names are rejected.
- Audit shows extraction/check-digit rates.
- PranW class cannot be canonicalized without acknowledgment.
- Audit works on generated fixture and, when owner runs it, both real datasets.

---

## Milestone 4: Canonical dataset and split

### Deliverables

- Exact and perceptual duplicate grouping.
- Union-find leakage groups.
- Deterministic group-aware split.
- Canonical one-class COCO output.
- Dataset fingerprint and notices.

### Acceptance criteria

- No exact duplicate crosses splits.
- No accepted near-duplicate group crosses splits.
- No normalized container number crosses splits.
- Same seed/config yields identical manifest hashes.
- Canonical COCO passes validator.

---

## Milestone 5: OCR crop dataset and review

### Deliverables

- Crop generator.
- docTR `labels.json` export.
- Quality metadata.
- Review queues and decision store.
- Orientation classification.
- Rejected-sample report.

### Acceptance criteria

- Crops use ground-truth boxes and native source pixels.
- Labels inherit source split.
- No unresolved ambiguous label enters training by default.
- Every crop traces to source image and annotation.
- Review overrides change canonical results deterministically.

---

## Milestone 6: Detector training and evaluation

### Deliverables

- RF-DETR Small training adapter.
- MPS device handling.
- Run manifests and TensorBoard.
- Resume support.
- Detector evaluation and threshold sweep.
- ONNX export and parity test.

### Acceptance criteria

- Generated-data smoke run completes.
- Owner can run one-epoch MPS smoke test.
- Checkpoint resumes.
- Evaluation report is generated.
- Exported ONNX passes checker and CPU inference.
- No Roboflow API is used.

---

## Milestone 7: OCR training and evaluation

### Deliverables

- docTR CRNN training adapter.
- Restricted vocabulary.
- MPS training support.
- Exact-match and CER reports.
- ONNX export and parity.
- PARSeq configuration scaffold.

### Acceptance criteria

- Fixture smoke run learns/overfits a tiny dataset as a sanity check.
- Vocabulary mapping is exported.
- Checkpoint reload reproduces prediction.
- ONNX prediction agrees with PyTorch according to parity gate.
- Owner can run real-data training command on M4 Max.

---

## Milestone 8: Still-image end-to-end runtime

### Deliverables

- ONNX detector adapter.
- ONNX recognizer adapter.
- Crop and transform pipeline.
- ISO validation.
- Image and directory CLI.
- Annotated output option.
- End-to-end report.

### Acceptance criteria

- Runtime installation excludes training dependencies.
- Image inference uses only local files.
- Raw and normalized OCR are reported.
- Invalid/no-read cases are explicit.
- Bundle metadata controls preprocessing.

---

## Milestone 9: Tracking, consensus, video, and RTSP

### Deliverables

- IoU tracker.
- Weighted consensus.
- Video source.
- RTSP source with bounded queue.
- Reconnect logic.
- Event JSONL.
- Optional annotated preview.

### Acceptance criteria

- Queue remains bounded under slow inference.
- Stale frames are dropped.
- Reconnect test succeeds against a simulated source.
- Three-of-window consensus emits one event.
- Duplicate suppression works.
- Credentials are redacted.

---

## Milestone 10: API, Docker, and offline release path

### Deliverables

- FastAPI optional service.
- Runtime Dockerfile.
- Bundle verifier.
- Offline test.
- Release documentation.
- Security and license notices.

### Acceptance criteria

- Docker runs non-root.
- Image inference succeeds with `--network none`.
- Missing or tampered model fails hash verification.
- API binds loopback by default.
- Release checklist blocks absent license/attribution files.

---

## Milestone 11: Real training and model release

This milestone requires the repository owner’s local datasets and M4 Max.

### Owner-run tasks

1. Register both archives.
2. Run complete audit.
3. Review PranW class contact sheet.
4. Review filename failures and invalid check digits.
5. Build canonical detector and OCR datasets.
6. Run detector baseline.
7. Run OCR baseline.
8. Run end-to-end evaluation.
9. Add field-camera samples.
10. Select thresholds.
11. Export and verify models.
12. Complete model card and license review.
13. Publish model bundle.

### Acceptance criteria

- Real metrics recorded with no fabricated values.
- Test split remains locked.
- Model bundle passes offline test.
- License and dataset attribution are complete.
- Known failure modes are documented.

---

## 46. GitHub issues Jules should create

Create at least the following issues, linked to milestones.

1. Scaffold package, CLI, quality tooling, and CI.
2. Implement ISO 6346 parser and check digit.
3. Implement safe dataset archive registration and extraction.
4. Implement COCO validator.
5. Implement source filename label parser.
6. Build dataset audit and contact sheets.
7. Add PranW class-semantic confirmation gate.
8. Implement exact and perceptual duplicate grouping.
9. Implement group-aware deterministic split.
10. Build canonical one-class COCO dataset.
11. Build OCR crop dataset and docTR labels.
12. Implement manual review store and UI/CSV flow.
13. Implement MPS device doctor and training run manifests.
14. Implement RF-DETR Small training adapter.
15. Implement detector metrics and threshold calibration.
16. Implement detector ONNX export and parity tests.
17. Implement docTR CRNN training adapter.
18. Implement OCR metrics and error analysis.
19. Implement recognizer ONNX export and parity tests.
20. Implement model-bundle manifest and verification.
21. Implement ONNX detector runtime.
22. Implement ONNX recognizer runtime.
23. Implement crop-quality and orientation transforms.
24. Implement vertical-unstack experiment.
25. Implement still-image and directory inference CLI.
26. Implement IoU tracker.
27. Implement temporal consensus and duplicate suppression.
28. Implement video inference.
29. Implement RTSP decoder, bounded queue, and reconnect.
30. Implement local FastAPI service.
31. Implement non-root runtime Docker image.
32. Implement offline no-network smoke test.
33. Complete licensing, attribution, model card, and citation files.
34. Add field-data import path and protected field-test split.
35. Benchmark RF-DETR Nano versus Small.
36. Benchmark CRNN versus PARSeq.
37. Add optional synthetic OCR generator.
38. Add Core ML export and parity experiment.
39. Add multi-camera scheduling design.
40. Prepare v0.1.0 release.

---

## 47. Detailed acceptance checklist

### Data integrity

- [ ] Both source archive hashes are recorded.
- [ ] Source archives are never changed.
- [ ] Every source image reference resolves.
- [ ] Every accepted bounding box is valid.
- [ ] `objects` class semantics are visually confirmed.
- [ ] All accepted OCR labels are traceable to a filename or manual override.
- [ ] Invalid check-digit candidates are reviewed or excluded.
- [ ] Exact duplicates do not cross splits.
- [ ] Near-duplicate groups do not cross splits.
- [ ] Same normalized identifier does not cross splits.
- [ ] Canonical split is reproducible.
- [ ] Test split is locked.

### Detector

- [ ] RF-DETR Small uses a core Apache-licensed checkpoint.
- [ ] Training run records checkpoint source and hash.
- [ ] One-epoch MPS smoke test succeeds or a documented blocker exists.
- [ ] Resume works.
- [ ] Test metrics are generated only after selection.
- [ ] Operational threshold is calibrated.
- [ ] ONNX parity passes.

### OCR

- [ ] Vocabulary is exactly documented.
- [ ] Labels are 11 normalized characters unless explicitly reviewed.
- [ ] Training and test crops inherit source split.
- [ ] CRNN baseline is trained and evaluated.
- [ ] Vertical/tall subset is separately measured.
- [ ] ONNX parity passes.

### Runtime

- [ ] Runtime package does not import training frameworks.
- [ ] Model bundle hashes are checked.
- [ ] Image inference is fully offline.
- [ ] RTSP credentials are redacted.
- [ ] Frame queue is bounded.
- [ ] Stream reconnects.
- [ ] No frames are saved by default.
- [ ] Invalid/no-read results are explicit.
- [ ] Event requires multi-frame support by default.
- [ ] Duplicate events are suppressed.

### Distribution

- [ ] Apache-2.0 source license included.
- [ ] Dataset attribution included.
- [ ] Third-party notices included.
- [ ] Model license included.
- [ ] Model card completed.
- [ ] SHA256SUMS included.
- [ ] Docker runs as non-root.
- [ ] Offline `--network none` test passes.
- [ ] No API key or model download is required.

---

## 48. Makefile targets

Provide convenient wrappers:

```makefile
setup:
	uv sync --extra dev

setup-train:
	uv sync --extra dev --extra train --extra export

setup-runtime:
	uv sync --extra runtime --extra rtsp --extra api

format:
	uv run ruff format .

lint:
	uv run ruff check .
	uv run mypy src

test:
	uv run pytest

check: format-check lint test

data-audit:
	uv run container-id data audit --config configs/data/sources.local.yaml

data-build:
	uv run container-id data build-canonical --config configs/data/canonical.yaml
	uv run container-id data build-ocr --config configs/data/ocr.yaml

train-detector:
	uv run container-id train detector --config configs/train/detector-rfdetr-small.yaml

train-ocr:
	uv run container-id train ocr --config configs/train/ocr-crnn-mobilenet-v3-small.yaml

offline-test:
	bash scripts/verify_offline.sh
```

Do not make a target mutate data or delete runs without explicit confirmation.

---

## 49. Release versioning

Use semantic versioning for application code and separate model-bundle versioning.

Example:

```text
Application: v0.1.0
Model bundle: container-id-models-0.1.0
Canonical dataset schema: v1
Bundle manifest schema: v1
```

A model bundle update may occur without an application feature release when the manifest contract is unchanged.

Record compatibility in manifest fields:

```text
minimum_application_version
maximum_tested_application_version
```

---

## 50. Required owner-run first sequence

After Jules completes the repository, the owner should be able to perform this exact sequence.

### 50.1 Install

```bash
git clone <new-repository-url>
cd open-container-id
bash scripts/bootstrap_macos.sh
uv sync --extra dev --extra train --extra export --extra runtime --extra rtsp
uv run container-id doctor
```

### 50.2 Place data

```bash
mkdir -p data/raw/downloads
cp ~/Downloads/<pranw-coco-export>.zip data/raw/downloads/pranw-v7-coco.zip
cp ~/Downloads/<dasad-coco-export>.zip data/raw/downloads/dasad-v1-coco.zip
cp configs/data/sources.example.yaml configs/data/sources.local.yaml
```

Update local archive paths when necessary.

### 50.3 Audit

```bash
uv run container-id data register --config configs/data/sources.local.yaml
uv run container-id data extract --config configs/data/sources.local.yaml
uv run container-id data audit --config configs/data/sources.local.yaml
```

Open the generated `report.md` and contact sheets.

### 50.4 Confirm semantics

```bash
uv run container-id data acknowledge-audit \
  --audit-dir artifacts/audit/<run-id> \
  --confirm-class pranw_container_number_v7:objects=container_number
```

Only do this after visual confirmation.

### 50.5 Review labels

```bash
uv run container-id data review --audit-dir artifacts/audit/<run-id>
```

Resolve ambiguous, invalid-check-digit, and multiple-box samples.

### 50.6 Build datasets

```bash
uv run container-id data build-canonical --config configs/data/canonical.yaml
uv run container-id data build-ocr --config configs/data/ocr.yaml
uv run container-id data summarize --dataset data/processed/detection-v1
uv run container-id data summarize --dataset data/processed/ocr-v1
```

### 50.7 Smoke train

```bash
uv run container-id train detector \
  --config configs/train/detector-rfdetr-small.yaml \
  --override training.epochs=1

uv run container-id train ocr \
  --config configs/train/ocr-crnn-mobilenet-v3-small.yaml \
  --override training.epochs=1
```

### 50.8 Full training

```bash
uv run container-id train detector --config configs/train/detector-rfdetr-small.yaml
uv run container-id train ocr --config configs/train/ocr-crnn-mobilenet-v3-small.yaml
```

### 50.9 Evaluate and export

```bash
uv run container-id evaluate detector --run-dir runs/detector/<run-id>
uv run container-id evaluate ocr --run-dir runs/ocr/<run-id>
uv run container-id export detector --run-dir runs/detector/<run-id>
uv run container-id export ocr --run-dir runs/ocr/<run-id>
uv run container-id evaluate end-to-end --config configs/evaluate/end_to_end.yaml
```

### 50.10 Build and test bundle

```bash
uv run container-id bundle build \
  --detector runs/detector/<run-id>/exports/detector.onnx \
  --recognizer runs/ocr/<run-id>/exports/recognizer.onnx \
  --config configs/runtime/default.yaml \
  --output dist/models/container-id-models-0.1.0

uv run container-id bundle verify dist/models/container-id-models-0.1.0
bash scripts/verify_offline.sh dist/models/container-id-models-0.1.0
```

### 50.11 Test RTSP

```bash
export CONTAINER_ID_RTSP_URL='rtsp://user:password@host/stream'
uv run container-id rtsp run \
  --models dist/models/container-id-models-0.1.0 \
  --config configs/runtime/rtsp.example.yaml
```

---

## 51. Fail-fast conditions and decisions that must not be hidden

The pipeline must stop or require explicit override when:

1. The PranW `objects` class does not consistently represent number regions.
2. The filename does not reliably represent the visible number for a source.
3. A source image has multiple different container numbers but one filename label.
4. Duplicate images have conflicting labels.
5. The canonical split leaks a container number or near duplicate.
6. MPS silently falls back to CPU for substantial operations without owner consent.
7. ONNX export changes predictions beyond parity tolerance.
8. The runtime attempts to download a model.
9. Required attribution or model-license files are absent from a release.
10. The model bundle hash fails.
11. A field deployment would expose RTSP credentials in logs.
12. Offline inference cannot start after reboot without network access.

When a condition occurs, produce a clear report and actionable next step.

---

## 52. Known risks and mitigations

### Risk: Filename labels are incomplete or inconsistent

Mitigation:

- conservative parser;
- check-digit validation;
- manual review;
- rejected-sample report;
- no automatic speculative labels.

### Risk: PranW version 7 contains augmented duplicates

Mitigation:

- source-family grouping;
- pHash clustering;
- complete re-split;
- no reliance on original split.

### Risk: 640 × 640 stretch preprocessing reduces OCR fidelity

Mitigation:

- retain higher-resolution dasad crops;
- add actual-camera data;
- use PranW primarily for detection when OCR crops are insufficient;
- report OCR metrics by source;
- never upsample and treat it as recovered detail.

### Risk: Vertical text remains difficult

Mitigation:

- dedicated tall subset;
- rotation candidates;
- vertical unstacking;
- synthetic stacked-text data;
- PARSeq comparison;
- future character-level detector if needed.

### Risk: MPS operation incompatibility

Mitigation:

- doctor command;
- one-epoch smoke test;
- explicit fallback flag;
- pinned tested versions;
- resumable runs;
- document any CPU fallback.

### Risk: Public validation results overstate field performance

Mitigation:

- locked group-aware test;
- separate actual-camera field test;
- event-level metrics;
- no production claim based solely on Roboflow validation.

### Risk: License ambiguity for trained weights

Mitigation:

- use permissive core frameworks/checkpoints;
- maintain full provenance;
- preserve CC BY attribution;
- complete model-license review before release;
- do not rehost third-party trained weights.

### Risk: RTSP stream falls behind

Mitigation:

- bounded latest-frame queue;
- selected frame rate;
- latency metrics;
- frame dropping;
- separate decode and inference components.

### Risk: Valid-check-digit false positives

A random or wrong candidate can occasionally satisfy the check digit.

Mitigation:

- minimum OCR confidence;
- multi-frame support;
- competing-candidate margin;
- detector confidence and crop quality;
- check digit is one signal, not the only signal.

---

## 53. Definition of done

The repository is complete for version 0.1 when:

1. A fresh developer can install and run tests without downloading datasets.
2. The owner can register the two supplied ZIP files locally.
3. The audit produces accurate summaries and visual review outputs.
4. The owner can confirm class semantics and review filename labels.
5. The canonical detector and OCR datasets are reproducible and leakage-resistant.
6. RF-DETR Small and docTR CRNN training commands run natively on the M4 Max using MPS or produce a documented, actionable compatibility report.
7. Both models can be evaluated and exported to ONNX.
8. Exported models pass parity checks.
9. A model bundle can be built, hashed, inspected, and verified.
10. Still-image inference works with only runtime dependencies.
11. Video and RTSP inference use bounded queues and multi-frame consensus.
12. RTSP reconnect and credential redaction are tested.
13. The runtime functions without internet access.
14. The Docker image runs as non-root and passes a no-network smoke test.
15. Code, dataset, checkpoint, and model provenance are documented.
16. Dataset attribution and third-party notices are present.
17. No raw dataset, secrets, private frames, or unapproved weights are committed.
18. README, training, deployment, security, troubleshooting, and release documents are complete.
19. Real model metrics are clearly separated from targets and are not invented.
20. The owner can publish a GitHub release containing code and a legally reviewed model bundle.

---

## 54. Primary references for implementation

These references should be linked in the relevant repository documentation and revisited when pinning dependencies.

### Datasets

- PranW Container Number Detection, version 7:  
  `https://universe.roboflow.com/pranw/container-number-detection-wcunq/dataset/7`
- dasad Container number, version 1:  
  `https://universe.roboflow.com/dasad/container-number-pmov4-tvflz/dataset/1`
- Creative Commons Attribution 4.0:  
  `https://creativecommons.org/licenses/by/4.0/`

### Detector

- RF-DETR documentation:  
  `https://rfdetr.roboflow.com/latest/`
- RF-DETR training:  
  `https://rfdetr.roboflow.com/learn/train/`
- RF-DETR dataset formats:  
  `https://rfdetr.roboflow.com/latest/learn/train/dataset-formats/`
- RF-DETR export:  
  `https://rfdetr.roboflow.com/latest/learn/export/`

### OCR

- docTR documentation:  
  `https://mindee.github.io/doctr/`
- docTR custom training:  
  `https://mindee.github.io/doctr/latest/using_doctr/custom_models_training.html`
- docTR recognition training format:  
  `https://github.com/mindee/doctr/blob/main/references/recognition/README.md`
- docTR ONNX export:  
  `https://mindee.github.io/doctr/latest/using_doctr/using_model_export.html`

### Apple training

- PyTorch MPS backend:  
  `https://docs.pytorch.org/docs/stable/notes/mps.html`
- Apple accelerated PyTorch training:  
  `https://developer.apple.com/metal/pytorch/`

### Runtime

- ONNX Runtime installation:  
  `https://onnxruntime.ai/docs/install/`
- ONNX Runtime execution providers:  
  `https://onnxruntime.ai/docs/execution-providers/`
- ONNX Runtime Core ML provider:  
  `https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html`

### Container identification

- BIC container identification number:  
  `https://www.bic-code.org/identification-number/`
- BIC container markings:  
  `https://www.bic-code.org/marking-of-containers/`
- BIC check-digit calculator and explanation:  
  `https://www.bic-code.org/check-digit-calculator/`

---

## 55. Final direction to Jules

Build the repository so that all data-specific and model-specific behavior is explicit, testable, and reproducible. The most important engineering priorities are:

1. correct dataset interpretation;
2. prevention of train/test leakage;
3. reliable filename-derived OCR labels;
4. exact ISO 6346 validation;
5. separate, exportable detector and recognizer models;
6. local runtime with no cloud dependency;
7. fresh-frame RTSP processing;
8. multi-frame consensus rather than single-frame guesses;
9. model and dataset provenance;
10. open-source release readiness.

Do not optimize for a polished UI before the data, training, evaluation, and offline runtime contracts are correct. Build a dependable CLI-first system with clean interfaces, comprehensive tests, and documentation that allows the repository owner to train and release the models from the M4 Max without relying on hosted inference.
