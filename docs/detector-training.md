# Detector Training

The detector models in this project use the Roboflow DETR (RF-DETR) architecture. We typically rely on the `RFDETRSmall` or `RFDETRNano` variants to strike a balance between precision and edge-device inference speed.

## Prerequisites

Ensure you have run the canonical data build process successfully.
```bash
uv run container-id data build-canonical --config canonical.yaml
```

## Running the Training Job

Use the CLI to kick off the detector training:

```bash
uv run container-id train detector --config rfdetr_config.yaml
```

The config should specify:
- Number of epochs
- Batch size
- Path to your canonical dataset (e.g., `data/processed/detection-v1`)

## Monitoring

During training, metrics are logged and checkpoint artifacts are dumped to `runs/detector/<timestamp>/`. You can monitor convergence using TensorBoard if installed:
```bash
tensorboard --logdir runs/
```

## Evaluation

Once complete, evaluate the model against the locked `test` split:

```bash
uv run container-id evaluate detector --model runs/detector/latest/best.pt --data data/processed/detection-v1/test
```
This produces precision-recall curves and optimal threshold calibrations.
