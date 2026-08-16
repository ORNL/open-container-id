# Training on macOS (MPS)

Open Container ID natively supports Apple's Metal Performance Shaders (MPS), providing significant acceleration on M-series chips (like the M4 Max) without relying on Dockerized Linux GPU runtimes.

## Environment Check

First, verify that your environment is utilizing native ARM64 Python and that PyTorch detects the MPS backend:

```bash
uv run container-id train doctor
```

Output should confirm `mps` device availability. If it reports `cpu`, ensure you installed Python natively (not via Rosetta) and have installed the proper PyTorch distribution.

## Best Practices & Troubleshooting

1. **Batch Size Tuning**: Start with a small batch size (`--batch 4`) and increment. MPS shares system RAM with the GPU (Unified Memory). If you run out of memory, the system will swap heavily and training will stall.
2. **Docker**: **Do not use Docker for MPS training.** Docker Desktop on macOS runs a Linux VM, which cannot pass through the Metal backend to the container. All training must occur on bare metal.
3. **Common Errors**:
   - `NotImplementedError: The operator 'aten::xyz.out' is not currently implemented for the MPS device.` -> PyTorch MPS support is still growing. If this occurs in RF-DETR or docTR, set `PYTORCH_ENABLE_MPS_FALLBACK=1` in your environment to force the unsupported op onto the CPU.
4. **Resuming**: Training runs write to the `runs/` directory. You can resume interrupted jobs using `--resume runs/detector/latest/weights.pt`.
