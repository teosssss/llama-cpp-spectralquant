# SpectralQuant: PCA-Based KV Cache Compression for Apple Silicon

## Overview

SpectralQuant extends TurboQuant with **calibration-based PCA rotation** for improved quality. Instead of using a fixed rotation matrix (WHT), SpectralQuant learns an optimal per-region rotation matrix from representative data, resulting in better compression quality for model-specific KV distributions.

### Shared Design with TurboQuant Tom implementation

Like TurboQuant, SpectralQuant uses several optimizations:
- **Pre-rotate queries**: Rotates Q once at attention time using precomputed matrices, avoiding redundant rotation during K dequant at every decode step.
- **QJL removed**: Uses MSE-only quantization (dropping QJL from the original paper) to dedicate all bits to PolarQuant.
- **Asymmetric K/V support**: Supports mixed quantization types (e.g., K=q8_0, V=turbo3_pca).
- **Flash Attention required**: Turbo types require flash attention for correct computation.

### SpectralQuant rotation

SpectralQuant uses **calibrated PCA rotation** instead of fixed WHT. The rotation matrices are computed from your specific model's KV cache patterns, not from a predetermined transformation.

#### Theoretical Background

SpectralQuant calibration builds an **uncentered second-moment matrix** on normalized 128-dim KV blocks:

1. **Normalize each block**: `x = block / ||block||`
2. **Accumulate second moment**: `M = E[x x^T]`
3. **Eigendecompose**: `M = U Lambda U^T`
4. **Rotate at runtime**: `y = U^T x`

The stored PCA statistics are in the **rotated PCA coordinate system**:

- `means[j] = E[y_j]`
- `variances[j] = Var(y_j) = E[y_j^2] - means[j]^2`

Runtime does **not** switch to centered PCA. It keeps the same rotation and only shifts the scalar codebook:

```text
centroid[j, k] = means[j] + q_k * sqrt(max(variances[j], 1e-6)) / sigma_ref
sigma_ref = 1 / sqrt(128)
```

So SpectralQuant uses:

- PCA rotation learned from calibration
- centered variance per rotated coordinate
- mean-shifted Turbo centroids in PCA space

## Compression Rate

Compression rate is computed relative to FP16 KV cache:
```
compression = 16 / avg_bpv
avg_bpv = (K_bpv + V_bpv) / 2
```

## Supported Types

| Type | Bits/val | Storage | Compression vs FP16 | Description |
|------|----------|---------|-------------|-------------|
| `turbo3_pca` | 3.125 | 50 bytes/128 | ~5.1x | 3-bit per 32 elements, PCA rotation |
| `turbo4_pca` | 4.25 | 68 bytes/128 | ~3.8x | 4-bit per 32 elements, PCA rotation |
| `turbo4333_pca` | 3.25 | 52 bytes/128 | ~4.9x | 128-dim split 4x32: 4-3-3-3 bits |
| `turbo4322_pca` | 2.75 | 44 bytes/128 | ~5.8x | 128-dim split 4x32: 4-3-2-2 bits |
| `turbo4211_pca` | 2.25 | 36 bytes/128 | ~7.1x | 128-dim split 4x32: 4-2-1-1 bits |

### Per-Region Compression (turbo4333_pca, turbo4322_pca, turbo4211_pca)

The 128-dim group is split into 4 regions of 32 elements each, aligned with Metal's SIMD group size. PCA orders components by importance (highest variance first), so the first region captures the most signal and receives higher bit precision.

Each region stores: 1 norm value (32 bits) + centroid indices (region_bits × 32 elements):

- **turbo4333_pca**: 4-3-3-3 bits per region
  - `(4+4 + 3+4 + 3+4 + 3+4) × 32 / 128 = 3.25 bits/val`
- **turbo4322_pca**: 4-3-2-2 bits per region
  - `(4+4 + 3+4 + 2+4 + 2+4) × 32 / 128 = 2.75 bits/val`
- **turbo4211_pca**: 4-2-1-1 bits per region
  - `(4+4 + 2+4 + 1+4 + 1+4) × 32 / 128 = 2.25 bits/val`

**Requires calibration**: Unlike turbo3/turbo4, spectralquant types require a calibration JSON file computed from your model.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4/M5)
- Metal backend enabled
- Model with head_dim divisible by 128

## Calibration

### Why Calibrate?

PCA rotation matrices are optimized for specific model architectures. Calibration collects statistics from your model's actual KV cache patterns to compute optimal rotation matrices per layer/region.

### Running Calibration

```bash
# Build the llama-kv-calibrate tool (included in build)
cmake --build build --target llama-kv-calibrate

# Run PCA calibration with representative text (shorthand for --kv-empvar-calibrate with --kv-calibration-mode turbo3_pca)
./build/bin/llama-kv-calibrate \
  -m models/YOUR_MODEL.gguf \
  -f calibration_text.txt \
  --kv-pca-calibrate \
  --kv-empvar-calibrate-out calibration.json \
  -c 512 \
  --chunks 4
```

**Calibration text**: Use 1000-10000 tokens of representative text (model's training domain works best). More data = better rotation matrices.

**Context size**: Use your target context size (`-c`). Larger contexts capture more KV patterns.

### Output

The tool generates `calibration.json` containing:
- Per-region rotation matrices (128x128 matrices per layer)
- Per-region rotated-space `means`
- Per-region rotated-space centered `variances`
- Model hash for verification
- Statistics for debugging

## Running Inference

### Basic Usage

```bash
# Set the calibration file
export GGML_METAL_TURBO_PCA_JSON_FILE=calibration.json

# Run with turbo3_pca KV cache
./build/bin/llama-cli \
  -m models/YOUR_MODEL.gguf \
  -ctk turbo3_pca \
  -ctv turbo3_pca \
  -p "Your prompt here"
```

### Asymmetric Mode (Mixed K/V Types)

SpectralQuant supports asymmetric K/V types for quality/speed tradeoffs:

```bash
# K=q8_0 for attention scoring, V=turbo3_pca for memory
./build/bin/llama-cli \
  -m models/YOUR_MODEL.gguf \
  -ctk q8_0 \
  -ctv turbo3_pca \
  -p "Your prompt here"
```

This uses q8_0 for K (higher precision for attention computation) while compressing V with turbo3_pca.

## Running Tests

### Perplexity Test

```bash
# Set calibration file
export GGML_METAL_TURBO_PCA_JSON_FILE=calibration.json

# Run perplexity
./build/bin/llama-perplexity \
  -m models/YOUR_MODEL.gguf \
  -f wikitext/test.raw \
  -c 512 \
  -ctk turbo3_pca \
  -ctv turbo3_pca \
  --chunks 10
```

`llama-bench` is currently not a valid validation path for these PCA cache types in this branch. It does not account for the new cache-type requirements and can give misleading results.

Use `llama-cli` for manual decode checks and `llama-perplexity` for quantitative comparisons instead.

## Troubleshooting

**"turbo*_pca cache type requires a precomputed calibration JSON"**
- Set `GGML_METAL_TURBO_PCA_JSON_FILE` environment variable
- Or ensure calibration.json is in the current directory

**"calibration JSON is for a different model"**
- Regenerate calibration for the specific model
- Each model needs its own calibration file

**Slow inference**
- Ensure Metal is actually being used (check for "Metal" in logs)
- Verify calibration file loaded correctly

## Technical Notes

- PCA rotation operates on 128-element groups (matching head_dim)
- V uses transposed rotation for dequant path efficiency
- Calibration runs inference locally - no data leaves your machine
- Pre-rotate queries: Q is rotated once at attention time, not K during dequant
- 128-dim groups split into 4x32 regions aligned with Metal SIMD group size
- Per-region bit allocation prioritizes high-variance PCA components
- PCA calibration stores rotated-space means plus centered variances
- Runtime codebooks are mean-shifted, not zero-centered
