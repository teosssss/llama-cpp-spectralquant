# PCA Plan for Metal Attention

This document describes how Turbo PCA compression is implemented in the Metal attention path, including rotated-space means and centered variances.

## Overview

The Metal implementation of Turbo PCA is split across three layers:

1. **Graph-level rotations** (`src/llama-graph.cpp`): Applies the PCA rotation matrices at graph construction time.
2. **Metal quant/dequant** (`ggml/src/ggml-metal/ggml-metal.metal`): Quantizes and dequantizes in PCA space, using rotated-space means and centered variances.
3. **Metal dispatch/glue** (`ggml/src/ggml-metal/ggml-metal-ops.cpp`): Loads PCA calibration tables and binds rotation, variance, and mean buffers to Metal kernels.

## PCA Statistics

The PCA calibration does **not** use centered PCA at runtime.

Calibration computes, per 128-dim group:

- `x = block / ||block||`
- `M = E[x x^T]`
- `M = U Lambda U^T`
- `y = U^T x`
- `means[j] = E[y_j]`
- `variances[j] = Var(y_j) = E[y_j^2] - means[j]^2`

Both `means` and `variances` live in the **rotated PCA coordinate system**. They are not original-space statistics.

Runtime keeps the existing PCA rotation:

- quantization works on `y = U^T x`
- dequantization reconstructs `y_hat`
- graph code still rotates queries and outputs as before

The codebook for each PCA coordinate is:

```text
c[j, k] = means[j] + q_k * sqrt(max(variances[j], 1e-6)) / sigma_ref
sigma_ref = 1 / sqrt(128)
```

So the change versus the old PCA path is:

- old: zero-centered codebook scaled by second moment
- new: mean-shifted codebook scaled by centered variance

## Graph-Level PCA Rotations (Standard Attention)

Standard MHA/MQA attention uses three graph-level rotations for Turbo PCA:

### K/V Cache Rotations

When storing new tokens to the KV cache, the K and V tensors are rotated before being written:

```cpp
// src/llama-graph.cpp, line 2178
if (llm_graph_is_turbo_pca_type(k->type)) {
    k_cur = llm_graph_apply_turbo_pca(ctx0, k_cur, mctx_cur->get_turbo_pca_k_rot_t());
}
// src/llama-graph.cpp, line 2181
if (llm_graph_is_turbo_pca_type(v->type)) {
    v_cur = llm_graph_apply_turbo_pca(ctx0, v_cur, mctx_cur->get_turbo_pca_v_rot_t());
}
```

- **`llm_graph_apply_turbo_pca`**: Applies rotation using `U @ x` where `U` is the (non-transposed) rotation matrix (via `.get_turbo_pca_v_rot()`).
- **`llm_graph_apply_turbo_pca_from_t`**: Applies rotation using `U^T @ x` where `U^T` is the transposed rotation matrix (via `.get_turbo_pca_k_rot_t()`, `.get_turbo_pca_v_rot_t()`).

The rotation matrices are 128×128 and stored as 3D tensors `[128, 128, n_groups]`. For non-128-aligned head dimensions, padding is applied before multiplication.

### Q Rotation into K Basis

Queries are rotated into the K basis for attention computation:

```cpp
// src/llama-graph.cpp, line 2196
if (llm_graph_is_turbo_pca_type(k->type)) {
    q = llm_graph_apply_turbo_pca(ctx0, q, mctx_cur->get_turbo_pca_k_rot_t());
}
```

### Attention Output Mapped Back with V Basis

After attention computation, the output is inverse-rotated using the V basis to restore the original representation:

```cpp
// src/llama-graph.cpp, line 2214
if (llm_graph_is_turbo_pca_type(v->type) || llm_graph_is_turbo_wht_type(v->type)) {
    // Extract original V head_dim after inverse WHT
    cur = llm_graph_apply_turbo_pca(ctx0, cur, mctx ? mctx->get_turbo_pca_v_rot() : nullptr);
}
```

## Graph-Level PCA Rotations (MLA)

Multi-head latent attention (MLA) has a different rotation strategy because the V head dimension differs from K/Q (e.g., V=512, K/Q=576 in DeepSeek-V2). MLA applies rotations at two distinct stages:

### K/Q-Side Rotations

These rotations use the K-basis (`.get_turbo_pca_k_rot_t()`) because both K and Q need to share the same basis for attention computation:

```cpp
// src/llama-graph.cpp, line 1917 (Flash Attention path)
if (v_mla) {
    cur = llm_graph_apply_turbo_pca_from_t(ctx0, cur, mctx ? mctx->get_turbo_pca_k_rot_t() : nullptr);
}

// src/llama-graph.cpp, line 2001 (non-FA path)
if (v_mla) {
    kqv = llm_graph_apply_turbo_pca_from_t(ctx0, kqv, mctx ? mctx->get_turbo_pca_k_rot_t() : nullptr);
}
```

- **`llm_graph_apply_turbo_pca_from_t`** performs `U^T @ x` (transposed rotation) because the output must be projected from the K-basis space.

### V/Output-Side Rotations

These rotations use the V-basis (`.get_turbo_pca_v_rot()`) to recover the original V representation after attention:

```cpp
// src/llama-graph.cpp, line 2019
if (!v_mla) {
    kqv = llm_graph_apply_turbo_pca(ctx0, kqv, mctx ? mctx->get_turbo_pca_v_rot() : nullptr);
}
```

The distinction exists because:
- K, Q, and attention output operate in the K-basis space (shared across K/Q heads)
- V outputs must be mapped back to the V-basis space for the projection head

## Metal Flash Attention PCA Path

The Metal Flash Attention kernels bind separate K/V tables for both centered variances and means:

- `empvar_k`, `empvar_v`
- `empmean_k`, `empmean_v`
- matching `*_dim` integers

The same rule applies in vec and non-vec kernels:

- K dequant uses K PCA stats
- V dequant uses V PCA stats
- table coordinates are local PCA coordinates inside each 128-dim group

The dequant helpers reconstruct shifted centroids inline:

```text
mean  = empmean[coord]
var   = empvar[coord]
scale = sqrt(max(var, 1e-6)) / sigma_ref
value = mean + q[idx] * scale
```

This same formula is used by:

- Flash Attention dequant
- `SET_ROWS` quantization
- norm correction during quantization
- regular dequant helpers

## Metal Dispatch/Glue

The `ggml_metal_turbo_empvar_select` helper now selects both the variance table and the mean table, plus dims and metadata.

### Binding Logic

The selector reads tensor metadata to determine:

- whether the tensor uses WHT-only empvar or PCA stats
- whether the tensor is K or V
- which per-kind mean and variance tables to bind

`kv_kind` comes from tensor metadata:

- `1` = K
- `2` = V

For PCA types, this matters because K and V have different:

- rotations
- centered variances
- means

### State Initialization

The Metal-side calibration state now stores, for PCA-capable formats:

- K variances
- V variances
- K means
- V means

This state is populated from the PCA calibration JSON at model load time.

## Debugging Checklist

Use these search terms and file locations to debug Turbo PCA issues in Metal:

| Issue | Search Term | Location |
|-------|-------------|----------|
| PCA type detection | `llm_graph_is_turbo_pca_type` | `src/llama-graph.cpp:29` |
| Graph rotation functions | `llm_graph_apply_turbo_pca` | `src/llama-graph.cpp:44` |
| Transposed rotation | `llm_graph_apply_turbo_pca_from_t` | `src/llama-graph.cpp:74` |
| Kv cache rotation tensors | `get_turbo_pca_k_rot_t` | `src/llama-kv-cache.h:171` |
| MLA rotation divergence | `v_mla.*turbo_pca` | `src/llama-graph.cpp:1917` |
| Mean kernel args | `empmean_k` | `ggml/src/ggml-metal/ggml-metal.metal` |
| Variance kernel args | `empvar_k` | `ggml/src/ggml-metal/ggml-metal.metal` |
| Shifted centroid helpers | `scaled_shifted` | `ggml/src/ggml-metal/ggml-metal.metal` |
| Dispatch selector | `ggml_metal_turbo_empvar_select` | `ggml/src/ggml-metal/ggml-metal-ops.cpp` |
| Type classification | `ggml_metal_is_turbo3_empvar_type` | `ggml/src/ggml-metal/ggml-metal-ops.cpp:414` |
| State singleton | `ggml_metal_turbo_empvar_get_state` | `ggml/src/ggml-metal/ggml-metal-ops.cpp` |
