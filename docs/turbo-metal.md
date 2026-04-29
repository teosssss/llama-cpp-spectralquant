# Turbo PCA/Empvar in Metal Attention

This document describes how Turbo PCA (Principal Component Analysis) compression and empirical variance (empvar) scaling are implemented in the Metal attention path.

## Overview

The Metal implementation of Turbo PCA/empvar is split across three layers:

1. **Graph-level rotations** (`src/llama-graph.cpp`): Handles tensor reshaping, transpositions, and matrix multiplications that apply the PCA rotation matrices at graph construction time.

2. **Metal FA dequant/empvar scaling** (`ggml/src/ggml-metal/ggml-metal.metal`): Provides optimized dequantization functions that apply empirical variance scaling within the Metal kernel during Flash Attention.

3. **Metal dispatch/glue** (`ggml/src/ggml-metal/ggml-metal-ops.cpp`): Binds empirical variance buffers and dimensions to Metal kernel arguments via `ggml_metal_turbo_empvar_select`.

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

## Metal Flash Attention Empvar Path

The Metal Flash Attention kernels support empirical variance scaling through dynamic buffers passed to the kernel.

### K and V Empvar Pointers

Two separate empirical variance tables are passed to the kernel:

```cpp
// ggml/src/ggml-metal/ggml-metal.metal, line 7569-7572
constant float * empvar_k [[buffer(8)]],
constant float * empvar_v [[buffer(9)]],
constant int32_t & empvar_k_dim [[buffer(10)]],
constant int32_t & empvar_v_dim [[buffer(11)]],

// ggml/src/ggml-metal/ggml-metal.metal, line 8706-8709 (non-vec path)
constant float * empvar_k [[buffer(9)]],
constant float * empvar_v [[buffer(10)]],
constant int32_t & empvar_k_dim [[buffer(11)]],
constant int32_t & empvar_v_dim [[buffer(12)]],
```

- `empvar_k`: Empirical variance array for K dequantization (dimension varies by calibration)
- `empvar_v`: Empirical variance array for V dequantization (dimension varies by calibration)

### Dequant Helper Calls

The dequant helpers apply scaling inline during dequantization:

```cpp
// ggml/src/ggml-metal/ggml-metal.metal, line 7727
deq_k_t4(pk + i/nl_k, i%nl_k, mk, empvar_k, empvar_k_dim, i*4);

// ggml/src/ggml-metal/ggml-metal.metal, line 7827
deq_v_t4(pv4 + i/nl_v, i%nl_v, mv, empvar_v, empvar_v_dim, i*4);
```

### `turbo_apply_empvar_scale_4x4`

The scaling is applied by the helper function:

```cpp
// ggml/src/ggml-metal/ggml-metal.metal, line 477
static inline void turbo_apply_empvar_scale_4x4(
        thread float4 & scaled,
        constant float * empvar,
        int empvar_dim,
        int base_coord) {
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        scaled[k] *= turbo_empvar_sigma_scale(empvar, empvar_dim, base_coord + g*4 + k);
    }
}
```

The `turbo_empvar_sigma_scale` function computes `sigma / sigma_ref` where `sigma = sqrt(max(empvar[coord], 1e-6))` and `sigma_ref = 1/sqrt(128)`.

## Metal Dispatch/Glue

The `ggml_metal_turbo_empvar_select` function binds empvar tables to Metal kernels:

```cpp
// ggml/src/ggml-metal/ggml-metal-ops.cpp, line 418
static void ggml_metal_turbo_empvar_select(
        const ggml_tensor * t,
        const float ** table,
        int32_t * dim,
        int32_t * mode,
        int32_t * wht_group,
        int32_t * kv_kind);
```

### Binding Logic

The function reads tensor metadata to determine which variance table to use:

```cpp
// ggml/src/ggml-metal/ggml-metal-ops.cpp, line 435-446
if (t != nullptr && ggml_metal_is_turbo3_empvar_type(t->type)) {
    const auto & empvar = ggml_metal_turbo_empvar_get_state();
    if (empvar.mode == GGML_METAL_TURBO_EMPVAR_WHT_ONLY) {
        if (local_kv_kind == 1) {
            local_table = empvar.k;
            local_dim = empvar.k_dim;
        } else if (local_kv_kind == 2) {
            local_table = empvar.v;
            local_dim = empvar.v_dim;
        }
    }
}
```

- `kv_kind`: Read from tensor metadata via `ggml_metal_turbo_read_meta`. Values are `1` for K, `2` for V.
- `mode`: Controls whether variance scaling is applied or baked into constants.

### State Initialization

The state is stored in `ggml_metal_turbo_empvar_state`:

```cpp
// ggml/src/ggml-metal/ggml-metal-ops.cpp, line 39
struct ggml_metal_turbo_empvar_state {
    bool  initialized = false;
    int32_t mode = GGML_METAL_TURBO_EMPVAR_DISABLED;
    int32_t k_dim = 0;
    int32_t v_dim = 0;
    float k[GGML_METAL_TURBO_EMPVAR_MAX_DIM] = {};
    float v[GGML_METAL_TURBO_EMPVAR_MAX_DIM] = {};
};
```

This state is populated from calibration data at model loading time.

## Debugging Checklist

Use these search terms and file locations to debug Turbo PCA/empvar issues in Metal:

| Issue | Search Term | Location |
|-------|-------------|----------|
| PCA type detection | `llm_graph_is_turbo_pca_type` | `src/llama-graph.cpp:29` |
| Graph rotation functions | `llm_graph_apply_turbo_pca` | `src/llama-graph.cpp:44` |
| Transposed rotation | `llm_graph_apply_turbo_pca_from_t` | `src/llama-graph.cpp:74` |
| Kv cache rotation tensors | `get_turbo_pca_k_rot_t` | `src/llama-kv-cache.h:171` |
| MLA rotation divergence | `v_mla.*turbo_pca` | `src/llama-graph.cpp:1917` |
| Empvar kernel args | `empvar_k.*buffer` | `ggml/src/ggml-metal/ggml-metal.metal:7569` |
| Empvar scale helper | `turbo_apply_empvar_scale_4x4` | `ggml/src/ggml-metal/ggml-metal.metal:477` |
| Dispatch selector | `ggml_metal_turbo_empvar_select` | `ggml/src/ggml-metal/ggml-metal-ops.cpp:418` |
| Type classification | `ggml_metal_is_turbo3_empvar_type` | `ggml/src/ggml-metal/ggml-metal-ops.cpp:414` |
| State singleton | `ggml_metal_turbo_empvar_get_state` | `ggml/src/ggml-metal/ggml-metal-ops.cpp:302` |