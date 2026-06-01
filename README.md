# lau-math-cuda

GPU-accelerated implementation of the core Lau math primitives. CUDA counterpart to [lau-math-c](https://github.com/SuperInstance/lau-math-c), handling operations where N > 16 and parallel fleet computation.

## GPU Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| **CUDA Toolkit** | 12.0+ | 12.4+ |
| **Compute Capability** | SM 7.0 (Volta) | SM 8.7+ (Ampere) |
| **VRAM** | 2 GB | 7 GB (RTX 4050) |
| **Driver** | R525+ | R550+ |

## SM Compatibility Matrix

| GPU | SM Version | Architecture | Status |
|---|---|---|---|
| **Jetson Orin** | 8.7 | Ampere | ✅ Primary target |
| **RTX 4050** | 8.9 | Ada Lovelace | ✅ Primary target |
| RTX 3090/4090 | 8.6/8.9 | Ampere/Ada | ✅ Supported |
| A100 | 8.0 | Ampere | ✅ Supported |
| V100 | 7.0 | Volta | ⚠️ Untested |
| GTX 1080 | 6.1 | Pascal | ❌ Not supported |

## Memory Budget (RTX 4050 — 7 GB VRAM)

| Operation | Matrix Size | Batch | Memory | Feasible? |
|---|---|---|---|---|
| Agent fleet update | 16×16 | 1,000 | ~2 MB | ✅ |
| Agent fleet update | 64×64 | 1,000 | ~31 MB | ✅ |
| Agent fleet update | 256×256 | 1,000 | ~500 MB | ✅ |
| Agent fleet update | 512×512 | 1,000 | ~2 GB | ✅ |
| Eigendecomposition | 1024×1024 | 1 | ~34 MB | ✅ |
| Eigendecomposition | 4096×4096 | 1 | ~540 MB | ✅ |
| Eigendecomposition | 4096×4096 | 8 | ~4.3 GB | ✅ (tight) |
| Matrix multiply | 4096×4096 | 1 | ~384 MB | ✅ |
| Heat kernel (CSR) | 10K nodes | 1,000 timesteps | ~80 MB | ✅ |

### Worst Case Budget

```
Fleet of 1000 agents, each with 256×256 belief matrix:
  Belief matrices: 1000 × 256 × 256 × 8 = 500 MB
  Gradient buffers: 500 MB
  cuBLAS/cuSOLVER workspace: ~200 MB
  ────────────────────────────────────────
  Total: ~1.2 GB  (fits in 7 GB with headroom)
```

## Architecture

```
                    lau_dispatch.h  (Unified API)
                         │
                   lau_dispatch.cu  (Auto-detect GPU)
                    ╱      │      ╲
              ┌─ CUDA ──┐  │   ┌─ CPU Fallback ─┐
              │         │  │   │  (lau-math-c)   │
              ▼         ▼  ▼   ▼                 │
         Matrix    Laplacian  Heat    Agent    Conservation
          Ops        Ops     Kernel   Fleet      Ops
        (cuBLAS)  (cuSPARSE) (Taylor) (Kernel)  (Atomics)
                  (cuSOLVER)
```

## Modules

### `lau_matrix_cuda.cu` — CuBLAS Matrix Operations
- **Matrix multiply**: `lau_matmul()` — batched DGEMM via cuBLAS, 16×16 to 4096×4096
- **Batched inverse**: `lau_batch_inverse()` — cuSOLVER getrf/getri per matrix
- **Eigendecomposition**: `lau_batch_eigendecompose()` — cuSOLVER Dsyevd, symmetric matrices

### `lau_laplacian_cuda.cu` — Graph Laplacian on GPU
- **CSR construction**: `lau_build_laplacian_csr()` — parallel kernel builds L = D - A
- **Batched spectral gap**: `lau_batch_spectral_gap()` — power iteration per graph
- **Power iteration**: `lau_power_iteration()` — dominant eigenvalue via Rayleigh quotient

### `lau_heat_kernel_cuda.cu` — Parallel Heat Kernel
- **Batched computation**: `lau_heat_kernel_batched()` — Taylor series e^{-tL}, multiple t values
- **Multi-source**: `lau_heat_kernel_multi_source()` — fixed t, multiple source vectors
- **Buffer reuse**: Global workspace shared across calls to minimize allocations

### `lau_agent_fleet_cuda.cu` — Fleet Updates
- **Batched update**: `lau_fleet_batch_update()` — 1000+ agents simultaneously
- **Warp vote sync**: `lau_fleet_sync_vote()` — convergence detection across fleet
- **Per-agent belief**: Each agent maintains its own N×N matrix on GPU

### `lau_conservation_cuda.cu` — Conservation Laws
- **Noether charge**: `lau_verify_conservation_batch()` — parallel ratio computation
- **CRDT merge**: `lau_crdt_merge()` — atomic averaging of divergent states
- **Fleet reduction**: `lau_conservation_reduce()` — global conservation ratio

### `lau_dispatch.cu/h` — Dispatch Layer
- Auto-detects GPU capability at init
- Routes to CUDA kernels when available
- Falls back to CPU (lau-math-c) when no GPU present
- Unified API: same code compiles with or without CUDA

## Building

```bash
# Prerequisites
# - CUDA Toolkit 12+ with nvcc in PATH
# - cuBLAS, cuSOLVER, cuSPARSE development headers

make all           # Build static + shared libraries
make test          # Build test binary
make run_test      # Build and run 65 tests
make clean         # Remove artifacts
```

## Testing

65 tests covering all modules:
- **Initialization**: GPU detection, SM version, VRAM query
- **Matrix multiply**: 1×1 through 4096×4096, batched, rectangular
- **Matrix inverse**: 2×2 through 256×256, roundtrip verification
- **Eigendecomposition**: identity, diagonal, symmetric, batched
- **Laplacian**: CSR construction, spectral gap, power iteration
- **Heat kernel**: zero-t, conservation, batching, combined pipelines
- **Agent fleet**: 1/10/100/1000 agents, convergence voting
- **Conservation**: perfect/drift conservation, CRDT merge, 10K agents
- **Memory benchmarks**: VRAM budget validation for 7 GB target
- **Performance benchmarks**: timing for key operations

Tests auto-skip GPU-dependent operations when CUDA is unavailable.

## Integration with lau-math-c

The dispatch layer is designed to call into `lau-math-c` for CPU fallback:
- Link against both `liblau_math_cuda.a` and `liblau_math_c.a`
- Call `lau_init()` at startup — it auto-detects and routes
- Same header (`lau_dispatch.h`) works everywhere

## License

Part of the Lau framework. See parent repository for license information.
