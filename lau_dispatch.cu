/*
 * lau_dispatch.cu — Dispatch layer: auto-detect GPU, unified API
 *
 * Routes to CUDA kernels when GPU available, falls back to CPU stubs.
 */

#include "lau_dispatch.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define CUDA_CHECK_QUIET(call)                                         \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "[dispatch] CUDA error: %s\n",             \
                    cudaGetErrorString(err));                           \
        }                                                               \
    } while (0)

/* ── Global state ──────────────────────────────────────────────── */

static struct {
    int initialized;
    LauBackend backend;
    char gpu_name[256];
    int sm_version;
    size_t vram_bytes;
} g_dispatch = {0};

/* ── Capability query ──────────────────────────────────────────── */

LauBackend lau_get_backend(void) { return g_dispatch.backend; }
const char *lau_get_gpu_name(void) {
    return (g_dispatch.backend == LAU_BACKEND_CUDA) ? g_dispatch.gpu_name : NULL;
}
int lau_get_sm_version(void) {
    return (g_dispatch.backend == LAU_BACKEND_CUDA) ? g_dispatch.sm_version : 0;
}
size_t lau_get_vram_bytes(void) {
    return (g_dispatch.backend == LAU_BACKEND_CUDA) ? g_dispatch.vram_bytes : 0;
}

/* ── Init / Shutdown ───────────────────────────────────────────── */

int lau_init(void)
{
    if (g_dispatch.initialized) return 0;

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "[dispatch] No CUDA GPU detected, using CPU fallback\n");
        g_dispatch.backend = LAU_BACKEND_CPU;
        g_dispatch.gpu_name[0] = '\0';
        g_dispatch.sm_version = 0;
        g_dispatch.vram_bytes = 0;
        g_dispatch.initialized = 1;
        return 0;
    }

    /* Use device 0 */
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[dispatch] Failed to query GPU properties\n");
        g_dispatch.backend = LAU_BACKEND_CPU;
        g_dispatch.initialized = 1;
        return 0;
    }

    g_dispatch.backend = LAU_BACKEND_CUDA;
    strncpy(g_dispatch.gpu_name, prop.name, sizeof(g_dispatch.gpu_name) - 1);
    g_dispatch.sm_version = prop.major * 10 + prop.minor;
    g_dispatch.vram_bytes = prop.totalGlobalMem;

    printf("[dispatch] GPU: %s (SM %d.%d, %.1f GB VRAM)\n",
           prop.name, prop.major, prop.minor,
           (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    /* Validate SM version */
    if (prop.major < 8) {
        fprintf(stderr, "[dispatch] WARNING: GPU compute capability %d.%d is below "
                "recommended SM 8.7 (Ampere). Some kernels may not compile.\n",
                prop.major, prop.minor);
    }

    g_dispatch.initialized = 1;
    return 0;
}

void lau_shutdown(void)
{
    if (!g_dispatch.initialized) return;

    if (g_dispatch.backend == LAU_BACKEND_CUDA) {
        cudaError_t err = cudaDeviceReset();
        if (err != cudaSuccess) {
            fprintf(stderr, "[dispatch] cudaDeviceReset failed: %s\n",
                    cudaGetErrorString(err));
        }
    }

    memset(&g_dispatch, 0, sizeof(g_dispatch));
}

/* ── CPU fallback stubs ────────────────────────────────────────── */
/* These provide basic functionality when no GPU is available.
 * For full CPU implementation, link against lau-math-c. */

static int cpu_matmul(const double *A, const double *B, double *C,
                      int M, int K, int N, int batch_size)
{
    for (int b = 0; b < batch_size; b++) {
        const double *Ab = A + (size_t)b * M * K;
        const double *Bb = B + (size_t)b * K * N;
        double *Cb = C + (size_t)b * M * N;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                double sum = 0.0;
                for (int p = 0; p < K; p++) {
                    sum += Ab[i * K + p] * Bb[p * N + j];
                }
                Cb[i * N + j] = sum;
            }
        }
    }
    return 0;
}

static int cpu_batch_inverse(const double *A, double *Ainv,
                             int N, int batch_size)
{
    /* Gauss-Jordan elimination for each matrix */
    for (int b = 0; b < batch_size; b++) {
        const double *Ab = A + (size_t)b * N * N;
        double *Ib = Ainv + (size_t)b * N * N;

        /* Copy A into Ainv, augment with identity */
        double *aug = (double *)calloc(N * 2 * N, sizeof(double));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                aug[i * 2 * N + j] = Ab[i * N + j];
            }
            aug[i * 2 * N + N + i] = 1.0;
        }

        /* Forward elimination */
        for (int col = 0; col < N; col++) {
            double pivot = aug[col * 2 * N + col];
            if (fabs(pivot) < 1e-15) { free(aug); return -4; }
            for (int j = 0; j < 2 * N; j++) aug[col * 2 * N + j] /= pivot;
            for (int row = 0; row < N; row++) {
                if (row == col) continue;
                double factor = aug[row * 2 * N + col];
                for (int j = 0; j < 2 * N; j++) {
                    aug[row * 2 * N + j] -= factor * aug[col * 2 * N + j];
                }
            }
        }

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                Ib[i * N + j] = aug[i * 2 * N + N + j];

        free(aug);
    }
    return 0;
}

/* ── Dispatch wrappers ─────────────────────────────────────────── */

/* Forward declarations for CUDA implementations */
extern int lau_matmul_cuda(const double *, const double *, double *,
                           int, int, int, int);
extern int lau_batch_inverse_cuda(const double *, double *, int, int);
extern int lau_batch_eigendecompose_cuda(const double *, double *,
                                         double *, int, int);

int lau_matmul(const double *A, const double *B, double *C,
               int M, int K, int N, int batch_size)
{
    if (g_dispatch.backend == LAU_BACKEND_CUDA) {
        return lau_matmul_cuda(A, B, C, M, K, N, batch_size);
    }
    return cpu_matmul(A, B, C, M, K, N, batch_size);
}

int lau_batch_inverse(const double *A, double *Ainv,
                      int N, int batch_size)
{
    if (g_dispatch.backend == LAU_BACKEND_CUDA) {
        return lau_batch_inverse_cuda(A, Ainv, N, batch_size);
    }
    return cpu_batch_inverse(A, Ainv, N, batch_size);
}

int lau_batch_eigendecompose(const double *A, double *eigenvalues,
                             double *eigenvectors, int N, int batch_size)
{
    if (g_dispatch.backend == LAU_BACKEND_CUDA) {
        return lau_batch_eigendecompose_cuda(A, eigenvalues, eigenvectors,
                                             N, batch_size);
    }
    /* CPU fallback: stub - not implemented for CPU */
    fprintf(stderr, "[dispatch] Eigendecomposition CPU fallback not implemented\n");
    (void)A; (void)eigenvalues; (void)eigenvectors; (void)N; (void)batch_size;
    return -99;
}

/* Stub dispatch for Laplacian ops (CPU fallback not implemented) */
int lau_build_laplacian_csr(const int *row_ptr, const int *col_idx,
                            const double *adj_weights, double *vals_out,
                            int num_nodes, int nnz)
{
    if (g_dispatch.backend == LAU_BACKEND_CUDA) {
        return lau_build_laplacian_csr_cuda(row_ptr, col_idx, adj_weights,
                                            vals_out, num_nodes, nnz);
    }
    fprintf(stderr, "[dispatch] Laplacian CSR CPU fallback not implemented\n");
    return -99;
}

int lau_batch_spectral_gap(const int *row_ptrs, const int *col_idxs,
                           const double *vals, const int *graph_sizes,
                           const int *graph_nnz, double *spectral_gaps,
                           int fleet_size)
{
    if (g_dispatch.backend == LAU_BACKEND_CUDA) {
        return lau_batch_spectral_gap_cuda(row_ptrs, col_idxs, vals,
                                           graph_sizes, graph_nnz,
                                           spectral_gaps, fleet_size);
    }
    fprintf(stderr, "[dispatch] Spectral gap CPU fallback not implemented\n");
    return -99;
}

int lau_power_iteration(const int *row_ptr, const int *col_idx,
                        const double *vals, int N, int nnz,
                        double *dominant_eigenvalue, double *eigenvector_out,
                        int max_iter, double tol)
{
    if (g_dispatch.backend == LAU_BACKEND_CUDA) {
        return lau_power_iteration_cuda(row_ptr, col_idx, vals, N, nnz,
                                        dominant_eigenvalue, eigenvector_out,
                                        max_iter, tol);
    }
    fprintf(stderr, "[dispatch] Power iteration CPU fallback not implemented\n");
    return -99;
}

int lau_heat_kernel_batched(const int *row_ptr, const int *col_idx,
                            const double *vals, int N, int nnz,
                            const double *x, double *y,
                            const double *t_values, int batch_size)
{
    if (g_dispatch.backend == LAU_BACKEND_CUDA) {
        return lau_heat_kernel_batched_cuda(row_ptr, col_idx, vals, N, nnz,
                                            x, y, t_values, batch_size);
    }
    fprintf(stderr, "[dispatch] Heat kernel CPU fallback not implemented\n");
    return -99;
}

int lau_heat_kernel_multi_source(const int *row_ptr, const int *col_idx,
                                 const double *vals, int N, int nnz,
                                 const double *sources, double *results,
                                 double t, int num_sources)
{
    if (g_dispatch.backend == LAU_BACKEND_CUDA) {
        return lau_heat_kernel_multi_source_cuda(row_ptr, col_idx, vals, N, nnz,
                                                 sources, results, t, num_sources);
    }
    fprintf(stderr, "[dispatch] Heat kernel multi-source CPU fallback not implemented\n");
    return -99;
}

int lau_fleet_batch_update(double *belief_matrices,
                           const double *updates,
                           int mat_dim, int num_agents,
                           double learning_rate)
{
    if (g_dispatch.backend == LAU_BACKEND_CUDA) {
        return lau_fleet_batch_update_cuda(belief_matrices, updates,
                                           mat_dim, num_agents, learning_rate);
    }
    /* CPU fallback */
    int mat_size = mat_dim * mat_dim;
    for (int a = 0; a < num_agents; a++) {
        for (int i = 0; i < mat_size; i++) {
            belief_matrices[(size_t)a * mat_size + i] +=
                learning_rate * updates[(size_t)a * mat_size + i];
        }
    }
    return 0;
}

int lau_fleet_sync_vote(const int *convergence_flags,
                        int *all_converged, int num_agents)
{
    int result = 1;
    for (int i = 0; i < num_agents; i++) result &= convergence_flags[i];
    *all_converged = result;
    return 0;
}

int lau_verify_conservation_batch(const double *charges_in,
                                  const double *charges_out,
                                  double *ratios, int num_agents)
{
    if (g_dispatch.backend == LAU_BACKEND_CUDA) {
        return lau_verify_conservation_batch_cuda(charges_in, charges_out,
                                                  ratios, num_agents);
    }
    for (int i = 0; i < num_agents; i++) {
        ratios[i] = (fabs(charges_in[i]) < 1e-15) ? 0.0 : charges_out[i] / charges_in[i];
    }
    return 0;
}

int lau_crdt_merge(const double *local_states, double *merged_out,
                   int state_len, int num_agents)
{
    if (g_dispatch.backend == LAU_BACKEND_CUDA) {
        return lau_crdt_merge_cuda(local_states, merged_out, state_len, num_agents);
    }
    for (int i = 0; i < state_len; i++) {
        double sum = 0.0;
        for (int a = 0; a < num_agents; a++) sum += local_states[(size_t)a * state_len + i];
        merged_out[i] = sum / num_agents;
    }
    return 0;
}

int lau_conservation_reduce(const double *ratios, double *global_ratio,
                            int num_agents)
{
    if (g_dispatch.backend == LAU_BACKEND_CUDA) {
        return lau_conservation_reduce_cuda(ratios, global_ratio, num_agents);
    }
    double sum = 0.0;
    for (int i = 0; i < num_agents; i++) sum += ratios[i];
    *global_ratio = sum / num_agents;
    return 0;
}
