#ifndef LAU_DISPATCH_H
#define LAU_DISPATCH_H

/*
 * lau_dispatch.h — Unified API for Lau math primitives
 * Auto-dispatches to CUDA (GPU) or CPU (lau-math-c) fallback.
 *
 * Targets:
 *   - RTX 4050 (SM 8.9, 20 SMs, 7 GB VRAM)
 *   - Jetson Orin (SM 8.7, Ampere)
 */

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Capability query ─────────────────────────────────────────── */

typedef enum {
    LAU_BACKEND_CPU = 0,
    LAU_BACKEND_CUDA = 1,
} LauBackend;

/* Returns which backend will be used. Call after lau_init(). */
LauBackend lau_get_backend(void);

/* Returns GPU name string (NULL if CPU fallback). */
const char *lau_get_gpu_name(void);

/* Returns SM version (e.g. 89 for SM 8.9), 0 if no GPU. */
int lau_get_sm_version(void);

/* Returns available VRAM in bytes, 0 if no GPU. */
size_t lau_get_vram_bytes(void);

/* ── Initialization / Teardown ────────────────────────────────── */

/* Initialize the dispatch layer. Detects GPU, loads CPU fallback if needed.
 * Returns 0 on success, nonzero on error. */
int lau_init(void);

/* Teardown all resources. */
void lau_shutdown(void);

/* ── Matrix operations (lau_matrix_cuda) ──────────────────────── */

/* C = A * B   where A is MxK, B is KxN, C is MxN.
 * batch_size > 1 performs batched multiply. */
int lau_matmul(const double *A, const double *B, double *C,
               int M, int K, int N, int batch_size);

/* Invert N_out matrices of dim N each, stored contiguously. */
int lau_batch_inverse(const double *A, double *A_inv,
                      int N, int batch_size);

/* Eigendecomposition: eigenvalues + eigenvectors for batch of N-dim matrices. */
int lau_batch_eigendecompose(const double *A, double *eigenvalues,
                             double *eigenvectors, int N, int batch_size);

/* ── Laplacian operations (lau_laplacian_cuda) ────────────────── */

/* Build graph Laplacian L = D - A in sparse CSR format.
 * nnz_diag_row: number of non-zeros per row (degree + 1 for diagonal).
 * Returns 0 on success, fills row_ptr/col_idx/vals. */
int lau_build_laplacian_csr(const int *row_ptr, const int *col_idx,
                            const double *adj_weights,
                            double *vals_out,
                            int num_nodes, int nnz);

/* Batched spectral gap (smallest nonzero eigenvalue) for fleet_size graphs. */
int lau_batch_spectral_gap(const int *row_ptrs, const int *col_idxs,
                           const double *vals,
                           const int *graph_sizes, const int *graph_nnz,
                           double *spectral_gaps,
                           int fleet_size);

/* Power iteration on GPU: dominant eigenvalue of sparse matrix. */
int lau_power_iteration(const int *row_ptr, const int *col_idx,
                        const double *vals, int N, int nnz,
                        double *dominant_eigenvalue, double *eigenvector_out,
                        int max_iter, double tol);

/* ── Heat kernel (lau_heat_kernel_cuda) ───────────────────────── */

/* Compute H = e^{-tL} applied to a vector x, batched across agents.
 * L is in CSR format. Results in y[batch_size * N]. */
int lau_heat_kernel_batched(const int *row_ptr, const int *col_idx,
                            const double *vals, int N, int nnz,
                            const double *x, double *y,
                            const double *t_values, int batch_size);

/* Heat kernel at fixed t for multiple source vectors. */
int lau_heat_kernel_multi_source(const int *row_ptr, const int *col_idx,
                                 const double *vals, int N, int nnz,
                                 const double *sources, double *results,
                                 double t, int num_sources);

/* ── Agent fleet (lau_agent_fleet_cuda) ───────────────────────── */

/* Agent belief state: an NxN matrix + metadata. */
typedef struct {
    double *belief_matrix;   /* NxN, row-major */
    double *gradient;        /* NxN workspace */
    double noether_charge;   /* conservation invariant */
    int N;                   /* matrix dimension */
    int id;                  /* agent id */
} LauAgent;

/* Batched agent update: update num_agents agents simultaneously.
 * Each agent has its own belief matrix of dimension mat_dim.
 * belief_matrices: [num_agents * mat_dim * mat_dim]
 * updates:         [num_agents * mat_dim * mat_dim]  (gradient step)
 * learning_rate:   scalar step size */
int lau_fleet_batch_update(double *belief_matrices,
                           const double *updates,
                           int mat_dim, int num_agents,
                           double learning_rate);

/* Global synchronization across fleet via warp vote.
 * convergence_flags: [num_agents], set to 1 when agent converged. */
int lau_fleet_sync_vote(const int *convergence_flags,
                        int *all_converged,
                        int num_agents);

/* ── Conservation (lau_conservation_cuda) ─────────────────────── */

/* Verify Noether charge conservation for each agent.
 * charges_in: [num_agents] pre-step charges.
 * charges_out: [num_agents] post-step charges.
 * ratios: [num_agents] output ratio (charge_out / charge_in). */
int lau_verify_conservation_batch(const double *charges_in,
                                  const double *charges_out,
                                  double *ratios,
                                  int num_agents);

/* CRDT merge: merge divergent belief states using atomic operations.
 * local_states: [num_agents * state_len] divergent copies.
 * merged_out:   [state_len] consensus result. */
int lau_crdt_merge(const double *local_states, double *merged_out,
                   int state_len, int num_agents);

/* Conservation ratio reduction across fleet. */
int lau_conservation_reduce(const double *ratios, double *global_ratio,
                            int num_agents);

#ifdef __cplusplus
}
#endif

#endif /* LAU_DISPATCH_H */
