#include "tcmodel.h"
#include "tc_ops.h"
#include "tc_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void check_equal(const float *cpu, const float *gpu, int n, const char *name) {
    float max_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(cpu[i] - gpu[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("%-20s: max diff = %.6f\n", name, max_diff);
    if (n >= 5) {
        printf("  CPU[0..4]: %.4f %.4f %.4f %.4f %.4f\n", cpu[0], cpu[1], cpu[2], cpu[3], cpu[4]);
        printf("  GPU[0..4]: %.4f %.4f %.4f %.4f %.4f\n", gpu[0], gpu[1], gpu[2], gpu[3], gpu[4]);
    }
    if (max_diff > 1e-4f) {
        printf("FAIL: %s exceeded tolerance.\n", name);
        exit(1);
    }
}

#define ALIGN16K(x) (((x) + 16383) & ~16383)

int main(void) {
    srand(42);
    printf("--- tc_metalcheck ---\n");
    if (!tc_metal_init(NULL)) {
        printf("Metal not available.\n");
        return 1;
    }

    int M = 128, N = 128, K = 256;
    float *A, *B, *C_cpu, *C_gpu;
    
    posix_memalign((void**)&A, 16384, (size_t)M * K * sizeof(float));
    posix_memalign((void**)&B, 16384, (size_t)N * K * sizeof(float));
    posix_memalign((void**)&C_cpu, 16384, (size_t)M * N * sizeof(float));
    posix_memalign((void**)&C_gpu, 16384, (size_t)M * N * sizeof(float));

    // Fill with random
    for (int i = 0; i < M * K; i++) A[i] = ((float)rand() / RAND_MAX) - 0.5f;
    for (int i = 0; i < N * K; i++) B[i] = ((float)rand() / RAND_MAX) - 0.5f;
    memset(C_cpu, 0, (size_t)M * N * sizeof(float));
    memset(C_gpu, 0, (size_t)M * N * sizeof(float));

    // CPU test
    // tc_gemm_nt(C_cpu, A, B, M, N, K);

    // GPU test
    // tc_metal_gemm_nt(C_gpu, A, B, M, N, K);
    // tc_metal_commit_and_wait();

    // check_equal(C_cpu, C_gpu, M * N, "tc_metal_gemm_nt");

    // RMSNorm test
    int T = 64, D = 256;
    float *x, *g, *rms_cpu, *xhat_cpu, *y_cpu;
    float *rms_gpu, *xhat_gpu, *y_gpu;
    
    posix_memalign((void**)&x, 16384, (size_t)T * D * sizeof(float));
    posix_memalign((void**)&g, 16384, (size_t)D * sizeof(float));
    posix_memalign((void**)&rms_cpu, 16384, (size_t)T * sizeof(float));
    posix_memalign((void**)&xhat_cpu, 16384, (size_t)T * D * sizeof(float));
    posix_memalign((void**)&y_cpu, 16384, (size_t)T * D * sizeof(float));
    
    posix_memalign((void**)&rms_gpu, 16384, (size_t)T * sizeof(float));
    posix_memalign((void**)&xhat_gpu, 16384, (size_t)T * D * sizeof(float));
    posix_memalign((void**)&y_gpu, 16384, (size_t)T * D * sizeof(float));

    for (int i = 0; i < T * D; i++) x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < D; i++) g[i] = ((float)rand() / RAND_MAX) + 0.5f;

    tc_rmsnorm_fwd(x, g, rms_cpu, xhat_cpu, y_cpu, T, D, 1e-5f);
    
    tc_metal_rmsnorm_fwd(x, g, rms_gpu, xhat_gpu, y_gpu, T, D, 1e-5f);
    tc_metal_commit_and_wait();

    check_equal(rms_cpu, rms_gpu, T, "rmsnorm_rms");
    check_equal(xhat_cpu, xhat_gpu, T * D, "rmsnorm_xhat");
    check_equal(y_cpu, y_gpu, T * D, "rmsnorm_y");

    // SwiGLU test
    int FFD = 512;
    float *hgate, *hup, *hsilu_cpu, *hsilu_gpu;
    posix_memalign((void**)&hgate, 16384, (size_t)T * FFD * sizeof(float));
    posix_memalign((void**)&hup, 16384, (size_t)T * FFD * sizeof(float));
    posix_memalign((void**)&hsilu_cpu, 16384, (size_t)T * FFD * sizeof(float));
    posix_memalign((void**)&hsilu_gpu, 16384, (size_t)T * FFD * sizeof(float));
    
    for (int i = 0; i < T * FFD; i++) {
        hgate[i] = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;
        hup[i] = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;
    }
    
    tc_swiglu_fwd(hgate, hup, hsilu_cpu, T, FFD);
    tc_metal_swiglu_fwd(hgate, hup, hsilu_gpu, T, FFD);
    tc_metal_commit_and_wait();
    
    check_equal(hsilu_cpu, hsilu_gpu, T * FFD, "swiglu_fwd");
    
    // WKV test
    float *k, *v, *r_sig, *decay, *state_cpu, *state_gpu, *wkv_cpu, *wkv_gpu;
    posix_memalign((void**)&k, 16384, (size_t)T * D * sizeof(float));
    posix_memalign((void**)&v, 16384, (size_t)T * D * sizeof(float));
    posix_memalign((void**)&r_sig, 16384, (size_t)T * D * sizeof(float));
    posix_memalign((void**)&decay, 16384, (size_t)D * sizeof(float));
    posix_memalign((void**)&state_cpu, 16384, (size_t)T * D * sizeof(float));
    posix_memalign((void**)&state_gpu, 16384, (size_t)T * D * sizeof(float));
    posix_memalign((void**)&wkv_cpu, 16384, (size_t)T * D * sizeof(float));
    posix_memalign((void**)&wkv_gpu, 16384, (size_t)T * D * sizeof(float));

    for (int i = 0; i < T * D; i++) {
        k[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        v[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        r_sig[i] = ((float)rand() / RAND_MAX);
    }
    for (int i = 0; i < D; i++) decay[i] = ((float)rand() / RAND_MAX) * 5.0f - 2.5f;

    tc_wkv_fwd(k, v, r_sig, decay, state_cpu, wkv_cpu, T, D);
    tc_metal_wkv_fwd(k, v, r_sig, decay, state_gpu, wkv_gpu, T, D);
    tc_metal_commit_and_wait();
    
    check_equal(state_cpu, state_gpu, T * D, "wkv_state");
    check_equal(wkv_cpu, wkv_gpu, T * D, "wkv_out");

    // Attention test
    int H = 8, n_kv_heads = 2;
    int kv_D = (D / H) * n_kv_heads;
    float *Q, *k_att, *v_att, *attn_w_cpu, *attn_w_gpu, *attn_ctx_cpu, *attn_ctx_gpu;
    posix_memalign((void**)&Q, 16384, (size_t)T * D * sizeof(float));
    posix_memalign((void**)&k_att, 16384, (size_t)T * kv_D * sizeof(float));
    posix_memalign((void**)&v_att, 16384, (size_t)T * kv_D * sizeof(float));
    posix_memalign((void**)&attn_w_cpu, 16384, (size_t)H * T * T * sizeof(float));
    posix_memalign((void**)&attn_w_gpu, 16384, (size_t)H * T * T * sizeof(float));
    memset(attn_w_cpu, 0, (size_t)H * T * T * sizeof(float));
    memset(attn_w_gpu, 0, (size_t)H * T * T * sizeof(float));
    posix_memalign((void**)&attn_ctx_cpu, 16384, (size_t)T * D * sizeof(float));
    posix_memalign((void**)&attn_ctx_gpu, 16384, (size_t)T * D * sizeof(float));

    for (int i = 0; i < T * D; i++) Q[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < T * kv_D; i++) {
        k_att[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        v_att[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    tc_attn_fwd(Q, k_att, v_att, attn_w_cpu, attn_ctx_cpu, T, D, H, n_kv_heads);
    tc_metal_attn_fwd(Q, k_att, v_att, attn_w_gpu, attn_ctx_gpu, T, D, H, n_kv_heads);
    tc_metal_commit_and_wait();

    check_equal(attn_w_cpu, attn_w_gpu, H * T * T, "attn_w");
    check_equal(attn_ctx_cpu, attn_ctx_gpu, T * D, "attn_ctx");
    
    // 6. RMSNorm bwd
    float *dy = NULL, *dgamma_cpu = NULL, *dgamma_gpu = NULL, *dx_cpu = NULL, *dx_gpu = NULL;
    posix_memalign((void**)&dy, 16384, ALIGN16K((size_t)T * D * sizeof(float)));
    posix_memalign((void**)&dgamma_cpu, 16384, ALIGN16K((size_t)D * sizeof(float)));
    posix_memalign((void**)&dgamma_gpu, 16384, ALIGN16K((size_t)D * sizeof(float)));
    posix_memalign((void**)&dx_cpu, 16384, ALIGN16K((size_t)T * D * sizeof(float)));
    posix_memalign((void**)&dx_gpu, 16384, ALIGN16K((size_t)T * D * sizeof(float)));
    for (int i = 0; i < T * D; i++) dy[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    memset(dgamma_cpu, 0, D * sizeof(float));
    memset(dgamma_gpu, 0, D * sizeof(float));
    memset(dx_cpu, 0, T * D * sizeof(float));
    memset(dx_gpu, 0, T * D * sizeof(float));
    tc_rmsnorm_bwd(dy, xhat_cpu, g, rms_cpu, T, D, dgamma_cpu, dx_cpu);
    tc_metal_rmsnorm_bwd(dy, xhat_cpu, g, rms_cpu, T, D, dgamma_gpu, dx_gpu);
    tc_metal_commit_and_wait();
    check_equal(dgamma_cpu, dgamma_gpu, D, "rmsnorm_bwd_dgamma");
    check_equal(dx_cpu, dx_gpu, T * D, "rmsnorm_bwd_dx");

    // 7. SwiGLU bwd
    float *dhsilu = NULL, *dhgate_cpu = NULL, *dhgate_gpu = NULL, *dhup_cpu = NULL, *dhup_gpu = NULL;
    posix_memalign((void**)&dhsilu, 16384, ALIGN16K((size_t)T * FFD * sizeof(float)));
    posix_memalign((void**)&dhgate_cpu, 16384, ALIGN16K((size_t)T * FFD * sizeof(float)));
    posix_memalign((void**)&dhgate_gpu, 16384, ALIGN16K((size_t)T * FFD * sizeof(float)));
    posix_memalign((void**)&dhup_cpu, 16384, ALIGN16K((size_t)T * FFD * sizeof(float)));
    posix_memalign((void**)&dhup_gpu, 16384, ALIGN16K((size_t)T * FFD * sizeof(float)));
    for (int i = 0; i < T * FFD; i++) dhsilu[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    memset(dhgate_cpu, 0, T * FFD * sizeof(float));
    memset(dhgate_gpu, 0, T * FFD * sizeof(float));
    memset(dhup_cpu, 0, T * FFD * sizeof(float));
    memset(dhup_gpu, 0, T * FFD * sizeof(float));
    tc_swiglu_bwd(dhsilu, hgate, hup, dhgate_cpu, dhup_cpu, T, FFD);
    tc_metal_swiglu_bwd(dhsilu, hgate, hup, dhgate_gpu, dhup_gpu, T, FFD);
    tc_metal_commit_and_wait();
    check_equal(dhgate_cpu, dhgate_gpu, T * FFD, "swiglu_bwd_dhgate");
    check_equal(dhup_cpu, dhup_gpu, T * FFD, "swiglu_bwd_dhup");

    // 8. WKV bwd
    float *dwkv = NULL, *dstate_cpu = NULL, *dstate_gpu = NULL;
    float *drpre_cpu = NULL, *drpre_gpu = NULL, *dk_cpu = NULL, *dk_gpu = NULL;
    float *dv_cpu = NULL, *dv_gpu = NULL, *ddecay_cpu = NULL, *ddecay_gpu = NULL;
    posix_memalign((void**)&dwkv, 16384, ALIGN16K((size_t)T * D * sizeof(float)));
    posix_memalign((void**)&dstate_cpu, 16384, ALIGN16K((size_t)T * D * sizeof(float)));
    posix_memalign((void**)&dstate_gpu, 16384, ALIGN16K((size_t)T * D * sizeof(float)));
    posix_memalign((void**)&drpre_cpu, 16384, ALIGN16K((size_t)T * D * sizeof(float)));
    posix_memalign((void**)&drpre_gpu, 16384, ALIGN16K((size_t)T * D * sizeof(float)));
    posix_memalign((void**)&dk_cpu, 16384, ALIGN16K((size_t)T * D * sizeof(float)));
    posix_memalign((void**)&dk_gpu, 16384, ALIGN16K((size_t)T * D * sizeof(float)));
    posix_memalign((void**)&dv_cpu, 16384, ALIGN16K((size_t)T * D * sizeof(float)));
    posix_memalign((void**)&dv_gpu, 16384, ALIGN16K((size_t)T * D * sizeof(float)));
    posix_memalign((void**)&ddecay_cpu, 16384, ALIGN16K((size_t)D * sizeof(float)));
    posix_memalign((void**)&ddecay_gpu, 16384, ALIGN16K((size_t)D * sizeof(float)));
    for (int i = 0; i < T * D; i++) dwkv[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    memset(dstate_cpu, 0, T * D * sizeof(float));
    memset(dstate_gpu, 0, T * D * sizeof(float));
    memset(drpre_cpu, 0, T * D * sizeof(float));
    memset(drpre_gpu, 0, T * D * sizeof(float));
    memset(dk_cpu, 0, T * D * sizeof(float));
    memset(dk_gpu, 0, T * D * sizeof(float));
    memset(dv_cpu, 0, T * D * sizeof(float));
    memset(dv_gpu, 0, T * D * sizeof(float));
    memset(ddecay_cpu, 0, D * sizeof(float));
    memset(ddecay_gpu, 0, D * sizeof(float));
    tc_wkv_bwd(dwkv, k, v, r_sig, decay, state_cpu, dstate_cpu, drpre_cpu, dk_cpu, dv_cpu, ddecay_cpu, T, D);
    tc_metal_wkv_bwd(dwkv, k, v, r_sig, decay, state_cpu, dstate_gpu, drpre_gpu, dk_gpu, dv_gpu, ddecay_gpu, T, D);
    tc_metal_commit_and_wait();
    check_equal(drpre_cpu, drpre_gpu, T * D, "wkv_bwd_drpre");
    check_equal(dk_cpu, dk_gpu, T * D, "wkv_bwd_dk");
    check_equal(dv_cpu, dv_gpu, T * D, "wkv_bwd_dv");
    check_equal(ddecay_cpu, ddecay_gpu, D, "wkv_bwd_ddecay");

    // 9. Attn bwd
    float *dctx = NULL, *dQ_cpu = NULL, *dQ_gpu = NULL, *dK_cpu = NULL, *dK_gpu = NULL, *dV_cpu = NULL, *dV_gpu = NULL;
    posix_memalign((void**)&dctx, 16384, ALIGN16K((size_t)T * D * sizeof(float)));
    posix_memalign((void**)&dQ_cpu, 16384, ALIGN16K((size_t)T * D * sizeof(float)));
    posix_memalign((void**)&dQ_gpu, 16384, ALIGN16K((size_t)T * D * sizeof(float)));
    posix_memalign((void**)&dK_cpu, 16384, ALIGN16K((size_t)T * kv_D * sizeof(float)));
    posix_memalign((void**)&dK_gpu, 16384, ALIGN16K((size_t)T * kv_D * sizeof(float)));
    posix_memalign((void**)&dV_cpu, 16384, ALIGN16K((size_t)T * kv_D * sizeof(float)));
    posix_memalign((void**)&dV_gpu, 16384, ALIGN16K((size_t)T * kv_D * sizeof(float)));
    for (int i = 0; i < T * D; i++) dctx[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    memset(dQ_cpu, 0, T * D * sizeof(float));
    memset(dQ_gpu, 0, T * D * sizeof(float));
    memset(dK_cpu, 0, T * kv_D * sizeof(float));
    memset(dK_gpu, 0, T * kv_D * sizeof(float));
    memset(dV_cpu, 0, T * kv_D * sizeof(float));
    memset(dV_gpu, 0, T * kv_D * sizeof(float));
    tc_attn_bwd(dctx, attn_w_cpu, Q, k_att, v_att, dQ_cpu, dK_cpu, dV_cpu, T, D, H, n_kv_heads);
    tc_metal_attn_bwd(dctx, attn_w_cpu, Q, k_att, v_att, dQ_gpu, dK_gpu, dV_gpu, T, D, H, n_kv_heads);
    tc_metal_commit_and_wait();
    check_equal(dQ_cpu, dQ_gpu, T * D, "attn_bwd_dQ");
    check_equal(dK_cpu, dK_gpu, T * kv_D, "attn_bwd_dK");
    check_equal(dV_cpu, dV_gpu, T * kv_D, "attn_bwd_dV");

    tc_metal_shutdown();
    printf("tc_metalcheck all unit tests PASS.\n");
    return 0;
}
