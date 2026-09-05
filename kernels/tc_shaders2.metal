#include <metal_stdlib>
using namespace metal;

// ----------------------------------------------------------------------------
// CE Loss Forward (3-pass Softmax)
// ----------------------------------------------------------------------------
kernel void ce_fwd(device const float *logits [[buffer(0)]],
                   device const int *targets [[buffer(1)]],
                   device float *probs [[buffer(2)]],
                   device atomic_float *loss_out [[buffer(3)]],
                   device atomic_int *n_active_out [[buffer(4)]],
                   constant int &V [[buffer(5)]],
                   uint3 gid [[threadgroup_position_in_grid]],
                   uint tid [[thread_position_in_threadgroup]],
                   uint tgid [[threads_per_threadgroup]]) {
    // gid.x = token index (batch * T)
    int target = targets[gid.x];
    if (target < 0) return; // Masked out
    
    device const float *lgt = logits + gid.x * V;
    device float *prb = probs + gid.x * V;
    
    threadgroup float tg_max[32];
    threadgroup float tg_sum[32];
    
    // Pass 1: Max
    float local_max = -1e30f;
    for (int i = tid; i < V; i += tgid) {
        float val = lgt[i];
        if (val > local_max) local_max = val;
    }
    float simd_max = simd_max(local_max);
    if (tid % 32 == 0) tg_max[tid / 32] = simd_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid < 32) {
        float val = (tid < (tgid / 32)) ? tg_max[tid] : -1e30f;
        simd_max = simd_max(val);
        if (tid == 0) tg_max[0] = simd_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maxl = tg_max[0];
    
    // Pass 2: Exp and Sum
    float local_sum = 0.0f;
    for (int i = tid; i < V; i += tgid) {
        float p = exp(lgt[i] - maxl);
        prb[i] = p;
        local_sum += p;
    }
    float simd_sum_val = simd_sum(local_sum);
    if (tid % 32 == 0) tg_sum[tid / 32] = simd_sum_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid < 32) {
        float val = (tid < (tgid / 32)) ? tg_sum[tid] : 0.0f;
        simd_sum_val = simd_sum(val);
        if (tid == 0) tg_sum[0] = simd_sum_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float suml = tg_sum[0];
    
    // Pass 3: Normalize
    for (int i = tid; i < V; i += tgid) {
        prb[i] /= suml;
    }
    
    // Loss update
    if (tid == 0) {
        float prob = prb[target];
        if (prob < 1e-9f) prob = 1e-9f;
        float loss_val = -log(prob);
        atomic_fetch_add_explicit(loss_out, loss_val, memory_order_relaxed);
        atomic_fetch_add_explicit(n_active_out, 1, memory_order_relaxed);
    }
}

// ----------------------------------------------------------------------------
// CE Loss Backward
// ----------------------------------------------------------------------------
kernel void ce_bwd(device const float *probs [[buffer(0)]],
                   device const int *targets [[buffer(1)]],
                   device float *dlogits [[buffer(2)]],
                   constant float &weight [[buffer(3)]],
                   constant int &n_active [[buffer(4)]],
                   constant int &V [[buffer(5)]],
                   uint3 gid [[threadgroup_position_in_grid]],
                   uint tid [[thread_position_in_threadgroup]],
                   uint tgid [[threads_per_threadgroup]]) {
    int target = targets[gid.x];
    if (target < 0 || n_active == 0 || weight <= 0.0f) return;
    
    float scale = weight / (float)n_active;
    device const float *prb = probs + gid.x * V;
    device float *dl = dlogits + gid.x * V;
    
    for (int i = tid; i < V; i += tgid) {
        float p = prb[i];
        float ind = (i == target) ? 1.0f : 0.0f;
        // The MTP targets could share the dlogits buffer if not careful? No, they have independent heads
        // But atomic adds to dlogits is safe. In C it does dl[i] += scale * (p - ind).
        float val = scale * (p - ind);
        atomic_fetch_add_explicit((device atomic_float*)&dl[i], val, memory_order_relaxed);
    }
}

// ----------------------------------------------------------------------------
// Token Shift (Time Mix Mixers)
// ----------------------------------------------------------------------------
kernel void token_shift_fwd(device const float *x [[buffer(0)]],
                            device const float *mix_k [[buffer(1)]],
                            device const float *mix_v [[buffer(2)]],
                            device const float *mix_r [[buffer(3)]],
                            device float *xk [[buffer(4)]],
                            device float *xv [[buffer(5)]],
                            device float *xr [[buffer(6)]],
                            constant int &T [[buffer(7)]],
                            constant int &D [[buffer(8)]],
                            uint3 gid [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint tgid [[threads_per_threadgroup]]) {
    int b = gid.y;
    int t = gid.x;
    
    device const float *cur = x + (b * T + t) * D;
    device const float *prev = (t > 0) ? x + (b * T + t - 1) * D : nullptr;
    
    device float *out_xk = xk + (b * T + t) * D;
    device float *out_xv = xv + (b * T + t) * D;
    device float *out_xr = xr + (b * T + t) * D;
    
    for (int i = tid; i < D; i += tgid) {
        float mk = 1.0f / (1.0f + exp(-mix_k[i]));
        float mv = 1.0f / (1.0f + exp(-mix_v[i]));
        float mr = 1.0f / (1.0f + exp(-mix_r[i]));
        
        float p_val = prev ? prev[i] : 0.0f;
        float c_val = cur[i];
        
        out_xk[i] = mk * c_val + (1.0f - mk) * p_val;
        out_xv[i] = mv * c_val + (1.0f - mv) * p_val;
        out_xr[i] = mr * c_val + (1.0f - mr) * p_val;
    }
}
