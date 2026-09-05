#include <metal_stdlib>
using namespace metal;

// ----------------------------------------------------------------------------
// RMSNorm
// ----------------------------------------------------------------------------
kernel void rmsnorm_fwd(device const float *x [[buffer(0)]],
                        device const float *gamma [[buffer(1)]],
                        device float *rms [[buffer(2)]],
                        device float *xhat [[buffer(3)]],
                        device float *y [[buffer(4)]],
                        constant int &D [[buffer(5)]],
                        constant float &eps [[buffer(6)]],
                        uint3 gid [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint tgid [[threads_per_threadgroup]]) {
    // Each threadgroup processes one token: gid.x = token index (batch * T)
    device const float *x_t = x + gid.x * D;
    device float *xhat_t = xhat + gid.x * D;
    device float *y_t = y + gid.x * D;
    
    threadgroup float sum_sq[32]; // Assumes max threadgroup size is a multiple of SIMD width (32)
    float local_sq = 0.0f;
    for (int i = tid; i < D; i += tgid) {
        float val = x_t[i];
        local_sq += val * val;
    }
    
    float simd_sq = simd_sum(local_sq);
    if (tid % 32 == 0) {
        sum_sq[tid / 32] = simd_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid < 32) {
        float val = (tid < (tgid / 32)) ? sum_sq[tid] : 0.0f;
        simd_sq = simd_sum(val);
        if (tid == 0) {
            float rms_val = sqrt(simd_sq / (float)D + eps);
            rms[gid.x] = rms_val;
            sum_sq[0] = rms_val; // reuse for broadcast
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float inv_rms = 1.0f / sum_sq[0];
    for (int i = tid; i < D; i += tgid) {
        float norm_val = x_t[i] * inv_rms;
        xhat_t[i] = norm_val;
        y_t[i] = norm_val * gamma[i];
    }
}

kernel void rmsnorm_bwd(device const float *dy [[buffer(0)]],
                        device const float *xhat [[buffer(1)]],
                        device const float *gamma [[buffer(2)]],
                        device const float *rms [[buffer(3)]],
                        device float *dx [[buffer(4)]],
                        device float *dgamma [[buffer(5)]],
                        constant int &D [[buffer(6)]],
                        uint3 gid [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint tgid [[threads_per_threadgroup]]) {
    // gid.x = token index (batch * T)
    device const float *dy_t = dy + gid.x * D;
    device const float *xhat_t = xhat + gid.x * D;
    device float *dx_t = dx + gid.x * D;
    float rms_val = rms[gid.x];
    
    threadgroup float sum_dy_gamma_xhat[32];
    
    float local_sum = 0.0f;
    for (int i = tid; i < D; i += tgid) {
        local_sum += dy_t[i] * gamma[i] * xhat_t[i];
    }
    float simd_sum_val = simd_sum(local_sum);
    if (tid % 32 == 0) sum_dy_gamma_xhat[tid / 32] = simd_sum_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid < 32) {
        float val = (tid < (tgid / 32)) ? sum_dy_gamma_xhat[tid] : 0.0f;
        simd_sum_val = simd_sum(val);
        if (tid == 0) {
            sum_dy_gamma_xhat[0] = simd_sum_val / (float)D;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float mean_dy_gamma_xhat = sum_dy_gamma_xhat[0];
    float inv_rms = 1.0f / rms_val;
    for (int i = tid; i < D; i += tgid) {
        float dgamma_val = dy_t[i] * xhat_t[i];
        atomic_fetch_add_explicit((device atomic_float*)&dgamma[i], dgamma_val, memory_order_relaxed);
        dx_t[i] = inv_rms * (dy_t[i] * gamma[i] - xhat_t[i] * mean_dy_gamma_xhat);
    }
}

// ----------------------------------------------------------------------------
// SwiGLU
// ----------------------------------------------------------------------------
kernel void swiglu_fwd(device const float *hgate [[buffer(0)]],
                       device const float *hup [[buffer(1)]],
                       device float *hsilu [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    float x = hgate[gid];
    float silu = x / (1.0f + exp(-x));
    hsilu[gid] = silu * hup[gid];
}

kernel void swiglu_bwd(device const float *dhsilu [[buffer(0)]],
                       device const float *hgate [[buffer(1)]],
                       device const float *hup [[buffer(2)]],
                       device float *dhgate [[buffer(3)]],
                       device float *dhup [[buffer(4)]],
                       uint gid [[thread_position_in_grid]]) {
    float dx = dhsilu[gid];
    float hg = hgate[gid];
    float hu = hup[gid];
    
    float sig = 1.0f / (1.0f + exp(-hg));
    float silu = hg * sig;
    float dsilu = sig + hg * sig * (1.0f - sig);
    
    dhup[gid] = dx * silu;
    dhgate[gid] = dx * hu * dsilu;
}

// ----------------------------------------------------------------------------
// RoPE
// ----------------------------------------------------------------------------
kernel void rope_fwd(device float *q [[buffer(0)]],
                     device float *k [[buffer(1)]],
                     constant int &T [[buffer(2)]],
                     constant int &D [[buffer(3)]],
                     constant int &n_heads [[buffer(4)]],
                     constant int &n_kv_heads [[buffer(5)]],
                     uint3 gid [[threadgroup_position_in_grid]],
                     uint tid [[thread_position_in_threadgroup]]) {
    // gid.x = token index t within a sequence
    // gid.y = batch index b
    int t = gid.x;
    int b = gid.y;
    int seq_offset = (b * T + t) * D;
    int Dh = D / n_heads;
    
    // Process Q
    for (int h = 0; h < n_heads; h++) {
        int i = tid * 2;
        if (i < Dh) {
            float theta = (float)t * pow(10000.0f, -(float)i / (float)Dh);
            float cos_t = cos(theta);
            float sin_t = sin(theta);
            int idx0 = seq_offset + h * Dh + i;
            int idx1 = idx0 + 1;
            float q0 = q[idx0];
            float q1 = q[idx1];
            q[idx0] = q0 * cos_t - q1 * sin_t;
            q[idx1] = q1 * cos_t + q0 * sin_t;
        }
    }
    
    // Process K
    int kv_Dh = Dh;
    int kv_seq_offset = (b * T + t) * (kv_Dh * n_kv_heads);
    for (int h = 0; h < n_kv_heads; h++) {
        int i = tid * 2;
        if (i < kv_Dh) {
            float theta = (float)t * pow(10000.0f, -(float)i / (float)kv_Dh);
            float cos_t = cos(theta);
            float sin_t = sin(theta);
            int idx0 = kv_seq_offset + h * kv_Dh + i;
            int idx1 = idx0 + 1;
            float k0 = k[idx0];
            float k1 = k[idx1];
            k[idx0] = k0 * cos_t - k1 * sin_t;
            k[idx1] = k1 * cos_t + k0 * sin_t;
        }
    }
}

kernel void rope_bwd(device float *dq [[buffer(0)]],
                     device float *dk [[buffer(1)]],
                     constant int &T [[buffer(2)]],
                     constant int &D [[buffer(3)]],
                     constant int &n_heads [[buffer(4)]],
                     constant int &n_kv_heads [[buffer(5)]],
                     uint3 gid [[threadgroup_position_in_grid]],
                     uint tid [[thread_position_in_threadgroup]]) {
    int t = gid.x;
    int b = gid.y;
    int seq_offset = (b * T + t) * D;
    int Dh = D / n_heads;
    
    for (int h = 0; h < n_heads; h++) {
        int i = tid * 2;
        if (i < Dh) {
            float theta = (float)t * pow(10000.0f, -(float)i / (float)Dh);
            float cos_t = cos(theta);
            float sin_t = sin(theta);
            int idx0 = seq_offset + h * Dh + i;
            int idx1 = idx0 + 1;
            float dq0 = dq[idx0];
            float dq1 = dq[idx1];
            dq[idx0] = dq0 * cos_t + dq1 * sin_t;
            dq[idx1] = dq1 * cos_t - dq0 * sin_t;
        }
    }
    
    int kv_Dh = Dh;
    int kv_seq_offset = (b * T + t) * (kv_Dh * n_kv_heads);
    for (int h = 0; h < n_kv_heads; h++) {
        int i = tid * 2;
        if (i < kv_Dh) {
            float theta = (float)t * pow(10000.0f, -(float)i / (float)kv_Dh);
            float cos_t = cos(theta);
            float sin_t = sin(theta);
            int idx0 = kv_seq_offset + h * kv_Dh + i;
            int idx1 = idx0 + 1;
            float dk0 = dk[idx0];
            float dk1 = dk[idx1];
            dk[idx0] = dk0 * cos_t + dk1 * sin_t;
            dk[idx1] = dk1 * cos_t - dk0 * sin_t;
        }
    }
}
