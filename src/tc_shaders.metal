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
                        uint gid [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint tgid [[threads_per_threadgroup]]) {
    device const float *x_t = x + gid * D;
    device float *xhat_t = xhat + gid * D;
    device float *y_t = y + gid * D;
    
    threadgroup float sum_sq[32]; 
    float local_sq = 0.0f;
    for (int i = tid; i < D; i += tgid) {
        float val = x_t[i];
        local_sq += val * val;
    }
    
    float simd_sq = simd_sum(local_sq);
    if (tid % 32 == 0) sum_sq[tid / 32] = simd_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid < 32) {
        float val = (tid < (tgid / 32)) ? sum_sq[tid] : 0.0f;
        simd_sq = simd_sum(val);
        if (tid == 0) {
            float rms_val = 1.0f / sqrt(simd_sq / (float)D + eps);
            rms[gid] = rms_val;
            sum_sq[0] = rms_val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float inv_rms = sum_sq[0];
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
                        uint gid [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint tgid [[threads_per_threadgroup]]) {
    device const float *dy_t = dy + gid * D;
    device const float *xhat_t = xhat + gid * D;
    device float *dx_t = dx + gid * D;
    float rms_val = rms[gid];
    
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
        if (tid == 0) sum_dy_gamma_xhat[0] = simd_sum_val / (float)D;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float mean_dy_gamma_xhat = sum_dy_gamma_xhat[0];
    float inv_rms = rms_val;
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
                     uint3 tid [[thread_position_in_threadgroup]],
                     uint3 tgid [[threads_per_threadgroup]]) {
    int t = gid.x;
    int b = gid.y;
    int seq_offset = (b * T + t) * D;
    int Dh = D / n_heads;
    
    for (int h = 0; h < n_heads; h++) {
        int i = tid.x * 2;
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
    
    int kv_Dh = Dh;
    int kv_seq_offset = (b * T + t) * (kv_Dh * n_kv_heads);
    for (int h = 0; h < n_kv_heads; h++) {
        int i = tid.x * 2;
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
                     uint3 tid [[thread_position_in_threadgroup]],
                     uint3 tgid [[threads_per_threadgroup]]) {
    int t = gid.x;
    int b = gid.y;
    int seq_offset = (b * T + t) * D;
    int Dh = D / n_heads;
    
    for (int h = 0; h < n_heads; h++) {
        int i = tid.x * 2;
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
        int i = tid.x * 2;
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

// ----------------------------------------------------------------------------
// CE Loss Forward (3-pass Softmax)
// ----------------------------------------------------------------------------
kernel void ce_fwd(device const float *logits [[buffer(0)]],
                   device const int *targets [[buffer(1)]],
                   device float *probs [[buffer(2)]],
                   device atomic_float *loss_out [[buffer(3)]],
                   device atomic_int *n_active_out [[buffer(4)]],
                   constant int &V [[buffer(5)]],
                   uint gid [[threadgroup_position_in_grid]],
                   uint tid [[thread_position_in_threadgroup]],
                   uint tgid [[threads_per_threadgroup]]) {
    int target = targets[gid];
    if (target < 0) return;
    
    device const float *lgt = logits + gid * V;
    device float *prb = probs + gid * V;
    
    threadgroup float tg_max[32];
    threadgroup float tg_sum[32];
    
    // Pass 1: Max
    float local_max = -1e30f;
    for (int i = tid; i < V; i += tgid) {
        float val = 30.0f * tanh(lgt[i] / 30.0f);
        if (val > local_max) local_max = val;
    }
    float simd_max_val = simd_max(local_max);
    if (tid % 32 == 0) tg_max[tid / 32] = simd_max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid < 32) {
        float val = (tid < (tgid / 32)) ? tg_max[tid] : -1e30f;
        simd_max_val = simd_max(val);
        if (tid == 0) tg_max[0] = simd_max_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maxl = tg_max[0];
    
    // Pass 2: Exp and Sum
    float local_sum = 0.0f;
    for (int i = tid; i < V; i += tgid) {
        float val = 30.0f * tanh(lgt[i] / 30.0f);
        float p = exp(val - maxl);
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
                   device const float *logits [[buffer(6)]],
                   uint gid [[threadgroup_position_in_grid]],
                   uint tid [[thread_position_in_threadgroup]],
                   uint tgid [[threads_per_threadgroup]]) {
    int target = targets[gid];
    device float *dl = dlogits + gid * V;

    if (target < 0 || n_active == 0 || weight <= 0.0f) {
        for (int i = tid; i < V; i += tgid) dl[i] = 0.0f;
        return;
    }
    
    float scale = weight / (float)n_active;
    device const float *prb = probs + gid * V;
    device const float *lgt = logits + gid * V;
    
    for (int i = tid; i < V; i += tgid) {
        float p = prb[i];
        float ind = (i == target) ? 1.0f : 0.0f;
        float d_softcap = scale * (p - ind);
        
        // Derivative of 30 * tanh(x / 30) is 1 - tanh^2(x / 30)
        float t = tanh(lgt[i] / 30.0f);
        float val = d_softcap * (1.0f - t * t);
        
        dl[i] = val;
    }
}

// ----------------------------------------------------------------------------
// Attention Forward
// ----------------------------------------------------------------------------
kernel void attn_fwd(device const float *Q [[buffer(0)]],
                     device const float *K [[buffer(1)]],
                     device const float *V [[buffer(2)]],
                     device float *attn_w [[buffer(3)]],
                     device float *attn_ctx [[buffer(4)]],
                     constant int &T [[buffer(5)]],
                     constant int &D [[buffer(6)]],
                     constant int &H [[buffer(7)]],
                     constant int &n_kv_heads [[buffer(8)]],
                     uint3 gid [[threadgroup_position_in_grid]],
                     uint3 tid [[thread_position_in_threadgroup]],
                     uint3 tgid [[threads_per_threadgroup]]) {
    int t = gid.x;
    int h = gid.y;
    int b = gid.z;
    
    int Dh = D / H;
    int kv_h = h / (H / n_kv_heads);
    int kv_D = (D / H) * n_kv_heads;
    int kv_Dh = D / H;
    
    float invsqrt_dh = 1.0f / sqrt((float)Dh);
    
    threadgroup float scores[128]; // max_seq_len is 128
    
    device const float *qh = Q + (b * T + t) * D + h * Dh;
    
    float maxs = -1e30f;
    for (int u = 0; u <= t; u++) {
        device const float *kh = K + (b * T + u) * kv_D + kv_h * kv_Dh;
        
        float local_dot = 0.0f;
        for (int i = tid.x; i < Dh; i += tgid.x) {
            local_dot += qh[i] * kh[i];
        }
        float simd_dot = simd_sum(local_dot);
        
        threadgroup float sum_dot[32];
        if (tid.x % 32 == 0) sum_dot[tid.x / 32] = simd_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid.x < 32) {
            float val = (tid.x < (tgid.x / 32)) ? sum_dot[tid.x] : 0.0f;
            simd_dot = simd_sum(val);
            if (tid.x == 0) {
                float s = simd_dot * invsqrt_dh;
                scores[u] = s;
                if (s > maxs) maxs = s;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    threadgroup float tg_max[1];
    if (tid.x == 0) tg_max[0] = maxs;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maxs = tg_max[0];
    
    float sum_w = 0.0f;
    for (int u = 0; u <= t; u++) {
        if (tid.x == 0) {
            float w = exp(scores[u] - maxs);
            scores[u] = w;
            sum_w += w;
        }
    }
    
    threadgroup float tg_sum[1];
    if (tid.x == 0) tg_sum[0] = sum_w;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum_w = tg_sum[0];
    
    device float *wrow = attn_w + (b * H * T * T) + (h * T + t) * T;
    
    for (int i = tid.x; i < Dh; i += tgid.x) {
        float ctx_val = 0.0f;
        for (int u = 0; u <= t; u++) {
            float w = scores[u] / sum_w;
            if (i == 0) wrow[u] = w;
            device const float *vh = V + (b * T + u) * kv_D + kv_h * kv_Dh;
            ctx_val += w * vh[i];
        }
        attn_ctx[(b * T + t) * D + h * Dh + i] = ctx_val;
    }
}

// ----------------------------------------------------------------------------
// Attention Backward
// ----------------------------------------------------------------------------
kernel void attn_bwd(device const float *dctx [[buffer(0)]],
                     device const float *attn_w [[buffer(1)]],
                     device const float *Q [[buffer(2)]],
                     device const float *K [[buffer(3)]],
                     device const float *V [[buffer(4)]],
                     device float *dQ [[buffer(5)]],
                     device float *dK [[buffer(6)]],
                     device float *dV [[buffer(7)]],
                     constant int &T [[buffer(8)]],
                     constant int &D [[buffer(9)]],
                     constant int &H [[buffer(10)]],
                     constant int &n_kv_heads [[buffer(11)]],
                     uint3 gid [[threadgroup_position_in_grid]],
                     uint3 tid [[thread_position_in_threadgroup]],
                     uint3 tgid [[threads_per_threadgroup]]) {
    int t = gid.x;
    int h = gid.y;
    int b = gid.z;
    
    int Dh = D / H;
    int kv_h = h / (H / n_kv_heads);
    int kv_D = (D / H) * n_kv_heads;
    int kv_Dh = D / H;
    float invsqrt_dh = 1.0f / sqrt((float)Dh);
    
    device const float *dctx_h = dctx + (b * T + t) * D + h * Dh;
    device const float *wrow = attn_w + (b * H * T * T) + (h * T + t) * T;
    
    threadgroup float dw[128]; // Max seq len
    
    for (int u = 0; u <= t; u++) {
        device const float *vh = V + (b * T + u) * kv_D + kv_h * kv_Dh;
        device float *dvh = dV + (b * T + u) * kv_D + kv_h * kv_Dh;
        
        float local_dot = 0.0f;
        for (int i = tid.x; i < Dh; i += tgid.x) {
            float dctx_val = dctx_h[i];
            local_dot += dctx_val * vh[i];
            float dv_val = wrow[u] * dctx_val;
            atomic_fetch_add_explicit((device atomic_float*)&dvh[i], dv_val, memory_order_relaxed);
        }
        
        float simd_dot = simd_sum(local_dot);
        threadgroup float sum_dot[32];
        if (tid.x % 32 == 0) sum_dot[tid.x / 32] = simd_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid.x < 32) {
            float val = (tid.x < (tgid.x / 32)) ? sum_dot[tid.x] : 0.0f;
            simd_dot = simd_sum(val);
            if (tid.x == 0) dw[u] = simd_dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float dot_val = 0.0f;
    for (int u = 0; u <= t; u++) dot_val += dw[u] * wrow[u];
    
    device float *dq_h = dQ + (b * T + t) * D + h * Dh;
    
    for (int i = tid.x; i < Dh; i += tgid.x) {
        float q_val = Q[(b * T + t) * D + h * Dh + i];
        float dq_val = 0.0f;
        for (int u = 0; u <= t; u++) {
            device const float *kh = K + (b * T + u) * kv_D + kv_h * kv_Dh;
            device float *dkh = dK + (b * T + u) * kv_D + kv_h * kv_Dh;
            
            float w = wrow[u];
            float dw_u = dw[u];
            float ds = w * (dw_u - dot_val) * invsqrt_dh;
            dq_val += ds * kh[i];
            
            float dk_val = ds * q_val;
            atomic_fetch_add_explicit((device atomic_float*)&dkh[i], dk_val, memory_order_relaxed);
        }
        dq_h[i] = dq_val;
    }
}

// ----------------------------------------------------------------------------
// WKV Forward
// ----------------------------------------------------------------------------
kernel void wkv_fwd(device const float *k [[buffer(0)]],
                    device const float *v [[buffer(1)]],
                    device const float *r_sig [[buffer(2)]],
                    device const float *decay [[buffer(3)]],
                    device float *state [[buffer(4)]],
                    device float *wkv [[buffer(5)]],
                    constant int &T [[buffer(6)]],
                    constant int &D [[buffer(7)]],
                    uint gid [[threadgroup_position_in_grid]],
                    uint tid [[thread_position_in_threadgroup]],
                    uint tgid [[threads_per_threadgroup]]) {
    int b = gid;
    
    for (int i = tid; i < D; i += tgid) {
        float dc = 1.0f / (1.0f + exp(-decay[i]));
        float st_prev = 0.0f;
        for (int t = 0; t < T; t++) {
            int idx = (b * T + t) * D + i;
            float k_val = k[idx];
            float v_val = v[idx];
            float r_val = r_sig[idx];
            
            float st = dc * st_prev + (1.0f - dc) * (k_val * v_val);
            state[idx] = st;
            wkv[idx] = r_val * st;
            st_prev = st;
        }
    }
}

// ----------------------------------------------------------------------------
// WKV Backward
// ----------------------------------------------------------------------------
kernel void wkv_bwd(device const float *dwkv [[buffer(0)]],
                    device const float *k [[buffer(1)]],
                    device const float *v [[buffer(2)]],
                    device const float *r_sig [[buffer(3)]],
                    device const float *decay [[buffer(4)]],
                    device const float *state [[buffer(5)]],
                    device float *dstate [[buffer(6)]],
                    device float *drpre [[buffer(7)]],
                    device float *dk [[buffer(8)]],
                    device float *dv [[buffer(9)]],
                    device float *ddecay [[buffer(10)]],
                    constant int &T [[buffer(11)]],
                    constant int &D [[buffer(12)]],
                    uint gid [[threadgroup_position_in_grid]],
                    uint tid [[thread_position_in_threadgroup]],
                    uint tgid [[threads_per_threadgroup]]) {
    int b = gid;
    
    for (int i = tid; i < D; i += tgid) {
        float dc = 1.0f / (1.0f + exp(-decay[i]));
        float local_ddecay = 0.0f;
        
        for (int t = T - 1; t >= 0; t--) {
            int idx = (b * T + t) * D + i;
            float r_val = r_sig[idx];
            float st_val = state[idx];
            float dw = dwkv[idx];
            
            float drs_i = dw * st_val;
            float dst_i = dw * r_val + dstate[idx];
            
            drpre[idx] = drs_i * r_val * (1.0f - r_val);
            
            float sp = (t > 0) ? state[(b * T + t - 1) * D + i] : 0.0f;
            float dgate = 1.0f - dc;
            
            float kt = k[idx];
            float vt = v[idx];
            
            dk[idx] = dst_i * dgate * vt;
            dv[idx] = dst_i * dgate * kt;
            
            if (t > 0) {
                dstate[(b * T + t - 1) * D + i] += dst_i * dc;
            }
            local_ddecay += dst_i * (sp - kt * vt) * dc * (1.0f - dc);
        }
        atomic_fetch_add_explicit((device atomic_float*)&ddecay[i], local_ddecay, memory_order_relaxed);
    }
}

// ----------------------------------------------------------------------------
// Element-wise Add (Residuals)
// ----------------------------------------------------------------------------
kernel void add_fwd(device const float *a [[buffer(0)]],
                    device const float *b [[buffer(1)]],
                    device float *out [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    out[id] = a[id] + b[id];
}

// ----------------------------------------------------------------------------
// Time Mixing
// ----------------------------------------------------------------------------
inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

kernel void tmix_fwd(device const float *x [[buffer(0)]],
                     device const float *mix_w [[buffer(1)]],
                     device float *out [[buffer(2)]],
                     constant int &T [[buffer(3)]],
                     constant int &D [[buffer(4)]],
                     uint3 gid [[threadgroup_position_in_grid]], // t, b, 1
                     uint3 tid [[thread_position_in_threadgroup]],
                     uint3 tgid [[threads_per_threadgroup]]) {
    int t = gid.x;
    int b = gid.y;
    device const float *cur = x + (b * T + t) * D;
    device const float *prev = (t > 0) ? (x + (b * T + (t - 1)) * D) : nullptr;
    device float *out_t = out + (b * T + t) * D;
    
    for (int i = tid.x; i < D; i += tgid.x) {
        float m = sigmoid(mix_w[i]);
        float p_ = prev ? prev[i] : 0.0f;
        out_t[i] = m * cur[i] + (1.0f - m) * p_;
    }
}

kernel void tmix_bwd(device const float *dout [[buffer(0)]],
                     device const float *x [[buffer(1)]],
                     device const float *mix_w [[buffer(2)]],
                     device float *dx [[buffer(3)]],
                     device float *dmix_w [[buffer(4)]],
                     constant int &T [[buffer(5)]],
                     constant int &D [[buffer(6)]],
                     uint3 gid [[threadgroup_position_in_grid]], // t, b, 1
                     uint3 tid [[thread_position_in_threadgroup]],
                     uint3 tgid [[threads_per_threadgroup]]) {
    int t = gid.x;
    int b = gid.y;
    device const float *dout_t = dout + (b * T + t) * D;
    device const float *cur = x + (b * T + t) * D;
    device const float *prev = (t > 0) ? (x + (b * T + (t - 1)) * D) : nullptr;
    
    for (int i = tid.x; i < D; i += tgid.x) {
        float mix_val = mix_w[i];
        float m = sigmoid(mix_val);
        float d_m = m * (1.0f - m);
        float p_ = prev ? prev[i] : 0.0f;
        
        float d = dout_t[i];
        
        // d(mix_w)
        float dmix = d * (cur[i] - p_) * d_m;
        if (dmix_w) atomic_fetch_add_explicit((device atomic_float*)&dmix_w[i], dmix, memory_order_relaxed);
        
        // dx
        atomic_fetch_add_explicit((device atomic_float*)&dx[(b * T + t) * D + i], d * m, memory_order_relaxed);
        if (prev) {
            atomic_fetch_add_explicit((device atomic_float*)&dx[(b * T + (t - 1)) * D + i], d * (1.0f - m), memory_order_relaxed);
        }
    }
}

// ----------------------------------------------------------------------------
// Embedding Lookup
// ----------------------------------------------------------------------------
kernel void embed_fwd(device const float *embed [[buffer(0)]],
                      device const int *ids [[buffer(1)]],
                      device float *out [[buffer(2)]],
                      constant int &T [[buffer(3)]],
                      constant int &D [[buffer(4)]],
                      uint3 gid [[threadgroup_position_in_grid]],
                      uint3 tid [[thread_position_in_threadgroup]],
                      uint3 tgid [[threads_per_threadgroup]]) {
    int t = gid.x;
    int b = gid.y;
    int id = ids[b * T + t];
    device const float *emb = embed + id * D;
    device float *out_t = out + (b * T + t) * D;
    for (int i = tid.x; i < D; i += tgid.x) {
        out_t[i] = emb[i];
    }
}

kernel void embed_bwd(device const float *dout [[buffer(0)]],
                      device const int *ids [[buffer(1)]],
                      device float *dembed [[buffer(2)]],
                      constant int &T [[buffer(3)]],
                      constant int &D [[buffer(4)]],
                      uint3 gid [[threadgroup_position_in_grid]],
                      uint3 tid [[thread_position_in_threadgroup]],
                      uint3 tgid [[threads_per_threadgroup]]) {
    int t = gid.x;
    int b = gid.y;
    int id = ids[b * T + t];
    device const float *dout_t = dout + (b * T + t) * D;
    for (int i = tid.x; i < D; i += tgid.x) {
        atomic_fetch_add_explicit((device atomic_float*)&dembed[id * D + i], dout_t[i], memory_order_relaxed);
    }
}
