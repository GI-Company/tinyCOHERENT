#include <metal_stdlib>
using namespace metal;

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
                    uint3 gid [[threadgroup_position_in_grid]],
                    uint tid [[thread_position_in_threadgroup]],
                    uint tgid [[threads_per_threadgroup]]) {
    // WKV is a recurrence. It must be computed sequentially over T for each channel.
    // gid.x = batch index b
    int b = gid.x;
    
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
                     uint tid [[thread_position_in_threadgroup]],
                     uint tgid [[threads_per_threadgroup]]) {
    // gid.x = token index t
    // gid.y = head index h
    // gid.z = batch index b
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
    
    // Pass 1 & 2: Dot product for all u <= t and Max
    float maxs = -1e30f;
    for (int u = 0; u <= t; u++) {
        device const float *kh = K + (b * T + u) * kv_D + kv_h * kv_Dh;
        
        float local_dot = 0.0f;
        for (int i = tid; i < Dh; i += tgid) {
            local_dot += qh[i] * kh[i];
        }
        float simd_dot = simd_sum(local_dot);
        
        threadgroup float sum_dot[32];
        if (tid % 32 == 0) sum_dot[tid / 32] = simd_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid < 32) {
            float val = (tid < (tgid / 32)) ? sum_dot[tid] : 0.0f;
            simd_dot = simd_sum(val);
            if (tid == 0) {
                float s = simd_dot * invsqrt_dh;
                scores[u] = s;
                if (s > maxs) maxs = s;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Broadcast maxs
    threadgroup float tg_max[1];
    if (tid == 0) tg_max[0] = maxs;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maxs = tg_max[0];
    
    // Pass 3: Exp and Sum
    float sum_w = 0.0f;
    for (int u = 0; u <= t; u++) {
        if (tid == 0) {
            float w = exp(scores[u] - maxs);
            scores[u] = w; // Store unnormalized weight back to scores
            sum_w += w;
        }
    }
    
    // Broadcast sum_w
    threadgroup float tg_sum[1];
    if (tid == 0) tg_sum[0] = sum_w;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum_w = tg_sum[0];
    
    // Write normalized weights and accumulate ctx
    device float *wrow = attn_w + (b * H * T * T) + (h * T + t) * T;
    
    for (int i = tid; i < Dh; i += tgid) {
        float ctx_val = 0.0f;
        for (int u = 0; u <= t; u++) {
            float w = scores[u] / sum_w;
            if (i == 0) wrow[u] = w; // only one thread writes wrow
            device const float *vh = V + (b * T + u) * kv_D + kv_h * kv_Dh;
            ctx_val += w * vh[i];
        }
        attn_ctx[(b * T + t) * D + h * Dh + i] = ctx_val;
    }
}
