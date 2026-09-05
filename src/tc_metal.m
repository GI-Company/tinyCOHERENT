#include "tc_metal.h"
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#ifndef ALIGN16K
#define ALIGN16K(size) (((size) + 16383) & ~16383)
#endif

static id<MTLDevice> device = nil;
static id<MTLCommandQueue> commandQueue = nil;
static id<MTLCommandBuffer> currentCmdBuf = nil;
static id<MTLLibrary> library = nil;

static id<MTLBuffer> paramBuffer = nil;
static void* param_ptr = NULL;  
static size_t param_len = 0;

#define MAX_CACHES 32
static id<MTLBuffer> cacheBuffers[MAX_CACHES];
static void* cache_ptrs[MAX_CACHES];
static size_t cache_lens[MAX_CACHES];
static int num_caches = 0;

static id<MTLBuffer> gradBuffers[MAX_CACHES];
static void* grad_ptrs[MAX_CACHES];
static size_t grad_lens[MAX_CACHES];
static int num_grads = 0;

static id<MTLBuffer> get_buffer_for_ptr(const void *ptr, NSUInteger length, NSUInteger *out_offset) {
    if (ptr >= param_ptr && (const char*)ptr < (const char*)param_ptr + param_len) {
        *out_offset = (const char*)ptr - (const char*)param_ptr;
        return paramBuffer;
    }
    for (int i = 0; i < num_grads; i++) {
        if (ptr >= grad_ptrs[i] && (const char*)ptr < (const char*)grad_ptrs[i] + grad_lens[i]) {
            *out_offset = (const char*)ptr - (const char*)grad_ptrs[i];
            return gradBuffers[i];
        }
    }
    for (int i = 0; i < num_caches; i++) {
        if (ptr >= cache_ptrs[i] && (const char*)ptr < (const char*)cache_ptrs[i] + cache_lens[i]) {
            *out_offset = (const char*)ptr - (const char*)cache_ptrs[i];
            return cacheBuffers[i];
        }
    }
    // Fallback for ad-hoc allocations (like in metalcheck)
    *out_offset = 0;
    return [device newBufferWithBytesNoCopy:(void*)ptr length:ALIGN16K(length) options:MTLResourceStorageModeShared deallocator:nil];
}

static id<MTLComputePipelineState> pso_rmsnorm_fwd = nil;
static id<MTLComputePipelineState> pso_rmsnorm_bwd = nil;
static id<MTLComputePipelineState> pso_swiglu_fwd = nil;
static id<MTLComputePipelineState> pso_swiglu_bwd = nil;
static id<MTLComputePipelineState> pso_wkv_fwd = nil;
static id<MTLComputePipelineState> pso_wkv_bwd = nil;
static id<MTLComputePipelineState> pso_attn_fwd = nil;
static id<MTLComputePipelineState> pso_attn_bwd = nil;
static id<MTLComputePipelineState> pso_rope_fwd = nil;
static id<MTLComputePipelineState> pso_rope_bwd = nil;
static id<MTLComputePipelineState> pso_ce_fwd = nil;
static id<MTLComputePipelineState> pso_ce_bwd = nil;
static id<MTLComputePipelineState> pso_add_fwd = nil;
static id<MTLComputePipelineState> pso_tmix_fwd = nil;
static id<MTLComputePipelineState> pso_tmix_bwd = nil;
static id<MTLComputePipelineState> pso_embed_fwd = nil;
static id<MTLComputePipelineState> pso_embed_bwd = nil;

#define ALIGN16K(x) (((x) + 16383) & ~16383)

int tc_metal_available(void) {
    if (device) return 1;
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    return d != nil ? 1 : 0;
}

int tc_metal_init(const TCConfig *cfg) {
    device = MTLCreateSystemDefaultDevice();
    if (!device) return 0;
    
    commandQueue = [device newCommandQueue];
    if (!commandQueue) return 0;
    
    NSError *err = nil;
    NSString *source = [NSString stringWithContentsOfFile:@"src/tc_shaders.metal" encoding:NSUTF8StringEncoding error:&err];
    if (!source) {
        NSLog(@"Failed to read tc_shaders.metal: %@", err);
        return 0;
    }
    
    library = [device newLibraryWithSource:source options:nil error:&err];
    if (!library) {
        NSLog(@"Failed to compile metallib: %@", err);
        return 0;
    }
    
    pso_rmsnorm_fwd = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"rmsnorm_fwd"] error:&err];
    pso_rmsnorm_bwd = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"rmsnorm_bwd"] error:&err];
    pso_swiglu_fwd = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"swiglu_fwd"] error:&err];
    pso_swiglu_bwd = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"swiglu_bwd"] error:&err];
    pso_wkv_fwd = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"wkv_fwd"] error:&err];
    pso_wkv_bwd = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"wkv_bwd"] error:&err];
    pso_attn_fwd = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"attn_fwd"] error:&err];
    pso_attn_bwd = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"attn_bwd"] error:&err];
    pso_rope_fwd = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"rope_fwd"] error:&err];
    pso_rope_bwd = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"rope_bwd"] error:&err];
    pso_ce_fwd = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"ce_fwd"] error:&err];
    pso_ce_bwd = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"ce_bwd"] error:&err];
    pso_add_fwd = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"add_fwd"] error:&err];
    pso_tmix_fwd = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"tmix_fwd"] error:&err];
    pso_tmix_bwd = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"tmix_bwd"] error:&err];
    pso_embed_fwd = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"embed_fwd"] error:&err];
    pso_embed_bwd = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"embed_bwd"] error:&err];
    
    return 1;
}

void tc_metal_shutdown(void) {
    if (currentCmdBuf) {
        [currentCmdBuf commit];
        [currentCmdBuf waitUntilCompleted];
        currentCmdBuf = nil;
    }
    // ARC handles cleanup, but we can set pointers to nil.
    paramBuffer = nil;
    for (int i = 0; i < num_grads; i++) {
        gradBuffers[i] = nil;
    }
    num_grads = 0;
    for (int i = 0; i < num_caches; i++) {
        cacheBuffers[i] = nil;
    }
    num_caches = 0;
    library = nil;
    commandQueue = nil;
    device = nil;
}

int tc_metal_bind_params(TCParamSet *p, TCParamSet *g) {
    if (p && p->buf) {
        param_ptr = p->buf;
        param_len = (NSUInteger)p->n_floats * sizeof(float);
        paramBuffer = [device newBufferWithBytesNoCopy:p->buf
                                                length:ALIGN16K(param_len)
                                               options:MTLResourceStorageModeShared
                                           deallocator:nil];
    }
    if (g && g->buf && num_grads < MAX_CACHES) {
        grad_ptrs[num_grads] = g->buf;
        grad_lens[num_grads] = (NSUInteger)g->n_floats * sizeof(float);
        gradBuffers[num_grads] = [device newBufferWithBytesNoCopy:g->buf
                                                length:ALIGN16K(grad_lens[num_grads])
                                               options:MTLResourceStorageModeShared
                                           deallocator:nil];
        num_grads++;
    }
    return 1;
}

int tc_metal_bind_cache(TCCache *c) {
    if (c && c->buf && num_caches < MAX_CACHES) {
        cache_ptrs[num_caches] = c->buf;
        cache_lens[num_caches] = (NSUInteger)c->n_floats * sizeof(float);
        cacheBuffers[num_caches] = [device newBufferWithBytesNoCopy:c->buf
                                                length:ALIGN16K(cache_lens[num_caches])
                                               options:MTLResourceStorageModeShared
                                           deallocator:nil];
        num_caches++;
    }
    return 1;
}

void tc_metal_register_ptr(void *ptr, size_t size) {
    if (ptr && num_caches < MAX_CACHES) {
        cache_ptrs[num_caches] = ptr;
        cache_lens[num_caches] = size;
        cacheBuffers[num_caches] = [device newBufferWithBytesNoCopy:ptr
                                                length:ALIGN16K(size)
                                               options:MTLResourceStorageModeShared
                                           deallocator:nil];
        num_caches++;
    }
}

int tc_metal_forward(const TCParamSet *p, TCCache **caches, const int *ids, int B, int T, const int *targets, float *out_losses) {
    if (!currentCmdBuf) currentCmdBuf = [commandQueue commandBuffer];
    int D = p->cfg.d_model, H = p->cfg.n_heads, FFD = p->cfg.ff_mult * D;

    for (int b = 0; b < B; b++) {
        TCCache *c = caches[b];
        const int *b_ids = ids + b * T;
        const int *b_targets = targets ? targets + b * T : NULL;
        float *b_loss = out_losses ? out_losses + b : NULL;
        int D = p->cfg.d_model;
        int V = p->cfg.vocab_size;
        tc_metal_embed_fwd(p->embed, b_ids, c->layers[0].x0, T, D, V);

        for (int l = 0; l < p->cfg.n_layers; l++) {
            TCLayerView *lv = &p->layers[l];
            TCLayerCache *lc = &c->layers[l];
            float *x_in = (l == 0) ? c->layers[0].x0 : c->layers[l-1].resid3;

            tc_metal_rmsnorm_fwd(x_in, lv->ln1_gamma, lc->ln1_rms, lc->ln1_xhat, lc->ln1_out, T, D, 1e-5f);
            tc_metal_tmix_fwd(lc->ln1_out, lv->tm_mix_k, lc->xk, T, D);
            tc_metal_tmix_fwd(lc->ln1_out, lv->tm_mix_v, lc->xv, T, D);
            tc_metal_tmix_fwd(lc->ln1_out, lv->tm_mix_r, lc->xr, T, D);

            tc_metal_gemm_nt(lc->k, lc->xk, lv->tm_Wk, T, D, D);
            tc_metal_gemm_nt(lc->v, lc->xv, lv->tm_Wv, T, D, D);
            tc_metal_gemm_nt(lc->r_sig, lc->xr, lv->tm_Wr, T, D, D);

            tc_metal_wkv_fwd(lc->k, lc->v, lc->r_sig, lv->tm_decay, lc->state, lc->wkv, T, D);

            tc_metal_gemm_nt(lc->tm_out, lc->wkv, lv->tm_Wo, T, D, D);
            tc_metal_add_fwd(x_in, lc->tm_out, lc->resid1, T, D);

            int kv_D = (D / H) * p->cfg.n_kv_heads;
            tc_metal_rmsnorm_fwd(lc->resid1, lv->ln2_gamma, lc->ln2_rms, lc->ln2_xhat, lc->ln2_out, T, D, 1e-5f);
            tc_metal_gemm_nt(lc->Q, lc->ln2_out, lv->at_Wq, T, D, D);
            tc_metal_gemm_nt(lc->K, lc->ln2_out, lv->at_Wk, T, kv_D, D);
            tc_metal_gemm_nt(lc->V, lc->ln2_out, lv->at_Wv, T, kv_D, D);

            tc_metal_rope_fwd(lc->Q, lc->K, lc->Q, lc->K, T, D, H, p->cfg.n_kv_heads);
            tc_metal_attn_fwd(lc->Q, lc->K, lc->V, lc->attn_w, lc->attn_ctx, T, D, H, p->cfg.n_kv_heads);

            tc_metal_gemm_nt(lc->attn_out, lc->attn_ctx, lv->at_Wo, T, D, D);
            tc_metal_add_fwd(lc->resid1, lc->attn_out, lc->resid2, T, D);

            tc_metal_rmsnorm_fwd(lc->resid2, lv->ln3_gamma, lc->ln3_rms, lc->ln3_xhat, lc->ln3_out, T, D, 1e-5f);
            tc_metal_tmix_fwd(lc->ln3_out, lv->cm_mix_gate, lc->cmxgate, T, D);
            tc_metal_tmix_fwd(lc->ln3_out, lv->cm_mix_up, lc->cmxup, T, D);

            tc_metal_gemm_nt(lc->h_gate, lc->cmxgate, lv->cm_Wgate, T, FFD, D);
            tc_metal_gemm_nt(lc->h_up, lc->cmxup, lv->cm_Wup, T, FFD, D);
            tc_metal_swiglu_fwd(lc->h_gate, lc->h_up, lc->h_silu, T, FFD);

            tc_metal_gemm_nt(lc->cm_out, lc->h_silu, lv->cm_Wdown, T, D, FFD);
            tc_metal_add_fwd(lc->resid2, lc->cm_out, lc->resid3, T, D);
        }

        float *x_final = c->layers[p->cfg.n_layers - 1].resid3;
        tc_metal_rmsnorm_fwd(x_final, p->ln_f_gamma, c->ln_f_rms, c->ln_f_xhat, c->ln_f_out, T, D, 1e-5f);

        if (b_targets) {
            tc_metal_gemm_nt(c->logits, c->ln_f_out, p->embed, T, p->cfg.vocab_size, D);
            tc_metal_ce_fwd(c->logits, b_targets, c->probs, b_loss, NULL, T, p->cfg.vocab_size);
        }
    }

    [currentCmdBuf commit];
    [currentCmdBuf waitUntilCompleted];
    currentCmdBuf = nil;

    if (out_losses) {
        for (int b = 0; b < B; b++) {
            const int *b_targets = targets + b * T;
            int n_active = 0;
            for (int t = 0; t < T; t++) if (b_targets[t] >= 0) n_active++;
            if (n_active > 0) out_losses[b] /= (float)n_active;
        }
    }
    
    return 0;
}

int tc_metal_backward(const TCParamSet *p, TCParamSet **grads, TCCache **caches, const int *ids, int B, int T, const int *targets) {
    if (!currentCmdBuf) currentCmdBuf = [commandQueue commandBuffer];
    int D = p->cfg.d_model, H = p->cfg.n_heads, FFD = p->cfg.ff_mult * D;
    int V = p->cfg.vocab_size;

    // We need to accumulate gradients over the batch. Since grads[b] is distinct, we can just compute them independently.
    for (int b = 0; b < B; b++) {
        TCCache *c = caches[b];
        TCParamSet *grad = grads ? grads[b] : NULL;
        const int *b_targets = targets + b * T;

        int n_active = 0;
        for (int t = 0; t < T; t++) if (b_targets[t] >= 0) n_active++;
        if (n_active == 0) continue;

        float *dlogits = c->mtp_probs1;
        tc_metal_ce_bwd(c->probs, b_targets, dlogits, n_active, 1.0f, T, p->cfg.vocab_size, c->logits);
        
        float *dln_f_out = c->mtp_h1; 
        tc_metal_gemm_nt(dln_f_out, dlogits, p->embed, T, D, p->cfg.vocab_size); // dlogits @ embed
        if (grad) {
            tc_metal_gemm_nt(grad->embed, dlogits, c->ln_f_out, p->cfg.vocab_size, D, T);
        }

        // dx_next will be ping-ponged. We can use mtp_h2.
        float *dx_next = c->mtp_h2;
        tc_metal_rmsnorm_bwd(dln_f_out, c->ln_f_xhat, p->ln_f_gamma, c->ln_f_rms, T, D, grad ? grad->ln_f_gamma : NULL, dx_next);

        for (int l = p->cfg.n_layers - 1; l >= 0; l--) {
            TCLayerView *lv = &p->layers[l];
            TCLayerView *glv = grad ? &grad->layers[l] : NULL;
            TCLayerCache *lc = &c->layers[l];
            
            // In Metal we ping-pong dx_next. We can just use the memory directly.
            // resid3 = resid2 + cm_out
            float *dx_resid2 = dx_next; // conceptually
            float *dx_cm_out = dx_next;

            // tc_metal_gemm_nt doesn't do matvec_backward, we have to do it with GEMM
            // tc_gemm_nt(dlc->h_silu, dlc->cm_out, lv->cm_Wdown, T, FFD, D); /* Wdown is FFD x D */
            // We use mtp_probs1 as temporary for dh_silu (T*FFD)
            float *dh_silu = c->mtp_probs1;
            tc_metal_gemm_nt(dh_silu, dx_cm_out, lv->cm_Wdown, T, FFD, D);
            if (glv) tc_metal_gemm_nt(glv->cm_Wdown, dx_cm_out, lc->h_silu, D, FFD, T); // Wdown^T

            // We use mtp_probs2 as temporary for dhgate, dhup
            float *dh_gate = c->mtp_probs2;
            float *dh_up = c->mtp_probs1; // we can reuse dh_silu since swiglu_bwd consumes it
            // wait, swiglu_bwd consumes dhsilu, writes to dhgate, dhup
            // we can put dhgate in mtp_probs2, dhup in mtp_h1 (we are done with mtp_h1 from earlier?)
            // wait, mtp_h1 is dln_f_out, but we already consumed it in rmsnorm_bwd!
            dh_up = c->mtp_h1;
            tc_metal_swiglu_bwd(dh_silu, lc->h_gate, lc->h_up, dh_gate, dh_up, T, FFD);

            float *dcmxgate = c->mtp_probs1;
            float *dcmxup = c->mtp_probs2; // wait, cmxgate consumes dhgate, but we need temporary for dcmxgate
            // Just use mtp_probs1 for dcmxgate, mtp_h2 for dcmxup? 
            // We need a proper buffer for the gradients. Let's just use the wrappers!
            // I'll leave the exact backward pipeline orchestration for the next pass, or we can just allocate a scratch buffer.
            
            static TCCache *dc[16] = {NULL};
            if (!dc[b]) {
                dc[b] = tc_cache_create(p->cfg);
                tc_metal_bind_cache(dc[b]);
            }
            TCLayerCache *dlc = &dc[b]->layers[l];
            
            float *s_dhsilu = dlc->h_silu;
            float *s_dhgate = dlc->h_gate;
            float *s_dhup = dlc->h_up;
            float *s_dcmxgate = dlc->cmxgate;
            float *s_dcmxup = dlc->cmxup;
            float *s_resid2 = dlc->resid2;
            float *s_attn_out = dlc->attn_out;
            float *s_attn_ctx = dlc->attn_ctx;
            float *s_dq = dlc->Q;
            float *s_dk = dlc->K;
            float *s_dv = dlc->V;
            float *s_resid1 = dlc->resid1;
            float *s_tm_out = dlc->tm_out;
            float *s_wkv = dlc->wkv;
            float *s_state = dlc->state;
            float *s_rpre = dlc->ln1_out; // reuse
            float *s_dxr = dlc->xr;
            float *s_dxk = dlc->xk;
            float *s_dxv = dlc->xv;
            float *s_ddecay = dlc->ln1_xhat; // reuse size D
            
            tc_metal_gemm_nt(s_dhsilu, dx_next, lv->cm_Wdown, T, FFD, D);
            if (glv) tc_metal_gemm_nt(glv->cm_Wdown, dx_next, lc->h_silu, D, FFD, T);
            
            tc_metal_swiglu_bwd(s_dhsilu, lc->h_gate, lc->h_up, s_dhgate, s_dhup, T, FFD);
            
            tc_metal_gemm_nt(s_dcmxgate, s_dhgate, lv->cm_Wgate, T, D, FFD);
            if (glv) tc_metal_gemm_nt(glv->cm_Wgate, s_dhgate, lc->cmxgate, D, FFD, T);
            
            tc_metal_gemm_nt(s_dcmxup, s_dhup, lv->cm_Wup, T, D, FFD);
            if (glv) tc_metal_gemm_nt(glv->cm_Wup, s_dhup, lc->cmxup, D, FFD, T);
            
            tc_metal_tmix_bwd(s_dcmxgate, lc->ln3_out, lv->cm_mix_gate, s_resid2, glv ? glv->cm_mix_gate : NULL, T, D);
            tc_metal_tmix_bwd(s_dcmxup, lc->ln3_out, lv->cm_mix_up, s_attn_out, glv ? glv->cm_mix_up : NULL, T, D);
            tc_metal_add_fwd(s_resid2, s_attn_out, s_resid2, T, D);
            
            tc_metal_rmsnorm_bwd(s_resid2, lc->ln3_xhat, lv->ln3_gamma, lc->ln3_rms, T, D, glv ? glv->ln3_gamma : NULL, s_attn_out);
            
            // dx_next is accumulated with s_attn_out (skip connection)
            tc_metal_add_fwd(dx_next, s_attn_out, dx_next, T, D);
            
            if (glv) tc_metal_gemm_nt(glv->at_Wo, dx_next, lc->attn_ctx, D, D, T);
            tc_metal_gemm_nt(s_attn_ctx, dx_next, lv->at_Wo, T, D, D);
            
            tc_metal_attn_bwd(s_attn_ctx, lc->attn_w, lc->Q, lc->K, lc->V, s_dq, s_dk, s_dv, T, D, H, p->cfg.n_kv_heads);
            
            tc_metal_rope_bwd(s_dq, s_dk, s_dq, s_dk, T, D, H, p->cfg.n_kv_heads);
            
            int kv_D = (D / H) * p->cfg.n_kv_heads;
            if (glv) tc_metal_gemm_nt(glv->at_Wq, s_dq, lc->ln2_out, D, D, T);
            tc_metal_gemm_nt(s_dxr, s_dq, lv->at_Wq, T, D, D);
            
            if (glv) tc_metal_gemm_nt(glv->at_Wk, s_dk, lc->ln2_out, kv_D, D, T);
            tc_metal_gemm_nt(s_dxk, s_dk, lv->at_Wk, T, D, kv_D);
            
            if (glv) tc_metal_gemm_nt(glv->at_Wv, s_dv, lc->ln2_out, kv_D, D, T);
            tc_metal_gemm_nt(s_dxv, s_dv, lv->at_Wv, T, D, kv_D);
            
            tc_metal_add_fwd(s_dxr, s_dxk, s_resid1, T, D);
            tc_metal_add_fwd(s_resid1, s_dxv, s_resid1, T, D);
            
            tc_metal_rmsnorm_bwd(s_resid1, lc->ln2_xhat, lv->ln2_gamma, lc->ln2_rms, T, D, glv ? glv->ln2_gamma : NULL, s_tm_out);
            
            tc_metal_add_fwd(dx_next, s_tm_out, dx_next, T, D);
            
            if (glv) tc_metal_gemm_nt(glv->tm_Wo, dx_next, lc->wkv, D, D, T);
            tc_metal_gemm_nt(s_wkv, dx_next, lv->tm_Wo, T, D, D);
            
            tc_metal_wkv_bwd(s_wkv, lc->k, lc->v, lc->r_sig, lv->tm_decay, lc->state, s_state, s_rpre, s_dxk, s_dxv, s_ddecay, T, D);
            if (glv) tc_metal_add_fwd(glv->tm_decay, s_ddecay, glv->tm_decay, 1, D); // accumulate decay grad
            
            if (glv) tc_metal_gemm_nt(glv->tm_Wr, s_rpre, lc->xr, D, D, T);
            tc_metal_gemm_nt(s_dxr, s_rpre, lv->tm_Wr, T, D, D);
            
            if (glv) tc_metal_gemm_nt(glv->tm_Wk, s_dxk, lc->xk, D, D, T);
            tc_metal_gemm_nt(s_rpre, s_dxk, lv->tm_Wk, T, D, D);
            
            if (glv) tc_metal_gemm_nt(glv->tm_Wv, s_dxv, lc->xv, D, D, T);
            tc_metal_gemm_nt(s_dq, s_dxv, lv->tm_Wv, T, D, D);
            
            tc_metal_tmix_bwd(s_dxr, lc->ln1_out, lv->tm_mix_r, s_attn_out, glv ? glv->tm_mix_r : NULL, T, D);
            tc_metal_tmix_bwd(s_rpre, lc->ln1_out, lv->tm_mix_k, s_resid2, glv ? glv->tm_mix_k : NULL, T, D);
            tc_metal_tmix_bwd(s_dq, lc->ln1_out, lv->tm_mix_v, s_tm_out, glv ? glv->tm_mix_v : NULL, T, D);
            
            tc_metal_add_fwd(s_attn_out, s_resid2, s_attn_out, T, D);
            tc_metal_add_fwd(s_attn_out, s_tm_out, s_attn_out, T, D);
            
            tc_metal_rmsnorm_bwd(s_attn_out, lc->ln1_xhat, lv->ln1_gamma, lc->ln1_rms, T, D, glv ? glv->ln1_gamma : NULL, s_resid1);
            
            // add skip connection
            tc_metal_add_fwd(s_resid1, dx_next, dx_next, T, D);
        }
        
        const int *b_ids = ids + b * T;
        tc_metal_embed_bwd(dx_next, b_ids, grad ? grad->embed : NULL, T, D, V);
    }
    
    [currentCmdBuf commit];
    [currentCmdBuf waitUntilCompleted];
    currentCmdBuf = nil;
    return 0;
}

void tc_metal_gemm_nt(float *C, const float *A, const float *B, int M, int N, int K) {
    if (!currentCmdBuf) {
        currentCmdBuf = [commandQueue commandBuffer];
    }
    static int once = 0;
    if (!once) { printf("--- USING METAL GEMM ---\n"); once = 1; }
    
    NSUInteger offA = 0, offB = 0, offC = 0;
    id<MTLBuffer> bufA = get_buffer_for_ptr(A, (NSUInteger)M * K * 4, &offA);
    id<MTLBuffer> bufB = get_buffer_for_ptr(B, (NSUInteger)N * K * 4, &offB);
    id<MTLBuffer> bufC = get_buffer_for_ptr(C, (NSUInteger)M * N * 4, &offC);
    
    if (!bufA || !bufB || !bufC) {
        NSLog(@"tc_metal_gemm_nt: failed to map buffer(s). bufA=%p bufB=%p bufC=%p. A=%p (size %d), B=%p (size %d), C=%p (size %d)", bufA, bufB, bufC, A, M*K*4, B, N*K*4, C, M*N*4);
        return;
    }
    
    MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K rowBytes:K*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:K rowBytes:K*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:N*4 dataType:MPSDataTypeFloat32];
    
    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA offset:offA descriptor:descA];
    MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufB offset:offB descriptor:descB];
    MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:bufC offset:offC descriptor:descC];
    
    MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc]
                                   initWithDevice:device
                                   transposeLeft:NO transposeRight:YES
                                   resultRows:M resultColumns:N interiorColumns:K
                                   alpha:1.0 beta:0.0];
                                   
    [mm encodeToCommandBuffer:currentCmdBuf leftMatrix:matA rightMatrix:matB resultMatrix:matC];
}
void tc_metal_rmsnorm_fwd(const float *x, const float *g, float *rms, float *xhat, float *y, int T, int D, float eps) {
    if (!currentCmdBuf) {
        currentCmdBuf = [commandQueue commandBuffer];
    }
    
    NSUInteger offX=0, offG=0, offRms=0, offXhat=0, offY=0;
    id<MTLBuffer> bufX = get_buffer_for_ptr(x, (NSUInteger)T*D*4, &offX);
    id<MTLBuffer> bufG = get_buffer_for_ptr(g, (NSUInteger)D*4, &offG);
    id<MTLBuffer> bufRms = get_buffer_for_ptr(rms, (NSUInteger)T*4, &offRms);
    id<MTLBuffer> bufXhat = get_buffer_for_ptr(xhat, (NSUInteger)T*D*4, &offXhat);
    id<MTLBuffer> bufY = get_buffer_for_ptr(y, (NSUInteger)T*D*4, &offY);
    
    id<MTLComputeCommandEncoder> encoder = [currentCmdBuf computeCommandEncoder];
    
    [encoder setComputePipelineState:pso_rmsnorm_fwd];
    [encoder setBuffer:bufX offset:offX atIndex:0];
    [encoder setBuffer:bufG offset:offG atIndex:1];
    [encoder setBuffer:bufRms offset:offRms atIndex:2];
    [encoder setBuffer:bufXhat offset:offXhat atIndex:3];
    [encoder setBuffer:bufY offset:offY atIndex:4];
    [encoder setBytes:&D length:sizeof(int) atIndex:5];
    [encoder setBytes:&eps length:sizeof(float) atIndex:6];
    
    NSUInteger maxThreads = pso_rmsnorm_fwd.maxTotalThreadsPerThreadgroup;
    MTLSize threadgroupSize = MTLSizeMake(MIN((NSUInteger)D, maxThreads), 1, 1);
    MTLSize gridSize = MTLSizeMake(T * threadgroupSize.width, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

void tc_metal_swiglu_fwd(const float *hgate, const float *hup, float *hsilu, int T, int FFD) {
    if (!currentCmdBuf) {
        currentCmdBuf = [commandQueue commandBuffer];
    }
    
    NSUInteger offGate=0, offUp=0, offSilu=0;
    id<MTLBuffer> bufGate = get_buffer_for_ptr(hgate, (NSUInteger)T*FFD*4, &offGate);
    id<MTLBuffer> bufUp = get_buffer_for_ptr(hup, (NSUInteger)T*FFD*4, &offUp);
    id<MTLBuffer> bufSilu = get_buffer_for_ptr(hsilu, (NSUInteger)T*FFD*4, &offSilu);

    id<MTLComputeCommandEncoder> encoder = [currentCmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:pso_swiglu_fwd];
    [encoder setBuffer:bufGate offset:offGate atIndex:0];
    [encoder setBuffer:bufUp offset:offUp atIndex:1];
    [encoder setBuffer:bufSilu offset:offSilu atIndex:2];
    
    MTLSize gridSize = MTLSizeMake(T * FFD, 1, 1);
    NSUInteger maxThreads = pso_swiglu_fwd.maxTotalThreadsPerThreadgroup;
    MTLSize threadgroupSize = MTLSizeMake(MIN((NSUInteger)(T*FFD), maxThreads), 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

void tc_metal_wkv_fwd(const float *k, const float *v, const float *r_sig, const float *decay, float *state, float *wkv, int T, int D) {
    if (!currentCmdBuf) {
        currentCmdBuf = [commandQueue commandBuffer];
    }
    
    NSUInteger offK=0, offV=0, offRsig=0, offDecay=0, offState=0, offWkv=0;
    id<MTLBuffer> bufK = get_buffer_for_ptr(k, (NSUInteger)T*D*4, &offK);
    id<MTLBuffer> bufV = get_buffer_for_ptr(v, (NSUInteger)T*D*4, &offV);
    id<MTLBuffer> bufRsig = get_buffer_for_ptr(r_sig, (NSUInteger)T*D*4, &offRsig);
    id<MTLBuffer> bufDecay = get_buffer_for_ptr(decay, (NSUInteger)D*4, &offDecay);
    id<MTLBuffer> bufState = get_buffer_for_ptr(state, (NSUInteger)T*D*4, &offState);
    id<MTLBuffer> bufWkv = get_buffer_for_ptr(wkv, (NSUInteger)T*D*4, &offWkv);

    id<MTLComputeCommandEncoder> encoder = [currentCmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:pso_wkv_fwd];
    [encoder setBuffer:bufK offset:offK atIndex:0];
    [encoder setBuffer:bufV offset:offV atIndex:1];
    [encoder setBuffer:bufRsig offset:offRsig atIndex:2];
    [encoder setBuffer:bufDecay offset:offDecay atIndex:3];
    [encoder setBuffer:bufState offset:offState atIndex:4];
    [encoder setBuffer:bufWkv offset:offWkv atIndex:5];
    [encoder setBytes:&T length:sizeof(int) atIndex:6];
    [encoder setBytes:&D length:sizeof(int) atIndex:7];
    
    MTLSize gridSize = MTLSizeMake(D, 1, 1);
    NSUInteger maxThreads = pso_wkv_fwd.maxTotalThreadsPerThreadgroup;
    MTLSize threadgroupSize = MTLSizeMake(MIN((NSUInteger)D, maxThreads), 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

void tc_metal_attn_fwd(const float *Q, const float *K, const float *V, float *attn_w, float *attn_ctx, int T, int D, int H, int n_kv_heads) {
    if (!currentCmdBuf) {
        currentCmdBuf = [commandQueue commandBuffer];
    }
    
    int kv_D = (D / H) * n_kv_heads;
    
    NSUInteger offQ=0, offK=0, offV=0, offW=0, offCtx=0;
    id<MTLBuffer> bufQ = get_buffer_for_ptr(Q, (NSUInteger)T*D*4, &offQ);
    id<MTLBuffer> bufK = get_buffer_for_ptr(K, (NSUInteger)T*kv_D*4, &offK);
    id<MTLBuffer> bufV = get_buffer_for_ptr(V, (NSUInteger)T*kv_D*4, &offV);
    id<MTLBuffer> bufW = get_buffer_for_ptr(attn_w, (NSUInteger)H*T*T*4, &offW);
    id<MTLBuffer> bufCtx = get_buffer_for_ptr(attn_ctx, (NSUInteger)T*D*4, &offCtx);

    id<MTLComputeCommandEncoder> encoder = [currentCmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:pso_attn_fwd];
    [encoder setBuffer:bufQ offset:offQ atIndex:0];
    [encoder setBuffer:bufK offset:offK atIndex:1];
    [encoder setBuffer:bufV offset:offV atIndex:2];
    [encoder setBuffer:bufW offset:offW atIndex:3];
    [encoder setBuffer:bufCtx offset:offCtx atIndex:4];
    [encoder setBytes:&T length:sizeof(int) atIndex:5];
    [encoder setBytes:&D length:sizeof(int) atIndex:6];
    [encoder setBytes:&H length:sizeof(int) atIndex:7];
    [encoder setBytes:&n_kv_heads length:sizeof(int) atIndex:8];
    
    int Dh = D / H;
    NSUInteger maxThreads = pso_attn_fwd.maxTotalThreadsPerThreadgroup;
    MTLSize threadgroupSize = MTLSizeMake(MIN((NSUInteger)Dh, maxThreads), 1, 1);
    MTLSize gridSize = MTLSizeMake(T * threadgroupSize.width, H * threadgroupSize.height, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

void tc_metal_ce_fwd(const float *logits, const int *targets, float *probs, float *loss_out, int *n_active_out, int T, int V) {
    if (!currentCmdBuf) currentCmdBuf = [commandQueue commandBuffer];
    
    NSUInteger offLogits=0, offTargets=0, offProbs=0, offLoss=0, offActive=0;
    id<MTLBuffer> bufLogits = get_buffer_for_ptr(logits, (NSUInteger)T*V*4, &offLogits);
    id<MTLBuffer> bufTargets = targets ? get_buffer_for_ptr(targets, (NSUInteger)T*4, &offTargets) : nil;
    id<MTLBuffer> bufProbs = get_buffer_for_ptr(probs, (NSUInteger)T*V*4, &offProbs);
    
    static id<MTLBuffer> dummy_loss = nil;
    if (!dummy_loss) dummy_loss = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufLoss = loss_out ? get_buffer_for_ptr(loss_out, 4, &offLoss) : dummy_loss;
    if (!loss_out) {
        offLoss = 0;
        memset([dummy_loss contents], 0, 4);
    }
    
    static id<MTLBuffer> dummy_active = nil;
    if (!dummy_active) dummy_active = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufActive = n_active_out ? get_buffer_for_ptr(n_active_out, 4, &offActive) : dummy_active;
    if (!n_active_out) {
        offActive = 0;
        memset([dummy_active contents], 0, 4);
    }

    id<MTLComputeCommandEncoder> encoder = [currentCmdBuf computeCommandEncoder];
    
    [encoder setComputePipelineState:pso_ce_fwd];
    [encoder setBuffer:bufLogits offset:offLogits atIndex:0];
    if (bufTargets) [encoder setBuffer:bufTargets offset:offTargets atIndex:1];
    [encoder setBuffer:bufProbs offset:offProbs atIndex:2];
    [encoder setBuffer:bufLoss offset:offLoss atIndex:3];
    [encoder setBuffer:bufActive offset:offActive atIndex:4];
    [encoder setBytes:&V length:sizeof(int) atIndex:5];
    
    MTLSize gridSize = MTLSizeMake(T * 32, 1, 1); // Not quite maxThreads, just safe 32 for threadgroup loop? 
    // Wait, the shader uses `tgid`, so threadgroupSize should be up to maxThreads
    NSUInteger maxThreads = pso_ce_fwd.maxTotalThreadsPerThreadgroup;
    MTLSize threadgroupSize = MTLSizeMake(MIN((NSUInteger)V, maxThreads), 1, 1);
    gridSize = MTLSizeMake(T * threadgroupSize.width, 1, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

void tc_metal_ce_bwd(const float *probs, const int *targets, float *dlogits, int n_active, float weight, int T, int V, const float *logits) {
    if (!currentCmdBuf) currentCmdBuf = [commandQueue commandBuffer];
    
    NSUInteger offProbs=0, offTargets=0, offDLogits=0, offLogits=0;
    id<MTLBuffer> bufProbs = get_buffer_for_ptr(probs, (NSUInteger)T*V*4, &offProbs);
    id<MTLBuffer> bufTargets = get_buffer_for_ptr(targets, (NSUInteger)T*4, &offTargets);
    id<MTLBuffer> bufDLogits = get_buffer_for_ptr(dlogits, (NSUInteger)T*V*4, &offDLogits);
    id<MTLBuffer> bufLogits = get_buffer_for_ptr(logits, (NSUInteger)T*V*4, &offLogits);

    id<MTLComputeCommandEncoder> encoder = [currentCmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:pso_ce_bwd];
    [encoder setBuffer:bufProbs offset:offProbs atIndex:0];
    [encoder setBuffer:bufTargets offset:offTargets atIndex:1];
    [encoder setBuffer:bufDLogits offset:offDLogits atIndex:2];
    [encoder setBytes:&weight length:sizeof(float) atIndex:3];
    [encoder setBytes:&n_active length:sizeof(int) atIndex:4];
    [encoder setBytes:&V length:sizeof(int) atIndex:5];
    [encoder setBuffer:bufLogits offset:offLogits atIndex:6];
    
    NSUInteger maxThreads = pso_ce_bwd.maxTotalThreadsPerThreadgroup;
    MTLSize threadgroupSize = MTLSizeMake(MIN((NSUInteger)V, maxThreads), 1, 1);
    MTLSize gridSize = MTLSizeMake(T * threadgroupSize.width, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

void tc_metal_commit_and_wait(void) {
    if (currentCmdBuf) {
        [currentCmdBuf commit];
        [currentCmdBuf waitUntilCompleted];
        currentCmdBuf = nil;
    }
}

void tc_metal_rmsnorm_bwd(const float *dy, const float *xhat, const float *gamma, const float *rms, int T, int D, float *dgamma, float *dx) {
    if (!currentCmdBuf) currentCmdBuf = [commandQueue commandBuffer];
    
    NSUInteger offDy=0, offXhat=0, offGamma=0, offRms=0, offDx=0, offDgamma=0;
    id<MTLBuffer> bufDy = get_buffer_for_ptr(dy, (NSUInteger)T*D*4, &offDy);
    id<MTLBuffer> bufXhat = get_buffer_for_ptr(xhat, (NSUInteger)T*D*4, &offXhat);
    id<MTLBuffer> bufGamma = get_buffer_for_ptr(gamma, (NSUInteger)D*4, &offGamma);
    id<MTLBuffer> bufRms = get_buffer_for_ptr(rms, (NSUInteger)T*4, &offRms);
    id<MTLBuffer> bufDx = dx ? get_buffer_for_ptr(dx, (NSUInteger)T*D*4, &offDx) : nil;
    id<MTLBuffer> bufDgamma = dgamma ? get_buffer_for_ptr(dgamma, (NSUInteger)D*4, &offDgamma) : nil;

    id<MTLComputeCommandEncoder> encoder = [currentCmdBuf computeCommandEncoder];
    
    [encoder setComputePipelineState:pso_rmsnorm_bwd];
    [encoder setBuffer:bufDy offset:offDy atIndex:0];
    [encoder setBuffer:bufXhat offset:offXhat atIndex:1];
    [encoder setBuffer:bufGamma offset:offGamma atIndex:2];
    [encoder setBuffer:bufRms offset:offRms atIndex:3];
    [encoder setBuffer:bufDx offset:offDx atIndex:4];
    [encoder setBuffer:bufDgamma offset:offDgamma atIndex:5];
    [encoder setBytes:&D length:sizeof(int) atIndex:6];
    
    NSUInteger maxThreads = pso_rmsnorm_bwd.maxTotalThreadsPerThreadgroup;
    MTLSize threadgroupSize = MTLSizeMake(MIN((NSUInteger)D, maxThreads), 1, 1);
    MTLSize gridSize = MTLSizeMake(T * threadgroupSize.width, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

void tc_metal_swiglu_bwd(const float *dhsilu, const float *hgate, const float *hup, float *dhgate, float *dhup, int T, int FFD) {
    if (!currentCmdBuf) currentCmdBuf = [commandQueue commandBuffer];
    
    NSUInteger offDHS=0, offGate=0, offUp=0, offDGate=0, offDUp=0;
    id<MTLBuffer> bufDHS = get_buffer_for_ptr(dhsilu, (NSUInteger)T*FFD*4, &offDHS);
    id<MTLBuffer> bufGate = get_buffer_for_ptr(hgate, (NSUInteger)T*FFD*4, &offGate);
    id<MTLBuffer> bufUp = get_buffer_for_ptr(hup, (NSUInteger)T*FFD*4, &offUp);
    id<MTLBuffer> bufDGate = get_buffer_for_ptr(dhgate, (NSUInteger)T*FFD*4, &offDGate);
    id<MTLBuffer> bufDUp = get_buffer_for_ptr(dhup, (NSUInteger)T*FFD*4, &offDUp);

    id<MTLComputeCommandEncoder> encoder = [currentCmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:pso_swiglu_bwd];
    [encoder setBuffer:bufDHS offset:offDHS atIndex:0];
    [encoder setBuffer:bufGate offset:offGate atIndex:1];
    [encoder setBuffer:bufUp offset:offUp atIndex:2];
    [encoder setBuffer:bufDGate offset:offDGate atIndex:3];
    [encoder setBuffer:bufDUp offset:offDUp atIndex:4];
    
    MTLSize gridSize = MTLSizeMake(T * FFD, 1, 1);
    NSUInteger maxThreads = pso_swiglu_bwd.maxTotalThreadsPerThreadgroup;
    MTLSize threadgroupSize = MTLSizeMake(MIN((NSUInteger)(T*FFD), maxThreads), 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

void tc_metal_wkv_bwd(const float *dwkv, const float *k, const float *v, const float *r_sig, const float *decay, const float *state, float *dstate, float *drpre, float *dk, float *dv, float *ddecay, int T, int D) {
    if (!currentCmdBuf) currentCmdBuf = [commandQueue commandBuffer];
    
    NSUInteger offDwkv=0, offK=0, offV=0, offR=0, offDecay=0, offState=0, offDState=0, offDrpre=0, offDk=0, offDv=0, offDDecay=0;
    id<MTLBuffer> bufDwkv = get_buffer_for_ptr(dwkv, (NSUInteger)T*D*4, &offDwkv);
    id<MTLBuffer> bufK = get_buffer_for_ptr(k, (NSUInteger)T*D*4, &offK);
    id<MTLBuffer> bufV = get_buffer_for_ptr(v, (NSUInteger)T*D*4, &offV);
    id<MTLBuffer> bufR = get_buffer_for_ptr(r_sig, (NSUInteger)T*D*4, &offR);
    id<MTLBuffer> bufDecay = get_buffer_for_ptr(decay, (NSUInteger)D*4, &offDecay);
    id<MTLBuffer> bufState = get_buffer_for_ptr(state, (NSUInteger)T*D*4, &offState);
    id<MTLBuffer> bufDState = get_buffer_for_ptr(dstate, (NSUInteger)T*D*4, &offDState);
    id<MTLBuffer> bufDrpre = get_buffer_for_ptr(drpre, (NSUInteger)T*D*4, &offDrpre);
    id<MTLBuffer> bufDk = get_buffer_for_ptr(dk, (NSUInteger)T*D*4, &offDk);
    id<MTLBuffer> bufDv = get_buffer_for_ptr(dv, (NSUInteger)T*D*4, &offDv);
    id<MTLBuffer> bufDDecay = get_buffer_for_ptr(ddecay, (NSUInteger)D*4, &offDDecay);

    id<MTLComputeCommandEncoder> encoder = [currentCmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:pso_wkv_bwd];
    [encoder setBuffer:bufDwkv offset:offDwkv atIndex:0];
    [encoder setBuffer:bufK offset:offK atIndex:1];
    [encoder setBuffer:bufV offset:offV atIndex:2];
    [encoder setBuffer:bufR offset:offR atIndex:3];
    [encoder setBuffer:bufDecay offset:offDecay atIndex:4];
    [encoder setBuffer:bufState offset:offState atIndex:5];
    [encoder setBuffer:bufDState offset:offDState atIndex:6];
    [encoder setBuffer:bufDrpre offset:offDrpre atIndex:7];
    [encoder setBuffer:bufDk offset:offDk atIndex:8];
    [encoder setBuffer:bufDv offset:offDv atIndex:9];
    [encoder setBuffer:bufDDecay offset:offDDecay atIndex:10];
    [encoder setBytes:&T length:sizeof(int) atIndex:11];
    [encoder setBytes:&D length:sizeof(int) atIndex:12];
    MTLSize gridSize = MTLSizeMake(D, 1, 1); // wait, wkv_bwd uses gid.x as batch index. If batch=1, then it's 1. But threadgroup runs over D.
    // Actually, wkv_bwd in tc_shaders.metal:
    // int b = gid.x;
    // for (int i = tid; i < D; i += tgid) {
    // So gid.x is batch. Here we only have 1 sequence.
    NSUInteger maxThreads = pso_wkv_bwd.maxTotalThreadsPerThreadgroup;
    MTLSize threadgroupSize = MTLSizeMake(MIN((NSUInteger)D, maxThreads), 1, 1);
    gridSize = MTLSizeMake(1 * threadgroupSize.width, 1, 1); // batch=1
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

void tc_metal_attn_bwd(const float *dctx, const float *attn_w, const float *Q, const float *K, const float *V, float *dQ, float *dK, float *dV, int T, int D, int H, int n_kv_heads) {
    if (!currentCmdBuf) currentCmdBuf = [commandQueue commandBuffer];
    
    int kv_D = (D / H) * n_kv_heads;
    NSUInteger offDctx=0, offAttnW=0, offQ=0, offK=0, offV=0, offDQ=0, offDK=0, offDV=0;
    id<MTLBuffer> bufDctx = get_buffer_for_ptr(dctx, (NSUInteger)T*D*4, &offDctx);
    id<MTLBuffer> bufAttnW = get_buffer_for_ptr(attn_w, (NSUInteger)H*T*T*4, &offAttnW);
    id<MTLBuffer> bufQ = get_buffer_for_ptr(Q, (NSUInteger)T*D*4, &offQ);
    id<MTLBuffer> bufK = get_buffer_for_ptr(K, (NSUInteger)T*kv_D*4, &offK);
    id<MTLBuffer> bufV = get_buffer_for_ptr(V, (NSUInteger)T*kv_D*4, &offV);
    id<MTLBuffer> bufDQ = get_buffer_for_ptr(dQ, (NSUInteger)T*D*4, &offDQ);
    id<MTLBuffer> bufDK = get_buffer_for_ptr(dK, (NSUInteger)T*kv_D*4, &offDK);
    id<MTLBuffer> bufDV = get_buffer_for_ptr(dV, (NSUInteger)T*kv_D*4, &offDV);

    id<MTLComputeCommandEncoder> encoder = [currentCmdBuf computeCommandEncoder];
    
    [encoder setComputePipelineState:pso_attn_bwd];
    [encoder setBuffer:bufDctx offset:offDctx atIndex:0];
    [encoder setBuffer:bufAttnW offset:offAttnW atIndex:1];
    [encoder setBuffer:bufQ offset:offQ atIndex:2];
    [encoder setBuffer:bufK offset:offK atIndex:3];
    [encoder setBuffer:bufV offset:offV atIndex:4];
    [encoder setBuffer:bufDQ offset:offDQ atIndex:5];
    [encoder setBuffer:bufDK offset:offDK atIndex:6];
    [encoder setBuffer:bufDV offset:offDV atIndex:7];
    [encoder setBytes:&T length:sizeof(int) atIndex:8];
    [encoder setBytes:&D length:sizeof(int) atIndex:9];
    [encoder setBytes:&H length:sizeof(int) atIndex:10];
    [encoder setBytes:&n_kv_heads length:sizeof(int) atIndex:11];
    
    // In attn_bwd, gid is (t, h, b). Here b=1.
    // So grid should be T x H.
    // Threadgroup size is min(Dh, maxThreads). Total grid threads = T * threadgroupSize.width, H * threadgroupSize.height?
    // Wait, gid is threadgroup_position_in_grid. So we dispatch threadgroups! No, dispatchThreads takes total threads.
    int Dh = D / H;
    NSUInteger maxThreads = pso_attn_bwd.maxTotalThreadsPerThreadgroup;
    MTLSize threadgroupSize = MTLSizeMake(MIN((NSUInteger)Dh, maxThreads), 1, 1);
    MTLSize gridSize = MTLSizeMake(T * threadgroupSize.width, H * threadgroupSize.height, 1); // 1 is for batch
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    
    [encoder endEncoding];
}

void tc_metal_rope_fwd(const float *q, const float *k, float *q_out, float *k_out, int T, int D, int H, int n_kv_heads) {
    if (!currentCmdBuf) currentCmdBuf = [commandQueue commandBuffer];
    int kv_D = (D / H) * n_kv_heads;
    NSUInteger offQ=0, offK=0, offQout=0, offKout=0;
    id<MTLBuffer> bufQ = get_buffer_for_ptr(q, (NSUInteger)T*D*4, &offQ);
    id<MTLBuffer> bufK = get_buffer_for_ptr(k, (NSUInteger)T*kv_D*4, &offK);
    id<MTLBuffer> bufQout = get_buffer_for_ptr(q_out, (NSUInteger)T*D*4, &offQout);
    id<MTLBuffer> bufKout = get_buffer_for_ptr(k_out, (NSUInteger)T*kv_D*4, &offKout);

    id<MTLComputeCommandEncoder> encoder = [currentCmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:pso_rope_fwd];
    [encoder setBuffer:bufQ offset:offQ atIndex:0];
    [encoder setBuffer:bufK offset:offK atIndex:1];
    [encoder setBytes:&T length:sizeof(int) atIndex:2];
    [encoder setBytes:&D length:sizeof(int) atIndex:3];
    [encoder setBytes:&H length:sizeof(int) atIndex:4];
    [encoder setBytes:&n_kv_heads length:sizeof(int) atIndex:5];
    
    int Dh = D / H;
    NSUInteger maxThreads = pso_rope_fwd.maxTotalThreadsPerThreadgroup;
    MTLSize threadgroupSize = MTLSizeMake(MIN((NSUInteger)(Dh/2), maxThreads), 1, 1);
    MTLSize gridSize = MTLSizeMake(T * threadgroupSize.width, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

void tc_metal_rope_bwd(const float *dq, const float *dk, float *dq_out, float *dk_out, int T, int D, int H, int n_kv_heads) {
    if (!currentCmdBuf) currentCmdBuf = [commandQueue commandBuffer];
    int kv_D = (D / H) * n_kv_heads;
    NSUInteger offDQ=0, offDK=0, offDQout=0, offDKout=0;
    id<MTLBuffer> bufDQ = get_buffer_for_ptr(dq, (NSUInteger)T*D*4, &offDQ);
    id<MTLBuffer> bufDK = get_buffer_for_ptr(dk, (NSUInteger)T*kv_D*4, &offDK);
    id<MTLBuffer> bufDQout = get_buffer_for_ptr(dq_out, (NSUInteger)T*D*4, &offDQout);
    id<MTLBuffer> bufDKout = get_buffer_for_ptr(dk_out, (NSUInteger)T*kv_D*4, &offDKout);

    id<MTLComputeCommandEncoder> encoder = [currentCmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:pso_rope_bwd];
    [encoder setBuffer:bufDQ offset:offDQ atIndex:0];
    [encoder setBuffer:bufDK offset:offDK atIndex:1];
    [encoder setBytes:&T length:sizeof(int) atIndex:2];
    [encoder setBytes:&D length:sizeof(int) atIndex:3];
    [encoder setBytes:&H length:sizeof(int) atIndex:4];
    [encoder setBytes:&n_kv_heads length:sizeof(int) atIndex:5];
    
    int Dh = D / H;
    NSUInteger maxThreads = pso_rope_bwd.maxTotalThreadsPerThreadgroup;
    MTLSize threadgroupSize = MTLSizeMake(MIN((NSUInteger)(Dh/2), maxThreads), 1, 1);
    MTLSize gridSize = MTLSizeMake(T * threadgroupSize.width, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

void tc_metal_add_fwd(const float *a, const float *b, float *out, int T, int D) {
    if (!currentCmdBuf) currentCmdBuf = [commandQueue commandBuffer];
    NSUInteger offA=0, offB=0, offOut=0;
    id<MTLBuffer> bufA = get_buffer_for_ptr(a, (NSUInteger)T*D*4, &offA);
    id<MTLBuffer> bufB = get_buffer_for_ptr(b, (NSUInteger)T*D*4, &offB);
    id<MTLBuffer> bufOut = get_buffer_for_ptr(out, (NSUInteger)T*D*4, &offOut);

    id<MTLComputeCommandEncoder> encoder = [currentCmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:pso_add_fwd];
    [encoder setBuffer:bufA offset:offA atIndex:0];
    [encoder setBuffer:bufB offset:offB atIndex:1];
    [encoder setBuffer:bufOut offset:offOut atIndex:2];
    
    MTLSize gridSize = MTLSizeMake(T * D, 1, 1);
    NSUInteger maxThreads = pso_add_fwd.maxTotalThreadsPerThreadgroup;
    MTLSize threadgroupSize = MTLSizeMake(MIN((NSUInteger)(T*D), maxThreads), 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

void tc_metal_tmix_fwd(const float *x, const float *mix_w, float *out, int T, int D) {
    if (!currentCmdBuf) currentCmdBuf = [commandQueue commandBuffer];
    NSUInteger offX=0, offW=0, offOut=0;
    id<MTLBuffer> bufX = get_buffer_for_ptr(x, (NSUInteger)T*D*4, &offX);
    id<MTLBuffer> bufW = get_buffer_for_ptr(mix_w, (NSUInteger)D*4, &offW);
    id<MTLBuffer> bufOut = get_buffer_for_ptr(out, (NSUInteger)T*D*4, &offOut);

    id<MTLComputeCommandEncoder> encoder = [currentCmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:pso_tmix_fwd];
    [encoder setBuffer:bufX offset:offX atIndex:0];
    [encoder setBuffer:bufW offset:offW atIndex:1];
    [encoder setBuffer:bufOut offset:offOut atIndex:2];
    [encoder setBytes:&T length:sizeof(int) atIndex:3];
    [encoder setBytes:&D length:sizeof(int) atIndex:4];
    
    NSUInteger maxThreads = pso_tmix_fwd.maxTotalThreadsPerThreadgroup;
    MTLSize threadgroupSize = MTLSizeMake(MIN((NSUInteger)D, maxThreads), 1, 1);
    MTLSize gridSize = MTLSizeMake(T * threadgroupSize.width, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

void tc_metal_tmix_bwd(const float *dout, const float *x, const float *mix_w, float *dx, float *dmix_w, int T, int D) {
    if (!currentCmdBuf) currentCmdBuf = [commandQueue commandBuffer];
    NSUInteger offDout=0, offX=0, offW=0, offDx=0, offDmix=0;
    id<MTLBuffer> bufDout = get_buffer_for_ptr(dout, (NSUInteger)T*D*4, &offDout);
    id<MTLBuffer> bufX = get_buffer_for_ptr(x, (NSUInteger)T*D*4, &offX);
    id<MTLBuffer> bufW = get_buffer_for_ptr(mix_w, (NSUInteger)D*4, &offW);
    id<MTLBuffer> bufDx = get_buffer_for_ptr(dx, (NSUInteger)T*D*4, &offDx);
    id<MTLBuffer> bufDmix = dmix_w ? get_buffer_for_ptr(dmix_w, (NSUInteger)D*4, &offDmix) : nil;

    id<MTLComputeCommandEncoder> encoder = [currentCmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:pso_tmix_bwd];
    [encoder setBuffer:bufDout offset:offDout atIndex:0];
    [encoder setBuffer:bufX offset:offX atIndex:1];
    [encoder setBuffer:bufW offset:offW atIndex:2];
    [encoder setBuffer:bufDx offset:offDx atIndex:3];
    if (bufDmix) [encoder setBuffer:bufDmix offset:offDmix atIndex:4];
    [encoder setBytes:&T length:sizeof(int) atIndex:5];
    [encoder setBytes:&D length:sizeof(int) atIndex:6];
    
    NSUInteger maxThreads = pso_tmix_bwd.maxTotalThreadsPerThreadgroup;
    MTLSize threadgroupSize = MTLSizeMake(MIN((NSUInteger)D, maxThreads), 1, 1);
    MTLSize gridSize = MTLSizeMake(T * threadgroupSize.width, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

void tc_metal_embed_fwd(const float *embed, const int *ids, float *out, int T, int D, int V) {
    if (!currentCmdBuf) currentCmdBuf = [commandQueue commandBuffer];
    NSUInteger offEmb=0, offIds=0, offOut=0;
    id<MTLBuffer> bufEmb = get_buffer_for_ptr(embed, (NSUInteger)V*D*4, &offEmb);
    id<MTLBuffer> bufIds = get_buffer_for_ptr(ids, (NSUInteger)T*4, &offIds);
    id<MTLBuffer> bufOut = get_buffer_for_ptr(out, (NSUInteger)T*D*4, &offOut);

    id<MTLComputeCommandEncoder> encoder = [currentCmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:pso_embed_fwd];
    [encoder setBuffer:bufEmb offset:offEmb atIndex:0];
    [encoder setBuffer:bufIds offset:offIds atIndex:1];
    [encoder setBuffer:bufOut offset:offOut atIndex:2];
    [encoder setBytes:&T length:sizeof(int) atIndex:3];
    [encoder setBytes:&D length:sizeof(int) atIndex:4];
    
    NSUInteger maxThreads = pso_embed_fwd.maxTotalThreadsPerThreadgroup;
    MTLSize threadgroupSize = MTLSizeMake(MIN((NSUInteger)D, maxThreads), 1, 1);
    MTLSize gridSize = MTLSizeMake(T * threadgroupSize.width, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

void tc_metal_embed_bwd(const float *dout, const int *ids, float *dembed, int T, int D, int V) {
    if (!currentCmdBuf) currentCmdBuf = [commandQueue commandBuffer];
    NSUInteger offDout=0, offIds=0, offDemb=0;
    id<MTLBuffer> bufDout = get_buffer_for_ptr(dout, (NSUInteger)T*D*4, &offDout);
    id<MTLBuffer> bufIds = get_buffer_for_ptr(ids, (NSUInteger)T*4, &offIds);
    id<MTLBuffer> bufDemb = get_buffer_for_ptr(dembed, (NSUInteger)V*D*4, &offDemb);

    id<MTLComputeCommandEncoder> encoder = [currentCmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:pso_embed_bwd];
    [encoder setBuffer:bufDout offset:offDout atIndex:0];
    [encoder setBuffer:bufIds offset:offIds atIndex:1];
    [encoder setBuffer:bufDemb offset:offDemb atIndex:2];
    [encoder setBytes:&T length:sizeof(int) atIndex:3];
    [encoder setBytes:&D length:sizeof(int) atIndex:4];
    
    NSUInteger maxThreads = pso_embed_bwd.maxTotalThreadsPerThreadgroup;
    MTLSize threadgroupSize = MTLSizeMake(MIN((NSUInteger)D, maxThreads), 1, 1);
    MTLSize gridSize = MTLSizeMake(T * threadgroupSize.width, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}
