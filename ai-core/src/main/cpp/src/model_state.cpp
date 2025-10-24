#include "model_state.h"
#include "utils/logger.h"

#include <cstring>      // memcpy
#include <algorithm>
#include <sstream>
#include <vector>
#include <jni.h>

//////////////////////////////////////////////////////////////////////
// Basic helpers
//////////////////////////////////////////////////////////////////////

std::vector<llama_token> ModelState::tokenize(const std::string& text) const {
    if (!model) return {};

    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) return {};

    int32_t guess = static_cast<int32_t>(text.size() + 8);
    std::vector<llama_token> toks((size_t)guess);

    int32_t n = llama_tokenize(vocab, text.c_str(),
                               static_cast<int32_t>(text.size()),
                               toks.data(), static_cast<int32_t>(toks.size()),
                               true, true);
    if (n < 0) {
        // Error – allocate exact space and retry
        toks.resize((size_t)(-n));
        n = llama_tokenize(vocab, text.c_str(),
                           static_cast<int32_t>(text.size()),
                           toks.data(), static_cast<int32_t>(toks.size()),
                           true, true);
    }
    if (n < 0) {
        LOG_ERROR("ModelState::tokenize: tokenisation failed");
        return {};
    }
    toks.resize((size_t)n);
    return toks;
}

std::string ModelState::detokenize_single(llama_token t) const {
    if (!model) return {};
    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) return {};

    char buffer[512];
    int n = llama_token_to_piece(vocab, t, buffer, sizeof(buffer) - 1, 0, true);
    if (n < 0) {                               // buffer too small
        std::string out((size_t)(-n), '\0');
        llama_token_to_piece(vocab, t, out.data(), -n, 0, true);
        return out;
    }
    return std::string(buffer, buffer + n);
}

llama_token ModelState::space_token() const {
    if (!model) return 0;
    const llama_vocab* vocab = llama_model_get_vocab(model);
    llama_token out[4];
    int n = llama_tokenize(vocab, " ", 1, out, 4, true, true);
    return (n > 0) ? out[0] : 0;
}

// ------------------------------------------------------------------
// Resource management
// ------------------------------------------------------------------
void ModelState::release() {
    if (grammar_sampler) { llama_sampler_free(grammar_sampler); grammar_sampler = nullptr; }
    if (sampler) { llama_sampler_free(sampler); sampler = nullptr; }
    if (ctx) { llama_free(ctx); ctx = nullptr; }
    if (model) { llama_model_free(model); model = nullptr; }
    llama_backend_free();
    LOG_INFO("ModelState: all resources released");
}

// ------------------------------------------------------------------
// KV cache / sampler reset
// ------------------------------------------------------------------
void ModelState::prepare_for_generation() {
    if (!ctx) return;
    llama_memory_t mem = llama_get_memory(ctx);
    if (mem) llama_memory_clear(mem, true);   // wipe KV cache

    if (sampler) llama_sampler_reset(sampler);
}

// ------------------------------------------------------------------
// Sampler chain
// ------------------------------------------------------------------
void ModelState::rebuild_sampler(int top_k, float top_p, float temp, float min_p) {
    if (sampler) { llama_sampler_free(sampler); sampler = nullptr; }

    llama_sampler_chain_params p = llama_sampler_chain_default_params();
    llama_sampler* chain = llama_sampler_chain_init(p);

    // Add grammar sampler first if we have tools
    if (tools_enabled && grammar_sampler)
        llama_sampler_chain_add(chain, grammar_sampler);

    llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));

    if (top_p < 1.0f)
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));

    if (std::abs(temp - 1.0f) > 1e-3f)
        llama_sampler_chain_add(chain, llama_sampler_init_temp(temp));

    if (temp > 0.0f)
        llama_sampler_chain_add(chain, llama_sampler_init_dist(-1));

    if (min_p > 0.0f)
        llama_sampler_chain_add(chain, llama_sampler_init_min_p(min_p, 1));

    sampler = chain;
    llama_sampler_reset(sampler);
}

// ------------------------------------------------------------------
// Prompt decoding
// ------------------------------------------------------------------
bool ModelState::decode_prompt(const std::vector<llama_token>& toks) {
    if (!ctx || toks.empty()) return true;

    llama_batch batch = llama_batch_init(batch_size, 0, 1);
    int32_t pos = 0, idx = 0;
    while (idx < toks.size()) {
        int32_t take = std::min<int32_t>(batch_size,
                                         static_cast<int32_t>(toks.size() - idx));
        batch.n_tokens = take;
        for (int i = 0; i < take; ++i) {
            batch.token[i] = toks[idx + i];
            batch.pos[i]   = pos + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == take - 1);
        }
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERROR("ModelState::decode_prompt: llama_decode failed");
            llama_batch_free(batch);
            return false;
        }
        pos += take;
        idx += static_cast<size_t>(take);
    }
    llama_batch_free(batch);
    return true;
}

// ------------------------------------------------------------------
// Warm‑up to prime the model
// ------------------------------------------------------------------
void ModelState::warmup_context() {
    llama_token space = space_token();
    if (space == 0) return;

    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.n_tokens = 1;
    batch.token[0] = space;
    batch.pos[0]   = 0;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = true;

    llama_decode(ctx, batch);
    llama_batch_free(batch);
}

// ------------------------------------------------------------------
// State persistence
// ------------------------------------------------------------------
jlong ModelState::get_state_size() const { return llama_state_get_size(ctx); }

void* ModelState::get_state_data(void* buffer, size_t size) const {
    if (!ctx) return nullptr;
    return reinterpret_cast<void *>(llama_state_get_data(ctx, static_cast<uint8_t *>(buffer),
                                                         size));
}

bool ModelState::load_state_data(const void* data, size_t size) {
    if (!ctx) return false;
    size_t n = llama_state_set_data(ctx, static_cast<const uint8_t*>(data), size);
    return n == size;
}