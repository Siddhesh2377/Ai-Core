/*=============================================================
 *   model_state.h
 *=============================================================
 *
 *  RAII wrapper that owns:
 *    - `llama_model*`
 *    - `llama_context*`
 *    - `llama_sampler*` (chain, optional grammar sampler)
 *
 *  Public interface is intentionally tiny – only call
 *  `release()`, `is_ready()`, `prepare_for_generation()`,
 *  `rebuild_sampler()`, `tokenize()`, `detokenize_single()`,
 *  `decode_prompt()`, `warmup_context()`, `state_*()` functions.
 *
 *  The implementation lives in model_state.cpp.
 *============================================================*/

#pragma once

#include <string>
#include <vector>
#include <atomic>
#include <jni.h>
#include "llama.h"

class ToolCallState; // forward

struct ModelState {
    // ------------------------------------------------------------------
    // Llama objects
    // ------------------------------------------------------------------
    llama_model*      model = nullptr;
    llama_context*    ctx   = nullptr;
    llama_sampler*    sampler        = nullptr;
    llama_sampler*    grammar_sampler = nullptr; // used when tools enabled

    // ------------------------------------------------------------------
    // Runtime parameters
    // ------------------------------------------------------------------
    std::string system_prompt = "You are a helpful assistant.";
    std::string chat_template_override;
    std::string tools_json;
    bool tools_enabled = false;

    int ctx_size = 2048;
    int batch_size = 512;

    // ------------------------------------------------------------------
    // Per‑token handling helpers
    // ------------------------------------------------------------------
    std::string utf8_carry{}; // focus on the current stream's carry

    // ------------------------------------------------------------------
    // Methods
    // ------------------------------------------------------------------
    void release();                                  // free all objects
    bool is_ready() const { return model && ctx; }

    void prepare_for_generation();                    // clear KV & reset sampler
    void rebuild_sampler(int top_k, float top_p,
                         float temp, float min_p);    // create a new sampler chain

    // Tokenisation helpers
    std::vector<llama_token> tokenize(const std::string& text) const;
    std::string detokenize_single(llama_token t) const;
    llama_token space_token() const;                  // token for a single space

    // Prompt handling
    bool decode_prompt(const std::vector<llama_token>& toks);

    // Warm‑up decoding (single space)
    void warmup_context();

    // State persistence helpers (thin wrappers)
    jlong  get_state_size() const;
    void*  get_state_data(void* buffer, size_t size) const;
    bool   load_state_data(const void* data, size_t size);
};