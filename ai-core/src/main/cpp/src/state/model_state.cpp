#include "model_state.h"
#include "../utils/logger.h"

#include <cstring>
#include <algorithm>
#include <sstream>
#include <vector>
#include <jni.h>
#include <src/llama-sampling.h>
#include "llama.h"

//////////////////////////////////////////////////////////////////////
// Sampler rebuild
//////////////////////////////////////////////////////////////////////

void ModelState::rebuild_sampler(
        int topK,
        float topP,
        float temp,
        float minP,
        int mirostat,
        float mirostatTau,
        float mirostatEta,
        int seed) {

    LOG_INFO("=== rebuild_sampler START ===");
    LOG_INFO("Parameters: topK=%d, topP=%.2f, temp=%.2f, minP=%.2f", topK, topP, temp, minP);
    LOG_INFO("Mirostat: mode=%d, tau=%.2f, eta=%.2f, seed=%d", mirostat, mirostatTau, mirostatEta, seed);

    // Free existing sampler
    if (sampler) {
        LOG_INFO("Freeing existing sampler chain");
        llama_sampler_free(sampler);
        sampler = nullptr;
    }

    const llama_vocab *vocab = llama_model_get_vocab(model);
    if (!vocab) {
        LOG_ERROR("❌ Failed to get vocab for sampler rebuild");
        return;
    }
    LOG_INFO("✓ Vocab retrieved for sampler");

    // Initialize default chain parameters
    auto sparams = llama_sampler_chain_default_params();
    LOG_INFO("Creating sampler chain...");
    llama_sampler *chain = llama_sampler_chain_init(sparams);
    if (!chain) {
        LOG_ERROR("❌ Failed to create sampler chain");
        return;
    }
    LOG_INFO("✓ Sampler chain created");

    // Add grammar sampler first (if any tools exist)
    if (tools_enabled && grammar_sampler) {
        LOG_INFO("Adding grammar sampler to chain (tools enabled)");
        llama_sampler_chain_add(chain, grammar_sampler);
    } else {
        LOG_INFO("No grammar sampler (tools_enabled=%d, grammar_sampler=%p)",
                 static_cast<int>(tools_enabled.load()), grammar_sampler);
    }

    // --- Mirostat branch ---
    if (mirostat > 0) {
        LOG_INFO("Using Mirostat sampling (mode=%d)", mirostat);
        int n_vocab = llama_vocab_n_tokens(vocab);
        LOG_INFO("Vocab size: %d tokens", n_vocab);

        auto *mirostatSampler = llama_sampler_init_mirostat(
                n_vocab,
                seed,
                mirostatTau,
                mirostatEta,
                100 // m window
        );

        if (mirostatSampler) {
            llama_sampler_chain_add(chain, mirostatSampler);
            LOG_INFO("✓ Mirostat sampler added");
        } else {
            LOG_ERROR("❌ Failed to create Mirostat sampler");
        }
    }
        // --- Standard sampling branch ---
    else {
        LOG_INFO("Using standard sampling");

        LOG_INFO("Adding top-k sampler (k=%d)", topK);
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(topK));

        if (topP < 1.0f) {
            LOG_INFO("Adding top-p sampler (p=%.2f)", topP);
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(topP, 1));
        } else {
            LOG_INFO("Skipping top-p (p=%.2f >= 1.0)", topP);
        }

        if (std::abs(temp - 1.0f) > 1e-3f) {
            LOG_INFO("Adding temperature sampler (temp=%.2f)", temp);
            llama_sampler_chain_add(chain, llama_sampler_init_temp(temp));
        } else {
            LOG_INFO("Skipping temperature (temp=%.2f ~= 1.0)", temp);
        }

        if (temp > 0.0f) {
            LOG_INFO("Adding distribution sampler");
            llama_sampler_chain_add(chain, llama_sampler_init_dist(-1));
        }

        if (minP > 0.0f) {
            LOG_INFO("Adding min-p sampler (minP=%.2f)", minP);
            llama_sampler_chain_add(chain, llama_sampler_init_min_p(minP, 1));
        } else {
            LOG_INFO("Skipping min-p (minP=%.2f <= 0.0)", minP);
        }
    }

    sampler = chain;
    LOG_INFO("Resetting sampler state...");
    llama_sampler_reset(sampler);

    LOG_INFO("✓ Sampler rebuilt successfully");
    LOG_INFO("=== rebuild_sampler END ===");
}

//////////////////////////////////////////////////////////////////////
// Tokenization
//////////////////////////////////////////////////////////////////////

std::vector<llama_token> ModelState::tokenize(const std::string &text) const {
    LOG_INFO("=== tokenize START ===");
    LOG_INFO("Input text length: %zu bytes", text.size());

    if (!model) {
        LOG_ERROR("❌ Model is null");
        return {};
    }

    const llama_vocab *vocab = llama_model_get_vocab(model);
    if (!vocab) {
        LOG_ERROR("❌ Vocab is null");
        return {};
    }
    LOG_INFO("✓ Model and vocab ready");

    int32_t guess = static_cast<int32_t>(text.size() + 8);
    LOG_INFO("Initial token buffer size: %d", guess);
    std::vector<llama_token> toks(static_cast<size_t>(guess));

    LOG_INFO("Calling llama_tokenize (first attempt)...");
    int32_t n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                               toks.data(), static_cast<int32_t>(toks.size()), true, true);

    if (n < 0) {
        LOG_WARN("First attempt failed (needed %d tokens), retrying with exact size", -n);
        toks.resize(static_cast<size_t>(-n));
        n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                           toks.data(), static_cast<int32_t>(toks.size()), true, true);
    }

    if (n < 0) {
        LOG_ERROR("❌ Tokenization failed after retry");
        return {};
    }

    toks.resize(static_cast<size_t>(n));
    LOG_INFO("✓ Tokenization successful: %d tokens", n);
    LOG_INFO("=== tokenize END ===");
    return toks;
}

//////////////////////////////////////////////////////////////////////
// Detokenization
//////////////////////////////////////////////////////////////////////

std::string ModelState::detokenize_single(llama_token t) const {
    if (!model) {
        LOG_ERROR("detokenize_single: model is null");
        return {};
    }

    const llama_vocab *vocab = llama_model_get_vocab(model);
    if (!vocab) {
        LOG_ERROR("detokenize_single: vocab is null");
        return {};
    }

    char buffer[512];
    int n = llama_token_to_piece(vocab, t, buffer, sizeof(buffer) - 1, 0, false);

    if (n < 0) {
        LOG_INFO("Token %d needs larger buffer (%d bytes), retrying", static_cast<int>(t), -n);
        std::string out(static_cast<size_t>(-n), '\0');
        n = llama_token_to_piece(vocab, t, out.data(), -n, 0, false);
        if (n < 0) {
            LOG_ERROR("❌ Failed to detokenize token %d", static_cast<int>(t));
            return {};
        }
        return out;
    }

    return std::string(buffer, static_cast<size_t>(n));
}

std::string ModelState::detokenize_buffered(llama_token t) {
    // Get raw token bytes
    std::string piece = detokenize_single(t);
    if (piece.empty()) {
        LOG_WARN("detokenize_buffered: empty piece for token %d", static_cast<int>(t));
        return {};
    }

    LOG_INFO("Token %d -> '%s' (%zu bytes)", static_cast<int>(t), piece.c_str(), piece.size());

    // Add to carry buffer
    utf8_carry_buffer += piece;
    LOG_INFO("Carry buffer size: %zu bytes", utf8_carry_buffer.size());

    // Extract complete UTF-8 characters
    std::string complete_chars;
    size_t i = 0;

    while (i < utf8_carry_buffer.size()) {
        auto c = static_cast<unsigned char>(utf8_carry_buffer[i]);
        size_t char_len = 0;

        // Determine UTF-8 character length
        if ((c & 0x80) == 0x00) {
            char_len = 1; // ASCII (0xxxxxxx)
        } else if ((c & 0xE0) == 0xC0) {
            char_len = 2; // 2-byte (110xxxxx)
        } else if ((c & 0xF0) == 0xE0) {
            char_len = 3; // 3-byte (1110xxxx)
        } else if ((c & 0xF8) == 0xF0) {
            char_len = 4; // 4-byte (11110xxx) - EMOJIS!
        } else {
            // Invalid UTF-8 start byte - skip it
            LOG_WARN("Invalid UTF-8 start byte: 0x%02X at position %zu", c, i);
            i++;
            continue;
        }

        LOG_INFO("UTF-8 char at pos %zu: len=%zu, start=0x%02X", i, char_len, c);

        // Check if we have enough bytes for complete character
        if (i + char_len > utf8_carry_buffer.size()) {
            LOG_INFO("Incomplete UTF-8 char (need %zu, have %zu), keeping in buffer",
                     char_len, utf8_carry_buffer.size() - i);
            break;
        }

        // Validate continuation bytes
        bool valid = true;
        for (size_t j = 1; j < char_len; ++j) {
            auto cont = static_cast<unsigned char>(utf8_carry_buffer[i + j]);
            if ((cont & 0xC0) != 0x80) { // Must be 10xxxxxx
                valid = false;
                LOG_WARN("Invalid UTF-8 continuation byte at offset %zu: 0x%02X", j, cont);
                break;
            }
        }

        if (valid) {
            // Complete valid UTF-8 character
            std::string utf8_char = utf8_carry_buffer.substr(i, char_len);
            LOG_INFO("✓ Complete UTF-8 char: '%s' (%zu bytes)", utf8_char.c_str(), char_len);
            complete_chars.append(utf8_char);
            i += char_len;
        } else {
            // Invalid sequence - skip the start byte
            LOG_WARN("Skipping invalid UTF-8 sequence starting at %zu", i);
            i++;
        }
    }

    // Remove processed characters from buffer
    utf8_carry_buffer = utf8_carry_buffer.substr(i);
    LOG_INFO("Remaining in buffer: %zu bytes, returning: %zu bytes ('%s')",
             utf8_carry_buffer.size(), complete_chars.size(), complete_chars.c_str());

    return complete_chars;
}

std::string ModelState::flush_utf8_buffer() {
    LOG_INFO("=== flush_utf8_buffer ===");
    std::string remaining = utf8_carry_buffer;
    utf8_carry_buffer.clear();

    if (!remaining.empty()) {
        LOG_WARN("⚠️ Flushing incomplete UTF-8 sequence: %zu bytes", remaining.size());
        for (size_t i = 0; i < remaining.size(); ++i) {
            LOG_WARN("  [%zu] = 0x%02X", i, static_cast<unsigned char>(remaining[i]));
        }
    } else {
        LOG_INFO("Buffer was empty (normal)");
    }

    return remaining;
}

//////////////////////////////////////////////////////////////////////
// Helper functions
//////////////////////////////////////////////////////////////////////

llama_token ModelState::space_token() const {
    if (!model) return 0;
    const llama_vocab *vocab = llama_model_get_vocab(model);
    llama_token out[4];
    int n = llama_tokenize(vocab, " ", 1, out, 4, true, true);
    llama_token space = (n > 0) ? out[0] : 0;
    LOG_INFO("Space token: %d", static_cast<int>(space));
    return space;
}

//////////////////////////////////////////////////////////////////////
// Resource management
//////////////////////////////////////////////////////////////////////

void ModelState::release() {
    LOG_INFO("=== ModelState::release START ===");

    if (grammar_sampler) {
        LOG_INFO("Freeing grammar sampler");
        llama_sampler_free(grammar_sampler);
        grammar_sampler = nullptr;
    }

    if (sampler) {
        LOG_INFO("Freeing sampler");
        llama_sampler_free(sampler);
        sampler = nullptr;
    }

    if (ctx) {
        LOG_INFO("Freeing context");
        llama_free(ctx);
        ctx = nullptr;
    }

    if (model) {
        LOG_INFO("Freeing model");
        llama_model_free(model);
        model = nullptr;
    }

    utf8_carry_buffer.clear();
    LOG_INFO("UTF-8 buffer cleared");

    llama_backend_free();
    LOG_INFO("✓ ModelState: all resources released");
    LOG_INFO("=== ModelState::release END ===");
}

//////////////////////////////////////////////////////////////////////
// Generation preparation
//////////////////////////////////////////////////////////////////////

void ModelState::prepare_for_generation() {
    LOG_INFO("=== prepare_for_generation START ===");

    if (!ctx) {
        LOG_WARN("Context is null, cannot prepare");
        return;
    }

    LOG_INFO("Clearing KV cache...");
    llama_memory_t mem = llama_get_memory(ctx);
    if (mem) {
        llama_memory_clear(mem, true);
        LOG_INFO("✓ KV cache cleared");
    } else {
        LOG_WARN("Failed to get memory handle");
    }

    if (sampler) {
        LOG_INFO("Resetting sampler state...");
        llama_sampler_reset(sampler);
        LOG_INFO("✓ Sampler reset");
    } else {
        LOG_WARN("Sampler is null");
    }

    LOG_INFO("Clearing UTF-8 buffer...");
    utf8_carry_buffer.clear();

    LOG_INFO("=== prepare_for_generation END ===");
}

//////////////////////////////////////////////////////////////////////
// Prompt decoding
//////////////////////////////////////////////////////////////////////

bool ModelState::decode_prompt(const std::vector<llama_token> &toks) const {
    LOG_INFO("=== decode_prompt START ===");
    LOG_INFO("Tokens to decode: %zu", toks.size());

    if (!ctx) {
        LOG_ERROR("❌ Context is null");
        return false;
    }

    if (toks.empty()) {
        LOG_WARN("Empty token vector, nothing to decode");
        return true;
    }

    LOG_INFO("Creating batch (batch_size=%d)", batch_size);
    llama_batch batch = llama_batch_init(batch_size, 0, 1);
    if (!batch.token) {
        LOG_ERROR("❌ Failed to allocate batch");
        return false;
    }
    LOG_INFO("✓ Batch created");

    int32_t pos = 0, idx = 0;
    int decode_count = 0;

    while (idx < static_cast<int32_t>(toks.size())) {
        int32_t take = std::min<int32_t>(batch_size, static_cast<int32_t>(toks.size()) - idx);
        LOG_INFO("Decode iteration %d: processing %d tokens (pos=%d, idx=%d)",
                 decode_count, take, pos, idx);

        batch.n_tokens = take;
        for (int i = 0; i < take; ++i) {
            batch.token[i] = toks[static_cast<size_t>(idx + i)];
            batch.pos[i] = pos + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == take - 1); // Only last token needs logits
        }

        LOG_INFO("Calling llama_decode for %d tokens...", take);
        int result = llama_decode(ctx, batch);
        if (result != 0) {
            LOG_ERROR("❌ llama_decode failed with code: %d (pos=%d, count=%d)",
                      result, pos, take);
            llama_batch_free(batch);
            return false;
        }
        LOG_INFO("✓ llama_decode successful");

        pos += take;
        idx += take;
        decode_count++;
    }

    llama_batch_free(batch);
    LOG_INFO("✓ decode_prompt completed: %d decode calls, %d tokens total",
             decode_count, static_cast<int>(toks.size()));
    LOG_INFO("=== decode_prompt END ===");
    return true;
}

//////////////////////////////////////////////////////////////////////
// Warm-up
//////////////////////////////////////////////////////////////////////

void ModelState::warmup_context() const {
    LOG_INFO("=== warmup_context START ===");

    llama_token space = space_token();
    if (space == 0) {
        LOG_WARN("Space token is 0, skipping warmup");
        return;
    }
    LOG_INFO("Using space token %d for warmup", static_cast<int>(space));

    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.n_tokens = 1;
    batch.token[0] = space;
    batch.pos[0] = 0;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = true;

    LOG_INFO("Decoding warmup token...");
    int result = llama_decode(ctx, batch);
    if (result != 0) {
        LOG_WARN("Warmup decode failed with code: %d", result);
    } else {
        LOG_INFO("✓ Warmup successful");
    }

    llama_batch_free(batch);
    LOG_INFO("=== warmup_context END ===");
}

//////////////////////////////////////////////////////////////////////
// State persistence
//////////////////////////////////////////////////////////////////////

jlong ModelState::get_state_size() const {
    if (!ctx) {
        LOG_WARN("get_state_size: context is null");
        return 0;
    }
    jlong size = llama_state_get_size(ctx);
    LOG_INFO("State size: %lld bytes", size);
    return size;
}

void *ModelState::get_state_data(void *buffer, size_t size) const {
    LOG_INFO("get_state_data: copying %zu bytes", size);
    if (!ctx) {
        LOG_ERROR("Context is null");
        return nullptr;
    }
    return reinterpret_cast<void *>(llama_state_get_data(ctx, static_cast<uint8_t *>(buffer), size));
}

bool ModelState::load_state_data(const void *data, size_t size) const {
    LOG_INFO("load_state_data: loading %zu bytes", size);
    if (!ctx) {
        LOG_ERROR("Context is null");
        return false;
    }
    size_t n = llama_state_set_data(ctx, static_cast<const uint8_t *>(data), size);
    bool success = (n == size);
    LOG_INFO("load_state_data: %s (loaded %zu / %zu bytes)",
             success ? "SUCCESS" : "FAILED", n, size);
    return success;
}