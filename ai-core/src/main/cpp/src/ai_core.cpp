/*=============================================================
 *   ai_core.cpp - WITH COMPREHENSIVE LOGGING
 *=============================================================*/

#include "state/model_state.h"
#include "utils/jni_utils.h"
#include "utils/utf8_utils.h"
#include "chat/chat_template.h"

#include "llama.h"
#include "ggml-backend.h"
#include "cpu/cpu_helper.h"
#include "utils/logger.h"
#include "state/global_state.h"
#include "tool_calling/tool_call_state.h"
#include <sstream>
#include <algorithm>

#include <jni.h>
#include <string>
#include <mutex>

/*  --------------------------------------------------------------
 *      Global state and guard
 *  -------------------------------------------------------------- */
static std::mutex g_init_mtx;
static std::atomic<bool> g_stop_requested{false};

/*  --------------------------------------------------------------
 *      Helper – build & init grammar when tools enabled
 *  -------------------------------------------------------------- */
static void maybe_init_grammar() {
    LOG_INFO("maybe_init_grammar: tools_enabled=%d", static_cast<int>(g_state.tools_enabled));
    if (!g_state.tools_enabled) {
        LOG_INFO("Tools disabled, skipping grammar init");
        return;
    }

    LOG_INFO("Initializing tool-call grammar");
    const std::string grammar = chat::build_tool_grammar(g_state.tools_json);
    LOG_INFO("Grammar built, length=%zu", grammar.size());

    if (!grammar.empty()) {
        if (g_state.grammar_sampler) {
            LOG_INFO("Freeing existing grammar sampler");
            llama_sampler_free(g_state.grammar_sampler);
        }

        const llama_vocab* vocab = llama_model_get_vocab(g_state.model);
        if (!vocab) {
            LOG_ERROR("Failed to get vocab for grammar");
            g_state.tools_enabled = false;
            return;
        }

        g_state.grammar_sampler = llama_sampler_init_grammar(vocab, grammar.c_str(), "root");
        if (!g_state.grammar_sampler) {
            LOG_ERROR("Tool grammar initialization failed");
            g_state.tools_enabled = false;
        } else {
            LOG_INFO("Grammar sampler initialized successfully");
        }
    } else {
        LOG_WARN("Empty grammar string generated");
    }
}

/*  --------------------------------------------------------------
 *      JNI: load model & init context
 *  -------------------------------------------------------------- */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1core_NativeLib_nativeInit(JNIEnv* env, jobject,
                                          jstring jpath,
                                          jint jthreads,
                                          jint ctxSize,
                                          jfloat temp,
                                          jint topK,
                                          jfloat topP,
                                          jfloat minP,
                                          jint mirostat,
                                          jfloat mirostatTau,
                                          jfloat mirostatEta,
                                          jint seed) {
    LOG_INFO("=== nativeInit START ===");
    std::lock_guard<std::mutex> lk(g_init_mtx);

    const std::string path = utf8::from_jstring(env, jpath);
    LOG_INFO("Model path: %s", path.c_str());
    LOG_INFO("Parameters: threads=%d, ctx=%d, temp=%.2f, topK=%d, topP=%.2f, minP=%.2f",
             static_cast<int>(jthreads), static_cast<int>(ctxSize), temp,
             static_cast<int>(topK), topP, minP);
    LOG_INFO("Mirostat: mode=%d, tau=%.2f, eta=%.2f, seed=%d",
             static_cast<int>(mirostat), mirostatTau, mirostatEta, static_cast<int>(seed));

    LOG_INFO("Releasing old state...");
    g_state.release();

    LOG_INFO("Initializing llama backend...");
    llama_backend_init();

    int phys = count_physical_cores();
    int nthreads = (jthreads > 0) ? static_cast<int>(jthreads) : phys;
    LOG_INFO("Physical cores detected: %d, using threads: %d", phys, nthreads);

    LOG_INFO("Setting up model params...");
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_mmap = true;
    mparams.use_mlock = false;
    mparams.check_tensors = true;
    LOG_INFO("Model params: gpu_layers=%d, mmap=%d, mlock=%d, check_tensors=%d",
             mparams.n_gpu_layers, mparams.use_mmap, mparams.use_mlock, mparams.check_tensors);

    LOG_INFO("Loading model from file...");
    g_state.model = llama_model_load_from_file(path.c_str(), mparams);
    if (!g_state.model) {
        LOG_ERROR("FATAL: Failed to load model from '%s'", path.c_str());
        g_state.release();
        return JNI_FALSE;
    }
    LOG_INFO("✓ Model loaded successfully");

    LOG_INFO("Setting up context params...");
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = ctxSize;
    cparams.n_batch = 512;
    cparams.n_ubatch = 256;
    cparams.n_threads = nthreads;
    cparams.n_threads_batch = nthreads;
    cparams.offload_kqv = false;
    cparams.n_seq_max = 1;
    cparams.no_perf = false;
    LOG_INFO("Context params: n_ctx=%d, n_batch=%d, n_ubatch=%d, n_threads=%d, n_seq_max=%d",
             cparams.n_ctx, cparams.n_batch, cparams.n_ubatch, cparams.n_threads, cparams.n_seq_max);

    LOG_INFO("Creating context from model...");
    g_state.ctx = llama_init_from_model(g_state.model, cparams);
    if (!g_state.ctx) {
        LOG_ERROR("FATAL: Failed to create context");
        g_state.release();
        return JNI_FALSE;
    }
    LOG_INFO("✓ Context created successfully");

    g_state.ctx_size = ctxSize;
    g_state.batch_size = cparams.n_batch;
    LOG_INFO("State configured: ctx_size=%d, batch_size=%d", g_state.ctx_size, g_state.batch_size);

    LOG_INFO("Building sampler chain...");
    g_state.rebuild_sampler(static_cast<int>(topK),
                            topP,
                            temp,
                            minP,
                            mirostat,
                            mirostatTau,
                            mirostatEta,
                            seed);
    LOG_INFO("✓ Sampler chain built");

    LOG_INFO("Warming up context...");
    g_state.warmup_context();
    LOG_INFO("✓ Context warmed up");

    LOG_INFO("Initializing grammar (if needed)...");
    maybe_init_grammar();

    LOG_INFO("=== nativeInit COMPLETED SUCCESSFULLY ===");
    return JNI_TRUE;
}

/*  --------------------------------------------------------------
 *      JNI: release resources
 *  -------------------------------------------------------------- */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1core_NativeLib_nativeRelease(JNIEnv*, jobject) {
    LOG_INFO("=== nativeRelease called ===");
    std::lock_guard<std::mutex> lk(g_init_mtx);
    g_state.release();
    LOG_INFO("=== nativeRelease completed ===");
    return JNI_TRUE;
}

/*  --------------------------------------------------------------
 *      JNI: configuration setters
 *  -------------------------------------------------------------- */
extern "C" JNIEXPORT void JNICALL
Java_com_mp_ai_1core_NativeLib_nativeSetSystemPrompt(JNIEnv* env, jobject,
                                                     jstring jprompt) {
    g_state.system_prompt = utf8::from_jstring(env, jprompt);
    LOG_INFO("System prompt updated: %zu bytes, first 100 chars: '%.100s'",
             g_state.system_prompt.size(), g_state.system_prompt.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_mp_ai_1core_NativeLib_nativeSetChatTemplate(JNIEnv* env, jobject,
                                                     jstring jtemplate) {
    g_state.chat_template_override = utf8::from_jstring(env, jtemplate);
    LOG_INFO("Chat template override set: %zu bytes, first 100 chars: '%.100s'",
             g_state.chat_template_override.size(), g_state.chat_template_override.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_mp_ai_1core_NativeLib_nativeSetToolsJson(JNIEnv* env, jobject,
                                                  jstring jtools) {
    g_state.tools_json = utf8::from_jstring(env, jtools);
    g_state.tools_enabled = !g_state.tools_json.empty();
    LOG_INFO("Tools JSON set: %zu bytes, enabled=%d, content: '%.200s'",
             g_state.tools_json.size(), static_cast<int>(g_state.tools_enabled),
             g_state.tools_json.c_str());
    maybe_init_grammar();
}

/*  --------------------------------------------------------------
 *      JNI: request stop
 *  -------------------------------------------------------------- */
extern "C" JNIEXPORT void JNICALL
Java_com_mp_ai_1core_NativeLib_nativeStopGeneration(JNIEnv*, jobject) {
    g_stop_requested.store(true);
    LOG_INFO("⚠️ Stop generation requested");
}

/*  --------------------------------------------------------------
 *      JNI: clear KV cache (fast reset)
 *  -------------------------------------------------------------- */
extern "C" JNIEXPORT void JNICALL
Java_com_mp_ai_1core_NativeLib_nativeClearMemory(JNIEnv*, jobject) {
    LOG_INFO("Clearing KV cache...");
    if (g_state.ctx) {
        llama_memory_t mem = llama_get_memory(g_state.ctx);
        if (mem) {
            llama_memory_clear(mem, true);
            LOG_INFO("✓ KV cache cleared");
        } else {
            LOG_WARN("Failed to get memory handle");
        }
    } else {
        LOG_WARN("Context is null, cannot clear memory");
    }
}

/*  --------------------------------------------------------------
 *      JNI: streamable generation
 *  -------------------------------------------------------------- */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1core_NativeLib_nativeGenerateStream(JNIEnv* env, jobject,
                                                    jstring jprompt,
                                                    jint max_tokens,
                                                    jobject jcallback) {
    LOG_INFO("════════════════════════════════════════════════════");
    LOG_INFO("=== nativeGenerateStream START ===");
    LOG_INFO("════════════════════════════════════════════════════");
    LOG_INFO("Requested max_tokens: %d", static_cast<int>(max_tokens));

    if (!g_state.is_ready()) {
        LOG_ERROR("❌ Model not ready: model=%p, ctx=%p", g_state.model, g_state.ctx);
        jni::on_error(env, jcallback, "Model not initialized");
        return JNI_FALSE;
    }
    LOG_INFO("✓ Model state ready check: PASSED");

    LOG_INFO("Preparing for generation...");
    g_state.prepare_for_generation();
    g_stop_requested.store(false);
    LOG_INFO("✓ State prepared, stop flag reset");

    const std::string user_msg = utf8::from_jstring(env, jprompt);
    LOG_INFO("User message (%zu bytes): '%s'", user_msg.size(), user_msg.c_str());

    const llama_vocab* vocab = llama_model_get_vocab(g_state.model);
    if (!vocab) {
        LOG_ERROR("❌ Failed to get vocab from model");
        jni::on_error(env, jcallback, "Failed to get vocab");
        return JNI_FALSE;
    }
    LOG_INFO("✓ Vocab retrieved");

    // Build the final prompt + template
    std::string system = g_state.system_prompt;
    LOG_INFO("System prompt: %zu bytes", system.size());

    if (g_state.tools_enabled) {
        LOG_INFO("Tools enabled, building preamble...");
        std::string preamble = chat::build_tool_preamble(g_state.tools_json);
        LOG_INFO("Tool preamble: %zu bytes", preamble.size());
        system += "\n" + preamble;
        LOG_INFO("System + preamble: %zu bytes", system.size());
    }

    LOG_INFO("Applying chat template...");
    const std::string prompt = chat::apply_template(g_state.model,
                                                    system,
                                                    user_msg,
                                                    g_state.chat_template_override,
                                                    true);

    LOG_INFO("✓ Final prompt size: %zu bytes", prompt.size());
    LOG_INFO("First 300 chars: '%.300s'", prompt.c_str());
    if (prompt.size() > 300) {
        LOG_INFO("Last 100 chars: '%.100s'", prompt.c_str() + prompt.size() - 100);
    }

    // Tokenise prompt
    LOG_INFO("Tokenizing prompt...");
    std::vector<llama_token> prompt_toks = g_state.tokenize(prompt);
    if (prompt_toks.empty()) {
        LOG_ERROR("❌ Tokenization returned empty vector");
        jni::on_error(env, jcallback, "Tokenisation failed");
        return JNI_FALSE;
    }
    LOG_INFO("✓ Tokenization successful: %zu tokens", prompt_toks.size());
    LOG_INFO("First 10 tokens: ", "");
    for (size_t i = 0; i < std::min<size_t>(10, prompt_toks.size()); ++i) {
        LOG_INFO("  [%zu] = %d", i, static_cast<int>(prompt_toks[i]));
    }

    // Limit generation by context
    int32_t available = g_state.ctx_size - static_cast<int32_t>(prompt_toks.size()) - 8;
    LOG_INFO("Context calculation:");
    LOG_INFO("  ctx_size = %d", g_state.ctx_size);
    LOG_INFO("  prompt_tokens = %zu", prompt_toks.size());
    LOG_INFO("  reserved = 8");
    LOG_INFO("  available = %d", available);

    if (available <= 0) {
        LOG_ERROR("❌ Context overflow: available=%d", available);
        jni::on_error(env, jcallback, "Context overflow – shorten your prompt");
        return JNI_TRUE;
    }

    auto to_generate = static_cast<int32_t>(max_tokens > 0 ? max_tokens : 128);
    to_generate = std::min(to_generate, available);
    LOG_INFO("✓ Will generate: %d tokens (requested=%d, available=%d)",
             to_generate, static_cast<int>(max_tokens), available);

    // Feed prompt first
    LOG_INFO("──────────────────────────────────────────────────");
    LOG_INFO("Starting prompt decode (%zu tokens)...", prompt_toks.size());
    if (!g_state.decode_prompt(prompt_toks)) {
        LOG_ERROR("❌ decode_prompt() returned false");
        jni::on_error(env, jcallback, "Decoding prompt failed");
        return JNI_TRUE;
    }
    LOG_INFO("✓ Prompt decode completed successfully");
    LOG_INFO("──────────────────────────────────────────────────");

    /* ---------------------------------------------------------
     *  Streaming loop – one token at a time
     * -------------------------------------------------------- */
    LOG_INFO("════════════════════════════════════════════════════");
    LOG_INFO("=== STARTING TOKEN GENERATION LOOP ===");
    LOG_INFO("════════════════════════════════════════════════════");

    ToolCallState tool_state;
    llama_token eos = llama_vocab_eos(vocab);
    llama_token eot = llama_vocab_eot(vocab);
    LOG_INFO("Special tokens: EOS=%d, EOT=%d", static_cast<int>(eos), static_cast<int>(eot));

    llama_batch single = llama_batch_init(1, 0, 1);
    LOG_INFO("Batch initialized for streaming (size=1)");

    g_state.utf8_carry_buffer.clear();
    LOG_INFO("UTF-8 carry buffer cleared");

    int tokens_generated = 0;
    for (int i = 0; i < to_generate && !g_stop_requested.load(); ++i) {
        if (i % 5 == 0) {
            LOG_INFO("─── Generation step %d/%d ───", i, to_generate);
        }

        // Sample & accept
        LOG_INFO("[%d] Sampling token...", i);
        llama_token tok = llama_sampler_sample(g_state.sampler, g_state.ctx, -1);
        LOG_INFO("[%d] ✓ Sampled token: %d", i, static_cast<int>(tok));

        llama_sampler_accept(g_state.sampler, tok);
        LOG_INFO("[%d] ✓ Token accepted by sampler", i);

        // Turn EOS into space if first token
        if (i == 0 && (tok == eos || tok == eot)) {
            LOG_INFO("[%d] ⚠️ First token is EOS/EOT (%d), converting to space", i, static_cast<int>(tok));
            tok = g_state.space_token();
            LOG_INFO("[%d] Converted to space token: %d", i, static_cast<int>(tok));
        }

        if (tok == eos || tok == eot) {
            LOG_INFO("[%d] 🛑 Generated EOS/EOT token (%d) - STOPPING", i, static_cast<int>(tok));
            break;
        }

        // Use buffered detokenization
        LOG_INFO("[%d] Detokenizing token %d...", i, static_cast<int>(tok));
        std::string complete_chars = g_state.detokenize_buffered(tok);
        LOG_INFO("[%d] Detokenized: '%s' (len=%zu, buffer_size=%zu)",
                 i, complete_chars.c_str(), complete_chars.size(), g_state.utf8_carry_buffer.size());

        // Only process if we got complete UTF-8 characters
        if (!complete_chars.empty()) {
            LOG_INFO("[%d] Processing %zu complete UTF-8 chars", i, complete_chars.size());

            // Tool-call detection
            bool complete = false;
            if (g_state.tools_enabled) {
                LOG_INFO("[%d] Checking for tool patterns...", i);
                complete = tool_state.accumulate(complete_chars);
                if (complete) {
                    LOG_INFO("[%d] ✓ Tool call pattern completed!", i);
                    std::string name, payload;
                    if (tool_state.extract_tool_call(name, payload)) {
                        LOG_INFO("[%d] Tool call extracted: name='%s', payload='%s'",
                                 i, name.c_str(), payload.c_str());
                        jni::on_toolcall(env, jcallback, name, payload);
                        break;
                    }
                    tool_state.reset();
                }
            }

            // Emit complete UTF-8 characters
            if (!tool_state.is_collecting()) {
                LOG_INFO("[%d] 📤 Emitting to callback: '%s'", i, complete_chars.c_str());
                jni::on_token(env, jcallback, complete_chars);
                tokens_generated++;
            } else {
                LOG_INFO("[%d] ⏸️ Tool state collecting - not emitting yet", i);
            }
        } else {
            LOG_INFO("[%d] ⏳ No complete UTF-8 chars yet (buffering)", i);
        }

        // Prepare batch for next token
        single.n_tokens = 1;
        single.token[0] = tok;
        single.pos[0] = static_cast<int32_t>(prompt_toks.size() + i);
        single.n_seq_id[0] = 1;
        single.seq_id[0][0] = 0;
        single.logits[0] = true;
        LOG_INFO("[%d] Prepared batch: pos=%d", i, single.pos[0]);

        LOG_INFO("[%d] Calling llama_decode...", i);
        int decode_result = llama_decode(g_state.ctx, single);
        if (decode_result != 0) {
            LOG_ERROR("[%d] ❌ llama_decode FAILED with code: %d", i, decode_result);
            jni::on_error(env, jcallback, "llama_decode failed during generation");
            break;
        }
        LOG_INFO("[%d] ✓ llama_decode successful", i);

        if (env->ExceptionCheck()) {
            LOG_ERROR("[%d] ❌ Java exception detected - aborting", i);
            env->ExceptionClear();
            break;
        }
    }

    LOG_INFO("════════════════════════════════════════════════════");
    LOG_INFO("=== TOKEN GENERATION LOOP ENDED ===");
    LOG_INFO("════════════════════════════════════════════════════");
    LOG_INFO("Tokens generated: %d", tokens_generated);

    LOG_INFO("Flushing UTF-8 buffer...");
    std::string remaining = g_state.flush_utf8_buffer();
    if (!remaining.empty()) {
        LOG_INFO("Flushed remaining bytes: '%s' (len=%zu)", remaining.c_str(), remaining.size());
        jni::on_token(env, jcallback, remaining);
    } else {
        LOG_INFO("No remaining bytes in buffer");
    }

    LOG_INFO("Freeing batch...");
    llama_batch_free(single);

    LOG_INFO("Flushing carry...");
    utf8::flush_carry(env, jcallback);

    LOG_INFO("Calling on_done callback...");
    jni::on_done(env, jcallback);

    LOG_INFO("════════════════════════════════════════════════════");
    LOG_INFO("=== nativeGenerateStream COMPLETED ===");
    LOG_INFO("════════════════════════════════════════════════════");
    return JNI_TRUE;
}

/*  --------------------------------------------------------------
 *      JNI: kernel-level diagnostics
 *  -------------------------------------------------------------- */
extern "C" JNIEXPORT void JNICALL
Java_com_mp_ai_1core_NativeLib_llamaPrintTimings(JNIEnv*, jobject) {
    LOG_INFO("Printing system info and timings...");
    llama_print_system_info();
    llama_perf_context_print(g_state.ctx);
}

/*  --------------------------------------------------------------
 *      JNI: model information
 *  -------------------------------------------------------------- */

static const char* detect_model_architecture(llama_model* model) {
    if (!model) return "unknown";

    char arch_buf[128] = {0};
    int32_t arch_len = llama_model_meta_val_str(model, "general.architecture", arch_buf, sizeof(arch_buf));

    if (arch_len > 0) {
        std::string arch(arch_buf);
        std::transform(arch.begin(), arch.end(), arch.begin(), ::tolower);

        if (arch.find("llama") != std::string::npos) return "llama";
        if (arch.find("qwen") != std::string::npos) return "qwen";
        if (arch.find("gemma") != std::string::npos) return "gemma";
        if (arch.find("phi") != std::string::npos) return "phi";
        if (arch.find("mistral") != std::string::npos) return "mistral";
        if (arch.find("mixtral") != std::string::npos) return "mixtral";
        if (arch.find("yi") != std::string::npos) return "yi";
        if (arch.find("deepseek") != std::string::npos) return "deepseek";
        if (arch.find("command") != std::string::npos) return "command-r";
        if (arch.find("starcoder") != std::string::npos) return "starcoder";

        return arch_buf;
    }

    char name_buf[256] = {0};
    int32_t name_len = llama_model_meta_val_str(model, "general.name", name_buf, sizeof(name_buf));

    if (name_len > 0) {
        std::string name(name_buf);
        std::transform(name.begin(), name.end(), name.begin(), ::tolower);

        if (name.find("qwen") != std::string::npos) return "qwen";
        if (name.find("gemma") != std::string::npos) return "gemma";
        if (name.find("phi") != std::string::npos) return "phi";
        if (name.find("mistral") != std::string::npos) return "mistral";
        if (name.find("llama") != std::string::npos) return "llama";
        if (name.find("yi-") != std::string::npos) return "yi";
        if (name.find("deepseek") != std::string::npos) return "deepseek";
    }

    return "unknown";
}

static const char* get_model_name(llama_model* model) {
    if (!model) return "";
    static char name_buf[256] = {0};
    llama_model_meta_val_str(model, "general.name", name_buf, sizeof(name_buf));
    return name_buf;
}

static const char* get_model_description(llama_model* model) {
    if (!model) return "";
    static char desc_buf[512] = {0};
    llama_model_meta_val_str(model, "general.description", desc_buf, sizeof(desc_buf));
    return desc_buf;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_mp_ai_1core_NativeLib_nativeGetModelInfo(JNIEnv* env, jobject) {
    if (!g_state.model) return env->NewStringUTF("{}");

    const llama_vocab* vocab = llama_model_get_vocab(g_state.model);
    std::ostringstream json;

    json << "{";

    const char* arch = detect_model_architecture(g_state.model);
    const char* name = get_model_name(g_state.model);
    const char* desc = get_model_description(g_state.model);

    json << R"("architecture":")" << chat::json_escape(arch) << "\",";
    json << R"("name":")" << chat::json_escape(name) << "\",";
    json << R"("description":")" << chat::json_escape(desc) << "\",";

    json << "\"n_vocab\":" << (vocab ? llama_vocab_n_tokens(vocab) : 0) << ",";
    json << "\"n_ctx_train\":" << llama_model_n_ctx_train(g_state.model) << ",";
    json << "\"n_embd\":" << llama_model_n_embd(g_state.model) << ",";
    json << "\"n_layer\":" << llama_model_n_layer(g_state.model) << ",";
    json << "\"n_head\":" << llama_model_n_head(g_state.model) << ",";
    json << "\"n_head_kv\":" << llama_model_n_head_kv(g_state.model) << ",";

    if (vocab) {
        json << "\"bos\":" << llama_vocab_bos(vocab) << ",";
        json << "\"eos\":" << llama_vocab_eos(vocab) << ",";
        json << "\"eot\":" << llama_vocab_eot(vocab) << ",";
        json << "\"nl\":" << llama_vocab_nl(vocab) << ",";

        const char* vocab_type = "unknown";
        switch (llama_vocab_type(vocab)) {
            case LLAMA_VOCAB_TYPE_SPM: vocab_type = "spm"; break;
            case LLAMA_VOCAB_TYPE_BPE: vocab_type = "bpe"; break;
            case LLAMA_VOCAB_TYPE_WPM: vocab_type = "wpm"; break;
        }
        json << R"("vocab_type":")" << vocab_type << "\",";
    }

    char arch_buf[128] = {0};
    llama_model_meta_val_str(g_state.model, "general.architecture", arch_buf, sizeof(arch_buf));
    std::string architecture(arch_buf);

    const char* tmpl = llama_model_chat_template(g_state.model, nullptr);
    std::string template_str;
    const char* template_type = "custom";

    if (tmpl && *tmpl) {
        template_str = std::string(tmpl);
    } else {
        std::transform(architecture.begin(), architecture.end(), architecture.begin(), ::tolower);

        if (architecture.find("llama") != std::string::npos) {
            template_str = "{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] }}{% endif %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% endif %}{% if message['role'] == 'assistant' %}{{ message['content'] }}{% endif %}{% endfor %}";
            template_type = "llama";
        }
        else if (architecture.find("qwen") != std::string::npos) {
            template_str = "<|im_start|>system\n{{ system }}<|im_end|>\n<|im_start|>user\n{{ user }}<|im_end|>\n<|im_start|>assistant\n";
            template_type = "chatml";
        }
        else if (architecture.find("gemma") != std::string::npos) {
            template_str = "<start_of_turn>system\n{{ system }}<end_of_turn>\n<start_of_turn>user\n{{ user }}<end_of_turn>\n<start_of_turn>model\n";
            template_type = "gemma";
        }
        else if (architecture.find("phi") != std::string::npos) {
            template_str = "<|system|>\n{{ system }}<|end|>\n<|user|>\n{{ user }}<|end|>\n<|assistant|>\n";
            template_type = "phi";
        }
        else if (architecture.find("mistral") != std::string::npos || architecture.find("mixtral") != std::string::npos) {
            template_str = "[INST] {{ system }}\n{{ user }} [/INST]";
            template_type = "llama";
        }
        else {
            template_str = "<|im_start|>system\n{{ system }}<|im_end|>\n<|im_start|>user\n{{ user }}<|im_end|>\n<|im_start|>assistant\n";
            template_type = "chatml";
        }
    }

    if (tmpl && *tmpl) {
        if (template_str.find("<|im_start|>") != std::string::npos)
            template_type = "chatml";
        else if (template_str.find("<start_of_turn>") != std::string::npos)
            template_type = "gemma";
        else if (template_str.find("[INST]") != std::string::npos)
            template_type = "llama";
        else if (template_str.find("<|system|>") != std::string::npos)
            template_type = "phi";
    }

    json << R"("chat_template":")" << chat::json_escape(template_str) << "\",";
    json << R"("template_type":")" << template_type << "\",";
    json << R"("system":")" << chat::json_escape(llama_print_system_info()) << "\"";
    json << "}";

    return env->NewStringUTF(json.str().c_str());
}

/*  --------------------------------------------------------------
 *      JNI: persistence helpers
 *  -------------------------------------------------------------- */
extern "C" JNIEXPORT jlong JNICALL
Java_com_mp_ai_1core_NativeLib_nativeGetStateSize(JNIEnv*, jobject) {
    jlong size = g_state.get_state_size();
    LOG_INFO("State size: %lld bytes", size);
    return size;
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_mp_ai_1core_NativeLib_nativeGetStateData(JNIEnv* env, jobject) {
    jlong sz = g_state.get_state_size();
    if (!sz) {
        LOG_WARN("State size is 0, returning null");
        return nullptr;
    }

    LOG_INFO("Getting state data: %lld bytes", sz);
    jbyteArray arr = env->NewByteArray(static_cast<jsize>(sz));
    if (!arr) {
        LOG_ERROR("Failed to allocate byte array");
        return nullptr;
    }

    void* buffer = env->GetByteArrayElements(arr, nullptr);
    g_state.get_state_data(buffer, static_cast<size_t>(sz));
    env->ReleaseByteArrayElements(arr, (jbyte*)buffer, 0);
    LOG_INFO("✓ State data retrieved");
    return arr;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1core_NativeLib_nativeLoadStateData(JNIEnv* env, jobject,
                                                   jbyteArray arr) {
    if (!arr) {
        LOG_ERROR("Null array provided");
        return JNI_FALSE;
    }
    jbyte* buf = env->GetByteArrayElements(arr, nullptr);
    auto len = static_cast<size_t>(env->GetArrayLength(arr));
    LOG_INFO("Loading state data: %zu bytes", len);
    bool ok = g_state.load_state_data(buf, len);
    env->ReleaseByteArrayElements(arr, buf, JNI_ABORT);
    LOG_INFO("Load state: %s", ok ? "SUCCESS" : "FAILED");
    return ok ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1core_NativeLib_nativeSaveStateFile(JNIEnv* env, jobject,
                                                   jstring jpath) {
    const char* path = env->GetStringUTFChars(jpath, nullptr);
    LOG_INFO("Saving state to file: %s", path);
    bool ok = llama_state_save_file(g_state.ctx, path, nullptr, 0);
    env->ReleaseStringUTFChars(jpath, path);
    LOG_INFO("Save state file: %s", ok ? "SUCCESS" : "FAILED");
    return ok ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1core_NativeLib_nativeLoadStateFile(JNIEnv* env, jobject,
                                                   jstring jpath) {
    const char* path = env->GetStringUTFChars(jpath, nullptr);
    LOG_INFO("Loading state from file: %s", path);
    bool ok = llama_state_load_file(g_state.ctx, path, nullptr, 0, nullptr);
    env->ReleaseStringUTFChars(jpath, path);
    LOG_INFO("Load state file: %s", ok ? "SUCCESS" : "FAILED");
    return ok ? JNI_TRUE : JNI_FALSE;
}