/*=============================================================
 *   ai_core.cpp - OPTIMIZED LOGGING VERSION
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

// Enable detailed logging only in debug builds
#ifdef NDEBUG
#define LOG_VERBOSE(...)
#define LOG_DETAIL(...)
#else
#define LOG_VERBOSE(...) LOG_INFO(__VA_ARGS__)
    #define LOG_DETAIL(...) LOG_INFO(__VA_ARGS__)
#endif

/*  --------------------------------------------------------------
 *      Global state and guard
 *  -------------------------------------------------------------- */
static std::mutex g_init_mtx;
static std::atomic<bool> g_stop_requested{false};

/*  --------------------------------------------------------------
 *      Helper – build & init grammar when tools enabled
 *  -------------------------------------------------------------- */
static void maybe_init_grammar() {
    if (!g_state.tools_enabled) {
        return;
    }

    const std::string grammar = chat::build_tool_grammar(g_state.tools_json);

    if (!grammar.empty()) {
        if (g_state.grammar_sampler) {
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
        }
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
    LOG_INFO("Initializing model...");
    std::lock_guard<std::mutex> lk(g_init_mtx);

    const std::string path = utf8::from_jstring(env, jpath);
    LOG_VERBOSE("Model path: %s, threads=%d, ctx=%d", path.c_str(),
                static_cast<int>(jthreads), static_cast<int>(ctxSize));

    g_state.release();
    llama_backend_init();

    int phys = count_physical_cores();
    int nthreads = (jthreads > 0) ? static_cast<int>(jthreads) : phys;

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_mmap = true;
    mparams.use_mlock = false;
    mparams.check_tensors = true;

    g_state.model = llama_model_load_from_file(path.c_str(), mparams);
    if (!g_state.model) {
        LOG_ERROR("Failed to load model from '%s'", path.c_str());
        g_state.release();
        return JNI_FALSE;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = ctxSize;
    cparams.n_batch = 512;
    cparams.n_ubatch = 256;
    cparams.n_threads = nthreads;
    cparams.n_threads_batch = nthreads;
    cparams.offload_kqv = false;
    cparams.n_seq_max = 1;
    cparams.no_perf = false;

    g_state.ctx = llama_init_from_model(g_state.model, cparams);
    if (!g_state.ctx) {
        LOG_ERROR("Failed to create context");
        g_state.release();
        return JNI_FALSE;
    }

    g_state.ctx_size = ctxSize;
    g_state.batch_size = cparams.n_batch;

    g_state.rebuild_sampler(static_cast<int>(topK), topP, temp, minP,
                            mirostat, mirostatTau, mirostatEta, seed);
    g_state.warmup_context();
    maybe_init_grammar();

    LOG_INFO("Model initialized successfully");
    return JNI_TRUE;
}

/*  --------------------------------------------------------------
 *      JNI: release resources
 *  -------------------------------------------------------------- */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1core_NativeLib_nativeRelease(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lk(g_init_mtx);
    g_state.release();
    LOG_INFO("Resources released");
    return JNI_TRUE;
}

/*  --------------------------------------------------------------
 *      JNI: configuration setters
 *  -------------------------------------------------------------- */
extern "C" JNIEXPORT void JNICALL
Java_com_mp_ai_1core_NativeLib_nativeSetSystemPrompt(JNIEnv* env, jobject,
                                                     jstring jprompt) {
    g_state.system_prompt = utf8::from_jstring(env, jprompt);
    LOG_VERBOSE("System prompt updated: %zu bytes", g_state.system_prompt.size());
}

extern "C" JNIEXPORT void JNICALL
Java_com_mp_ai_1core_NativeLib_nativeSetChatTemplate(JNIEnv* env, jobject,
                                                     jstring jtemplate) {
    g_state.chat_template_override = utf8::from_jstring(env, jtemplate);
    LOG_VERBOSE("Chat template set: %zu bytes", g_state.chat_template_override.size());
}

extern "C" JNIEXPORT void JNICALL
Java_com_mp_ai_1core_NativeLib_nativeSetToolsJson(JNIEnv* env, jobject,
                                                  jstring jtools) {
    g_state.tools_json = utf8::from_jstring(env, jtools);
    g_state.tools_enabled = !g_state.tools_json.empty();
    LOG_VERBOSE("Tools enabled: %d", static_cast<int>(g_state.tools_enabled));
    maybe_init_grammar();
}

/*  --------------------------------------------------------------
 *      JNI: request stop
 *  -------------------------------------------------------------- */
extern "C" JNIEXPORT void JNICALL
Java_com_mp_ai_1core_NativeLib_nativeStopGeneration(JNIEnv*, jobject) {
    g_stop_requested.store(true);
    LOG_INFO("Stop requested");
}

/*  --------------------------------------------------------------
 *      JNI: clear KV cache (fast reset)
 *  -------------------------------------------------------------- */
extern "C" JNIEXPORT void JNICALL
Java_com_mp_ai_1core_NativeLib_nativeClearMemory(JNIEnv*, jobject) {
    if (g_state.ctx) {
        llama_memory_t mem = llama_get_memory(g_state.ctx);
        if (mem) {
            llama_memory_clear(mem, true);
            LOG_VERBOSE("KV cache cleared");
        }
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
    LOG_INFO("Starting generation (max_tokens=%d)", static_cast<int>(max_tokens));

    if (!g_state.is_ready()) {
        LOG_ERROR("Model not initialized");
        jni::on_error(env, jcallback, "Model not initialized");
        return JNI_FALSE;
    }

    g_state.prepare_for_generation();
    g_stop_requested.store(false);

    const std::string user_msg = utf8::from_jstring(env, jprompt);
    const llama_vocab* vocab = llama_model_get_vocab(g_state.model);

    if (!vocab) {
        LOG_ERROR("Failed to get vocab");
        jni::on_error(env, jcallback, "Failed to get vocab");
        return JNI_FALSE;
    }

    // Build prompt with optional tool preamble
    std::string system = g_state.system_prompt;
    if (g_state.tools_enabled) {
        system += "\n" + chat::build_tool_preamble(g_state.tools_json);
    }

    const std::string prompt = chat::apply_template(g_state.model, system, user_msg,
                                                    g_state.chat_template_override, true);
    LOG_VERBOSE("Prompt size: %zu bytes", prompt.size());

    // Tokenize
    std::vector<llama_token> prompt_toks = g_state.tokenize(prompt);
    if (prompt_toks.empty()) {
        LOG_ERROR("Tokenization failed");
        jni::on_error(env, jcallback, "Tokenisation failed");
        return JNI_FALSE;
    }

    // Check context limits
    int32_t available = g_state.ctx_size - static_cast<int32_t>(prompt_toks.size()) - 8;
    if (available <= 0) {
        LOG_ERROR("Context overflow (prompt=%zu, ctx=%d)",
                  prompt_toks.size(), g_state.ctx_size);
        jni::on_error(env, jcallback, "Context overflow – shorten your prompt");
        return JNI_TRUE;
    }

    auto to_generate = static_cast<int32_t>(max_tokens > 0 ? max_tokens : 128);
    to_generate = std::min(to_generate, available);
    LOG_VERBOSE("Generating %d tokens (available=%d)", to_generate, available);

    // Decode prompt
    if (!g_state.decode_prompt(prompt_toks)) {
        LOG_ERROR("Prompt decode failed");
        jni::on_error(env, jcallback, "Decoding prompt failed");
        return JNI_TRUE;
    }

    /* ---------------------------------------------------------
     *  Streaming loop – CRITICAL PATH, minimal logging
     * -------------------------------------------------------- */
    ToolCallState tool_state;
    llama_token eos = llama_vocab_eos(vocab);
    llama_token eot = llama_vocab_eot(vocab);

    llama_batch single = llama_batch_init(1, 0, 1);
    g_state.utf8_carry_buffer.clear();

    int tokens_generated = 0;
    for (int i = 0; i < to_generate && !g_stop_requested.load(); ++i) {
        // Sample & accept
        llama_token tok = llama_sampler_sample(g_state.sampler, g_state.ctx, -1);
        llama_sampler_accept(g_state.sampler, tok);

        // Convert EOS to space if first token
        if (i == 0 && (tok == eos || tok == eot)) {
            tok = g_state.space_token();
        }

        // Check for end
        if (tok == eos || tok == eot) {
            LOG_DETAIL("Generated EOS/EOT at token %d", i);
            break;
        }

        // Detokenize with buffering
        std::string complete_chars = g_state.detokenize_buffered(tok);

        if (!complete_chars.empty()) {
            // Tool-call detection
            if (g_state.tools_enabled) {
                bool complete = tool_state.accumulate(complete_chars);
                if (complete) {
                    std::string name, payload;
                    if (tool_state.extract_tool_call(name, payload)) {
                        LOG_INFO("Tool call: %s", name.c_str());
                        jni::on_toolcall(env, jcallback, name, payload);
                        break;
                    }
                    tool_state.reset();
                }
            }

            // Emit complete UTF-8 characters
            if (!tool_state.is_collecting()) {
                jni::on_token(env, jcallback, complete_chars);
                tokens_generated++;
            }
        }

        // Prepare next batch
        single.n_tokens = 1;
        single.token[0] = tok;
        single.pos[0] = static_cast<int32_t>(prompt_toks.size() + i);
        single.n_seq_id[0] = 1;
        single.seq_id[0][0] = 0;
        single.logits[0] = true;

        int decode_result = llama_decode(g_state.ctx, single);
        if (decode_result != 0) {
            LOG_ERROR("Decode failed at token %d (code=%d)", i, decode_result);
            jni::on_error(env, jcallback, "llama_decode failed during generation");
            break;
        }

        if (env->ExceptionCheck()) {
            LOG_ERROR("Java exception during generation");
            env->ExceptionClear();
            break;
        }
    }

    LOG_INFO("Generation complete (%d tokens)", tokens_generated);

    // Cleanup
    std::string remaining = g_state.flush_utf8_buffer();
    if (!remaining.empty()) {
        jni::on_token(env, jcallback, remaining);
    }

    llama_batch_free(single);
    utf8::flush_carry(env, jcallback);
    jni::on_done(env, jcallback);

    return JNI_TRUE;
}

/*  --------------------------------------------------------------
 *      JNI: kernel-level diagnostics
 *  -------------------------------------------------------------- */
extern "C" JNIEXPORT void JNICALL
Java_com_mp_ai_1core_NativeLib_llamaPrintTimings(JNIEnv*, jobject) {
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
    return g_state.get_state_size();
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_mp_ai_1core_NativeLib_nativeGetStateData(JNIEnv* env, jobject) {
    jlong sz = g_state.get_state_size();
    if (!sz) return nullptr;

    jbyteArray arr = env->NewByteArray(static_cast<jsize>(sz));
    if (!arr) {
        LOG_ERROR("Failed to allocate state array");
        return nullptr;
    }

    void* buffer = env->GetByteArrayElements(arr, nullptr);
    g_state.get_state_data(buffer, static_cast<size_t>(sz));
    env->ReleaseByteArrayElements(arr, (jbyte*)buffer, 0);
    return arr;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1core_NativeLib_nativeLoadStateData(JNIEnv* env, jobject,
                                                   jbyteArray arr) {
    if (!arr) return JNI_FALSE;

    jbyte* buf = env->GetByteArrayElements(arr, nullptr);
    auto len = static_cast<size_t>(env->GetArrayLength(arr));
    bool ok = g_state.load_state_data(buf, len);
    env->ReleaseByteArrayElements(arr, buf, JNI_ABORT);

    if (!ok) {
        LOG_ERROR("Failed to load state data");
    }
    return ok ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1core_NativeLib_nativeSaveStateFile(JNIEnv* env, jobject,
                                                   jstring jpath) {
    const char* path = env->GetStringUTFChars(jpath, nullptr);
    bool ok = llama_state_save_file(g_state.ctx, path, nullptr, 0);
    env->ReleaseStringUTFChars(jpath, path);

    if (!ok) {
        LOG_ERROR("Failed to save state to file");
    }
    return ok ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1core_NativeLib_nativeLoadStateFile(JNIEnv* env, jobject,
                                                   jstring jpath) {
    const char* path = env->GetStringUTFChars(jpath, nullptr);
    bool ok = llama_state_load_file(g_state.ctx, path, nullptr, 0, nullptr);
    env->ReleaseStringUTFChars(jpath, path);

    if (!ok) {
        LOG_ERROR("Failed to load state from file");
    }
    return ok ? JNI_TRUE : JNI_FALSE;
}