// ai_core_streaming_with_tools.cpp
// Streaming generation for Android with llama.cpp + UTF‑safe JNI bridging + tool calling (GBNF)
// Package: com.mp.ai_core

#include <jni.h>
#include <string>
#include <vector>
#include <sstream>
#include <memory>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <atomic>

#include "llama.h"
#include "cpu_helper.h"
#include <src/llama-mmap.h>
#include "src/llama-io.h"
#include "ggml-backend.h"

#if defined(__ANDROID__)

#include <android/log.h>

#define LOG_TAG "ai_core"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define LOGI(...)
#define LOGE(...)
#endif

// -----------------------------------------------------------------------------
// Globals
// -----------------------------------------------------------------------------
static llama_model *g_model = nullptr;
static llama_context *g_ctx = nullptr;
static llama_sampler *g_sampler = nullptr;   // main chain
static llama_sampler *g_sampler_grammar = nullptr;   // optional grammar sampler (front of chain)

static std::string g_system_prompt = "You are a helpful assistant.";
static std::string g_chat_template_override;
static std::atomic<bool> g_stop_requested(false);

// Tools
static std::string g_tools_json;  // OpenAI-style tools array
static bool g_tools_enabled = false;

// Runtime params (saved at init so we rebuild chains consistently)
static int g_ctx_size = 2048;
static int g_n_batch = 512;
static int g_init_top_k = 20;
static float g_init_top_p = 0.9f;
static float g_init_temp = 0.7f;
static float g_init_min_p = 0.0f;

// Carry buffer for streaming UTF-8 that may be split mid-codepoint.
static thread_local std::string g_utf8_carry;

// Tool-call streaming accumulator
static std::string g_tool_accum;
static int g_brace_depth = 0;
static bool g_in_tool_json = false;

static void log_gpu_info() {
    LOGI("=== GPU Diagnostic ===");

    // Detect any registered device
    int count = ggml_backend_dev_count();
    if (count == 0) {
        LOGE("No GPU devices found – will run purely on CPU");
        return;
    }

    for (int i=0; i<count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char *name = ggml_backend_dev_name(dev);
        const char *desc = ggml_backend_dev_description(dev);
        LOGI("GPU %d: %s (%s)", i, name, desc ? desc : "unknown");
    }
}

// -----------------------------------------------------------------------------
// UTF helpers (UTF-16 <-> UTF-8, streaming-safe)
// -----------------------------------------------------------------------------
static inline void push_u16(std::u16string &o, uint32_t cp) {
    if (cp <= 0xFFFFu) {
        if (cp >= 0xD800u && cp <= 0xDFFFu) cp = 0xFFFDu; // disallow lone surrogates
        o.push_back((char16_t) cp);
    } else {
        cp -= 0x10000u;
        o.push_back((char16_t) (0xD800u + (cp >> 10)));
        o.push_back((char16_t) (0xDC00u + (cp & 0x3FF)));
    }
}

static inline bool decode_one_utf8(const std::string &s, size_t &i, uint32_t &cp) {
    if (i >= s.size()) return false;
    unsigned char b0 = (unsigned char) s[i];
    size_t rem = s.size() - i;

    if (b0 < 0x80) {
        cp = b0;
        i += 1;
        return true;
    }
    if ((b0 >> 5) == 0x6) {
        if (rem < 2) return false;
        unsigned char b1 = (unsigned char) s[i + 1];
        if ((b1 & 0xC0) != 0x80) {
            i++;
            cp = 0xFFFDu;
            return true;
        }
        cp = ((b0 & 0x1F) << 6) | (b1 & 0x3F);
        i += 2;
        return true;
    }
    if ((b0 >> 4) == 0xE) {
        if (rem < 3) return false;
        unsigned char b1 = (unsigned char) s[i + 1], b2 = (unsigned char) s[i + 2];
        if ((b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80) {
            i++;
            cp = 0xFFFDu;
            return true;
        }
        cp = ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F);
        i += 3;
        return true;
    }
    if ((b0 >> 3) == 0x1E) {
        if (rem < 4) return false;
        unsigned char b1 = (unsigned char) s[i + 1], b2 = (unsigned char) s[i +
                                                                            2], b3 = (unsigned char) s[
                i + 3];
        if ((b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80 || (b3 & 0xC0) != 0x80) {
            i++;
            cp = 0xFFFDu;
            return true;
        }
        cp = ((b0 & 0x07) << 18) | ((b1 & 0x3F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F);
        i += 4;
        return true;
    }
    i++;
    cp = 0xFFFDu;
    return true;
}

static void
utf8_to_utf16_with_carry(const std::string &in, std::u16string &out, std::string &carry) {
    std::string s = carry + in;
    carry.clear();
    size_t i = 0;
    while (i < s.size()) {
        size_t before = i;
        uint32_t cp = 0;
        bool ok = decode_one_utf8(s, i, cp);
        if (!ok) {
            carry.assign(s.begin() + before, s.end());
            break;
        }
        push_u16(out, cp);
    }
}

static std::string jstr_to_utf8(JNIEnv *env, jstring js) {
    if (!js) return {};
    jsize n = env->GetStringLength(js);
    const jchar *p = env->GetStringChars(js, nullptr); // UTF-16
    std::string out;
    out.reserve((size_t) n);
    for (jsize i = 0; i < n;) {
        uint32_t cp;
        uint16_t w1 = p[i++];
        if (w1 >= 0xD800 && w1 <= 0xDBFF && i < n) {
            uint16_t w2 = p[i];
            if (w2 >= 0xDC00 && w2 <= 0xDFFF) {
                ++i;
                cp = 0x10000u + (((w1 - 0xD800u) << 10) | (w2 - 0xDC00u));
            }
            else cp = 0xFFFDu;
        } else if (w1 >= 0xDC00 && w1 <= 0xDFFF) { cp = 0xFFFDu; }
        else { cp = w1; }
        if (cp < 0x80) out.push_back((char) cp);
        else if (cp < 0x800) {
            out.push_back((char) (0xC0 | (cp >> 6)));
            out.push_back((char) (0x80 | (cp & 0x3F)));
        }
        else if (cp < 0x10000) {
            out.push_back((char) (0xE0 | (cp >> 12)));
            out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char) (0x80 | (cp & 0x3F)));
        } else {
            out.push_back((char) (0xF0 | (cp >> 18)));
            out.push_back((char) (0x80 | ((cp >> 12) & 0x3F)));
            out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char) (0x80 | (cp & 0x3F)));
        }
    }
    env->ReleaseStringChars(js, p);
    return out;
}

static jstring utf8_to_jstring(JNIEnv *env, const std::string &utf8, std::string &carry) {
    std::u16string u16;
    utf8_to_utf16_with_carry(utf8, u16, carry);
    if (u16.empty()) return nullptr;
    return env->NewString(reinterpret_cast<const jchar *>(u16.data()), (jsize) u16.size());
}

static void flush_utf8_carry(JNIEnv *env, jobject cb) {
    if (g_utf8_carry.empty()) return;
    std::string tmp = g_utf8_carry;
    tmp.append("\xEF\xBF\xBD", 3); // U+FFFD
    jclass cls = env->GetObjectClass(cb);
    if (!cls) {
        g_utf8_carry.clear();
        return;
    }
    jmethodID mid = env->GetMethodID(cls, "onToken", "(Ljava/lang/String;)V");
    if (!mid) {
        g_utf8_carry.clear();
        return;
    }
    std::string dummy;
    jstring jtok = utf8_to_jstring(env, tmp, dummy);
    if (jtok) {
        env->CallVoidMethod(cb, mid, jtok);
        env->DeleteLocalRef(jtok);
    }
    g_utf8_carry.clear();
}

// -----------------------------------------------------------------------------
// Small JSON utils
// -----------------------------------------------------------------------------
static std::string json_escape(const std::string &s) {
    std::ostringstream o;
    for (auto c: s) {
        switch (c) {
            case '\\':
                o << "\\\\";
                break;
            case '"':
                o << "\\\"";
                break;
            case '\n':
                o << "\\n";
                break;
            case '\r':
                o << "\\r";
                break;
            case '\t':
                o << "\\t";
                break;
            default:
                o << c;
                break;
        }
    }
    return o.str();
}

// -----------------------------------------------------------------------------
// Kotlin callback helpers (add onToolCall)
// -----------------------------------------------------------------------------
static void jni_on_error(JNIEnv *env, jobject cb, const char *msg) {
    jclass cls = env->GetObjectClass(cb);
    if (!cls) return;
    jmethodID mid = env->GetMethodID(cls, "onError", "(Ljava/lang/String;)V");
    if (!mid) return;
    std::string dummy;
    jstring jmsg = utf8_to_jstring(env, std::string(msg ? msg : "error"), dummy);
    if (!jmsg) jmsg = env->NewStringUTF("error");
    env->CallVoidMethod(cb, mid, jmsg);
    env->DeleteLocalRef(jmsg);
}

static void jni_on_token(JNIEnv *env, jobject cb, const std::string &s) {
    jclass cls = env->GetObjectClass(cb);
    if (!cls) return;
    jmethodID mid = env->GetMethodID(cls, "onToken", "(Ljava/lang/String;)V");
    if (!mid) return;
    jstring jtok = utf8_to_jstring(env, s, g_utf8_carry);
    if (jtok) {
        env->CallVoidMethod(cb, mid, jtok);
        env->DeleteLocalRef(jtok);
    }
}

static void
jni_on_toolcall(JNIEnv *env, jobject cb, const std::string &name, const std::string &payloadUtf8) {
    jclass cls = env->GetObjectClass(cb);
    if (!cls) return;
    jmethodID mid = env->GetMethodID(cls, "onToolCall", "(Ljava/lang/String;Ljava/lang/String;)V");
    if (!mid) return;
    jstring jname = env->NewStringUTF(name.c_str());
    std::string dummy;
    jstring jpayload = utf8_to_jstring(env, payloadUtf8, dummy);
    if (!jpayload) jpayload = env->NewStringUTF(payloadUtf8.c_str());
    env->CallVoidMethod(cb, mid, jname, jpayload);
    env->DeleteLocalRef(jname);
    env->DeleteLocalRef(jpayload);
}

static void jni_on_done(JNIEnv *env, jobject cb) {
    jclass cls = env->GetObjectClass(cb);
    if (!cls) return;
    jmethodID mid = env->GetMethodID(cls, "onDone", "()V");
    if (!mid) return;
    env->CallVoidMethod(cb, mid);
}

// -----------------------------------------------------------------------------
// Resource management
// -----------------------------------------------------------------------------
static void free_everything() {
    if (g_sampler_grammar) {
        llama_sampler_free(g_sampler_grammar);
        g_sampler_grammar = nullptr;
    }
    if (g_sampler) {
        llama_sampler_free(g_sampler);
        g_sampler = nullptr;
    }
    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
    llama_backend_free();
}

extern "C" JNIEXPORT void JNICALL
Java_com_mp_ai_1core_NativeLib_nativeStopGeneration(JNIEnv *, jobject) {
    g_stop_requested = true;
    LOGI("Stop generation requested");
}

// Add this diagnostic function to your C++ file
// Call it right after llama_model_load_from_file succeeds
// -----------------------------------------------------------------------------
// Tools helpers
// -----------------------------------------------------------------------------
static std::vector<std::string> extract_tool_names(const std::string &tools_json) {
    std::vector<std::string> out;
    size_t pos = 0;
    while (true) {
        size_t k = tools_json.find("\"name\"", pos);
        if (k == std::string::npos) break;
        size_t colon = tools_json.find(':', k);
        if (colon == std::string::npos) break;
        size_t q1 = tools_json.find('"', colon + 1);
        if (q1 == std::string::npos) break;
        size_t q2 = tools_json.find('"', q1 + 1);
        if (q2 == std::string::npos) break;
        std::string name = tools_json.substr(q1 + 1, q2 - q1 - 1);
        if (!name.empty()) out.push_back(name);
        pos = q2 + 1;
    }
    return out;
}

static std::string build_toolcall_gbnf(const std::string &tools_json) {
    const auto names = extract_tool_names(tools_json);
    std::ostringstream g;
    g << R"(root         ::= json
json         ::= ws toolcall ws
toolcall     ::= "{" ws "\"tool_calls\"" ws ":" ws "[" ws call ws "]" ws "}"
call         ::= "{" ws "\"name\"" ws ":" ws toolname ws "," ws "\"arguments\"" ws ":" ws object ws "}"
)";
    g << "toolname     ::= ";
    if (!names.empty()) {
        for (size_t i = 0; i < names.size(); ++i) {
            if (i) g << " | ";
            g << "\"\\\"" << names[i] << "\\\"\"";
        }
    } else { g << "\"\\\"unknown\\\"\""; }
    g << "\n";
    g << R"(
object       ::= "{" ws "}"
           | "{" ws member (ws "," ws member)* ws "}"
member       ::= string ws ":" ws value
value        ::= string | number | object | "true" | "false" | "null"
string       ::= "\"" [^"]* "\""
number       ::= [0-9]+ ("." [0-9]+)?
ws           ::= [ \t\n\r]*
)";
    return g.str();
}

static std::string tool_preamble(const std::string &toolsJson) {
    return std::string("You may call tools by emitting ONLY the JSON object:\n"
                       "{\"tool_calls\":[{\"name\":\"NAME\",\"arguments\":{...}}]}\n"
                       "Available tools (OpenAI schema):\n") + toolsJson + "\n";
}

static bool looks_like_toolcall_json(const std::string &s) {
    return s.find("\"tool_calls\"") != std::string::npos && s.size() > 10;
}

static bool maybe_collect_tool_json_chunk(const std::string &piece) {
    if (!g_tools_enabled) return false;
    for (char c: piece) {
        if (!g_in_tool_json) {
            if (c == '{') {
                g_in_tool_json = true;
                g_brace_depth = 1;
                g_tool_accum.clear();
                g_tool_accum.push_back(c);
            }
        } else {
            g_tool_accum.push_back(c);
            if (c == '{') ++g_brace_depth; else if (c == '}') --g_brace_depth;
            if (g_brace_depth == 0) return true; // object closed
        }
    }
    return false;
}

static void rebuild_sampler_chain(bool with_grammar_first) {
    if (g_sampler) {
        llama_sampler_free(g_sampler);
        g_sampler = nullptr;
    }
    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    llama_sampler *chain = llama_sampler_chain_init(sp);
    if (with_grammar_first && g_sampler_grammar) llama_sampler_chain_add(chain, g_sampler_grammar);
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(g_init_top_k));
    if (g_init_top_p < 1.0f)
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(g_init_top_p, 1));
    if (g_init_temp != 1.0f) llama_sampler_chain_add(chain, llama_sampler_init_temp(g_init_temp));
    if (g_init_temp > 0.0f) llama_sampler_chain_add(chain, llama_sampler_init_dist(-1));
    if (g_init_min_p > 0.0f)
        llama_sampler_chain_add(chain, llama_sampler_init_min_p(g_init_min_p, 1));
    g_sampler = chain;
    llama_sampler_reset(g_sampler);
}

static bool enable_tool_grammar_if_needed() {
    if (!g_tools_enabled) return false;
    LOGI("TOOLS JSON set (%zu bytes)", g_tools_json.size());
    const std::string gbnf = build_toolcall_gbnf(g_tools_json);
    if (g_sampler_grammar) {
        llama_sampler_free(g_sampler_grammar);
        g_sampler_grammar = nullptr;
    }
    const llama_vocab *vocab = llama_model_get_vocab(g_model);
    g_sampler_grammar = llama_sampler_init_grammar(vocab, gbnf.c_str(), "root");
    if (!g_sampler_grammar) {
        LOGE("grammar init failed");
        return false;
    }
    rebuild_sampler_chain(/*with_grammar_first=*/true);
    LOGI("Tool grammar enabled");
    return true;
}

// Feed tokens in chunks <= g_n_batch; return false on decode error
static bool
decode_tokens_chunked(llama_context *ctx, const std::vector<llama_token> &toks, int32_t start_pos,
                      int32_t n_batch, JNIEnv *env, jobject jcb) {
    if (toks.empty()) return true;
    llama_batch batch = llama_batch_init(n_batch, /*embd*/0, /*n_seq_max*/1);
    int32_t pos = start_pos;
    size_t i = 0;
    while (i < toks.size()) {
        const int32_t take = (int32_t) std::min<size_t>(n_batch, toks.size() - i);
        batch.n_tokens = take;
        for (int32_t t = 0; t < take; ++t) {
            batch.token[t] = toks[i + t];
            batch.pos[t] = pos + t;
            batch.n_seq_id[t] = 1;
            batch.seq_id[t][0] = 0;
            batch.logits[t] = (t == take - 1);
        }
        int rc = llama_decode(ctx, batch);
        if (rc != 0) {
            llama_batch_free(batch);
            if (rc == 1) jni_on_error(env, jcb, "decode failed: no KV slot (context overflow)");
            else jni_on_error(env, jcb, "decode() failed on prompt chunk");
            return false;
        }
        pos += take;
        i += (size_t) take;
    }
    llama_batch_free(batch);
    return true;
}

// -----------------------------------------------------------------------------
// JNI API
// -----------------------------------------------------------------------------
extern "C" JNIEXPORT void JNICALL
Java_com_mp_ai_1core_NativeLib_nativeSetSystemPrompt(JNIEnv *env, jobject, jstring jprompt) {
    g_system_prompt = jstr_to_utf8(env, jprompt);
    LOGI("System prompt updated (%zu bytes)", g_system_prompt.size());
}

extern "C" JNIEXPORT void JNICALL
Java_com_mp_ai_1core_NativeLib_nativeSetChatTemplate(JNIEnv *env, jobject, jstring jtemplate) {
    g_chat_template_override = jstr_to_utf8(env, jtemplate);
    LOGI("Chat template override set (%zu bytes)", g_chat_template_override.size());
}

extern "C" JNIEXPORT void JNICALL
Java_com_mp_ai_1core_NativeLib_nativeSetToolsJson(JNIEnv *env, jobject, jstring jtools) {
    g_tools_json = jstr_to_utf8(env, jtools);
    g_tools_enabled = !g_tools_json.empty();
    LOGI("Tools json set (%zu bytes); \nenabled=%d, \ntool = %s", g_tools_json.size(),
         (int) g_tools_enabled, g_tools_json.c_str());
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1core_NativeLib_nativeRelease(JNIEnv *, jobject) {
    free_everything();
    return JNI_TRUE;
}

static std::mutex g_init_mtx;

extern "C" JNIEXPORT void JNICALL
Java_com_mp_ai_1core_NativeLib_llamaPrintTimings(JNIEnv *env, jobject thiz) {
    llama_print_system_info();
    llama_perf_context_print(g_ctx);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1core_NativeLib_nativeInit(JNIEnv *env, jobject, jstring jpath, jint jthreads,
                                          jint gpuLayers, jboolean useMMAP, jboolean /*useMLOCK*/,
                                          jint ctxSize, jfloat temp, jint topK, jfloat topP,
                                          jfloat minP) {
    std::lock_guard<std::mutex> _lock(g_init_mtx);

    const std::string path = jstr_to_utf8(env, jpath);
    free_everything();
    llama_backend_init();



    const int physCores = count_physical_cores();


    // FIXED: Proper GPU layer handling
    int gpu_layers = gpuLayers;
    if (gpu_layers < 0) {
        // -1 means ALL layers - get actual layer count from model later
        gpu_layers = 999; // Will be clamped by llama.cpp to actual layer count
        LOGI("GPU: Offloading ALL layers (requested -1)");
    } else if (gpu_layers == 0) {
        LOGI("GPU: CPU-only mode (0 layers)");
    } else {
        LOGI("GPU: Offloading %d layers", gpu_layers);
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = gpu_layers;
    mparams.use_mmap = useMMAP;
    mparams.use_mlock = false;
    mparams.check_tensors = true;

    LOGI("Loading model: %s", path.c_str());
    g_model = llama_model_load_from_file(path.c_str(), mparams);
    if (!g_model) {
        LOGE("Failed to load model: %s", path.c_str());
        free_everything();
        return JNI_FALSE;
    }
    log_gpu_info();
    // Log actual GPU offload info
    int32_t n_layer = llama_model_n_layer(g_model);
    LOGI("Model has %d layers total", n_layer);
    if (gpu_layers > 0) {
        int actual_offloaded = std::min(gpu_layers, n_layer);
        LOGI("Actually offloading %d/%d layers to GPU", actual_offloaded, n_layer);
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = ctxSize;
    cparams.n_batch = 254;
    cparams.n_ubatch = 128;

    // FIXED: Enable KQV offloading for GPU
    cparams.offload_kqv = true; // true if using GPU

    cparams.n_seq_max = 1;
    cparams.n_threads = jthreads > 0 ? jthreads : physCores;
    cparams.n_threads_batch = cparams.n_threads;
    cparams.no_perf = false;

    LOGI("Creating context (ctx_size=%d, offload_kqv=%s)",
         ctxSize, cparams.offload_kqv ? "true" : "false");

    g_ctx = llama_init_from_model(g_model, cparams);
    if (!g_ctx) {
        LOGE("Failed to create context");
        free_everything();
        return JNI_FALSE;
    }

    // Persist runtime knobs
    g_ctx_size = ctxSize;
    g_n_batch = cparams.n_batch;
    g_init_top_k = topK;
    g_init_top_p = topP;
    g_init_temp = temp;
    g_init_min_p = minP;

    // Warm-up single token
    {
        const llama_vocab *vocab = llama_model_get_vocab(g_model);
        llama_token sp[4];
        int nsp = llama_tokenize(vocab, " ", 1, sp, 4, true, true);
        llama_batch warm = llama_batch_init(1, 0, 1);
        if (nsp > 0) {
            warm.n_tokens = 1;
            warm.token[0] = sp[0];
            warm.pos[0] = 0;
            warm.n_seq_id[0] = 1;
            warm.seq_id[0][0] = 0;
            warm.logits[0] = true;
            (void) llama_decode(g_ctx, warm);
        }
        llama_batch_free(warm);
    }

    // Initial sampler chain
    rebuild_sampler_chain(/*with_grammar_first=*/false);

    LOGI("Model initialized successfully:");
    LOGI("  - GPU layers: %d/%d", std::min(gpu_layers, n_layer), n_layer);
    LOGI("  - KQV offload: %s", cparams.offload_kqv ? "enabled" : "disabled");
    LOGI("  - Batch size: %d", cparams.n_batch);
    LOGI("  - Context size: %d", ctxSize);

    return JNI_TRUE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_mp_ai_1core_NativeLib_nativeGetModelInfo(JNIEnv *env, jobject) {
    if (!g_model) return env->NewStringUTF("");
    const llama_vocab *vocab = llama_model_get_vocab(g_model);
    std::ostringstream oss;
    oss << "{";
    oss << "\"core\":" << "{" << "\"n_vocab\":" << (vocab ? llama_vocab_n_tokens(vocab) : 0)
        << ",\"n_ctx_train\":" << llama_model_n_ctx_train(g_model) << ",\"n_embd\":"
        << llama_model_n_embd(g_model) << ",\"n_layer\":" << llama_model_n_layer(g_model)
        << ",\"n_head\":" << llama_model_n_head(g_model) << ",\"n_head_kv\":"
        << llama_model_n_head_kv(g_model) << "},";
    if (vocab) {
        oss << "\"special\":" << "{" << "\"bos\":" << llama_vocab_bos(vocab) << ",\"eos\":"
            << llama_vocab_eos(vocab) << ",\"eot\":" << llama_vocab_eot(vocab) << ",\"nl\":"
            << llama_vocab_nl(vocab) << "},";
    }

    // Get chat template from model metadata
    const char *chat_template = llama_model_chat_template(g_model, nullptr);
    if (chat_template && *chat_template != '\0') {
        oss << R"("chat_template":")" << json_escape(chat_template) << "\",";
    } else {
        oss << "\"chat_template\":null,";
    }

    oss << R"("system":")" << json_escape(llama_print_system_info()) << "\"";
    oss << "}";
    const std::string out = oss.str();
    return env->NewStringUTF(out.c_str());
}

// -----------------------------------------------------------------------------
// Chat templating & token utils
// -----------------------------------------------------------------------------
static std::string apply_chat_template(const llama_model *model, const std::string &system_msg,
                                       const std::string &user_msg, bool add_assistant) {
    const char *tmpl = nullptr;
    if (!g_chat_template_override.empty()) {
        tmpl = g_chat_template_override.c_str();
        LOGI("Using Custom Chat-Template");
    }
    else {
        tmpl = llama_model_chat_template(model, nullptr);
        LOGI("Using model chat template %s", tmpl ? "(ok)" : "(missing)");
    }

    if (!tmpl || *tmpl == '\0') {
        std::string out;
        if (!system_msg.empty()) {
            out += "System: ";
            out += system_msg;
            out += "\n";
        }
        out += "User: ";
        out += user_msg;
        out += "\nAssistant: ";
        return out;
    }

    std::vector<llama_chat_message> msgs;
    if (!system_msg.empty()) msgs.push_back({"system", system_msg.c_str()});
    msgs.push_back({"user", user_msg.c_str()});
    int32_t need = llama_chat_apply_template(tmpl, msgs.data(), (int32_t) msgs.size(),
                                             add_assistant, nullptr, 0);
    if (need < 0) need = -need;
    std::string out((size_t) need, '\0');
    int32_t written = llama_chat_apply_template(tmpl, msgs.data(), (int32_t) msgs.size(),
                                                add_assistant, out.data(), need);
    if (written < 0) written = -written;
    out.resize((size_t) written);
    return out;
}

static std::string detok_piece(const llama_vocab *vocab, llama_token tok) {
    char tmp[512];
    int n = llama_token_to_piece(vocab, tok, tmp, (int) sizeof(tmp), 0, true);
    if (n < 0) {
        std::string out;
        out.resize((size_t) (-n));
        llama_token_to_piece(vocab, tok, out.data(), -n, 0, true);
        return out;
    }
    return std::string(tmp, tmp + n);
}

// -----------------------------------------------------------------------------
// Generation (crash‑safe: chunked prompt decode + context guard)
// -----------------------------------------------------------------------------
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1core_NativeLib_nativeGenerateStream(JNIEnv *env, jobject /*this*/, jstring jprompt,
                                                    jint max_tokens, jobject jcallback) {
    if (!g_ctx || !g_model) {
        jni_on_error(env, jcallback, "Not initialized");
        return JNI_FALSE;
    }

    // Reset between turns
    {
        llama_memory_t mem = llama_get_memory(g_ctx);
        if (mem) llama_memory_clear(mem, /*data=*/true);
        if (!g_sampler) rebuild_sampler_chain(false); else llama_sampler_reset(g_sampler);
    }
    g_stop_requested = false;
    g_in_tool_json = false;
    g_tool_accum.clear();
    g_brace_depth = 0;

    const std::string user_prompt = jstr_to_utf8(env, jprompt);
    const llama_vocab *vocab = llama_model_get_vocab(g_model);

    // Grammar chain if tools enabled
    if (g_tools_enabled) enable_tool_grammar_if_needed(); else rebuild_sampler_chain(false);

    // Compose system prompt (+ tools preamble)
    std::string system_msg = g_system_prompt;
    if (g_tools_enabled) system_msg += std::string("\n") + tool_preamble(g_tools_json);

    std::string rendered = apply_chat_template(g_model, system_msg, user_prompt, true);
    LOGI("rendered.size=%d", (int) rendered.size());

    // Tokenize
    std::vector<llama_token> toks;
    {
        int32_t guess = (int32_t) rendered.size() + 8;
        toks.resize((size_t) guess);
        int32_t n = llama_tokenize(vocab, rendered.c_str(), (int32_t) rendered.size(), toks.data(),
                                   (int32_t) toks.size(), true, true);
        if (n < 0) {
            toks.resize((size_t) (-n));
            n = llama_tokenize(vocab, rendered.c_str(), (int32_t) rendered.size(), toks.data(),
                               (int32_t) toks.size(), true, true);
        }
        if (n < 0) {
            jni_on_error(env, jcallback, "tokenize failed");
            return JNI_FALSE;
        }
        toks.resize((size_t) n);
        LOGI("prompt toks = %d", (int) toks.size());
    }

    // Clamp generation to available context (leave headroom of 8)
    int32_t room = g_ctx_size - (int32_t) toks.size() - 8;
    if (room <= 0) {
        jni_on_error(env, jcallback, "context overflow before generation (reduce prompt)");
        flush_utf8_carry(env, jcallback);
        jni_on_done(env, jcallback);
        return JNI_TRUE;
    }
    int32_t to_gen = std::min<int32_t>(max_tokens > 0 ? max_tokens : 128, room);

    // Feed prompt in chunks <= g_n_batch
    if (!decode_tokens_chunked(g_ctx, toks, /*start_pos=*/0, g_n_batch, env, jcallback)) {
        flush_utf8_carry(env, jcallback);
        jni_on_done(env, jcallback);
        return JNI_TRUE;
    }

    // Streaming loop
    llama_batch one = llama_batch_init(1, 0, 1);
    int32_t cur_pos = (int32_t) toks.size();
    const llama_token eos = llama_vocab_eos(vocab);
    const llama_token eot = llama_vocab_eot(vocab);

    for (int i = 0; i < to_gen; ++i, ++cur_pos) {
        if (g_stop_requested) {
            LOGI("Generation stopped by user");
            break;
        }
        llama_token tok = llama_sampler_sample(g_sampler, g_ctx, -1);
        llama_sampler_accept(g_sampler, tok);
        if (i == 0 && (tok == eos || tok == eot)) {
            llama_token sp[4];
            int nsp = llama_tokenize(vocab, " ", 1, sp, 4, true, true);
            if (nsp > 0) tok = sp[0];
        }
        if (tok == eos || tok == eot) break;

        std::string piece = detok_piece(vocab, tok);
        LOGI("TOKENS :: %s :: %d", piece.c_str(), tok);
        bool completed = false;
        if (g_tools_enabled) completed = maybe_collect_tool_json_chunk(piece);
        if (completed) {
            if (looks_like_toolcall_json(g_tool_accum)) {
                std::string name = "tool";
                size_t p = g_tool_accum.find("\"name\"");
                if (p != std::string::npos) {
                    p = g_tool_accum.find('"', g_tool_accum.find(':', p) + 1);
                    size_t q = g_tool_accum.find('"', p + 1);
                    if (p != std::string::npos && q != std::string::npos && q > p)
                        name = g_tool_accum.substr(p + 1, q - p - 1);
                }
                jni_on_toolcall(env, jcallback, name, g_tool_accum);
                g_in_tool_json = false;
                g_tool_accum.clear();
                break; // stop this turn; app will run tool and call again
            } else {
                g_in_tool_json = false;
                g_tool_accum.clear();
            }
        }
        if (!(g_tools_enabled && g_in_tool_json)) jni_on_token(env, jcallback, piece);

        one.n_tokens = 1;
        one.token[0] = tok;
        one.pos[0] = cur_pos;
        one.n_seq_id[0] = 1;
        one.seq_id[0][0] = 0;
        one.logits[0] = true;
        int rc = llama_decode(g_ctx, one);
        if (rc != 0) {
            if (rc == 1)
                jni_on_error(env, jcallback, "decode failed during generation: no KV slot");
            else jni_on_error(env, jcallback, "decode failed during generation");
            break;
        }
        if (env->ExceptionCheck()) {
            LOGE("Java exception during callback");
            env->ExceptionClear();
            break;
        }
    }

    llama_batch_free(one);
    flush_utf8_carry(env, jcallback);
    jni_on_done(env, jcallback);
    return JNI_TRUE;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_mp_ai_1core_NativeLib_nativeGetStateSize(JNIEnv *env, jobject thiz) {
    if (!g_ctx) {
        LOGE("No active context");
        return 0;
    }
    return llama_state_get_size(g_ctx);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1core_NativeLib_nativeLoadStateData(JNIEnv *env, jobject thiz, jbyteArray arr) {
    if (!g_ctx) {
        LOGE("No active context");
        return JNI_FALSE;
    }
    jbyte *buf = env->GetByteArrayElements(arr, nullptr);
    size_t sz = (size_t) env->GetArrayLength(arr);
    size_t nbytes = llama_state_set_data(g_ctx, (const uint8_t *) buf, sz);
    env->ReleaseByteArrayElements(arr, buf, 0);
    return nbytes == sz ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1core_NativeLib_nativeLoadStateFile(JNIEnv *env, jobject thiz, jstring jpath) {
    const char *path = env->GetStringUTFChars(jpath, nullptr);
    bool ok = false;

    if (g_ctx && path) {
        llama_file file(path, "rb");

        const uint32_t magic = file.read_u32();
        const uint32_t version = file.read_u32();

        /* accept only valid session files */
        if (magic == LLAMA_SESSION_MAGIC && version == LLAMA_SESSION_VERSION) {
            const size_t n_token_count = file.read_u32();
            std::vector<llama_token> tokens(n_token_count);
            size_t n_token_read = 0;
            ok = llama_state_load_file(g_ctx, path, tokens.data(), tokens.size(), &n_token_read);
        }
    }

    env->ReleaseStringUTFChars(jpath, path);
    return ok ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1core_NativeLib_nativeSaveStateFile(JNIEnv *env, jobject thiz, jstring jpath) {
    const char *path = env->GetStringUTFChars(jpath, nullptr);
    bool ok = false;

    if (g_ctx && path) {
        ok = llama_state_save_file(g_ctx, path, nullptr, 0);
    }

    env->ReleaseStringUTFChars(jpath, path);
    return ok ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_mp_ai_1core_NativeLib_nativeGetStateData(JNIEnv *env, jobject thiz) {
    if (!g_ctx) { /* no context -> nothing to return */
        return nullptr;
    }

    /* get how much data the state represents */
    size_t sz = llama_state_get_size(g_ctx);
    if (sz == 0) { /* empty state */
        return nullptr;
    }

    /* allocate a Java byte[] of the exact size */
    jbyteArray arr = env->NewByteArray(static_cast<jsize>(sz));
    if (!arr) { /* allocation failed */
        return nullptr;
    }

    /* obtain a writable buffer from the Java array */
    jbyte *buf = env->GetByteArrayElements(arr, nullptr);
    /* write the state into that buffer */
    llama_state_get_data(g_ctx, reinterpret_cast<uint8_t *>(buf), sz);
    /* release the buffer back to the JVM (0 = copy back, no free) */
    env->ReleaseByteArrayElements(arr, buf, 0);

    return arr;
}



// Also add a JNI function to expose this info to Kotlin
extern "C" JNIEXPORT jstring JNICALL
Java_com_mp_ai_1core_NativeLib_nativeGetBackendInfo(JNIEnv *env, jobject) {
    std::ostringstream info;

    info << "Backend Info:\n";

#ifdef GGML_USE_OPENCL
    info << "OpenCL: COMPILED IN\n";
#else
    info << "OpenCL: NOT COMPILED\n";
#endif

#ifdef GGML_OPENCL
    info << "GGML_OPENCL: DEFINED\n";
#else
    info << "GGML_OPENCL: NOT DEFINED\n";
#endif

    if (g_model) {
        info << "Model loaded: YES\n";
        info << "Layers: " << llama_model_n_layer(g_model) << "\n";
    } else {
        info << "Model loaded: NO\n";
    }

    if (g_ctx) {
        info << "Context created: YES\n";
    } else {
        info << "Context created: NO\n";
    }

    return env->NewStringUTF(info.str().c_str());
}

// Fixed initialization function with proper return type
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1core_NativeLib_nativeInitForEmbeddings(JNIEnv *env, jobject, jstring jpath,
                                                       jint jthreads, jint gpuLayers,
                                                       jboolean useMMAP, jint ctxSize) {
    if (!env || !jpath) {
        LOGE("Invalid JNI parameters");
        return JNI_FALSE;
    }

    std::lock_guard<std::mutex> _lock(g_init_mtx);

    // Convert Java string
    const char *cstr = env->GetStringUTFChars(jpath, nullptr);
    if (!cstr) {
        LOGE("Failed to get path string");
        return JNI_FALSE;
    }

    const std::string path(cstr);
    env->ReleaseStringUTFChars(jpath, cstr);

    LOGI("Initializing model for embeddings: %s", path.c_str());

    // Clean up existing resources
    free_everything();
    llama_backend_init();

    const int physCores = count_physical_cores();
    int gpu_layers = gpuLayers;
    if (gpu_layers < 0) gpu_layers = 0;
    if (gpu_layers > 32) gpu_layers = 32;  // More reasonable upper limit

    // Model parameters
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = gpu_layers;
    mparams.use_mmap = useMMAP;
    mparams.use_mlock = false;
    mparams.check_tensors = true;

    LOGI("Loading model with %d GPU layers", gpu_layers);
    g_model = llama_model_load_from_file(path.c_str(), mparams);
    if (!g_model) {
        LOGE("Failed to load model: %s", path.c_str());
        free_everything();
        return JNI_FALSE;
    }

    LOGI("Model loaded successfully");

    // Context parameters - CRITICAL: Enable embeddings
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = ctxSize > 0 ? ctxSize : 2048;
    cparams.n_batch = 256;
    cparams.n_ubatch = 64;
    cparams.offload_kqv = true;  // Can help with GPU acceleration
    cparams.n_seq_max = 1;
    cparams.n_threads = jthreads > 0 ? jthreads : physCores;
    cparams.n_threads_batch = cparams.n_threads;
    cparams.no_perf = true;

    // MOST IMPORTANT: Enable embeddings
    cparams.embeddings = true;

    LOGI("Creating context with embeddings enabled");
    g_ctx = llama_init_from_model(g_model, cparams);
    if (!g_ctx) {
        LOGE("Failed to create context");
        free_everything();
        return JNI_FALSE;
    }

    // Persist runtime knobs
    g_ctx_size = cparams.n_ctx;
    g_n_batch = cparams.n_batch;

    // Verify embeddings are actually enabled
    int32_t n_embd = llama_model_n_embd(g_model);
    if (n_embd <= 0) {
        LOGE("Model does not support embeddings (n_embd = %d)", n_embd);
        free_everything();
        return JNI_FALSE;
    }

    LOGI("Embedding model initialized successfully:");
    LOGI("  - GPU layers: %d", gpu_layers);
    LOGI("  - Context size: %d", cparams.n_ctx);
    LOGI("  - Batch size: %d", cparams.n_batch);
    LOGI("  - Embedding dim: %d", n_embd);
    LOGI("  - Embeddings enabled: true");

    return JNI_TRUE;
}


extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_mp_ai_1core_NativeLib_embed(JNIEnv *env, jobject /*this*/, jstring jtext) {
    if (!g_ctx || !g_model) {
        LOGE("Embed: Context or model not initialized");
        return nullptr;
    }

    const std::string text = jstr_to_utf8(env, jtext);
    if (text.empty()) {
        LOGE("Embed: Empty input text");
        return nullptr;
    }

    LOGI("Embed: Processing text (length=%d)", (int) text.size());

    // Get vocab from model (matching nativeGenerateStream pattern)
    const llama_vocab *vocab = llama_model_get_vocab(g_model);
    if (!vocab) {
        LOGE("Embed: Failed to get vocab");
        return nullptr;
    }

    // Clear context state before embedding (following nativeGenerateStream pattern)
    {
        llama_memory_t mem = llama_get_memory(g_ctx);
        if (mem) {
            llama_memory_clear(mem, /*data=*/true);
        }
    }

    // Tokenize text (following nativeGenerateStream pattern exactly)
    std::vector<llama_token> toks;
    {
        int32_t guess = (int32_t) text.size() + 8;
        toks.resize((size_t) guess);
        int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(),
                                   (int32_t) toks.size(), true, true);
        if (n < 0) {
            toks.resize((size_t) (-n));
            n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(),
                               (int32_t) toks.size(), true, true);
        }
        if (n < 0) {
            LOGE("Embed: Tokenization failed");
            return nullptr;
        }
        toks.resize((size_t) n);
        LOGI("Embed: Tokenized to %d tokens", (int) toks.size());
    }

    if (toks.empty()) {
        LOGE("Embed: No tokens generated");
        return nullptr;
    }

    // Check context capacity
    if ((int32_t) toks.size() >= g_ctx_size) {
        LOGE("Embed: Text too long for context (%d tokens, max %d)", (int) toks.size(), g_ctx_size);
        return nullptr;
    }

    // Create batch for ALL tokens - this is crucial for embeddings
    llama_batch batch = llama_batch_init((int32_t) toks.size(), 0, 1);
    if (!batch.token) {
        LOGE("Embed: Failed to create batch");
        return nullptr;
    }

    // Fill batch - CRITICAL: ALL tokens must be marked correctly for embeddings
    for (int32_t i = 0; i < (int32_t) toks.size(); ++i) {
        batch.token[i] = toks[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        // For embeddings: all tokens should have logits = true or at least the last one
        batch.logits[i] = (i == (int32_t) toks.size() - 1); // Only last token needs logits
    }
    batch.n_tokens = (int32_t) toks.size();

    LOGI("Embed: Created batch with %d tokens, decoding...", batch.n_tokens);

    // Decode the entire batch at once
    int rc = llama_decode(g_ctx, batch);
    if (rc != 0) {
        LOGE("Embed: Decode failed with code %d", rc);
        llama_batch_free(batch);
        return nullptr;
    }

    LOGI("Embed: Decode completed successfully");
    llama_batch_free(batch);

    // Get embeddings dimension
    int32_t n_embd = llama_model_n_embd(g_model);
    if (n_embd <= 0) {
        LOGE("Embed: Invalid embedding dimension: %d", n_embd);
        return nullptr;
    }

    // Try to get embeddings using different methods
    const float *embeddings = nullptr;

    // Method 1: Try sequence-specific embeddings first
    embeddings = llama_get_embeddings_seq(g_ctx, 0);
    if (embeddings) {
        LOGI("Embed: Got embeddings via llama_get_embeddings_seq");
    } else {
        // Method 2: Try regular embeddings
        embeddings = llama_get_embeddings(g_ctx);
        if (embeddings) {
            LOGI("Embed: Got embeddings via llama_get_embeddings");
        } else {
            LOGE("Embed: Both embedding methods failed");

            // Debug info
            LOGI("Embed: Debug - n_embd=%d, context valid=%s", n_embd, g_ctx ? "yes" : "no");
            return nullptr;
        }
    }

    LOGI("Embed: Got embeddings, dimension=%d", n_embd);

    // Verify embeddings are not all zeros (common issue)
    bool all_zero = true;
    for (int32_t i = 0; i < std::min(n_embd, 10); ++i) {
        if (embeddings[i] != 0.0f) {
            all_zero = false;
            break;
        }
    }

    if (all_zero) {
        LOGI("Embed: Warning - embeddings appear to be all zeros");
    } else {
        LOGI("Embed: Embeddings look valid (first few values non-zero)");
    }

    // Create Java float array
    jfloatArray result = env->NewFloatArray(n_embd);
    if (!result) {
        LOGE("Embed: Failed to create Java array");
        return nullptr;
    }

    // Copy embeddings to Java array
    env->SetFloatArrayRegion(result, 0, n_embd, embeddings);

    if (env->ExceptionCheck()) {
        LOGE("Embed: Exception while copying embeddings");
        env->ExceptionClear();
        return nullptr;
    }

    LOGI("Embed: Successfully created embedding array");
    return result;
}