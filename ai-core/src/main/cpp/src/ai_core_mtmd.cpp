/*=============================================================
 *   ai_core_mtmd.cpp
 *=============================================================
 *
 *  Android JNI interface for multimodal (image+text) models.
 *  - Extends ai_core.cpp with vision capabilities
 *  - Supports models like LFM2-VL, LLaVA, Qwen-VL, etc.
 *  - Handles image preprocessing and encoding
 *  - Thread-safe bitmap and chunk management
 *============================================================*/

#include "state/model_state.h"
#include "state/global_state.h"
#include "utils/jni_utils.h"
#include "utils/utf8_utils.h"
#include "utils/logger.h"

#include "llama.h"
#include "tools/mtmd/mtmd.h"
#include "tools/mtmd/mtmd-helper.h"


#include <jni.h>
#include <string>
#include <vector>
#include <mutex>
#include <sstream>
#include <memory>

/*  --------------------------------------------------------------
 *      Global multimodal state
 *  -------------------------------------------------------------- */
struct MTMDState {
    mtmd_context* ctx = nullptr;
    std::string mmproj_path;
    bool initialized = false;

    void release() {
        if (ctx) {
            mtmd_free(ctx);
            ctx = nullptr;
        }
        initialized = false;
        mmproj_path.clear();
    }

    bool is_ready() const {
        return initialized && ctx != nullptr;
    }
};

static MTMDState g_mtmd_state;
static std::mutex g_mtmd_mtx;

/*  --------------------------------------------------------------
 *      JNI: Initialize multimodal context
 *  -------------------------------------------------------------- */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1core_MtmdLib_nativeInitMTMD(JNIEnv* env, jobject,
                                              jstring jmmproj_path,
                                              jboolean use_gpu,
                                              jint n_threads) {
    std::lock_guard<std::mutex> lk(g_mtmd_mtx);

    if (!g_state.is_ready()) {
        LOG_ERROR("Base model must be initialized before MTMD");
        return JNI_FALSE;
    }

    const std::string mmproj_path = utf8::from_jstring(env, jmmproj_path);
    LOG_INFO("Initializing MTMD with projector: %s", mmproj_path.c_str());

    // Release any existing context
    g_mtmd_state.release();

    // Setup MTMD context parameters
    mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu = static_cast<bool>(use_gpu);
    params.print_timings = false;
    params.n_threads = static_cast<int>(n_threads > 0 ? n_threads : 4);
    params.verbosity = GGML_LOG_LEVEL_INFO;
    params.media_marker = mtmd_default_marker(); // "<__media__>"

    // Initialize MTMD context
    g_mtmd_state.ctx = mtmd_init_from_file(
            mmproj_path.c_str(),
            g_state.model,
            params
    );

    if (!g_mtmd_state.ctx) {
        LOG_ERROR("Failed to initialize MTMD context");
        return JNI_FALSE;
    }

    g_mtmd_state.mmproj_path = mmproj_path;
    g_mtmd_state.initialized = true;

    // Log capabilities
    LOG_INFO("MTMD initialized successfully");
    LOG_INFO("Vision support: %d", mtmd_support_vision(g_mtmd_state.ctx));
    LOG_INFO("Audio support: %d", mtmd_support_audio(g_mtmd_state.ctx));
    LOG_INFO("Default marker: %s", mtmd_default_marker());

    return JNI_TRUE;
}

/*  --------------------------------------------------------------
 *      JNI: Release multimodal resources
 *  -------------------------------------------------------------- */
extern "C" JNIEXPORT void JNICALL
Java_com_mp_ai_1core_MtmdLib_nativeReleaseMTMD(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lk(g_mtmd_mtx);
    g_mtmd_state.release();
    LOG_INFO("MTMD resources released");
}

/*  --------------------------------------------------------------
 *      JNI: Check if MTMD is initialized
 *  -------------------------------------------------------------- */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1core_MtmdLib_nativeIsMTMDReady(JNIEnv*, jobject) {
    return g_mtmd_state.is_ready() ? JNI_TRUE : JNI_FALSE;
}

/*  --------------------------------------------------------------
 *      JNI: Get MTMD capabilities
 *  -------------------------------------------------------------- */
extern "C" JNIEXPORT jstring JNICALL
Java_com_mp_ai_1core_MtmdLib_nativeGetMTMDInfo(JNIEnv* env, jobject) {
    if (!g_mtmd_state.is_ready()) {
        return env->NewStringUTF("{}");
    }

    std::ostringstream json;
    json << "{";
    json << "\"vision_support\":" << (mtmd_support_vision(g_mtmd_state.ctx) ? "true" : "false") << ",";
    json << "\"audio_support\":" << (mtmd_support_audio(g_mtmd_state.ctx) ? "true" : "false") << ",";
    json << "\"default_marker\":\"" << mtmd_default_marker() << "\",";
    json << "\"use_non_causal\":" << (mtmd_decode_use_non_causal(g_mtmd_state.ctx) ? "true" : "false") << ",";
    json << "\"use_mrope\":" << (mtmd_decode_use_mrope(g_mtmd_state.ctx) ? "true" : "false");

    int audio_bitrate = mtmd_get_audio_bitrate(g_mtmd_state.ctx);
    if (audio_bitrate > 0) {
        json << ",\"audio_bitrate\":" << audio_bitrate;
    }

    json << "}";

    return env->NewStringUTF(json.str().c_str());
}

/*  --------------------------------------------------------------
 *      JNI: Generate with image input (streaming)
 *  -------------------------------------------------------------- */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1core_MtmdLib_nativeGenerateStreamWithImage(
        JNIEnv* env, jobject,
        jstring jprompt,
        jbyteArray jimage_data,
        jint image_width,
        jint image_height,
        jint max_tokens,
        jobject jcallback) {

    if (!g_state.is_ready() || !g_mtmd_state.is_ready()) {
        jni::on_error(env, jcallback, "Model or MTMD not initialized");
        return JNI_FALSE;
    }

    if (!mtmd_support_vision(g_mtmd_state.ctx)) {
        jni::on_error(env, jcallback, "Vision not supported by this model");
        return JNI_FALSE;
    }

    // Prepare for new generation
    g_state.prepare_for_generation();

    const std::string user_prompt = utf8::from_jstring(env, jprompt);
    LOG_INFO("Multimodal generation: prompt_len=%zu, image=%dx%d",
             user_prompt.size(), image_width, image_height);

    // Extract image data from Java byte array
    jbyte* img_bytes = env->GetByteArrayElements(jimage_data, nullptr);
    jsize img_len = env->GetArrayLength(jimage_data);

    if (!img_bytes || img_len <= 0) {
        jni::on_error(env, jcallback, "Invalid image data");
        return JNI_FALSE;
    }

    // Create bitmap from raw RGB data
    // Expected format: RGBRGBRGB... (3 bytes per pixel)
    mtmd_bitmap* bitmap = mtmd_bitmap_init(
            static_cast<uint32_t>(image_width),
            static_cast<uint32_t>(image_height),
            reinterpret_cast<const unsigned char*>(img_bytes)
    );

    env->ReleaseByteArrayElements(jimage_data, img_bytes, JNI_ABORT);

    if (!bitmap) {
        jni::on_error(env, jcallback, "Failed to create bitmap");
        return JNI_FALSE;
    }

    // Set optional bitmap ID for KV cache tracking
    mtmd_bitmap_set_id(bitmap, "user_image_0");

    // Tokenize prompt with image
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    mtmd_input_text text_input = {
            user_prompt.c_str(),
            true,  // add_special
            true   // parse_special
    };

    const mtmd_bitmap* bitmaps[] = { bitmap };
    int32_t tokenize_result = mtmd_tokenize(
            g_mtmd_state.ctx,
            chunks,
            &text_input,
            bitmaps,
            1
    );

    if (tokenize_result != 0) {
        LOG_ERROR("Tokenization failed with code: %d", tokenize_result);
        mtmd_bitmap_free(bitmap);
        mtmd_input_chunks_free(chunks);
        jni::on_error(env, jcallback, "Tokenization failed");
        return JNI_FALSE;
    }

    // Get total token count
    size_t n_tokens = mtmd_helper_get_n_tokens(chunks);
    llama_pos n_pos = mtmd_helper_get_n_pos(chunks);
    LOG_INFO("Tokenized: %zu tokens, %d positions", n_tokens, n_pos);

    // Check context size
    int32_t available = g_state.ctx_size - static_cast<int32_t>(n_tokens) - 8;
    if (available <= 0) {
        mtmd_bitmap_free(bitmap);
        mtmd_input_chunks_free(chunks);
        jni::on_error(env, jcallback, "Context overflow - image+prompt too large");
        return JNI_TRUE;
    }

    auto to_generate = static_cast<int32_t>(max_tokens > 0 ? max_tokens : 128);
    to_generate = std::min(to_generate, available);

    // Evaluate chunks (image + text)
    llama_pos new_n_past = 0;
    int32_t eval_result = mtmd_helper_eval_chunks(
            g_mtmd_state.ctx,
            g_state.ctx,
            chunks,
            0,              // n_past
            0,              // seq_id
            g_state.batch_size,
            false,          // logits_last
            &new_n_past
    );

    // Clean up
    mtmd_bitmap_free(bitmap);
    mtmd_input_chunks_free(chunks);

    if (eval_result != 0) {
        LOG_ERROR("Chunk evaluation failed with code: %d", eval_result);
        jni::on_error(env, jcallback, "Failed to process image+text");
        return JNI_TRUE;
    }

    LOG_INFO("Image+text processed, starting generation (n_past=%d)", new_n_past);

    /* ---------------------------------------------------------
     *  Generation loop - same as text-only
     * -------------------------------------------------------- */
    const llama_vocab* vocab = llama_model_get_vocab(g_state.model);
    llama_token eos = llama_vocab_eos(vocab);
    llama_token eot = llama_vocab_eot(vocab);

    llama_batch single = llama_batch_init(1, 0, 1);

    for (int i = 0; i < to_generate; ++i) {
        // Sample next token
        llama_token tok = llama_sampler_sample(g_state.sampler, g_state.ctx, -1);
        llama_sampler_accept(g_state.sampler, tok);

        if (tok == eos || tok == eot) break;

        // Decode token
        std::string piece = g_state.detokenize_single(tok);
        jni::on_token(env, jcallback, piece);

        // Prepare next batch
        single.n_tokens = 1;
        single.token[0] = tok;
        single.pos[0] = new_n_past + i;
        single.n_seq_id[0] = 1;
        single.seq_id[0][0] = 0;
        single.logits[0] = true;

        if (llama_decode(g_state.ctx, single) != 0) {
            jni::on_error(env, jcallback, "Decode failed during generation");
            break;
        }

        if (env->ExceptionCheck()) {
            LOG_ERROR("Java exception during callback");
            env->ExceptionClear();
            break;
        }
    }

    llama_batch_free(single);
    utf8::flush_carry(env, jcallback);
    jni::on_done(env, jcallback);

    LOG_INFO("Multimodal generation complete");
    return JNI_TRUE;
}

/*  --------------------------------------------------------------
 *      JNI: Load image from file (helper for testing)
 *  -------------------------------------------------------------- */
extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_mp_ai_1core_MtmdLib_nativeLoadImageFromFile(
        JNIEnv* env, jobject,
        jstring jpath) {

    if (!g_mtmd_state.is_ready()) {
        return nullptr;
    }

    const std::string path = utf8::from_jstring(env, jpath);
    mtmd_bitmap* bitmap = mtmd_helper_bitmap_init_from_file(
            g_mtmd_state.ctx,
            path.c_str()
    );

    if (!bitmap) {
        LOG_ERROR("Failed to load image from: %s", path.c_str());
        return nullptr;
    }

    uint32_t nx = mtmd_bitmap_get_nx(bitmap);
    uint32_t ny = mtmd_bitmap_get_ny(bitmap);
    const unsigned char* data = mtmd_bitmap_get_data(bitmap);
    size_t n_bytes = mtmd_bitmap_get_n_bytes(bitmap);

    LOG_INFO("Loaded image: %dx%d, %zu bytes", nx, ny, n_bytes);

    // Copy data to Java byte array
    jbyteArray result = env->NewByteArray(static_cast<jsize>(n_bytes));
    if (result) {
        env->SetByteArrayRegion(result, 0, static_cast<jsize>(n_bytes),
                                reinterpret_cast<const jbyte*>(data));
    }

    mtmd_bitmap_free(bitmap);
    return result;
}

/*  --------------------------------------------------------------
 *      JNI: Get default media marker
 *  -------------------------------------------------------------- */
extern "C" JNIEXPORT jstring JNICALL
Java_com_mp_ai_1core_MtmdLib_nativeGetMediaMarker(JNIEnv* env, jobject) {
    return env->NewStringUTF(mtmd_default_marker());
}