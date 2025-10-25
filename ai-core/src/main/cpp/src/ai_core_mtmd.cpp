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

// ------------------------------------------------------------
//  JNI: Generate with image input (streaming)
// ------------------------------------------------------------
extern "C"
JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1core_MtmdLib_nativeGenerateStreamWithImage(
        JNIEnv*                env,
        jobject,
        jstring                jprompt,
        jbyteArray             jimage_data,
        jint                   image_width,
        jint                   image_height,
        jint                   max_tokens,
        jobject                jcallback)
{
    /* --- 1️⃣  Sanity checks --------------------------------- */
    if (!g_state.is_ready() || !g_mtmd_state.is_ready()) {
        jni::on_error(env, jcallback,
                      "Model or MTMD not initialized");
        return JNI_FALSE;
    }

    if (!mtmd_support_vision(g_mtmd_state.ctx)) {
        jni::on_error(env, jcallback,
                      "Vision not supported by this model");
        return JNI_FALSE;
    }

    /* --- 2️⃣  Prepare for new generation ------------------- */
    g_state.prepare_for_generation();

    /* --- 3️⃣  Read prompt and format it -------------------- */
    std::string prompt_str = utf8::from_jstring(env, jprompt);

    // Add media marker if not present
    std::string media_marker = mtmd_default_marker();
    if (prompt_str.find(media_marker) == std::string::npos) {
        // Place media marker at the beginning for vision models
        prompt_str = media_marker + "\n" + prompt_str;
    }

    LOG_INFO("Multimodal: prompt_len=%zu, image=%dx%d, marker='%s'",
             prompt_str.size(), image_width, image_height, media_marker.c_str());

    /* --- 4️⃣  Pull raw RGB bytes from Java ----------------- */
    jbyte* img_bytes = env->GetByteArrayElements(jimage_data, nullptr);
    jsize  img_len   = env->GetArrayLength(jimage_data);

    if (img_bytes == nullptr || img_len <= 0) {
        jni::on_error(env, jcallback, "Invalid image data");
        return JNI_FALSE;
    }

    // Verify image data size matches expected dimensions (RGB = 3 bytes per pixel)
    jsize expected_size = static_cast<jsize>(image_width * image_height * 3);
    if (img_len != expected_size) {
        LOG_WARN("Image data size mismatch: got %d, expected %d", img_len, expected_size);
    }

    /* --- 5️⃣  Create mtmd_bitmap instance ------------------ */
    mtmd_bitmap* bitmap = mtmd_bitmap_init(
            static_cast<uint32_t>(image_width),
            static_cast<uint32_t>(image_height),
            reinterpret_cast<const unsigned char*>(img_bytes));

    env->ReleaseByteArrayElements(jimage_data, img_bytes, JNI_ABORT);

    if (bitmap == nullptr) {
        jni::on_error(env, jcallback, "Failed to create bitmap");
        return JNI_FALSE;
    }

    // Set unique ID for KV cache management
    mtmd_bitmap_set_id(bitmap, "user_image_0");
    LOG_INFO("Bitmap created: %dx%d, id='user_image_0'",
             mtmd_bitmap_get_nx(bitmap), mtmd_bitmap_get_ny(bitmap));

    /* --- 6️⃣  Tokenize prompt + image --------------------- */
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    if (chunks == nullptr) {
        mtmd_bitmap_free(bitmap);
        jni::on_error(env, jcallback, "Failed to initialize input chunks");
        return JNI_FALSE;
    }

    // Configure text input
    // Note: add_special should typically be false if you've already formatted
    // the prompt with chat template. Set to true if using raw text.
    mtmd_input_text text_input = {
            const_cast<char*>(prompt_str.c_str()),
            false,      // add_special - false since we're handling formatting
            true        // parse_special - true to parse media markers
    };

    const mtmd_bitmap* bm_arr[] = { bitmap };
    int32_t tokenize_rc = mtmd_tokenize(
            g_mtmd_state.ctx,
            chunks,
            &text_input,
            bm_arr,
            1);

    if (tokenize_rc != 0) {
        LOG_ERROR("Tokenization failed with code %d", tokenize_rc);
        LOG_ERROR("Prompt: '%s'", prompt_str.c_str());
        LOG_ERROR("Media marker: '%s'", media_marker.c_str());
        mtmd_bitmap_free(bitmap);
        mtmd_input_chunks_free(chunks);
        jni::on_error(env, jcallback, "Tokenization failed");
        return JNI_FALSE;
    }

    /* --- 7️⃣  Verify context space ------------------------ */
    size_t n_tokens = mtmd_helper_get_n_tokens(chunks);
    llama_pos n_pos = mtmd_helper_get_n_pos(chunks);
    LOG_INFO("Tokenized: %zu tokens, %d positions", n_tokens, n_pos);

    int32_t available = g_state.ctx_size - static_cast<int32_t>(n_tokens) - 8;
    if (available <= 0) {
        LOG_ERROR("Context overflow: ctx_size=%d, n_tokens=%zu, available=%d",
                  g_state.ctx_size, n_tokens, available);
        mtmd_bitmap_free(bitmap);
        mtmd_input_chunks_free(chunks);
        jni::on_error(env, jcallback,
                      "Context overflow – image+prompt too large");
        return JNI_TRUE;
    }

    /* --- 8️⃣  Determine tokens to generate ---------------- */
    int32_t to_generate = static_cast<int32_t>(max_tokens > 0 ? max_tokens : 128);
    to_generate = std::min(to_generate, available);
    LOG_INFO("Will generate up to %d tokens (available: %d)", to_generate, available);

    /* --- 9️⃣  Evaluate image + prompt chunks -------------- */
    llama_pos new_n_past = 0;
    int32_t eval_rc = mtmd_helper_eval_chunks(
            g_mtmd_state.ctx,       // vision context
            g_state.ctx,            // text model context
            chunks,
            0,                      // n_past (starting position)
            0,                      // seq_id
            g_state.batch_size,     // n_batch
            true,                   // logits_last - MUST be true for generation!
            &new_n_past);           // output: position after evaluation

// Cleanup bitmap and chunks - no longer needed
    mtmd_bitmap_free(bitmap);
    mtmd_input_chunks_free(chunks);

// Check evaluation result
// mtmd_helper_eval_chunks returns 0 on SUCCESS, non-zero on ERROR
    if (eval_rc != 0) {  // ✅ CORRECT: non-zero is error
        LOG_ERROR("Chunk evaluation failed with code %d", eval_rc);
        jni::on_error(env, jcallback, "Failed to process image+prompt");
        return JNI_TRUE;
    }

    LOG_INFO("Image+prompt processed successfully (new_n_past=%d)", new_n_past);
    LOG_INFO("Starting generation of %d tokens...", to_generate);

    /* --- 🔟  Generation loop ----------------------------- */
/* --- 🔟  Generation loop ----------------------------- */
    const llama_vocab* vocab = llama_model_get_vocab(g_state.model);
    llama_token eos = llama_vocab_eos(vocab);
    llama_token eot = llama_vocab_eot(vocab);

// Initialize single-token batch for autoregressive decoding
    llama_batch batch = llama_batch_init(1, 0, 1);

    int generated_count = 0;
    for (int i = 0; i < to_generate; ++i) {
        // Sample next token
        llama_token tok = llama_sampler_sample(g_state.sampler, g_state.ctx, -1);
        llama_sampler_accept(g_state.sampler, tok);

        // Check for end-of-sequence
        if (tok == eos || tok == eot) {
            LOG_INFO("EOS reached after %d tokens", generated_count);
            break;
        }

        // Detokenize and send to callback
        std::string piece = g_state.detokenize_single(tok);
        jni::on_token(env, jcallback, piece.c_str());
        generated_count++;

        // Prepare batch for next decode step (manual method)
        batch.n_tokens = 1;
        batch.token[0] = tok;
        batch.pos[0] = new_n_past + i;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;

        // Decode the token
        int decode_rc = llama_decode(g_state.ctx, batch);
        if (decode_rc != 0) {
            LOG_ERROR("Decode failed with code %d at token %d", decode_rc, i);
            jni::on_error(env, jcallback, "Decode failed");
            break;
        }

        // Check for Java exceptions
        if (env->ExceptionCheck()) {
            LOG_ERROR("Java callback threw exception");
            env->ExceptionClear();
            break;
        }
    }

// Cleanup
    llama_batch_free(batch);

// Flush any remaining UTF-8 data and signal completion
    utf8::flush_carry(env, jcallback);
    jni::on_done(env, jcallback);

    LOG_INFO("Multimodal generation completed (%d tokens generated)", generated_count);

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