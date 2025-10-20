#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>

// Include actual llama.cpp headers
#include "common.h"
#include "sampling.h"
#include "llama.h"
#include "tools/mtmd/mtmd.h"
#include "tools/mtmd/mtmd-helper.h"
#include "common/common.h"

#define LOG_TAG "MMNativeLib"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Global state
struct mm_state {
    llama_model* model = nullptr;
    llama_context* lctx = nullptr;
    mtmd_context* ctx_vision = nullptr;
    common_sampler* smpl = nullptr;
    llama_batch batch;
    llama_pos n_past = 0;
    int n_threads = 4;
    bool initialized = false;
};

static mm_state* g_state = nullptr;

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1core_MMNativeLib_nativeMMInit(
        JNIEnv* env,
        jobject,
        jstring jMainModelPath,
        jstring jMmModelPath,
        jint numThreads) {

    const char* mainModelPath = env->GetStringUTFChars(jMainModelPath, nullptr);
    const char* mmModelPath = env->GetStringUTFChars(jMmModelPath, nullptr);

    LOGI("=== Initializing MM ===");
    LOGI("Model: %s", mainModelPath);
    LOGI("MMProj: %s", mmModelPath);
    LOGI("Threads: %d", numThreads);

    // Cleanup existing state
    if (g_state) {
        if (g_state->smpl) common_sampler_free(g_state->smpl);
        if (g_state->batch.n_tokens > 0) llama_batch_free(g_state->batch);
        if (g_state->ctx_vision) mtmd_free(g_state->ctx_vision);
        if (g_state->lctx) llama_free(g_state->lctx);
        if (g_state->model) llama_model_free(g_state->model);
        delete g_state;
    }

    g_state = new mm_state();
    g_state->n_threads = numThreads;

    // Initialize llama backend
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    // Load model (CPU only)
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0; // keep LLaMA on CPU
    g_state->model = llama_model_load_from_file(mainModelPath, model_params);
    if (!g_state->model) {
        LOGE("Failed to load model");
        env->ReleaseStringUTFChars(jMainModelPath, mainModelPath);
        env->ReleaseStringUTFChars(jMmModelPath, mmModelPath);
        return JNI_FALSE;
    }
    LOGI("✓ Model loaded");

    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 4096;
    ctx_params.n_threads = numThreads;
    ctx_params.n_threads_batch = numThreads;
    g_state->lctx = llama_init_from_model(g_state->model, ctx_params);
    if (!g_state->lctx) {
        LOGE("Failed to create context");
        llama_model_free(g_state->model);
        env->ReleaseStringUTFChars(jMainModelPath, mainModelPath);
        env->ReleaseStringUTFChars(jMmModelPath, mmModelPath);
        return JNI_FALSE;
    }
    LOGI("✓ Context created");

    // Initialize vision context (GPU)
    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = false;
    mparams.n_threads = numThreads;
    mparams.verbosity = GGML_LOG_LEVEL_CONT;

    g_state->ctx_vision = mtmd_init_from_file(mmModelPath, g_state->model, mparams);
    if (!g_state->ctx_vision) {
        LOGE("Failed to load vision model");
        llama_free(g_state->lctx);
        llama_model_free(g_state->model);
        env->ReleaseStringUTFChars(jMainModelPath, mainModelPath);
        env->ReleaseStringUTFChars(jMmModelPath, mmModelPath);
        return JNI_FALSE;
    }


    int vision_layers = llama_model_n_layer(g_state->model);
    LOGI("Vision model layers: %d", vision_layers);


    LOGI("✓ Vision model loaded");

    // Initialize sampler
    common_params params;
    params.sampling.temp = 0.7f;
    params.sampling.top_k = 40;
    params.sampling.top_p = 0.9f;
    g_state->smpl = common_sampler_init(g_state->model, params.sampling);

    // Initialize batch
    g_state->batch = llama_batch_init(512, 0, 1);
    g_state->n_past = 0;
    g_state->initialized = true;

    env->ReleaseStringUTFChars(jMainModelPath, mainModelPath);
    env->ReleaseStringUTFChars(jMmModelPath, mmModelPath);

    LOGI("=== Initialization Complete ===");
    return JNI_TRUE;
}



JNIEXPORT void JNICALL
Java_com_mp_ai_1core_MMNativeLib_nativeMMGenerateStreaming(
        JNIEnv* env,
        jobject thiz,
        jstring jInput,
        jstring jImagePath,
        jint maxTokens,
        jobject jCallback) {

    if (!g_state || !g_state->initialized) {
        LOGE("MM not initialized");
        return;
    }

    const char* input = env->GetStringUTFChars(jInput, nullptr);
    const char* imagePath = env->GetStringUTFChars(jImagePath, nullptr);

    // --- Add system prompt ---
    static const std::string SYSTEM_PROMPT =
            "<|im_start|>system\n"
            "You are a helpful multimodal assistant. Provide concise and accurate responses based on the image and user input.\n"
            "<|im_end|>\n";

    std::string prompt = "here is an image: <__media__> describe it in simple 2 sentence.";
    //std::string prompt = SYSTEM_PROMPT + mtmd_default_marker() + "\n" + input;

    LOGI("Prepared prompt: %s", prompt.c_str());

    // ---- Image decode ----
    mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_file(g_state->ctx_vision, imagePath);
    if (!bmp) {
        LOGE("Failed to load image at path: %s", imagePath);
        env->ReleaseStringUTFChars(jInput, input);
        env->ReleaseStringUTFChars(jImagePath, imagePath);
        return;
    }

    // ---- Prepare text input ----
    mtmd_input_text text;
    text.text = prompt.c_str();
    text.add_special = true;
    text.parse_special = true;

    // ---- Init chunks ----
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    if (!chunks) {
        LOGE("Failed to initialize chunks");
        mtmd_bitmap_free(bmp);
        env->ReleaseStringUTFChars(jInput, input);
        env->ReleaseStringUTFChars(jImagePath, imagePath);
        return;
    }

    const mtmd_bitmap* bitmaps[1] = {bmp};

    if (mtmd_tokenize(g_state->ctx_vision, chunks, &text, bitmaps, 1) != 0) {
        LOGE("Tokenization failed");
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
        env->ReleaseStringUTFChars(jInput, input);
        env->ReleaseStringUTFChars(jImagePath, imagePath);
        return;
    }

    llama_pos new_n_past;
    if (mtmd_helper_eval_chunks(g_state->ctx_vision, g_state->lctx, chunks,
                                g_state->n_past, 0, 512, true, &new_n_past) != 0) {
        LOGE("Eval failed");
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
        env->ReleaseStringUTFChars(jInput, input);
        env->ReleaseStringUTFChars(jImagePath, imagePath);
        return;
    }

    g_state->n_past = new_n_past;

    const llama_vocab* vocab = llama_model_get_vocab(g_state->model);

    // ---- Token sampling loop ----
    jclass cbClass = env->GetObjectClass(jCallback);
    jmethodID onTokenMethod = env->GetMethodID(cbClass, "onToken", "(Ljava/lang/String;)V");
    jmethodID onCompleteMethod = env->GetMethodID(cbClass, "onComplete", "()V");

    std::vector<llama_token> generated_tokens;
    for (int i = 0; i < maxTokens; i++) {
        llama_token token_id = common_sampler_sample(g_state->smpl, g_state->lctx, -1);
        common_sampler_accept(g_state->smpl, token_id, true);

        if (llama_vocab_is_eog(vocab, token_id)) break;

        generated_tokens.push_back(token_id);

        common_batch_clear(g_state->batch);
        common_batch_add(g_state->batch, token_id, g_state->n_past, {0}, true);
        if (llama_decode(g_state->lctx, g_state->batch)) break;

        g_state->n_past++;

        std::string piece = common_token_to_piece(g_state->lctx, token_id);
        jstring jPiece = env->NewStringUTF(piece.c_str());
        env->CallVoidMethod(jCallback, onTokenMethod, jPiece);
        env->DeleteLocalRef(jPiece);
    }

    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bmp);

    env->ReleaseStringUTFChars(jInput, input);
    env->ReleaseStringUTFChars(jImagePath, imagePath);

    env->CallVoidMethod(jCallback, onCompleteMethod);
}


JNIEXPORT void JNICALL
Java_com_mp_ai_1core_MMNativeLib_nativeMMFree(JNIEnv*, jobject) {
    LOGI("Freeing MM state");

    if (g_state) {
        if (g_state->smpl) common_sampler_free(g_state->smpl);
        if (g_state->batch.n_tokens > 0) llama_batch_free(g_state->batch);
        if (g_state->ctx_vision) mtmd_free(g_state->ctx_vision);
        if (g_state->lctx) llama_free(g_state->lctx);
        if (g_state->model) llama_model_free(g_state->model);
        delete g_state;
        g_state = nullptr;
    }

    llama_backend_free();
    LOGI("MM state freed");
}

} // extern "C"