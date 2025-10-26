/*=============================================================
 *   ai_core_embed.cpp
 *=============================================================
 *
 *  Separate JNI entry point dedicated to embedding‑generation.  The
 *  life‑cycle is identical to the main generation code but the
 *  context is configured with `have_embeddings=true` and no
 *  prompt parsing / chat templating.
 *
 *  Functions:
 *    - nativeInitForEmbeddings  – load model, create context for embeddings
 *    - embed                    – return a single float[] of embeddings
 *    - nativeRelease (shared)
 *    - state persistence helpers (shared) – reuse the same functions
 *============================================================*/

#include "state/model_state.h"
#include "utils/jni_utils.h"
#include "utils/utf8_utils.h"

#include "llama.h"
#include "cpu/cpu_helper.h"

#include <jni.h>
#include <string>
#include <mutex>
#include "utils/logger.h"

static ModelState g_state;
static std::mutex g_init_mtx;

/*  --------------------------------------------------------------
 *      ENDEVMODE: initialise model for embeddings
 *  -------------------------------------------------------------- */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1core_EmbedLib_nativeInitForEmbeddings(JNIEnv *env, jobject, jstring jpath,
                                                      jint jthreads, jint ctxSize) {
    std::lock_guard<std::mutex> lk(g_init_mtx);
    const std::string path = utf8::from_jstring(env, jpath);
    g_state.release();
    llama_backend_init();

    int phys = count_physical_cores();
    int nthreads = (jthreads > 0) ? static_cast<int>(jthreads) : phys;
    LOG_INFO("Embedding init: model=%s, threads=%d, ctx=%d", path.c_str(), nthreads, ctxSize);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;                  // CPU
    mparams.use_mmap = true;
    mparams.use_mlock = false;
    mparams.check_tensors = true;

    g_state.model = llama_model_load_from_file(path.c_str(), mparams);
    if (!g_state.model) {
        LOG_ERROR("Failed to load model for embeddings");
        g_state.release();
        return JNI_FALSE;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = ctxSize;
    cparams.n_batch = 256;
    cparams.n_ubatch = 64;
    cparams.n_threads = nthreads;
    cparams.n_threads_batch = nthreads;
    cparams.no_perf = true;
    cparams.embeddings = true;          // crucial flag
    cparams.offload_kqv = false;        // CPU only

    g_state.ctx = llama_init_from_model(g_state.model, cparams);
    if (!g_state.ctx) {
        LOG_ERROR("Failed to create embedding context");
        g_state.release();
        return JNI_FALSE;
    }

    g_state.ctx_size = ctxSize;
    g_state.batch_size = cparams.n_batch;
    LOG_INFO("Embedding model initialised successfully");
    return JNI_TRUE;
}
/*  --------------------------------------------------------------
 *      EMBED: returns a float[] of size n_embd
 *  -------------------------------------------------------------- */
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_mp_ai_1core_EmbedLib_embed(JNIEnv *env, jobject, jstring jtext) {
    if (!g_state.is_ready()) {
        LOG_ERROR("embed – model not initialised");
        return nullptr;
    }

    const std::string txt = utf8::from_jstring(env, jtext);
    if (txt.empty()) {
        LOG_ERROR("embed – empty input text");
        return nullptr;
    }

    LOG_INFO("Embedding text (len %zu)", txt.size());

    const llama_vocab *vocab = llama_model_get_vocab(g_state.model);
    if (!vocab) {
        LOG_ERROR("embed – failed to get vocab");
        return nullptr;
    }

    /* ---------- Tokenise text ---------------------------------------- */
    std::vector<llama_token> toks = g_state.tokenize(txt);
    if (toks.empty()) {
        LOG_ERROR("embed – tokenisation produced 0 tokens");
        return nullptr;
    }
    if (static_cast<int32_t>(toks.size()) >= g_state.ctx_size) {
        LOG_ERROR("embed – token count (%d) exceeds context size (%d)",
                  static_cast<int>(toks.size()), g_state.ctx_size);
        return nullptr;
    }

    /* ---------- Build a single-token batch (all tokens processed) ----- */
    llama_batch batch = llama_batch_init(static_cast<int32_t>(toks.size()), 0, 1);
    if (!batch.token) {
        LOG_ERROR("embed – batch allocation failed");
        return nullptr;
    }

    for (int i = 0; i < (int) toks.size(); ++i) {
        batch.token[i] = toks[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == (int) toks.size() - 1);   // only last token needs logits
    }
    batch.n_tokens = static_cast<int32_t>(toks.size());

    /* ---------- Decode ---------------------------------------------- */
    int rc = llama_decode(g_state.ctx, batch);
    if (rc != 0) {
        LOG_ERROR("embed – llama_decode returned %d", rc);
        llama_batch_free(batch);
        return nullptr;
    }
    llama_batch_free(batch);

    /* ---------- Grab embeddings ------------------------------------- */
    const float *emb = llama_get_embeddings(g_state.ctx);
    if (!emb) {
        LOG_ERROR("embed – llama_get_embeddings returned null");
        return nullptr;
    }

    int n_embd = llama_model_n_embd(g_state.model);
    jfloatArray out = env->NewFloatArray(n_embd);
    if (!out) {
        LOG_ERROR("embed – failed to allocate Java float array");
        return nullptr;
    }
    env->SetFloatArrayRegion(out, 0, n_embd, const_cast<float *>(emb));
    if (env->ExceptionCheck()) {
        LOG_ERROR("embed – exception during SetFloatArrayRegion");
        env->ExceptionClear();
        return nullptr;
    }
    return out;
}

/*  --------------------------------------------------------------
 *      Re‑use the same helpers for release / state persistence
 *  -------------------------------------------------------------- */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1core_EmbedLib_nativeRelease(JNIEnv *, jobject) {
    g_state.release();
    return JNI_TRUE;
}