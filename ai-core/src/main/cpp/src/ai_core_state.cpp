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

// -----------------------------------------------------------------
// State persistence wrappers – identical to the main module
// -----------------------------------------------------------------
extern "C" JNIEXPORT jlong JNICALL
Java_com_mp_ai_1core_StateLib_nativeGetStateSize(JNIEnv *, jobject) {
    return g_state.get_state_size();
}
extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_mp_ai_1core_StateLib_nativeGetStateData(JNIEnv *env, jobject) {
    jlong sz = g_state.get_state_size();
    if (!sz) return nullptr;
    jbyteArray arr = env->NewByteArray(static_cast<jsize>(sz));
    if (!arr) return nullptr;
    void *buf = env->GetByteArrayElements(arr, nullptr);
    g_state.get_state_data(buf, static_cast<size_t>(sz));
    env->ReleaseByteArrayElements(arr, (jbyte *) buf, 0);
    return arr;
}
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1core_StateLib_nativeLoadStateData(JNIEnv *env, jobject, jbyteArray arr) {
    if (!arr) return JNI_FALSE;
    jbyte *buf = env->GetByteArrayElements(arr, nullptr);
    size_t sz = static_cast<size_t>(env->GetArrayLength(arr));
    bool ok = g_state.load_state_data(buf, sz);
    env->ReleaseByteArrayElements(arr, buf, JNI_ABORT);
    return ok ? JNI_TRUE : JNI_FALSE;
}
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1core_StateLib_nativeSaveStateFile(JNIEnv *env, jobject, jstring jpath) {
    const char *path = env->GetStringUTFChars(jpath, nullptr);
    bool ok = llama_state_save_file(g_state.ctx, path, nullptr, 0);
    env->ReleaseStringUTFChars(jpath, path);
    return ok ? JNI_TRUE : JNI_FALSE;
}
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1core_StateLib_nativeLoadStateFile(JNIEnv *env, jobject, jstring jpath) {
    const char *path = env->GetStringUTFChars(jpath, nullptr);
    bool ok = llama_state_load_file(g_state.ctx, path, nullptr, 0, nullptr);
    env->ReleaseStringUTFChars(jpath, path);
    return ok ? JNI_TRUE : JNI_FALSE;
}