/*
 *  EmbeddingJNI.kt
 *
 *  Thin JNI wrapper that exposes the *embedding* side of the native
 *  library (`ai_core.so`).  Only the functions that live in
 *  `ai_core_embed.cpp` are referenced here – everything else is kept
 *  in `ai_core.so` so that we avoid any accidental dependency on
 *  the generation code.
 *
 *  The class is deliberately minimal:
 *      1.  One‑time `System.loadLibrary("ai_core")` in the companion.
 *      2.  External native declarations that match exactly the
 *          function signatures in `ai_core_embed.cpp`.
 *      3.  A small helper API (`loadModel`, `embed`, `state`) that
 *          is safe to call from coroutine scope guarantees.
 *
 *  Because the embedding functionality is usually a blocking
 *  operation (full‑sentence encoding), all public methods are marked
 *  `suspend` so that they can be invoked from a `Dispatchers.IO`
 *  context without blocking the UI thread.
 *
 *  ───────────────────────────────────────────────────────────────────────
 */

package com.mp.ai_core

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * A dedicated JNI wrapper for the *embedding* portion of the native
 * library (`ai_core_embed.cpp`).
 *
 * Only one instance lives per JVM.  The companion holds the native
 * library load and a static factory.
 */
class EmbedLib private constructor() {

    companion object {
        private const val TAG = "EmbeddingJNI"

        init { System.loadLibrary("ai_core_embed") }

        /**
         * Return the one and only JNI wrapper instance.
         */
        @Synchronized
        fun getInstance(): EmbedLib = EmbedLib()
    }

    /* ----------------------------------------------------------------
     *  External JNI declarations (must be exact matches to the C++)
     * ---------------------------------------------------------------- */

    /** Initialise an embedding model (CPU only). */
    external fun nativeInitForEmbeddings(
        path: String,
        jthreads: Int,
        nCtx: Int
    ): Boolean

    /** Load a previous embedding session. */
    external fun nativeLoadStateFile(path: String): Boolean

    /** Save the current embedding session. */
    external fun nativeSaveStateFile(path: String): Boolean

    /** Return length of the current state buffer. */
    external fun nativeGetStateSize(): Long

    /** Return a copy of the state buffer or null. */
    external fun nativeGetStateData(): ByteArray?

    /** Replace the current state buffer with the supplied data. */
    external fun nativeLoadStateData(state: ByteArray): Boolean

    /** Generate an embedding for a single string. */
    external fun embed(text: String): FloatArray?

    /** Gracefully free the native context. */
    external fun nativeRelease(): Boolean

    /* ----------------------------------------------------------------
     *  Kotlin helper API – coroutine friendly wrappers
     * ---------------------------------------------------------------- */

    /**
     * Load an embedding model.
     *
     * @param path      Path to the .bin or .mgf file.
     * @param threads   Number of CPU threads to use (default: half of the
     *                  available processors).
     * @param ctxSize   Token context size (default: 512).
     * @return `true` if the native side started properly.
     */
    suspend fun loadModel(
        path: String,
        threads: Int = Runtime.getRuntime().availableProcessors() / 2,
        ctxSize: Int = 512
    ): Boolean = withContext(Dispatchers.IO) {
        try {
            val ok = nativeInitForEmbeddings(path, threads, ctxSize)
            if (!ok) Log.e(TAG, "Embedding init failed: $path")
            ok
        } catch (t: Throwable) {
            Log.e(TAG, "Embedding init exception", t)
            false
        }
    }

    /**
     * Encode a single piece of text – returns a nullable FloatArray.
     * The call is dispatched to `Dispatchers.IO` because it may block.
     */
    suspend fun encode(text: String): FloatArray? = withContext(Dispatchers.IO) {
        try {
            embed(text) ?: run {
                Log.w(TAG, "Empty embedding returned for: \"$text\"")
                null
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Embedding error", t)
            null
        }
    }

    /**
     * Save and load the embedding KV‑cache/session.
     */
    suspend fun saveSession(path: String): Boolean = withContext(Dispatchers.IO) {
        try { nativeSaveStateFile(path) }
        catch (t: Throwable) { Log.e(TAG, "saveSession error", t); false }
    }

    suspend fun loadSession(path: String): Boolean = withContext(Dispatchers.IO) {
        try { nativeLoadStateFile(path) }
        catch (t: Throwable) { Log.e(TAG, "loadSession error", t); false }
    }

    suspend fun getStateData(): ByteArray? = withContext(Dispatchers.IO) { nativeGetStateData() }
    suspend fun loadStateData(state: ByteArray): Boolean = withContext(Dispatchers.IO) {
        try { nativeLoadStateData(state) } catch (t: Throwable) {
            Log.e(TAG, "loadStateData error", t); false
        }
    }
    suspend fun getStateSize(): Long = withContext(Dispatchers.IO) { nativeGetStateSize() }

    /**
     * Safely free the embedded context – called when the app is
     * shutting down or when you want to switch models.
     */
    suspend fun release() = withContext(Dispatchers.IO) {
        try { nativeRelease() } catch (t: Throwable) { Log.e(TAG, "release error", t) }
    }
}