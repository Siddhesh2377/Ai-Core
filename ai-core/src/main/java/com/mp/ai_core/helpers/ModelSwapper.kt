package com.mp.ai_core.helpers

import com.mp.ai_core.*
import kotlinx.coroutines.*
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * Small helper that guarantees *only one* of the two
 * NativeLib / EmbedLib instances is active at any time.
 *
 * All heavy operations happen in the supplied CoroutineScope,
 * so the caller can decide the dispatcher (normally Dispatchers.IO).
 *
 * The swapper also remembers the *current* generation model
 * path so that it can be restored after the embedding step.
 */
class ModelSwapper(
    private val svcScope: CoroutineScope
) {
    private val lock = Mutex()                // serialises swaps
    private var genPath: String? = null       // last loaded generation model

    /* --------------------------------------------------------------
     *  Swapping helpers – called from the binder implementation.
     * -------------------------------------------------------------- */
    suspend fun <R> usingEmbedding(
        embedPath: String,
        embedAction: suspend (EmbedLib) -> R
    ): R {
        lock.withLock {
            val oldGen = genPath

            // 1️⃣  Unload generation (if any)
            oldGen?.let { unloadGeneration(it) }

            // 2️⃣  Load embedding
            val embedLib = EmbedLib.getInstance()
            val loaded     = embedLib.loadModel(embedPath)

            if (!loaded) throw RuntimeException("Embedding init failed: $embedPath")

            // 3️⃣  Run the requested action
            val result: R = try { embedAction(embedLib) } finally {
                // 4️⃣  Reload the text model (if it existed)
                oldGen?.let { reloadGeneration(it) }
            }
            return result
        }
    }

    private suspend fun unloadGeneration(path: String) {
        withContext(svcScope.coroutineContext) {
            // call on binder thread – no UI involved
            lib.nativeRelease()
            genPath = null
        }
    }

    private suspend fun reloadGeneration(path: String) {
        withContext(svcScope.coroutineContext) {
            lib.init(
                path, threads = 0,
                ctxSize = 4096, temp = 0.7f, topK = 20, topP = 0.9f, minP = 0.0f
            )
            genPath = path
        }
    }

    /* keep a *single* holder of the generation lib -------------------------------- */
    private val lib = NativeLib.getInstance()
}