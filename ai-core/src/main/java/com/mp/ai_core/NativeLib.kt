package com.mp.ai_core

import android.util.Log
import androidx.annotation.Keep
import com.mp.ai_core.services.IGenerationCallback
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.onFailure

// Keep the tag consistent with the previous code
private const val TAG = "MicroAI"

/**
 * A singleton‑style wrapper around the native inference library.
 *
 * Only the *generation* model is kept – all embedding‑related
 * methods and instances have been removed.  The coroutines below
 * are deliberately light‑weight: we use a dedicated CoroutineScope
 * with a single SupervisorJob and `Dispatchers.Default/IO` for
 * heavy work.  No blocking `when { }` or heavy `Thread` creates.
 *
 * NOTE: The companion’s `getInstance()` is the only public
 * factory.  It guarantees that a single native instance
 * lives per JVM, and that it is cleaned up on `releaseAll()`.
 */
class NativeLib private constructor(private val instanceId: String) {

    // -----------------------------------------------------------------
    //  Companion – factory & instance cache
    // -----------------------------------------------------------------
    companion object {
        private val instances = mutableMapOf<String, NativeLib>()
        private var coroutineScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
        init { System.loadLibrary("ai_core") }   // one-time NDK load

        /**
         * Returns the long‑lived generation instance.
         */
        fun getInstance(): NativeLib =
            instances.getOrPut("generation") { NativeLib("generation") }

        /**
         * Releases a particular instance – frees the native context.
         * It *does not* destroy the singleton mix‑in from the NDK.
         */
        fun releaseInstance(id: String) {
            instances[id]?.let { it.nativeRelease(); instances.remove(id) }
        }

        /** Tear down *all* living instances – e.g. when the app exits. */
        fun releaseAll() { instances.values.forEach { it.nativeRelease() }; instances.clear() }
    }

    // -------------------------------------------------------------
    //  External natives
    // -------------------------------------------------------------
    /** NDK helpers – exported by the C++ side.  All wrap the same context. */
    external fun nativeGetStateSize(): Long
    external fun nativeLoadStateData(state: ByteArray): Boolean
    external fun nativeGetStateData(): ByteArray?
    external fun nativeLoadStateFile(path: String): Boolean
    external fun nativeSaveStateFile(path: String): Boolean
    external fun nativeRelease(): Boolean
    external fun nativeSetChatTemplate(template: String)
    external fun nativeSetToolsJson(toolsJson: String)
    external fun nativeSetSystemPrompt(prompt: String)
    external fun nativeGetModelInfo(): String
    external fun nativeStopGeneration()
    external fun nativeGenerateStream(prompt: String,
                                      maxTokens: Int,
                                      callback: StreamCallback): Boolean

    /** Native init‑fn that reflects the *simpler* C++ signature. */
    external fun nativeInit(path: String,
                            threads: Int,
                            ctxSize: Int,
                            temp: Float,
                            topK: Int,
                            topP: Float,
                            minP: Float,
                            mirostat: Int,
                            mirostatTau: Float,
                            mirostatEta: Float,
                            seed: Int): Boolean

    // -------------------------------------------------------------
    //  State flags – keep them local (no static global weak refs)
    // -------------------------------------------------------------
    private var isModelInitialized = false

    // -------------------------------------------------------------
    //  Initialisation helpers
    // -------------------------------------------------------------
    /**
     * Bootstraps an inference model.  Only calls the one‑argument
     * nativeInit; everything else (GPU, mmap, etc.) is hard‑wired
     * to the CPU‑only path in C++.
     */
    @JvmOverloads
    fun init(path: String,
             threads: Int = Runtime.getRuntime().availableProcessors() / 2,
             ctxSize: Int = 4096,
             temp: Float = 0.7f,
             topK: Int = 20,
             topP: Float = 0.9f,
             minP: Float = 0.0f,
             mirostat: Int,
             mirostatTau: Float ,
             mirostatEta: Float ,
             seed: Int,
    ): Boolean {
        nativeRelease()   // graceful cleanup of any pre‑existing ctx
        return try {
            val ok = nativeInit(path, threads, ctxSize, temp, topK, topP, minP, mirostat, mirostatTau, mirostatEta, seed)
            if (ok) {
                isModelInitialized = true
                Log.i("$instanceId.nn", "Model initialised: $path")
            } else {
                Log.e("$instanceId.nn", "Model init failed: $path")
            }
            ok
        } catch (t: Throwable) {
            Log.e("$instanceId.nn", "init error", t); false
        }
    }

    fun isReady() = isModelInitialized

    // -------------------------------------------------------------
    //  High‑level generation – backed by coroutines & a Channel
    // -------------------------------------------------------------
    @OptIn(DelicateCoroutinesApi::class)
    suspend fun generateStreaming(
        prompt: String,
        maxTokens: Int = 512,
        callback: IGenerationCallback,
        toolsJson: String = ""
    ) {
        if (!isReady()) {
            callback.onError("Model not initialised – call init() first")
            return
        }

        if (!coroutineScope.isActive) {
            Log.w(TAG, "Scope inactive, recreating...")
            coroutineScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
        }

        nativeSetToolsJson(toolsJson.ifEmpty { "" })

        val ch = Channel<String>(capacity = 256)
        val batchMs = 35L

        // Consumer job
        val consumerJob = coroutineScope.launch {
            val sb = StringBuilder()
            var lastFlush = System.nanoTime()

            suspend fun flush(force: Boolean) {
                if (sb.isNotEmpty() &&
                    (force || (System.nanoTime() - lastFlush) / 1_000_000L >= batchMs)
                ) {
                    val chunk = sb.toString()
                    sb.setLength(0)
                    lastFlush = System.nanoTime()
                    withContext(Dispatchers.Main.immediate) {
                        val clean = chunk.replace("[PAD]", "")
                            .replace(Regex("\\[unused\\d+]"), "")
                        if (clean.isNotBlank()) callback.onToken(clean)
                    }
                }
            }

            try {
                for (token in ch) {
                    sb.append(token)
                    flush(false)
                }
                flush(true)
            } catch (e: Throwable) {
                Log.e(TAG, "Channel error", e)
                callback.onError(e.localizedMessage ?: "Channel error")
            }
        }

        val cb = object : StreamCallback {
            override fun onToken(token: String) {
                try {
                    // Check if still active
                    if (ch.isClosedForSend) {
                        Log.w(TAG, "Channel closed, stopping native generation")
                        nativeStopGeneration()
                        return
                    }

                    val result = ch.trySend(token)
                    if (result.isFailure) {
                        Log.w(TAG, "Failed to send token, error: ${result.exceptionOrNull()}")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error in onToken", e)
                }
            }

            override fun onToolCall(name: String, argsJson: String) {
                CoroutineScope(Dispatchers.Main.immediate).launch {
                    callback.onToolCall(name, argsJson)
                }
            }

            override fun onDone() {
                ch.close()
            }

            override fun onError(message: String) {
                ch.close()
                callback.onError(message)
            }
        }

        // ✅ Use a separate scope that won't be cancelled
        val generationScope = CoroutineScope(Dispatchers.IO + SupervisorJob())

        try {
            val generationJob = generationScope.launch {
                val ok = nativeGenerateStream(prompt, maxTokens, cb)
                if (!ok) {
                    callback.onError("nativeGenerateStream returned false")
                }
            }

            // ✅ Wait for BOTH jobs to complete
            generationJob.join()
            consumerJob.join()

        } finally {
            ch.close()
            generationScope.cancel()
        }

        callback.onDone()
    }

    // -------------------------------------------------------------
    //  Helper public methods – they are thin wrappers so we keep
    //  them in a single place but they are trivial
    // -------------------------------------------------------------
    fun setSystemPrompt(prompt: String) = nativeSetSystemPrompt(prompt)
    fun setChatTemplate(template: String) = nativeSetChatTemplate(template)
    fun setToolsJson(toolsJson: String) = nativeSetToolsJson(toolsJson)
    fun stop() = nativeStopGeneration()
}

/**
 * Lightweight callback the native layer uses to push tokens and tool calls
 * back to the Kotlin side.  Methods are intentionally minimal – that is
 * also what the C++ code expects.
 */
@Keep
interface StreamCallback {
    fun onToken(token: String)
    fun onToolCall(name: String, argsJson: String)
    fun onDone()
    fun onError(message: String)
}