package com.mp.ai_core

import android.util.Log
import androidx.annotation.Keep
import com.mp.ai_core.text.IGenerationCallback
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel

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
                            minP: Float): Boolean

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
             minP: Float = 0.0f): Boolean {
        nativeRelease()   // graceful cleanup of any pre‑existing ctx
        return try {
            val ok = nativeInit(path, threads, ctxSize, temp, topK, topP, minP)
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
    suspend fun generateStreaming(
        prompt: String,
        maxTokens: Int = 512,
        callback: IGenerationCallback,
        toolsJson: String = ""
    ) {
        // Quick guard – can be omitted if you trust callers
        if (!isReady()) {
            callback.onError("Model not initialised – call init() first")
            return
        }

        // Set/clear tool JSON for this turn
        nativeSetToolsJson(toolsJson.ifEmpty { "" })

        // Channel that passes raw tokens from the JNI layer to the UI
        val ch = Channel<String>(capacity = 256)
        val batchMs = 35L
        val coroutineScope = CoroutineScope(Dispatchers.Default + SupervisorJob())

        coroutineScope.launch {
            // ---- Batching that runs on a background thread ----
            val sb = StringBuilder()
            var lastFlush = System.nanoTime()

            suspend fun flush (force: Boolean) {
                if (sb.isNotEmpty() &&
                    (force || (System.nanoTime() - lastFlush) / 1_000_000L >= batchMs)
                ) {
                    val chunk = sb.toString()
                    sb.setLength(0)
                    lastFlush = System.nanoTime()
                    withContext(Dispatchers.Main.immediate) {
                        // Clean token – commonly `[PAD]` and `[unusedX]` are useless
                        val clean = chunk.replace("[PAD]", "")
                            .replace(Regex("\\[unused\\d+]"), "")
                        if (clean.isNotBlank()) callback.onToken(clean)
                    }
                }
            }

            // Consume the channel
            try {
                for (token in ch) {
                    sb.append(token)
                    flush(false)
                }
                flush(true)   // one last flush
            } catch (e: Throwable) {
                Log.e(TAG, "Channel error", e)
                callback.onError(e.localizedMessage ?: "Channel error")
            }
        }

        // -----------------------------------------------
        //  JNI side – feed tokens to the channel
        // -----------------------------------------------
        val cb = object : StreamCallback {
            override fun onToken(tok: String) {
                // `trySend` is non‑blocking; if buffer is full we silently drop
                // this token.  The normal flow down the channel gives us backpressure
                // if we run out of callers.
                ch.trySend(tok).isSuccess
            }

            override fun onToolCall(name: String, argsJson: String) {
                // Jump to UI thread – caller may use dispatchers.AFTER_EVENT
                CoroutineScope(Dispatchers.Main.immediate).launch { callback.onToolCall(name, argsJson) }
            }

            override fun onDone() { ch.close() }
            override fun onError(msg: String) { ch.close(); callback.onError(msg) }
        }

        // Run the JNI call on IO – it is blocking until model finishes
        val parent = coroutineScope.launch(Dispatchers.IO) {
            val ok = nativeGenerateStream(prompt, maxTokens, cb)
            if (!ok) callback.onError("nativeGenerateStream returned false")
        }

        // Wait for the flow to finish – we cancel the batcher afterwards
        parent.join()
        coroutineScope.cancel()
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