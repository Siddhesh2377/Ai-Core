package com.mp.ai_core

import android.util.Log
import androidx.annotation.Keep
import com.mp.ai_core.services.IGenerationCallback
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.receiveAsFlow
import kotlin.math.sqrt

const val TAG = "MicroAI"

class NativeLib private constructor(private val instanceId: String) {
    companion object {
        private val instances = mutableMapOf<String, NativeLib>()

        init {
            System.loadLibrary("ai_core")
        }

        fun getGenerationInstance(): NativeLib {
            return instances.getOrPut("generation") {
                NativeLib("generation")
            }
        }

        fun getGenerationServiceInstance(): NativeLib =
            instances.getOrPut("service") { NativeLib("service") }

        fun getEmbeddingInstance(): NativeLib {
            return instances.getOrPut("embedding") {
                NativeLib("embedding")
            }
        }

        fun releaseInstance(instanceId: String) {
            instances[instanceId]?.let { instance ->
                instance.nativeRelease()
                instances.remove(instanceId)
            }
        }

        fun releaseAllInstances() {
            instances.values.forEach { it.nativeRelease() }
            instances.clear()
        }
    }

    private var isMainModelInitialized = false
    private var isEmbeddingModelInitialized = false

    fun initModel(
        path: String,
        threads: Int = Runtime.getRuntime().availableProcessors() / 2,
        gpuLayers: Int = 0,
        useMMAP: Boolean = true,
        useMLOCK: Boolean = false,
        ctxSize: Int = 4096,
        temp: Float = 0.7f,
        topK: Int = 20,
        topP: Float = 0.9f,
        minP: Float = 0.0f
    ): Boolean {
        // Only release this instance, not all instances
        nativeRelease()
        return try {
            val ok = nativeInit(
                path, threads, gpuLayers, useMMAP, useMLOCK,
                ctxSize, temp, topK, topP, minP
            )
            if (ok) {
                isMainModelInitialized = true
                isEmbeddingModelInitialized = false
                Log.i("NativeLib-$instanceId", "Main model initialized: $path")
            } else {
                Log.e("NativeLib-$instanceId", "Main model initialization failed: $path")
            }
            ok
        } catch (t: Throwable) {
            Log.e("NativeLib-$instanceId", "Main model init error", t)
            false
        }
    }

    fun initEmbeddingModel(
        path: String,
        threads: Int = 4,
        gpuLayers: Int = 0,
        useMMAP: Boolean = true,
        ctxSize: Int = 512
    ): Boolean {
        nativeRelease()
        Log.d("NativeLib-$instanceId", "Initializing embedding model: $path")
        return try {
            val ok = nativeInitForEmbeddings(
                path, threads, gpuLayers, useMMAP, ctxSize
            )
            if (ok) {
                isEmbeddingModelInitialized = true
                isMainModelInitialized = false
                Log.i("NativeLib-$instanceId", "Embedding model initialized successfully")
            } else {
                Log.e("NativeLib-$instanceId", "Embedding model initialization failed: $path")
            }
            ok
        } catch (t: Throwable) {
            Log.e("NativeLib-$instanceId", "Embedding model init error", t)
            false
        }
    }

    // Add method to check model state
    fun isGenerationReady(): Boolean = isMainModelInitialized
    fun isEmbeddingReady(): Boolean = isEmbeddingModelInitialized

    // Other external functions remain the same
    external fun nativeInit(
        path: String,
        threads: Int,
        gpuLayers: Int,
        useMMAP: Boolean,
        useMLOCK: Boolean,
        ctxSize: Int,
        temp: Float,
        topK: Int,
        topP: Float,
        minP: Float
    ): Boolean

    external fun nativeRelease(): Boolean
    external fun nativeSetChatTemplate(template: String)
    external fun nativeInitForEmbeddings(
        path: String,
        jthreads: Int,
        nGpuLayers: Int,
        useMMAP: Boolean,
        nCtx: Int
    ): Boolean
    external fun nativeGenerateStream(
        prompt: String,
        maxTokens: Int,
        callback: StreamCallback
    ): Boolean
    external fun embed(text: String): FloatArray?
    external fun nativeSetToolsJson(toolsJson: String)
    external fun nativeSetSystemPrompt(prompt: String)
    external fun nativeGetModelInfo(): String
    external fun nativeStopGeneration()

    fun setSystemPrompt(prompt: String) = nativeSetSystemPrompt(prompt)

    suspend fun generateStreaming(
        prompt: String,
        maxTokens: Int = 512,
        callback: IGenerationCallback,
        toolsJson: String = "",
    ) {
        // Guard: model present?
        val modelInfo = runCatching { nativeGetModelInfo() }.getOrNull()
        if (modelInfo.isNullOrEmpty()) {
            val err = "No model loaded. Please call initModel() first."
            Log.e(TAG, err)
            callback.onError(err)
            return
        }

        // Enable/disable tools for this turn
        if (toolsJson.isNotEmpty()) nativeSetToolsJson(toolsJson) else nativeSetToolsJson("")

        val tokenCh = Channel<String>(capacity = 256)
        val batchPeriodMs = 35L

        coroutineScope {
            val batcherJob = launch(Dispatchers.Default) {
                val sb = StringBuilder()
                var lastFlush = System.nanoTime()

                fun flush(force: Boolean = false) {
                    if (sb.isNotEmpty() && (force || (System.nanoTime() - lastFlush) / 1_000_000 >= batchPeriodMs)) {
                        val chunk = sb.toString()
                        sb.setLength(0)
                        lastFlush = System.nanoTime()

                        launch(Dispatchers.Main.immediate) {
                            val cleanToken = chunk
                                .replace("[PAD]", "")
                                .replace(Regex("\\[unused\\d+]"), "")
                            if (cleanToken.isNotEmpty()) {
                                launch(Dispatchers.Main) {
                                    callback.onToken(cleanToken)
                                }
                            }
                        }
                    }
                }

                try {
                    for (tok in tokenCh) {
                        sb.append(tok)
                        flush(false)
                    }
                } finally {
                    flush(true)
                }
            }

            val cb = object : StreamCallback {
                override fun onToken(token: String) {
                    if (!tokenCh.trySend(token).isSuccess) {
                        Log.w(TAG, "Token dropped due to backpressure")
                    }
                }

                override fun onToolCall(name: String, argsJson: String) {
                    launch(Dispatchers.Main.immediate) {
                        callback.onToolCall(name, argsJson)
                    }
                }

                override fun onDone() {
                    tokenCh.close()
                }

                override fun onError(message: String) {
                    launch(Dispatchers.Main.immediate) { onError(message) }
                    tokenCh.close()
                }
            }

            val parentJob = launch(Dispatchers.IO) {
                try {
                    val success = nativeGenerateStream(prompt, maxTokens, cb)
                    if (!success) {
                        throw IllegalStateException("nativeGenerateStream returned false")
                    }
                } catch (t: Throwable) {
                    Log.e(TAG, "nativeGenerateStream error", t)
                    withContext(Dispatchers.Main.immediate) { callback.onError(t.message ?: "Native error") }
                } finally {
                    tokenCh.close()
                }
            }

            parentJob.join()
            batcherJob.cancel()
            batcherJob.join()
            withContext(Dispatchers.Main.immediate) { callback.onDone() }
        }
    }

}

@Keep
interface StreamCallback {
    fun onToken(token: String)
    fun onToolCall(name: String, argsJson: String)
    fun onDone()
    fun onError(message: String)
}

class EmbeddingManager {
    companion object {
        @Volatile
        private var instance: EmbeddingManager? = null

        fun getInstance(): EmbeddingManager {
            return instance ?: synchronized(this) {
                instance ?: EmbeddingManager().also { instance = it }
            }
        }
    }

    private val embeddingLib = NativeLib.getEmbeddingInstance()
    private var initialized = false
    private var embeddingDim = -1

    suspend fun initializeEmbedding(modelPath: String): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            Log.d("EmbeddingManager", "Initializing embedding model: $modelPath")
            val ok = embeddingLib.initEmbeddingModel(modelPath, 4, 0, true, 512)
            if (!ok) {
                Log.e("EmbeddingManager", "Failed to initialize embedding model: $modelPath")
                return@withContext Result.failure(Exception("Failed to initialize embedding model"))
            }

            // Warmup
            val warmup = embeddingLib.embed("warmup")
            if (warmup == null || warmup.isEmpty()) {
                Log.e("EmbeddingManager", "Warmup failed")
                return@withContext Result.failure(Exception("Warmup failed"))
            }

            embeddingDim = warmup.size
            initialized = true
            Log.i("EmbeddingManager", "Embedding model ready (dim=$embeddingDim)")
            Result.success(Unit)
        } catch (e: Exception) {
            Log.e("EmbeddingManager", "Initialization error", e)
            Result.failure(e)
        }
    }

    suspend fun getEmbedding(text: String): Result<FloatArray> = withContext(Dispatchers.Default) {
        if (!initialized) {
            return@withContext Result.failure(Exception("Embedding model not initialized"))
        }
        try {
            val embedding = embeddingLib.embed(text)
            if (embedding != null && embedding.isNotEmpty()) {
                Result.success(embedding)
            } else {
                Result.failure(Exception("Empty embedding returned"))
            }
        } catch (e: Exception) {
            Log.e("EmbeddingManager", "Error generating embedding", e)
            Result.failure(e)
        }
    }

    suspend fun getEmbeddings(
        texts: List<String>,
    ): List<Result<FloatArray>> = withContext(Dispatchers.Default) {
        texts.map { text ->
            getEmbedding(text)
        }
    }

    fun cosineSimilarity(embedding1: FloatArray, embedding2: FloatArray): Float {
        if (embedding1.size != embedding2.size) {
            Log.w("EmbeddingManager", "Embedding size mismatch: ${embedding1.size} vs ${embedding2.size}")
            return 0f
        }
        var dotProduct = 0f
        var norm1 = 0f
        var norm2 = 0f
        for (i in embedding1.indices) {
            dotProduct += embedding1[i] * embedding2[i]
            norm1 += embedding1[i] * embedding1[i]
            norm2 += embedding2[i] * embedding2[i]
        }
        return if (norm1 == 0f || norm2 == 0f) {
            0f
        } else {
            dotProduct / (sqrt(norm1) * sqrt(norm2))
        }
    }

    fun release() {
        initialized = false
        Log.i("EmbeddingManager", "Embedding manager released")
    }
}
