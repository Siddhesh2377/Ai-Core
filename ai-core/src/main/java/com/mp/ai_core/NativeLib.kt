package com.mp.ai_core

import android.util.Log
import androidx.annotation.Keep
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.receiveAsFlow
import kotlin.math.sqrt

private const val TAG = "MicroAI"

class NativeLib {

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

    // Fixed: Return Boolean instead of Unit
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

    // Keep the simple embed function
    external fun embed(text: String): FloatArray?

    external fun nativeSetToolsJson(toolsJson: String)
    external fun nativeSetSystemPrompt(prompt: String)
    external fun nativeGetModelInfo(): String
    external fun nativeStopGeneration()

    companion object {
        init {
            System.loadLibrary("ai_core")
        }
    }

    /** Initialize model safely */
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
        return try {
            val ok = nativeInit(
                path, threads, gpuLayers, useMMAP, useMLOCK,
                ctxSize, temp, topK, topP, minP
            )
            if (!ok) {
                Log.e(TAG, "Model initialization failed at path: $path")
            }
            ok
        } catch (t: Throwable) {
            Log.e(TAG, "Model init error", t)
            false
        }
    }

    suspend fun getEmbedding(text: String): Result<FloatArray> = withContext(Dispatchers.Default) {
        val modelInfo = runCatching { nativeGetModelInfo() }.getOrNull()
        if (modelInfo.isNullOrEmpty()) {
            val err = "No model loaded. Please call initModel() first."
            Log.e(TAG, err)
            return@withContext Result.failure(Exception(err))
        }

        try {
            val embedding = embed(text)
            if (embedding != null) {
                Log.d(TAG, "Embedding generated successfully with size: ${embedding.size}")
                Result.success(embedding)
            } else {
                Log.e(TAG, "Embedding is null")
                Result.failure(Exception("Failed to generate embedding"))
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error generating embedding", e)
            Result.failure(e)
        }
    }

    /** System prompt setter */
    fun setSystemPrompt(prompt: String) = nativeSetSystemPrompt(prompt)

    /** Streamed generation with model check */
    fun generateStreaming(
        prompt: String,
        maxTokens: Int = 512,
        uiScope: CoroutineScope,
        onStart: () -> Unit,
        onGenerate: (String) -> Unit,
        onError: (String) -> Unit,
        onDone: () -> Unit,
        toolsJson: String? = null,
        onToolCall: (name: String, argsJson: String) -> Unit = { _, _ -> }
    ): Job {
        // Guard: model present?
        val modelInfo = runCatching { nativeGetModelInfo() }.getOrNull()
        if (modelInfo.isNullOrEmpty()) {
            val err = "No model loaded. Please call initModel() first."
            Log.e(TAG, err)
            onError(err)
            return SupervisorJob().apply { complete() }
        }

        // Enable/disable tools for this turn
        if (toolsJson != null) nativeSetToolsJson(toolsJson) else nativeSetToolsJson("")

        val tokenCh = Channel<String>(capacity = 256)

        val batchPeriodMs = 35L
        val batcherJob = uiScope.launch(Dispatchers.Default) {
            val sb = StringBuilder()
            var lastFlush = System.nanoTime()
            fun flush(force: Boolean = false) {
                if (sb.isNotEmpty() && (force || (System.nanoTime() - lastFlush) / 1_000_000 >= batchPeriodMs)) {
                    val chunk = sb.toString()
                    sb.setLength(0)
                    lastFlush = System.nanoTime()
                    uiScope.launch(Dispatchers.Main.immediate) { onGenerate(chunk) }
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
                uiScope.launch(Dispatchers.Main.immediate) {
                    onToolCall(name, argsJson)
                }
            }
            override fun onDone() { tokenCh.close() }
            override fun onError(message: String) {
                uiScope.launch(Dispatchers.Main.immediate) { onError(message) }
                tokenCh.close()
            }
        }

        onStart()

        val parentJob = uiScope.launch(Dispatchers.IO) {
            try {
                nativeGenerateStream(prompt, maxTokens, cb)
            } catch (t: Throwable) {
                Log.e(TAG, "nativeGenerateStream error", t)
                withContext(Dispatchers.Main.immediate) { onError(t.message ?: "Native error") }
            } finally {
                tokenCh.close()
            }
        }

        parentJob.invokeOnCompletion {
            batcherJob.cancel()
            uiScope.launch {
                batcherJob.join()
                withContext(Dispatchers.Main.immediate) { onDone() }
            }
        }
        return parentJob
    }
}

@Keep
interface StreamCallback {
    fun onToken(token: String)
    fun onToolCall(name: String, argsJson: String)
    fun onDone()
    fun onError(message: String)
}

class EmbeddingManager(private val nativeLib: NativeLib) {
    private var isInitialized = false

    // Initialize embedding model using the same model as chat
    suspend fun initializeEmbedding(
        modelPath: String,
        contextSize: Int = 2048,
        gpuLayers: Int = 0
    ): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            Log.i("EmbeddingManager", "Initializing embedding model at: $modelPath")

            val success = nativeLib.nativeInitForEmbeddings(
                modelPath,
                4, // threads
                gpuLayers,
                false, // useMMAP
                contextSize
            )

            if (success) {
                isInitialized = true
                Log.i("EmbeddingManager", "Embedding model initialized successfully")
                Result.success(Unit)
            } else {
                Log.e("EmbeddingManager", "Failed to initialize embedding model")
                Result.failure(Exception("Failed to initialize embedding model"))
            }
        } catch (e: Exception) {
            Log.e("EmbeddingManager", "Error initializing embedding model", e)
            Result.failure(e)
        }
    }

    // Get embedding using the simple embed function
    suspend fun getEmbedding(
        text: String,
        meanPool: Boolean = true // This parameter is ignored for now
    ): Result<FloatArray> = withContext(Dispatchers.Default) {
        if (!isInitialized) {
            return@withContext Result.failure(Exception("Embedding model not initialized"))
        }

        try {
            Log.d("EmbeddingManager", "Getting embedding for text: ${text.take(50)}...")
            val embedding = nativeLib.embed(text)

            if (embedding != null && embedding.isNotEmpty()) {
                Log.d("EmbeddingManager", "Embedding generated successfully with size: ${embedding.size}")
                Result.success(embedding)
            } else {
                Log.e("EmbeddingManager", "Empty or null embedding returned")
                Result.failure(Exception("Empty embedding returned"))
            }
        } catch (e: Exception) {
            Log.e("EmbeddingManager", "Error generating embedding", e)
            Result.failure(e)
        }
    }

    // Batch embeddings
    suspend fun getEmbeddings(
        texts: List<String>,
        meanPool: Boolean = true
    ): List<Result<FloatArray>> = withContext(Dispatchers.Default) {
        texts.map { text ->
            getEmbedding(text, meanPool)
        }
    }

    // Calculate cosine similarity
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

    // Clean up resources - just mark as not initialized since we're using the shared model
    fun release() {
        isInitialized = false
        Log.i("EmbeddingManager", "Embedding manager released")
    }
}