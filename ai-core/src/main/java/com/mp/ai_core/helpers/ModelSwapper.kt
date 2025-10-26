package com.mp.ai_core.helpers

import android.util.Log
import com.mp.ai_core.*
import com.mp.ai_core.text.IGenerationCallback
import kotlinx.coroutines.*
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.concurrent.atomic.AtomicBoolean

private const val TAG = "ModelSwapper"

class ModelSwapper(private val scope: CoroutineScope) {

    private val lock = Mutex()
    private val genLib = NativeLib.getInstance()
    private val embedLib = EmbedLib.getInstance()
    private val mtmdLib = MtmdLib.getInstance()
    private val stateLib = StateLib.getInstance()

    private var currentGenPath: String? = null
    private var currentEmbedPath: String? = null
    private var currentMmprojPath: String? = null

    private var generationJob: Job? = null
    private val _isGenerating = AtomicBoolean(false)
    private var activeCallback: IGenerationCallback? = null

    suspend fun loadGeneration(path: String, threads: Int = 0, ctxSize: Int = 4096, temp: Float = 0.7f, topK: Int = 20, topP: Float = 0.9f, minP: Float = 0f): Boolean = withContext(Dispatchers.IO) {
        Log.d(TAG, "loadGeneration: path=$path, threads=$threads, ctxSize=$ctxSize")
        lock.withLock {
            if (currentEmbedPath != null) {
                Log.d(TAG, "Unloading embedding model first")
                embedLib.release()
                currentEmbedPath = null
            }

            val ok = genLib.init(path, threads, ctxSize, temp, topK, topP, minP)
            if (ok) {
                currentGenPath = path
                Log.i(TAG, "Generation model loaded: $path")
            } else {
                Log.e(TAG, "Failed to load generation model: $path")
            }
            ok
        }
    }

    suspend fun unloadGeneration() = withContext(Dispatchers.IO) {
        Log.d(TAG, "unloadGeneration")
        lock.withLock {
            stopGeneration()
            if (currentMmprojPath != null) {
                mtmdLib.release()
                currentMmprojPath = null
            }
            genLib.nativeRelease()
            currentGenPath = null
            Log.i(TAG, "Generation model unloaded")
        }
    }

    suspend fun loadMultimodal(mmprojPath: String, threads: Int = 4): Boolean = withContext(Dispatchers.IO) {
        Log.d(TAG, "loadMultimodal: path=$mmprojPath, threads=$threads")
        lock.withLock {
            if (currentGenPath == null) {
                Log.e(TAG, "Base model must be loaded first")
                return@withLock false
            }
            if (currentMmprojPath != null) {
                Log.w(TAG, "Projector already loaded: $currentMmprojPath")
                return@withLock false
            }
            val ok = mtmdLib.init(mmprojPath, threads)
            if (ok) {
                currentMmprojPath = mmprojPath
                Log.i(TAG, "Multimodal projector loaded: $mmprojPath")
            } else {
                Log.e(TAG, "Failed to load multimodal projector: $mmprojPath")
            }
            ok
        }
    }

    suspend fun unloadMultimodal() = withContext(Dispatchers.IO) {
        Log.d(TAG, "unloadMultimodal")
        lock.withLock {
            mtmdLib.release()
            currentMmprojPath = null
            Log.i(TAG, "Multimodal projector unloaded")
        }
    }

    suspend fun loadEmbedding(path: String, threads: Int = Runtime.getRuntime().availableProcessors() / 2, ctxSize: Int = 512): Boolean = withContext(Dispatchers.IO) {
        Log.d(TAG, "loadEmbedding: path=$path, threads=$threads, ctxSize=$ctxSize")
        lock.withLock {
            if (currentGenPath != null) {
                Log.d(TAG, "Unloading generation model first")
                stopGeneration()
                if (currentMmprojPath != null) {
                    mtmdLib.release()
                    currentMmprojPath = null
                }
                genLib.nativeRelease()
                currentGenPath = null
            }

            val ok = embedLib.loadModel(path, threads, ctxSize)
            if (ok) {
                currentEmbedPath = path
                Log.i(TAG, "Embedding model loaded: $path")
            } else {
                Log.e(TAG, "Failed to load embedding model: $path")
            }
            ok
        }
    }

    suspend fun unloadEmbedding() = withContext(Dispatchers.IO) {
        Log.d(TAG, "unloadEmbedding")
        lock.withLock {
            embedLib.release()
            currentEmbedPath = null
            Log.i(TAG, "Embedding model unloaded")
        }
    }

    suspend fun generateText(prompt: String, maxTokens: Int, toolsJson: String, callback: IGenerationCallback): Boolean {
        Log.d(TAG, "generateText: prompt=$prompt, maxTokens=$maxTokens, currentGenPath=$currentGenPath, isGenerating=${_isGenerating.get()}")

        if (currentGenPath == null) {
            Log.e(TAG, "Cannot generate: no model loaded")
            withContext(Dispatchers.Main) { callback.onError("No model loaded") }
            return false
        }

        if (_isGenerating.get()) {
            Log.w(TAG, "Cannot generate: already generating")
            withContext(Dispatchers.Main) { callback.onError("Already generating") }
            return false
        }

        activeCallback = callback
        generationJob = scope.launch {
            try {
                Log.i(TAG, "Starting text generation")
                _isGenerating.set(true)
                genLib.generateStreaming(prompt, maxTokens, callback, toolsJson)
                Log.i(TAG, "Text generation completed")
            } catch (e: Exception) {
                Log.e(TAG, "Generation error", e)
                withContext(Dispatchers.Main) { callback.onError("Native error: ${e.localizedMessage}") }
            } finally {
                Log.d(TAG, "Cleaning up generation state")
                _isGenerating.set(false)
                activeCallback = null
                generationJob = null
            }
        }
        return true
    }

    suspend fun generateWithImage(prompt: String, imageData: ByteArray, imageWidth: Int, imageHeight: Int, maxTokens: Int, toolsJson: String, callback: IGenerationCallback): Boolean {
        Log.d(TAG, "generateWithImage: prompt=$prompt, imageSize=${imageWidth}x$imageHeight, mtmdReady=${mtmdLib.isReady()}, isGenerating=${_isGenerating.get()}")

        if (!mtmdLib.isReady()) {
            Log.e(TAG, "Cannot generate: multimodal not ready")
            withContext(Dispatchers.Main) { callback.onError("Multimodal not ready") }
            return false
        }

        if (_isGenerating.get()) {
            Log.w(TAG, "Cannot generate: already generating")
            withContext(Dispatchers.Main) { callback.onError("Already generating") }
            return false
        }

        activeCallback = callback
        generationJob = scope.launch {
            try {
                Log.i(TAG, "Starting multimodal generation")
                _isGenerating.set(true)
                if (toolsJson.isNotEmpty()) genLib.setToolsJson(toolsJson)

                val streamCb = object : StreamCallback {
                    override fun onToken(token: String) {
                        Log.v(TAG, "Token: $token")
                        scope.launch(Dispatchers.Main) { callback.onToken(token) }
                    }
                    override fun onToolCall(name: String, argsJson: String) {
                        Log.d(TAG, "ToolCall: name=$name, args=$argsJson")
                        scope.launch(Dispatchers.Main) { callback.onToolCall(name, argsJson) }
                    }
                    override fun onDone() {
                        Log.i(TAG, "Generation done")
                        scope.launch(Dispatchers.Main) { callback.onDone() }
                    }
                    override fun onError(message: String) {
                        Log.e(TAG, "Generation error: $message")
                        scope.launch(Dispatchers.Main) { callback.onError(message) }
                    }
                }

                Log.d(TAG, "Calling native generateWithImage")
                val ok = mtmdLib.nativeGenerateStreamWithImage(prompt, imageData, imageWidth, imageHeight, maxTokens, streamCb)
                if (!ok) {
                    Log.e(TAG, "Native generation returned false")
                    withContext(Dispatchers.Main) { callback.onError("Native generation failed") }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Multimodal generation error", e)
                withContext(Dispatchers.Main) { callback.onError("Error: ${e.localizedMessage}") }
            } finally {
                Log.d(TAG, "Cleaning up multimodal generation state")
                _isGenerating.set(false)
                activeCallback = null
                generationJob = null
            }
        }
        return true
    }

    fun stopGeneration() {
        Log.d(TAG, "stopGeneration: job=$generationJob, isGenerating=${_isGenerating.get()}")
        generationJob?.cancel()
        genLib.nativeStopGeneration()
        _isGenerating.set(false)
        activeCallback = null
        generationJob = null
        Log.i(TAG, "Generation stopped")
    }

    suspend fun embed(text: String): FloatArray? = withContext(Dispatchers.IO) {
        lock.withLock {
            if (currentEmbedPath == null) null
            else embedLib.embed(text)
        }
    }

    fun setSystemPrompt(prompt: String) = genLib.setSystemPrompt(prompt)
    fun setChatTemplate(template: String) = genLib.setChatTemplate(template)
    fun setToolsJson(toolsJson: String) = genLib.setToolsJson(toolsJson)
    fun getModelInfo(): String = genLib.nativeGetModelInfo()

    suspend fun getStateSize(): Long = withContext(Dispatchers.IO) { stateLib.getStateSize() }
    suspend fun getStateData(): ByteArray? = withContext(Dispatchers.IO) { stateLib.getStateData() }
    suspend fun loadStateData(state: ByteArray): Boolean = withContext(Dispatchers.IO) { stateLib.loadStateData(state) }
    suspend fun saveStateFile(path: String): Boolean = withContext(Dispatchers.IO) { stateLib.saveSession(path) }
    suspend fun loadStateFile(path: String): Boolean = withContext(Dispatchers.IO) { stateLib.loadSession(path) }

    fun isMultimodalReady(): Boolean = mtmdLib.isReady()
    fun getMultimodalInfo(): String = if (mtmdLib.isReady()) mtmdLib.nativeGetMTMDInfo() else "{}"
    fun isGenerating(): Boolean = _isGenerating.get()
    fun isGenActive(): Boolean = currentGenPath != null
    fun isEmbedActive(): Boolean = currentEmbedPath != null
    fun isMtmdActive(): Boolean = currentMmprojPath != null
}