package com.mp.ai_core.text

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.graphics.drawable.IconCompat
import com.mp.ai_core.*
import com.mp.ai_core.R
import com.mp.ai_core.helpers.ModelSwapper
import kotlinx.coroutines.*
import java.util.concurrent.atomic.AtomicBoolean

private const val TAG = "GenerationService"

class GenerationService : Service() {

    /* 1️⃣  The generation lib (text‑generation) */
    private val lib = NativeLib.getInstance()
    private val mtmdLib = MtmdLib.getInstance()  // ← ADD THIS

    /* 2️⃣  The “swap” helper that temporarily loads the embedding lib */
    private val svcScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val swapper = ModelSwapper(svcScope)

    private var currentGenPath: String? = null
    private var currentMmprojPath: String? = null

    /* State helpers – give the compiler an explicit type to avoid recursion */
    private var activeCallback: IGenerationCallback? = null
    private var generationJob: Job? = null
    private val _isGenerating = AtomicBoolean(false)

    /* AIDL binder – explicit type for the anonymous inner class */
    private val binder: IGenerationService.Stub = object : IGenerationService.Stub() {

        override fun loadModel(
            path: String,
            threads: Int,
            gpuLayers: Int,
            useMMap: Boolean,
            ctxSize: Int,
            temp: Float,
            topK: Int,
            topP: Float,
            minP: Float
        ): Boolean {
            val logTag = "loadModel"
            Log.d(TAG, "$logTag: path=$path, threads=$threads, gpuLayers=$gpuLayers, useMMap=$useMMap, ctxSize=$ctxSize, temp=$temp, topK=$topK, topP=$topP, minP=$minP")
            if (currentGenPath != null) {
                Log.w(TAG, "$logTag: Model already loaded at $currentGenPath")
                return false
            }
            val ok = lib.init(
                path, threads, ctxSize, temp, topK, topP, minP
            )
            if (ok) {
                currentGenPath = path
                Log.i(TAG, "$logTag: Model loaded successfully at $path")
            } else {
                Log.e(TAG, "$logTag: Failed to load model at $path")
            }
            return ok
        }

        override fun unloadModel() {
            Log.d(TAG, "unloadModel: Unloading model at $currentGenPath")
            lib.nativeRelease()
            currentGenPath = null
            Log.i(TAG, "unloadModel: Model unloaded")
        }

        override fun generate(
            prompt: String, maxTokens: Int, toolCallingJson: String, callback: IGenerationCallback
        ): Boolean {
            val logTag = "generate"
            Log.d(TAG, "$logTag: prompt=$prompt, maxTokens=$maxTokens, toolCallingJson=$toolCallingJson")
            if (currentGenPath == null) {
                Log.w(TAG, "$logTag: No model loaded – cannot generate")
                return false
            }
            if (_isGenerating.get()) {
                Log.w(TAG, "$logTag: Generation already in progress")
                return false
            }

            activeCallback = callback
            generationJob = svcScope.launch {
                try {
                    Log.d(TAG, "$logTag: Starting generation with prompt=$prompt")
                    _isGenerating.set(true)
                    lib.generateStreaming(
                        prompt, maxTokens, activeCallback!!, toolsJson = toolCallingJson
                    )
                    Log.d(TAG, "$logTag: Generation completed")
                } catch (e: Exception) {
                    Log.e(TAG, "$logTag: Native generation error", e)
                    activeCallback?.onError("Native error: ${e.localizedMessage}")
                } finally {
                    Log.d(TAG, "$logTag: Resetting generation state")
                    _isGenerating.set(false)
                    activeCallback = null
                    generationJob = null
                }
            }
            return true
        }

        override fun loadMultimodalProjector(mmprojPath: String, threads: Int): Boolean {
            Log.d(TAG, "loadMultimodalProjector: path=$mmprojPath, threads=$threads")

            if (!lib.isReady()) {
                Log.e(TAG, "Base model must be loaded first")
                return false
            }

            if (currentMmprojPath != null) {
                Log.w(TAG, "Projector already loaded at $currentMmprojPath")
                return false
            }

            val ok = mtmdLib.init(mmprojPath, threads)
            if (ok) {
                currentMmprojPath = mmprojPath
                Log.i(TAG, "Multimodal projector loaded: $mmprojPath")
            }
            return ok
        }

        override fun unloadMultimodalProjector() {
            Log.d(TAG, "unloadMultimodalProjector")
            mtmdLib.release()
            currentMmprojPath = null
        }

        override fun isMultimodalReady(): Boolean {
            return mtmdLib.isReady()
        }

        override fun getMultimodalInfo(): String {
            return if (mtmdLib.isReady()) {
                mtmdLib.nativeGetMTMDInfo()
            } else {
                "{}"
            }
        }

        override fun generateWithImage(
            prompt: String,
            imageData: ByteArray,
            imageWidth: Int,
            imageHeight: Int,
            maxTokens: Int,
            toolCallingJson: String,
            callback: IGenerationCallback
        ): Boolean {
            Log.d(TAG, "generateWithImage: prompt=$prompt, image=${imageWidth}x${imageHeight}")

            if (!mtmdLib.isReady()) {
                Log.e(TAG, "Multimodal not ready")
                return false
            }

            if (_isGenerating.get()) {
                Log.w(TAG, "Generation already in progress")
                return false
            }

            activeCallback = callback
            generationJob = svcScope.launch {
                try {
                    _isGenerating.set(true)

                    // Set tools if provided
                    if (toolCallingJson.isNotEmpty()) {
                        lib.setToolsJson(toolCallingJson)
                    }

                    // Create streaming callback
                    val streamCallback = object : StreamCallback {
                        override fun onToken(token: String) {
                            CoroutineScope(Dispatchers.Main.immediate).launch {
                                callback.onToken(token)
                            }
                        }

                        override fun onToolCall(name: String, argsJson: String) {
                            CoroutineScope(Dispatchers.Main.immediate).launch {
                                callback.onToolCall(name, argsJson)
                            }
                        }

                        override fun onDone() {
                            CoroutineScope(Dispatchers.Main.immediate).launch {
                                callback.onDone()
                            }
                        }

                        override fun onError(message: String) {
                            CoroutineScope(Dispatchers.Main.immediate).launch {
                                callback.onError(message)
                            }
                        }
                    }

                    // Call native
                    val ok = mtmdLib.nativeGenerateStreamWithImage(
                        prompt, imageData, imageWidth, imageHeight,
                        maxTokens, streamCallback
                    )

                    if (!ok) {
                        callback.onError("Native generation failed")
                    }

                } catch (e: Exception) {
                    Log.e(TAG, "generateWithImage error", e)
                    callback.onError("Error: ${e.localizedMessage}")
                } finally {
                    _isGenerating.set(false)
                    activeCallback = null
                    generationJob = null
                }
            }

            return true
        }

        override fun stopGeneration() {
            val logTag = "stopGeneration"
            Log.d(TAG, "$logTag: Stopping generation")
            generationJob?.cancel()
            lib.nativeStopGeneration()
            activeCallback?.onError("Generation cancelled")
            _isGenerating.set(false)
            activeCallback = null
            generationJob = null
            Log.d(TAG, "$logTag: Generation stopped")
        }


        override fun isGenerating(): Boolean {
            val isGenerating = _isGenerating
            Log.d(TAG, "isGenerating: $isGenerating")
            return isGenerating.get()
        }

        /* ------------------------------------------------------------------
         *  Configuration – forwarded directly to the lib
         * ------------------------------------------------------------------ */
        override fun setSystemPrompt(prompt: String) {
            Log.d(TAG, "setSystemPrompt: prompt=$prompt")
            lib.setSystemPrompt(prompt)
        }

        override fun setChatTemplate(template: String) {
            Log.d(TAG, "setChatTemplate: template=$template")
            lib.nativeSetChatTemplate(template)
        }

        override fun setToolsJson(toolsJson: String) {
            Log.d(TAG, "setToolsJson: toolsJson=$toolsJson")
            lib.nativeSetToolsJson(toolsJson)
        }

        /* ------------------------------------------------------------------
         *  State (meta) – the same as before
         * ------------------------------------------------------------------ */
        override fun getStateSize(): Long {
            val size = lib.nativeGetStateSize()
            Log.d(TAG, "getStateSize: size=$size")
            return size
        }

        override fun getStateData(): ByteArray? {
            val stateData = lib.nativeGetStateData()
            Log.d(TAG, "getStateData: stateData size=${stateData?.size ?: 0}")
            return stateData
        }

        override fun loadStateData(state: ByteArray?): Boolean {
            val result = (state != null && lib.nativeLoadStateData(state))
            Log.d(TAG, "loadStateData: result=$result")
            return result
        }

        override fun saveStateFile(filePath: String?): Boolean {
            val result = (filePath?.let { lib.nativeSaveStateFile(it) } ?: false)
            Log.d(TAG, "saveStateFile: filePath=$filePath, result=$result")
            return result
        }

        override fun loadStateFile(filePath: String?): Boolean {
            val result = (filePath?.let { lib.nativeLoadStateFile(it) } ?: false)
            Log.d(TAG, "loadStateFile: filePath=$filePath, result=$result")
            return result
        }

        /* ------------------------------------------------------------------
         *  **Embedding** – the big change!
         * ------------------------------------------------------------------ */
        override fun embed(text: String?): FloatArray? {
            val logTag = "embed"
            if (text.isNullOrEmpty()) {
                Log.w(TAG, "$logTag: text is null or empty")
                return null
            }

            // Use a coroutine to call the suspend function
            val result = svcScope.async {
                swapper.usingEmbedding(embedPath = currentGenPath
                    ?: throw IllegalStateException("No model loaded – cannot embed")
                ) { embedLib ->
                    embedLib.embed(text)     // result type is FloatArray?
                }
            }

            // Block to get the result. This is on the IO dispatcher, so it shouldn't block the main thread.
            return runBlocking {
                try {
                    val vector = result.await()
                    Log.d(TAG, "$logTag: Embedding successful: ${vector?.joinToString(", ")}")
                    vector
                } catch (e: Exception) {
                    Log.e(TAG, "$logTag: Error during embedding", e)
                    null
                }
            }
        }

        /* ------------------------------------------------------------------
         *  Extra meta
         * ------------------------------------------------------------------ */
        override fun getModelInfo(): String {
            val modelInfo = lib.nativeGetModelInfo()
            Log.d(TAG, "getModelInfo: modelInfo=$modelInfo")
            return modelInfo
        }
    }

    /* ------------------------------------------------------------------ *//*  Service lifecycle
     * ------------------------------------------------------------------ */
    override fun onBind(intent: Intent?): IBinder? {
        Log.d(TAG, "onBind: Binding service")
        return binder
    }

    override fun onCreate() {
        super.onCreate()
        startForeground(1, buildNotification())
        Log.i(TAG, "GenerationService created")
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            generationJob?.cancel()
            lib.nativeStopGeneration()
            lib.nativeRelease()
            svcScope.cancel()
            Log.i(TAG, "GenerationService destroyed")
        } catch (e: Exception) {
            Log.e(TAG, "Error during service destruction", e)
        }
    }
    /* ------------------------------------------------------------------ *//*  Notification helper
     * ------------------------------------------------------------------ */
    private fun buildNotification(): Notification {
        val chId = "ai_core_service"
        val mgr = getSystemService(NotificationManager::class.java)

        val ch = NotificationChannel(
            chId, "AI Core Service", NotificationManager.IMPORTANCE_LOW
        )
        mgr.createNotificationChannel(ch)

        return NotificationCompat.Builder(this, chId).setContentTitle("AI Core Service")
            .setContentText("LLM Engine ready…")
            .setSmallIcon(IconCompat.createWithResource(this, R.drawable.privicy))
            .setPriority(NotificationCompat.PRIORITY_LOW).setOngoing(true).build()
    }
}