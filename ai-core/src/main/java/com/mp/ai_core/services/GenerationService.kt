package com.mp.ai_core.services

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.graphics.drawable.IconCompat
import com.mp.ai_core.NativeLib
import com.mp.ai_core.R
import kotlinx.coroutines.*
import java.io.File

private const val TAG = "GenerationService"

class GenerationService : Service() {

    /* 1️⃣  One global instance per-process */
    private val lib = NativeLib.getGenerationInstance()

    /* 2️⃣  Service scope – all heavy work on Dispatchers.IO */
    private val svcScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    /* 3️⃣  Keep track whether a model is loaded */
    private var currentModelPath: String? = null

    /* 4️⃣  Track active generation */
    private var activeCallback: IGenerationCallback? = null
    private var generationJob: Job? = null

    /* 5️⃣  Binder implementation */
    private val binder = object : IGenerationService.Stub() {

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
            if (currentModelPath != null) {
                Log.w(TAG, "Model already loaded: $currentModelPath")
                return false
            }

            return try {
                val ok = lib.initModel(
                    path, threads, gpuLayers, useMMap, /*useMLOCK=*/false,
                    ctxSize, temp, topK, topP, minP
                )
                if (ok) {
                    currentModelPath = path
                    Log.i(TAG, "Model loaded: $path")
                } else {
                    Log.e(TAG, "Failed to load model: $path")
                }
                ok
            } catch (e: Throwable) {
                Log.e(TAG, "initModel exception", e)
                false
            }
        }

        override fun unloadModel() {
            if (currentModelPath == null) {
                Log.w(TAG, "No model to unload")
                return
            }

            try {
                // Stop any active generation first
                if (isGenerating) {
                    stopGeneration()
                }

                lib.nativeRelease()
                currentModelPath = null
                Log.i(TAG, "Model unloaded")
            } catch (e: Throwable) {
                Log.e(TAG, "Error unloading model", e)
            }
        }

        override fun generate(
            prompt: String,
            maxTokens: Int,
            toolCallingJson: String,
            callback: IGenerationCallback
        ): Boolean {
            if (currentModelPath == null) {
                Log.e(TAG, "Cannot generate: no model loaded")
                callback.onError("No model loaded")
                return false
            }

            if (isGenerating) {
                Log.w(TAG, "Generation already in progress")
                callback.onError("Generation already in progress")
                return false
            }

            activeCallback = callback
            generationJob = svcScope.launch {
                try {
                    lib.generateStreaming(
                        prompt = prompt,
                        maxTokens = maxTokens,
                        callback = callback,
                        toolsJson = toolCallingJson
                    )
                } catch (e: CancellationException) {
                    Log.d(TAG, "Generation cancelled")
                    callback.onError("Generation cancelled by user")
                } catch (t: Throwable) {
                    Log.e(TAG, "Generation error", t)
                    callback.onError("Native error: ${t.localizedMessage}")
                } finally {
                    activeCallback = null
                    generationJob = null
                }
            }

            return true
        }

        override fun stopGeneration() {
            try {
                val cb = activeCallback
                val job = generationJob

                // Cancel coroutine
                job?.cancel()

                // Stop native generation
                lib.nativeStopGeneration()

                // Notify callback
                cb?.onError("Generation stopped by user")

                activeCallback = null
                generationJob = null

                Log.i(TAG, "Generation stopped")
            } catch (e: Throwable) {
                Log.e(TAG, "Error stopping generation", e)
            }
        }

        override fun isGenerating(): Boolean {
            return generationJob?.isActive == true
        }

        /* ---------- Configuration helpers ---------- */
        override fun setSystemPrompt(prompt: String) {
            try {
                lib.setSystemPrompt(prompt)
                Log.d(TAG, "System prompt set (${prompt.length} chars)")
            } catch (e: Throwable) {
                Log.e(TAG, "Error setting system prompt", e)
            }
        }

        override fun setChatTemplate(template: String) {
            try {
                lib.nativeSetChatTemplate(template)
                Log.d(TAG, "Chat template set (${template.length} chars)")
            } catch (e: Throwable) {
                Log.e(TAG, "Error setting chat template", e)
            }
        }

        override fun setToolsJson(toolsJson: String) {
            try {
                lib.nativeSetToolsJson(toolsJson)
                Log.d(TAG, "Tools JSON set (${toolsJson.length} chars)")
            } catch (e: Throwable) {
                Log.e(TAG, "Error setting tools JSON", e)
            }
        }

        /* ---------- State Management ---------- */
        override fun getStateSize(): Long {
            return try {
                if (currentModelPath == null) {
                    Log.w(TAG, "No model loaded, state size is 0")
                    0L
                } else {
                    val size = lib.nativeGetStateSize()
                    Log.d(TAG, "State size: $size bytes (${size / 1024} KB)")
                    size
                }
            } catch (e: Throwable) {
                Log.e(TAG, "Error getting state size", e)
                0L
            }
        }

        override fun getStateData(): ByteArray? {
            return try {
                if (currentModelPath == null) {
                    Log.w(TAG, "No model loaded, cannot get state data")
                    null
                } else {
                    val data = lib.nativeGetStateData()
                    if (data != null) {
                        Log.i(TAG, "State data retrieved: ${data.size} bytes")
                    } else {
                        Log.w(TAG, "State data is null")
                    }
                    data
                }
            } catch (e: Throwable) {
                Log.e(TAG, "Error getting state data", e)
                null
            }
        }

        override fun loadStateData(state: ByteArray?): Boolean {
            return try {
                if (currentModelPath == null) {
                    Log.e(TAG, "No model loaded, cannot load state")
                    false
                } else if (state == null || state.isEmpty()) {
                    Log.e(TAG, "Invalid state data")
                    false
                } else {
                    val ok = lib.nativeLoadStateData(state)
                    if (ok) {
                        Log.i(TAG, "State loaded successfully (${state.size} bytes)")
                    } else {
                        Log.e(TAG, "Failed to load state data")
                    }
                    ok
                }
            } catch (e: Throwable) {
                Log.e(TAG, "Error loading state data", e)
                false
            }
        }

        override fun saveStateFile(filePath: String?): Boolean {
            return try {
                if (currentModelPath == null) {
                    Log.e(TAG, "No model loaded, cannot save state")
                    false
                } else if (filePath.isNullOrBlank()) {
                    Log.e(TAG, "Invalid file path")
                    false
                } else {
                    val file = File(filePath)

                    // Create parent directories if needed
                    file.parentFile?.mkdirs()

                    val ok = lib.nativeSaveStateFile(filePath)
                    if (ok) {
                        Log.i(TAG, "State saved to: $filePath (${file.length()} bytes)")
                    } else {
                        Log.e(TAG, "Failed to save state to: $filePath")
                    }
                    ok
                }
            } catch (e: Throwable) {
                Log.e(TAG, "Error saving state file", e)
                false
            }
        }

        override fun loadStateFile(filePath: String?): Boolean {
            return try {
                if (currentModelPath == null) {
                    Log.e(TAG, "No model loaded, cannot load state")
                    false
                } else if (filePath.isNullOrBlank()) {
                    Log.e(TAG, "Invalid file path")
                    false
                } else {
                    val file = File(filePath)
                    if (!file.exists()) {
                        Log.e(TAG, "State file not found: $filePath")
                        return false
                    }

                    val ok = lib.nativeLoadStateFile(filePath)
                    if (ok) {
                        Log.i(TAG, "State loaded from: $filePath (${file.length()} bytes)")
                    } else {
                        Log.e(TAG, "Failed to load state from: $filePath")
                    }
                    ok
                }
            } catch (e: Throwable) {
                Log.e(TAG, "Error loading state file", e)
                false
            }
        }

        /* ---------- Embedding ---------- */
        override fun embed(text: String?): FloatArray? {
            return try {
                if (text.isNullOrEmpty()) {
                    Log.w(TAG, "Empty text for embedding")
                    null
                } else {
                    val embedding = lib.embed(text)
                    if (embedding != null) {
                        Log.d(TAG, "Embedding generated: ${embedding.size} dimensions")
                    } else {
                        Log.w(TAG, "Embedding is null")
                    }
                    embedding
                }
            } catch (e: Throwable) {
                Log.e(TAG, "Error generating embedding", e)
                null
            }
        }

        /* ---------- Meta ---------- */
        override fun getModelInfo(): String {
            return try {
                val info = lib.nativeGetModelInfo()
                Log.d(TAG, "Model info: $info")
                info
            } catch (e: Throwable) {
                Log.e(TAG, "Error getting model info", e)
                "{}"
            }
        }
    }

    /* ----- Service lifecycle ----- */
    override fun onBind(intent: Intent?) = binder

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "GenerationService created")
        startForeground(1, buildNotification())
    }

    override fun onDestroy() {
        super.onDestroy()

        try {
            // Stop any active generation
            if (generationJob?.isActive == true) {
                generationJob?.cancel()
                lib.nativeStopGeneration()
            }

            // Release model
            lib.nativeRelease()

            // Cancel scope
            svcScope.cancel()

            Log.i(TAG, "GenerationService destroyed")
        } catch (e: Throwable) {
            Log.e(TAG, "Error during cleanup", e)
        }
    }

    /* Notification helper */
    private fun buildNotification(): Notification {
        val chId = "ai_core_service"
        val mgr = getSystemService(NotificationManager::class.java)

        val ch = NotificationChannel(
            chId,
            "AI Core Service",
            NotificationManager.IMPORTANCE_LOW
        )
        mgr.createNotificationChannel(ch)

        return NotificationCompat.Builder(this, chId)
            .setContentTitle("AI Core Service")
            .setContentText("LLM Engine ready...")
            .setSmallIcon(IconCompat.createWithResource(this, R.drawable.privicy))
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setOngoing(true)
            .build()
    }
}