package com.mp.ai_core.text

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
import com.mp.ai_core.helpers.ModelSwapper
import kotlinx.coroutines.*

private const val TAG = "GenerationService"

class GenerationService : Service() {

    /* 1️⃣  The generation lib (text‑generation) */
    private val lib = NativeLib.getInstance()

    /* 2️⃣  The “swap” helper that temporarily loads the embedding lib */
    private val svcScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val swapper = ModelSwapper(svcScope)

    private var currentGenPath: String? = null

    /* State helpers – give the compiler an explicit type to avoid recursion */
    private var activeCallback: IGenerationCallback? = null
    private var generationJob: Job? = null
    private val _isGenerating: Boolean
        get() = svcScope.coroutineContext[Job]?.isActive == true

    /* AIDL binder – explicit type for the anonymous inner class */
    private val binder: IGenerationService.Stub = object : IGenerationService.Stub() {

        /* ------------------------------------------------------------------
         *  Model management – the same as you had before
         * ------------------------------------------------------------------ */
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
            if (currentGenPath != null) return false
            val ok = lib.init(
                path, threads, ctxSize, temp, topK, topP, minP
            )
            if (ok) currentGenPath = path
            return ok
        }

        override fun unloadModel() {
            lib.nativeRelease()
            currentGenPath = null
        }

        /* ------------------------------------------------------------------
         *  Generation
         * ------------------------------------------------------------------ */
        override fun generate(
            prompt: String, maxTokens: Int, toolCallingJson: String, callback: IGenerationCallback
        ): Boolean {
            if (currentGenPath == null) return false
            if (_isGenerating) return false

            activeCallback = callback
            generationJob = svcScope.launch {
                try {
                    lib.generateStreaming(
                        prompt, maxTokens, activeCallback!!, toolsJson = toolCallingJson
                    )
                } catch (e: Throwable) {
                    Log.e(TAG, "Native generation error", e)
                    activeCallback?.onError("Native error: ${e.localizedMessage}")
                } finally {
                    activeCallback = null
                    generationJob = null
                }
            }
            return true
        }

        override fun stopGeneration() {
            generationJob?.cancel()
            lib.nativeStopGeneration()
            activeCallback?.onError("Generation cancelled")
            activeCallback = null
            generationJob = null
        }

        override fun isGenerating() = _isGenerating

        /* ------------------------------------------------------------------
         *  Configuration – forwarded directly to the lib
         * ------------------------------------------------------------------ */
        override fun setSystemPrompt(prompt: String) = lib.setSystemPrompt(prompt)
        override fun setChatTemplate(template: String) = lib.nativeSetChatTemplate(template)
        override fun setToolsJson(toolsJson: String) = lib.nativeSetToolsJson(toolsJson)

        /* ------------------------------------------------------------------
         *  State (meta) – the same as before
         * ------------------------------------------------------------------ */
        override fun getStateSize() =
            lib.nativeGetStateSize().also { Log.d(TAG, "State size: $it") }

        override fun getStateData(): ByteArray? =
            lib.nativeGetStateData().also { Log.d(TAG, "State data size: ${it?.size ?: 0}") }

        override fun loadStateData(state: ByteArray?): Boolean =
            (state != null && lib.nativeLoadStateData(state))

        override fun saveStateFile(filePath: String?): Boolean =
            (filePath?.let { lib.nativeSaveStateFile(it) } ?: false)

        override fun loadStateFile(filePath: String?): Boolean =
            (filePath?.let { lib.nativeLoadStateFile(it) } ?: false)

        /* ------------------------------------------------------------------
         *  **Embedding** – the big change!
         * ------------------------------------------------------------------ */
        override fun embed(text: String?): FloatArray? {
            if (text.isNullOrEmpty()) return null

            // Use a coroutine to call the suspend function
            val result = svcScope.async {
                swapper.usingEmbedding(embedPath = currentGenPath
                    ?: throw IllegalStateException("No model loaded – cannot embed")
                ) { embedLib ->
                    embedLib.embed(text)     // result type is FloatArray?
                }
            }

            // Block to get the result. This is on the IO dispatcher, so it shouldn't block the main thread.
            return runBlocking { result.await() }
        }

        /* ------------------------------------------------------------------
         *  Extra meta
         * ------------------------------------------------------------------ */
        override fun getModelInfo(): String = lib.nativeGetModelInfo()
    }

    /* ------------------------------------------------------------------ *//*  Service lifecycle
     * ------------------------------------------------------------------ */
    override fun onBind(intent: Intent?) = binder

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
        } finally {
            Log.i(TAG, "GenerationService destroyed")
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