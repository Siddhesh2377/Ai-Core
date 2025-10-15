package com.mp.ai_core.services

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.graphics.drawable.IconCompat
import com.mp.ai_core.NativeLib
import com.mp.ai_core.R
import kotlinx.coroutines.*

private const val TAG = "GenerationService"

class GenerationService : Service() {

    /* 1️⃣  One global instance per-process */
    private val lib = NativeLib.getGenerationInstance()

    /* 2️⃣  Service scope – all heavy work on Dispatchers.IO */
    private val svcScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    /* 3️⃣  Keep track whether a model is loaded */
    private var currentModelPath: String? = null

    /* 4️⃣  Binder implementation */
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
            if (currentModelPath != null) return false // only one model per service instance
            return try {
                val ok = lib.initModel(
                    path, threads, gpuLayers, useMMap, /*useMLOCK=*/false,
                    ctxSize, temp, topK, topP, minP
                )
                if (ok) currentModelPath = path
                ok
            } catch (e: Throwable) {
                Log.e(TAG, "initModel failed", e)
                false
            }
        }

        override fun unloadModel() {
            currentModelPath = null
            lib.nativeRelease()
        }

        override fun generate(
            prompt: String,
            maxTokens: Int,
            toolCallingJson: String,
            callback: IGenerationCallback
        ): Boolean {
            svcScope.launch {
                try {
                    lib.generateStreaming(prompt, maxTokens, callback, toolCallingJson)
                } catch (t: Throwable) {
                    Log.e(TAG, "nativeGenerateStream error", t)
                    callback.onError("Native error: ${t.localizedMessage}")
                }
            }
            return true
        }

        override fun stopGeneration() {
            svcScope.coroutineContext.cancelChildren()
            lib.nativeStopGeneration()
        }

        /* ---------- Configuration helpers ---------- */
        override fun setSystemPrompt(prompt: String) {
            lib.setSystemPrompt(prompt)
        }

        override fun setChatTemplate(template: String) {
            lib.nativeSetChatTemplate(template)
        }

        override fun setToolsJson(toolsJson: String) {
            lib.nativeSetToolsJson(toolsJson)
        }

        /* ---------- Embedding ---------- */
        override fun embed(text: String?): FloatArray? {
            return if (text.isNullOrEmpty()) null else lib.embed(text)
        }

        /* ---------- Meta ---------- */
        override fun getModelInfo(): String {
            return lib.nativeGetModelInfo()
        }
    }

    /* ----- Binder–to‑native wrapper ---- */
    private fun internalCallback(binderCb: IGenerationCallback) =
        object : com.mp.ai_core.StreamCallback {
            override fun onToken(token: String) = binderCb.onToken(token)
            override fun onToolCall(name: String, argsJson: String) =
                binderCb.onToolCall(name, argsJson)
            override fun onDone() = binderCb.onDone()
            override fun onError(message: String) = binderCb.onError(message)
        }

    /* ----- Service lifecycle ----- */
    override fun onBind(intent: Intent?) = binder

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "GenerationService up")
        startForeground(1, buildNotification("LLM Engine running…"))
    }

    override fun onDestroy() {
        super.onDestroy()
        lib.nativeRelease()
        svcScope.cancel()
        Log.i(TAG, "GenerationService terminated")
    }

    /* Notification helper */
    private fun buildNotification(content: String): Notification {
        val chId = "ai_core_service"
        val mgr = getSystemService(NotificationManager::class.java)
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            val ch = NotificationChannel(chId, "AI Core", NotificationManager.IMPORTANCE_LOW)
            mgr.createNotificationChannel(ch)
        }
        return NotificationCompat.Builder(this, chId)
            .setContentTitle("AI Core Service")
            .setContentText(content)
            .setSmallIcon(IconCompat.createWithResource(this, R.drawable.privicy))
            .build()
    }
}