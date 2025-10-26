package com.mp.ai_core.services

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.graphics.drawable.IconCompat
import com.mp.ai_core.R
import com.mp.ai_core.helpers.ModelSwapper
import kotlinx.coroutines.*

class GenerationService : Service() {

    private val svcScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val swapper = ModelSwapper(svcScope)

    private val binder = object : IGenerationService.Stub() {

        override fun loadTextGenerationModel(path: String, threads: Int, gpuLayers: Int, useMMap: Boolean, ctxSize: Int, temp: Float, topK: Int, topP: Float, minP: Float): Boolean {
            val deferred = CompletableDeferred<Boolean>()
            svcScope.launch {
                val result = swapper.loadGeneration(path, threads, ctxSize, temp, topK, topP, minP)
                deferred.complete(result)
            }
            return runBlocking { deferred.await() }
        }

        override fun unloadTextGenerationModel() {
            svcScope.launch { swapper.unloadGeneration() }
        }

        override fun loadMultimodalProjector(mmprojPath: String, threads: Int): Boolean {
            val deferred = CompletableDeferred<Boolean>()
            svcScope.launch {
                val result = swapper.loadMultimodal(mmprojPath, threads)
                deferred.complete(result)
            }
            return runBlocking { deferred.await() }
        }

        override fun unloadMultimodalProjector() {
            svcScope.launch { swapper.unloadMultimodal() }
        }

        override fun loadEmbedModel(path: String?, threads: Int, ctxSize: Int): Boolean {
            if (path.isNullOrEmpty()) return false
            val deferred = CompletableDeferred<Boolean>()
            svcScope.launch {
                val result = swapper.loadEmbedding(path, threads, ctxSize)
                deferred.complete(result)
            }
            return runBlocking { deferred.await() }
        }

        override fun unLoadEmbeddingModel() {
            svcScope.launch { swapper.unloadEmbedding() }
        }

        override fun generateText(prompt: String, maxTokens: Int, toolCallingJson: String, callback: IGenerationCallback): Boolean {
            svcScope.launch { swapper.generateText(prompt, maxTokens, toolCallingJson, callback) }
            return true
        }

        override fun generateWithImage(prompt: String, imageData: ByteArray, imageWidth: Int, imageHeight: Int, maxTokens: Int, toolCallingJson: String, callback: IGenerationCallback): Boolean {
            svcScope.launch { swapper.generateWithImage(prompt, imageData, imageWidth, imageHeight, maxTokens, toolCallingJson, callback) }
            return true
        }

        override fun stopTextGeneration() = swapper.stopGeneration()

        override fun embed(text: String?): FloatArray? {
            if (text.isNullOrEmpty()) return null
            val deferred = CompletableDeferred<FloatArray?>()
            svcScope.launch {
                val result = swapper.embed(text)
                deferred.complete(result)
            }
            return runBlocking { deferred.await() }
        }

        override fun isGenerating(): Boolean = swapper.isGenerating()
        override fun isMultimodalReady(): Boolean = swapper.isMultimodalReady()
        override fun getMultimodalInfo(): String = swapper.getMultimodalInfo()
        override fun getModelInfo(): String = swapper.getModelInfo()
        override fun setSystemPrompt(prompt: String) = swapper.setSystemPrompt(prompt)
        override fun setChatTemplate(template: String) = swapper.setChatTemplate(template)
        override fun setToolsJson(toolsJson: String) = swapper.setToolsJson(toolsJson)

        override fun getStateSize(): Long {
            val deferred = CompletableDeferred<Long>()
            svcScope.launch { deferred.complete(swapper.getStateSize()) }
            return runBlocking { deferred.await() }
        }

        override fun getStateData(): ByteArray? {
            val deferred = CompletableDeferred<ByteArray?>()
            svcScope.launch { deferred.complete(swapper.getStateData()) }
            return runBlocking { deferred.await() }
        }

        override fun loadStateData(state: ByteArray?): Boolean {
            if (state == null) return false
            val deferred = CompletableDeferred<Boolean>()
            svcScope.launch { deferred.complete(swapper.loadStateData(state)) }
            return runBlocking { deferred.await() }
        }

        override fun saveStateFile(filePath: String?): Boolean {
            if (filePath == null) return false
            val deferred = CompletableDeferred<Boolean>()
            svcScope.launch { deferred.complete(swapper.saveStateFile(filePath)) }
            return runBlocking { deferred.await() }
        }

        override fun loadStateFile(filePath: String?): Boolean {
            if (filePath == null) return false
            val deferred = CompletableDeferred<Boolean>()
            svcScope.launch { deferred.complete(swapper.loadStateFile(filePath)) }
            return runBlocking { deferred.await() }
        }
    }

    override fun onBind(intent: Intent?): IBinder = binder
    override fun onCreate() {
        super.onCreate()
        startForeground(1, buildNotification())
    }
    override fun onDestroy() {
        super.onDestroy()
        swapper.stopGeneration()
        svcScope.cancel()
    }

    private fun buildNotification(): Notification {
        val chId = "ai_core_service"
        val mgr = getSystemService(NotificationManager::class.java)
        val ch = NotificationChannel(chId, "AI Core Service", NotificationManager.IMPORTANCE_LOW)
        mgr.createNotificationChannel(ch)
        return NotificationCompat.Builder(this, chId)
            .setContentTitle("AI Core Service")
            .setContentText("LLM Engine ready…")
            .setSmallIcon(IconCompat.createWithResource(this, R.drawable.privicy))
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setOngoing(true)
            .build()
    }
}