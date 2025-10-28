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
import com.mp.ai_core.R
import com.mp.ai_core.audio.stt.STTEngine
import com.mp.ai_core.audio.tts.TtsEngine
import kotlinx.coroutines.*
import kotlinx.coroutines.future.asCompletableFuture
import org.json.JSONObject

private const val TAG = "AudioService"

class AudioService : Service() {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val ttsEngine = TtsEngine(scope)
    private val sttEngine = STTEngine()

    private val binder = object : IAudioService.Stub() {

        //region TTS

        override fun initializeTts(
            modelDir: String,
            modelName: String,
            voices: String,
            dataDir: String
        ): Boolean {
            val deferred = CompletableDeferred<Boolean>()
            scope.launch {
                deferred.complete(
                    ttsEngine.initialize(modelDir, modelName, voices, dataDir)
                )
            }
            return deferred.asCompletableFuture().get()
        }

        override fun releaseTts() {
            scope.launch { ttsEngine.release() }
        }

        override fun isTtsReady(): Boolean = ttsEngine.isReady()

        override fun getTtsSampleRate(): Int = ttsEngine.getSampleRate()

        override fun getTtsNumSpeakers(): Int = ttsEngine.getNumSpeakers()

        override fun generateTts(text: String, speakerId: Int, callback: IAudioCallback) {
            scope.launch {
                ttsEngine.generate(text, speakerId, callback)
            }
        }

        override fun stopTts() {
            scope.launch { ttsEngine.stop() }
        }

        //endregion

        //region STT

        override fun initializeStt(modelDir: String, modelType: Int, numThreads: Int): Boolean {
            val deferred = CompletableDeferred<Boolean>()
            scope.launch {
                deferred.complete(
                    sttEngine.initialize(modelDir, modelType, null, numThreads).isSuccess
                )
            }
            return deferred.asCompletableFuture().get()
        }

        override fun releaseStt() {
            scope.launch { sttEngine.release() }
        }

        override fun isSttReady(): Boolean = sttEngine.isReady()

        override fun transcribeFile(filePath: String, sampleRate: Int, callback: ISttCallback) {
            scope.launch {
                val result = sttEngine.transcribeFile(filePath)
                    .getOrElse { "ERROR: ${it.message}" }

                if (result.startsWith("ERROR:")) {
                    callback.onError(result)
                } else {
                    callback.onResult(result)
                }
            }
        }

        override fun transcribeSamples(
            samples: FloatArray,
            sampleRate: Int,
            callback: ISttCallback
        ) {
            scope.launch {
                val result = sttEngine.transcribeSamples(samples, sampleRate)
                    .getOrElse { "ERROR: ${it.message}" }

                if (result.startsWith("ERROR:")) {
                    callback.onError(result)
                } else {
                    callback.onResult(result)
                }
            }
        }

        override fun getActiveStreamCount(): Int = sttEngine.activeStreamCount()

        override fun getCurrentModelType(): Int = sttEngine.getCurrentModelType()

        //endregion

        //region Info

        override fun getAudioInfo(): String {
            val json = JSONObject()

            json.put("tts", JSONObject().apply {
                put("ready", ttsEngine.isReady())
                if (ttsEngine.isReady()) {
                    put("sample_rate", ttsEngine.getSampleRate())
                    put("speakers", ttsEngine.getNumSpeakers())
                }
            })

            json.put("stt", JSONObject().apply {
                put("ready", sttEngine.isReady())
                if (sttEngine.isReady()) {
                    put("active_streams", sttEngine.activeStreamCount())
                    put("model_type", sttEngine.getCurrentModelType())
                }
            })

            return json.toString()
        }

        //endregion
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onCreate() {
        super.onCreate()
        startForeground(2, buildNotification())
        Log.i(TAG, "AudioService created")
    }

    override fun onDestroy() {
        super.onDestroy()
        scope.launch {
            ttsEngine.stop()
            sttEngine.release()
        }
        scope.cancel()
        Log.i(TAG, "AudioService destroyed")
    }

    private fun buildNotification(): Notification {
        val channelId = "audio_service"
        val manager = getSystemService(NotificationManager::class.java)
        val channel = NotificationChannel(
            channelId,
            "Audio Service",
            NotificationManager.IMPORTANCE_LOW
        )
        manager.createNotificationChannel(channel)

        return NotificationCompat.Builder(this, channelId)
            .setContentTitle("Audio Service")
            .setContentText("TTS/STT Engine ready")
            .setSmallIcon(IconCompat.createWithResource(this, R.drawable.privicy))
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setOngoing(true)
            .build()
    }

    private fun <T> Deferred<T>.await(): T = kotlinx.coroutines.runBlocking { await() }
}