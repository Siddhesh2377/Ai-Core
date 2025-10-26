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
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking

private const val TAG = "AudioService"

class AudioService : Service() {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    private val ttsEngine = TtsEngine(scope)
    private val sttEngine = STTEngine()

    // ----------------------------------------------------------------------------
    // AIDL binding implementation
    // ----------------------------------------------------------------------------
    @OptIn(ExperimentalCoroutinesApi::class)
    private val binder = object : IAudioService.Stub() {

        /* --------- TTS --------- */
        override fun initializeTts(
            modelDir: String,
            modelName: String,
            voices: String,
            dataDir: String,
        ): Boolean = runBlocking {
            Log.d(TAG, "initializeTts: $modelDir/$modelName")
            val d = CompletableDeferred<Boolean>()
            scope.launch {
                d.complete(
                    ttsEngine.initialize(
                        modelDir, modelName, voices, dataDir
                    )
                )
            }
            return@runBlocking d.await()
        }

        override fun releaseTts() {
            scope.launch {
                ttsEngine.release()
            }
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
            scope.launch {
                ttsEngine.stop()
            }
        }

        /* --------- STT --------- */
        override fun initializeStt(modelDir: String, modelType: Int, numThreads: Int): Boolean = runBlocking {
            Log.d(TAG, "initializeStt: $modelDir $modelType")
            val d = CompletableDeferred<Boolean>()
            scope.launch {
                d.complete(
                    sttEngine.initialize(
                        modelDir, modelType, null, numThreads
                    ).isSuccess
                )
            }
            return@runBlocking d.await()
        }

        override fun releaseStt() {
            scope.launch { sttEngine.release() }
        }

        override fun isSttReady(): Boolean = sttEngine.isReady()

        override fun transcribeFile(filePath: String, sampleRate: Int, callback: ISttCallback) {
            scope.launch {
                val result = sttEngine.transcribeFile(filePath).getOrElse { "ERROR: ${it.message}" }
                if (result.startsWith("ERROR:")) callback.onError(result)
                else callback.onResult(result)
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
                if (result.startsWith("ERROR:")) callback.onError(result)
                else callback.onResult(result)
            }
        }


        override fun getActiveStreamCount(): Int = sttEngine.activeStreamCount()

        override fun getCurrentModelType(): Int = sttEngine.getCurrentModelType()

        override fun getAudioInfo(): String = buildString {
            append("TTS Ready: ${ttsEngine.isReady()}\n")
            if (ttsEngine.isReady()) {
                append("Sample Rate: ${ttsEngine.getSampleRate()}\n")
                append("Speakers: ${ttsEngine.getNumSpeakers()}\n")
            }
            append("STT Ready: ${sttEngine.isReady()}\n")
            if (sttEngine.isReady()) {
                append("Active Streams: ${sttEngine.activeStreamCount()}\n")
                append("Model Type: ${sttEngine.getCurrentModelType()}")
            }
        }
    }

    // ----------------------------------------------------------------------------
    // Service lifecycle
    // ----------------------------------------------------------------------------
    override fun onBind(intent: Intent?): IBinder = binder

    override fun onCreate() {
        super.onCreate()
        startForeground(2, buildNotification())
        Log.i(TAG, "AudioService created")
    }

    override fun onDestroy() {
        super.onDestroy()
        ttsEngine.stop()
        scope.cancel()
        scope.launch { sttEngine.release() }
        Log.i(TAG, "AudioService destroyed")
    }

    // ----------------------------------------------------------------------------
    // Notification helper
    // ----------------------------------------------------------------------------
    private fun buildNotification(): Notification {
        val chId = "audio_service"
        val mgr = getSystemService(NotificationManager::class.java)
        val ch = NotificationChannel(chId, "Audio Service", NotificationManager.IMPORTANCE_LOW)
        mgr.createNotificationChannel(ch)

        return NotificationCompat.Builder(this, chId).setContentTitle("Audio Service")
            .setContentText("TTS/STT Engine ready")
            .setSmallIcon(IconCompat.createWithResource(this, R.drawable.privicy))
            .setPriority(NotificationCompat.PRIORITY_LOW).setOngoing(true).build()
    }
}