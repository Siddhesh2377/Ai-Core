package com.mp.ai_core.tts

import android.util.Log
import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.getOfflineTtsConfig
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Implementation of ITtsService.
 * Handles TTS lifecycle, synchronization, and streaming output.
 */
class TtsServiceImpl : ITtsService {

    private var tts: OfflineTts? = null
    private val stopped = AtomicBoolean(false)
    private var config: TtsConfig? = null
    private val ttsLock = Mutex()
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    companion object {
        private const val TAG = "TtsServiceImpl"
    }

    override suspend fun initialize(config: TtsConfig) = ttsLock.withLock {
        if (tts != null) {
            Log.w(TAG, "TTS already initialized — releasing old instance")
            releaseUnsafe()
        }

        val offlineConfig = getOfflineTtsConfig(
            modelDir = config.modelDir ?: "",
            modelName = config.modelName ?: "",
            acousticModelName = config.acousticModelName ?: "",
            vocoder = config.vocoder ?: "",
            voices = config.voices ?: "",
            lexicon = config.lexicon ?: "",
            dataDir = config.dataDir ?: "",
            dictDir = "",
            ruleFsts = config.ruleFsts ?: "",
            ruleFars = config.ruleFars ?: "",
            isKitten = config.isKitten ?: false,
        )

        tts = OfflineTts(config = offlineConfig)
        stopped.set(false)
        this.config = config
        Log.i(TAG, "✅ TTS initialized successfully")
    }

    override fun generateAudioStream(text: String, speakerId: Int): Flow<AudioChunk> = callbackFlow {
        val job = serviceScope.launch {
            ttsLock.withLock {
                val ttsInstance = tts ?: throw IllegalStateException("TTS not initialized")
                stopped.set(false)
                ttsInstance.currentSid = speakerId

                fun callback(samples: FloatArray, progress: Float): Int {
                    if (stopped.get()) return 0
                    trySend(AudioChunk(samples.copyOf(), progress))
                    return 1
                }

                try {
                    ttsInstance.generateWithCallback(text, ::callback)
                } catch (e: Exception) {
                    Log.e(TAG, "Error during TTS generation", e)
                    close(e)
                } finally {
                    close()
                }
            }
        }

        awaitClose {
            stopped.set(true)
            job.cancel()
        }
    }

    override fun getTtsInfo(): TtsInfo? = tts?.let {
        TtsInfo(
            sampleRate = it.sampleRate(),
            numSpeakers = it.numSpeakers(),
            isReady = true
        )
    }

    override fun stop() {
        stopped.set(true)
    }

    override fun release() {
        serviceScope.launch {
            ttsLock.withLock { releaseUnsafe() }
        }
    }

    private fun releaseUnsafe() {
        stopped.set(true)
        tts?.release()
        tts = null
        config = null
        Log.i(TAG, "🧹 TTS released safely")
    }

    override fun isInitialized(): Boolean = tts != null
}
