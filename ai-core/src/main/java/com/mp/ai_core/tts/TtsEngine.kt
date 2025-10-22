package com.mp.ai_core.tts

import android.content.Context
import android.util.Log
import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.getOfflineTtsConfig
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Implementation of ITtsService.
 * This class encapsulates all TTS operations and manages the lifecycle.
 */
class TtsServiceImpl : ITtsService {

    private var tts: OfflineTts? = null
    private val stopped = AtomicBoolean(false)
    private var config: TtsConfig? = null

    companion object {
        private const val TAG = "TtsServiceImpl"
    }

    override suspend fun initialize(config: TtsConfig) {
        Log.i(TAG, "Initializing TTS with config")
        this.config = config

        if (tts != null) {
            Log.w(TAG, "TTS already initialized, releasing old instance")
            release()
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
        Log.i(TAG, "TTS initialized successfully")
    }

    override fun generateAudioStream(text: String, speakerId: Int): Flow<AudioChunk> = callbackFlow {
        val ttsInstance = tts ?: throw IllegalStateException("TTS not initialized")

        stopped.set(false)
        ttsInstance.currentSid = speakerId

        Log.i(TAG, "Starting audio generation for text: ${text.take(50)}...")

        fun callback(samples: FloatArray, progress: Float): Int {
            return if (!stopped.get()) {
                val samplesCopy = samples.copyOf()
                val sent = trySend(AudioChunk(samplesCopy, progress)).isSuccess
                if (sent) {
                    Log.d(TAG, "Sent audio chunk, progress: $progress")
                    1 // Continue generation
                } else {
                    Log.w(TAG, "Failed to send audio chunk")
                    0
                }
            } else {
                Log.i(TAG, "Generation stopped by user")
                0
            }
        }

        try {
            // This call blocks until generation is complete
            ttsInstance.generateWithCallback(
                text = text,
                callback = ::callback
            )
            Log.i(TAG, "Audio generation completed")
            close() // Close the flow when done
        } catch (e: Exception) {
            Log.e(TAG, "Error during audio generation", e)
            close(e) // Close with error
        }

        awaitClose {
            Log.d(TAG, "Flow closed")
        }
    }

    override fun getTtsInfo(): TtsInfo? {
        return tts?.let {
            TtsInfo(
                sampleRate = it.sampleRate(),
                numSpeakers = it.numSpeakers(),
                isReady = true
            )
        }
    }

    override fun stop() {
        Log.i(TAG, "Stopping TTS generation")
        stopped.set(true)
    }

    override fun release() {
        Log.i(TAG, "Releasing TTS resources")
        stopped.set(true)
        tts?.release()
        tts = null
        config = null
    }

    override fun isInitialized(): Boolean = tts != null
}
