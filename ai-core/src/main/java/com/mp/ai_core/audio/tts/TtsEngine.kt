package com.mp.ai_core.audio.tts

import android.util.Log
import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.getOfflineTtsConfig
import com.mp.ai_core.services.IAudioCallback
import kotlinx.coroutines.*
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.concurrent.atomic.AtomicBoolean

private const val TAG = "TtsEngine"

class TtsEngine(private val scope: CoroutineScope) {

    private val lock = Mutex()
    private var tts: OfflineTts? = null
    private val isStopped = AtomicBoolean(false)
    private var generationJob: Job? = null

    suspend fun initialize(
        modelDir: String,
        modelName: String,
        voices: String,
        dataDir: String,
    ): Boolean = withContext(Dispatchers.IO) {
        lock.withLock {
            Log.d(TAG, "Initializing TTS: modelDir=$modelDir, modelName=$modelName")
            
            if (tts != null) {
                Log.w(TAG, "TTS already initialized, releasing old instance")
                releaseUnsafe()
            }

            try {
                val config = getOfflineTtsConfig(
                    modelDir = modelDir,
                    modelName = modelName,
                    acousticModelName = "",
                    vocoder = "",
                    voices = voices,
                    lexicon = "",
                    dataDir = dataDir,
                    dictDir = "",
                    ruleFsts = "",
                    ruleFars = "",
                    isKitten = false
                )

                tts = OfflineTts(config = config)
                isStopped.set(false)
                Log.i(TAG, "TTS initialized successfully")
                true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize TTS", e)
                false
            }
        }
    }

    suspend fun generate(text: String, speakerId: Int, callback: IAudioCallback) {
        if (!isReady()) {
            withContext(Dispatchers.Main) { callback.onError("TTS not initialized") }
            return
        }

        generationJob = scope.launch(Dispatchers.IO) {
            lock.withLock {
                val ttsInstance = tts ?: run {
                    withContext(Dispatchers.Main) { callback.onError("TTS instance is null") }
                    return@launch
                }

                try {
                    Log.d(TAG, "Starting TTS generation: text=$text, speakerId=$speakerId")
                    isStopped.set(false)
                    ttsInstance.currentSid = speakerId

                    val audioCallback = { samples: FloatArray, progress: Float ->
                        if (isStopped.get()) {
                            0
                        } else {
                            try {
                                callback.onAudioChunk(samples, progress)
                                1
                            } catch (e: Exception) {
                                Log.e(TAG, "Callback error", e)
                                0
                            }
                        }
                    }

                    ttsInstance.generateWithCallback(text, audioCallback)
                    
                    if (!isStopped.get()) {
                        withContext(Dispatchers.Main) { callback.onComplete() }
                        Log.i(TAG, "TTS generation completed")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "TTS generation error", e)
                    withContext(Dispatchers.Main) { callback.onError("Generation failed: ${e.message}") }
                }
            }
        }
    }

    fun stop() {
        Log.d(TAG, "Stopping TTS generation")
        isStopped.set(true)
        generationJob?.cancel()
        generationJob = null
    }

    suspend fun release() = withContext(Dispatchers.IO) {
        lock.withLock { releaseUnsafe() }
    }

    private fun releaseUnsafe() {
        Log.d(TAG, "Releasing TTS")
        stop()
        tts?.release()
        tts = null
        Log.i(TAG, "TTS released")
    }

    fun isReady(): Boolean = tts != null

    fun getSampleRate(): Int = tts?.sampleRate() ?: 0

    fun getNumSpeakers(): Int = tts?.numSpeakers() ?: 0
}