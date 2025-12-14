package com.mp.ai_core.audio.stt

import android.util.Log
import com.k2fsa.sherpa.onnx.*
import kotlinx.coroutines.*
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File
import java.nio.charset.Charset
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlin.coroutines.CoroutineContext

private const val TAG = "STTEngine"

class STTEngine(
    private val scope: CoroutineContext = Dispatchers.IO
) {

    private val mutex = Mutex()
    private var recognizer: OfflineRecognizer? = null
    private val isLoading = AtomicBoolean(false)
    private val activeStreams = AtomicInteger(0)

    /** Current model type (as used in getOfflineModelConfig) */
    private var currentModelType: Int = -1

    /** Status of the engine */
    enum class Status { UNINITIALIZED, LOADING, READY, ERROR }

    /** Result of a detailed transcription */
    data class DetailedResult(
        val text: String,
        val tokens: List<String>,
        val timestamps: List<Float>,
        val language: String,
        val emotion: String,
        val event: String,
        val durations: List<Float>
    )

    // ------------------------------------------------------------------
    // init / release
    // ------------------------------------------------------------------
    suspend fun initialize(
        modelDir: String,
        encoder: String,
        decoder: String,
        tokens: String,
        numThreads: Int = 2
    ): Result<Unit> = withContext(scope) {
        if (isLoading.get()) return@withContext Result.failure(
            IllegalStateException("Initialization already in progress")
        )

        isLoading.set(true)
        try {
            mutex.withLock {
                val config = OfflineRecognizerConfig(
                    featConfig = FeatureConfig(),
                    modelConfig = OfflineModelConfig(
                        whisper = OfflineWhisperModelConfig(
                            encoder = "$modelDir/$encoder",
                            decoder = "$modelDir/$decoder",
                        ),
                        tokens = "$modelDir/$tokens",
                        modelType = "whisper",
                        numThreads = numThreads,
                    ),

                    decodingMethod = "greedy_search"
                )

                recognizer?.release()
                recognizer = OfflineRecognizer(config = config)
            }
            Result.success(Unit)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize STT", e)
            Result.failure(e)
        } finally {
            isLoading.set(false)
        }
    }

    suspend fun release() {
        mutex.withLock {
            recognizer?.release()
            recognizer = null
            currentModelType = -1
            activeStreams.set(0)
        }
    }

    fun isReady() = recognizer != null

    // ------------------------------------------------------------------
    // transcription (file & raw samples)
    // ------------------------------------------------------------------
    suspend fun transcribeFile(path: String): Result<String> = withContext(scope) {
        val rec = recognizer ?: return@withContext Result.failure(
            IllegalStateException("STT not initialized")
        )

        val file = File(path)
        if (!file.exists())
            return@withContext Result.failure(IllegalArgumentException("File not found"))

        val stream = rec.createStream()
        activeStreams.incrementAndGet()
        try {
            val wave = WaveReader.readWave(path)
            if (wave.samples.isEmpty())
                return@withContext Result.failure(IllegalArgumentException("Empty audio"))

            stream.acceptWaveform(wave.samples, wave.sampleRate)
            rec.decode(stream)
            val result = rec.getResult(stream)
            Result.success(result.text.trim())
        } catch (e: Exception) {
            Result.failure(e)
        } finally {
            stream.release()
            activeStreams.decrementAndGet()
        }
    }

    suspend fun transcribeSamples(samples: FloatArray, rate: Int = 16000): Result<String> =
        withContext(scope) {
            val rec = recognizer ?: return@withContext Result.failure(
                IllegalStateException("STT not initialized")
            )
            val stream = rec.createStream()
            activeStreams.incrementAndGet()
            try {
                stream.acceptWaveform(samples, rate)
                rec.decode(stream)
                val result = rec.getResult(stream)
                Result.success(result.text.trim())
            } catch (e: Exception) {
                Result.failure(e)
            } finally {
                stream.release()
                activeStreams.decrementAndGet()
            }
        }

    suspend fun transcribeDetailed(path: String): Result<DetailedResult> =
        withContext(scope) {
            val rec = recognizer ?: return@withContext Result.failure(
                IllegalStateException("STT not initialized")
            )
            val stream = rec.createStream()
            activeStreams.incrementAndGet()
            try {
                val wav = WaveReader.readWave(path)
                stream.acceptWaveform(wav.samples, wav.sampleRate)
                rec.decode(stream)
                val res = rec.getResult(stream)
                Result.success(
                    DetailedResult(
                        text = res.text.trim(),
                        tokens = res.tokens.toList(),
                        timestamps = res.timestamps.toList(),
                        language = res.lang,
                        emotion = res.emotion,
                        event = res.event,
                        durations = res.durations.toList()
                    )
                )
            } catch (e: Exception) {
                Result.failure(e)
            } finally {
                stream.release()
                activeStreams.decrementAndGet()
            }
        }

    // ------------------------------------------------------------------
    // state query helpers
    // ------------------------------------------------------------------
    fun activeStreamCount() = activeStreams.get()
    fun getCurrentModelType() = currentModelType
    fun status() = when {
        recognizer == null -> Status.UNINITIALIZED
        isLoading.get()   -> Status.LOADING
        else              -> Status.READY
    }
}