package com.mp.ai_core.stt

import android.content.Context
import android.content.res.AssetManager
import android.util.Log
import com.k2fsa.sherpa.onnx.*
import kotlinx.coroutines.*
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/**
 * High-performance singleton manager for Sherpa ONNX Speech-to-Text.
 * Thread-safe and optimized for use in Android .aar libraries.
 */
object SherpaSTTManager {
    private const val TAG = "SherpaSTTManager"
    private const val DEFAULT_SAMPLE_RATE = 16000
    private const val MIN_THREADS = 2
    private const val MAX_THREADS = 4

    // State management
    @Volatile
    private var recognizer: OfflineRecognizer? = null
    private val isInitialized = AtomicBoolean(false)
    private val isLoading = AtomicBoolean(false)
    private val activeStreams = AtomicInteger(0)

    // Configuration
    @Volatile
    private var currentModelType: Int = -1
    @Volatile
    private var assetManager: AssetManager? = null

    // Coroutine scope for async operations
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    // Synchronization lock
    private val lock = Any()

    /**
     * Current loading/initialization status
     */
    val status: STTStatus
        get() = when {
            isLoading.get() -> STTStatus.LOADING
            isInitialized.get() -> STTStatus.READY
            else -> STTStatus.UNINITIALIZED
        }

    /**
     * Check if the manager is ready to transcribe
     */
    val isReady: Boolean
        get() = isInitialized.get() && recognizer != null

    /**
     * Number of active transcription streams
     */
    val activeStreamCount: Int
        get() = activeStreams.get()

    /**
     * Initialize the STT engine with specified model.
     * This method is thread-safe and idempotent.
     *
     * @param context Application or Activity context
     * @param modelType Model type identifier (see getOfflineModelConfig) - only used if modelConfig is null
     * @param modelConfig Custom model configuration with file paths. If null, uses built-in models from assets
     * @param numThreads Number of threads for inference (2-4 recommended)
     * @param useAssets If true, load from assets; if false, load from file system
     * @return Result indicating success or failure
     */
    suspend fun initialize(
        modelDir: String,
        modelType: Int = 2,
        modelConfig: OfflineModelConfig? = null,
        numThreads: Int = MIN_THREADS,
    ): Result<Unit> = withContext(Dispatchers.IO) {
        // Quick check without locking
        if (isInitialized.get() && currentModelType == modelType && modelConfig == null) {
            Log.d(TAG, "Already initialized with model type $modelType")
            return@withContext Result.success(Unit)
        }

        // Check if already loading
        if (!isLoading.compareAndSet(false, true)) {
            Log.w(TAG, "Initialization already in progress")
            return@withContext Result.failure(
                IllegalStateException("Initialization already in progress")
            )
        }

        try {
            synchronized(lock) {
                // Double-check after acquiring lock
                if (isInitialized.get() && currentModelType == modelType && modelConfig == null) {
                    isLoading.set(false)
                    return@withContext Result.success(Unit)
                }

                Log.i(TAG, "Initializing STT with ${if (modelConfig != null) "custom config" else "model type: $modelType"}")

                // Release existing recognizer if switching models
                if (recognizer != null) {
                    Log.i(TAG, "Releasing existing recognizer")
                    releaseInternal()
                }

                // Get or use model configuration
                val finalModelConfig = modelConfig ?: run {
                    getOfflineModelConfig(type = modelType, modelDir = modelDir)
                        ?: return@withContext Result.failure(
                            IllegalArgumentException("Invalid model type: $modelType")
                        )
                }

                // Configure threading (clamp between MIN and MAX)
                finalModelConfig.numThreads = numThreads.coerceIn(MIN_THREADS, MAX_THREADS)

                // Create recognizer config
                val config = OfflineRecognizerConfig(
                    modelConfig = finalModelConfig,
                    decodingMethod = "greedy_search",
                    maxActivePaths = 4
                )

                // Create recognizer (with or without AssetManager)
                recognizer = OfflineRecognizer(
                    config = config
                )

                currentModelType = modelType
                isInitialized.set(true)

                Log.i(TAG, "STT initialized successfully (model: $modelType, threads: ${modelConfig?.numThreads})")
                Result.success(Unit)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize STT", e)
            isInitialized.set(false)
            recognizer = null
            Result.failure(e)
        } finally {
            isLoading.set(false)
        }
    }

    /**
     * Transcribe audio from a WAV file.
     *
     * @param filePath Path to WAV file
     * @param sampleRate Expected sample rate (default: 16000)
     * @return Result containing transcription text or error
     */
    suspend fun transcribeFile(
        filePath: String,
        sampleRate: Int = DEFAULT_SAMPLE_RATE
    ): Result<String> = withContext(Dispatchers.IO) {
        // Verify recognizer is valid
        val currentRecognizer = recognizer
        if (!ensureReady() || currentRecognizer == null) {
            Log.e(TAG, "Recognizer not ready: isInitialized=${isInitialized.get()}, recognizer=$currentRecognizer")
            return@withContext Result.failure(
                IllegalStateException("STT not initialized. Call initialize() first.")
            )
        }

        val file = File(filePath)
        if (!file.exists()) {
            return@withContext Result.failure(
                IllegalArgumentException("File not found: $filePath")
            )
        }

        if (file.length() < 44) {
            return@withContext Result.failure(
                IllegalArgumentException("Invalid WAV file (too small): $filePath")
            )
        }

        var stream: OfflineStream? = null

        try {
            // Create stream with null check
            stream = synchronized(lock) {
                if (currentRecognizer != recognizer || recognizer == null) {
                    Log.e(TAG, "Recognizer changed or became null during operation")
                    null
                } else {
                    try {
                        activeStreams.incrementAndGet()
                        recognizer?.createStream()
                    } catch (e: Exception) {
                        activeStreams.decrementAndGet()
                        Log.e(TAG, "Failed to create stream", e)
                        null
                    }
                }
            }

            if (stream == null) {
                return@withContext Result.failure(
                    IllegalStateException("Failed to create recognition stream")
                )
            }

            // Read audio data
            val waveData = WaveReader.readWave(filePath)

            if (waveData.samples.isEmpty()) {
                return@withContext Result.failure(
                    IllegalArgumentException("No audio samples in file")
                )
            }

            Log.d(TAG, "Transcribing ${waveData.samples.size} samples at ${waveData.sampleRate} Hz")

            // Process audio
            stream.acceptWaveform(waveData.samples, sampleRate)

            synchronized(lock) {
                recognizer?.decode(stream)
            }

            // Get result
            val result = synchronized(lock) {
                recognizer?.getResult(stream)
            }

            val text = result?.text?.trim() ?: ""

            Log.d(TAG, "Transcription result: ${if (text.isEmpty()) "(empty)" else text}")

            if (text.isEmpty()) {
                Result.success("") // Return empty string, not an error
            } else {
                Result.success(text)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Transcription error", e)
            Result.failure(e)
        } finally {
            stream?.release()
            activeStreams.decrementAndGet()
        }
    }

    /**
     * Transcribe audio samples directly.
     *
     * @param samples Audio samples (Float array)
     * @param sampleRate Sample rate of the audio
     * @return Result containing transcription text or error
     */
    suspend fun transcribeSamples(
        samples: FloatArray,
        sampleRate: Int = DEFAULT_SAMPLE_RATE
    ): Result<String> = withContext(Dispatchers.IO) {
        val currentRecognizer = recognizer
        if (!ensureReady() || currentRecognizer == null) {
            return@withContext Result.failure(
                IllegalStateException("STT not initialized")
            )
        }

        if (samples.isEmpty()) {
            return@withContext Result.failure(
                IllegalArgumentException("Empty audio samples")
            )
        }

        var stream: OfflineStream? = null

        try {
            stream = synchronized(lock) {
                if (currentRecognizer != recognizer || recognizer == null) {
                    null
                } else {
                    try {
                        activeStreams.incrementAndGet()
                        recognizer?.createStream()
                    } catch (e: Exception) {
                        activeStreams.decrementAndGet()
                        null
                    }
                }
            }

            if (stream == null) {
                return@withContext Result.failure(
                    IllegalStateException("Failed to create recognition stream")
                )
            }

            stream.acceptWaveform(samples, sampleRate)

            synchronized(lock) {
                recognizer?.decode(stream)
            }

            val result = synchronized(lock) {
                recognizer?.getResult(stream)
            }

            val text = result?.text?.trim() ?: ""

            Result.success(text)
        } catch (e: Exception) {
            Log.e(TAG, "Transcription error", e)
            Result.failure(e)
        } finally {
            stream?.release()
            activeStreams.decrementAndGet()
        }
    }

    /**
     * Transcribe with detailed result including timestamps and tokens.
     *
     * @param filePath Path to WAV file
     * @param sampleRate Expected sample rate
     * @return Result containing detailed transcription result
     */
    suspend fun transcribeDetailed(
        filePath: String,
        sampleRate: Int = DEFAULT_SAMPLE_RATE
    ): Result<TranscriptionResult> = withContext(Dispatchers.IO) {
        if (!ensureReady()) {
            return@withContext Result.failure(
                IllegalStateException("STT not initialized")
            )
        }

        val stream = createStream() ?: return@withContext Result.failure(
            IllegalStateException("Failed to create stream")
        )

        try {
            val waveData = WaveReader.readWave(filePath)
            stream.acceptWaveform(waveData.samples, sampleRate)
            recognizer?.decode(stream)

            val result = recognizer?.getResult(stream)

            if (result != null) {
                Result.success(
                    TranscriptionResult(
                        text = result.text.trim(),
                        tokens = result.tokens.toList(),
                        timestamps = result.timestamps.toList(),
                        language = result.lang,
                        emotion = result.emotion,
                        event = result.event,
                        durations = result.durations.toList()
                    )
                )
            } else {
                Result.failure(IllegalStateException("Failed to get result"))
            }
        } catch (e: Exception) {
            Log.e(TAG, "Detailed transcription error", e)
            Result.failure(e)
        } finally {
            stream.release()
            activeStreams.decrementAndGet()
        }
    }

    /**
     * Create a new stream for processing.
     * Internal use - manages stream count.
     */
    private fun createStream(): OfflineStream? {
        return try {
            synchronized(lock) {
                val rec = recognizer
                if (rec == null) {
                    Log.e(TAG, "Cannot create stream: recognizer is null")
                    return null
                }
                activeStreams.incrementAndGet()
                rec.createStream()
            }
        } catch (e: Exception) {
            activeStreams.decrementAndGet()
            Log.e(TAG, "Failed to create stream", e)
            null
        }
    }

    /**
     * Ensure the manager is ready for use.
     */
    private fun ensureReady(): Boolean {
        val ready = isInitialized.get() && recognizer != null
        if (!ready) {
            Log.w(TAG, "Manager not ready: isInitialized=${isInitialized.get()}, recognizer=${recognizer != null}")
        }
        return ready
    }

    /**
     * Update configuration for an already initialized recognizer.
     *
     * @param config New configuration
     * @return Result indicating success or failure
     */
    fun updateConfig(config: OfflineRecognizerConfig): Result<Unit> {
        if (!ensureReady()) {
            return Result.failure(
                IllegalStateException("STT not initialized")
            )
        }

        return try {
            synchronized(lock) {
                recognizer?.setConfig(config)
                Log.i(TAG, "Configuration updated successfully")
                Result.success(Unit)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to update config", e)
            Result.failure(e)
        }
    }

    /**
     * Release all resources.
     * Call this when STT is no longer needed or when app is destroyed.
     */
    fun release() {
        synchronized(lock) {
            if (!isInitialized.get()) {
                Log.d(TAG, "Already released or not initialized")
                return
            }

            Log.i(TAG, "Releasing STT resources...")
            releaseInternal()

            // Cancel all pending coroutines
            scope.coroutineContext.cancelChildren()

            Log.i(TAG, "STT resources released")
        }
    }

    /**
     * Internal release without locking (assumes lock is held)
     */
    private fun releaseInternal() {
        try {
            recognizer?.release()
            recognizer = null
            isInitialized.set(false)
            currentModelType = -1
            activeStreams.set(0)
            assetManager = null
        } catch (e: Exception) {
            Log.e(TAG, "Error during release", e)
        }
    }

    /**
     * Get current model type
     */
    fun getCurrentModelType(): Int = currentModelType

    /**
     * Get all available model types
     */
    fun getAvailableModels(): List<ModelInfo> = listOf(
        ModelInfo(0, "Paraformer Chinese", "zh"),
        ModelInfo(1, "Zipformer English Multi-dataset", "en"),
        ModelInfo(2, "Whisper Tiny English", "en"),
        ModelInfo(3, "Whisper Base English", "en"),
        ModelInfo(4, "Zipformer Chinese Wenetspeech", "zh"),
        ModelInfo(5, "Zipformer Chinese Multi", "zh"),
        // Add more as needed
    )
}

/**
 * Status enum for STT Manager
 */
enum class STTStatus {
    UNINITIALIZED,
    LOADING,
    READY,
    ERROR
}

/**
 * Detailed transcription result
 */
data class TranscriptionResult(
    val text: String,
    val tokens: List<String>,
    val timestamps: List<Float>,
    val language: String,
    val emotion: String,
    val event: String,
    val durations: List<Float>
)

/**
 * Model information
 */
data class ModelInfo(
    val type: Int,
    val name: String,
    val language: String
)

/**
 * Sealed class for model file paths
 */
sealed class ModelPaths {
    data class Paraformer(
        val modelPath: String,
        val tokensPath: String
    ) : ModelPaths()

    data class Whisper(
        val encoderPath: String,
        val decoderPath: String,
        val tokensPath: String,
        val language: String = "en",
        val task: String = "transcribe"
    ) : ModelPaths()

    data class Transducer(
        val encoderPath: String,
        val decoderPath: String,
        val joinerPath: String,
        val tokensPath: String
    ) : ModelPaths()

    data class NemoCTC(
        val modelPath: String,
        val tokensPath: String
    ) : ModelPaths()

    data class SenseVoice(
        val modelPath: String,
        val tokensPath: String,
        val language: String = "",
        val useITN: Boolean = true
    ) : ModelPaths()
}