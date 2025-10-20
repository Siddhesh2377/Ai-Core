package com.mp.ai_core.stt

import android.content.res.AssetManager
import android.util.Log
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.getOfflineModelConfig

object SimulateStreamingAsr {
    private const val TAG = "SimulateStreamingAsr"
    private const val DEFAULT_MODEL_TYPE = 2
    private const val DEFAULT_NUM_THREADS = 2

    @Volatile
    private var _recognizer: OfflineRecognizer? = null

    val recognizer: OfflineRecognizer
        get() = _recognizer ?: throw IllegalStateException(
            "Recognizer not initialized. Call initOfflineRecognizer() first."
        )

    val isInitialized: Boolean
        get() = _recognizer != null

    /**
     * Initialize the offline recognizer with the specified model type.
     * This method is thread-safe and will only initialize once.
     * 
     * @param assetManager AssetManager to access model files from assets
     * @param modelType Model type to use (default: 2)
     * @param numThreads Number of threads for inference (default: 2)
     * @throws IllegalArgumentException if model config is not available for the specified type
     */
    fun initOfflineRecognizer(
        assetManager: AssetManager,
        modelType: Int = DEFAULT_MODEL_TYPE,
        numThreads: Int = DEFAULT_NUM_THREADS
    ) {
        if (_recognizer != null) {
            Log.d(TAG, "Recognizer already initialized, skipping")
            return
        }

        synchronized(this) {
            // Double-check locking
            if (_recognizer != null) {
                return
            }

            Log.i(TAG, "Initializing sherpa-onnx offline recognizer with model type: $modelType")

            val modelConfig = getOfflineModelConfig(type = modelType)
                ?: throw IllegalArgumentException("Model config not available for type: $modelType")

            // Set number of threads (minimum 2 for better performance)
            modelConfig.numThreads = maxOf(numThreads, 2)

            val config = OfflineRecognizerConfig(modelConfig = modelConfig)

            try {
                _recognizer = OfflineRecognizer(
                    assetManager = assetManager,
                    config = config
                )
                Log.i(TAG, "Sherpa-onnx offline recognizer initialized successfully (threads: ${modelConfig.numThreads})")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize recognizer", e)
                throw e
            }
        }
    }

    /**
     * Release the recognizer resources.
     * Call this when the recognizer is no longer needed.
     */
    fun release() {
        synchronized(this) {
            _recognizer?.release()
            _recognizer = null
            Log.i(TAG, "Recognizer released")
        }
    }

    /**
     * Check if recognizer is ready to use.
     */
    fun ensureInitialized() {
        if (_recognizer == null) {
            throw IllegalStateException(
                "Recognizer not initialized. Call initOfflineRecognizer() first."
            )
        }
    }
}