// ISherpaSTTService.aidl
package com.mp.ai_core.stt;

// Place this file in: src/main/aidl/com/mp/ai_core/stt/ISherpaSTTService.aidl

/**
 * AIDL Interface for Sherpa STT Service
 * This enables cross-process communication between your .aar and main app
 */
interface ISherpaSTTService {

    /**
     * Initialize the STT engine
     * @param modelType Model type to use
     * @param numThreads Number of threads for inference
     * @return true if successful, false otherwise
     */
    boolean initialize(String modelDir, int modelType, int numThreads);

    /**
     * Check if the service is ready
     * @return true if initialized and ready
     */
    boolean isReady();

    /**
     * Get current status
     * @return 0=UNINITIALIZED, 1=LOADING, 2=READY, 3=ERROR
     */
    int getStatus();

    /**
     * Transcribe audio from file path
     * @param filePath Absolute path to WAV file
     * @param sampleRate Sample rate of audio
     * @return Transcription text or error message
     */
    String transcribeFile(String filePath, int sampleRate);

    /**
     * Transcribe audio samples
     * @param samples Float array of audio samples
     * @param sampleRate Sample rate of audio
     * @return Transcription text or error message
     */
    String transcribeSamples(in float[] samples, int sampleRate);

    /**
     * Get active stream count
     * @return Number of active transcription streams
     */
    int getActiveStreamCount();

    /**
     * Get current model type
     * @return Current model type or -1 if not initialized
     */
    int getCurrentModelType();

    /**
     * Release all resources
     */
    void release();

    /**
     * Check if specific model type is available
     * @param modelType Model type to check
     * @return true if available
     */
    boolean isModelAvailable(int modelType);
}