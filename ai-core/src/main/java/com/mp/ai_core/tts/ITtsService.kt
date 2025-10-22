package com.mp.ai_core.tts

import kotlinx.coroutines.flow.Flow

/**
 * Main interface for TTS operations.
 * This is the contract between your app and the TTS module.
 */
interface ITtsService {
    /**
     * Initialize the TTS engine with configuration
     */
    suspend fun initialize(config: TtsConfig)

    /**
     * Generate audio with streaming samples
     * @param text Text to convert to speech
     * @param speakerId Voice ID to use
     * @return Flow of audio samples as they're generated
     */
    fun generateAudioStream(text: String, speakerId: Int): Flow<AudioChunk>

    /**
     * Get TTS information
     */
    fun getTtsInfo(): TtsInfo?

    /**
     * Stop current generation
     */
    fun stop()

    /**
     * Release resources
     */
    fun release()

    /**
     * Check if TTS is initialized
     */
    fun isInitialized(): Boolean
}

/**
 * Represents a chunk of audio data
 */
data class AudioChunk(
    val samples: FloatArray,
    val progress: Float
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as AudioChunk
        if (!samples.contentEquals(other.samples)) return false
        if (progress != other.progress) return false
        return true
    }

    override fun hashCode(): Int {
        var result = samples.contentHashCode()
        result = 31 * result + progress.hashCode()
        return result
    }
}

/**
 * TTS engine information
 */
data class TtsInfo(
    val sampleRate: Int,
    val numSpeakers: Int,
    val isReady: Boolean
)

/**
 * Configuration for TTS
 */
data class TtsConfig(
    val modelDir: String? = null,
    val modelName: String? = null,
    val acousticModelName: String? = null,
    val vocoder: String? = null,
    val voices: String? = null,
    val ruleFsts: String? = null,
    val ruleFars: String? = null,
    val lexicon: String? = null,
    val dataDir: String? = null,
    val lang: String? = null,
    val lang2: String? = null,
    val isKitten: Boolean? = null
)