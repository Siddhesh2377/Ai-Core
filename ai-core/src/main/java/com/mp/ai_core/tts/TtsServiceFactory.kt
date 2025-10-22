package com.mp.ai_core.tts

/**
 * Factory to create TTS service instances.
 * Use this in your app to get the service.
 */
object TtsServiceFactory {
    fun createTtsService(): ITtsService {
        return TtsServiceImpl()
    }
}