package com.mp.ai_core

class MMNativeLib {

    init {
        System.loadLibrary("mm_ai_core")
    }

    external fun nativeMMInit(mainModelPath: String, mmModelPath: String, numThread: Int): Boolean

    external fun nativeMMGenerateStreaming(
        input: String,
        imagePath: String,
        maxTokens: Int,
        callback: MMGenerateCallback
    )


    external fun nativeMMFree()
}

interface MMGenerateCallback {
    fun onToken(token: String)
    fun onComplete()
}