package com.mp.ai_core

import android.util.Log
import androidx.annotation.Keep

class MtmdLib private constructor() {

    companion object {
        @Volatile
        private var instance: MtmdLib? = null

        init {
            System.loadLibrary("ai_core")
        }

        fun getInstance(): MtmdLib =
            instance ?: synchronized(this) {
                instance ?: MtmdLib().also { instance = it }
            }
    }

    private var isInitialized = false

    // External natives
    external fun nativeInitMTMD(
        mmprojPath: String,
        useGpu: Boolean,
        nThreads: Int
    ): Boolean

    external fun nativeReleaseMTMD()
    external fun nativeIsMTMDReady(): Boolean
    external fun nativeGetMTMDInfo(): String

    external fun nativeStop()

    external fun nativeGenerateStreamWithImage(
        prompt: String,
        imageData: ByteArray,
        imageWidth: Int,
        imageHeight: Int,
        maxTokens: Int,
        callback: StreamCallback
    ): Boolean

    external fun nativeLoadImageFromFile(path: String): ByteArray?
    external fun nativeGetMediaMarker(): String

    // High-level init
    fun init(mmprojPath: String, nThreads: Int = 4): Boolean {
        if (isInitialized) {
            Log.w("MtmdLib", "Already initialized")
            return true
        }

        val ok = nativeInitMTMD(mmprojPath, useGpu = true, nThreads)
        if (ok) {
            isInitialized = true
            Log.i("MtmdLib", "MTMD initialized: $mmprojPath")
        } else {
            Log.e("MtmdLib", "MTMD init failed")
        }
        return ok
    }

    fun release() {
        nativeReleaseMTMD()
        isInitialized = false
    }

    fun isReady() = isInitialized && nativeIsMTMDReady()
}