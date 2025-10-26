package com.mp.ai_core

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class StateLib private constructor() {

    companion object {
        private const val TAG = "StateLibJNI"

        init { System.loadLibrary("ai_core") }

        /**
         * Return the one and only JNI wrapper instance.
         */
        @Synchronized
        fun getInstance(): StateLib = StateLib()
    }

    external fun nativeGetStateData(): ByteArray?

    /** Load a previous embedding session. */
    external fun nativeLoadStateFile(path: String): Boolean

    /** Save the current embedding session. */
    external fun nativeSaveStateFile(path: String): Boolean

    /** Return length of the current state buffer. */
    external fun nativeGetStateSize(): Long

    /** Replace the current state buffer with the supplied data. */
    external fun nativeLoadStateData(state: ByteArray): Boolean


    /**
     * Save and load the embedding KV‑cache/session.
     */
    suspend fun saveSession(path: String): Boolean = withContext(Dispatchers.IO) {
        try { nativeSaveStateFile(path) }
        catch (t: Throwable) { Log.e(TAG, "saveSession error", t); false }
    }

    suspend fun loadSession(path: String): Boolean = withContext(Dispatchers.IO) {
        try { nativeLoadStateFile(path) }
        catch (t: Throwable) { Log.e(TAG, "loadSession error", t); false }
    }

    suspend fun getStateData(): ByteArray? = withContext(Dispatchers.IO) { nativeGetStateData() }
    suspend fun loadStateData(state: ByteArray): Boolean = withContext(Dispatchers.IO) {
        try { nativeLoadStateData(state) } catch (t: Throwable) {
            Log.e(TAG, "loadStateData error", t); false
        }
    }
    suspend fun getStateSize(): Long = withContext(Dispatchers.IO) { nativeGetStateSize() }
}