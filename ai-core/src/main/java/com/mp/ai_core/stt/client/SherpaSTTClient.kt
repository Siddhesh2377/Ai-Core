package com.mp.ai_core.stt.client

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import android.util.Log
import com.mp.ai_core.stt.ISherpaSTTService
import com.mp.ai_core.stt.SherpaSTTService
import kotlinx.coroutines.*
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Client for accessing SherpaSTTService from your main app.
 * This handles the service binding and provides a clean API.
 * 
 * Usage:
 * ```
 * val sttClient = SherpaSTTClient(context)
 * sttClient.connect { connected ->
 *     if (connected) {
 *         sttClient.initialize(modelType = 2, numThreads = 2) { success ->
 *             if (success) {
 *                 sttClient.transcribeFile("/path/to/audio.wav") { result ->
 *                     println("Transcription: $result")
 *                 }
 *             }
 *         }
 *     }
 * }
 * 
 * // Don't forget to disconnect
 * sttClient.disconnect()
 * ```
 */
class SherpaSTTClient(private val context: Context) {
    
    companion object {
        private const val TAG = "SherpaSTTClient"
    }

    private var service: ISherpaSTTService? = null
    private val isConnected = AtomicBoolean(false)
    private val isConnecting = AtomicBoolean(false)
    
    private val clientScope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    
    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            Log.i(TAG, "Service connected")
            service = ISherpaSTTService.Stub.asInterface(binder)
            isConnected.set(true)
            isConnecting.set(false)
            
            connectionCallbacks.forEach { it(true) }
            connectionCallbacks.clear()
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            Log.w(TAG, "Service disconnected")
            service = null
            isConnected.set(false)
            isConnecting.set(false)
        }
    }
    
    private val connectionCallbacks = mutableListOf<(Boolean) -> Unit>()

    /**
     * Connect to the STT service
     * 
     * @param callback Called with true if connected, false if failed
     */
    fun connect(callback: (Boolean) -> Unit) {
        if (isConnected.get()) {
            callback(true)
            return
        }

        if (isConnecting.get()) {
            connectionCallbacks.add(callback)
            return
        }

        isConnecting.set(true)
        connectionCallbacks.add(callback)

        try {
            val intent = Intent(context, SherpaSTTService::class.java).apply {
                action = SherpaSTTService.ACTION_STT_SERVICE
            }
            
            val bound = context.bindService(
                intent,
                serviceConnection,
                Context.BIND_AUTO_CREATE
            )

            if (!bound) {
                Log.e(TAG, "Failed to bind service")
                isConnecting.set(false)
                connectionCallbacks.forEach { it(false) }
                connectionCallbacks.clear()
            }

            // Timeout after 10 seconds
            clientScope.launch {
                delay(10000)
                if (isConnecting.get()) {
                    Log.e(TAG, "Service connection timeout")
                    isConnecting.set(false)
                    connectionCallbacks.forEach { it(false) }
                    connectionCallbacks.clear()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to connect to service", e)
            isConnecting.set(false)
            connectionCallbacks.forEach { it(false) }
            connectionCallbacks.clear()
        }
    }

    /**
     * Suspend version of connect
     */
    @OptIn(ExperimentalCoroutinesApi::class)
    suspend fun connectAsync(): Boolean = suspendCancellableCoroutine { cont ->
        connect { success ->
            cont.resume(success) {}
        }
    }

    /**
     * Disconnect from the service
     */
    fun disconnect() {
        if (isConnected.get()) {
            try {
                context.unbindService(serviceConnection)
                Log.i(TAG, "Service disconnected")
            } catch (e: Exception) {
                Log.e(TAG, "Error disconnecting", e)
            }
        }
        service = null
        isConnected.set(false)
        isConnecting.set(false)
        clientScope.cancel()
    }

    /**
     * Initialize the STT engine
     */
    fun initialize(
        modelPath: String,
        modelType: Int = 2,
        numThreads: Int = 2,
        callback: (Boolean) -> Unit
    ) {
        clientScope.launch(Dispatchers.IO) {
            val result = try {
                service?.initialize(modelPath, modelType, numThreads) ?: false
            } catch (e: Exception) {
                Log.e(TAG, "Initialize failed", e)
                false
            }
            withContext(Dispatchers.Main) {
                callback(result)
            }
        }
    }

    /**
     * Suspend version of initialize
     */
    suspend fun initializeAsync(
        modelDir: String,
        modelType: Int = 2,
        numThreads: Int = 2
    ): Boolean = withContext(Dispatchers.IO) {
        try {
            service?.initialize(modelDir,modelType, numThreads) ?: false
        } catch (e: Exception) {
            Log.e(TAG, "Initialize failed", e)
            false
        }
    }

    /**
     * Check if service is ready
     */
    fun isReady(): Boolean {
        return try {
            service?.isReady() ?: false
        } catch (e: Exception) {
            Log.e(TAG, "isReady failed", e)
            false
        }
    }

    /**
     * Get current status
     */
    fun getStatus(): Int {
        return try {
            service?.status ?: 0
        } catch (e: Exception) {
            Log.e(TAG, "getStatus failed", e)
            0
        }
    }

    /**
     * Transcribe audio file
     */
    fun transcribeFile(
        filePath: String,
        sampleRate: Int = 16000,
        callback: (String) -> Unit
    ) {
        clientScope.launch(Dispatchers.IO) {
            val result = try {
                service?.transcribeFile(filePath, sampleRate) ?: "ERROR: Service not connected"
            } catch (e: Exception) {
                Log.e(TAG, "Transcribe failed", e)
                "ERROR: ${e.message}"
            }
            withContext(Dispatchers.Main) {
                callback(result)
            }
        }
    }

    /**
     * Suspend version of transcribeFile
     */
    suspend fun transcribeFileAsync(
        filePath: String,
        sampleRate: Int = 16000
    ): String = withContext(Dispatchers.IO) {
        try {
            service?.transcribeFile(filePath, sampleRate) ?: "ERROR: Service not connected"
        } catch (e: Exception) {
            Log.e(TAG, "Transcribe failed", e)
            "ERROR: ${e.message}"
        }
    }

    /**
     * Transcribe audio samples
     */
    fun transcribeSamples(
        samples: FloatArray,
        sampleRate: Int = 16000,
        callback: (String) -> Unit
    ) {
        clientScope.launch(Dispatchers.IO) {
            val result = try {
                service?.transcribeSamples(samples, sampleRate) ?: "ERROR: Service not connected"
            } catch (e: Exception) {
                Log.e(TAG, "Transcribe failed", e)
                "ERROR: ${e.message}"
            }
            withContext(Dispatchers.Main) {
                callback(result)
            }
        }
    }

    /**
     * Suspend version of transcribeSamples
     */
    suspend fun transcribeSamplesAsync(
        samples: FloatArray,
        sampleRate: Int = 16000
    ): String = withContext(Dispatchers.IO) {
        try {
            service?.transcribeSamples(samples, sampleRate) ?: "ERROR: Service not connected"
        } catch (e: Exception) {
            Log.e(TAG, "Transcribe failed", e)
            "ERROR: ${e.message}"
        }
    }

    /**
     * Get active stream count
     */
    fun getActiveStreamCount(): Int {
        return try {
            service?.activeStreamCount ?: 0
        } catch (e: Exception) {
            Log.e(TAG, "getActiveStreamCount failed", e)
            0
        }
    }

    /**
     * Get current model type
     */
    fun getCurrentModelType(): Int {
        return try {
            service?.currentModelType ?: -1
        } catch (e: Exception) {
            Log.e(TAG, "getCurrentModelType failed", e)
            -1
        }
    }

    /**
     * Release resources
     */
    fun release() {
        try {
            service?.release()
        } catch (e: Exception) {
            Log.e(TAG, "Release failed", e)
        }
    }

    /**
     * Check if model is available
     */
    fun isModelAvailable(modelType: Int): Boolean {
        return try {
            service?.isModelAvailable(modelType) ?: false
        } catch (e: Exception) {
            Log.e(TAG, "isModelAvailable failed", e)
            false
        }
    }

    /**
     * Check if currently connected
     */
    fun isConnectedToService(): Boolean = isConnected.get()
}