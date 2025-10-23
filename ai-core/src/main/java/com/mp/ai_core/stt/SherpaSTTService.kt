package com.mp.ai_core.stt

import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.util.Log
import kotlinx.coroutines.*

/**
 * AIDL-based service for Sherpa STT.
 * This service runs in a separate process for better isolation and resource management.
 * 
 * Add to AndroidManifest.xml:
 * <service
 *     android:name=".stt.SherpaSTTService"
 *     android:enabled="true"
 *     android:exported="true"
 *     android:process=":stt">
 *     <intent-filter>
 *         <action android:name="com.mp.ai_core.stt.STT_SERVICE" />
 *     </intent-filter>
 * </service>
 */
class SherpaSTTService : Service() {
    
    companion object {
        private const val TAG = "SherpaSTTService"
        const val ACTION_STT_SERVICE = "com.mp.ai_core.stt.STT_SERVICE"
    }

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    
    private val binder = object : ISherpaSTTService.Stub() {
        
        override fun initialize(modelDir: String, modelType: Int, numThreads: Int): Boolean {
            Log.d(TAG, "AIDL: initialize(modelType=$modelType, numThreads=$numThreads)")
            
            return try {
                val result = runBlocking {
                    SherpaSTTManager.initialize(
                        modelDir = modelDir,
                        modelType = modelType,
                        numThreads = numThreads
                    )
                }
                result.isSuccess
            } catch (e: Exception) {
                Log.e(TAG, "AIDL: initialize failed", e)
                false
            }
        }

        override fun isReady(): Boolean {
            return SherpaSTTManager.isReady
        }

        override fun getStatus(): Int {
            return when (SherpaSTTManager.status) {
                STTStatus.UNINITIALIZED -> 0
                STTStatus.LOADING -> 1
                STTStatus.READY -> 2
                STTStatus.ERROR -> 3
            }
        }

        override fun transcribeFile(filePath: String, sampleRate: Int): String {
            Log.d(TAG, "AIDL: transcribeFile(path=$filePath, rate=$sampleRate)")
            
            return try {
                val result = runBlocking {
                    SherpaSTTManager.transcribeFile(filePath, sampleRate)
                }
                
                result.getOrElse { e ->
                    "ERROR: ${e.message}"
                }
            } catch (e: Exception) {
                Log.e(TAG, "AIDL: transcribeFile failed", e)
                "ERROR: ${e.message}"
            }
        }

        override fun transcribeSamples(samples: FloatArray, sampleRate: Int): String {
            Log.d(TAG, "AIDL: transcribeSamples(samples=${samples.size}, rate=$sampleRate)")
            
            return try {
                val result = runBlocking {
                    SherpaSTTManager.transcribeSamples(samples, sampleRate)
                }
                
                result.getOrElse { e ->
                    "ERROR: ${e.message}"
                }
            } catch (e: Exception) {
                Log.e(TAG, "AIDL: transcribeSamples failed", e)
                "ERROR: ${e.message}"
            }
        }

        override fun getActiveStreamCount(): Int {
            return SherpaSTTManager.activeStreamCount
        }

        override fun getCurrentModelType(): Int {
            return SherpaSTTManager.getCurrentModelType()
        }

        override fun release() {
            Log.d(TAG, "AIDL: release()")
            SherpaSTTManager.release()
        }

        override fun isModelAvailable(modelType: Int): Boolean {
            return try {
               // com.k2fsa.sherpa.onnx.getOfflineModelConfig(modelType) != null
                true
            } catch (e: Exception) {
                false
            }
        }
    }

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "Service created")
    }

    override fun onBind(intent: Intent?): IBinder {
        Log.i(TAG, "Service bound")
        return binder
    }

    override fun onUnbind(intent: Intent?): Boolean {
        Log.i(TAG, "Service unbound")
        return super.onUnbind(intent)
    }

    override fun onDestroy() {
        super.onDestroy()
        Log.i(TAG, "Service destroyed, releasing resources")
        SherpaSTTManager.release()
        serviceScope.cancel()
    }
}