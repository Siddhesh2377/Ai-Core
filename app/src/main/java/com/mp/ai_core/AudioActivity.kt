package com.mp.ai_core

import android.content.ComponentName
import android.content.Intent
import android.content.ServiceConnection
import android.os.Bundle
import android.os.IBinder
import android.os.RemoteException
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Button
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.mp.ai_core.services.AudioService
import com.mp.ai_core.services.IAudioCallback
import com.mp.ai_core.services.IAudioService
import com.mp.ai_core.services.ISttCallback
import com.mp.ai_core.ui.theme.AiCoreTheme
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class AudioActivity : ComponentActivity() {

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            iAudioService = IAudioService.Stub.asInterface(service)
            try {
                ttsReady.value = iAudioService?.initializeTts(
                    "/storage/emulated/0/Download/audio/kokoro-en-v0_19",
                    "model.onnx",
                    "voices.bin",
                    "/storage/emulated/0/Download/audio/kokoro-en-v0_19/espeak-ng-data"
                ) ?: false
//                sttReady.value = iAudioService?.initializeStt(
//                    "/storage/emulated/0/Download/audio/sherpa-onnx-whisper-tiny", 2, 4
//                ) ?: false
            } catch (e: RemoteException) {
                Log.e(TAG, "RemoteException onServiceConnected", e)
            }
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            iAudioService = null
            ttsReady.value = false
            sttReady.value = false
        }
    }

    private var iAudioService: IAudioService? = null

    private val ttsReady = mutableStateOf(false)
    private val sttReady = mutableStateOf(false)

    private var ttsText = mutableStateOf("")
    private var sttText = mutableStateOf("")

    private val coroutineScope = CoroutineScope(Dispatchers.Main)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        val intent = Intent(this, AudioService::class.java)
        bindService(intent, serviceConnection, BIND_AUTO_CREATE)

        setContent {
            AiCoreTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    AudioControlLayout()
                }
            }
        }
    }

    @Composable
    private fun AudioControlLayout() {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp)
        ) {
            Text(
                text = "TTS & STT Demo",
                fontSize = 24.sp,
                modifier = Modifier.padding(bottom = 16.dp)
            )

            // TTS Controls
            TextField(
                value = ttsText.value,
                onValueChange = { ttsText.value = it },
                label = { Text("Enter text for TTS") },
                modifier = Modifier.fillMaxWidth()
            )

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 8.dp)
            ) {
                Button(
                    onClick = { initializeTts() },
                    enabled = !ttsReady.value,
                    modifier = Modifier.weight(1f)
                ) {
                    Text("Initialize TTS")
                }
                Spacer(modifier = Modifier.width(8.dp))
                Button(
                    onClick = { generateTts() },
                    enabled = ttsReady.value,
                    modifier = Modifier.weight(1f)
                ) {
                    Text("Generate TTS")
                }
            }

            // STT Controls
            Text(
                text = "STT Output:",
                fontSize = 18.sp,
                modifier = Modifier.padding(top = 16.dp, bottom = 8.dp)
            )
            TextField(
                value = sttText.value,
                onValueChange = { sttText.value = it },
                label = { Text("STT Result") },
                modifier = Modifier.fillMaxWidth(),
                readOnly = true
            )

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 8.dp)
            ) {
                Button(
                    onClick = { initializeStt() },
                    enabled = !sttReady.value,
                    modifier = Modifier.weight(1f)
                ) {
                    Text("Initialize STT")
                }
                Spacer(modifier = Modifier.width(8.dp))
                Button(
                    onClick = { generateStt() },
                    enabled = sttReady.value,
                    modifier = Modifier.weight(1f)
                ) {
                    Text("Generate STT")
                }
            }
        }
    }

    private fun initializeTts() {
        coroutineScope.launch {
            ttsText.value = ""
            iAudioService?.releaseTts()
            ttsReady.value = iAudioService?.initializeTts(
                "/storage/emulated/0/Download/audio/kokoro-en-v0_19",
                "model.onnx",
                "voices.bin",
                "/storage/emulated/0/Download/audio/kokoro-en-v0_19/espeak-ng-data"
            ) ?: false
        }
    }

    private fun generateTts() {
        coroutineScope.launch {
            iAudioService?.generateTts(
                ttsText.value, 0, object : IAudioCallback.Stub() {
                    override fun onAudioChunk(samples: FloatArray?) {
                        Log.d(TAG, "TTS Chunk: Size: ${samples?.size}")
                    }

                    override fun onComplete() {
                        Log.d(TAG, "TTS Complete")
                    }

                    override fun onError(error: String?) {
                        Log.e(TAG, "TTS Error: $error")
                    }
                })
        }
    }

    private fun initializeStt() {
        coroutineScope.launch {
            sttText.value = ""
            iAudioService?.releaseStt()
            sttReady.value = iAudioService?.initializeStt(
                "/storage/emulated/0/Download/audio/sherpa-onnx-whisper-tiny", 2, 4
            ) ?: false
        }
    }

    private fun generateStt() {
        coroutineScope.launch {
            // Assuming you have a recorded audio file path
            val filePath = "dummyFilePath.wav" // Change this to your actual recorded file path
            iAudioService?.transcribeFile(
                filePath, 16000, // Sample rate of the audio file
                object : ISttCallback.Stub() {
                    override fun onResult(text: String?) {
                        sttText.value = text ?: "No result"
                    }

                    override fun onError(error: String?) {
                        sttText.value = "Error: ${error ?: "Unknown error"}"
                    }
                })
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            unbindService(serviceConnection)
        } catch (e: Exception) {
            Log.e(TAG, "Exception unbinding service", e)
        }
    }

    companion object {
        private const val TAG = "AudioActivity"
    }
}