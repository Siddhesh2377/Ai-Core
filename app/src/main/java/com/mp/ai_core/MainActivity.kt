// MainActivity.kt
// ---------- 1️⃣  Imports ----------
package com.mp.ai_core

import android.content.ComponentName
import android.content.Intent
import android.content.ServiceConnection
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.os.Bundle
import android.os.IBinder
import android.os.RemoteException
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.State
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.lifecycleScope
import com.mp.ai_core.services.GenerationService
import com.mp.ai_core.services.IGenerationCallback
import com.mp.ai_core.services.IGenerationService
import com.mp.ai_core.tts.TtsEngine
import com.mp.ai_core.ui.theme.AiCoreTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.consumeAsFlow
import kotlinx.coroutines.flow.scan
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

private const val TAG = "MainActivity"

class MainActivity : ComponentActivity() {

    /* ---------- 1️⃣  Service handling ---------- */
    private var service: IGenerationService? = null
    private var modelLoaded = false

    /* ---------- 2️⃣  TTS -------------------------------------- */
    private lateinit var ttsChannel: Channel<FloatArray>
    private var audioTrack: AudioTrack? = null
    private var isPlaying = false
    private var ttsConfigLoaded = false

    /* ---------- 3️⃣  Service connection ---------- */
    private val svcConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            service = IGenerationService.Stub.asInterface(binder)
            Log.i(TAG, "Service connected")
            lifecycleScope.launch { loadModelOnce(service!!) }
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            service = null
            Log.w(TAG, "Service disconnected")
            modelLoaded = false
        }
    }

    /* ---------- 4️⃣  LLM channel -------------------------------- */
    private var tokenChannel: Channel<String> = Channel(Channel.CONFLATED)

    private var answerFlow: StateFlow<String> =
        tokenChannel.consumeAsFlow().scan("") { acc, token -> acc + token }.stateIn(
                scope = lifecycleScope,
                started = SharingStarted.WhileSubscribed(5_000),
                initialValue = ""
            )

    /* ---------- 5️⃣  LLM job --------------------------------- */
    private var genJob: Job? = null

    /* ---------- 6️⃣  UI state ------------------------------- */
    private val _generating = mutableStateOf(false)
    private val generating: State<Boolean> get() = _generating

    private val _tokenCount = mutableIntStateOf(0)
    private val tokenCount: State<Int> get() = _tokenCount

    private val _reachedFirstToken = mutableStateOf(false)
    private val reachedFirstToken: State<Boolean> get() = _reachedFirstToken

    /* ---------- 7️⃣  life‑cycle --------------------------------- */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        val svcIntent = Intent(this, GenerationService::class.java)
        startForegroundService(svcIntent)  // foreground service
        bindService(svcIntent, svcConnection, BIND_AUTO_CREATE)

        setContent { AiCoreTheme { AppUi() } }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopTts()          // release AudioTrack if playing
        // Service remains up; unbind only if you want
    }

    /* ---------- 8️⃣  UI ---------------------------------- */
    @Composable
    private fun AppUi() {
        var query by remember { mutableStateOf("") }
        var ttsText by remember { mutableStateOf("Hi bro..!") }
        val answer by answerFlow.collectAsState()

        Scaffold {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(it)
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Text(
                    text = "ToolNeuron RAG Demo", style = MaterialTheme.typography.headlineSmall
                )

                BasicTextField(
                    value = query,
                    onValueChange = { query = it },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(56.dp)
                        .padding(horizontal = 12.dp, vertical = 8.dp)
                        .background(Color.DarkGray)
                )

                Row(
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Button(
                        onClick = {
                            if (query.isNotBlank()) {
                                lifecycleScope.launch { shutdownPreviousJob() }
                                lifecycleScope.launch { runGeneration(query) }
                            }
                        }) { Text("Ask") }

                    Button(
                        onClick = { lifecycleScope.launch { shutdownPreviousJob() } }) { Text("Stop") }
                }

                /* ------ TTS UI ------ */
                Spacer(modifier = Modifier.height(24.dp))

                Text(
                    text = "Text‑to‑Speech", fontWeight = FontWeight.SemiBold, fontSize = 18.sp
                )

                BasicTextField(
                    value = ttsText,
                    onValueChange = { ttsText = it },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(48.dp)
                        .padding(horizontal = 12.dp, vertical = 8.dp)
                        .background(Color.DarkGray)
                )

                Row(
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.padding(top = 8.dp)
                ) {
                    Button(
                        onClick = { loadTtsModelOnce() }) { Text("Load TTS") }

                    Button(
                        onClick = { lifecycleScope.launch { runTts(ttsText) } },
                        enabled = ttsText.isNotBlank()
                    ) { Text("Synthesize") }

                    Button(
                        onClick = { stopTts() }, enabled = isPlaying
                    ) { Text("Stop") }
                }

                /* ------ LLM answer ------ */
                Text(
                    "Answer:", fontWeight = FontWeight.SemiBold, fontSize = 18.sp
                )

                Box(
                    Modifier
                        .fillMaxWidth()
                        .heightIn(min = 120.dp)
                        .padding(12.dp)
                ) {
                    if (generating.value && !reachedFirstToken.value) {
                        RobotDecodePlaceholder(
                            modifier = Modifier
                                .fillMaxSize()
                                .padding(12.dp),
                            active = true,
                            base = "Decoding …",
                            tokenCount = tokenCount.value
                        )
                    }

                    if (answer.isNotEmpty()) {
                        Text(
                            answer, fontSize = 16.sp, color = MaterialTheme.colorScheme.primary
                        )
                    } else {
                        Spacer(modifier = Modifier.height(16.dp))
                    }
                }
            }
        }
    }

    /* ---------- 9️⃣  LLM helpers -------------------------------- */
    private suspend fun shutdownPreviousJob() {
        genJob?.cancelAndJoin()
        _generating.value = false
        _tokenCount.intValue = 0
    }

    private suspend fun runGeneration(prompt: String) {
        while (service == null) delay(50)

        tokenChannel = Channel(Channel.CONFLATED)
        answerFlow = tokenChannel.consumeAsFlow().scan("") { acc, token -> acc + token }.stateIn(
                scope = lifecycleScope,
                started = SharingStarted.WhileSubscribed(5_000),
                initialValue = ""
            )

        _generating.value = true
        _tokenCount.value = 0

        val cb = object : IGenerationCallback.Stub() {
            override fun onToken(token: String?) {
                val data = token ?: ""
                tokenChannel.trySend(data)

                if (!_reachedFirstToken.value) _reachedFirstToken.value = true
                _tokenCount.value++
            }

            override fun onToolCall(name: String?, argsJson: String?) {
                tokenChannel.trySend("[TOOL $name] $argsJson")
            }

            override fun onDone() {
                tokenChannel.close()
                _generating.value = false
            }

            override fun onError(err: String) {
                tokenChannel.trySend("ERROR: $err")
                tokenChannel.close()
                _generating.value = false
            }
        }

        genJob = lifecycleScope.launch {
            try {
                service?.generate(prompt, 512, cb) ?: tokenChannel.close()
            } catch (e: RemoteException) {
                tokenChannel.trySend("RPC failure: ${e.message}")
                tokenChannel.close()
                _generating.value = false
            }
        }
    }

    /* ---------- 🔧  Load LLM model once ---------- */
    private fun loadModelOnce(service: IGenerationService) {
        if (modelLoaded) return

        val modelPath = "/storage/emulated/0/Download/Models/lucy_128k-Q3_K_S.gguf"

        val ok = try {
            service.loadModelK(
                path = modelPath,
                threads = Runtime.getRuntime().availableProcessors() - 1,
                gpuLayers = 10,
                useMMap = true,
                ctxSize = 4096,
                temp = 0.7f,
                topK = 40,
                topP = 0.9f,
                minP = 0f
            )
        } catch (e: RemoteException) {
            Log.e(TAG, "loadModel RPC failed", e)
            false
        }

        Log.i(TAG, "Model load result: $ok")
        modelLoaded = ok
    }

    private fun IGenerationService.loadModelK(
        path: String,
        threads: Int,
        gpuLayers: Int,
        useMMap: Boolean,
        ctxSize: Int,
        temp: Float,
        topK: Int,
        topP: Float,
        minP: Float
    ): Boolean = loadModel(
        path, threads, gpuLayers, useMMap, ctxSize, temp, topK, topP, minP
    )

    /* ---------- 🔧  TTS helpers -------------------------------- */
    private fun loadTtsModelOnce() {
        if (ttsConfigLoaded) return

        // Hard‑coded configuration – replace with your model folder / names
        val json = """
        {
          "modelDir": "kitten-nano-en-v0_1-fp16",
          "modelName": "model.fp16.onnx",
          "voices": "voices.bin",
          "dataDir": "kitten-nano-en-v0_1-fp16/espeak-ng-data",
          "lang": "eng",
          "isKitten": true
        }
        """.trimIndent()

        try {
            TtsEngine.loadFromJson(this, json)
        } catch (e: RemoteException) {
            Log.e(TAG, "TTS load RPC failed", e)
            false
        }
    }

    private fun runTts(text: String) {
        if (service == null) {
            Log.w(TAG, "Service not bound – abort TTS")
            return
        }

        /* 1) start the service request */
        lifecycleScope.launch(Dispatchers.IO) {
            TtsEngine.generateAudio(text)
        }

        /* 2) render the stream from the channel */
        lifecycleScope.launch(Dispatchers.IO) {
            ttsChannel = TtsEngine.samplesChannel
            if (ttsChannel != null){
                for (samples in ttsChannel) {
                    audioTrack?.write(samples, 0, samples.size, AudioTrack.WRITE_BLOCKING)
                }
                audioTrack?.stop()     // after the stream ends
                audioTrack?.release()
                audioTrack = null
                isPlaying = false
            }
        }

        initialiseAudioTrackIfNeeded()
        isPlaying = true
    }

    private fun initialiseAudioTrackIfNeeded() {
        if (audioTrack != null) return
        val sampleRate = TtsEngine.tts!!.sampleRate()
        val bufLength = AudioTrack.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT
        )
        Log.i(TAG, "sampleRate: $sampleRate, buffLength: $bufLength")

        val attr = AudioAttributes.Builder().setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
            .setUsage(AudioAttributes.USAGE_MEDIA).build()

        val format = AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).setSampleRate(sampleRate).build()

        audioTrack = AudioTrack(
            attr, format, bufLength, AudioTrack.MODE_STREAM, AudioManager.AUDIO_SESSION_ID_GENERATE
        ).apply { play() }

        isPlaying = true
    }

    private fun stopTts() {
        ttsChannel.cancel()
        audioTrack?.stop()
        audioTrack?.flush()
        audioTrack?.release()
        audioTrack = null
        isPlaying = false
    }
}