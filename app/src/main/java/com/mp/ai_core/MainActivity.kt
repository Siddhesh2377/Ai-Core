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
import androidx.compose.foundation.border
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
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Divider
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
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
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
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
        startForegroundService(svcIntent)
        bindService(svcIntent, svcConnection, BIND_AUTO_CREATE)

        setContent { AiCoreTheme { AppUi() } }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopTts()
    }

    /* ---------- 8️⃣  UI ---------------------------------- */
    @Composable
    private fun AppUi() {
        var query by remember { mutableStateOf("") }
        var ttsText by remember { mutableStateOf("Hi bro..!") }
        val answer by answerFlow.collectAsState()

        Scaffold(
            containerColor = MaterialTheme.colorScheme.background
        ) { padding ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding)
                    .padding(horizontal = 20.dp, vertical = 16.dp)
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(20.dp)
            ) {
                // Header
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    Text(
                        text = "ToolNeuron AI",
                        style = MaterialTheme.typography.headlineMedium,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary
                    )
                    Text(
                        text = "RAG Demo Assistant",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))

                // LLM Section
                ElevatedCard(
                    modifier = Modifier.fillMaxWidth(),
                    elevation = CardDefaults.elevatedCardElevation(defaultElevation = 4.dp)
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(20.dp),
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        Text(
                            text = "💬 Ask Question",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold
                        )

                        // Query Input
                        Surface(
                            modifier = Modifier
                                .fillMaxWidth()
                                .heightIn(min = 80.dp),
                            shape = RoundedCornerShape(12.dp),
                            color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                            border = androidx.compose.foundation.BorderStroke(
                                1.dp,
                                MaterialTheme.colorScheme.outline.copy(alpha = 0.3f)
                            )
                        ) {
                            BasicTextField(
                                value = query,
                                onValueChange = { query = it },
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(16.dp),
                                textStyle = TextStyle(
                                    fontSize = 16.sp,
                                    color = MaterialTheme.colorScheme.onSurface
                                ),
                                decorationBox = { innerTextField ->
                                    if (query.isEmpty()) {
                                        Text(
                                            "Type your question here...",
                                            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
                                            fontSize = 16.sp
                                        )
                                    }
                                    innerTextField()
                                }
                            )
                        }

                        // Action Buttons
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            Button(
                                onClick = {
                                    if (query.isNotBlank()) {
                                        lifecycleScope.launch { shutdownPreviousJob() }
                                        lifecycleScope.launch { runGeneration(query) }
                                    }
                                },
                                modifier = Modifier.weight(1f),
                                enabled = query.isNotBlank() && !generating.value,
                                shape = RoundedCornerShape(10.dp)
                            ) {
                                Text("Ask", fontSize = 16.sp)
                            }

                            OutlinedButton(
                                onClick = { lifecycleScope.launch { shutdownPreviousJob() } },
                                modifier = Modifier.weight(1f),
                                enabled = generating.value,
                                shape = RoundedCornerShape(10.dp)
                            ) {
                                Text("Stop", fontSize = 16.sp)
                            }
                        }

                        // Status Indicator
                        if (generating.value) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(16.dp),
                                    strokeWidth = 2.dp
                                )
                                Text(
                                    "Generating... ${tokenCount.value} tokens",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.primary
                                )
                            }
                        }
                    }
                }

//                // Answer Section
//                ElevatedCard(
//                    modifier = Modifier.fillMaxWidth(),
//                    elevation = CardDefaults.elevatedCardElevation(defaultElevation = 4.dp)
//                ) {
//                    Column(
//                        modifier = Modifier
//                            .fillMaxWidth()
//                            .padding(20.dp),
//                        verticalArrangement = Arrangement.spacedBy(12.dp)
//                    ) {
//                        Text(
//                            text = "📝 Answer",
//                            style = MaterialTheme.typography.titleMedium,
//                            fontWeight = FontWeight.SemiBold
//                        )
//
//                        Surface(
//                            modifier = Modifier
//                                .fillMaxWidth()
//                                .heightIn(min = 150.dp),
//                            shape = RoundedCornerShape(12.dp),
//                            color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f)
//                        ) {
//                            Box(
//                                modifier = Modifier
//                                    .fillMaxSize()
//                                    .padding(16.dp)
//                            ) {
//                                when {
//                                    generating.value && !reachedFirstToken.value -> {
//                                        RobotDecodePlaceholder(
//                                            modifier = Modifier.align(Alignment.Center),
//                                            active = true,
//                                            base = "Decoding",
//                                            tokenCount = tokenCount.value
//                                        )
//                                    }
//                                    answer.isNotEmpty() -> {
//                                        Text(
//                                            answer,
//                                            fontSize = 15.sp,
//                                            lineHeight = 22.sp,
//                                            color = MaterialTheme.colorScheme.onSurface
//                                        )
//                                    }
//                                    else -> {
//                                        Text(
//                                            "Your answer will appear here...",
//                                            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f),
//                                            fontSize = 15.sp,
//                                            modifier = Modifier.align(Alignment.Center)
//                                        )
//                                    }
//                                }
//                            }
//                        }
//                    }
//                }

                // TTS Section
                ElevatedCard(
                    modifier = Modifier.fillMaxWidth(),
                    elevation = CardDefaults.elevatedCardElevation(defaultElevation = 4.dp)
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(20.dp),
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "🔊 Text-to-Speech",
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.SemiBold
                            )

                            if (!ttsConfigLoaded) {
                                OutlinedButton(
                                    onClick = { loadTtsModelOnce() },
                                    shape = RoundedCornerShape(8.dp)
                                ) {
                                    Text("Load Model", fontSize = 14.sp)
                                }
                            } else {
                                Surface(
                                    shape = RoundedCornerShape(8.dp),
                                    color = MaterialTheme.colorScheme.primaryContainer
                                ) {
                                    Text(
                                        "Model Loaded",
                                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                                        fontSize = 13.sp,
                                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                                        fontWeight = FontWeight.Medium
                                    )
                                }
                            }
                        }

                        // TTS Input
                        Surface(
                            modifier = Modifier
                                .fillMaxWidth()
                                .heightIn(min = 80.dp),
                            shape = RoundedCornerShape(12.dp),
                            color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                            border = androidx.compose.foundation.BorderStroke(
                                1.dp,
                                MaterialTheme.colorScheme.outline.copy(alpha = 0.3f)
                            )
                        ) {
                            BasicTextField(
                                value = ttsText,
                                onValueChange = { ttsText = it },
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(16.dp),
                                textStyle = TextStyle(
                                    fontSize = 16.sp,
                                    color = MaterialTheme.colorScheme.onSurface
                                ),
                                decorationBox = { innerTextField ->
                                    if (ttsText.isEmpty()) {
                                        Text(
                                            "Enter text to synthesize...",
                                            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
                                            fontSize = 16.sp
                                        )
                                    }
                                    innerTextField()
                                }
                            )
                        }

                        // TTS Controls
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            Button(
                                onClick = { lifecycleScope.launch { runTts(ttsText) } },
                                enabled = ttsText.isNotBlank() && ttsConfigLoaded && !isPlaying,
                                modifier = Modifier.weight(1f),
                                shape = RoundedCornerShape(10.dp)
                            ) {
                                Text("Synthesize", fontSize = 16.sp)
                            }

                            OutlinedButton(
                                onClick = { stopTts() },
                                enabled = isPlaying,
                                modifier = Modifier.weight(1f),
                                shape = RoundedCornerShape(10.dp),
                                colors = ButtonDefaults.outlinedButtonColors(
                                    contentColor = MaterialTheme.colorScheme.error
                                )
                            ) {
                                Text("Stop", fontSize = 16.sp)
                            }
                        }

                        if (isPlaying) {
                            LinearProgressIndicator(
                                modifier = Modifier.fillMaxWidth()
                            )
                        }
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
        _reachedFirstToken.value = false
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
        _reachedFirstToken.value = false

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
                service?.generate(prompt, 512, "", cb) ?: tokenChannel.close()
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

        val json = """
        {
          "modelDir": "kokoro-en-v0_19",
          "modelName": "model.onnx",
          "voices": "voices.bin",
          "dataDir": "kokoro-en-v0_19/espeak-ng-data",
          "lang": "eng",
        }
        """.trimIndent()

        try {
            TtsEngine.loadFromJson(this, json)
            ttsConfigLoaded = true
        } catch (e: RemoteException) {
            Log.e(TAG, "TTS load RPC failed", e)
        }
    }

    private fun runTts(text: String) {
        if (service == null) {
            Log.w(TAG, "Service not bound – abort TTS")
            return
        }

        lifecycleScope.launch(Dispatchers.IO) {
            TtsEngine.generateAudio(text)
        }

        lifecycleScope.launch(Dispatchers.IO) {
            ttsChannel = TtsEngine.samplesChannel
            if (ttsChannel != null){
                for (samples in ttsChannel) {
                    audioTrack?.write(samples, 0, samples.size, AudioTrack.WRITE_BLOCKING)
                }
                audioTrack?.stop()
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
        val minBufSizeBytes = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_FLOAT
        )
        val minBufSizeFrames = minBufSizeBytes / 4

        val attr = AudioAttributes.Builder()
            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
            .setUsage(AudioAttributes.USAGE_MEDIA)
            .build()

        val format = AudioFormat.Builder()
            .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
            .setSampleRate(sampleRate)
            .build()

        audioTrack = AudioTrack(
            attr,
            format,
            minBufSizeFrames,
            AudioTrack.MODE_STREAM,
            AudioManager.AUDIO_SESSION_ID_GENERATE
        ).apply { play() }

        isPlaying = true
    }

    private fun stopTts() {
        ttsChannel.cancel()
        audioTrack?.stop()
        audioTrack?.release()
        audioTrack = null
        isPlaying = false
    }
}