package com.mp.ai_core

import android.annotation.SuppressLint
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.media.MediaPlayer
import android.net.Uri
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DividerDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.mp.ai_core.tts.TtsEngine
import com.mp.ai_core.ui.theme.AiCoreTheme
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.time.TimeSource

class TTSActivity : ComponentActivity() {
    companion object {
        private const val TAG = "sherpa-onnx-tts-engine"
        private const val OUTPUT_FILENAME = "generated.wav"
    }

    private var mediaPlayer: MediaPlayer? = null
    private lateinit var track: AudioTrack
    private var stopped: Boolean = false
    private var samplesChannel = Channel<FloatArray>()

    @OptIn(ExperimentalMaterial3Api::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        initializeTts()
        initAudioTrack()

        setContent {
            AiCoreTheme {
                TtsScreen(
                    onGenerate = ::onGenerate, onPlay = ::onClickPlay, onStop = ::onClickStop
                )
            }
        }
    }

    override fun onDestroy() {
        stopMediaPlayer()
        track.release()
        super.onDestroy()
    }

    private fun initializeTts() {
        Log.i(TAG, "Start to initialize TTS")
        TtsEngine.createTts(this)
        Log.i(TAG, "Finish initializing TTS")
    }

    private fun initAudioTrack() {
        Log.i(TAG, "Start to initialize AudioTrack")
        val sampleRate = TtsEngine.tts!!.sampleRate()
        val bufLength = AudioTrack.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT
        )
        Log.i(TAG, "sampleRate: $sampleRate, buffLength: $bufLength")

        val attr = AudioAttributes.Builder().setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
            .setUsage(AudioAttributes.USAGE_MEDIA).build()

        val format = AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).setSampleRate(sampleRate).build()

        track = AudioTrack(
            attr, format, bufLength, AudioTrack.MODE_STREAM, AudioManager.AUDIO_SESSION_ID_GENERATE
        )
        track.play()
        Log.i(TAG, "Finish initializing AudioTrack")
    }

    @SuppressLint("DefaultLocale")
    private suspend fun onGenerate(text: String, speakerId: Int): String {
        return withContext(Dispatchers.IO) {
            stopped = false
            track.pause()
            track.flush()
            track.play()

            // Launch audio playback coroutine
            launch {
                for (samples in samplesChannel) {
                    track.write(samples, 0, samples.size, AudioTrack.WRITE_BLOCKING)
                    if (stopped) break
                }
            }

            val timeSource = TimeSource.Monotonic
            val startTime = timeSource.markNow()

            val audio = TtsEngine.tts!!.generateWithCallback(
                text = text,
                sid = speakerId,
                callback = ::callback,
            )

            val elapsed = startTime.elapsedNow().inWholeMilliseconds.toFloat() / 1000
            val audioDuration = audio.samples.size / TtsEngine.tts!!.sampleRate().toFloat()

            val filename = application.filesDir.absolutePath + "/$OUTPUT_FILENAME"
            audio.save(filename)

            String.format(
                "Threads: %d\nElapsed: %.3f s\nAudio: %.3f s\nRTF: %.3f",
                TtsEngine.tts!!.config.model.numThreads,
                elapsed,
                audioDuration,
                elapsed / audioDuration
            )
        }
    }

    private fun stopMediaPlayer() {
        mediaPlayer?.stop()
        mediaPlayer?.release()
        mediaPlayer = null
    }

    private fun onClickPlay() {
        val filename = application.filesDir.absolutePath + "/$OUTPUT_FILENAME"
        stopMediaPlayer()
        mediaPlayer = MediaPlayer.create(
            applicationContext, Uri.fromFile(File(filename))
        )
        mediaPlayer?.start()
    }

    private fun onClickStop() {
        stopped = true
        track.pause()
        track.flush()
        stopMediaPlayer()
    }

    private fun callback(samples: FloatArray): Int {
        return if (!stopped) {
            val samplesCopy = samples.copyOf()
            CoroutineScope(Dispatchers.IO).launch {
                samplesChannel.send(samplesCopy)
            }
            1
        } else {
            track.stop()
            Log.i(TAG, "Callback stopped")
            0
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TtsScreen(
    onGenerate: suspend (String, Int) -> String, onPlay: () -> Unit, onStop: () -> Unit
) {
    var inputText by remember { mutableStateOf("") }
    var speakerId by remember { mutableStateOf("0") }
    var isGenerating by remember { mutableStateOf(false) }
    var hasGenerated by remember { mutableStateOf(false) }
    var statsText by remember { mutableStateOf("") }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    val scope = rememberCoroutineScope()
    val scrollState = rememberScrollState()

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        "Text-to-Speech Engine", style = MaterialTheme.typography.titleLarge
                    )
                }, colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    titleContentColor = MaterialTheme.colorScheme.onPrimaryContainer
                )
            )
        }) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .verticalScroll(scrollState)
                .padding(16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Speaker ID Input
            OutlinedTextField(
                value = speakerId,
                onValueChange = { newValue ->
                    if (newValue.isEmpty() || newValue.all { it.isDigit() }) {
                        speakerId = newValue
                    }
                },
                label = { Text("Speaker ID") },
                placeholder = { Text("0") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.fillMaxWidth(),
                enabled = !isGenerating,
                singleLine = true
            )

            // Text Input
            OutlinedTextField(
                value = inputText,
                onValueChange = { inputText = it },
                label = { Text("Enter text to synthesize") },
                placeholder = { Text("Type something...") },
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 150.dp),
                enabled = !isGenerating,
                maxLines = 8,
                shape = RoundedCornerShape(12.dp)
            )

            // Error Message
            errorMessage?.let { error ->
                Card(
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.errorContainer
                    ), modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = error,
                        color = MaterialTheme.colorScheme.onErrorContainer,
                        modifier = Modifier.padding(12.dp),
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }

            // Action Buttons
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Button(
                    onClick = {
                        if (inputText.isBlank()) {
                            errorMessage = "Please enter some text to synthesize"
                            return@Button
                        }

                        errorMessage = null
                        isGenerating = true
                        hasGenerated = false

                        scope.launch {
                            try {
                                val stats = onGenerate(
                                    inputText, speakerId.toIntOrNull() ?: 0
                                )
                                statsText = stats
                                hasGenerated = true
                            } catch (e: Exception) {
                                errorMessage = "Generation failed: ${e.message}"
                            } finally {
                                isGenerating = false
                            }
                        }
                    },
                    enabled = !isGenerating,
                    modifier = Modifier.weight(1f),
                    contentPadding = PaddingValues(vertical = 12.dp)
                ) {
                    if (isGenerating) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(20.dp),
                            strokeWidth = 2.dp,
                            color = MaterialTheme.colorScheme.onPrimary
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Generating...")
                    } else {
                        Icon(
                            imageVector = Icons.Default.Refresh,
                            contentDescription = null,
                            modifier = Modifier.size(20.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Generate")
                    }
                }

                OutlinedButton(
                    onClick = onPlay,
                    enabled = hasGenerated && !isGenerating,
                    modifier = Modifier.weight(1f),
                    contentPadding = PaddingValues(vertical = 12.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.PlayArrow,
                        contentDescription = null,
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Play")
                }

                OutlinedButton(
                    onClick = {
                        onStop()
                        isGenerating = false
                    },
                    modifier = Modifier.weight(1f),
                    contentPadding = PaddingValues(vertical = 12.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Stop,
                        contentDescription = null,
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Stop")
                }
            }

            // Statistics Card
            if (statsText.isNotEmpty()) {
                Card(
                    modifier = Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.secondaryContainer
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        Text(
                            text = "Generation Statistics",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.onSecondaryContainer
                        )
                        HorizontalDivider(
                            modifier = Modifier.padding(vertical = 8.dp),
                            thickness = DividerDefaults.Thickness,
                            color = MaterialTheme.colorScheme.onSecondaryContainer.copy(alpha = 0.3f)
                        )
                        Text(
                            text = statsText,
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSecondaryContainer
                        )
                    }
                }
            }
        }
    }
}