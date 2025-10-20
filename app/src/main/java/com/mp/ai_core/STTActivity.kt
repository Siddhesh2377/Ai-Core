package com.mp.ai_core

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.SoundPool
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RawRes
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.k2fsa.sherpa.onnx.OfflineStream
import com.k2fsa.sherpa.onnx.WaveReader
import com.mp.ai_core.stt.SimulateStreamingAsr
import com.mp.ai_core.ui.theme.AiCoreTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

class STTActivity : ComponentActivity() {
    companion object {
        private const val TAG = "STTActivity"
        private const val SAMPLE_RATE = 16000
        private const val MODEL_TYPE = 2
    }

    private var recordingJob: Job? = null
    private var isRecording = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val requestPermissionLauncher = registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { granted ->
            if (!granted) {
                Toast.makeText(this, "Mic permission required", Toast.LENGTH_SHORT).show()
            }
        }
        requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)

        // Initialize recognizer
        try {
            SimulateStreamingAsr.initOfflineRecognizer(
                assetManager = assets,
                modelType = MODEL_TYPE
            )
            Log.i(TAG, "Recognizer initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize recognizer", e)
            Toast.makeText(this, "Failed to initialize recognizer", Toast.LENGTH_LONG).show()
        }

        setContent {
            var config by remember { mutableStateOf(NeuralThemes.MidnightMoss) }
            val neuralState = rememberNeuralNetworkState()
            var textResult by remember { mutableStateOf("") }
            var isLoading by remember { mutableStateOf(false) }
            var isRecordingState by remember { mutableStateOf(false) }
            val scope = rememberCoroutineScope()

            val interfaceSuccess = rememberSoundPlayer(this, R.raw.interface_success)
            val interfaceError = rememberSoundPlayer(this, R.raw.error_interface)

            AiCoreTheme {
                Scaffold { padding ->
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(padding)
                            .padding(horizontal = 24.dp, vertical = 32.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.SpaceBetween
                    ) {
                        // Title
                        Text(
                            text = "AI Voice Transcription",
                            fontSize = 28.sp,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.onBackground,
                            textAlign = TextAlign.Center
                        )

                        // Neural Animation Card
                        Card(
                            modifier = Modifier
                                .size(280.dp)
                                .animateContentSize(),
                            shape = CircleShape,
                            colors = CardDefaults.cardColors(
                                containerColor = Color.Transparent
                            ),
                            elevation = CardDefaults.cardElevation(0.dp)
                        ) {
                            FuturisticNeuralAnimation(
                                modifier = Modifier.fillMaxSize(),
                                config = config,
                                state = neuralState
                            )
                        }

                        // Recording Button
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            RecordingButton(
                                isRecording = isRecordingState,
                                isLoading = isLoading,
                                isEnabled = !isLoading && SimulateStreamingAsr.isInitialized,
                                config = config,
                                onClick = {
                                    if (!isRecording) {
                                        // Start recording
                                        interfaceSuccess()
                                        isRecording = true
                                        isRecordingState = true
                                        textResult = ""
                                        scope.launch {
                                            config = NeuralThemes.CrimsonBlood
                                            neuralState.spike(1f)
                                            neuralState.setAllSpeeds(2f)
                                        }
                                        recordingJob = scope.launch(Dispatchers.IO) {
                                            startRecording()
                                        }
                                    } else {
                                        // Stop recording
                                        interfaceError()
                                        isRecording = false
                                        isRecordingState = false
                                        config = NeuralThemes.SageGarden
                                        scope.launch {
                                            neuralState.setAllSpeeds(1f)
                                            recordingJob?.join()

                                            withContext(Dispatchers.Main) {
                                                isLoading = true
                                            }

                                            neuralState.spike(0.8f)
                                            val text = transcribeLastRecording()

                                            withContext(Dispatchers.Main) {
                                                textResult = text
                                                isLoading = false

                                                // Play appropriate sound based on result
                                                if (text.startsWith("Error") || text == "File not found" ||
                                                    text == "Invalid audio file" || text == "No audio data" ||
                                                    text == "Recognizer not ready"
                                                ) {
                                                    interfaceError()
                                                    config = NeuralThemes.CrimsonBlood
                                                } else {
                                                    interfaceSuccess()
                                                    config = NeuralThemes.ArcticTeal
                                                }
                                            }

                                            neuralState.spike(0.5f)
                                            delay(300)
                                            neuralState.resetSpike()
                                        }
                                    }
                                }
                            )

                            Spacer(modifier = Modifier.height(20.dp))

                            // Status text
                            AnimatedVisibility(
                                visible = isRecordingState || isLoading,
                                enter = fadeIn(),
                                exit = fadeOut()
                            ) {
                                Text(
                                    text = when {
                                        isRecordingState -> "● Recording..."
                                        isLoading -> "⚡ Processing..."
                                        else -> ""
                                    },
                                    fontSize = 16.sp,
                                    color = config.primaryColor,
                                    fontWeight = FontWeight.Medium
                                )
                            }
                        }

                        // Result Card
                        ResultCard(
                            text = textResult,
                            config = config
                        )
                    }
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        SimulateStreamingAsr.release()
    }

    @SuppressLint("MissingPermission")
    private suspend fun startRecording() {
        val bufferSize = AudioRecord.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )

        val recorder = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize
        )

        val pcmFile = File(getExternalFilesDir(null), "recorded_audio.pcm")

        try {
            FileOutputStream(pcmFile).use { fos ->
                val buffer = ByteArray(bufferSize)
                recorder.startRecording()
                Log.i(TAG, "Recording started")

                while (isRecording) {
                    val read = recorder.read(buffer, 0, buffer.size)
                    if (read > 0) {
                        fos.write(buffer, 0, read)
                    }
                }

                recorder.stop()
                recorder.release()
                Log.i(TAG, "Recording stopped, PCM size: ${pcmFile.length()} bytes")
            }

            val wavFile = File(getExternalFilesDir(null), "recorded_audio.wav")
            pcmToWav(pcmFile, wavFile, SAMPLE_RATE)

            if (wavFile.exists() && wavFile.length() > 44) {
                Log.i(TAG, "WAV file created: ${wavFile.path}, size: ${wavFile.length()} bytes")
            } else {
                Log.e(TAG, "WAV file creation failed or file too small")
            }

            pcmFile.delete()
        } catch (e: Exception) {
            Log.e(TAG, "Recording error: ${e.message}", e)
            recorder.stop()
            recorder.release()
        }
    }

    private fun pcmToWav(pcmFile: File, wavFile: File, sampleRate: Int) {
        val pcmData = pcmFile.readBytes()
        val channels = 1
        val bitsPerSample = 16
        val byteRate = sampleRate * channels * bitsPerSample / 8
        val blockAlign = channels * bitsPerSample / 8

        val header = ByteBuffer.allocate(44).apply {
            order(ByteOrder.LITTLE_ENDIAN)
            put("RIFF".toByteArray(Charsets.US_ASCII))
            putInt(36 + pcmData.size)
            put("WAVE".toByteArray(Charsets.US_ASCII))
            put("fmt ".toByteArray(Charsets.US_ASCII))
            putInt(16)
            putShort(1.toShort())
            putShort(channels.toShort())
            putInt(sampleRate)
            putInt(byteRate)
            putShort(blockAlign.toShort())
            putShort(bitsPerSample.toShort())
            put("data".toByteArray(Charsets.US_ASCII))
            putInt(pcmData.size)
        }

        FileOutputStream(wavFile).use {
            it.write(header.array())
            it.write(pcmData)
        }

        Log.i(TAG, "WAV created: rate=$sampleRate, channels=$channels, size=${pcmData.size} bytes")
    }

    private suspend fun transcribeLastRecording(): String = withContext(Dispatchers.IO) {
        val audioFile = File(getExternalFilesDir(null), "recorded_audio.wav")

        try {
            SimulateStreamingAsr.ensureInitialized()

            if (!audioFile.exists()) {
                Log.e(TAG, "Audio file not found: ${audioFile.path}")
                return@withContext "File not found"
            }

            if (audioFile.length() < 44) {
                Log.e(TAG, "Audio file too small: ${audioFile.length()} bytes")
                return@withContext "Invalid audio file"
            }

            Log.i(TAG, "Starting transcription: ${audioFile.path}, size: ${audioFile.length()} bytes")

            val waveData = WaveReader.readWave(audioFile.path)

            if (waveData.samples.isEmpty()) {
                Log.e(TAG, "No audio samples found")
                return@withContext "No audio data"
            }

            Log.i(TAG, "Loaded ${waveData.samples.size} samples at ${waveData.sampleRate} Hz")

            val recognizer = SimulateStreamingAsr.recognizer
            val stream: OfflineStream = recognizer.createStream()

            stream.acceptWaveform(waveData.samples, SAMPLE_RATE)
            recognizer.decode(stream)

            val result = recognizer.getResult(stream)
            stream.release()

            Log.i(TAG, "Transcription result: ${result.text}")

            return@withContext result.text.ifEmpty { "No speech detected" }
        } catch (e: IllegalStateException) {
            Log.e(TAG, "Recognizer not initialized: ${e.message}", e)
            return@withContext "Recognizer not ready"
        } catch (e: Exception) {
            Log.e(TAG, "Transcription error: ${e.message}", e)
            return@withContext "Error: ${e.message}"
        }
    }
}

@Composable
private fun RecordingButton(
    isRecording: Boolean,
    isLoading: Boolean,
    isEnabled: Boolean,
    config: NeuralAnimationConfig,
    onClick: () -> Unit
) {
    val infiniteTransition = rememberInfiniteTransition(label = "pulse")
    val scale by infiniteTransition.animateFloat(
        initialValue = 1f,
        targetValue = 1.15f,
        animationSpec = infiniteRepeatable(
            animation = tween(1200, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "scale"
    )

    Box(
        contentAlignment = Alignment.Center,
        modifier = Modifier.size(120.dp)
    ) {
        // Outer ring when recording
        if (isRecording) {
            Surface(
                modifier = Modifier
                    .size(110.dp)
                    .scale(scale),
                shape = CircleShape,
                color = Color.Transparent,
                border = BorderStroke(3.dp, config.primaryColor.copy(alpha = 0.5f))
            ) {}
        }

        // Main button with icon
        OutlinedButton(
            onClick = onClick,
            enabled = isEnabled,
            modifier = Modifier.size(90.dp),
            shape = CircleShape,
            colors = ButtonDefaults.outlinedButtonColors(
                containerColor = when {
                    isRecording -> config.primaryColor.copy(alpha = 0.2f)
                    isLoading -> config.accentColor.copy(alpha = 0.2f)
                    else -> config.secondaryColor.copy(alpha = 0.15f)
                },
                contentColor = when {
                    isRecording -> config.primaryColor
                    isLoading -> config.accentColor
                    else -> config.secondaryColor
                }
            ),
            border = BorderStroke(
                width = 2.dp,
                color = when {
                    isRecording -> config.primaryColor
                    isLoading -> config.accentColor
                    else -> config.secondaryColor
                }
            ),
            contentPadding = PaddingValues(0.dp)
        ) {
            when {
                isRecording -> Icon(
                    imageVector = Icons.Default.Stop,
                    contentDescription = "Stop Recording",
                    modifier = Modifier.size(40.dp)
                )
                isLoading -> Icon(
                    imageVector = Icons.Default.Mic,
                    contentDescription = "Processing",
                    modifier = Modifier.size(36.dp)
                )
                else -> Icon(
                    imageVector = Icons.Default.Mic,
                    contentDescription = "Start Recording",
                    modifier = Modifier.size(36.dp)
                )
            }
        }
    }
}

@Composable
private fun ResultCard(
    text: String,
    config: NeuralAnimationConfig
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 120.dp, max = 220.dp),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(
            containerColor = Color(0xFF0A0E27).copy(alpha = 0.85f)
        ),
        elevation = CardDefaults.cardElevation(0.dp),
        border = BorderStroke(1.dp, config.primaryColor.copy(alpha = 0.2f))
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp),
            contentAlignment = if (text.isEmpty()) Alignment.Center else Alignment.TopStart
        ) {
            if (text.isEmpty()) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    Icon(
                        imageVector = Icons.Default.Mic,
                        contentDescription = null,
                        tint = config.primaryColor.copy(alpha = 0.4f),
                        modifier = Modifier.size(32.dp)
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = "Tap the button to start recording",
                        fontSize = 15.sp,
                        color = Color.White.copy(alpha = 0.6f),
                        textAlign = TextAlign.Center,
                        fontWeight = FontWeight.Light
                    )
                }
            } else {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .verticalScroll(rememberScrollState())
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            imageVector = Icons.Default.Mic,
                            contentDescription = null,
                            tint = config.primaryColor,
                            modifier = Modifier.size(16.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = "Transcription",
                            fontSize = 13.sp,
                            color = config.primaryColor,
                            fontWeight = FontWeight.Bold
                        )
                    }
                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = text,
                        fontSize = 15.sp,
                        color = Color.White,
                        lineHeight = 22.sp,
                        fontWeight = FontWeight.Normal
                    )
                }
            }
        }
    }
}

@Composable
fun rememberSoundPlayer(context: Context, @RawRes soundResId: Int): () -> Unit {
    val soundPool = remember {
        SoundPool.Builder().setMaxStreams(5).setAudioAttributes(
            AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_ASSISTANCE_SONIFICATION)
                .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION).build()
        ).build()
    }

    var soundId by remember { mutableIntStateOf(0) }

    LaunchedEffect(Unit) {
        soundId = soundPool.load(context, soundResId, 1)
    }

    DisposableEffect(Unit) {
        onDispose {
            soundPool.release()
        }
    }

    return {
        if (soundId != 0) {
            soundPool.play(soundId, 1f, 1f, 1, 0, 1f)
        }
    }
}