package com.mp.ai_core

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.media.MediaPlayer
import android.os.Bundle
import android.os.RemoteException
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.scale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.mp.ai_core.tts.TtsEngine
import com.mp.ai_core.ui.theme.AiCoreTheme
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class DialogLine(
    val voiceId: Int, val text: String, val speakerName: String
)

class TTSActivity : ComponentActivity() {
    companion object {
        private const val TAG = "sherpa-onnx-tts"
        private const val OUTPUT_FILENAME = "generated.wav"
    }

    private var mediaPlayer: MediaPlayer? = null
    private lateinit var audioTrack: AudioTrack
    private var isStopped = false
    private val samplesChannel = Channel<FloatArray>(Channel.UNLIMITED)
    private var playbackJob: Job? = null

    @OptIn(ExperimentalMaterial3Api::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        initializeTts()
        initAudioTrack()

        setContent {
            AiCoreTheme {
                TtsConversationScreen(
                    onPlayConversation = ::playConversation, onStop = ::stopPlayback
                )
            }
        }
    }

    override fun onDestroy() {
        stopMediaPlayer()
        stopPlayback()
        audioTrack.release()
        samplesChannel.close()
        super.onDestroy()
    }

    private fun initializeTts() {
        Log.i(TAG, "Initializing TTS")
        val modelDir = "kokoro-int8-multi-lang-v1_1"

        val json2 = """
        {
          "modelDir": "$modelDir",
          "modelName": "model.int8.onnx",
          "voices": "voices.bin",
          "dataDir": "$modelDir/espeak-ng-data",
          "lang": "eng",
          "lexicon": "lexicon-gb-en.txt",
          "ruleFsts": "$modelDir/phone-zh.fst,$modelDir/date-zh.fst,$modelDir/number-zh.fst"
        }
        """.trimIndent()


        val json = """
        {
         "modelDir" = "kokoro-en-v0_19",
         "modelName" = "model.onnx",
         "voices" = "voices.bin",
         "dataDir" = "kokoro-en-v0_19/espeak-ng-data",
         "lang" = "eng"
        }
        """.trimIndent()

        try {
            TtsEngine.loadFromJson(this, json)
            Log.i(TAG, "TTS initialized successfully")
        } catch (e: RemoteException) {
            Log.e(TAG, "TTS initialization failed", e)
        }
    }

    private fun initAudioTrack() {
        val sampleRate = TtsEngine.tts!!.sampleRate()
        val bufLength = AudioTrack.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT
        )

        val attr = AudioAttributes.Builder().setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
            .setUsage(AudioAttributes.USAGE_MEDIA).build()

        val format = AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).setSampleRate(sampleRate).build()

        audioTrack = AudioTrack(
            attr,
            format,
            bufLength,
            AudioTrack.MODE_STREAM,
            android.media.AudioManager.AUDIO_SESSION_ID_GENERATE
        )
        audioTrack.play()
    }

    private suspend fun playConversation(
        conversation: List<DialogLine>, onLineChange: (Int) -> Unit
    ) = withContext(Dispatchers.IO) {
        isStopped = false
        audioTrack.pause()
        audioTrack.flush()
        audioTrack.play()

        // Start audio playback coroutine
        playbackJob = launch {
            for (samples in samplesChannel) {
                if (isStopped) break
                audioTrack.write(samples, 0, samples.size, AudioTrack.WRITE_BLOCKING)
            }
        }

        // Generate and play each line
        conversation.forEachIndexed { index, line ->
            if (isStopped) return@withContext

            withContext(Dispatchers.Main) {
                onLineChange(index)
            }

            TtsEngine.tts?.apply {
                currentSid = line.voiceId
                generateWithCallback(
                    text = line.text, callback = ::callback)
            }

            // Small delay between lines
            delay(300)
        }

        withContext(Dispatchers.Main) {
            onLineChange(-1) // Reset
        }


    }

    fun callback(samples: FloatArray, progress: Float): Int {
        return if (!isStopped) {
            val samplesCopy = samples.copyOf()
            CoroutineScope(Dispatchers.IO).launch {
                samplesChannel.send(samplesCopy)
            }
            Log.d(TAG, "Callback called with progress: $progress")
            1
        } else {
            Log.i(TAG, "Callback stopped")
            0
        }
    }

    private fun stopPlayback() {
        isStopped = true
        playbackJob?.cancel()
        audioTrack.pause()
        audioTrack.flush()
        stopMediaPlayer()
    }

    private fun stopMediaPlayer() {
        mediaPlayer?.apply {
            stop()
            release()
        }
        mediaPlayer = null
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TtsConversationScreen(
    onPlayConversation: suspend (List<DialogLine>, (Int) -> Unit) -> Unit, onStop: () -> Unit
) {
    var currentLineIndex by remember { mutableStateOf(-1) }
    var isPlaying by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()
    val listState = rememberLazyListState()

    val conversation = remember {
        listOf(
            DialogLine(1, "Hey guys! Are we meeting today for the project?", "Bella"),
            DialogLine(6, "Hi Bella! Yes, I think around 5 PM works for me.", "Nicole"),
            DialogLine(9, "I might be a bit late, traffic is crazy here.", "Sarah"),
            DialogLine(10, "No worries Sarah, we can start without you.", "Sky"),
            DialogLine(3, "Cool. Should I bring the laptops or just notes?", "Adam"),
            DialogLine(6, "Laptops are fine. We can do some coding live.", "Michael"),
            DialogLine(2, "Hey everyone! Just got in, did I miss anything?", "Emma"),
            DialogLine(
                8,
                "Not much, Emma. We are deciding whether to meet in lab or cafe.",
                "Isabella"
            ),
            DialogLine(4, "I vote for the lab. More quiet and we have all tools there.", "George"),
            DialogLine(7, "Lab it is then! See you all at 5 PM.", "Lewis"),
            DialogLine(1, "Great! See you guys later.", "Bella")
        )
    }

    // Auto-scroll to current line
    LaunchedEffect(currentLineIndex) {
        if (currentLineIndex >= 0) {
            listState.animateScrollToItem(
                index = currentLineIndex, scrollOffset = -200
            )
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(
                            "Live Conversation", style = MaterialTheme.typography.titleLarge
                        )
                        if (isPlaying) {
                            Text(
                                "Playing...",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f)
                            )
                        }
                    }
                }, colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    titleContentColor = MaterialTheme.colorScheme.onPrimaryContainer
                )
            )
        }) { paddingValues ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            LazyColumn(
                state = listState,
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                itemsIndexed(conversation) { index, line ->
                    DialogCard(
                        line = line,
                        isActive = index == currentLineIndex,
                        isPast = index < currentLineIndex
                    )
                }

                item {
                    Spacer(modifier = Modifier.height(80.dp))
                }
            }

            // Floating Action Buttons
            Row(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(16.dp)
                    .fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                FloatingActionButton(
                    onClick = {
                        if (isPlaying) {
                            onStop()
                            isPlaying = false
                            currentLineIndex = -1
                        } else {
                            isPlaying = true
                            scope.launch {
                                try {
                                    onPlayConversation(conversation) { index ->
                                        currentLineIndex = index
                                    }
                                } finally {
                                    isPlaying = false
                                    currentLineIndex = -1
                                }
                            }
                        }
                    },
                    modifier = Modifier.weight(1f),
                    containerColor = if (isPlaying) MaterialTheme.colorScheme.error
                    else MaterialTheme.colorScheme.primary
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 24.dp),
                        horizontalArrangement = Arrangement.Center,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            imageVector = if (isPlaying) Icons.Default.Stop else Icons.Default.PlayArrow,
                            contentDescription = if (isPlaying) "Stop" else "Play"
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = if (isPlaying) "Stop" else "Play Conversation",
                            style = MaterialTheme.typography.labelLarge
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun DialogCard(
    line: DialogLine, isActive: Boolean, isPast: Boolean
) {
    val scale by animateFloatAsState(
        targetValue = if (isActive) 1.05f else 1f, animationSpec = spring(
            dampingRatio = Spring.DampingRatioMediumBouncy, stiffness = Spring.StiffnessLow
        ), label = "scale"
    )

    val containerColor by animateColorAsState(
        targetValue = when {
            isActive -> MaterialTheme.colorScheme.primaryContainer
            isPast -> MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
            else -> MaterialTheme.colorScheme.surface
        }, label = "containerColor"
    )

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .scale(scale), colors = CardDefaults.cardColors(
            containerColor = containerColor
        ), elevation = CardDefaults.cardElevation(
            defaultElevation = if (isActive) 8.dp else 2.dp
        ), shape = RoundedCornerShape(16.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = line.speakerName,
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.Bold,
                    color = if (isActive) MaterialTheme.colorScheme.primary
                    else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                )

                AnimatedVisibility(
                    visible = isActive, enter = scaleIn() + fadeIn(), exit = scaleOut() + fadeOut()
                ) {
                    Box(
                        modifier = Modifier
                            .size(8.dp)
                            .background(
                                MaterialTheme.colorScheme.primary, shape = RoundedCornerShape(50)
                            )
                    )
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = line.text,
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = if (isActive) FontWeight.SemiBold else FontWeight.Normal,
                color = when {
                    isActive -> MaterialTheme.colorScheme.onPrimaryContainer
                    isPast -> MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f)
                    else -> MaterialTheme.colorScheme.onSurface
                },
                modifier = Modifier.fillMaxWidth()
            )

            if (isActive) {
                Spacer(modifier = Modifier.height(12.dp))
                LinearProgressIndicator(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(3.dp),
                    color = MaterialTheme.colorScheme.primary
                )
            }
        }
    }
}