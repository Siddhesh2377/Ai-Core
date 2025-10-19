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
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
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
    val voiceId: Int,
    val text: String,
    val speakerName: String,
    val speed: Float = 1.0f
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

    private var currentLineCallback: ((Int) -> Unit)? = null
    private var pendingLineIndex = -1

    private val _currentProgress = mutableStateOf(0f)
    val currentProgress: State<Float> = _currentProgress

    @OptIn(ExperimentalMaterial3Api::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        initializeTts()
        initAudioTrack()

        setContent {
            AiCoreTheme {
                TtsConversationScreen(
                    onPlayConversation = ::playConversation,
                    onStop = ::stopPlayback,
                    currentProgress = currentProgress
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
         "modelDir": "kokoro-en-v0_19",
         "modelName": "model.onnx",
         "voices": "voices.bin",
         "dataDir": "kokoro-en-v0_19/espeak-ng-data",
         "lang": "eng"
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
            bufLength,
            AudioTrack.MODE_STREAM,
            android.media.AudioManager.AUDIO_SESSION_ID_GENERATE
        )
        audioTrack.play()
    }

    private suspend fun playConversation(
        conversation: List<DialogLine>,
        onLineChange: (Int) -> Unit
    ) = withContext(Dispatchers.IO) {
        isStopped = false
        audioTrack.pause()
        audioTrack.flush()
        audioTrack.play()

        playbackJob = launch {
            for (samples in samplesChannel) {
                if (isStopped) break
                audioTrack.write(samples, 0, samples.size, AudioTrack.WRITE_BLOCKING)
            }
        }

        conversation.forEachIndexed { index, line ->
            if (isStopped) return@withContext

            // Store pending line info
            pendingLineIndex = index
            currentLineCallback = onLineChange

            TtsEngine.tts?.apply {
                currentSid = line.voiceId
                currentSpeed = line.speed

                generateWithCallback(
                    text = line.text,
                    callback = ::callback
                )
            }
        }

        withContext(Dispatchers.Main) {
            onLineChange(-1)
            _currentProgress.value = 0f
        }
    }

    fun callback(samples: FloatArray, progress: Float): Int {
        return if (!isStopped) {
            // Update UI on FIRST callback when audio actually starts
            if (pendingLineIndex >= 0 && progress > 0f) {
                val lineToShow = pendingLineIndex
                CoroutineScope(Dispatchers.Main).launch {
                    currentLineCallback?.invoke(lineToShow)
                }
                pendingLineIndex = -1
            }

            val samplesCopy = samples.copyOf()
            CoroutineScope(Dispatchers.IO).launch {
                samplesChannel.send(samplesCopy)
            }
            _currentProgress.value = progress
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
        _currentProgress.value = 0f
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
    onPlayConversation: suspend (List<DialogLine>, (Int) -> Unit) -> Unit,
    onStop: () -> Unit,
    currentProgress: State<Float>
) {
    var currentLineIndex by remember { mutableIntStateOf(-1) }
    var isPlaying by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()
    val listState = rememberLazyListState()

    val conversation = remember {
        listOf(
            // Team Human vs Team Robot - Balanced speeds
            DialogLine(0, "Okay so hear me out, AI is definitely gonna take over and we're all cooked.", "Bella", 1.05f),
            DialogLine(5, "Bella that's so dramatic, AI can't even make a decent meme yet.", "Adam", 1.1f),
            DialogLine(2, "No literally, I asked ChatGPT for dating advice and it told me to touch grass.", "Nicole", 1.15f),
            DialogLine(6, "That's actually solid advice though, no cap.", "Michael", 1.05f),
            DialogLine(3, "Michael you're literally defending the robots right now, sus behavior.", "Sarah", 1.1f),
            DialogLine(9, "I mean statistically AI will replace like 80% of jobs by 2030.", "George", 1.0f),
            DialogLine(4, "George bestie, you're not helping the vibe right now.", "Sky", 1.15f),
            DialogLine(10, "But like imagine AI doing all the boring stuff while we just chill?", "Lewis", 1.05f),
            DialogLine(0, "Lewis that's what they want you to think, it's giving Skynet energy.", "Bella", 1.1f),
            DialogLine(5, "Skynet? Bella you need to stop watching old movies.", "Adam", 1.05f),
            DialogLine(2, "Okay but real talk, AI art is kind of stealing from actual artists.", "Nicole", 1.0f),
            DialogLine(6, "Fair point, but it's also making art accessible to everyone.", "Michael", 1.05f),
            DialogLine(3, "Accessible? More like putting people out of jobs, periodt.", "Sarah", 1.1f),
            DialogLine(9, "The industrial revolution had the same arguments honestly.", "George", 1.0f),
            DialogLine(4, "Yeah but the industrial revolution didn't have robots writing TikTok scripts.", "Sky", 1.15f),
            DialogLine(10, "Wait AI can write TikTok scripts? That's lowkey fire.", "Lewis", 1.1f),
            DialogLine(0, "Lewis I'm gonna need you to pick a side here.", "Bella", 1.15f),
            DialogLine(5, "There are no sides, we're just having a conversation.", "Adam", 1.0f),
            DialogLine(2, "Adam that's such a centrist take, I can't with you right now.", "Nicole", 1.15f),
            DialogLine(6, "Can we all agree AI making our coffee is a W though?", "Michael", 1.1f),
            DialogLine(3, "Michael the coffee machine isn't AI, it's just a machine.", "Sarah", 1.05f),
            DialogLine(9, "Technically any programmed automation could be considered basic AI.", "George", 1.0f),
            DialogLine(4, "George please stop being smart for like two seconds.", "Sky", 1.2f),
            DialogLine(10, "So bottom line, are we team human or team robot?", "Lewis", 1.05f),
            DialogLine(0, "Team human obviously, I'm not about to betray my species.", "Bella", 1.1f),
            DialogLine(5, "I'm team coexistence, we can vibe with the bots.", "Adam", 1.05f),
            DialogLine(2, "That's it, Adam is officially a robot spy, confirmed.", "Nicole", 1.2f),
            DialogLine(6, "This whole debate is unhinged and I love it.", "Michael", 1.1f)
        )
    }

    LaunchedEffect(currentLineIndex) {
        if (currentLineIndex >= 0) {
            listState.animateScrollToItem(
                index = currentLineIndex,
                scrollOffset = -300
            )
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                Brush.verticalGradient(
                    colors = listOf(
                        MaterialTheme.colorScheme.surface,
                        MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f)
                    )
                )
            )
    ) {
        Column(
            modifier = Modifier.fillMaxSize()
        ) {
            TopAppBar(
                title = {
                    Column {
                        Text(
                            "Live Conversation",
                            style = MaterialTheme.typography.titleLarge.copy(
                                fontWeight = FontWeight.Bold
                            )
                        )
                        if (isPlaying) {
                            Text(
                                "Now playing...",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.primary.copy(alpha = 0.8f)
                            )
                        }
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = Color.Transparent
                )
            )

            LazyColumn(
                state = listState,
                modifier = Modifier
                    .fillMaxSize()
                    .weight(1f),
                contentPadding = PaddingValues(horizontal = 24.dp, vertical = 16.dp),
                verticalArrangement = Arrangement.spacedBy(32.dp)
            ) {
                itemsIndexed(conversation) { index, line ->
                    LyricsStyleDialogLine(
                        line = line,
                        isActive = index == currentLineIndex,
                        isPast = index < currentLineIndex,
                        isFuture = index > currentLineIndex,
                        progress = if (index == currentLineIndex) currentProgress.value else 0f
                    )
                }

                item {
                    Spacer(modifier = Modifier.height(120.dp))
                }
            }
        }

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
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 32.dp)
                .size(72.dp),
            containerColor = if (isPlaying)
                MaterialTheme.colorScheme.error
            else
                MaterialTheme.colorScheme.primary,
            shape = CircleShape
        ) {
            Icon(
                imageVector = if (isPlaying) Icons.Default.Stop else Icons.Default.PlayArrow,
                contentDescription = if (isPlaying) "Stop" else "Play",
                modifier = Modifier.size(32.dp)
            )
        }
    }
}

@Composable
fun LyricsStyleDialogLine(
    line: DialogLine,
    isActive: Boolean,
    isPast: Boolean,
    isFuture: Boolean,
    progress: Float
) {
    val scale by animateFloatAsState(
        targetValue = when {
            isActive -> 0.94f
            isPast -> 0.96f
            else -> 0.94f
        },
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioNoBouncy,
            stiffness = Spring.StiffnessHigh
        ),
        label = "scale"
    )

    val alpha by animateFloatAsState(
        targetValue = when {
            isActive -> 1f
            isPast -> 0.35f
            else -> 0.48f
        },
        animationSpec = tween(200, easing = FastOutSlowInEasing),
        label = "alpha"
    )

    val blur by animateFloatAsState(
        targetValue = if (isActive) 0f else if (isFuture) 2f else 2.5f,
        animationSpec = tween(200, easing = FastOutSlowInEasing),
        label = "blur"
    )

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .scale(scale)
            .alpha(alpha)
            .blur(blur.dp),
        horizontalAlignment = if (isActive) Alignment.CenterHorizontally else Alignment.Start
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier.padding(bottom = 6.dp)
        ) {
            if (isActive) {
                Box(
                    modifier = Modifier
                        .size(8.dp)
                        .clip(CircleShape)
                        .background(MaterialTheme.colorScheme.primary)
                ) {
                    val infiniteTransition = rememberInfiniteTransition(label = "pulse")
                    val pulseScale by infiniteTransition.animateFloat(
                        initialValue = 1f,
                        targetValue = 1.6f,
                        animationSpec = infiniteRepeatable(
                            animation = tween(550, easing = FastOutSlowInEasing),
                            repeatMode = RepeatMode.Reverse
                        ),
                        label = "pulse"
                    )
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .scale(pulseScale)
                            .background(
                                MaterialTheme.colorScheme.primary.copy(alpha = 0.3f),
                                CircleShape
                            )
                    )
                }
            }

            Text(
                text = line.speakerName,
                style = MaterialTheme.typography.labelMedium.copy(
                    fontWeight = if (isActive) FontWeight.Bold else FontWeight.Medium,
                    fontSize = if (isActive) 13.sp else 11.sp
                ),
                color = if (isActive)
                    MaterialTheme.colorScheme.primary
                else
                    MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
            )
        }

        Box(
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(
                text = line.text,
                style = MaterialTheme.typography.headlineSmall.copy(
                    fontWeight = if (isActive) FontWeight.SemiBold else FontWeight.Normal,
                    fontSize = when {
                        isActive -> 26.sp
                        isPast -> 19.sp
                        else -> 20.sp
                    },
                    lineHeight = when {
                        isActive -> 34.sp
                        else -> 26.sp
                    }
                ),
                textAlign = if (isActive) TextAlign.Center else TextAlign.Start,
                color = when {
                    isActive -> MaterialTheme.colorScheme.primary
                    isPast -> MaterialTheme.colorScheme.onSurface.copy(alpha = 0.35f)
                    else -> MaterialTheme.colorScheme.onSurface.copy(alpha = 0.45f)
                },
                modifier = Modifier.fillMaxWidth()
            )
        }

        if (isActive) {
            Spacer(modifier = Modifier.height(14.dp))

            LinearProgressIndicator(
                progress = { progress },
                modifier = Modifier
                    .fillMaxWidth(0.7f)
                    .height(3.dp)
                    .clip(RoundedCornerShape(1.5.dp))
                    .align(Alignment.CenterHorizontally),
                color = MaterialTheme.colorScheme.primary,
                trackColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.15f),
            )
        }
    }
}