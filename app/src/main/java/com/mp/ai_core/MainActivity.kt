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
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.slideInVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
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
import com.mp.ai_core.ui.theme.AiCoreTheme
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch

private const val TAG = "MainActivity"

class MainActivity : ComponentActivity() {

    /* ---------- Service handling ---------- */
    private var service: IGenerationService? = null
    private var modelLoaded = false

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

    /* ---------- The LLM channel (global) ---------- */
    // A new channel is created each request – the old one is closed
    private var tokenChannel: Channel<String> = Channel(Channel.CONFLATED)

    private var answerFlow: StateFlow<String> =
        tokenChannel.consumeAsFlow().scan("") { acc, token -> acc + token }
            .stateIn(
                scope = lifecycleScope,
                started = SharingStarted.WhileSubscribed(5_000),
                initialValue = ""
            )

    /* ---------- Job – needed to cancel a running request ---------- */
    private var genJob: Job? = null

    /* ---------- UI state ---------- */
    private val _generating = mutableStateOf(false)
    private val generating: State<Boolean> get() = _generating

    /* ---------- Token counter ---------- */
    private val _tokenCount = mutableIntStateOf(0)
    private val tokenCount: State<Int> get() = _tokenCount

    private val _reachedFirstToken = mutableStateOf(false)
    private val reachedFirstToken: State<Boolean> get() = _reachedFirstToken

    /* ---------- Activity life‑cycle ---------- */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // 1️⃣ start the foreground service
        val svcIntent = Intent(this, GenerationService::class.java)
        startForegroundService(svcIntent)

        // 2️⃣ bind so we can call its RPC
        bindService(svcIntent, svcConnection, BIND_AUTO_CREATE)

        // 3️⃣ UI
        setContent { AiCoreTheme { AppUi() } }
    }

    override fun onDestroy() {
        super.onDestroy()
        // Service stays alive; you may unbind here if desired
    }

    /* ---------------------------------  UI ------------------------------------------------------ */
    @Composable
    private fun AppUi() {
        var query by remember { mutableStateOf("") }
        val answer by answerFlow.collectAsState()

        Scaffold {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(it)
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Text("ToolNeuron RAG Demo", style = MaterialTheme.typography.headlineSmall)

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
                        }
                    ) { Text("Ask") }

                    Button(onClick = { lifecycleScope.launch { shutdownPreviousJob() } }) {
                        Text("Stop")
                    }
                }

                Text(
                    "Answer:",
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 18.sp
                )

                Box(
                    Modifier
                        .fillMaxWidth()
                        .heightIn(min = 120.dp)
                        .padding(12.dp)
                ) {
                    /* ----------- 1️⃣ placeholder – only while *generating* and *no* token yet  ----------- */
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

                    /* ----------- 2️⃣ the live answer – always visible once *any* token arrives ----------- */
                    if (answer.isNotEmpty()) {
                        Text(answer, fontSize = 16.sp, color = MaterialTheme.colorScheme.primary)
                    } else {
                        // show an empty placeholder to keep the space reserved
                        Spacer(modifier = Modifier.height(16.dp))
                    }
                }
            }
        }
    }

    /* ---------------------------------  Service‑interaction helpers  --------------------------------- */
    /**
     * Cancels the current LLM job (if any) and waits for it to finish.
     * Resets the generating flag and the token counter so the placeholder disappears.
     */
    private suspend fun shutdownPreviousJob() {
        genJob?.cancelAndJoin()
        _generating.value = false
        _tokenCount.value = 0
    }

    /**
     * Calls the remote service to run `generate()`.
     * Opens a fresh channel/flow for each request and flips the generating flag.
     */
    private suspend fun runGeneration(prompt: String) {
        // Wait until the service is bound
        while (service == null) delay(50)

        // Create a brand‑new channel for this request
        tokenChannel = Channel(Channel.CONFLATED)
        answerFlow = tokenChannel.consumeAsFlow()
            .scan("") { acc, token -> acc + token }
            .stateIn(
                scope = lifecycleScope,
                started = SharingStarted.WhileSubscribed(5_000),
                initialValue = ""
            )

        _generating.value = true
        _tokenCount.value = 0

        // Callback that the service will call back through AIDL
        val cb = object : IGenerationCallback.Stub() {
            override fun onToken(token: String?) {
                val data = token ?: ""
                tokenChannel.trySend(data)

                // First token? Flip the flag once.
                if (! _reachedFirstToken.value) {
                    _reachedFirstToken.value = true
                }

                // Keep the visual counter alive
                _tokenCount.value++
            }

            override fun onToolCall(name: String?, argsJson: String?) {
                tokenChannel.trySend("[TOOL $name] $argsJson")
            }

            override fun onDone() {
                tokenChannel.close()
                _generating.value = false        // generation finished
            }

            override fun onError(err: String) {
                tokenChannel.trySend("ERROR: $err")
                tokenChannel.close()
                _generating.value = false
            }
        }

        // Launch the actual RPC in a coroutine
        genJob = lifecycleScope.launch {
            try {
                // `generate()` returns true on success; we ignore the result here
                service?.generate(prompt, 512, cb) ?: tokenChannel.close()
            } catch (e: RemoteException) {
                tokenChannel.trySend("RPC failure: ${e.message}")
                tokenChannel.close()
                _generating.value = false
            }
        }
    }

    /* ---------- Load shared model once per session ---------- */
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

    // wrapper that forwards to the AIDL call
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
}