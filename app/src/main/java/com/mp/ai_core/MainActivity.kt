package com.mp.ai_core

import android.Manifest
import android.app.Activity
import android.content.*
import android.net.Uri
import android.os.*
import android.provider.Settings
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.LocalActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.mp.ai_core.text.GenerationService
import com.mp.ai_core.text.IGenerationCallback
import com.mp.ai_core.text.IGenerationService
import com.mp.ai_core.ui.theme.AiCoreTheme
import kotlinx.coroutines.*
import java.io.File
import java.io.FileOutputStream
import kotlin.math.min
import androidx.core.net.toUri

class MainActivity : ComponentActivity() {

    /** -------- Service binding --------------------------------------------------- */
    var iService: IGenerationService? = null
    private lateinit var serviceConnection: ServiceConnection

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        requestPermissionsIfNeeded()

        // Start the foreground service (UI will bind to it)
        val startIntent = Intent(this, GenerationService::class.java)
        ContextCompat.startForegroundService(this, startIntent)

        setContent { AiCoreTheme { MainScreen() } }
    }

    override fun onStart() {
        super.onStart()
        bindService()
    }

    override fun onStop() {
        super.onStop()
        unbindService()
    }

    /** -------- Service helpers ------------------------------------------------- */
    private fun bindService() {
        serviceConnection = object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
                iService = binder as? IGenerationService
                doInitialisation()
            }

            override fun onServiceDisconnected(name: ComponentName?) {
                iService = null
            }
        }
        val bindIntent = Intent(this, GenerationService::class.java)
        bindService(bindIntent, serviceConnection, BIND_AUTO_CREATE)
    }

    private fun unbindService() {
        try {
            unbindService(serviceConnection)
        } catch (e: IllegalArgumentException) {
            // service not bound – ignore
        }
    }

    /** -------- Initialise the model once we are bound ----------------------------- */
    private fun doInitialisation() {
        // 1️⃣  Path to the generation model (user copies it somewhere they can read)
        val generationModelPath = "/storage/emulated/0/Download/Kodify-Nano-2.0.Q8_0.gguf"

        // 2️⃣  Copy the embedding model from the assets folder to a temporary file
        copyAssetToTemp("embedding.gguf")

        // 3️⃣  Tell the service that it now has the inbound file paths
        val ok = iService?.loadModel(
            generationModelPath,
            min(Runtime.getRuntime().availableProcessors(), 8),
            0,
            true,
            2048,
            0.8f,
            40,
            0.95f,
            0.1f
        ) ?: false

        if (!ok) {
            Toast.makeText(this, "Failed to initialise LLM – check paths", Toast.LENGTH_LONG).show()
        }
    }

    /** -------- Asset → temp file helper ---------------------------------------- */
    private fun copyAssetToTemp(assetName: String): File {
        val tempFile = File(cacheDir, assetName)
        assets.open(assetName).use { input ->
            FileOutputStream(tempFile).use { out ->
                input.copyTo(out)
            }
        }
        return tempFile
    }

    /** -------- Permission handling ---------------------------------------------- */
    private fun requestPermissionsIfNeeded() {
        // API ≥ 30 → need the MANAGE_EXTERNAL_STORAGE permission
        if (!Environment.isExternalStorageManager()) {
            startActivity(
                Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION).apply {
                    data = "package:$packageName".toUri()
                }
            )
        }
    }
}

/** --------------------------------------------------------------------------- */
/** Compose UI – very small demo                                                        */
/** --------------------------------------------------------------------------- */

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen() {
    // Grab the service – it might still be null until the binding completes
    val coroutineScope = rememberCoroutineScope()
    val service = LocalActivity.current?.let { act ->
        @Suppress("DEPRECATION") remember { (act as? MainActivity)?.iService }
    }

    var prompt by remember { mutableStateOf("") }
    var output by remember { mutableStateOf("") }
    var generating by remember { mutableStateOf(false) }

    var embedText by remember { mutableStateOf("") }
    var embedVector by remember { mutableStateOf<FloatArray?>(null) }

    Scaffold(topBar = {
        TopAppBar(title = { Text("Ai Core Demo") })
    }, content = { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
                .verticalScroll(rememberScrollState())
        ) {
            // ---------- Text generation ----------
            Text("Text Generation", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(8.dp))
            OutlinedTextField(
                value = prompt,
                onValueChange = { prompt = it },
                label = { Text("Prompt") },
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(Modifier.height(8.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(
                    onClick = {
                        generating = true
                        coroutineScope.launch(Dispatchers.IO) {
                            service?.generate(
                                prompt, 128, "{}", object : IGenerationCallback {
                                    override fun onToken(token: String) {
                                        coroutineScope.launch(Dispatchers.Main) {
                                            output += token
                                        }
                                    }

                                    override fun onToolCall(name: String, payload: String) {
                                        // not used in this demo
                                    }

                                    override fun onError(error: String) {
                                        coroutineScope.launch(Dispatchers.Main) {
                                            output += "\n[ERROR] $error"
                                        }
                                    }

                                    override fun onDone() {}
                                    override fun asBinder(): IBinder? {
                                        return null
                                    }
                                })
                            coroutineScope.launch(Dispatchers.Main) {
                                generating = false
                            }
                        }
                    }, enabled = !generating && service != null
                ) { Text("Generate") }

                Button(
                    onClick = { service?.stopGeneration() }, enabled = generating
                ) { Text("Stop") }
            }

            Spacer(Modifier.height(12.dp))
            Text("Output:", style = MaterialTheme.typography.bodyMedium)
            Text(
                buildAnnotatedString { append(output) },
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color(0xFFF0F0F0))
                    .padding(8.dp)
            )

            // ---------- Embedding ----------
            Spacer(Modifier.height(24.dp))
            Text("Text Embedding", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(8.dp))
            OutlinedTextField(
                value = embedText,
                onValueChange = { embedText = it },
                label = { Text("Text") },
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(Modifier.height(8.dp))
            Button(
                onClick = {
                    coroutineScope.launch(Dispatchers.IO) {
                        val vec = service?.embed(embedText)
                        coroutineScope.launch(Dispatchers.Main) {
                            embedVector = vec
                        }
                    }
                }, enabled = embedText.isNotBlank() && service != null
            ) { Text("Embed") }

            Spacer(Modifier.height(8.dp))
            embedVector?.let { vec ->
                Text("Vector size: ${vec.size}")
                Text(
                    vec.joinToString(
                        prefix = "[", postfix = "]", separator = ", "
                    ) { "%0.4f".format(it) })
            }
        }
    })
}