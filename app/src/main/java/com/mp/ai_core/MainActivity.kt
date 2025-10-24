package com.mp.ai_core

import android.content.ComponentName
import android.content.Intent
import android.content.ServiceConnection
import android.os.Bundle
import android.os.Environment
import android.os.IBinder
import android.provider.Settings
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.core.net.toUri
import com.mp.ai_core.text.GenerationService
import com.mp.ai_core.text.IGenerationCallback
import com.mp.ai_core.text.IGenerationService
import com.mp.ai_core.ui.theme.AiCoreTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileOutputStream
import kotlin.math.min

class MainActivity : ComponentActivity() {

    private val _serviceState = MutableStateFlow<IGenerationService?>(null)
    private val serviceState: StateFlow<IGenerationService?>
        get() = _serviceState.asStateFlow()

    private lateinit var serviceConnection: ServiceConnection

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        requestPermissionsIfNeeded()

        // Start the foreground service (UI will bind to it)
        val startIntent = Intent(this, GenerationService::class.java)
        ContextCompat.startForegroundService(this, startIntent)

        setContent {
            AiCoreTheme {
                MainScreen(serviceState = serviceState)
            }
        }
    }

    override fun onStart() {
        super.onStart()
        bindService()
    }

    override fun onStop() {
        super.onStop()
        unbindService()
    }

    private fun bindService() {
        serviceConnection = object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
                _serviceState.value = binder as? IGenerationService
                doInitialisation()
            }

            override fun onServiceDisconnected(name: ComponentName?) {
                _serviceState.value = null
            }
        }
        val bindIntent = Intent(this, GenerationService::class.java)
        bindService(bindIntent, serviceConnection, BIND_AUTO_CREATE)
    }

    private fun unbindService() {
        try {
            unbindService(serviceConnection)
        } catch (e: IllegalArgumentException) {
            // Service not bound – ignore
        }
    }

    private fun doInitialisation() {
        val generationModelPath = "/storage/emulated/0/Download/Kodify-Nano-2.0.Q8_0.gguf"
        copyAssetToTemp("embedding.gguf")

        val ok = _serviceState.value?.loadModel(
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

    private fun copyAssetToTemp(assetName: String): File {
        val tempFile = File(cacheDir, assetName)
        assets.open(assetName).use { input ->
            FileOutputStream(tempFile).use { out ->
                input.copyTo(out)
            }
        }
        return tempFile
    }

    private fun requestPermissionsIfNeeded() {
        if (!Environment.isExternalStorageManager()) {
            startActivity(
                Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION).apply {
                    data = "package:$packageName".toUri()
                })
        }
    }
}

fun logDebug(message: String) {
    android.util.Log.d("AiCoreDemo", message)
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(serviceState: StateFlow<IGenerationService?>) {
    val coroutineScope = rememberCoroutineScope()
    val service by serviceState.collectAsState(initial = null)

    var prompt by remember { mutableStateOf("Hello") }
    var output by remember { mutableStateOf("") }
    var generating by remember { mutableStateOf(false) }
    var embedText by remember { mutableStateOf("") }
    var embedVector by remember { mutableStateOf<FloatArray?>(null) }

    val context = LocalContext.current

    Scaffold(topBar = {
        TopAppBar(
            title = { Text("Ai Core Demo", style = MaterialTheme.typography.headlineLarge) })
    }, content = { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            service?.let { service ->
                // Text Generation Section
                Text("Text Generation", style = MaterialTheme.typography.titleMedium)
                OutlinedTextField(
                    value = prompt,
                    onValueChange = { prompt = it },
                    label = { Text("Prompt", style = MaterialTheme.typography.bodyMedium) },
                    modifier = Modifier.fillMaxWidth(),
                    textStyle = MaterialTheme.typography.bodyMedium
                )
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Button(
                        onClick = {
                            if (prompt.isNotBlank()) {
                                generating = true
                                coroutineScope.launch(Dispatchers.IO) {
                                    logDebug("Generate button clicked with prompt: $prompt")
                                    service.generate(
                                        prompt, 128, "{}", object : IGenerationCallback {
                                            override fun onToken(token: String) {
                                                logDebug("Generated token: $token")
                                                coroutineScope.launch(Dispatchers.Main) {
                                                    output += token
                                                }
                                            }

                                            override fun onToolCall(
                                                name: String,
                                                payload: String
                                            ) {
                                                logDebug("Tool call: $name, payload: $payload")
                                            }

                                            override fun onError(error: String) {
                                                logDebug("Error: $error")
                                                coroutineScope.launch(Dispatchers.Main) {
                                                    output += "\n[ERROR] $error"
                                                }
                                            }

                                            override fun onDone() {
                                                logDebug("Generation done")
                                                coroutineScope.launch(Dispatchers.Main) {
                                                    generating = false
                                                }
                                            }

                                            override fun asBinder(): IBinder? {
                                                return null
                                            }
                                        })
                                }
                            } else {
                                Toast.makeText(
                                    context,
                                    "Prompt cannot be empty",
                                    Toast.LENGTH_SHORT
                                ).show()
                            }
                        },
                        enabled = !generating && service != null,
                        modifier = Modifier.weight(1f)
                    ) { Text("Generate", style = MaterialTheme.typography.labelLarge) }

                    Button(
                        onClick = {
                            generating = false
                            coroutineScope.launch(Dispatchers.IO) {
                                service.stopGeneration()
                                logDebug("Stop button clicked")
                            }
                        }, enabled = generating, modifier = Modifier.weight(1f)
                    ) { Text("Stop", style = MaterialTheme.typography.labelLarge) }
                }

                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 8.dp),
                    elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp)
                    ) {
                        Text("Output:", style = MaterialTheme.typography.titleSmall)
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = buildAnnotatedString { append(output) },
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(8.dp),
                            style = MaterialTheme.typography.bodySmall
                        )
                    }
                }

                // Embedding Section
                Text("Text Embedding", style = MaterialTheme.typography.titleMedium)
                OutlinedTextField(
                    value = embedText,
                    onValueChange = { embedText = it },
                    label = { Text("Text", style = MaterialTheme.typography.bodyMedium) },
                    modifier = Modifier.fillMaxWidth(),
                    textStyle = MaterialTheme.typography.bodyMedium
                )
                Button (
                    onClick = {
                        coroutineScope.launch(Dispatchers.IO) {
                            logDebug("Embed button clicked with text: $embedText")
                            val vec = service.embed(embedText)
                            coroutineScope.launch(Dispatchers.Main) {
                                embedVector = vec
                                logDebug("Embed vector: ${vec?.joinToString(", ")}")
                            }
                        }
                    },
                    modifier = Modifier.fillMaxWidth()
                ){
                    Text("Embed")
                }

                embedVector?.let { vec ->
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 8.dp),
                        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
                    ) {
                        Column(
                            modifier = Modifier.padding(16.dp)
                        ) {
                            Text(
                                "Vector size: ${vec.size}",
                                style = MaterialTheme.typography.titleSmall
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = vec.joinToString(
                                    prefix = "[", postfix = "]", separator = ", "
                                ) { "%.4f".format(it) },
                                style = MaterialTheme.typography.bodySmall
                            )
                        }
                    }
                }
            } ?: run {
                Text("Service unavailable", style = MaterialTheme.typography.bodyLarge)
            }
        }
    })
}