package com.mp.ai_core

import android.content.ComponentName
import android.content.Intent
import android.content.ServiceConnection
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.Environment
import android.os.IBinder
import android.provider.Settings
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
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
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Divider
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.core.graphics.get
import androidx.core.graphics.scale
import androidx.core.net.toUri
import androidx.lifecycle.lifecycleScope
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
import java.nio.ByteBuffer
import kotlin.math.min

class MainActivity : ComponentActivity() {

    private val _serviceState = MutableStateFlow<IGenerationService?>(null)
    private val serviceState = _serviceState.asStateFlow()
    private lateinit var serviceConnection: ServiceConnection

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        requestStoragePermission()

        ContextCompat.startForegroundService(this, Intent(this, GenerationService::class.java))

        setContent {
            AiCoreTheme {
                MainScreen(serviceState)
            }
        }
    }

    override fun onStart() {
        super.onStart()
        bindToService()
    }

    override fun onStop() {
        super.onStop()
        unbindFromService()
    }

    private fun bindToService() {
        serviceConnection = object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
                _serviceState.value = (binder as? IGenerationService)?.also { initializeModels(it) }
            }

            override fun onServiceDisconnected(name: ComponentName?) {
                _serviceState.value = null
            }
        }
        bindService(
            Intent(this, GenerationService::class.java), serviceConnection, BIND_AUTO_CREATE
        )
    }

    private fun unbindFromService() {
        try {
            unbindService(serviceConnection)
        } catch (_: IllegalArgumentException) {
        }
    }

    private fun initializeModels(service: IGenerationService) {
        val vlmPath = "/storage/emulated/0/Download/VLM/LFM2-VL-450M-Q4_0.gguf"
        val projPath = "/storage/emulated/0/Download/VLM/mmproj-LFM2-VL-450M-Q8_0.gguf"
        val threads = min(Runtime.getRuntime().availableProcessors(), 8)

        lifecycleScope.launch(Dispatchers.IO) {
            val vlmOk = service.loadTextGenerationModel(
                vlmPath, threads, 0, true, 2048, 0.8f, 40, 0.95f, 0.1f
            )
            if (vlmOk) {
                val projOk = service.loadMultimodalProjector(projPath, threads)
                if (!projOk) {
                    launch(Dispatchers.Main) {
                        Toast.makeText(
                            this@MainActivity, "Projector load failed", Toast.LENGTH_SHORT
                        ).show()
                    }
                }
            } else {
                launch(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "VLM model load failed", Toast.LENGTH_SHORT)
                        .show()
                }
            }
        }

        copyAssetToCache("embedding.gguf")
    }

    private fun copyAssetToCache(assetName: String): File {
        val file = File(cacheDir, assetName)
        assets.open(assetName).use { input ->
            FileOutputStream(file).use { it.write(input.readBytes()) }
        }
        return file
    }

    private fun requestStoragePermission() {
        if (!Environment.isExternalStorageManager()) {
            startActivity(Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION).apply {
                data = "package:$packageName".toUri()
            })
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(serviceState: StateFlow<IGenerationService?>) {
    val service by serviceState.collectAsState()
    var selectedTab by remember { mutableIntStateOf(0) }
    val tabs = listOf("VLM", "Text", "Embedding")

    Scaffold(
        topBar = {
            Column {
                TopAppBar(
                    title = { Text("AI Core Demo") }, colors = TopAppBarDefaults.topAppBarColors(
                        containerColor = MaterialTheme.colorScheme.primaryContainer
                    )
                )
                TabRow(selectedTabIndex = selectedTab) {
                    tabs.forEachIndexed { index, title ->
                        Tab(
                            selected = selectedTab == index,
                            onClick = { selectedTab = index },
                            text = { Text(title) })
                    }
                }
            }
        }) { padding ->
        Box(modifier = Modifier.padding(padding)) {
            service?.let { svc ->
                when (selectedTab) {
                    0 -> VLMTab(svc)
                    1 -> TextTab(svc)
                    2 -> EmbeddingTab(svc)
                }
            } ?: Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(16.dp)
            ) {
                Text(
                    "Connecting to service...",
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

@Composable
fun VLMTab(service: IGenerationService) {
    val scope = rememberCoroutineScope()
    val context = LocalContext.current

    var modelPath by remember { mutableStateOf("/storage/emulated/0/Download/VLM/LFM2-VL-450M-Q4_0.gguf") }
    var projPath by remember { mutableStateOf("/storage/emulated/0/Download/VLM/mmproj-LFM2-VL-450M-Q8_0.gguf") }
    var isModelLoaded by remember { mutableStateOf(false) }
    var isLoading by remember { mutableStateOf(false) }

    var prompt by remember { mutableStateOf("Describe this image") }
    var output by remember { mutableStateOf("") }
    var isGenerating by remember { mutableStateOf(false) }

    val imagePath =
        "/data/data/com.mp.ai_core/files/fall-clipart-wallpaper-3840x2160-festive-decor-harvest-clipart-28488.jpg"

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text("Vision Language Model", style = MaterialTheme.typography.titleLarge)

        Card(
            modifier = Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(
                containerColor = if (isModelLoaded) MaterialTheme.colorScheme.primaryContainer
                else MaterialTheme.colorScheme.surfaceVariant
            )
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Model Loading", style = MaterialTheme.typography.titleMedium)
                Spacer(modifier = Modifier.height(8.dp))

                OutlinedTextField(
                    value = modelPath,
                    onValueChange = { modelPath = it },
                    label = { Text("Model Path") },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = !isModelLoaded && !isLoading,
                    singleLine = true
                )

                Spacer(modifier = Modifier.height(8.dp))

                OutlinedTextField(
                    value = projPath,
                    onValueChange = { projPath = it },
                    label = { Text("Projector Path") },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = !isModelLoaded && !isLoading,
                    singleLine = true
                )

                Spacer(modifier = Modifier.height(8.dp))

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = {
                            isLoading = true
                            scope.launch(Dispatchers.IO) {
                                val threads = min(Runtime.getRuntime().availableProcessors(), 8)
                                val ok = service.loadTextGenerationModel(
                                    modelPath, threads, 0, true, 2048, 0.8f, 40, 0.95f, 0.1f
                                )
                                if (ok) {
                                    val projOk = service.loadMultimodalProjector(projPath, threads)
                                    scope.launch(Dispatchers.Main) {
                                        if (projOk) {
                                            isModelLoaded = true
                                            Toast.makeText(
                                                context, "VLM loaded", Toast.LENGTH_SHORT
                                            ).show()
                                        } else {
                                            Toast.makeText(
                                                context, "Projector failed", Toast.LENGTH_SHORT
                                            ).show()
                                        }
                                        isLoading = false
                                    }
                                } else {
                                    scope.launch(Dispatchers.Main) {
                                        Toast.makeText(
                                            context, "Model load failed", Toast.LENGTH_SHORT
                                        ).show()
                                        isLoading = false
                                    }
                                }
                            }
                        }, enabled = !isModelLoaded && !isLoading, modifier = Modifier.weight(1f)
                    ) {
                        Text(if (isLoading) "Loading..." else "Load Model")
                    }

                    if (isModelLoaded) {
                        Button(
                            onClick = {
                                scope.launch(Dispatchers.IO) {
                                    service.unloadMultimodalProjector()
                                    service.unloadTextGenerationModel()
                                    scope.launch(Dispatchers.Main) {
                                        isModelLoaded = false
                                        Toast.makeText(
                                            context, "Model unloaded", Toast.LENGTH_SHORT
                                        ).show()
                                    }
                                }
                            },
                            modifier = Modifier.weight(1f),
                            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error)
                        ) {
                            Text("Unload")
                        }
                    }
                }
            }
        }

        Divider()

        OutlinedTextField(
            value = prompt,
            onValueChange = { prompt = it },
            label = { Text("Prompt") },
            modifier = Modifier.fillMaxWidth(),
            enabled = isModelLoaded && !isGenerating
        )

        Row(
            modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(
                onClick = {
                    if (prompt.isBlank()) {
                        Toast.makeText(context, "Enter a prompt", Toast.LENGTH_SHORT).show()
                        return@Button
                    }
                    isGenerating = true
                    output = ""
                    scope.launch(Dispatchers.IO) {
                        loadImageData(imagePath, 128)?.let { img ->
                            service.generateWithImage(
                                prompt,
                                img.bytes,
                                img.width,
                                img.height,
                                256,
                                "",
                                object : IGenerationCallback.Stub() {
                                    override fun onToken(token: String) {
                                        scope.launch(Dispatchers.Main) { output += token }
                                    }

                                    override fun onToolCall(name: String, payload: String) {}
                                    override fun onError(error: String) {
                                        scope.launch(Dispatchers.Main) {
                                            output += "\n[ERROR] $error"
                                            isGenerating = false
                                        }
                                    }

                                    override fun onDone() {
                                        scope.launch(Dispatchers.Main) { isGenerating = false }
                                    }
                                })
                        } ?: run {
                            scope.launch(Dispatchers.Main) {
                                Toast.makeText(context, "Failed to load image", Toast.LENGTH_SHORT)
                                    .show()
                                isGenerating = false
                            }
                        }
                    }
                }, enabled = isModelLoaded && !isGenerating, modifier = Modifier.weight(1f)
            ) {
                Text("Generate")
            }

            Button(
                onClick = {
                    scope.launch(Dispatchers.IO) { service.stopTextGeneration() }
                    isGenerating = false
                },
                enabled = isGenerating,
                modifier = Modifier.weight(1f),
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error)
            ) {
                Text("Stop")
            }
        }

        Card(
            modifier = Modifier.fillMaxWidth(), elevation = CardDefaults.cardElevation(4.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Output", style = MaterialTheme.typography.titleMedium)
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = output.ifEmpty { "Generated text will appear here..." },
                    style = MaterialTheme.typography.bodyMedium,
                    color = if (output.isEmpty()) MaterialTheme.colorScheme.onSurfaceVariant
                    else MaterialTheme.colorScheme.onSurface
                )
            }
        }
    }
}

@Composable
fun TextTab(service: IGenerationService) {
    val scope = rememberCoroutineScope()
    val context = LocalContext.current

    var modelPath by remember { mutableStateOf("/storage/emulated/0/Download/Kodify-Nano-2.0.Q8_0.gguf") }
    var isModelLoaded by remember { mutableStateOf(false) }
    var isLoading by remember { mutableStateOf(false) }

    var prompt by remember { mutableStateOf("Write a short story about AI") }
    var output by remember { mutableStateOf("") }
    var isGenerating by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text("Text Generation", style = MaterialTheme.typography.titleLarge)

        Card(
            modifier = Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(
                containerColor = if (isModelLoaded) MaterialTheme.colorScheme.primaryContainer
                else MaterialTheme.colorScheme.surfaceVariant
            )
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Model Loading", style = MaterialTheme.typography.titleMedium)
                Spacer(modifier = Modifier.height(8.dp))

                OutlinedTextField(
                    value = modelPath,
                    onValueChange = { modelPath = it },
                    label = { Text("Model Path") },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = !isModelLoaded && !isLoading,
                    singleLine = true
                )

                Spacer(modifier = Modifier.height(8.dp))

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = {
                            isLoading = true
                            scope.launch(Dispatchers.IO) {
                                service.unloadMultimodalProjector()
                                val threads = min(Runtime.getRuntime().availableProcessors(), 8)
                                val ok = service.loadTextGenerationModel(
                                    modelPath, threads, 0, true, 4096, 0.7f, 40, 0.95f, 0.05f
                                )
                                scope.launch(Dispatchers.Main) {
                                    if (ok) {
                                        isModelLoaded = true
                                        Toast.makeText(context, "Model loaded", Toast.LENGTH_SHORT)
                                            .show()
                                    } else {
                                        Toast.makeText(context, "Load failed", Toast.LENGTH_SHORT)
                                            .show()
                                    }
                                    isLoading = false
                                }
                            }
                        }, enabled = !isModelLoaded && !isLoading, modifier = Modifier.weight(1f)
                    ) {
                        Text(if (isLoading) "Loading..." else "Load Model")
                    }

                    if (isModelLoaded) {
                        Button(
                            onClick = {
                                scope.launch(Dispatchers.IO) {
                                    service.unloadTextGenerationModel()
                                    scope.launch(Dispatchers.Main) {
                                        isModelLoaded = false
                                        Toast.makeText(
                                            context, "Model unloaded", Toast.LENGTH_SHORT
                                        ).show()
                                    }
                                }
                            },
                            modifier = Modifier.weight(1f),
                            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error)
                        ) {
                            Text("Unload")
                        }
                    }
                }
            }
        }

        Divider()

        OutlinedTextField(
            value = prompt,
            onValueChange = { prompt = it },
            label = { Text("Prompt") },
            modifier = Modifier.fillMaxWidth(),
            enabled = isModelLoaded && !isGenerating,
            minLines = 3
        )

        Row(
            modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(
                onClick = {
                    if (prompt.isBlank()) {
                        Toast.makeText(context, "Enter a prompt", Toast.LENGTH_SHORT).show()
                        return@Button
                    }
                    isGenerating = true
                    output = ""
                    scope.launch(Dispatchers.IO) {
                        service.generateText(
                            prompt, 512, "", object : IGenerationCallback.Stub() {
                                override fun onToken(token: String) {
                                    scope.launch(Dispatchers.Main) { output += token }
                                }

                                override fun onToolCall(name: String, payload: String) {}
                                override fun onError(error: String) {
                                    scope.launch(Dispatchers.Main) {
                                        output += "\n[ERROR] $error"
                                        isGenerating = false
                                    }
                                }

                                override fun onDone() {
                                    scope.launch(Dispatchers.Main) { isGenerating = false }
                                }
                            })
                    }
                }, enabled = isModelLoaded && !isGenerating, modifier = Modifier.weight(1f)
            ) {
                Text("Generate")
            }

            Button(
                onClick = {
                    scope.launch(Dispatchers.IO) { service.stopTextGeneration() }
                    isGenerating = false
                },
                enabled = isGenerating,
                modifier = Modifier.weight(1f),
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error)
            ) {
                Text("Stop")
            }
        }

        Card(
            modifier = Modifier.fillMaxWidth(), elevation = CardDefaults.cardElevation(4.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Output", style = MaterialTheme.typography.titleMedium)
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = output.ifEmpty { "Generated text will appear here..." },
                    style = MaterialTheme.typography.bodyMedium,
                    color = if (output.isEmpty()) MaterialTheme.colorScheme.onSurfaceVariant
                    else MaterialTheme.colorScheme.onSurface
                )
            }
        }
    }
}

@Composable
fun EmbeddingTab(service: IGenerationService) {
    val scope = rememberCoroutineScope()
    val context = LocalContext.current

    var modelPath by remember { mutableStateOf("/data/data/com.mp.ai_core/cache/embedding.gguf") }
    var isModelLoaded by remember { mutableStateOf(false) }
    var isLoading by remember { mutableStateOf(false) }

    var text by remember { mutableStateOf("Hello, world!") }
    var embedding by remember { mutableStateOf("") }
    var isProcessing by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text("Text Embeddings", style = MaterialTheme.typography.titleLarge)

        Card(
            modifier = Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(
                containerColor = if (isModelLoaded) MaterialTheme.colorScheme.primaryContainer
                else MaterialTheme.colorScheme.surfaceVariant
            )
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Model Loading", style = MaterialTheme.typography.titleMedium)
                Spacer(modifier = Modifier.height(8.dp))

                OutlinedTextField(
                    value = modelPath,
                    onValueChange = { modelPath = it },
                    label = { Text("Model Path") },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = !isModelLoaded && !isLoading,
                    singleLine = true
                )

                Spacer(modifier = Modifier.height(8.dp))

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = {
                            isLoading = true
                            scope.launch(Dispatchers.IO) {
                                val threads = min(Runtime.getRuntime().availableProcessors(), 4)
                                val ok = service.loadEmbedModel(modelPath, threads, 512)
                                scope.launch(Dispatchers.Main) {
                                    if (ok) {
                                        isModelLoaded = true
                                        Toast.makeText(
                                            context, "Embedding model loaded", Toast.LENGTH_SHORT
                                        ).show()
                                    } else {
                                        Toast.makeText(context, "Load failed", Toast.LENGTH_SHORT)
                                            .show()
                                    }
                                    isLoading = false
                                }
                            }
                        }, enabled = !isModelLoaded && !isLoading, modifier = Modifier.weight(1f)
                    ) {
                        Text(if (isLoading) "Loading..." else "Load Model")
                    }

                    if (isModelLoaded) {
                        Button(
                            onClick = {
                                scope.launch(Dispatchers.IO) {
                                    service.unLoadEmbeddingModel()
                                    scope.launch(Dispatchers.Main) {
                                        isModelLoaded = false
                                        Toast.makeText(
                                            context, "Model unloaded", Toast.LENGTH_SHORT
                                        ).show()
                                    }
                                }
                            },
                            modifier = Modifier.weight(1f),
                            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error)
                        ) {
                            Text("Unload")
                        }
                    }
                }
            }
        }

        Divider()

        OutlinedTextField(
            value = text,
            onValueChange = { text = it },
            label = { Text("Text to embed") },
            modifier = Modifier.fillMaxWidth(),
            enabled = isModelLoaded && !isProcessing,
            minLines = 2
        )

        Button(
            onClick = {
                if (text.isBlank()) {
                    Toast.makeText(context, "Enter text", Toast.LENGTH_SHORT).show()
                    return@Button
                }
                isProcessing = true
                scope.launch(Dispatchers.IO) {
                    val result = service.embed(text)
                    scope.launch(Dispatchers.Main) {
                        embedding = if (result != null) {
                            "Dimension: ${result.size}\n\nFirst 10 values:\n" + result.take(10)
                                .joinToString("\n") { "%.6f".format(it) }
                        } else {
                            "Failed to generate embedding"
                        }
                        isProcessing = false
                    }
                }
            }, enabled = isModelLoaded && !isProcessing, modifier = Modifier.fillMaxWidth()
        ) {
            Text(if (isProcessing) "Processing..." else "Generate Embedding")
        }

        if (embedding.isNotEmpty()) {
            Card(
                modifier = Modifier.fillMaxWidth(), elevation = CardDefaults.cardElevation(4.dp)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text("Embedding Result", style = MaterialTheme.typography.titleMedium)
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = embedding,
                        style = MaterialTheme.typography.bodySmall,
                        fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace
                    )
                }
            }
        }
    }
}

data class ImageData(val bytes: ByteArray, val width: Int, val height: Int) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is ImageData) return false
        return width == other.width && height == other.height && bytes.contentEquals(other.bytes)
    }

    override fun hashCode(): Int = 31 * (31 * width + height) + bytes.contentHashCode()
}

fun loadImageData(path: String, maxDim: Int = 128): ImageData? {
    val bmp = BitmapFactory.decodeFile(path) ?: return null
    val scale = min(maxDim.toFloat() / bmp.width, maxDim.toFloat() / bmp.height)
    val scaled = bmp.scale((bmp.width * scale).toInt(), (bmp.height * scale).toInt())

    val buffer = ByteBuffer.allocate(scaled.width * scaled.height * 3)
    for (y in 0 until scaled.height) {
        for (x in 0 until scaled.width) {
            val pixel = scaled[x, y]
            buffer.put(((pixel shr 16) and 0xFF).toByte())
            buffer.put(((pixel shr 8) and 0xFF).toByte())
            buffer.put((pixel and 0xFF).toByte())
        }
    }
    return ImageData(buffer.array(), scaled.width, scaled.height)
}