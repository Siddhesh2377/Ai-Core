package com.mp.ai_core

import android.content.ComponentName
import android.content.Intent
import android.content.ServiceConnection
import android.graphics.Bitmap
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
import java.nio.ByteBuffer
import kotlin.math.min
import androidx.core.graphics.get
import androidx.core.graphics.scale

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
        val vlmModelPath = "/storage/emulated/0/Download/VLM/LFM2-VL-450M-Q4_0.gguf"
        val projectorPath = "/storage/emulated/0/Download/VLM/mmproj-LFM2-VL-450M-Q8_0.gguf"
        copyAssetToTemp("embedding.gguf") // for embedding

        val ok = _serviceState.value?.loadModel(
            vlmModelPath,
            min(Runtime.getRuntime().availableProcessors(), 8),
            0,
            true,
            2048,
            0.8f,
            40,
            0.95f,
            0.1f
        ) ?: false

        if (ok) {
            val projOk = _serviceState.value?.loadMultimodalProjector(
                projectorPath,
                min(Runtime.getRuntime().availableProcessors(), 8)
            ) ?: false

            if (!projOk) {
                Toast.makeText(this, "Failed to load multimodal projector", Toast.LENGTH_LONG).show()
            }
        } else {
            Toast.makeText(this, "Failed to load VLM model", Toast.LENGTH_LONG).show()
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

    var prompt by remember { mutableStateOf("Describe the image") }
    var output by remember { mutableStateOf("") }
    var generating by remember { mutableStateOf(false) }

    val context = LocalContext.current
    val imagePath = "/data/data/com.mp.ai_core/files/fall-clipart-wallpaper-3840x2160-festive-decor-harvest-clipart-28488.jpg"

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

                // Multimodal Generation Section
                Text("Multimodal Generation", style = MaterialTheme.typography.titleMedium)
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

                                    // Load image as Bitmap
                                    val imageData = loadBitmapReduced(imagePath, 128)
                                    val scaledBitmap = BitmapFactory.decodeFile(imagePath)?.scale(128, 128)


                                    logDebug("Generate with image clicked with prompt: $prompt")
                                    imageData?.let {
                                        service.generateWithImage(
                                            prompt,
                                            it.bytes,
                                            it.width,   // ✅ Correct dimensions
                                            it.height,  // ✅ Correct dimensions
                                            128,
                                            "{}",
                                            object : IGenerationCallback {
                                                override fun onToken(token: String) {
                                                    coroutineScope.launch(Dispatchers.Main) {
                                                        output += token
                                                    }
                                                }

                                                override fun onToolCall(name: String, payload: String) {}
                                                override fun onError(error: String) {
                                                    coroutineScope.launch(Dispatchers.Main) {
                                                        output += "\n[ERROR] $error"
                                                    }
                                                }

                                                override fun onDone() {
                                                    coroutineScope.launch(Dispatchers.Main) {
                                                        generating = false
                                                    }
                                                }

                                                override fun asBinder(): IBinder? = null
                                            }
                                        )
                                    }

                                }
                            } else {
                                Toast.makeText(context, "Prompt cannot be empty", Toast.LENGTH_SHORT).show()
                            }
                        },
                        enabled = !generating,
                        modifier = Modifier.weight(1f)
                    ) { Text("Generate with Image", style = MaterialTheme.typography.labelLarge) }

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
                    Column(modifier = Modifier.padding(16.dp)) {
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
            } ?: run {
                Text("Service unavailable", style = MaterialTheme.typography.bodyLarge)
            }
        }
    })
}

data class ImageData(val bytes: ByteArray, val width: Int, val height: Int) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as ImageData

        if (width != other.width) return false
        if (height != other.height) return false
        if (!bytes.contentEquals(other.bytes)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = width
        result = 31 * result + height
        result = 31 * result + bytes.contentHashCode()
        return result
    }
}

fun loadBitmapReduced(path: String, maxDim: Int = 1024): ImageData? {
    val bmp = BitmapFactory.decodeFile(path) ?: return null
    val scale = min(maxDim.toFloat() / bmp.width, maxDim.toFloat() / bmp.height)
    val scaledBmp = bmp.scale((bmp.width * scale).toInt(), (bmp.height * scale).toInt())

    val buffer = ByteBuffer.allocate(scaledBmp.width * scaledBmp.height * 3)
    for (y in 0 until scaledBmp.height) {
        for (x in 0 until scaledBmp.width) {
            val pixel = scaledBmp[x, y]
            buffer.put(((pixel shr 16) and 0xFF).toByte()) // R
            buffer.put(((pixel shr 8) and 0xFF).toByte())  // G
            buffer.put((pixel and 0xFF).toByte())           // B
        }
    }
    return ImageData(buffer.array(), scaledBmp.width, scaledBmp.height)
}
