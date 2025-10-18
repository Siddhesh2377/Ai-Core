package com.mp.ai_core

import android.content.Context
import android.os.Bundle
import android.os.IBinder
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.DividerDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import com.mp.ai_core.services.IGenerationCallback
import com.mp.ai_core.ui.theme.AiCoreTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File
import kotlin.math.roundToInt

private fun sessionFile(context: Context) = File(context.filesDir, "my_session.sess")

class TempActivity : ComponentActivity() {

    private var vm_state by mutableStateOf("")
    private var promptState by mutableStateOf(TextFieldValue("Hi"))
    private var modelPathState by mutableStateOf(TextFieldValue("/storage/emulated/0/Download/Models/lucy_128k-Q3_K_S.gguf"))
    private var stateSize by mutableLongStateOf(0L)
    private var isGenerating by mutableStateOf(false)
    private var isModelLoaded by mutableStateOf(false)
    private var useGPU by mutableStateOf(false)

    private var tokenCount by mutableIntStateOf(0)
    private var avgTokensPerSec by mutableFloatStateOf(0f)
    private var highestTokensPerSec by mutableFloatStateOf(0f)
    private var lowestTokensPerSec by mutableFloatStateOf(Float.MAX_VALUE)
    private var currentTokensPerSec by mutableFloatStateOf(0f)

    private val nativeLib = NativeLib.getGenerationInstance()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            AiCoreTheme {
                MainScreen()
            }
        }
    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    private fun MainScreen() {
        Scaffold(
            topBar = { TopAppBar(title = { Text("Llama-cpp Demo") }) },
            modifier = Modifier.fillMaxSize()
        ) { innerPadding ->
            Column(
                modifier = Modifier
                    .padding(innerPadding)
                    .padding(16.dp)
            ) {
                // MODEL PATH INPUT + LOAD / UNLOAD
                Text("Model Path:", style = MaterialTheme.typography.titleMedium)
                OutlinedTextField(
                    value = modelPathState,
                    onValueChange = { modelPathState = it },
                    colors = OutlinedTextFieldDefaults.colors(),
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 8.dp, bottom = 8.dp)
                )

                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier.padding(bottom = 16.dp)
                ) {
                    Button(
                        onClick = { loadModel() },
                        enabled = !isGenerating && !isModelLoaded
                    ) { Text("Load Model") }

                    Button(
                        onClick = {
                            unloadModel()
                            Toast.makeText(this@TempActivity, "Model unloaded", Toast.LENGTH_SHORT).show()
                        },
                        enabled = isModelLoaded
                    ) { Text("Unload Model") }
                }

                HorizontalDivider(modifier = Modifier.padding(bottom = 16.dp))

                // GPU Toggle
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 16.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "GPU Acceleration (OpenCL)",
                        style = MaterialTheme.typography.titleMedium
                    )
                    Switch(
                        checked = useGPU,
                        onCheckedChange = {
                            useGPU = it
                            if (isModelLoaded) loadModel()
                        },
                        enabled = !isGenerating && isModelLoaded
                    )
                }

                HorizontalDivider(modifier = Modifier.padding(bottom = 16.dp))

                // Token stats
                if (tokenCount > 0) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(bottom = 16.dp)
                    ) {
                        Text("Token Statistics", style = MaterialTheme.typography.titleMedium)
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(top = 8.dp),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Column {
                                Text("Tokens: $tokenCount", style = MaterialTheme.typography.bodySmall)
                                Text(
                                    "Average: ${avgTokensPerSec.roundToInt()} tok/s",
                                    style = MaterialTheme.typography.bodySmall
                                )
                            }
                            Column(horizontalAlignment = Alignment.End) {
                                Text(
                                    "Highest: ${highestTokensPerSec.roundToInt()} tok/s",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.primary
                                )
                                Text(
                                    "Lowest: ${
                                        if (lowestTokensPerSec == Float.MAX_VALUE) 0 else lowestTokensPerSec.roundToInt()
                                    } tok/s",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.secondary
                                )
                            }
                        }
                    }

                    HorizontalDivider(modifier = Modifier.padding(bottom = 16.dp))
                }

                // Prompt input
                Text("Prompt:", style = MaterialTheme.typography.titleMedium)
                BasicTextField(
                    value = promptState,
                    onValueChange = { promptState = it },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 8.dp, bottom = 24.dp)
                )

                // Buttons
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier.padding(bottom = 16.dp)
                ) {
                    Button(
                        onClick = { generatePrompt() },
                        enabled = !isGenerating && isModelLoaded && promptState.text.isNotBlank()
                    ) { Text("Generate") }

                    Button(onClick = { saveStateToFile() }, enabled = isModelLoaded) { Text("Save State") }
                    Button(onClick = { loadStateFromFile() }, enabled = isModelLoaded) { Text("Load State") }
                    Button(onClick = { nativeLib.nativeStopGeneration() }, enabled = isGenerating) { Text("Stop") }
                }

                // Model info
                Text(
                    text = "State size: ${stateSize / 1024} KiB (${stateSize} bytes)",
                    style = MaterialTheme.typography.bodySmall
                )
                Text(
                    text = "Mode: ${if (useGPU) "GPU (OpenCL)" else "CPU Only"}",
                    style = MaterialTheme.typography.bodySmall,
                    color = if (useGPU) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.secondary
                )

                HorizontalDivider(modifier = Modifier.padding(vertical = 12.dp))

                // Result area
                Text("Result:", style = MaterialTheme.typography.titleMedium)
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f)
                        .padding(top = 8.dp)
                        .verticalScroll(rememberScrollState()),
                ) {
                    Text(text = vm_state, modifier = Modifier.padding(12.dp))
                }
            }
        }
    }

    private fun loadModel() {
        val modelPath = modelPathState.text.trim()
        if (!File(modelPath).exists()) {
            Toast.makeText(this, "Model file not found!", Toast.LENGTH_LONG).show()
            return
        }

        lifecycleScope.launch(Dispatchers.IO) {
            isModelLoaded = false

            try {
                NativeLib.releaseInstance("generation")
            } catch (_: Exception) {}

            val gpuLayers = if (useGPU) 5 else 0
            runOnUiThread {
                Toast.makeText(
                    this@TempActivity,
                    "Loading model (${if (useGPU) "GPU" else "CPU"})...",
                    Toast.LENGTH_SHORT
                ).show()
            }

            val ok = nativeLib.initModel(
                path = modelPath,
                threads = Runtime.getRuntime().availableProcessors() / 2,
                gpuLayers = gpuLayers,
                ctxSize = 4096
            )

            runOnUiThread {
                if (!ok) {
                    Toast.makeText(
                        this@TempActivity,
                        "Failed to load model!",
                        Toast.LENGTH_LONG
                    ).show()
                } else {
                    isModelLoaded = true
                    stateSize = nativeLib.nativeGetStateSize()
                    Toast.makeText(
                        this@TempActivity,
                        "Model loaded successfully!",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }
        }
    }

    private fun unloadModel() {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                NativeLib.releaseInstance("generation")
                isModelLoaded = false
                stateSize = 0
            } catch (_: Exception) {}
        }
    }

    private fun generatePrompt() {
        val prompt = promptState.text
        isGenerating = true
        vm_state = ""

        // Reset stats for new generation
        tokenCount = 0
        val tokenTimes = mutableListOf<Long>()
        var lastTokenTime = System.currentTimeMillis()

        lifecycleScope.launch(Dispatchers.IO) {
            val start = System.currentTimeMillis()

            nativeLib.generateStreaming(
                prompt, maxTokens = 1256, callback = object : IGenerationCallback {
                    override fun onToken(token: String) {
                        val now = System.currentTimeMillis()
                        val timeSinceLastToken = now - lastTokenTime

                        if (tokenCount > 0 && timeSinceLastToken > 0) {
                            val tokensPerSec = 1000f / timeSinceLastToken
                            currentTokensPerSec = tokensPerSec
                            tokenTimes.add(timeSinceLastToken)

                            // Update highest/lowest
                            if (tokensPerSec > highestTokensPerSec) {
                                highestTokensPerSec = tokensPerSec
                            }
                            if (tokensPerSec < lowestTokensPerSec) {
                                lowestTokensPerSec = tokensPerSec
                            }
                        }

                        lastTokenTime = now
                        tokenCount++

                        launch(Dispatchers.Main) {
                            vm_state += token
                        }
                    }

                    override fun onToolCall(name: String, argsJson: String) {
                        launch(Dispatchers.Main) {
                            vm_state += "[TOOL: $name=$argsJson]"
                        }
                    }

                    override fun onDone() {
                        nativeLib.llamaPrintTimings()
                        isGenerating = false
                        stateSize = nativeLib.nativeGetStateSize()
                        val totalMs = System.currentTimeMillis() - start

                        // Calculate average tokens per second
                        if (tokenCount > 0 && totalMs > 0) {
                            avgTokensPerSec = (tokenCount * 1000f) / totalMs
                        }

                        // Reset lowest if it was never set
                        if (lowestTokensPerSec == Float.MAX_VALUE) {
                            lowestTokensPerSec = 0f
                        }

                        runOnUiThread {
                            Toast.makeText(
                                this@TempActivity,
                                "Done! $tokenCount tokens in $totalMs ms (${avgTokensPerSec.roundToInt()} tok/s avg)",
                                Toast.LENGTH_LONG
                            ).show()
                        }
                    }

                    override fun onError(message: String) {
                        isGenerating = false
                        runOnUiThread {
                            Toast.makeText(
                                this@TempActivity,
                                "Error: $message",
                                Toast.LENGTH_LONG
                            ).show()
                        }
                    }

                    override fun asBinder(): IBinder? {
                        return null
                    }
                }, toolsJson = ""
            )
        }
    }

    private fun saveStateToFile() {
        lifecycleScope.launch(Dispatchers.IO) {
            val data = nativeLib.nativeGetStateData()
            if (data == null) {
                launch(Dispatchers.Main) {
                    Toast.makeText(this@TempActivity, "No state to save", Toast.LENGTH_SHORT).show()
                }
                return@launch
            }

            val file = sessionFile(this@TempActivity)
            file.writeBytes(data)

            launch(Dispatchers.Main) {
                Toast.makeText(
                    this@TempActivity,
                    "Session (${data.size} bytes) written to ${file.name}",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
    }

    private fun loadStateFromFile() {
        lifecycleScope.launch(Dispatchers.IO) {
            val file = sessionFile(this@TempActivity)
            if (!file.exists()) {
                launch(Dispatchers.Main) {
                    Toast.makeText(this@TempActivity, "Session file not found", Toast.LENGTH_SHORT)
                        .show()
                }
                return@launch
            }

            val data = file.readBytes()
            val ok = nativeLib.nativeLoadStateData(data)

            launch(Dispatchers.Main) {
                if (ok) {
                    stateSize = nativeLib.nativeGetStateSize()
                    Toast.makeText(
                        this@TempActivity,
                        "Session restored (size $stateSize bytes)",
                        Toast.LENGTH_SHORT
                    ).show()
                } else {
                    Toast.makeText(
                        this@TempActivity,
                        "Failed to restore session",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }
        }
    }

    private fun updateStateSizeText() {
        stateSize = nativeLib.nativeGetStateSize()
    }

    override fun onDestroy() {
        super.onDestroy()
        NativeLib.releaseInstance("generation")
    }
}