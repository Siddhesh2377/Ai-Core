package com.mp.ai_core

import android.content.Context
import android.os.Bundle
import android.os.IBinder
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material3.Button
import androidx.compose.material3.DividerDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File
import kotlin.math.roundToInt

private const val MODEL_PATH = "/storage/emulated/0/Download/jamba-reasoning-3b-Q4_K_M.gguf"

private fun sessionFile(context: Context) = File(context.filesDir, "my_session.sess")

class TempActivity : ComponentActivity() {

    /* UI state holders */
    private var vm_state by mutableStateOf("")
    private var promptState by mutableStateOf(TextFieldValue(""))
    private var stateSize by mutableLongStateOf(0L)
    private var isGenerating by mutableStateOf(false)
    private var isModelLoaded by mutableStateOf(false)

    /* GPU toggle */
    private var useGPU by mutableStateOf(true)

    /* Token statistics */
    private var tokenCount by mutableIntStateOf(0)
    private var avgTokensPerSec by mutableFloatStateOf(0f)
    private var highestTokensPerSec by mutableFloatStateOf(0f)
    private var lowestTokensPerSec by mutableFloatStateOf(Float.MAX_VALUE)
    private var currentTokensPerSec by mutableFloatStateOf(0f)

    private val nativeLib = NativeLib.getGenerationInstance()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { MainScreen() }

        // Load the model on startup
        loadModel()
    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    private fun MainScreen() {
        Scaffold(
            topBar = { TopAppBar(title = { Text("Llama‑cpp Demo") }) },
            modifier = Modifier.fillMaxSize()
        ) { innerPadding ->
            Column(
                modifier = Modifier
                    .padding(innerPadding)
                    .padding(16.dp)
            ) {
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
                            loadModel()
                        },
                        enabled = !isGenerating && isModelLoaded
                    )
                }

                HorizontalDivider(modifier = Modifier.padding(bottom = 16.dp))

                // Token Statistics
                if (tokenCount > 0) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(bottom = 16.dp)
                    ) {
                        Text(
                            text = "Token Statistics",
                            style = MaterialTheme.typography.titleMedium
                        )

                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(top = 8.dp),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Column {
                                Text(
                                    text = "Tokens: $tokenCount",
                                    style = MaterialTheme.typography.bodySmall
                                )
                                Text(
                                    text = "Average: ${avgTokensPerSec.roundToInt()} tok/s",
                                    style = MaterialTheme.typography.bodySmall
                                )
                            }
                            Column(horizontalAlignment = Alignment.End) {
                                Text(
                                    text = "Highest: ${highestTokensPerSec.roundToInt()} tok/s",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.primary
                                )
                                Text(
                                    text = "Lowest: ${if (lowestTokensPerSec == Float.MAX_VALUE) 0 else lowestTokensPerSec.roundToInt()} tok/s",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.secondary
                                )
                            }
                        }

                        if (isGenerating) {
                            Text(
                                text = "Current: ${currentTokensPerSec.roundToInt()} tok/s",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.tertiary,
                                modifier = Modifier.padding(top = 4.dp)
                            )
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

                    Button(
                        onClick = { saveStateToFile() },
                        enabled = isModelLoaded
                    ) { Text("Save State") }

                    Button(
                        onClick = { loadStateFromFile() },
                        enabled = isModelLoaded
                    ) { Text("Load State") }

                    Button(
                        onClick = { nativeLib.nativeStopGeneration() },
                        enabled = isGenerating
                    ) { Text("Stop") }
                }

                // State size
                Text(
                    text = "State size: ${stateSize / 1024} KiB (${stateSize} bytes)",
                    style = MaterialTheme.typography.bodySmall
                )

                Text(
                    text = "Mode: ${if (useGPU) "GPU (OpenCL)" else "CPU Only"}",
                    style = MaterialTheme.typography.bodySmall,
                    color = if (useGPU) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.secondary
                )

                HorizontalDivider(
                    modifier = Modifier.padding(vertical = 12.dp),
                    thickness = DividerDefaults.Thickness,
                    color = DividerDefaults.color
                )

                // Result area
                Text("Result:", style = MaterialTheme.typography.titleMedium)
                Surface(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f)
                        .padding(top = 8.dp),
                    shape = MaterialTheme.shapes.medium,
                    tonalElevation = 1.dp
                ) {
                    Text(
                        text = vm_state,
                        modifier = Modifier.padding(12.dp)
                    )
                }
            }
        }
    }

    private fun loadModel() {
        lifecycleScope.launch(Dispatchers.IO) {
            isModelLoaded = false

            // Release existing model if loaded
            try {
                NativeLib.releaseInstance("generation")
            } catch (_: Exception) {
                // Ignore if nothing to release
            }

            val gpuLayers = if (useGPU) -1 else 0

            runOnUiThread {
                Toast.makeText(
                    this@TempActivity,
                    "Loading model with ${if (useGPU) "GPU" else "CPU"}...",
                    Toast.LENGTH_SHORT
                ).show()
            }

            val ok = nativeLib.initModel(
                path = MODEL_PATH,
                threads = Runtime.getRuntime().availableProcessors() / 2,
                gpuLayers = gpuLayers,
                ctxSize = 4096
            )

            if (!ok) {
                runOnUiThread {
                    Toast.makeText(
                        this@TempActivity,
                        "Failed to load model with ${if (useGPU) "GPU" else "CPU"}",
                        Toast.LENGTH_LONG
                    ).show()
                }
            } else {
                isModelLoaded = true
                stateSize = nativeLib.nativeGetStateSize()
                updateStateSizeText()

                runOnUiThread {
                    Toast.makeText(
                        this@TempActivity,
                        "Model loaded with ${if (useGPU) "GPU (OpenCL)" else "CPU"}!",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }
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