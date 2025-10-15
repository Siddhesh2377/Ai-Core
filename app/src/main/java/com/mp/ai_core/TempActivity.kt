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
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import com.mp.ai_core.services.IGenerationCallback
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File

/** Dummy path – replace with the real location of your .bin/.gguf file. */
private const val MODEL_PATH = "/storage/emulated/0/Download/Models/lucy_128k-Q3_K_S.gguf"

/** File that we will use to persist the KV‑cache / prompt state. */
private fun sessionFile(context: Context) = File(context.filesDir, "my_session.sess")

class TempActivity : ComponentActivity() {

    /* UI‑state holders */
    private var vm_state by mutableStateOf("")            // where we show the result
    private var promptState by mutableStateOf(TextFieldValue("")) // prompt input
    private var stateSize by mutableLongStateOf(0L)          // state size in bytes
    private var isGenerating by mutableStateOf(false)

    private val nativeLib = NativeLib.getGenerationInstance()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { MainScreen() }

        // Load the model once …
        lifecycleScope.launch(Dispatchers.IO) {
            val ok = nativeLib.initModel(
                path = MODEL_PATH,
                threads = Runtime.getRuntime().availableProcessors() / 2,
                gpuLayers = 0,
                ctxSize = 4096 // you can pick another value if your device can hold it
            )
            if (!ok) {
                runOnUiThread {
                    Toast.makeText(this@TempActivity, "Failed to load model", Toast.LENGTH_LONG)
                        .show()
                }
            } else {
                // Show the initial state size (should be 0 until we send a prompt)
                stateSize = nativeLib.nativeGetStateSize()
                updateStateSizeText()
            }
        }
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
                        enabled = !isGenerating && promptState.text.isNotBlank()
                    ) { Text("Generate") }

                    Button(onClick = { saveStateToFile() }) { Text("Save State") }

                    Button(onClick = { loadStateFromFile() }) { Text("Load State") }

                    Button(onClick = { nativeLib.nativeStopGeneration() }) {
                        Text("Stop")
                    }
                }

                // State size
                Text(
                    text = "State size: ${stateSize / 1024} KiB (${stateSize} bytes)",
                    style = MaterialTheme.typography.bodySmall
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
                        text = vm_state, modifier = Modifier.padding(12.dp)
                    )
                }
            }
        }
    }

    /** ------------------------------------------------------------------
     *  Generate a stream from the native side.
     *  The callback updates the LiveData shown in the screen.
     * ------------------------------------------------------------------ */
    private fun generatePrompt() {
        val prompt = promptState.text
        isGenerating = true
        vm_state = ""
        lifecycleScope.launch(Dispatchers.IO) {
            val start = System.currentTimeMillis()
            nativeLib.generateStreaming(
                prompt, maxTokens = 1256, callback = object : IGenerationCallback {
                    override fun onToken(token: String) {
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
                        val ms = System.currentTimeMillis() - start
                        Toast.makeText(this@TempActivity, "Done ($ms ms)", Toast.LENGTH_SHORT)
                            .show()
                    }

                    override fun onError(message: String) {
                        isGenerating = false
                        Toast.makeText(this@TempActivity, "Error: $message", Toast.LENGTH_LONG)
                            .show()
                    }

                    override fun asBinder(): IBinder? {
                        return null
                    }
                }, toolsJson = ""
            )
        }
    }

    /** ------------------------------------------------------------------
     *  Persist the *entire* in‑memory state to a file.
     *  (This is a single file, not a whole cache + prompt array.)
     * ------------------------------------------------------------------ */
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

    /** ------------------------------------------------------------------
     *  Load a previously saved state from a file.
     * ------------------------------------------------------------------ */
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

    /** Just a helper that updates the Rich‑text label that shows
     *  the size in KiB.  Job is trivial, but makes the life of
     *  the UI a bit cleaner. */
    private fun updateStateSizeText() {
        stateSize = nativeLib.nativeGetStateSize()
        // Trigger recomposition – the `stateSize` property is @Composable‑upgradeable
    }

    override fun onDestroy() {
        super.onDestroy()
        // Release the instance when we are exiting.
        NativeLib.releaseInstance("generation")
    }
}