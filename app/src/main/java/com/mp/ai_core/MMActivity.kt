package com.mp.ai_core

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.lifecycleScope
import coil.compose.AsyncImage
import com.mp.ai_core.ui.theme.AiCoreTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MMActivity : ComponentActivity() {

    private val mmLib = MMNativeLib()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            AiCoreTheme {
                MMUIScreen()
            }
        }
    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    fun MMUIScreen() {
        var prompt by remember { mutableStateOf("Describe the image") }
        var output by remember { mutableStateOf("") }
        var isLoading by remember { mutableStateOf(false) }
        var isInitialized by remember { mutableStateOf(false) }
        var errorMsg by remember { mutableStateOf("") }

        val modelPath = "/storage/emulated/0/Download/LFM2-VL-450M-Q4_0.gguf"
        val mmprojPath = "/storage/emulated/0/Download/mmproj-LFM2-VL-450M-F16.gguf"
        val imageFile = File("/storage/emulated/0/Download/ANIME-ENVIROMENT-WALLPAPER-4K.jpg")

        Scaffold(
            topBar = {
                CenterAlignedTopAppBar(
                    title = { Text("🧠 MM Vision Core", fontWeight = FontWeight.Bold) }
                )
            },
            containerColor = MaterialTheme.colorScheme.background
        ) { padding ->
            Column(
                modifier = Modifier
                    .padding(padding)
                    .fillMaxSize()
                    .padding(16.dp)
            ) {
                // Status indicator
                Text(
                    text = if (isInitialized) "✓ Model Loaded" else "⚠ Model Not Loaded",
                    color = if (isInitialized) MaterialTheme.colorScheme.primary
                    else MaterialTheme.colorScheme.error,
                    fontSize = 14.sp
                )

                Spacer(modifier = Modifier.height(8.dp))

                // Prompt input
                OutlinedTextField(
                    value = prompt,
                    onValueChange = { prompt = it },
                    label = { Text("Prompt") },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = !isLoading
                )

                Spacer(modifier = Modifier.height(12.dp))

                // Display the image
                AsyncImage(
                    model = imageFile,
                    contentDescription = "Input Image",
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(180.dp)
                )

                Spacer(modifier = Modifier.height(16.dp))

                // Initialize / Generate buttons row
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    // Initialize button
                    Button(
                        onClick = {
                            isLoading = true
                            errorMsg = ""
                            lifecycleScope.launch(Dispatchers.IO) {
                                try {
                                    val success = mmLib.nativeMMInit(
                                        modelPath,
                                        mmprojPath,
                                        Runtime.getRuntime().availableProcessors()
                                    )
                                    withContext(Dispatchers.Main) {
                                        if (success) {
                                            isInitialized = true
                                            output = "Model initialized successfully!"
                                        } else {
                                            errorMsg = "Failed to initialize model. Check paths."
                                        }
                                        isLoading = false
                                    }
                                } catch (e: Exception) {
                                    withContext(Dispatchers.Main) {
                                        errorMsg = "Error: ${e.message}"
                                        isLoading = false
                                    }
                                }
                            }
                        },
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(12.dp),
                        enabled = !isLoading && !isInitialized
                    ) {
                        if (isLoading && !isInitialized) {
                            CircularProgressIndicator(modifier = Modifier.size(20.dp))
                        } else {
                            Text("Initialize")
                        }
                    }

                    // Generate button
                    Button(
                        onClick = {
                            if (prompt.isNotBlank() && imageFile.exists()) {
                                isLoading = true
                                output = ""
                                errorMsg = ""
                                lifecycleScope.launch(Dispatchers.IO) {
                                    try {
                                        val callback = object : MMGenerateCallback {
                                            override fun onToken(token: String) {
                                                // Append token on the main thread
                                                lifecycleScope.launch(Dispatchers.Main) {
                                                    output += token
                                                }
                                            }

                                            override fun onComplete() {
                                                lifecycleScope.launch(Dispatchers.Main) {
                                                    isLoading = false
                                                }
                                            }
                                        }

                                        // Call JNI streaming function with callback
                                        mmLib.nativeMMGenerateStreaming(
                                            prompt,
                                            imageFile.absolutePath,
                                            256,       // max tokens
                                            callback    // pass it here
                                        )
                                    } catch (e: Exception) {
                                        withContext(Dispatchers.Main) {
                                            errorMsg = "Generation error: ${e.message}"
                                            isLoading = false
                                        }
                                    }
                                }

                            } else {
                                errorMsg = if (!imageFile.exists()) {
                                    "Image file not found: ${imageFile.absolutePath}"
                                } else {
                                    "Please enter a prompt"
                                }
                            }
                        },
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(12.dp),
                        enabled = !isLoading && isInitialized
                    ) {
                        if (isLoading && isInitialized) {
                            CircularProgressIndicator(modifier = Modifier.size(20.dp))
                        } else {
                            Text("Generate")
                        }
                    }
                }

                Spacer(modifier = Modifier.height(12.dp))

                // Error message
                if (errorMsg.isNotEmpty()) {
                    Text(
                        text = errorMsg,
                        color = MaterialTheme.colorScheme.error,
                        fontSize = 14.sp
                    )
                }

                Spacer(modifier = Modifier.height(12.dp))

                // Output box
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f)
                        .verticalScroll(rememberScrollState())
                        .padding(8.dp)
                ) {
                    if (output.isNotEmpty()) {
                        Text("Output:", fontWeight = FontWeight.Bold, fontSize = 14.sp)
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(output, fontSize = 16.sp)
                    } else if (isLoading) {
                        CircularProgressIndicator()
                    }
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        mmLib.nativeMMFree()
    }
}
