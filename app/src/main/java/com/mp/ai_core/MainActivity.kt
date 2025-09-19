package com.mp.ai_core

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.lifecycleScope
import com.mp.ai_core.ui.theme.AiCoreTheme
import kotlinx.coroutines.*
import kotlin.math.sqrt

class MainActivity : ComponentActivity() {
    private val native = NativeLib()
    private lateinit var embeddingManager: EmbeddingManager
    private val vectorStore = VectorStore()
    val m1 = "/storage/emulated/0/Download/Models/Kodify-Nano-2.0.Q8_0.gguf"

    @OptIn(ExperimentalMaterial3Api::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        initializeEmbeddings()

        setContent {
            AiCoreTheme{
                var query by remember { mutableStateOf("") }
                var answer by remember { mutableStateOf("") }
                val scope = rememberCoroutineScope()
                var job: Job? by remember { mutableStateOf(null) }

                Scaffold {
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(it)
                            .verticalScroll(rememberScrollState()),
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        Text(
                            text = "ToolNeuron RAG Demo",
                            style = MaterialTheme.typography.headlineSmall.copy(
                                fontWeight = FontWeight.Bold,
                            )
                        )

                        // Query Box
                        BasicTextField(
                            value = query,
                            onValueChange = { query = it },
                            textStyle = TextStyle(fontSize = 16.sp),
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(56.dp)
                                .padding(horizontal = 12.dp, vertical = 8.dp)
                        )

                        // Buttons
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(12.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Button(
                                onClick = {
                                    if (query.isNotBlank()) {
                                        job?.cancel()
                                        answer = ""
                                        job = scope.launch {
                                            runRag(
                                                query = query,
                                                uiScope = scope,
                                                onUpdate = { token -> answer += token },
                                                onDone = { Log.i("RAG", "Streaming complete") },
                                                onError = { err -> answer = "Error: $err" }
                                            )
                                        }
                                    } else {
                                        answer = "⚠️ Please enter a query"
                                    }
                                },
                                shape = RoundedCornerShape(10.dp)
                            ) {
                                Text("Ask")
                            }

                            OutlinedButton(
                                onClick = {
                                    CoroutineScope(Dispatchers.IO).launch {
                                        val ok = native.initModel(
                                            path = m1,
                                            threads = Runtime.getRuntime().availableProcessors()
                                                .coerceAtLeast(2) - 1,
                                            gpuLayers = 10,
                                            useMMAP = true,
                                            useMLOCK = false,
                                            ctxSize = 4096,
                                            temp = 0.7f,
                                            topK = 40,
                                            topP = 0.9f,
                                            minP = 0.0f
                                        )
                                        if (!ok) Log.e("AiCore", "Failed to init model at $m1")
                                    }
                                },
                                shape = RoundedCornerShape(10.dp)
                            ) {
                                Text("Load Model")
                            }
                        }

                        // Answer Box
                        Text(
                            text = "Answer:",
                            fontWeight = FontWeight.SemiBold,
                            fontSize = 18.sp,
                            color = Color.White
                        )
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .heightIn(min = 120.dp)
                                .padding(12.dp)
                        ) {
                            Text(
                                text = answer,
                                fontSize = 16.sp,
                            )
                        }
                    }
                }
            }
        }
    }

    private suspend fun runRag(
        query: String,
        uiScope: CoroutineScope,
        onUpdate: (String) -> Unit,
        onDone: () -> Unit,
        onError: (String) -> Unit
    ): Job? {
        if (query.isBlank()) {
            onError("Please enter a valid query")
            return null
        }

        val queryEmbedding = embeddingManager.getEmbedding(query).getOrElse {
            onError("Error embedding query: ${it.message}")
            return null
        }

        val topDocs = vectorStore.search(queryEmbedding, topK = 5)
        val context = topDocs.joinToString("\n") { it.text }
        val prompt = "Use the following context to answer:\n$context\n\nQuestion: $query"

        val ok = native.initModel(
            path = m1,
            threads = Runtime.getRuntime().availableProcessors().coerceAtLeast(2) - 1,
            gpuLayers = 10,
            useMMAP = true,
            useMLOCK = false,
            ctxSize = 4096,
            temp = 0.7f,
            topK = 40,
            topP = 0.9f,
            minP = 0.0f
        )
        if (!ok) {
            onError("Failed to init model at $m1")
            return null
        }

        return native.generateStreaming(
            prompt = prompt,
            maxTokens = 512,
            uiScope = uiScope,
            onStart = { onUpdate("") },
            onGenerate = { token -> onUpdate(token) },
            onError = { err -> onError(err) },
            onDone = { onDone() }
        )
    }

    private fun initializeEmbeddings() {
        lifecycleScope.launch {
            val modelPath = "/storage/emulated/0/Download/ai/all-MiniLM-L6-v2-Q8_0.gguf"
            embeddingManager = EmbeddingManager(native)

            withContext(Dispatchers.IO) {
                embeddingManager.initializeEmbedding(
                    modelPath = modelPath,
                ).onSuccess {
                    Log.i("MainActivity", "Embedding model ready")
                    seedDocs()
                }.onFailure {
                    Log.e("MainActivity", "Embedding init failed: ${it.message}")
                }
            }
        }
    }

    private suspend fun seedDocs() {
        val docs = listOf(
            //My Life
            "Rey Lost Her Leg Yesterday",
            "Rey is a 25-year-old woman who lives in Bangalore.",
            "Rey is an avid reader and loves to travel.",
            "Rey has a dog named Max who she adores.",
            "Rey works as a software engineer at a startup in Bangalore.",
            "Rey lost her leg in a car accident yesterday.",
            "Rey is currently recovering from her surgery.",
            
            // Tech / AI
            "Android is an open-source operating system developed by Google.",
            "RAG stands for Retrieval Augmented Generation, a method to ground LLMs with external knowledge.",
            "Kotlin is officially supported for Android development and preferred for Jetpack Compose.",
            "Tool-Neuron is an offline-first AI operating system for mobile, built around modular plugins.",

            // Science
            "Cells are the basic building blocks of life.",
            "The sun is the star at the center of our solar system and provides energy for life on Earth.",
            "Water is essential for all known forms of life, covering 71% of Earth's surface.",

            // History / General Knowledge
            "The printing press was invented by Johannes Gutenberg in the 15th century.",
            "The Great Wall of China was built to protect against invasions and raids.",
            "World War II lasted from 1939 to 1945 and involved most of the world’s nations."
        )

        docs.forEachIndexed { i, text ->
            embeddingManager.getEmbedding(text).onSuccess { emb ->
                vectorStore.addDocument("doc$i", text, emb)
            }
        }
        Log.i("MainActivity", "Seeded ${docs.size} docs into vector store")
    }
}


data class Doc(val id: String, val text: String, val embedding: FloatArray)

class VectorStore {
    private val docs = mutableListOf<Doc>()

    fun addDocument(id: String, text: String, embedding: FloatArray) {
        docs.add(Doc(id, text, embedding))
    }

    fun search(queryEmbedding: FloatArray, topK: Int = 5): List<Doc> {
        return docs
            .map { it to cosineSimilarity(it.embedding, queryEmbedding) }
            .sortedByDescending { it.second }
            .take(topK)
            .map { it.first }
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Double {
        val dot = a.zip(b).sumOf { (x, y) -> (x * y).toDouble() }
        val normA = sqrt(a.sumOf { (it * it).toDouble() })
        val normB = sqrt(b.sumOf { (it * it).toDouble() })
        return if (normA != 0.0 && normB != 0.0) dot / (normA * normB) else 0.0
    }
}