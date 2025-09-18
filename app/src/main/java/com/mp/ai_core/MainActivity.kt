package com.mp.ai_core

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
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



        // init LLM (m1)


        setContent {
            MaterialTheme {
                var query by remember { mutableStateOf("") }
                var answer by remember { mutableStateOf("") }
                val scope = rememberCoroutineScope()
                var job: Job? by remember { mutableStateOf(null) }

                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(36.dp)
                        .verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    BasicTextField(
                        value = query,
                        onValueChange = { query = it },
                        modifier = Modifier
                            .fillMaxSize()
                            .weight(1f)
                    )

                    Button(onClick = {
                        if (query.isNotBlank()) {
                            job?.cancel() // cancel previous stream if running
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
                    }) {
                        Text("Ask")
                    }

                    Button(onClick = {
                        CoroutineScope(Dispatchers.IO).launch {
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
                            if (!ok) Log.e("AiCore", "Failed to init model at $m1")
                        }
                    }) {
                        Text("Load M1")
                    }

                    Text(text = answer, fontWeight = FontWeight.Bold)
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

        val topDocs = vectorStore.search(queryEmbedding, topK = 3)
        val context = topDocs.joinToString("\n") { it.text }
        val prompt = "Use the following context to answer:\n$context\n\nQuestion: $query"

        // 1️⃣ Ensure the text-gen model is loaded
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

        // 2️⃣ Generate streaming output
        return  native.generateStreaming(
            prompt = prompt,
            maxTokens = 512,
            uiScope = uiScope,
            onStart = { onUpdate("") }, // clear previous text
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
            "Android is an open-source operating system developed by Google.",
            "RAG stands for Retrieval Augmented Generation.",
            "Kotlin is officially supported for Android development.",
            "Android is the King of the mobile OS world."
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

    fun search(queryEmbedding: FloatArray, topK: Int = 3): List<Doc> {
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