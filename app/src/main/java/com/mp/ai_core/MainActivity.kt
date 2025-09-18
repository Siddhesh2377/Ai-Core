package com.mp.ai_core

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking

// Pretend NativeLib + EmbeddingManager are imported from your JNI wrapper

class MainActivity : ComponentActivity() {

    private lateinit var generator: NativeLib      // m1: Kodify-Nano
    private lateinit var embedder: EmbeddingManager // m2: MiniLM
    private lateinit var vectorStore: VectorStore

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        generator = NativeLib()
        embedder = EmbeddingManager(generator)
        vectorStore = VectorStore(embedder)

        val m1 = "/storage/emulated/0/Download/Models/Kodify-Nano-2.0.Q8_0.gguf"
        val m2 = "/storage/emulated/0/Download/ai/all-MiniLM-L6-v2-Q8_0.gguf"

        // Init models
        generator.initModel(
            path = m1,
            nCtx = 4096
        )
        CoroutineScope(Dispatchers.IO).launch {
            embedder.initEmbeddings(
                modelPath = m2
            )
        }

        // Seed some docs for retrieval
        lifecycleScope.launch {
            vectorStore.addDocument("doc1", "Neuro-V is a modular AI automation app for Android.")
            vectorStore.addDocument("doc2", "Kodify-Nano is a small LLM optimized for offline inference.")
            vectorStore.addDocument("doc3", "MiniLM is used for generating embeddings for semantic search.")
        }

        setContent {
            MaterialTheme(colorScheme = lightColorScheme()) {
                RagChatUI(generator, vectorStore)
            }
        }
    }
}

@Composable
fun RagChatUI(generator: NativeLib, vectorStore: VectorStore) {
    var query by remember { mutableStateOf("") }
    var response by remember { mutableStateOf("") }
    var isLoading by remember { mutableStateOf(false) }

    val scope = rememberCoroutineScope()

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        Text("RAG Chat", style = MaterialTheme.typography.headlineSmall)
        Spacer(Modifier.height(16.dp))

        BasicTextField(
            value = query,
            onValueChange = { query = it },
            modifier = Modifier.fillMaxWidth().padding(8.dp)
        )
        Spacer(Modifier.height(8.dp))

        Button(
            onClick = {
                response = ""
                isLoading = true
                scope.launch {
                    ragQuery(
                        query = query,
                        vectorStore = vectorStore,
                        generator = generator,
                        scope = scope,
                        onGenerate = { chunk -> response += chunk },
                        onDone = { isLoading = false },
                        onError = { err ->
                            response = "Error: $err"
                            isLoading = false
                        }
                    )
                }
            }
        ) {
            Text("Ask")
        }

        Spacer(Modifier.height(16.dp))

        if (isLoading) Text("Generating...")

        Text(response, style = MaterialTheme.typography.bodyLarge)
    }
}

// ---------------------------
// RAG glue
// ---------------------------

fun ragQuery(
    query: String,
    vectorStore: VectorStore,
    generator: NativeLib,
    scope: CoroutineScope,
    onGenerate: (String) -> Unit,
    onDone: () -> Unit,
    onError: (String) -> Unit
) = generator.generateStreaming(
    prompt = buildPrompt(query, vectorStore),
    uiScope = scope,
    onStart = {},
    onGenerate = onGenerate,
    onError = onError,
    onDone = onDone
)

fun buildPrompt(query: String, vectorStore: VectorStore): String {
    val docs = vectorStore.query(query, k = 3)
    val context = docs.joinToString("\n") { it.text }
    return """
        You are a helpful assistant. Use the following context:
        ---
        $context
        ---
        Question: $query
    """.trimIndent()
}

// ---------------------------
// Minimal vector store
// ---------------------------

data class Document(val id: String, val text: String, val embedding: FloatArray)

class VectorStore(private val embedder: EmbeddingManager) {
    private val docs = mutableListOf<Document>()

    suspend fun addDocument(id: String, text: String) {
        embedder.getEmbedding(text).onSuccess { emb ->
            docs.add(Document(id, text, emb))
        }.onFailure { e ->
            Log.e("VectorStore", "Embedding failed: ${e.message}")
        }
    }

    fun query(query: String, k: Int = 3): List<Document> {
        val queryEmb = runBlocking { embedder.getEmbedding(query).getOrThrow() }
        return docs.sortedByDescending { embedder.cosineSimilarity(it.embedding, queryEmb) }.take(k)
    }
}