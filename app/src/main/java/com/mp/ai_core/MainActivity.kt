package com.mp.ai_core

import android.os.Build
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import com.mp.ai_core.ui.theme.AiCoreTheme
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch
import org.json.JSONObject
import kotlin.math.sqrt
import kotlin.text.ifEmpty

class MainActivity : ComponentActivity() {
    private val native = NativeLib()
    private var streamJob: Job? = null
    private lateinit var embeddingManager: EmbeddingManager


    @OptIn(ExperimentalMaterial3Api::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        initializeEmbeddings()
        val modelPath: MutableStateFlow<String?> = MutableStateFlow("")


        val m1 = "/storage/emulated/0/Download/Models/Kodify-Nano-2.0.Q8_0.gguf"
        val m2 = "/storage/emulated/0/Download/ai/all-MiniLM-L6-v2-Q8_0.gguf"

        CoroutineScope(Dispatchers.IO).launch {
            modelPath.value = "/storage/emulated/0/Download/ai/all-MiniLM-L6-v2-Q8_0.gguf"

            val ok = native.initModel(
                path = modelPath.value ?: "/storage/emulated/0/Download/ai/all-MiniLM-L6-v2-Q8_0.gguf",
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
            if (!ok) Log.e("AiSampleToolActivity", "Failed to init model at $modelPath")
        }

        // Slim system prompt: the JNI side will also inject tool preamble
        //native.setSystemPrompt("You are a concise assistant. Prefer calling tools when helpful.")

        setContent {
            MaterialTheme {
                var prompt by remember { mutableStateOf("Hello") }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            native.nativeStopGeneration()
            native.nativeRelease()
            embeddingManager.release()
        } catch (t: Throwable) {
            Log.w("AiSampleToolActivity", "release error", t)
        }
    }

    private fun initializeEmbeddings() {
        lifecycleScope.launch {
            val modelPath = "/storage/emulated/0/Download/ai/all-MiniLM-L6-v2-Q8_0.gguf"

            embeddingManager = EmbeddingManager(native)

            embeddingManager.initializeEmbedding(
                modelPath = modelPath,
                contextSize = 2048,
                gpuLayers = 0
            ).fold(
                onSuccess = {
                    Log.i("MainActivity", "Embedding model loaded successfully")
                    testEmbeddings()
                },
                onFailure = { error ->
                    Log.e("MainActivity", "Failed to load embedding model: ${error.message}")
                }
            )
        }
    }

    private suspend fun testEmbeddings() {
        // Test single embedding
        embeddingManager.getEmbedding("Hello world").fold(
            onSuccess = { embedding ->
                Log.i("MainActivity", "Embedding generated: ${embedding.size} dimensions")
                Log.i("MainActivity", "First few values: ${embedding.take(5)}")
            },
            onFailure = { error ->
                Log.e("MainActivity", "Embedding failed: ${error.message}")
            }
        )

        // Test similarity
        val text1 = "The cat sat on the mat"
        val text2 = "A cat was sitting on a mat"

        val embedding1Result = embeddingManager.getEmbedding(text1)
        val embedding2Result = embeddingManager.getEmbedding(text2)

        if (embedding1Result.isSuccess && embedding2Result.isSuccess) {
            val similarity = embeddingManager.cosineSimilarity(
                embedding1Result.getOrThrow(),
                embedding2Result.getOrThrow()
            )
            Log.i("MainActivity", "Similarity between texts: $similarity")
        }
    }
    private fun buildDeviceInfoJson(): String {
        return JSONObject()
            .put("manufacturer", Build.MANUFACTURER)
            .put("model", Build.MODEL)
            .put("sdk", Build.VERSION.SDK_INT)
            .put("device", Build.DEVICE)
            .toString()
    }

    fun runTool(name: String, argsJson: String): String {
        fun ok(obj: JSONObject) = obj.put("ok", true).toString()
        fun err(msg: String, extra: JSONObject? = null) =
            JSONObject().put("ok", false).put("error", msg).apply { if (extra != null) put("extra", extra) }.toString()

        return try {
            val root = JSONObject(argsJson)
            val args = root.optJSONArray("tool_calls")?.optJSONObject(0)?.optJSONObject("arguments")
                ?: root.optJSONObject("arguments") ?: root

            when (name) {
                "device_info" -> buildDeviceInfoJson()

                "sum" -> {
                    val a = args.optDouble("a")
                    val b = args.optDouble("b")
                    ok(JSONObject().put("a", a).put("b", b).put("sum", a + b))
                }

                "stats" -> {
                    val arr = args.getJSONArray("values")
                    val list = (0 until arr.length()).map { arr.getDouble(it) }
                    val n = list.size
                    val sum = list.sum()
                    val mean = sum / n
                    val sorted = list.sorted()
                    val median = if (n % 2 == 1) sorted[n/2] else (sorted[n/2 - 1] + sorted[n/2]) / 2.0
                    val variance = list.fold(0.0) { acc, x -> acc + (x - mean) * (x - mean) } / n
                    val stdev = kotlin.math.sqrt(variance)
                    ok(JSONObject()
                        .put("count", n)
                        .put("sum", sum)
                        .put("mean", mean)
                        .put("median", median)
                        .put("stdev", stdev))
                }

                "unit_convert" -> {
                    val quantity = args.getString("quantity")
                    val from = args.getString("from").lowercase()
                    val to = args.getString("to").lowercase()
                    val value = args.getDouble("value")

                    fun lengthToMeters(v: Double, u: String) = when (u) {
                        "m" -> v
                        "cm" -> v / 100.0
                        "km" -> v * 1000.0
                        "in" -> v * 0.0254
                        "ft" -> v * 0.3048
                        "mi" -> v * 1609.344
                        else -> Double.NaN
                    }
                    fun metersTo(v: Double, u: String) = when (u) {
                        "m" -> v
                        "cm" -> v * 100.0
                        "km" -> v / 1000.0
                        "in" -> v / 0.0254
                        "ft" -> v / 0.3048
                        "mi" -> v / 1609.344
                        else -> Double.NaN
                    }
                    fun tempToK(v: Double, u: String) = when (u) {
                        "k" -> v
                        "c" -> v + 273.15
                        "f" -> (v - 32.0) * 5.0/9.0 + 273.15
                        else -> Double.NaN
                    }
                    fun kTo(v: Double, u: String) = when (u) {
                        "k" -> v
                        "c" -> v - 273.15
                        "f" -> (v - 273.15) * 9.0/5.0 + 32.0
                        else -> Double.NaN
                    }

                    val result = when (quantity) {
                        "length" -> {
                            val m = lengthToMeters(value, from)
                            val out = metersTo(m, to)
                            out
                        }
                        "temperature" -> {
                            val k = tempToK(value, from)
                            val out = kTo(k, to)
                            out
                        }
                        else -> Double.NaN
                    }

                    if (result.isNaN()) err("Unsupported unit conversion", JSONObject(argsJson))
                    else ok(JSONObject().put("quantity", quantity).put("from", from).put("to", to).put("input", value).put("output", result))
                }

                "regex_extract" -> {
                    val pattern = args.getString("pattern")
                    val input = args.getString("input")
                    val flagsArr = args.optJSONArray("flags") ?: org.json.JSONArray()
                    var flags = 0
                    for (i in 0 until flagsArr.length()) {
                        flags = flags or when (flagsArr.getString(i)) {
                            "i" -> java.util.regex.Pattern.CASE_INSENSITIVE
                            "m" -> java.util.regex.Pattern.MULTILINE
                            "s" -> java.util.regex.Pattern.DOTALL
                            "u" -> java.util.regex.Pattern.UNICODE_CASE
                            else -> 0
                        }
                    }
                    val p = java.util.regex.Pattern.compile(pattern, flags)
                    val m = p.matcher(input)
                    val matches = org.json.JSONArray()
                    while (m.find()) {
                        val one = org.json.JSONArray()
                        for (g in 0..m.groupCount()) one.put(m.group(g))
                        matches.put(one)
                    }
                    ok(JSONObject().put("matches", matches))
                }

                "sort_and_filter" -> {
                    val itemsArr = args.getJSONArray("items")
                    val predsArr = args.optJSONArray("predicates") ?: org.json.JSONArray()
                    val sortBy = args.optString("sort_by", null)
                    val order = args.optString("order", "asc")

                    fun passesPred(o: JSONObject, key: String, op: String, v: Any?): Boolean {
                        val lhs = if (o.has(key)) o.get(key) else return false
                        fun asDoubleOrNull(x: Any?): Double? = when (x) {
                            is Number -> x.toDouble()
                            is String -> x.toDoubleOrNull()
                            is Boolean -> if (x) 1.0 else 0.0
                            else -> null
                        }
                        return when (op) {
                            "==" -> lhs == v
                            "!=" -> lhs != v
                            "contains" -> lhs is String && v is String && lhs.contains(v, ignoreCase = true)
                            ">", ">=", "<", "<=" -> {
                                val a = asDoubleOrNull(lhs) ?: return false
                                val b = asDoubleOrNull(v) ?: return false
                                when (op) {
                                    ">"  -> a >  b
                                    ">=" -> a >= b
                                    "<"  -> a <  b
                                    "<=" -> a <= b
                                    else -> false
                                }
                            }
                            else -> false
                        }
                    }

                    val filtered = buildList {
                        for (i in 0 until itemsArr.length()) {
                            val obj = itemsArr.getJSONObject(i)
                            var okItem = true
                            for (j in 0 until predsArr.length()) {
                                val p = predsArr.getJSONObject(j)
                                if (!passesPred(obj, p.getString("key"), p.getString("op"), p.get("value"))) {
                                    okItem = false; break
                                }
                            }
                            if (okItem) add(JSONObject(obj.toString())) // copy
                        }
                    }

                    val sorted = if (sortBy != null) {
                        filtered.sortedWith { a, b ->
                            val av = if (a.has(sortBy)) a.get(sortBy) else null
                            val bv = if (b.has(sortBy)) b.get(sortBy) else null
                            val cmp = when {
                                av == null && bv == null -> 0
                                av == null -> -1
                                bv == null -> 1
                                av is Number && bv is Number -> av.toDouble().compareTo(bv.toDouble())
                                else -> av.toString().compareTo(bv.toString(), ignoreCase = true)
                            }
                            if (order == "desc") -cmp else cmp
                        }
                    } else filtered

                    val outArr = org.json.JSONArray()
                    sorted.forEach { outArr.put(it) }
                    ok(JSONObject().put("result", outArr))
                }

                "json_path" -> {
                    val jsonStr = args.getString("json")
                    val path = args.getJSONArray("path").let { arr -> (0 until arr.length()).map { arr.getString(it) } }
                    val any = org.json.JSONTokener(jsonStr).nextValue()

                    fun getAt(any: Any?, segs: List<String>): Any? {
                        if (segs.isEmpty()) return any
                        val head = segs.first()
                        val tail = segs.drop(1)
                        return when (any) {
                            is JSONObject -> if (any.has(head)) getAt(any.get(head), tail) else null
                            is org.json.JSONArray -> {
                                val idx = head.toIntOrNull() ?: return null
                                if (idx in 0 until any.length()) getAt(any.get(idx), tail) else null
                            }
                            else -> null
                        }
                    }

                    val value = getAt(any, path)
                    ok(JSONObject().put("value", value))
                }

                else -> err("Unknown tool: $name")
            }
        } catch (t: Throwable) {
            err(t.message ?: "tool error")
        }
    }

    companion object {
        // Two tools: device_info() and sum(a,b)
        val SAMPLE_TOOLS_JSON = """
[
  {"type":"function","function":{"name":"searchWeb","description":"This Tool Helps In WebSearch","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":"[query]"}}}
]
""".trimIndent()

    }
}