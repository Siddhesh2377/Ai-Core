package com.mp.ai_core.tts

import android.content.Context
import android.content.res.AssetManager
import android.util.Log
import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.getOfflineTtsConfig
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlin.collections.iterator
import com.google.gson.Gson
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch

object TtsEngine {
    var tts: OfflineTts? = null
    const val TAG = "TtsEngine"
    private var stopped: Boolean = false
    // https://en.wikipedia.org/wiki/ISO_639-3
    // Example:
    // eng for English,
    // deu for German
    // cmn for Mandarin
    var lang: String? = null

    // if a model supports two languages, set also lang2
    var lang2: String? = null
    var samplesChannel = Channel<FloatArray>(Channel.CONFLATED)
    private var modelDir: String? = null
    private var modelName: String? = null
    private var acousticModelName: String? = null // for matcha tts
    private var vocoder: String? = null // for matcha tts
    private var voices: String? = null // for kokoro
    private var ruleFsts: String? = null
    private var ruleFars: String? = null
    private var lexicon: String? = null
    private var dataDir: String? = null
    private var assets: AssetManager? = null
    private var isKitten = false

    var scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    init {
        // The purpose of such a design is to make the CI test easier
        // Please see
        // https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/apk/generate-tts-apk-script.py
        //
        // For VITS -- begin
        modelName = null
        // For VITS -- end

        // For Matcha -- begin
        acousticModelName = null
        vocoder = null
        // For Matcha -- end

        // For Kokoro -- begin
        voices = null
        // For Kokoro -- end

        modelDir = null
        ruleFsts = null
        ruleFars = null
        lexicon = null
        dataDir = null
        lang = null
        lang2 = null

        // Please enable one and only one of the examples below

        // Example 1:
        // https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-vctk.tar.bz2
        // modelDir = "vits-vctk"
        // modelName = "vits-vctk.onnx"
        // lexicon = "lexicon.txt"
        // lang = "eng"

        // Example 2:
        // https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
        // https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
        // modelDir = "vits-piper-en_US-amy-low"
        // modelName = "en_US-amy-low.onnx"
        // dataDir = "vits-piper-en_US-amy-low/espeak-ng-data"
        // lang = "eng"

        // Example 3:
        // https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-icefall-zh-aishell3.tar.bz2
        // modelDir = "vits-icefall-zh-aishell3"
        // modelName = "model.onnx"
        // ruleFars = "vits-icefall-zh-aishell3/rule.far"
        // lexicon = "lexicon.txt"
        // lang = "zho"

        // Example 4:
        // https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/vits.html#csukuangfj-vits-zh-hf-fanchen-c-chinese-187-speakers
        // modelDir = "vits-zh-hf-fanchen-C"
        // modelName = "vits-zh-hf-fanchen-C.onnx"
        // lexicon = "lexicon.txt"
        // lang = "zho"

        // Example 5:
        // https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-coqui-de-css10.tar.bz2
        // This model does not need lexicon or dataDir
        // modelDir = "vits-coqui-de-css10"
        // modelName = "model.onnx"
        // lang = "deu"

        // Example 6
        // vits-melo-tts-zh_en
        // https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/vits.html#vits-melo-tts-zh-en-chinese-english-1-speaker
        // modelDir = "vits-melo-tts-zh_en"
        // modelName = "model.onnx"
        // lexicon = "lexicon.txt"
        // lang = "zho"
        // lang2 = "eng"

        // Example 7
        // matcha-icefall-zh-baker
        // https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/matcha.html#matcha-icefall-zh-baker-chinese-1-female-speaker
        // modelDir = "matcha-icefall-zh-baker"
        // acousticModelName = "model-steps-3.onnx"
        // vocoder = "vocos-22khz-univ.onnx"
        // lexicon = "lexicon.txt"
        // lang = "zho"

        // Example 8
        // matcha-icefall-en_US-ljspeech
        // https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/matcha.html#matcha-icefall-en-us-ljspeech-american-english-1-female-speaker
        // modelDir = "matcha-icefall-en_US-ljspeech"
        // acousticModelName = "model-steps-3.onnx"
        // vocoder = "vocos-22khz-univ.onnx"
        // dataDir = "matcha-icefall-en_US-ljspeech/espeak-ng-data"
        // lang = "eng"

        // Example 9
         //kokoro-en-v0_19
         modelDir = "kokoro-en-v0_19"
         modelName = "model.onnx"
         voices = "voices.bin"
         dataDir = "kokoro-en-v0_19/espeak-ng-data"
         lang = "eng"

        // Example 10
        // kokoro-multi-lang-v1_0
        // modelDir = "kokoro-multi-lang-v1_0"
        // modelName = "model.onnx"
        // voices = "voices.bin"
        // dataDir = "kokoro-multi-lang-v1_0/espeak-ng-data"
        // lexicon = "kokoro-multi-lang-v1_0/lexicon-us-en.txt,kokoro-multi-lang-v1_0/lexicon-zh.txt"
        // lang = "eng"
        // lang2 = "zho"
        // ruleFsts = "$modelDir/phone-zh.fst,$modelDir/date-zh.fst,$modelDir/number-zh.fst"
        //
        // This model supports many languages, e.g., English, Chinese, etc.
        // We set lang to eng here.

        // Example 11
        // kitten-nano-en-v0_1-fp16
//         modelDir = "kitten-nano-en-v0_1-fp16"
//         modelName = "model.fp16.onnx"
//         voices = "voices.bin"
//         dataDir = "kitten-nano-en-v0_1-fp16/espeak-ng-data"
//         lang = "eng"
//         isKitten = true
    }

    fun createTts(context: Context) {
        Log.i(TAG, "Init Next-gen Kaldi TTS")
        if (tts == null) {
            initTts(context)
        }
    }

    fun generateAudio(text: String) {
        tts!!.generateWithCallback(
            text = text,
            callback = ::callback,
        )
    }

    private fun callback(samples: FloatArray, process: Float): Int {
        if (!stopped) {
            val samplesCopy = samples.copyOf()
            scope.launch {
                samplesChannel.send(samplesCopy)
            }
            Log.i(TAG, " return 1 :: $samplesCopy")
            return 1
        } else {
            Log.i(TAG, " return 0")
            return 0
        }
    }

    fun stop() {
        tts?.release()
        scope.cancel()
    }

    private fun initTts(context: Context) {
        assets = context.assets

        if (dataDir != null) {
            val newDir = copyDataDir(context, dataDir!!)
            dataDir = "$newDir/$dataDir"
        }

        val config = getOfflineTtsConfig(
            modelDir = modelDir!!,
            modelName = modelName ?: "",
            acousticModelName = acousticModelName ?: "",
            vocoder = vocoder ?: "",
            voices = voices ?: "",
            lexicon = lexicon ?: "",
            dataDir = dataDir ?: "",
            dictDir = "",
            ruleFsts = ruleFsts ?: "",
            ruleFars = ruleFars ?: "",
            isKitten = isKitten,
        )

        tts = OfflineTts(assetManager = assets, config = config)
    }



    private fun copyDataDir(context: Context, dataDir: String): String {
        Log.i(TAG, "data dir is $dataDir")
        copyAssets(context, dataDir)

        val newDataDir = context.getExternalFilesDir(null)!!.absolutePath
        Log.i(TAG, "newDataDir: $newDataDir")
        return newDataDir
    }

    private fun copyAssets(context: Context, path: String) {
        val assets: Array<String>?
        try {
            assets = context.assets.list(path)
            if (assets!!.isEmpty()) {
                copyFile(context, path)
            } else {
                val fullPath = "${context.getExternalFilesDir(null)}/$path"
                val dir = File(fullPath)
                dir.mkdirs()
                for (asset in assets.iterator()) {
                    val p: String = if (path == "") "" else "$path/"
                    copyAssets(context, p + asset)
                }
            }
        } catch (ex: IOException) {
            Log.e(TAG, "Failed to copy $path. $ex")
        }
    }

    private fun copyFile(context: Context, filename: String) {
        try {
            val istream = context.assets.open(filename)
            val newFilename = context.getExternalFilesDir(null).toString() + "/" + filename
            val ostream = FileOutputStream(newFilename)
            // Log.i(TAG, "Copying $filename to $newFilename")
            val buffer = ByteArray(1024)
            var read = 0
            while (read != -1) {
                ostream.write(buffer, 0, read)
                read = istream.read(buffer)
            }
            istream.close()
            ostream.flush()
            ostream.close()
        } catch (ex: Exception) {
            Log.e(TAG, "Failed to copy $filename, $ex")
        }
    }

    private fun applyConfig(config: TtsConfig) {
        // overwrite any field that the client supplied
        if (config.modelDir != null)          modelDir = config.modelDir
        if (config.modelName != null)         modelName = config.modelName
        if (config.acousticModelName != null) acousticModelName = config.acousticModelName
        if (config.vocoder != null)           vocoder = config.vocoder
        if (config.voices != null)            voices = config.voices
        if (config.ruleFsts != null)          ruleFsts = config.ruleFsts
        if (config.ruleFars != null)          ruleFars = config.ruleFars
        if (config.lexicon != null)           lexicon = config.lexicon
        if (config.dataDir != null)           dataDir = config.dataDir
        if (config.lang != null)              lang = config.lang
        if (config.lang2 != null)             lang2 = config.lang2
        if (config.isKitten != null)          isKitten = config.isKitten == true
    }

    fun loadFromJson(context: Context, json: String?) {
        if (!json.isNullOrBlank()) {
            // Using Gson – you can replace it with any other parser
            val gson = Gson()
            val cfg = gson.fromJson(json, TtsConfig::class.java)
            applyConfig(cfg)
        }
        createTts(context)   // will re‑initialise with the updated config
    }
}
