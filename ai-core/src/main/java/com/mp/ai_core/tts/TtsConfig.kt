// TtsConfig.kt
package com.mp.ai_core.tts

import com.google.gson.Gson
import com.google.gson.annotations.SerializedName


/**
 * A tiny POJO that mirrors every mutable member of [TtsEngine].
 * We use Gson to (de)serialise it – the client can construct it in Java/Kotlin
 * or send a JSON string manually.
 */
data class TtsConfig(
    @SerializedName("modelDir")          var modelDir: String?          = null,
    @SerializedName("modelName")         var modelName: String?         = null,
    @SerializedName("acousticModelName") var acousticModelName: String? = null,
    @SerializedName("vocoder")           var vocoder: String?           = null,
    @SerializedName("voices")            var voices: String?            = null,
    @SerializedName("ruleFsts")          var ruleFsts: String?          = null,
    @SerializedName("ruleFars")          var ruleFars: String?          = null,
    @SerializedName("lexicon")           var lexicon: String?           = null,
    @SerializedName("dataDir")           var dataDir: String?           = null,
    @SerializedName("lang")              var lang: String?              = null,
    @SerializedName("lang2")             var lang2: String?             = null,
    @SerializedName("isKitten")          var isKitten: Boolean?         = null
)


fun TtsConfig.copyToJson(cfg: TtsConfig): String{
   return Gson().toJson(cfg)
}