package com.mp.ai_core

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class MMNativeLibTest {

    private val mmLib = MMNativeLib()

    @Test
    fun testMMGenerate_withImage() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val modelPath = "/home/home/Downloads/LFM2-VL-450M-Q4_0.gguf"
        val mmprojPath = "/home/home/Downloads/mmproj-LFM2-VL-450M-Q8_0.gguf"
        val imagePath = "/home/home/Downloads/pexels-dantemunozphoto-28821755.jpg"

        // Verify files exist before proceeding
        assertTrue("Model not found at $modelPath", File(modelPath).exists())
        assertTrue("MMProj not found at $mmprojPath", File(mmprojPath).exists())
        assertTrue("Image not found at $imagePath", File(imagePath).exists())

        val initOk = mmLib.nativeMMInit(modelPath, mmprojPath, Runtime.getRuntime().availableProcessors())
        assertTrue("Init failed", initOk)

        val prompt = "Describe this image briefly."
        val result = mmLib.nativeMMGenerate(prompt, imagePath, 64)

        Log.d("MMNativeLibTest", "🧠 Output: $result")

        assertNotNull(result)
        assertTrue("Empty response", result.isNotEmpty())
    }

    @Test
    fun testMMGenerate_textOnly() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val modelPath = "/sdcard/AI/models/lfm2-q4.gguf"
        val mmprojPath = "/sdcard/AI/models/mmproj-f16.gguf"

        val initOk = mmLib.nativeMMInit(modelPath, mmprojPath, 4)
        assertTrue("Init failed", initOk)

        val result = mmLib.nativeMMGenerate("Write a short haiku about mountains", "", 50)
        Log.d("MMNativeLibTest", "🧠 Text-only Output: $result")

        assertNotNull(result)
        assertTrue("Response is empty", result.isNotEmpty())
    }
}
