package com.mp.ai_core

import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import java.io.File

class MMNativeLibUnitTest {

    private lateinit var mmLib: MMNativeLib

    @Before
    fun setup() {
        mmLib = MMNativeLib()
    }

    @Test
    fun testMMGenerate_withImage() {
        val modelPath = "/home/home/Downloads/LFM2-VL-450M-Q4_0.gguf"
        val mmprojPath = "/home/home/Downloads/mmproj-LFM2-VL-450M-Q8_0.gguf"
        val imagePath = "/home/home/Downloads/pexels-dantemunozphoto-28821755.jpg"

        // sanity checks
        assertTrue("❌ Model not found at $modelPath", File(modelPath).exists())
        assertTrue("❌ MMProj not found at $mmprojPath", File(mmprojPath).exists())
        assertTrue("❌ Image not found at $imagePath", File(imagePath).exists())

        val initOk = mmLib.nativeMMInit(modelPath, mmprojPath, Runtime.getRuntime().availableProcessors())
        assertTrue("❌ Init failed", initOk)

        val prompt = "Describe this image briefly."
        val result = mmLib.nativeMMGenerate(prompt, imagePath, 64)

        println("🧠 Output: $result")

        assertNotNull("❌ Output is null", result)
        assertTrue("❌ Output is empty", result.isNotEmpty())
    }

    @Test
    fun testMMGenerate_textOnly() {
        val modelPath = "/home/home/Downloads/LFM2-VL-450M-Q4_0.gguf"
        val mmprojPath = "/home/home/Downloads/mmproj-LFM2-VL-450M-Q8_0.gguf"

        assertTrue("❌ Model not found", File(modelPath).exists())
        assertTrue("❌ MMProj not found", File(mmprojPath).exists())

        val initOk = mmLib.nativeMMInit(modelPath, mmprojPath, 4)
        assertTrue("❌ Init failed", initOk)

        val result = mmLib.nativeMMGenerate("Write a short haiku about mountains", "", 50)

        println("🧠 Text-only Output: $result")

        assertNotNull("❌ Output is null", result)
        assertTrue("❌ Response is empty", result.isNotEmpty())
    }
}
