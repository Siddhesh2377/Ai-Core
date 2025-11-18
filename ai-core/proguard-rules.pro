# Keep all classes in ai_core package
-keep class com.mp.ai_core.* { *; }
-keep class com.mp.ai_core.** { *; }

# Keep k2fsa sherpa onnx package (THIS IS THE CORRECT PACKAGE!)
-keep class com.k2fsa.sherpa.onnx.* { *; }
-keep class com.k2fsa.sherpa.onnx.** { *; }

# Keep all public methods, fields, and constructors
-keepclassmembers class com.k2fsa.sherpa.onnx.** {
    public *;
    private *;
}

# Specifically keep the classes you mentioned
-keep class com.k2fsa.sherpa.onnx.FeatureConfig { *; }
-keep class com.k2fsa.sherpa.onnx.HomophoneReplacerConfig { *; }
-keep class com.k2fsa.sherpa.onnx.OfflineRecognizer { *; }
-keep class com.k2fsa.sherpa.onnx.OfflineStream { *; }
-keep class com.k2fsa.sherpa.onnx.TtsKt { *; }
-keep class com.k2fsa.sherpa.onnx.WaveReader { *; }

# Keep Kotlin data classes (important for data classes)
-keepclassmembers class com.k2fsa.sherpa.onnx.** {
    <init>(...);
    *** component*();
    *** copy(...);
}

# Keep Kotlin top-level functions
-keep class com.k2fsa.sherpa.onnx.**Kt { *; }

# If using JNI (native methods)
-keepclasseswithmembernames class com.k2fsa.sherpa.onnx.** {
    native <methods>;
}