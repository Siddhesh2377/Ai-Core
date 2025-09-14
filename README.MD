# Ai‑Core

Native C/C++ core for **ToolNeuron** — JNI + `llama.cpp` bindings for fast, private, on‑device LLM inference on Android.

> Ai‑Core provides the low‑level runtime that ToolNeuron uses to run models fully offline. It wraps upstream `llama.cpp` behind a small C layer and exposes a stable JNI surface for Kotlin/Java apps.

---

## Relationship to ToolNeuron

Ai‑Core is a component of the main project **ToolNeuron**. For the app, plugin system, and full UX, see: [https://github.com/Siddhesh2377/ToolNeuron](https://github.com/Siddhesh2377/ToolNeuron)

---

## Highlights

* JNI bridge around the latest `llama.cpp` (kept in sync regularly).
* Minimal, portable C/C++ interface with Android‑first build.
* Fully offline execution; no network dependency.
* CMake‑driven native build; Gradle/NDK integration.

---

## Repository structure

The repository is organized as an Android multi‑module project with a native library and a sample app.

```
📦 Ai-Core
├─ ai-core          # Android Library module exposing JNI + native C/C++ core
│  ├─ src/main/cpp  # Native sources & CMake
│  │  ├─ CMakeLists.txt
│  │  ├─ ai_core.cpp
│  │  ├─ cpu_helper.cpp
│  │  └─ cpu_helper.h
│  └─ src/main/java/com/mp/ai_core/NativeLib.kt  # Kotlin JNI facade
└─ app              # Optional sample app module
```

**Native sources (current):** four key files power the core today — `ai_core.cpp`, `cpu_helper.cpp`, `cpu_helper.h`, and `CMakeLists.txt`.

> If you vendor `llama.cpp`, it typically lives under `ai-core/src/main/cpp/third_party/llama.cpp` (as a submodule) or is pulled via CMake. Exact placement may vary; update `CMakeLists.txt` accordingly.

---

## Requirements

* Android Studio (latest stable)
* Android NDK r26+ and CMake ≥ 3.22
* Android SDK Platform 23+
* JDK 17 recommended

---

## Build

### From the command line

```bash
# Build the native library (AAR) and sample app
./gradlew :ai-core:assembleRelease
./gradlew :app:installDebug
```

Artifacts:

* `ai-core` produces an AAR with `libai_core.so` for your configured ABIs.
* `app` is an optional sample to verify integration.

### Android Studio

Open the root `settings.gradle.kts` and sync. The project already includes `externalNativeBuild` with CMake under `ai-core/src/main/cpp`.

---

## Integrating Ai‑Core in your app

Add this repository as a submodule or include build, then depend on the library module.

**settings.gradle.kts**

```kotlin
include(":ai-core")
```

**app/build.gradle.kts**

```kotlin
dependencies {
    implementation(project(":ai-core"))
}
```

**Load the native library & call JNI**

```kotlin
object AiCoreNative {
    init { System.loadLibrary("ai_core") }
    // See NativeLib.kt for the exact exported functions.
}
```

> The reference Kotlin facade is in `ai-core/src/main/java/com/mp/ai_core/NativeLib.kt`. Keep signatures minimal and stable.

---

## Model formats

Ai‑Core targets `llama.cpp`‑compatible **GGUF** models. Ship models as app assets, dynamic downloads, or device‑local files. Ensure the selected quantization fits your device memory (e.g., Q4\_0 … Q8\_0). Model handling paths are provided from the Java/Kotlin side into JNI.

---

## Keeping `llama.cpp` up to date

Ai‑Core tracks upstream `llama.cpp`. When updating:

1. Fast‑forward your submodule or adjust the CMake fetch step.
2. Rebuild for all ABIs/targets you ship.
3. Run the sample app to smoke‑test model load and token generation.

Typical submodule refresh:

```bash
git submodule update --init --recursive
git submodule update --remote ai-core/src/main/cpp/third_party/llama.cpp
git add ai-core/src/main/cpp/third_party/llama.cpp
git commit -m "chore: sync llama.cpp to latest"
```

---

## Native layout and responsibilities

* **`ai_core.cpp`** — Public C API surface used by JNI; model lifecycle (load/free), inference loop, sampler controls.
* **`cpu_helper.*`** — Runtime CPU feature detection and thread suggestions (e.g., big.LITTLE hints).
* **`CMakeLists.txt`** — Flags, ABIs, STL, and inclusion of `llama.cpp` sources.
* **`NativeLib.kt`** — Thin Kotlin shim calling into the C API via `JNIEXPORT` functions.

---

## Usage sketch (conceptual)

Actual names may differ; see headers and `NativeLib.kt`.

**Kotlin**

```kotlin
object AiCore {
    init { System.loadLibrary("ai_core") }
    external fun load(modelPath: String, nThreads: Int, nCtx: Int): Long
    external fun infer(handle: Long, prompt: String, maxTokens: Int, temperature: Float): String
    external fun close(handle: Long)
}
```

**C entry points (simplified)**

```c
// ai_core.h (conceptual)
#ifdef __cplusplus
extern "C" {
#endif

typedef struct ai_core_handle ai_core_handle;
ai_core_handle* ai_core_load(const char* model_path, int n_threads, int n_ctx);
int ai_core_infer(ai_core_handle* h, const char* prompt, int max_tokens, float temperature, /*out*/ char* out_buf, int out_cap);
void ai_core_free(ai_core_handle* h);

#ifdef __cplusplus
}
#endif
```

---

## Performance notes

* Match `n_threads` to physical cores; test on target devices.
* Increase `n_ctx` cautiously to balance memory and coherence.
* Choose quantization to fit memory budgets; higher precision improves quality.
* When upstream supports it and you enable it in CMake, draft/speculative decoding can reduce latency for greedy sampling.

---

## Security & privacy

Inference is fully local. If your app uploads models or prompts anywhere, document that externally; Ai‑Core itself does not initiate network calls.

---

## License

Ai‑Core includes or links code from `llama.cpp` under its respective license. See repository LICENSE files for details.

---

## Links

* Main project (ToolNeuron): [https://github.com/Siddhesh2377/ToolNeuron](https://github.com/Siddhesh2377/ToolNeuron)
* Upstream engine (`llama.cpp`): [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
* This repository: [https://github.com/Siddhesh2377/Ai-Core](https://github.com/Siddhesh2377/Ai-Core)
