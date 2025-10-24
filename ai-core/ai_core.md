# ai_core.cpp – Native Text‑Generation API

Provides the **Core** JNI functions to:
- Load / release model (`nativeInit`, `nativeRelease`)
- Dynamic configuration (system prompt, template, tools)
- Streaming generation (`nativeGenerateStream`)
- KV‑cache handling (`nativeClearMemory`, state persistence)
- Diagnostics (`nativeGetModelInfo`, `llamaPrintTimings`)

All calls are **CPU‑only** (`n_gpu_layers = 0`).  
The file contains:
- `global_state` – singleton the whole library shares.
- `chat_template.cpp` – chat protocol / prompt rendering.
- `tool_call_state.cpp` – tool‑call parsing/accumulation.  
  Usage:  Call via `NativeLib` Kotlin wrapper.