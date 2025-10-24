# ai_core_embed.cpp – Embedding Engine

Specialised JNI functions for computing *sentence embeddings*:
- `nativeInitForEmbeddings` – initialise with `embeddings=true`.
- `embed` – return a `float[]` of size `n_embd`.
- State persistence (load / save KV‑cache) identical to `ai_core.cpp`.

Designed for **blocking** single‑sentence encoding; all Kotlin callers wrap in `Dispatchers.IO`.