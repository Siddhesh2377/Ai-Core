# ai_core_mtmd.cpp – Multimodal (Vision) Engine

Adds image decoding + tokenisation for *vision* models:
- `nativeInitMTMD` – initialise MX decoder (`mmproj.bin`).
- `nativeGenerateStreamWithImage` – stream output for image + text.
- `nativeLoadImageFromFile` – helper for debug tests.

Uses **mtmd** library, which is linked by `CMakeLists.txt`.  
Requires the base model (same as `ai_core.cpp`) to be loaded first.