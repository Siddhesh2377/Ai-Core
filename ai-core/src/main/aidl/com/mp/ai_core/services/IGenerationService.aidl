package com.mp.ai_core.services;

import com.mp.ai_core.services.IGenerationCallback;

/**
 *  All generation / embed / meta calls that the UI needs.
 */
interface IGenerationService {
    /* ---------- ordinary generation ---------- */
    boolean   loadModel(
        String path,
        int threads,
        int gpuLayers,
        boolean useMMap,
        int ctxSize,
        float temp,
        int topK,
        float topP,
        float minP
    );

    boolean generate(
        String prompt,
        int maxTokens,
        String toolCallingJSON,
        IGenerationCallback callback
    );

    void    stopGeneration();

    void    unloadModel();

    /* ---------- configuration ---------- */
    void    setSystemPrompt(String prompt);
    void    setChatTemplate(String template);
    void    setToolsJson(String toolsJson);  // helper to enable/disable tool calling

    /* ---------- embedding ---------- */
    float[] embed(String text);            // synchronous – blocking but small payload
    // for larger vocab you can replace the above with:
    // void embed(String text, IEmbeddingCallback cb);

    /* ---------- metadata ---------- */
    String  getModelInfo();    // JSON meta string
}