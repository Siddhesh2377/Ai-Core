package com.mp.ai_core.services;

import com.mp.ai_core.services.IGenerationCallback;

interface IGenerationService {
    /* ---------- Model Management ---------- */
    boolean loadModel(
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

    void unloadModel();

    /* ---------- Generation ---------- */
    boolean generate(
        String prompt,
        int maxTokens,
        String toolCallingJSON,
        IGenerationCallback callback
    );

    void stopGeneration();
    boolean isGenerating();

    /* ---------- Configuration ---------- */
    void setSystemPrompt(String prompt);
    void setChatTemplate(String template);
    void setToolsJson(String toolsJson);

    /* ---------- State Management ---------- */
    long getStateSize();
    byte[] getStateData();
    boolean loadStateData(in byte[] state);
    boolean saveStateFile(String filePath);
    boolean loadStateFile(String filePath);

    /* ---------- Embedding ---------- */
    float[] embed(String text);

    /* ---------- Metadata ---------- */
    String getModelInfo();
}