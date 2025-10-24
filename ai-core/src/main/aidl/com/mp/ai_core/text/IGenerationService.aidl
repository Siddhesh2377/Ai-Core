package com.mp.ai_core.text;

import com.mp.ai_core.text.IGenerationCallback;

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

    boolean loadMultimodalProjector(String mmprojPath, int threads);
    void unloadMultimodalProjector();
    boolean isMultimodalReady();
    String getMultimodalInfo();

    boolean generateWithImage(
        String prompt,
        in byte[] imageData,
        int imageWidth,
        int imageHeight,
        int maxTokens,
        String toolCallingJson,
        IGenerationCallback callback
    );
}