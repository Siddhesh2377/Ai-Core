package com.mp.ai_core.text;

import com.mp.ai_core.text.IGenerationCallback;

interface IGenerationService {
    //Common
    boolean isGenerating();
    String getModelInfo();

    //Text Generation Model
    boolean loadTextGenerationModel(
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
    void unloadTextGenerationModel();
    boolean generateText(
        String prompt,
        int maxTokens,
        String toolCallingJSON,
        IGenerationCallback callback
    );
    void stopTextGeneration();
    void setSystemPrompt(String prompt);
    void setChatTemplate(String template);
    void setToolsJson(String toolsJson);

    //State-Managment
    long getStateSize();
    byte[] getStateData();
    boolean loadStateData(in byte[] state);
    boolean saveStateFile(String filePath);
    boolean loadStateFile(String filePath);

    //Embedding Model
    boolean loadEmbedModel(String path, int threads, int ctxSize);
    float[] embed(String text);
    void unLoadEmbeddingModel();

    //Multi-Model ( VLM )
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