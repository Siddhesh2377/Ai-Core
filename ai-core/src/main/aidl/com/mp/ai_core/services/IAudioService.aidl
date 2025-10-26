package com.mp.ai_core.services;

import com.mp.ai_core.services.IAudioCallback;
import com.mp.ai_core.services.ISttCallback;

interface IAudioService {
    //TTS Functions
    boolean initializeTts(String modelDir, String modelName, String voices, String dataDir);
    void releaseTts();
    boolean isTtsReady();
    int getTtsSampleRate();
    int getTtsNumSpeakers();
    void generateTts(String text, int speakerId, IAudioCallback callback);
    void stopTts();

    //STT Functions
    boolean initializeStt(String modelDir, int modelType, int numThreads);
    void releaseStt();
    boolean isSttReady();
    void transcribeFile(String filePath, int sampleRate, ISttCallback callback);
    void transcribeSamples(in float[] samples, int sampleRate, ISttCallback callback);
    int getActiveStreamCount();
    int getCurrentModelType();

    //Misc Functions
    String getAudioInfo();
}