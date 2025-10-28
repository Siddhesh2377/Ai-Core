package com.mp.ai_core.services;

interface IAudioCallback {
    void onAudioChunk(in float[] samples);
    void onComplete();
    void onError(String error);
}