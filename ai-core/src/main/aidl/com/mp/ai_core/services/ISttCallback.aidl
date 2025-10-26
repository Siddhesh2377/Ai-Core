package com.mp.ai_core.services;

interface ISttCallback {
    void onResult(String text);
    void onError(String error);
}
