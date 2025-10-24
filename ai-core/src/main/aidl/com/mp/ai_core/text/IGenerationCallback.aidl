package com.mp.ai_core.text;

interface IGenerationCallback {
    void onToken(in String token);
    void onToolCall(in String name, in String argsJson);
    void onDone();
    void onError(in String error);
}