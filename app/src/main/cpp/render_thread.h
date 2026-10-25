#pragma once

#include <android/asset_manager.h>
#include <android/native_window.h>

#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

#include "renderer.h"

// Dedicated render thread: owns the Renderer, streams assets and renders frames at display rate
// (paced by FIFO presentation) independently of the Android UI thread.
class RenderThread {
public:
    void start(AAssetManager* assets);
    // Hands a new window (or nullptr) to the render thread and blocks until it has been adopted, so that
    // the caller can safely let the Surface be destroyed afterwards.
    void setWindow(ANativeWindow* window);
    void setEnvironment(const std::string& assetDir, bool startFlyby);
    void setPaused(bool paused);
    void stop();
    std::string status();
    bool isFlybyCompleted();
    void setDebugOptions(double startTime, bool freeze, const CameraOverride& override);

private:
    void run();

    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    AAssetManager* assets_ = nullptr;
    ANativeWindow* pendingWindow_ = nullptr;
    bool windowPending_ = false;
    std::string pendingAssetDir_;
    bool pendingStartFlyby_ = false;
    bool envPending_ = false;
    bool flybyCompleted_ = false;
    bool paused_ = false;
    bool quit_ = false;
    bool running_ = false;
    std::string status_ = "Starting…";

    double startTime_ = 0.0;
    bool freeze_ = false;
    CameraOverride override_;
    bool debugDirty_ = false;
};
