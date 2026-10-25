#include "render_thread.h"

#include <android/log.h>
#include <sys/prctl.h>

#include <chrono>
#include <cstdio>

#define LOG_TAG "BistroThread"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

void RenderThread::start(AAssetManager* assets) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (running_) return;
    assets_ = assets;
    quit_ = false;
    running_ = true;
    thread_ = std::thread(&RenderThread::run, this);
}

void RenderThread::setWindow(ANativeWindow* window) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!running_) {
        if (window) ANativeWindow_release(window);
        return;
    }
    if (windowPending_ && pendingWindow_) ANativeWindow_release(pendingWindow_);
    pendingWindow_ = window;
    windowPending_ = true;
    cv_.notify_all();
    cv_.wait(lock, [&] { return !windowPending_ || quit_; });
}

void RenderThread::setPaused(bool paused) {
    std::lock_guard<std::mutex> lock(mutex_);
    paused_ = paused;
    cv_.notify_all();
}

void RenderThread::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_) return;
        quit_ = true;
        cv_.notify_all();
    }
    if (thread_.joinable()) thread_.join();
    std::lock_guard<std::mutex> lock(mutex_);
    running_ = false;
    if (windowPending_ && pendingWindow_) ANativeWindow_release(pendingWindow_);
    pendingWindow_ = nullptr;
    windowPending_ = false;
}

std::string RenderThread::status() {
    std::lock_guard<std::mutex> lock(mutex_);
    return status_;
}

bool RenderThread::isFlybyCompleted() {
    std::lock_guard<std::mutex> lock(mutex_);
    return flybyCompleted_;
}

void RenderThread::setEnvironment(const std::string& assetDir, bool startFlyby) {
    std::lock_guard<std::mutex> lock(mutex_);
    pendingAssetDir_ = assetDir;
    pendingStartFlyby_ = startFlyby;
    envPending_ = true;
    flybyCompleted_ = false;
    cv_.notify_all();
}

void RenderThread::setDebugOptions(double startTime, bool freeze, const CameraOverride& override) {
    std::lock_guard<std::mutex> lock(mutex_);
    startTime_ = startTime;
    freeze_ = freeze;
    override_ = override;
    flybyCompleted_ = false;
    debugDirty_ = true;
    cv_.notify_all();
}

void RenderThread::run() {
    prctl(PR_SET_NAME, "BistroRender", 0, 0, 0);
    Renderer renderer;
    bool ok = renderer.init(assets_);
    if (!ok) {
        std::lock_guard<std::mutex> lock(mutex_);
        status_ = "Vulkan initialisation failed (Vulkan 1.3 with ASTC required)";
    }
    ANativeWindow* window = nullptr;
    using Clock = std::chrono::steady_clock;
    auto last = Clock::now();
    double wallTime = 0.0;
    double animTime = 0.0;
    bool freeze = false;
    bool flybyRunning = false;

    while (true) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            for (;;) {
                if (quit_) break;
                if (windowPending_) {
                    if (window) {
                        renderer.detachWindow();
                        ANativeWindow_release(window);
                    }
                    window = pendingWindow_;
                    pendingWindow_ = nullptr;
                    windowPending_ = false;
                    if (window && ok && !renderer.attachWindow(window)) {
                        status_ = "Failed to create swapchain";
                    }
                    cv_.notify_all();
                }
                if (envPending_) {
                    std::string dir = pendingAssetDir_;
                    bool startFlyby = pendingStartFlyby_;
                    envPending_ = false;
                    lock.unlock();
                    if (ok) renderer.setEnvironment(dir);
                    lock.lock();
                    animTime = 0.0;
                    flybyRunning = startFlyby;
                    flybyCompleted_ = false;
                    last = Clock::now();
                }
                if (debugDirty_) {
                    animTime = startTime_;
                    freeze = freeze_;
                    flybyRunning = true;
                    flybyCompleted_ = false;
                    renderer.setCameraOverride(override_);
                    debugDirty_ = false;
                }
                if (ok && window && !paused_) break;
                cv_.wait(lock);
                last = Clock::now();
            }
            if (quit_) break;
        }

        auto now = Clock::now();
        double dt = std::chrono::duration<double>(now - last).count();
        last = now;
        dt = std::min(dt, 0.1);
        wallTime += dt;
        if (renderer.isLoaded() && flybyRunning && !freeze) {
            animTime += dt;
            float totalLoop = renderer.loopDuration();
            if (!override_.enabled && totalLoop > 0.0f && animTime >= static_cast<double>(totalLoop)) {
                flybyRunning = false;
                std::lock_guard<std::mutex> lock(mutex_);
                flybyCompleted_ = true;
            }
        }

        renderer.drawFrame(renderer.isLoaded() ? animTime : wallTime, dt);

        char buf[256];
        if (!renderer.hasEnvironment()) {
            snprintf(buf, sizeof(buf), "Select an environment to begin");
        } else if (renderer.loadFailed()) {
            snprintf(buf, sizeof(buf), "Error: %s", renderer.loadStatus().c_str());
        } else if (!renderer.isLoaded()) {
            snprintf(buf, sizeof(buf), "%s… %d%%", renderer.loadStatus().c_str(),
                     static_cast<int>(renderer.loadProgress() * 100.0f));
        } else {
            const RenderStats& s = renderer.stats();
            const char* tour = s.cameraSegmentName.empty() ? "Fly-Through" : s.cameraSegmentName.c_str();
            snprintf(buf, sizeof(buf),
                     "%s · %.0f fps · GPU %.1f ms · %u×%u %s\n%u/%u clusters · %.2fM of %.1fM triangles drawn",
                     tour, s.fps, s.gpuMs, s.renderWidth, s.renderHeight, s.taa ? "TAA" : "4×MSAA",
                     s.visibleClusters[0] + s.visibleClusters[1] + s.visibleClusters[2], s.totalClusters,
                     static_cast<double>(s.visibleTriangles) / 1e6, static_cast<double>(s.totalTriangles) / 1e6);
            static int logCounter = 0;
            if (++logCounter % 120 == 0) {
                LOGI("stats: %.1f fps, GPU %.2f ms, clusters %u/%u/%u of %u, tris %.2fM, exposure %.3f, cam (%.1f %.1f %.1f) t=%.1f",
                     s.fps, s.gpuMs, s.visibleClusters[0], s.visibleClusters[1], s.visibleClusters[2],
                     s.totalClusters, s.visibleTriangles / 1e6, s.exposure, s.cameraPos.x, s.cameraPos.y, s.cameraPos.z, animTime);
            }
        }
        std::lock_guard<std::mutex> lock(mutex_);
        status_ = buf;
    }

    renderer.destroy();
    if (window) ANativeWindow_release(window);
    std::lock_guard<std::mutex> lock(mutex_);
    cv_.notify_all();
    LOGI("Render thread exiting");
}
