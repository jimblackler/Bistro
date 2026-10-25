#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <jni.h>

#include <memory>
#include <string>

#include "render_thread.h"

namespace {
std::unique_ptr<RenderThread> gThread;
jobject gAssetManagerRef = nullptr;
}  // namespace

extern "C" {

JNIEXPORT void JNICALL Java_com_android_demo_bistro_MainActivity_nativeCreate(JNIEnv* env, jobject,
                                                                             jobject assetManager) {
    if (gThread) return;
    gAssetManagerRef = env->NewGlobalRef(assetManager);
    gThread = std::make_unique<RenderThread>();
    gThread->start(AAssetManager_fromJava(env, gAssetManagerRef));
}

JNIEXPORT void JNICALL Java_com_android_demo_bistro_MainActivity_nativeSetSurface(JNIEnv* env, jobject,
                                                                                 jobject surface) {
    if (!gThread) return;
    ANativeWindow* window = surface ? ANativeWindow_fromSurface(env, surface) : nullptr;
    gThread->setWindow(window);
}

JNIEXPORT void JNICALL Java_com_android_demo_bistro_MainActivity_nativeSetPaused(JNIEnv*, jobject,
                                                                                jboolean paused) {
    if (gThread) gThread->setPaused(paused == JNI_TRUE);
}

JNIEXPORT void JNICALL Java_com_android_demo_bistro_MainActivity_nativeDestroy(JNIEnv* env, jobject) {
    if (gThread) {
        gThread->stop();
        gThread.reset();
    }
    if (gAssetManagerRef) {
        env->DeleteGlobalRef(gAssetManagerRef);
        gAssetManagerRef = nullptr;
    }
}

JNIEXPORT jstring JNICALL Java_com_android_demo_bistro_MainActivity_nativeGetStatus(JNIEnv* env, jobject) {
    std::string s = gThread ? gThread->status() : std::string();
    return env->NewStringUTF(s.c_str());
}

JNIEXPORT jboolean JNICALL Java_com_android_demo_bistro_MainActivity_nativeIsFlybyCompleted(JNIEnv*, jobject) {
    return (gThread && gThread->isFlybyCompleted()) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL Java_com_android_demo_bistro_MainActivity_nativeSetEnvironment(
        JNIEnv* env, jobject, jstring assetDir, jboolean startFlyby) {
    if (!gThread || !assetDir) return;
    const char* chars = env->GetStringUTFChars(assetDir, nullptr);
    std::string dir(chars ? chars : "");
    if (chars) env->ReleaseStringUTFChars(assetDir, chars);
    gThread->setEnvironment(dir, startFlyby == JNI_TRUE);
}

JNIEXPORT void JNICALL Java_com_android_demo_bistro_MainActivity_nativeSetDebugOptions(
        JNIEnv* env, jobject, jfloat startTime, jboolean freeze, jfloatArray camera) {
    if (!gThread) return;
    CameraOverride o;
    if (camera && env->GetArrayLength(camera) >= 6) {
        jfloat v[6];
        env->GetFloatArrayRegion(camera, 0, 6, v);
        o.enabled = true;
        o.eye = Vec3{v[0], v[1], v[2]};
        o.target = Vec3{v[3], v[4], v[5]};
    }
    gThread->setDebugOptions(startTime, freeze == JNI_TRUE, o);
}

}  // extern "C"
