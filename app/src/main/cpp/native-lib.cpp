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

JNIEXPORT void JNICALL Java_com_android_demo_bistro_MainActivity_nativeSetGraphicsSettings(
        JNIEnv*, jobject, jfloat renderScale, jint shadowSize, jfloat minClusterPixels, jfloat bloomScale,
        jfloat aoIntensityScale, jboolean ssrEnabled, jboolean contactShadowsEnabled, jboolean volumetricFogEnabled,
        jboolean aoEnabled, jboolean shadowPcfEnabled, jboolean taaEnabled, jboolean occlusionCulling) {
    if (!gThread) return;
    GraphicsSettings s;
    s.renderScale = renderScale;
    s.shadowSize = static_cast<uint32_t>(shadowSize);
    s.minClusterPixels = minClusterPixels;
    s.bloomScale = bloomScale;
    s.aoIntensityScale = aoIntensityScale;
    s.ssrEnabled = (ssrEnabled == JNI_TRUE);
    s.contactShadowsEnabled = (contactShadowsEnabled == JNI_TRUE);
    s.volumetricFogEnabled = (volumetricFogEnabled == JNI_TRUE);
    s.aoEnabled = (aoEnabled == JNI_TRUE);
    s.shadowPcfEnabled = (shadowPcfEnabled == JNI_TRUE);
    s.taaEnabled = (taaEnabled == JNI_TRUE);
    s.occlusionCulling = (occlusionCulling == JNI_TRUE);
    gThread->setGraphicsSettings(s);
}

JNIEXPORT void JNICALL Java_com_android_demo_bistro_MainActivity_nativeSetCameraFrozen(
        JNIEnv*, jobject, jboolean frozen) {
    if (gThread) gThread->setCameraFrozen(frozen == JNI_TRUE);
}

}  // extern "C"
