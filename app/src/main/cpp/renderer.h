#pragma once

#include <android/asset_manager.h>
#include <android/native_window.h>
#include <vulkan/vulkan.h>

#include <array>
#include <string>
#include <vector>

#include "camera_path.h"
#include "math_util.h"
#include "scene.h"
#include "vk_context.h"

// Optional camera override used for debugging and authoring the fly-through path.
struct CameraOverride {
    bool enabled = false;
    Vec3 eye;
    Vec3 target;
};

struct RenderStats {
    float fps = 0.0f;
    float gpuMs = 0.0f;
    uint32_t visibleClusters[3] = {0, 0, 0};
    uint32_t visibleTriangles = 0;
    bool taa = false;
    uint32_t totalClusters = 0;
    uint64_t totalTriangles = 0;
    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
    float exposure = 1.0f;
    Vec3 cameraPos;
    int cameraSegment = 0;
    std::string cameraSegmentName;
};

// Owns all Vulkan state and renders the active environment fly-through. Must only be used from the render thread.
class Renderer {
public:
    bool init(AAssetManager* assets);
    void setEnvironment(const std::string& assetDir);
    bool attachWindow(ANativeWindow* window);
    void detachWindow();
    // Renders one frame (the loading screen until all assets are resident).
    void drawFrame(double timeSeconds, double dtSeconds);
    void destroy();

    [[nodiscard]] bool hasWindow() const { return swapchain_ != VK_NULL_HANDLE; }
    [[nodiscard]] bool hasEnvironment() const { return !currentAssetDir_.empty(); }
    [[nodiscard]] const std::string& currentAssetDir() const { return currentAssetDir_; }
    [[nodiscard]] bool isLoaded() const { return !currentAssetDir_.empty() && scene_.loaded() && shadowReady_; }
    [[nodiscard]] float loadProgress() const { return currentAssetDir_.empty() ? 0.0f : scene_.progress(); }
    [[nodiscard]] const std::string& loadStatus() const { return scene_.status(); }
    [[nodiscard]] bool loadFailed() const { return scene_.failed() || initFailed_; }
    [[nodiscard]] const RenderStats& stats() const { return stats_; }
    [[nodiscard]] float loopDuration() const { return path_.loopDuration(); }
    void setCameraOverride(const CameraOverride& o) {
        cameraOverride_ = o;
        invalidateHistory();  // camera may have jumped; last frame's data is no longer valid
    }
    void invalidateHistory() {
        hizValid_ = false;
        taaHistoryValid_ = false;
        aoValid_ = false;
    }

private:
    static constexpr uint32_t kFramesInFlight = 2;
    static constexpr uint32_t kBloomLevels = 6;
    static constexpr uint32_t kMaxPointLights = 8;
    static constexpr uint32_t kMaxHizLevels = 16;

    struct FrameResources {
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkSemaphore acquired = VK_NULL_HANDLE;
        GpuBuffer uniforms;
        GpuBuffer draws;
        GpuBuffer counts;
        VkDescriptorSet set = VK_NULL_HANDLE;
        bool submitted = false;
        bool timestampsValid = false;
    };

    bool createGlobalResources();
    bool createDescriptorLayouts();
    bool createPipelines();
    VkPipeline createGraphicsPipeline(const uint32_t* vs, size_t vsSize, const uint32_t* fs, size_t fsSize,
                                      VkPipelineLayout layout, VkFormat colorFormat, VkFormat depthFormat,
                                      VkSampleCountFlagBits samples, bool vertexInput, VkCullModeFlags cull,
                                      bool depthTest, bool depthWrite, VkCompareOp depthCompare, int blendMode,
                                      bool alphaToCoverage, uint32_t alphaModeConstant, bool depthBias,
                                      const char* name, VkPipelineLayout layoutOverride = VK_NULL_HANDLE);
    VkPipeline createComputePipeline(const uint32_t* cs, size_t size, const char* name,
                                     VkPipelineLayout layout = VK_NULL_HANDLE);
    bool createSwapchain();
    void destroySwapchain();
    bool createRenderTargets();
    void destroyRenderTargets();
    void writeStaticDescriptors();
    void writeRenderTargetDescriptors();
    bool buildShadowMap();
    void updateUniforms(FrameResources& frame, double time, double dt);
    void recordScene(VkCommandBuffer cmd, FrameResources& frame, uint32_t imageIndex);
    void recordLoading(VkCommandBuffer cmd, uint32_t imageIndex, double time);
    void readBackStats(FrameResources& frame, uint32_t frameIndex);

    VulkanContext ctx_;
    AAssetManager* assets_ = nullptr;
    std::string currentAssetDir_;
    Scene scene_;
    CameraPath path_;
    CameraOverride cameraOverride_;
    RenderStats stats_;
    int lastCameraSegment_ = -1;
    bool initFailed_ = false;
    bool initialized_ = false;

    // Window / swapchain
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat swapchainFormat_ = VK_FORMAT_UNDEFINED;
    bool swapchainNeedsSrgbEncode_ = false;
    VkExtent2D swapchainExtent_{};   // native (pre-rotation) extent
    VkExtent2D logicalExtent_{};     // extent as seen by the user
    uint32_t preRotation_ = 0;       // 0, 1 (90), 2 (180), 3 (270)
    VkSurfaceTransformFlagBitsKHR preTransform_ = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    std::vector<VkImage> swapchainImages_;
    std::vector<VkImageView> swapchainViews_;
    std::vector<VkSemaphore> renderDone_;
    bool swapchainDirty_ = false;

    // Frame resources
    std::array<FrameResources, kFramesInFlight> frames_{};
    uint32_t frameIndex_ = 0;
    uint64_t frameCounter_ = 0;
    VkQueryPool timestampPool_ = VK_NULL_HANDLE;

    // Render targets (depend on window size)
    float renderScale_ = 1.0f;
    VkExtent2D renderExtent_{};
    VkSampleCountFlagBits msaaSamples_ = VK_SAMPLE_COUNT_1_BIT;
    VkFormat hdrFormat_ = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkFormat depthFormat_ = VK_FORMAT_D32_SFLOAT;
    GpuImage msaaColor_;
    GpuImage msaaDepth_;
    GpuImage hdrResolve_;
    GpuImage bloom_;
    GpuImage depthResolve_;  // single-sample depth (the depth attachment itself without MSAA)
    std::array<GpuImage, kFramesInFlight> taaOutput_;  // TAA output / history ping-pong
    bool taaEnabled_ = true;
    bool taaHistoryValid_ = false;
    uint32_t jitterIndex_ = 0;
    Mat4 prevViewProjJittered_ = Mat4::identity();
    Mat4 prevViewProjUnjittered_ = Mat4::identity();
    GpuImage hiz_;           // hierarchical depth pyramid for occlusion culling
    VkExtent2D hizExtent_{};
    uint32_t hizMipCount_ = 0;
    std::array<VkImageView, kMaxHizLevels> hizViews_{};
    std::array<VkDescriptorSet, kMaxHizLevels> hizSets_{};
    bool hizValid_ = false;
    bool occlusionCulling_ = true;
    bool buildHiz_ = true;
    bool skyDebug_ = false;  // `setprop debug.bistro.skydebug 1`: paint the sky magenta to find leaks
    // Screen-space AO (`setprop debug.bistro.ao 0` disables). Computed at half resolution after the main
    // pass into ao_[frame slot] and reprojected by the next frame's scene shading.
    bool aoEnabled_ = true;
    bool aoDebug_ = false;  // `setprop debug.bistro.aodebug 1`: show the ambient occlusion term only
    bool aoValid_ = false;
    VkExtent2D aoExtent_{};
    GpuImage aoRaw_;
    std::array<GpuImage, kFramesInFlight> ao_;
    std::array<VkDescriptorSet, kFramesInFlight * 2> aoSets_{};  // [slot * 2] = AO, [slot * 2 + 1] = blur
    VkDescriptorSetLayout aoSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout aoLayout_ = VK_NULL_HANDLE;
    VkPipeline aoPipeline_ = VK_NULL_HANDLE;
    VkPipeline aoBlurPipeline_ = VK_NULL_HANDLE;
    std::array<VkImageView, kBloomLevels> bloomViews_{};
    std::array<VkExtent2D, kBloomLevels> bloomExtents_{};

    // Static resources
    GpuImage shadowMap_;
    uint32_t shadowSize_ = 8192;
    Mat4 shadowViewProj_ = Mat4::identity();
    float shadowWorldTexel_ = 0.02f;
    bool shadowReady_ = false;
    GpuBuffer exposureBuffer_;
    GpuBuffer shadowDraws_;
    uint32_t shadowOpaqueCount_ = 0;
    uint32_t shadowMaskCount_ = 0;

    VkSampler materialSampler_ = VK_NULL_HANDLE;
    VkSampler linearClampSampler_ = VK_NULL_HANDLE;
    VkSampler envSampler_ = VK_NULL_HANDLE;
    VkSampler shadowSampler_ = VK_NULL_HANDLE;
    VkSampler nearestSampler_ = VK_NULL_HANDLE;

    VkDescriptorSetLayout globalSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout textureSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout hizSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout hizLayout_ = VK_NULL_HANDLE;
    VkPipeline hizPipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    // [0], [1] = HDR input for frame slot 0/1, [i+2] = bloom mip i
    std::array<VkDescriptorSet, kBloomLevels + 2> bloomSourceSets_{};
    VkDescriptorSetLayout taaSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout taaLayout_ = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, kFramesInFlight> taaSets_{};
    VkPipeline taaPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout postLayout_ = VK_NULL_HANDLE;

    VkPipeline cullPipeline_ = VK_NULL_HANDLE;
    VkPipeline exposurePipeline_ = VK_NULL_HANDLE;
    VkPipeline opaquePipeline_ = VK_NULL_HANDLE;
    VkPipeline maskPipeline_ = VK_NULL_HANDLE;
    VkPipeline blendPipeline_ = VK_NULL_HANDLE;
    VkPipeline skyPipeline_ = VK_NULL_HANDLE;
    VkPipeline shadowOpaquePipeline_ = VK_NULL_HANDLE;
    VkPipeline shadowMaskPipeline_ = VK_NULL_HANDLE;
    VkPipeline bloomDownPipeline_ = VK_NULL_HANDLE;
    VkPipeline bloomUpPipeline_ = VK_NULL_HANDLE;
    VkPipeline tonemapPipeline_ = VK_NULL_HANDLE;
    VkPipeline loadingPipeline_ = VK_NULL_HANDLE;
    VkFormat pipelineSwapchainFormat_ = VK_FORMAT_UNDEFINED;

    // Lighting configuration
    float envYaw_ = 0.0f;
    Vec3 sunDirection_{0, 1, 0};
    Vec3 sunColor_{1, 1, 1};
    std::vector<Vec4> pointLights_;  // pairs of (position+range, colour)

    double lastFpsTime_ = 0.0;
    double wallClock_ = 0.0;
    uint32_t fpsFrames_ = 0;
};
