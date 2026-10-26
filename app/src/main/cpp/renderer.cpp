#include "renderer.h"

#include <android/log.h>
#include <sys/system_properties.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include "vulkan_debug.h"

#define LOG_TAG "BistroRenderer"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

alignas(4) const uint32_t kCullComp[] =
#include "cull.comp.spv.h"
;
alignas(4) const uint32_t kHizComp[] =
#include "hiz.comp.spv.h"
;
alignas(4) const uint32_t kExposureComp[] =
#include "exposure.comp.spv.h"
;
alignas(4) const uint32_t kSceneVert[] =
#include "scene.vert.spv.h"
;
alignas(4) const uint32_t kSceneFrag[] =
#include "scene.frag.spv.h"
;
alignas(4) const uint32_t kShadowVert[] =
#include "shadow.vert.spv.h"
;
alignas(4) const uint32_t kShadowFrag[] =
#include "shadow.frag.spv.h"
;
alignas(4) const uint32_t kFullscreenVert[] =
#include "fullscreen.vert.spv.h"
;
alignas(4) const uint32_t kSkyFrag[] =
#include "sky.frag.spv.h"
;
alignas(4) const uint32_t kBloomDownFrag[] =
#include "bloom_down.frag.spv.h"
;
alignas(4) const uint32_t kBloomUpFrag[] =
#include "bloom_up.frag.spv.h"
;
alignas(4) const uint32_t kTonemapFrag[] =
#include "tonemap.frag.spv.h"
;
alignas(4) const uint32_t kTaaFrag[] =
#include "taa.frag.spv.h"
;
alignas(4) const uint32_t kLoadingFrag[] =
#include "loading.frag.spv.h"
;
alignas(4) const uint32_t kAoComp[] =
#include "ao.comp.spv.h"
;
alignas(4) const uint32_t kAoBlurComp[] =
#include "ao_blur.comp.spv.h"
;

constexpr float kFovY = 52.0f * 3.14159265f / 180.0f;
constexpr float kNearPlane = 0.1f;
constexpr uint32_t kPushConstantSize = 64;
constexpr VkShaderStageFlags kAllStages =
    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;

// Blend modes for createGraphicsPipeline.
constexpr int kBlendNone = 0;
constexpr int kBlendPremultiplied = 1;
constexpr int kBlendAdditive = 2;

struct DrawCommand {
    uint32_t indexCount;
    uint32_t instanceCount;
    uint32_t firstIndex;
    int32_t vertexOffset;
    uint32_t firstInstance;
};
static_assert(sizeof(DrawCommand) == sizeof(VkDrawIndexedIndirectCommand));

struct alignas(16) FrameUniforms {
    Mat4 viewProj;
    Mat4 invViewProj;
    Mat4 shadowViewProj;
    Mat4 prevViewProj;
    Mat4 reprojection;
    float cameraPos[4];
    float sunDirection[4];
    float sunColor[4];
    float envRotation[4];
    float sh[9][4];
    Vec4 frustumPlanes[6];
    float fogParams[4];
    float fogColor[4];
    float renderParams[4];
    float cullParams[4];
    float postParams[4];
    float shadowParams[4];
    float extraParams[4];
    float exposureParams[4];
    float exposureParams2[4];
    float occlusionParams[4];
    float taaParams[4];
    float aoParams[4];
    float lights[8][8];
    uint32_t lightCount[4];
};
static_assert(sizeof(FrameUniforms) == 1088, "FrameUniforms must match the std140 layout in common.glsl");

struct LoadingPush {
    float outputSize[2];
    float progress;
    float rotation;
    float time;
    float srgbEncode;
};

struct BloomPush {
    float texel[2];
    float a;
    float b;
};

// Reads an optional tuning override, e.g. `adb shell setprop debug.bistro.scale 0.6`.
float debugProperty(const char* name, float fallback) {
    char value[PROP_VALUE_MAX] = {};
    if (__system_property_get(name, value) > 0) return static_cast<float>(atof(value));
    return fallback;
}

Vec3 envToWorld(const Vec3& e, float yaw) {
    float c = std::cos(yaw), s = std::sin(yaw);
    return {c * e.x - s * e.z, e.y, s * e.x + c * e.z};
}

}  // namespace

// ------------------------------------------------------------------------------------------------
// Initialisation
// ------------------------------------------------------------------------------------------------

bool Renderer::init(AAssetManager* assets) {
    assets_ = assets;
    renderScale_ = std::clamp(debugProperty("debug.bistro.scale", renderScale_), 0.25f, 1.0f);
    occlusionCulling_ = debugProperty("debug.bistro.occlusion", 1.0f) > 0.5f;
    buildHiz_ = debugProperty("debug.bistro.hiz", 1.0f) > 0.5f;
    skyDebug_ = debugProperty("debug.bistro.skydebug", 0.0f) > 0.5f;
    aoEnabled_ = debugProperty("debug.bistro.ao", 1.0f) > 0.5f;
    aoDebug_ = debugProperty("debug.bistro.aodebug", 0.0f) > 0.5f;
    if (!ctx_.init()) {
        initFailed_ = true;
        return false;
    }
    if (!createDescriptorLayouts() || !createGlobalResources() || !createPipelines()) {
        initFailed_ = true;
        return false;
    }
    initialized_ = true;
    LOGI("Renderer initialised");
    return true;
}

void Renderer::setEnvironment(const std::string& assetDir) {
    if (!initialized_) return;
    if (assetDir == currentAssetDir_ && !scene_.failed() && !initFailed_) {
        lastCameraSegment_ = -1;
        shadowFitted_ = false;
        invalidateHistory();
        return;
    }
    vkDeviceWaitIdle(ctx_.device);
    scene_.destroy();
    ctx_.destroyImage(shadowMap_);
    ctx_.destroyBuffer(shadowDraws_);
    for (auto& f : frames_) {
        ctx_.destroyBuffer(f.draws);
    }
    clusterLightBounds_.clear();
    shadowReady_ = false;
    shadowFitted_ = false;
    shadowDirty_ = false;
    shadowOpaqueCount_ = 0;
    shadowMaskCount_ = 0;
    pointLights_.clear();
    lastCameraSegment_ = -1;
    invalidateHistory();
    initFailed_ = false;
    stats_ = RenderStats{};
    stats_.renderWidth = renderExtent_.width;
    stats_.renderHeight = renderExtent_.height;
    currentAssetDir_ = assetDir;
    if (exposureBuffer_.mapped) {
        float initExp[4] = {1.0f, 1.0f, 0.0f, 0.0f};
        memcpy(exposureBuffer_.mapped, initExp, sizeof(initExp));
    }
    if (!assetDir.empty()) {
        scene_.begin(&ctx_, assets_, assetDir);
        path_.loadFromManifest(scene_.manifestJson);
        LOGI("Switched environment to %s (%d camera loops, %.1f s total, %.0f m)",
             assetDir.c_str(), path_.loopCount(), path_.loopDuration(), path_.totalLength());
    }
}

bool Renderer::createDescriptorLayouts() {
    std::vector<VkDescriptorSetLayoutBinding> b;
    auto add = [&](uint32_t binding, VkDescriptorType type, uint32_t count = 1) {
        VkDescriptorSetLayoutBinding x{};
        x.binding = binding;
        x.descriptorType = type;
        x.descriptorCount = count;
        x.stageFlags = kAllStages;
        b.push_back(x);
    };
    add(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    add(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    add(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    add(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    add(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    add(5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    add(6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    add(7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    add(8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    add(9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    add(10, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    add(11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    add(13, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    add(14, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    add(15, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4096);

    std::vector<VkDescriptorBindingFlags> flags(b.size(), VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
    flags.back() |= VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;
    VkDescriptorSetLayoutBindingFlagsCreateInfo fci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    fci.bindingCount = static_cast<uint32_t>(flags.size());
    fci.pBindingFlags = flags.data();
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.pNext = &fci;
    lci.bindingCount = static_cast<uint32_t>(b.size());
    lci.pBindings = b.data();
    if (vkCreateDescriptorSetLayout(ctx_.device, &lci, nullptr, &globalSetLayout_) != VK_SUCCESS) return false;
    VulkanDebug::setObjectName(ctx_.device, globalSetLayout_, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                               "Global Descriptor Set Layout (Set 0)");

    VkDescriptorSetLayoutBinding tb{};
    tb.binding = 0;
    tb.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    tb.descriptorCount = 1;
    tb.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo tci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    tci.bindingCount = 1;
    tci.pBindings = &tb;
    if (vkCreateDescriptorSetLayout(ctx_.device, &tci, nullptr, &textureSetLayout_) != VK_SUCCESS) return false;
    VulkanDebug::setObjectName(ctx_.device, textureSetLayout_, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                               "Post-Process Texture Descriptor Set Layout (Set 1)");

    VkDescriptorSetLayoutBinding hb[2] = {};
    hb[0].binding = 0;
    hb[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    hb[0].descriptorCount = 1;
    hb[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    hb[1].binding = 1;
    hb[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    hb[1].descriptorCount = 1;
    hb[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo hci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    hci.bindingCount = 2;
    hci.pBindings = hb;
    if (vkCreateDescriptorSetLayout(ctx_.device, &hci, nullptr, &hizSetLayout_) != VK_SUCCESS) return false;
    VulkanDebug::setObjectName(ctx_.device, hizSetLayout_, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                               "Hi-Z Pyramid Descriptor Set Layout (Set 1)");

    VkDescriptorSetLayoutBinding taaBindings[3] = {};
    for (uint32_t i = 0; i < 3; ++i) {
        taaBindings[i].binding = i;
        taaBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        taaBindings[i].descriptorCount = 1;
        taaBindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo tlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    tlci.bindingCount = 3;
    tlci.pBindings = taaBindings;
    if (vkCreateDescriptorSetLayout(ctx_.device, &tlci, nullptr, &taaSetLayout_) != VK_SUCCESS) return false;
    VulkanDebug::setObjectName(ctx_.device, taaSetLayout_, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                               "TAA Resolve Descriptor Set Layout (Set 1)");

    // SSAO compute: 0 = scene depth, 1 = AO input (blur), 2 = AO output (storage).
    VkDescriptorSetLayoutBinding aoBindings[3] = {};
    for (uint32_t i = 0; i < 3; ++i) {
        aoBindings[i].binding = i;
        aoBindings[i].descriptorType =
            i == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        aoBindings[i].descriptorCount = 1;
        aoBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo alci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    alci.bindingCount = 3;
    alci.pBindings = aoBindings;
    if (vkCreateDescriptorSetLayout(ctx_.device, &alci, nullptr, &aoSetLayout_) != VK_SUCCESS) return false;
    VulkanDebug::setObjectName(ctx_.device, aoSetLayout_, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                               "SSAO Compute Descriptor Set Layout (Set 1)");

    VkPushConstantRange range{kAllStages, 0, kPushConstantSize};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &globalSetLayout_;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &range;
    if (vkCreatePipelineLayout(ctx_.device, &pli, nullptr, &pipelineLayout_) != VK_SUCCESS) return false;
    VulkanDebug::setObjectName(ctx_.device, pipelineLayout_, VK_OBJECT_TYPE_PIPELINE_LAYOUT, "Global Pipeline Layout");
    VkDescriptorSetLayout postSets[2] = {globalSetLayout_, textureSetLayout_};
    pli.setLayoutCount = 2;
    pli.pSetLayouts = postSets;
    if (vkCreatePipelineLayout(ctx_.device, &pli, nullptr, &postLayout_) != VK_SUCCESS) return false;
    VulkanDebug::setObjectName(ctx_.device, postLayout_, VK_OBJECT_TYPE_PIPELINE_LAYOUT, "Post-Process Pipeline Layout");
    VkDescriptorSetLayout hizSets[2] = {globalSetLayout_, hizSetLayout_};
    pli.pSetLayouts = hizSets;
    if (vkCreatePipelineLayout(ctx_.device, &pli, nullptr, &hizLayout_) != VK_SUCCESS) return false;
    VulkanDebug::setObjectName(ctx_.device, hizLayout_, VK_OBJECT_TYPE_PIPELINE_LAYOUT, "Hi-Z Compute Pipeline Layout");
    VkDescriptorSetLayout taaSets[2] = {globalSetLayout_, taaSetLayout_};
    pli.pSetLayouts = taaSets;
    if (vkCreatePipelineLayout(ctx_.device, &pli, nullptr, &taaLayout_) != VK_SUCCESS) return false;
    VulkanDebug::setObjectName(ctx_.device, taaLayout_, VK_OBJECT_TYPE_PIPELINE_LAYOUT, "TAA Resolve Pipeline Layout");
    VkDescriptorSetLayout aoSets[2] = {globalSetLayout_, aoSetLayout_};
    pli.pSetLayouts = aoSets;
    if (vkCreatePipelineLayout(ctx_.device, &pli, nullptr, &aoLayout_) != VK_SUCCESS) return false;
    VulkanDebug::setObjectName(ctx_.device, aoLayout_, VK_OBJECT_TYPE_PIPELINE_LAYOUT, "SSAO Compute Pipeline Layout");

    const uint32_t aoSetCount = static_cast<uint32_t>(aoSets_.size());
    VkDescriptorPoolSize sizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kFramesInFlight},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5 * kFramesInFlight},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         (4096 + 8) * kFramesInFlight + kBloomLevels + 2 + kMaxHizLevels + 3 * kFramesInFlight + 2 * aoSetCount},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kMaxHizLevels + aoSetCount},
    };
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = kFramesInFlight * 2 + kBloomLevels + 2 + kMaxHizLevels + aoSetCount;
    pci.poolSizeCount = 4;
    pci.pPoolSizes = sizes;
    if (vkCreateDescriptorPool(ctx_.device, &pci, nullptr, &descriptorPool_) != VK_SUCCESS) return false;
    VulkanDebug::setObjectName(ctx_.device, descriptorPool_, VK_OBJECT_TYPE_DESCRIPTOR_POOL, "Main Descriptor Pool");

    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        FrameResources& f = frames_[i];
        uint32_t variableCount = 4096;
        VkDescriptorSetVariableDescriptorCountAllocateInfo vci{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO};
        vci.descriptorSetCount = 1;
        vci.pDescriptorCounts = &variableCount;
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.pNext = &vci;
        ai.descriptorPool = descriptorPool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &globalSetLayout_;
        if (vkAllocateDescriptorSets(ctx_.device, &ai, &f.set) != VK_SUCCESS) return false;
        std::string name = "Global Descriptor Set [Frame " + std::to_string(i) + "]";
        VulkanDebug::setObjectName(ctx_.device, f.set, VK_OBJECT_TYPE_DESCRIPTOR_SET, name.c_str());
    }
    for (size_t i = 0; i < bloomSourceSets_.size(); ++i) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = descriptorPool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &textureSetLayout_;
        if (vkAllocateDescriptorSets(ctx_.device, &ai, &bloomSourceSets_[i]) != VK_SUCCESS) return false;
        std::string name = i < 2 ? ("Bloom Source Descriptor Set [HDR Frame " + std::to_string(i) + "]")
                                 : ("Bloom Source Descriptor Set [Mip " + std::to_string(i - 2) + "]");
        VulkanDebug::setObjectName(ctx_.device, bloomSourceSets_[i], VK_OBJECT_TYPE_DESCRIPTOR_SET, name.c_str());
    }
    for (size_t i = 0; i < hizSets_.size(); ++i) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = descriptorPool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &hizSetLayout_;
        if (vkAllocateDescriptorSets(ctx_.device, &ai, &hizSets_[i]) != VK_SUCCESS) return false;
        std::string name = "Hi-Z Pyramid Descriptor Set [Mip " + std::to_string(i) + "]";
        VulkanDebug::setObjectName(ctx_.device, hizSets_[i], VK_OBJECT_TYPE_DESCRIPTOR_SET, name.c_str());
    }
    for (size_t i = 0; i < taaSets_.size(); ++i) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = descriptorPool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &taaSetLayout_;
        if (vkAllocateDescriptorSets(ctx_.device, &ai, &taaSets_[i]) != VK_SUCCESS) return false;
        std::string name = "TAA Resolve Descriptor Set [Frame " + std::to_string(i) + "]";
        VulkanDebug::setObjectName(ctx_.device, taaSets_[i], VK_OBJECT_TYPE_DESCRIPTOR_SET, name.c_str());
    }
    for (size_t i = 0; i < aoSets_.size(); ++i) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = descriptorPool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &aoSetLayout_;
        if (vkAllocateDescriptorSets(ctx_.device, &ai, &aoSets_[i]) != VK_SUCCESS) return false;
        std::string name = std::string((i % 2 == 0) ? "SSAO Raw Compute Descriptor Set [Frame "
                                                    : "SSAO Blur Compute Descriptor Set [Frame ") +
                           std::to_string(i / 2) + "]";
        VulkanDebug::setObjectName(ctx_.device, aoSets_[i], VK_OBJECT_TYPE_DESCRIPTOR_SET, name.c_str());
    }
    return true;
}

bool Renderer::createGlobalResources() {
    VkDevice dev = ctx_.device;
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.maxLod = VK_LOD_CLAMP_NONE;
    sci.anisotropyEnable = ctx_.maxAnisotropy > 1.0f;
    sci.maxAnisotropy = ctx_.maxAnisotropy;
    if (vkCreateSampler(dev, &sci, nullptr, &materialSampler_) != VK_SUCCESS) return false;
    VulkanDebug::setObjectName(dev, materialSampler_, VK_OBJECT_TYPE_SAMPLER, "Material Anisotropic Repeat Sampler");

    sci.anisotropyEnable = VK_FALSE;
    sci.maxAnisotropy = 1.0f;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(dev, &sci, nullptr, &envSampler_) != VK_SUCCESS) return false;
    VulkanDebug::setObjectName(dev, envSampler_, VK_OBJECT_TYPE_SAMPLER, "Environment LatLong Sampler");

    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(dev, &sci, nullptr, &linearClampSampler_) != VK_SUCCESS) return false;
    VulkanDebug::setObjectName(dev, linearClampSampler_, VK_OBJECT_TYPE_SAMPLER, "Linear Clamp Sampler");

    sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    if (vkCreateSampler(dev, &sci, nullptr, &nearestSampler_) != VK_SUCCESS) return false;
    VulkanDebug::setObjectName(dev, nearestSampler_, VK_OBJECT_TYPE_SAMPLER, "Nearest Clamp Sampler");
    sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;

    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    sci.compareEnable = VK_TRUE;
    sci.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.maxLod = 0.0f;
    if (vkCreateSampler(dev, &sci, nullptr, &shadowSampler_) != VK_SUCCESS) return false;
    VulkanDebug::setObjectName(dev, shadowSampler_, VK_OBJECT_TYPE_SAMPLER, "Shadow Depth Compare Sampler");

    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        FrameResources& f = frames_[i];
        std::string suffix = " [Frame " + std::to_string(i) + "]";
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = ctx_.queueFamily;
        if (vkCreateCommandPool(dev, &pci, nullptr, &f.pool) != VK_SUCCESS) return false;
        VulkanDebug::setObjectName(dev, f.pool, VK_OBJECT_TYPE_COMMAND_POOL, ("Frame Command Pool" + suffix).c_str());
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = f.pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(dev, &ai, &f.cmd) != VK_SUCCESS) return false;
        VulkanDebug::setObjectName(dev, f.cmd, VK_OBJECT_TYPE_COMMAND_BUFFER, ("Frame Command Buffer" + suffix).c_str());
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        if (vkCreateFence(dev, &fci, nullptr, &f.fence) != VK_SUCCESS) return false;
        VulkanDebug::setObjectName(dev, f.fence, VK_OBJECT_TYPE_FENCE, ("Frame In-Flight Fence" + suffix).c_str());
        VkSemaphoreCreateInfo sci2{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        if (vkCreateSemaphore(dev, &sci2, nullptr, &f.acquired) != VK_SUCCESS) return false;
        VulkanDebug::setObjectName(dev, f.acquired, VK_OBJECT_TYPE_SEMAPHORE,
                                   ("Image Acquired Semaphore" + suffix).c_str());
        if (!ctx_.createBuffer(sizeof(FrameUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               f.uniforms, ("Frame Uniforms" + suffix).c_str())) {
            return false;
        }
        if (!ctx_.createBuffer(64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, f.counts,
                               ("Indirect Draw Counts" + suffix).c_str())) {
            return false;
        }
    }

    float initialExposure[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (!ctx_.createBuffer(16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, exposureBuffer_,
                           "Auto-Exposure State Buffer")) {
        return false;
    }
    memcpy(exposureBuffer_.mapped, initialExposure, sizeof(initialExposure));

    VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qci.queryCount = 2 * kFramesInFlight;
    if (vkCreateQueryPool(dev, &qci, nullptr, &timestampPool_) == VK_SUCCESS) {
        VulkanDebug::setObjectName(dev, timestampPool_, VK_OBJECT_TYPE_QUERY_POOL, "Frame Timestamp Query Pool");
    }

    // HDR and depth formats.
    const VkFormatFeatureFlags hdrFeatures = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                                             VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT |
                                             VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    hdrFormat_ = ctx_.isFormatSupported(VK_FORMAT_B10G11R11_UFLOAT_PACK32, hdrFeatures) &&
                         debugProperty("debug.bistro.hdr", 0.0f) < 0.5f
                     ? VK_FORMAT_B10G11R11_UFLOAT_PACK32
                     : VK_FORMAT_R16G16B16A16_SFLOAT;
    depthFormat_ = ctx_.isFormatSupported(VK_FORMAT_D32_SFLOAT, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
                       ? VK_FORMAT_D32_SFLOAT
                       : VK_FORMAT_D24_UNORM_S8_UINT;
    VkSampleCountFlags counts = ctx_.properties.limits.framebufferColorSampleCounts &
                                ctx_.properties.limits.framebufferDepthSampleCounts;
    // Default: no MSAA, temporal anti-aliasing instead (MSAA proved expensive with this geometry density on
    // tile-based GPUs). `setprop debug.bistro.msaa 4` switches to 4x MSAA + alpha-to-coverage.
    msaaSamples_ = VK_SAMPLE_COUNT_1_BIT;
    if (debugProperty("debug.bistro.msaa", 1.0f) >= 4.0f && (counts & VK_SAMPLE_COUNT_4_BIT))
        msaaSamples_ = VK_SAMPLE_COUNT_4_BIT;
    taaEnabled_ = msaaSamples_ == VK_SAMPLE_COUNT_1_BIT && debugProperty("debug.bistro.taa", 0.0f) > 0.5f;
    // SSAO reads the single-sample depth, which the MSAA path only resolves when building Hi-Z.
    if (msaaSamples_ != VK_SAMPLE_COUNT_1_BIT && !buildHiz_) aoEnabled_ = false;
    LOGI("HDR format %d, depth format %d, MSAA x%d", hdrFormat_, depthFormat_, msaaSamples_);
    return true;
}

VkPipeline Renderer::createComputePipeline(const uint32_t* cs, size_t size, const char* csName, const char* name,
                                           VkPipelineLayout layout) {
    VkShaderModule module = ctx_.createShaderModule(cs, size, csName);
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = module;
    ci.stage.pName = "main";
    ci.layout = layout ? layout : pipelineLayout_;
    VkPipeline p = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(ctx_.device, VK_NULL_HANDLE, 1, &ci, nullptr, &p) != VK_SUCCESS)
        LOGE("Failed to create compute pipeline %s", name);
    vkDestroyShaderModule(ctx_.device, module, nullptr);
    VulkanDebug::setObjectName(ctx_.device, p, VK_OBJECT_TYPE_PIPELINE, name);
    return p;
}

VkPipeline Renderer::createGraphicsPipeline(const uint32_t* vs, size_t vsSize, const char* vsName, const uint32_t* fs,
                                            size_t fsSize, const char* fsName, VkPipelineLayout layout,
                                            VkFormat colorFormat, VkFormat depthFormat, VkSampleCountFlagBits samples,
                                            bool vertexInput, VkCullModeFlags cull, bool depthTest, bool depthWrite,
                                            VkCompareOp depthCompare, int blendMode, bool alphaToCoverage,
                                            uint32_t alphaModeConstant, bool depthBias, const char* name,
                                            VkPipelineLayout layoutOverride) {
    if (layoutOverride) layout = layoutOverride;
    VkShaderModule vsm = ctx_.createShaderModule(vs, vsSize, vsName);
    VkShaderModule fsm = fs ? ctx_.createShaderModule(fs, fsSize, fsName) : VK_NULL_HANDLE;

    struct {
        uint32_t alphaMode;
        VkBool32 alphaToCoverage;
    } specData{alphaModeConstant, alphaToCoverage ? VK_TRUE : VK_FALSE};
    VkSpecializationMapEntry entries[2] = {{0, 0, sizeof(uint32_t)}, {1, sizeof(uint32_t), sizeof(VkBool32)}};
    VkSpecializationInfo spec{2, entries, sizeof(specData), &specData};

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vsm;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fsm;
    stages[1].pName = "main";
    stages[1].pSpecializationInfo = &spec;

    VkVertexInputBindingDescription binding{0, sizeof(bistro::GpuVertex), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription attrs[5] = {
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(bistro::GpuVertex, position)},
        {1, 0, VK_FORMAT_R16G16_SNORM, offsetof(bistro::GpuVertex, normal)},
        {2, 0, VK_FORMAT_R8G8B8A8_SNORM, offsetof(bistro::GpuVertex, tangent)},
        {3, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(bistro::GpuVertex, uv)},
        {4, 0, VK_FORMAT_R8G8B8A8_UNORM, offsetof(bistro::GpuVertex, extra)},
    };
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    if (vertexInput) {
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &binding;
        vi.vertexAttributeDescriptionCount = 5;
        vi.pVertexAttributeDescriptions = attrs;
    }
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = cull;
    // Both the projection matrix (r.at(1,1) = -f) and Vulkan's framebuffer coordinates (+Y down) flip Y,
    // so the two flips cancel and CCW triangles in world space remain counter-clockwise on screen.
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;
    rs.depthBiasEnable = depthBias;
    const float biasSign = (depthCompare == VK_COMPARE_OP_GREATER_OR_EQUAL || depthCompare == VK_COMPARE_OP_GREATER) ? 1.0f : 1.0f;
    if (depthCompare == VK_COMPARE_OP_GREATER_OR_EQUAL || depthCompare == VK_COMPARE_OP_GREATER) {
        // Reversed-Z (near = 1, far = 0): nudge alpha-masked decals slightly towards the camera so coplanar
        // warning labels and stencil decals never Z-fight with the underlying opaque hull.
        rs.depthBiasConstantFactor = 2.0f;
        rs.depthBiasSlopeFactor = 1.5f;
    } else {
        rs.depthBiasConstantFactor = 2.0f;
        rs.depthBiasSlopeFactor = 2.5f;
    }
    (void)biasSign;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = samples;
    ms.alphaToCoverageEnable = alphaToCoverage;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = depthTest;
    ds.depthWriteEnable = depthWrite;
    ds.depthCompareOp = depthCompare;
    VkPipelineColorBlendAttachmentState att{};
    att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                         VK_COLOR_COMPONENT_A_BIT;
    if (blendMode == kBlendPremultiplied) {
        att.blendEnable = VK_TRUE;
        att.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        att.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        att.colorBlendOp = VK_BLEND_OP_ADD;
        att.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        att.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        att.alphaBlendOp = VK_BLEND_OP_ADD;
    } else if (blendMode == kBlendAdditive) {
        att.blendEnable = VK_TRUE;
        att.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        att.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        att.colorBlendOp = VK_BLEND_OP_ADD;
        att.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        att.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        att.alphaBlendOp = VK_BLEND_OP_ADD;
    }
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = colorFormat != VK_FORMAT_UNDEFINED ? 1 : 0;
    cb.pAttachments = &att;
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dy.dynamicStateCount = 2;
    dy.pDynamicStates = dyn;

    VkPipelineRenderingCreateInfo ri{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    ri.colorAttachmentCount = colorFormat != VK_FORMAT_UNDEFINED ? 1 : 0;
    ri.pColorAttachmentFormats = &colorFormat;
    ri.depthAttachmentFormat = depthFormat;

    VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pci.pNext = &ri;
    pci.stageCount = fs ? 2 : 1;
    pci.pStages = stages;
    pci.pVertexInputState = &vi;
    pci.pInputAssemblyState = &ia;
    pci.pViewportState = &vp;
    pci.pRasterizationState = &rs;
    pci.pMultisampleState = &ms;
    pci.pDepthStencilState = &ds;
    pci.pColorBlendState = &cb;
    pci.pDynamicState = &dy;
    pci.layout = layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(ctx_.device, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline) != VK_SUCCESS) {
        LOGE("Failed to create pipeline %s", name);
    }
    vkDestroyShaderModule(ctx_.device, vsm, nullptr);
    if (fsm) vkDestroyShaderModule(ctx_.device, fsm, nullptr);
    VulkanDebug::setObjectName(ctx_.device, pipeline, VK_OBJECT_TYPE_PIPELINE, name);
    return pipeline;
}

bool Renderer::createPipelines() {
    cullPipeline_ = createComputePipeline(kCullComp, sizeof(kCullComp), "cull.comp", "GPU Cluster Culling (cull.comp)");
    exposurePipeline_ =
        createComputePipeline(kExposureComp, sizeof(kExposureComp), "exposure.comp", "Auto Exposure (exposure.comp)");
    aoPipeline_ = createComputePipeline(kAoComp, sizeof(kAoComp), "ao.comp", "SSAO Compute (ao.comp)", aoLayout_);
    aoBlurPipeline_ =
        createComputePipeline(kAoBlurComp, sizeof(kAoBlurComp), "ao_blur.comp", "SSAO Blur (ao_blur.comp)", aoLayout_);
    hizPipeline_ =
        createComputePipeline(kHizComp, sizeof(kHizComp), "hiz.comp", "Hi-Z Pyramid Build (hiz.comp)", hizLayout_);
    const VkCompareOp reversedZ = VK_COMPARE_OP_GREATER_OR_EQUAL;
    const VkCompareOp reversedZStrict = VK_COMPARE_OP_GREATER;
    opaquePipeline_ = createGraphicsPipeline(kSceneVert, sizeof(kSceneVert), "scene.vert", kSceneFrag,
                                             sizeof(kSceneFrag), "scene.frag", pipelineLayout_, hdrFormat_,
                                             depthFormat_, msaaSamples_, true, VK_CULL_MODE_NONE, true, true,
                                             reversedZStrict, kBlendNone, false, 0, false,
                                             "Scene Opaque (scene.vert + scene.frag)");
    maskPipeline_ = createGraphicsPipeline(kSceneVert, sizeof(kSceneVert), "scene.vert", kSceneFrag, sizeof(kSceneFrag),
                                           "scene.frag", pipelineLayout_, hdrFormat_, depthFormat_, msaaSamples_, true,
                                           VK_CULL_MODE_NONE, true, true, reversedZStrict, kBlendNone,
                                           msaaSamples_ != VK_SAMPLE_COUNT_1_BIT &&
                                               debugProperty("debug.bistro.a2c", 1.0f) > 0.5f,
                                           1, true, "Scene Alpha Mask (scene.vert + scene.frag)");
    blendPipeline_ = createGraphicsPipeline(kSceneVert, sizeof(kSceneVert), "scene.vert", kSceneFrag,
                                            sizeof(kSceneFrag), "scene.frag", pipelineLayout_, hdrFormat_, depthFormat_,
                                            msaaSamples_, true, VK_CULL_MODE_NONE, true, false, reversedZ,
                                            kBlendPremultiplied, false, 2, false,
                                            "Scene Transparent (scene.vert + scene.frag)");
    skyPipeline_ = createGraphicsPipeline(kFullscreenVert, sizeof(kFullscreenVert), "fullscreen.vert", kSkyFrag,
                                          sizeof(kSkyFrag), "sky.frag", pipelineLayout_, hdrFormat_, depthFormat_,
                                          msaaSamples_, false, VK_CULL_MODE_NONE, true, false, reversedZ, kBlendNone,
                                          false, 0, false, "Sky (fullscreen.vert + sky.frag)");
    shadowOpaquePipeline_ = createGraphicsPipeline(kShadowVert, sizeof(kShadowVert), "shadow.vert", nullptr, 0, nullptr,
                                                   pipelineLayout_, VK_FORMAT_UNDEFINED, VK_FORMAT_D16_UNORM,
                                                   VK_SAMPLE_COUNT_1_BIT, true, VK_CULL_MODE_NONE, true, true,
                                                   VK_COMPARE_OP_LESS_OR_EQUAL, kBlendNone, false, 0, true,
                                                   "Shadow Opaque (shadow.vert)");
    shadowMaskPipeline_ = createGraphicsPipeline(kShadowVert, sizeof(kShadowVert), "shadow.vert", kShadowFrag,
                                                 sizeof(kShadowFrag), "shadow.frag", pipelineLayout_,
                                                 VK_FORMAT_UNDEFINED, VK_FORMAT_D16_UNORM, VK_SAMPLE_COUNT_1_BIT, true,
                                                 VK_CULL_MODE_NONE, true, true, VK_COMPARE_OP_LESS_OR_EQUAL, kBlendNone,
                                                 false, 0, true, "Shadow Alpha Mask (shadow.vert + shadow.frag)");
    bloomDownPipeline_ = createGraphicsPipeline(kFullscreenVert, sizeof(kFullscreenVert), "fullscreen.vert",
                                                kBloomDownFrag, sizeof(kBloomDownFrag), "bloom_down.frag", postLayout_,
                                                hdrFormat_, VK_FORMAT_UNDEFINED, VK_SAMPLE_COUNT_1_BIT, false,
                                                VK_CULL_MODE_NONE, false, false, VK_COMPARE_OP_ALWAYS, kBlendNone,
                                                false, 0, false,
                                                "Bloom Downsample (fullscreen.vert + bloom_down.frag)");
    taaPipeline_ = createGraphicsPipeline(kFullscreenVert, sizeof(kFullscreenVert), "fullscreen.vert", kTaaFrag,
                                          sizeof(kTaaFrag), "taa.frag", taaLayout_, hdrFormat_, VK_FORMAT_UNDEFINED,
                                          VK_SAMPLE_COUNT_1_BIT, false, VK_CULL_MODE_NONE, false, false,
                                          VK_COMPARE_OP_ALWAYS, kBlendNone, false, 0, false,
                                          "TAA Resolve (fullscreen.vert + taa.frag)");
    bloomUpPipeline_ = createGraphicsPipeline(kFullscreenVert, sizeof(kFullscreenVert), "fullscreen.vert", kBloomUpFrag,
                                              sizeof(kBloomUpFrag), "bloom_up.frag", postLayout_, hdrFormat_,
                                              VK_FORMAT_UNDEFINED, VK_SAMPLE_COUNT_1_BIT, false, VK_CULL_MODE_NONE,
                                              false, false, VK_COMPARE_OP_ALWAYS, kBlendAdditive, false, 0, false,
                                              "Bloom Upsample (fullscreen.vert + bloom_up.frag)");
    return cullPipeline_ && exposurePipeline_ && hizPipeline_ && taaPipeline_ && opaquePipeline_ && maskPipeline_ && blendPipeline_ &&
           skyPipeline_ && shadowOpaquePipeline_ && shadowMaskPipeline_ && bloomDownPipeline_ && bloomUpPipeline_ &&
           aoPipeline_ && aoBlurPipeline_;
}

// ------------------------------------------------------------------------------------------------
// Window / swapchain
// ------------------------------------------------------------------------------------------------

bool Renderer::attachWindow(ANativeWindow* window) {
    if (!initialized_) return false;
    VkAndroidSurfaceCreateInfoKHR sci{VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR};
    sci.window = window;
    if (vkCreateAndroidSurfaceKHR(ctx_.instance, &sci, nullptr, &surface_) != VK_SUCCESS) {
        LOGE("Failed to create surface");
        return false;
    }
    VulkanDebug::setObjectName(ctx_.device, surface_, VK_OBJECT_TYPE_SURFACE_KHR, "Android Window Surface");
    VkBool32 supported = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(ctx_.physicalDevice, ctx_.queueFamily, surface_, &supported);
    if (!supported) {
        LOGE("Queue family cannot present to this surface");
        return false;
    }
    return createSwapchain();
}

void Renderer::detachWindow() {
    if (!ctx_.device) return;
    vkDeviceWaitIdle(ctx_.device);
    destroySwapchain();
    if (surface_) {
        vkDestroySurfaceKHR(ctx_.instance, surface_, nullptr);
        surface_ = VK_NULL_HANDLE;
    }
}

bool Renderer::createSwapchain() {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx_.physicalDevice, surface_, &caps);
    uint32_t count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx_.physicalDevice, surface_, &count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx_.physicalDevice, surface_, &count, formats.data());

    VkSurfaceFormatKHR chosen = formats[0];
    bool found = false;
    for (VkFormat pref : {VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_B8G8R8A8_SRGB}) {
        for (const auto& f : formats) {
            if (f.format == pref && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                chosen = f;
                found = true;
                break;
            }
        }
        if (found) break;
    }
    if (!found) {
        for (const auto& f : formats) {
            if (f.format == VK_FORMAT_R8G8B8A8_UNORM || f.format == VK_FORMAT_B8G8R8A8_UNORM) {
                chosen = f;
                break;
            }
        }
    }
    swapchainFormat_ = chosen.format;
    swapchainNeedsSrgbEncode_ = !(chosen.format == VK_FORMAT_R8G8B8A8_SRGB || chosen.format == VK_FORMAT_B8G8R8A8_SRGB);

    // Pre-rotation: render in the display's native orientation and rotate in our final pass so the
    // compositor does not need to (saves a full-screen rotation blit every frame).
    preTransform_ = caps.currentTransform;
    VkExtent2D extent = caps.currentExtent;
    if (extent.width == 0xFFFFFFFF) extent = {1920, 1080};
    switch (preTransform_) {
        case VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR: preRotation_ = 1; break;
        case VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR: preRotation_ = 2; break;
        case VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR: preRotation_ = 3; break;
        default:
            preRotation_ = 0;
            preTransform_ = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
            break;
    }
    logicalExtent_ = extent;
    swapchainExtent_ = extent;
    if (preRotation_ == 1 || preRotation_ == 3) std::swap(swapchainExtent_.width, swapchainExtent_.height);

    uint32_t imageCount = std::max(3u, caps.minImageCount);
    if (caps.maxImageCount > 0) imageCount = std::min(imageCount, caps.maxImageCount);

    VkCompositeAlphaFlagBitsKHR alpha = (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
                                            ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
                                            : VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = surface_;
    ci.minImageCount = imageCount;
    ci.imageFormat = swapchainFormat_;
    ci.imageColorSpace = chosen.colorSpace;
    ci.imageExtent = swapchainExtent_;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = preTransform_;
    ci.compositeAlpha = alpha;
    ci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    ci.clipped = VK_TRUE;
    VkResult res = vkCreateSwapchainKHR(ctx_.device, &ci, nullptr, &swapchain_);
    if (res != VK_SUCCESS) {
        LOGE("vkCreateSwapchainKHR failed (%d)", res);
        return false;
    }
    VulkanDebug::setObjectName(ctx_.device, swapchain_, VK_OBJECT_TYPE_SWAPCHAIN_KHR, "Main Swapchain");
    vkGetSwapchainImagesKHR(ctx_.device, swapchain_, &count, nullptr);
    swapchainImages_.resize(count);
    vkGetSwapchainImagesKHR(ctx_.device, swapchain_, &count, swapchainImages_.data());
    swapchainViews_.resize(count);
    renderDone_.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        std::string idxStr = " [" + std::to_string(i) + "]";
        VulkanDebug::setObjectName(ctx_.device, swapchainImages_[i], VK_OBJECT_TYPE_IMAGE,
                                   ("Swapchain Image" + idxStr).c_str());
        swapchainViews_[i] = ctx_.createView(swapchainImages_[i], swapchainFormat_, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1,
                                             ("Swapchain Image View" + idxStr).c_str());
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        vkCreateSemaphore(ctx_.device, &sci, nullptr, &renderDone_[i]);
        VulkanDebug::setObjectName(ctx_.device, renderDone_[i], VK_OBJECT_TYPE_SEMAPHORE,
                                   ("Render Complete Semaphore" + idxStr).c_str());
    }
    LOGI("Swapchain %ux%u (logical %ux%u), format %d, pre-rotation %u, %u images", swapchainExtent_.width,
         swapchainExtent_.height, logicalExtent_.width, logicalExtent_.height, swapchainFormat_, preRotation_, count);

    if (pipelineSwapchainFormat_ != swapchainFormat_) {
        if (tonemapPipeline_) vkDestroyPipeline(ctx_.device, tonemapPipeline_, nullptr);
        if (loadingPipeline_) vkDestroyPipeline(ctx_.device, loadingPipeline_, nullptr);
        tonemapPipeline_ = createGraphicsPipeline(kFullscreenVert, sizeof(kFullscreenVert), "fullscreen.vert",
                                                  kTonemapFrag, sizeof(kTonemapFrag), "tonemap.frag", pipelineLayout_,
                                                  swapchainFormat_, VK_FORMAT_UNDEFINED, VK_SAMPLE_COUNT_1_BIT, false,
                                                  VK_CULL_MODE_NONE, false, false, VK_COMPARE_OP_ALWAYS, kBlendNone,
                                                  false, 0, false,
                                                  "Tonemap + Present (fullscreen.vert + tonemap.frag)");
        loadingPipeline_ = createGraphicsPipeline(kFullscreenVert, sizeof(kFullscreenVert), "fullscreen.vert",
                                                  kLoadingFrag, sizeof(kLoadingFrag), "loading.frag", pipelineLayout_,
                                                  swapchainFormat_, VK_FORMAT_UNDEFINED, VK_SAMPLE_COUNT_1_BIT, false,
                                                  VK_CULL_MODE_NONE, false, false, VK_COMPARE_OP_ALWAYS, kBlendNone,
                                                  false, 0, false, "Loading Screen (fullscreen.vert + loading.frag)");
        pipelineSwapchainFormat_ = swapchainFormat_;
    }
    swapchainDirty_ = false;
    return createRenderTargets();
}

void Renderer::destroySwapchain() {
    destroyRenderTargets();
    for (auto v : swapchainViews_) vkDestroyImageView(ctx_.device, v, nullptr);
    for (auto s : renderDone_) vkDestroySemaphore(ctx_.device, s, nullptr);
    swapchainViews_.clear();
    renderDone_.clear();
    swapchainImages_.clear();
    if (swapchain_) vkDestroySwapchainKHR(ctx_.device, swapchain_, nullptr);
    swapchain_ = VK_NULL_HANDLE;
}

bool Renderer::createRenderTargets() {
    renderExtent_.width = std::max(16u, static_cast<uint32_t>(logicalExtent_.width * renderScale_) & ~1u);
    renderExtent_.height = std::max(16u, static_cast<uint32_t>(logicalExtent_.height * renderScale_) & ~1u);

    ImageDesc d;
    d.width = renderExtent_.width;
    d.height = renderExtent_.height;
    d.format = hdrFormat_;
    d.samples = msaaSamples_;
    d.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    d.transient = true;
    d.name = "MSAA HDR Colour";
    if (msaaSamples_ != VK_SAMPLE_COUNT_1_BIT && !ctx_.createImage(d, msaaColor_)) return false;

    d.format = depthFormat_;
    d.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    d.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    d.name = "MSAA Depth";
    if (msaaSamples_ != VK_SAMPLE_COUNT_1_BIT && !ctx_.createImage(d, msaaDepth_)) return false;

    d = ImageDesc{};
    d.width = renderExtent_.width;
    d.height = renderExtent_.height;
    d.format = hdrFormat_;
    d.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    d.name = "HDR Resolve";
    if (!ctx_.createImage(d, hdrResolve_)) return false;

    d.width = std::max(1u, renderExtent_.width / 2);
    d.height = std::max(1u, renderExtent_.height / 2);
    d.mipLevels = kBloomLevels;
    d.name = "Bloom Chain";
    if (!ctx_.createImage(d, bloom_)) return false;
    for (uint32_t i = 0; i < kBloomLevels; ++i) {
        std::string mipViewName = "Bloom Chain View [Mip " + std::to_string(i) + "]";
        bloomViews_[i] = ctx_.createView(bloom_.image, hdrFormat_, VK_IMAGE_ASPECT_COLOR_BIT, i, 1, mipViewName.c_str());
        bloomExtents_[i] = {std::max(1u, d.width >> i), std::max(1u, d.height >> i)};
    }
    // Depth resolve target and Hi-Z pyramid (power-of-two, at most the render resolution).
    d = ImageDesc{};
    d.width = renderExtent_.width;
    d.height = renderExtent_.height;
    d.format = depthFormat_;
    d.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    d.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    d.name = "Scene Depth";
    if (!ctx_.createImage(d, depthResolve_)) return false;

    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        ImageDesc t;
        t.width = renderExtent_.width;
        t.height = renderExtent_.height;
        t.format = hdrFormat_;
        t.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        t.name = i == 0 ? "TAA History [Frame 0]" : "TAA History [Frame 1]";
        if (!ctx_.createImage(t, taaOutput_[i])) return false;
    }
    taaHistoryValid_ = false;

    auto floorPow2 = [](uint32_t v) {
        uint32_t p = 1;
        while (p * 2 <= v) p *= 2;
        return p;
    };
    hizExtent_ = {floorPow2(renderExtent_.width), floorPow2(renderExtent_.height)};
    hizMipCount_ = 1;
    while ((std::max(hizExtent_.width, hizExtent_.height) >> hizMipCount_) > 0 && hizMipCount_ < kMaxHizLevels)
        hizMipCount_++;
    d = ImageDesc{};
    d.width = hizExtent_.width;
    d.height = hizExtent_.height;
    d.mipLevels = hizMipCount_;
    d.format = VK_FORMAT_R32_SFLOAT;
    d.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    d.name = "Hi-Z Pyramid";
    if (!ctx_.createImage(d, hiz_)) return false;
    for (uint32_t i = 0; i < hizMipCount_; ++i) {
        std::string mipViewName = "Hi-Z Pyramid View [Mip " + std::to_string(i) + "]";
        hizViews_[i] =
            ctx_.createView(hiz_.image, VK_FORMAT_R32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, i, 1, mipViewName.c_str());
    }
    {
        VkCommandBuffer cmd = ctx_.beginOneShot("Init Hi-Z & TAA Layouts Command Buffer");
        {
            ScopedCmdLabel label(cmd, "Initialize Hi-Z & TAA Image Layouts", DebugColors::Transition);
            imageBarrier(cmd, hiz_.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            for (uint32_t i = 0; i < kFramesInFlight; ++i) {
                imageBarrier(cmd, taaOutput_[i].image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_NONE, 0,
                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            }
        }
        ctx_.endOneShot(cmd);
    }
    hizValid_ = false;

    // Half-resolution SSAO targets (RG16F: visibility, view depth).
    aoExtent_ = {std::max(1u, (renderExtent_.width + 1) / 2), std::max(1u, (renderExtent_.height + 1) / 2)};
    {
        ImageDesc a;
        a.width = aoExtent_.width;
        a.height = aoExtent_.height;
        a.format = VK_FORMAT_R16G16_SFLOAT;
        a.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        a.name = "SSAO Raw";
        if (!ctx_.createImage(a, aoRaw_)) return false;
        for (uint32_t i = 0; i < kFramesInFlight; ++i) {
            a.name = i == 0 ? "SSAO Filtered [Frame 0]" : "SSAO Filtered [Frame 1]";
            if (!ctx_.createImage(a, ao_[i])) return false;
        }
        VkCommandBuffer cmd = ctx_.beginOneShot("Init SSAO Layouts Command Buffer");
        {
            ScopedCmdLabel label(cmd, "Initialize SSAO Image Layouts", DebugColors::Transition);
            for (GpuImage* img : {&aoRaw_, &ao_[0], &ao_[1]}) {
                imageBarrier(cmd, img->image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE, 0,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            }
        }
        ctx_.endOneShot(cmd);
    }
    aoValid_ = false;

    stats_.renderWidth = renderExtent_.width;
    stats_.renderHeight = renderExtent_.height;
    writeRenderTargetDescriptors();
    LOGI("Render targets %ux%u", renderExtent_.width, renderExtent_.height);
    return true;
}

void Renderer::destroyRenderTargets() {
    for (auto& v : bloomViews_) {
        if (v) vkDestroyImageView(ctx_.device, v, nullptr);
        v = VK_NULL_HANDLE;
    }
    for (auto& v : hizViews_) {
        if (v) vkDestroyImageView(ctx_.device, v, nullptr);
        v = VK_NULL_HANDLE;
    }
    ctx_.destroyImage(hiz_);
    ctx_.destroyImage(aoRaw_);
    for (auto& a : ao_) ctx_.destroyImage(a);
    ctx_.destroyImage(depthResolve_);
    for (auto& t : taaOutput_) ctx_.destroyImage(t);
    ctx_.destroyImage(bloom_);
    ctx_.destroyImage(hdrResolve_);
    ctx_.destroyImage(msaaDepth_);
    ctx_.destroyImage(msaaColor_);
}

// ------------------------------------------------------------------------------------------------
// Descriptors
// ------------------------------------------------------------------------------------------------

void Renderer::writeRenderTargetDescriptors() {
    std::vector<VkWriteDescriptorSet> writes;
    VkDescriptorImageInfo hdrInfo{linearClampSampler_, hdrResolve_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo bloomInfo{linearClampSampler_, bloomViews_[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo hizInfo{nearestSampler_, hiz_.view, VK_IMAGE_LAYOUT_GENERAL};
    // Post-processing in frame slot i reads taaOutput_[i] (written either by the TAA resolve pass or
    // directly by the main pass when TAA is off), while SSR during the main pass reads the previous
    // frame's HDR buffer taaOutput_[(i + 1) % kFramesInFlight].
    std::array<VkDescriptorImageInfo, kFramesInFlight> postInputs{};
    std::array<VkDescriptorImageInfo, kFramesInFlight> prevHdrInputs{};
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        postInputs[i] = {linearClampSampler_, taaOutput_[i].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        prevHdrInputs[i] = {linearClampSampler_, taaOutput_[(i + 1) % kFramesInFlight].view,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    }
    std::array<VkDescriptorImageInfo, 3 * kFramesInFlight> taaInfos{};
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        taaInfos[i * 3 + 0] = hdrInfo;
        taaInfos[i * 3 + 1] = {linearClampSampler_, taaOutput_[(i + 1) % kFramesInFlight].view,
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        taaInfos[i * 3 + 2] = {nearestSampler_, depthResolve_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = taaSets_[i];
        w.dstBinding = 0;
        w.descriptorCount = 3;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &taaInfos[i * 3];
        writes.push_back(w);
    }
    for (uint32_t fi = 0; fi < kFramesInFlight; ++fi) {
        FrameResources& f = frames_[fi];
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = f.set;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.dstBinding = 8;
        w.pImageInfo = &postInputs[fi];
        writes.push_back(w);
        w.dstBinding = 9;
        w.pImageInfo = &bloomInfo;
        writes.push_back(w);
        w.dstBinding = 13;
        w.pImageInfo = &hizInfo;
        writes.push_back(w);
        w.dstBinding = 14;
        w.pImageInfo = &prevHdrInputs[fi];
        writes.push_back(w);
    }
    // SSAO: frame slot i reads the AO written by the previous frame, i.e. slot (i + 1) % 2.
    std::array<VkDescriptorImageInfo, kFramesInFlight> aoPrevInfos{};
    std::array<VkDescriptorImageInfo, kFramesInFlight> aoOutInfos{};
    VkDescriptorImageInfo aoDepthInfo{nearestSampler_, depthResolve_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo aoRawSampled{nearestSampler_, aoRaw_.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo aoRawStorage{VK_NULL_HANDLE, aoRaw_.view, VK_IMAGE_LAYOUT_GENERAL};
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        aoPrevInfos[i] = {linearClampSampler_, ao_[(i + 1) % kFramesInFlight].view, VK_IMAGE_LAYOUT_GENERAL};
        aoOutInfos[i] = {VK_NULL_HANDLE, ao_[i].view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.dstSet = frames_[i].set;
        w.dstBinding = 10;
        w.pImageInfo = &aoPrevInfos[i];
        writes.push_back(w);
        for (uint32_t pass = 0; pass < 2; ++pass) {
            VkDescriptorSet set = aoSets_[i * 2 + pass];
            w.dstSet = set;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.dstBinding = 0;
            w.pImageInfo = &aoDepthInfo;
            writes.push_back(w);
            w.dstBinding = 1;
            w.pImageInfo = &aoRawSampled;
            writes.push_back(w);
            w.dstBinding = 2;
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w.pImageInfo = pass == 0 ? &aoRawStorage : &aoOutInfos[i];
            writes.push_back(w);
        }
    }
    std::array<VkDescriptorImageInfo, kMaxHizLevels> hizSrc{};
    std::array<VkDescriptorImageInfo, kMaxHizLevels> hizDst{};
    for (uint32_t i = 0; i < hizMipCount_; ++i) {
        hizSrc[i] = i == 0 ? VkDescriptorImageInfo{nearestSampler_, depthResolve_.view,
                                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}
                           : VkDescriptorImageInfo{nearestSampler_, hizViews_[i - 1], VK_IMAGE_LAYOUT_GENERAL};
        hizDst[i] = {VK_NULL_HANDLE, hizViews_[i], VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = hizSets_[i];
        w.descriptorCount = 1;
        w.dstBinding = 0;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &hizSrc[i];
        writes.push_back(w);
        w.dstBinding = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w.pImageInfo = &hizDst[i];
        writes.push_back(w);
    }
    std::array<VkDescriptorImageInfo, kBloomLevels + 2> srcInfos{};
    srcInfos[0] = postInputs[0];
    srcInfos[1] = postInputs[1];
    for (uint32_t i = 0; i < kBloomLevels; ++i)
        srcInfos[i + 2] = {linearClampSampler_, bloomViews_[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    for (uint32_t i = 0; i < kBloomLevels + 2; ++i) {
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = bloomSourceSets_[i];
        w.dstBinding = 0;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &srcInfos[i];
        writes.push_back(w);
    }
    vkUpdateDescriptorSets(ctx_.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void Renderer::writeStaticDescriptors() {
    const uint32_t capacity = static_cast<uint32_t>(scene_.clusters.size());
    std::vector<VkDescriptorImageInfo> texInfos(scene_.textures.size());
    VkImageView fallbackView = scene_.textures.size() >= 4 ? scene_.textures[scene_.textures.size() - 4].view : VK_NULL_HANDLE;
    for (size_t i = 0; i < scene_.textures.size(); ++i) {
        VkImageView view = scene_.textures[i].view ? scene_.textures[i].view : fallbackView;
        texInfos[i] = {materialSampler_, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    }
    VkDescriptorImageInfo shadowInfo{shadowSampler_, shadowMap_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo skyInfo{envSampler_, scene_.sky.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo specInfo{envSampler_, scene_.envSpecular.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorBufferInfo clusterInfo{scene_.clusterBuffer.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo materialInfo{scene_.materialBuffer.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo exposureInfo{exposureBuffer_.buffer, 0, VK_WHOLE_SIZE};

    for (uint32_t fi = 0; fi < kFramesInFlight; ++fi) {
        FrameResources& f = frames_[fi];
        if (!f.draws.buffer) {
            std::string drawsName = currentAssetDir_ + "/scene.bin Indirect Draws [Frame " + std::to_string(fi) + "]";
            ctx_.createBuffer(static_cast<VkDeviceSize>(capacity) * kBucketCount * sizeof(DrawCommand),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, f.draws, drawsName.c_str());
        }
        VkDescriptorBufferInfo uboInfo{f.uniforms.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo drawInfo{f.draws.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo countInfo{f.counts.buffer, 0, VK_WHOLE_SIZE};
        std::vector<VkWriteDescriptorSet> writes;
        auto buf = [&](uint32_t binding, VkDescriptorType type, const VkDescriptorBufferInfo* info) {
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = f.set;
            w.dstBinding = binding;
            w.descriptorCount = 1;
            w.descriptorType = type;
            w.pBufferInfo = info;
            writes.push_back(w);
        };
        auto img = [&](uint32_t binding, const VkDescriptorImageInfo* info, uint32_t n = 1) {
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = f.set;
            w.dstBinding = binding;
            w.descriptorCount = n;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo = info;
            writes.push_back(w);
        };
        buf(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &uboInfo);
        buf(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &clusterInfo);
        buf(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &materialInfo);
        buf(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &drawInfo);
        buf(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &countInfo);
        img(5, &shadowInfo);
        img(6, &skyInfo);
        img(7, &specInfo);
        buf(11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &exposureInfo);
        img(15, texInfos.data(), static_cast<uint32_t>(texInfos.size()));
        vkUpdateDescriptorSets(ctx_.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
}

// ------------------------------------------------------------------------------------------------
// Camera-fitted directional sun shadow map
// ------------------------------------------------------------------------------------------------

bool Renderer::buildShadowMap() {
    // Lighting set-up. The sun direction comes from the HDR environment map (so shadows match the sky);
    // the map is rotated so the sun's azimuth matches the directional light authored in the FBX.
    const EnvironmentInfo& env = scene_.environment;
    const RenderConfig& rc = scene_.renderConfig;
    Vec3 fbxSun{scene_.header.sunDirection[0], scene_.header.sunDirection[1], scene_.header.sunDirection[2]};
    float targetAzimuth = std::atan2(fbxSun.z, fbxSun.x);
    float envAzimuth = std::atan2(env.sunDirection.z, env.sunDirection.x);
    envYaw_ = targetAzimuth - envAzimuth + rc.sunAzimuthOffset;
    sunDirection_ = normalize(envToWorld(env.sunDirection, envYaw_));
    sunColor_ = env.sunIrradiance * rc.sunIntensityScale;
    pointLights_.clear();
    for (const auto& pl : scene_.pointLights) {
        if (pointLights_.size() / 2 >= kMaxPointLights) break;
        pointLights_.push_back(Vec4{pl.position[0], pl.position[1], pl.position[2], pl.range});
        float lightScale = rc.pointLightScale;
        pointLights_.push_back(Vec4{pl.color[0] * lightScale, pl.color[1] * lightScale, pl.color[2] * lightScale, 0});
    }
    LOGI("Sun direction (%.3f %.3f %.3f), env yaw %.2f rad", sunDirection_.x, sunDirection_.y, sunDirection_.z,
         envYaw_);

    // Precompute the fixed light-space view matrix and per-cluster light-space AABBs so the shadow map
    // orthographic projection can be tightly fitted around the active camera frustum without swimming.
    Vec3 up = std::fabs(sunDirection_.y) > 0.99f ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    Vec3 center{(scene_.header.boundsMin[0] + scene_.header.boundsMax[0]) * 0.5f,
                (scene_.header.boundsMin[1] + scene_.header.boundsMax[1]) * 0.5f,
                (scene_.header.boundsMin[2] + scene_.header.boundsMax[2]) * 0.5f};
    lightView_ = Mat4::lookAt(center + sunDirection_ * 200.0f, center, up);

    clusterLightBounds_.resize(scene_.clusters.size());
    sceneLightMin_[0] = sceneLightMin_[1] = sceneLightMin_[2] = 1e30f;
    sceneLightMax_[0] = sceneLightMax_[1] = sceneLightMax_[2] = -1e30f;
    for (size_t i = 0; i < scene_.clusters.size(); ++i) {
        const auto& c = scene_.clusters[i];
        float mn[3] = {1e30f, 1e30f, 1e30f}, mx[3] = {-1e30f, -1e30f, -1e30f};
        for (int k = 0; k < 8; ++k) {
            Vec4 p{(k & 1) ? c.aabbMax[0] : c.aabbMin[0], (k & 2) ? c.aabbMax[1] : c.aabbMin[1],
                   (k & 4) ? c.aabbMax[2] : c.aabbMin[2], 1.0f};
            Vec4 v = lightView_ * p;
            mn[0] = std::min(mn[0], v.x); mx[0] = std::max(mx[0], v.x);
            mn[1] = std::min(mn[1], v.y); mx[1] = std::max(mx[1], v.y);
            mn[2] = std::min(mn[2], v.z); mx[2] = std::max(mx[2], v.z);
        }
        for (int a = 0; a < 3; ++a) {
            clusterLightBounds_[i].mn[a] = mn[a];
            clusterLightBounds_[i].mx[a] = mx[a];
            sceneLightMin_[a] = std::min(sceneLightMin_[a], mn[a]);
            sceneLightMax_[a] = std::max(sceneLightMax_[a], mx[a]);
        }
    }

    std::string shadowMapName = currentAssetDir_ + " Camera-Fitted Sun Shadow Map";
    ImageDesc d;
    d.width = d.height = shadowSize_;
    d.format = VK_FORMAT_D16_UNORM;
    d.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    d.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    d.name = shadowMapName.c_str();
    if (!ctx_.createImage(d, shadowMap_)) return false;

    std::string shadowDrawsName = currentAssetDir_ + "/scene.bin Shadow Indirect Draws";
    const size_t drawCapacity = std::max<size_t>(1, scene_.clusters.size()) * kFramesInFlight;
    if (!ctx_.createBuffer(drawCapacity * sizeof(DrawCommand), VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, shadowDraws_,
                           shadowDrawsName.c_str())) {
        return false;
    }
    ctx_.flushUploads();

    writeStaticDescriptors();

    VkCommandBuffer cmd = ctx_.beginOneShot("Init Shadow Map Layout Command Buffer");
    {
        ScopedCmdLabel label(cmd, "Initialize Shadow Map Image Layout", DebugColors::Transition);
        imageBarrier(cmd, shadowMap_.image, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_NONE, 0,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }
    ctx_.endOneShot(cmd);

    shadowFitted_ = false;
    shadowDirty_ = true;
    shadowReady_ = true;
    return true;
}

void Renderer::updateShadowFit(const CameraPath::Pose& pose) {
    if (scene_.clusters.empty() || !shadowDraws_.mapped) return;

    Vec3 fwd = normalize(pose.target - pose.eye);
    const float focusDist = pose.indoor ? 10.0f : 24.0f;
    Vec3 focusWorld = pose.eye + fwd * focusDist;
    Vec4 focusLight = lightView_ * Vec4{focusWorld.x, focusWorld.y, focusWorld.z, 1.0f};

    if (!cameraOverride_.enabled) {
        // Fit once per camera loop so shadowViewProj_ is 100% invariant during the entire fly-through
        // (zero mid-flight refit jumps, zero caster popping, and zero shadow-border snapping).
        if (shadowFitted_ && pose.segment == shadowSegment_ && !shadowOverride_) {
            return;
        }
    } else {
        const float moved = std::hypot(focusLight.x - shadowCenterX_, focusLight.y - shadowCenterY_);
        if (shadowFitted_ && shadowOverride_ && pose.indoor == (shadowSegment_ == 1) && moved <= 2.0f) {
            return;
        }
    }

    float mnX = sceneLightMin_[0];
    float mxX = sceneLightMax_[0];
    float mnY = sceneLightMin_[1];
    float mxY = sceneLightMax_[1];

    float loopMin[2] = {0.0f, 0.0f};
    float loopMax[2] = {0.0f, 0.0f};
    const float lookAheadDist = pose.indoor ? 15.0f : 45.0f;
    const float pad = pose.indoor ? 16.0f : 40.0f;

    if (!cameraOverride_.enabled && path_.loopLightBounds(pose.segment, lightView_, lookAheadDist, loopMin, loopMax)) {
        mnX = std::max(sceneLightMin_[0], loopMin[0] - pad);
        mxX = std::min(sceneLightMax_[0], loopMax[0] + pad);
        mnY = std::max(sceneLightMin_[1], loopMin[1] - pad);
        mxY = std::min(sceneLightMax_[1], loopMax[1] + pad);
    } else {
        const float halfSpan = pose.indoor ? 24.0f : 68.0f;
        mnX = std::max(sceneLightMin_[0], focusLight.x - halfSpan);
        mxX = std::min(sceneLightMax_[0], focusLight.x + halfSpan);
        mnY = std::max(sceneLightMin_[1], focusLight.y - halfSpan);
        mxY = std::min(sceneLightMax_[1], focusLight.y + halfSpan);
    }

    if (mxX - mnX < 8.0f) {
        float c = 0.5f * (mnX + mxX);
        mnX = c - 4.0f;
        mxX = c + 4.0f;
    }
    if (mxY - mnY < 8.0f) {
        float c = 0.5f * (mnY + mxY);
        mnY = c - 4.0f;
        mxY = c + 4.0f;
    }

    const float spanX = mxX - mnX;
    const float spanY = mxY - mnY;
    const float worldTexel = std::max(spanX, spanY) / static_cast<float>(shadowSize_);

    shadowFitted_ = true;
    shadowSegment_ = cameraOverride_.enabled ? (pose.indoor ? 1 : 0) : pose.segment;
    shadowOverride_ = cameraOverride_.enabled;
    shadowCenterX_ = focusLight.x;
    shadowCenterY_ = focusLight.y;

    float mnZ = 1e30f;
    float mxZ = -1e30f;
    const size_t capacity = scene_.clusters.size();
    auto* dst = static_cast<DrawCommand*>(shadowDraws_.mapped) + static_cast<size_t>(frameIndex_) * capacity;
    uint32_t count = 0;

    for (uint32_t pass = 0; pass < 2; ++pass) {
        const uint32_t targetBucket = (pass == 0) ? kBucketOpaque : kBucketMask;
        for (uint32_t i = 0; i < capacity; ++i) {
            const auto& c = scene_.clusters[i];
            if (c.bucket != targetBucket) continue;
            const auto& b = clusterLightBounds_[i];
            if (b.mx[0] < mnX || b.mn[0] > mxX || b.mx[1] < mnY || b.mn[1] > mxY) continue;
            mnZ = std::min(mnZ, b.mn[2]);
            mxZ = std::max(mxZ, b.mx[2]);
            dst[count++] = {c.indexCount, 1, c.firstIndex, c.vertexOffset, i};
        }
        if (pass == 0) {
            shadowOpaqueCount_ = count;
        } else {
            shadowMaskCount_ = count - shadowOpaqueCount_;
        }
    }
    if (mnZ > mxZ) {
        mnZ = sceneLightMin_[2];
        mxZ = sceneLightMax_[2];
    } else {
        mnZ -= 4.0f;
        mxZ += 4.0f;
    }

    Mat4 lightProj = Mat4::orthographic(mnX, mxX, mnY, mxY, -mxZ - 1.0f, -mnZ + 1.0f);
    shadowViewProj_ = lightProj * lightView_;
    shadowWorldTexel_ = worldTexel;
    shadowDirty_ = true;
    LOGI("Loop-fitted shadow map (seg %d): %.1f x %.1f m (%.2f cm texels, depth %.1f m, %u opaque + %u masked clusters)",
         pose.segment, spanX, spanY, shadowWorldTexel_ * 100.0f, mxZ - mnZ, shadowOpaqueCount_, shadowMaskCount_);
}

// ------------------------------------------------------------------------------------------------
// Per-frame
// ------------------------------------------------------------------------------------------------

void Renderer::updateUniforms(FrameResources& frame, double time, double dt) {
    CameraPath::Pose pose = path_.evaluate(time);
    if (cameraOverride_.enabled) {
        pose.eye = cameraOverride_.eye;
        pose.target = cameraOverride_.target;
    } else if (pose.segment != lastCameraSegment_) {
        const std::string& segName = path_.loopName(pose.segment);
        if (lastCameraSegment_ >= 0) {
            LOGI("Switching camera to %s fly-through", segName.c_str());
        }
        lastCameraSegment_ = pose.segment;
        invalidateHistory();
    }
    updateShadowFit(pose);
    stats_.cameraPos = pose.eye;
    stats_.cameraSegment = pose.segment;
    stats_.cameraSegmentName = path_.loopName(pose.segment);
    float aspect = static_cast<float>(logicalExtent_.width) / static_cast<float>(logicalExtent_.height);
    Mat4 view = Mat4::lookAt(pose.eye, pose.target, Vec3{0, 1, 0});
    Mat4 proj = Mat4::perspectiveReversedInfinite(kFovY, aspect, kNearPlane);
    Mat4 viewProj = proj * view;

    // Sub-pixel jitter (Halton 2,3) for temporal anti-aliasing.
    float jitter[2] = {0.0f, 0.0f};
    if (taaEnabled_) {
        auto halton = [](uint32_t index, uint32_t base) {
            float f = 1.0f, r = 0.0f;
            for (uint32_t i = index; i > 0; i /= base) {
                f /= static_cast<float>(base);
                r += f * static_cast<float>(i % base);
            }
            return r;
        };
        jitterIndex_ = (jitterIndex_ % 8) + 1;
        jitter[0] = (halton(jitterIndex_, 2) - 0.5f) * 2.0f / static_cast<float>(renderExtent_.width);
        jitter[1] = (halton(jitterIndex_, 3) - 0.5f) * 2.0f / static_cast<float>(renderExtent_.height);
    }
    Mat4 jitterMat = Mat4::identity();
    jitterMat.at(0, 3) = jitter[0];
    jitterMat.at(1, 3) = jitter[1];
    Mat4 viewProjJittered = jitterMat * viewProj;

    FrameUniforms u{};
    if (!taaHistoryValid_ && !hizValid_) {
        prevViewProjJittered_ = viewProjJittered;
        prevViewProjUnjittered_ = viewProj;
    }
    u.viewProj = viewProjJittered;
    u.invViewProj = viewProjJittered.inverse();
    u.shadowViewProj = shadowViewProj_;
    u.prevViewProj = prevViewProjJittered_;
    u.reprojection = prevViewProjUnjittered_ * viewProj.inverse();
    prevViewProjJittered_ = viewProjJittered;
    prevViewProjUnjittered_ = viewProj;
    u.taaParams[0] = jitter[0];
    u.taaParams[1] = jitter[1];
    u.taaParams[2] = 0.9f;  // history feedback
    u.taaParams[3] = taaHistoryValid_ ? 1.0f : 0.0f;
    u.cameraPos[0] = pose.eye.x;
    u.cameraPos[1] = pose.eye.y;
    u.cameraPos[2] = pose.eye.z;
    u.cameraPos[3] = static_cast<float>(std::fmod(time, 10000.0));
    u.sunDirection[0] = sunDirection_.x;
    u.sunDirection[1] = sunDirection_.y;
    u.sunDirection[2] = sunDirection_.z;
    u.sunColor[0] = sunColor_.x;
    u.sunColor[1] = sunColor_.y;
    u.sunColor[2] = sunColor_.z;
    u.sunColor[3] = std::cos(0.0047f);  // angular radius of the sun
    u.envRotation[0] = std::cos(envYaw_);
    u.envRotation[1] = std::sin(envYaw_);
    u.envRotation[2] = skyDebug_ ? -1.0f : 1.0f;  // sky intensity (negative = debug highlight)
    u.envRotation[3] = 1.0f;  // IBL intensity
    for (int i = 0; i < 9; ++i)
        for (int c = 0; c < 3; ++c) u.sh[i][c] = scene_.environment.sh[i][c];
    extractFrustumPlanes(viewProj, u.frustumPlanes);

    const RenderConfig& rc = scene_.renderConfig;
    // Fog colour: configured override, or average sky radiance (SH band 0) tinted slightly warm by the sun.
    float avg[3];
    for (int c = 0; c < 3; ++c) avg[c] = scene_.environment.sh[0][c] * 0.282095f / 3.14159265f;
    u.fogParams[0] = rc.fogDensity;        // density per metre
    u.fogParams[1] = rc.fogHeightFalloff;  // height falloff
    u.fogParams[2] = rc.fogBaseHeight;     // base height
    u.fogParams[3] = rc.fogMaxOpacity;     // max opacity
    if (rc.fogColor.x > 0.0f || rc.fogColor.y > 0.0f || rc.fogColor.z > 0.0f) {
        u.fogColor[0] = rc.fogColor.x;
        u.fogColor[1] = rc.fogColor.y;
        u.fogColor[2] = rc.fogColor.z;
    } else {
        u.fogColor[0] = avg[0] * 1.05f;
        u.fogColor[1] = avg[1] * 1.0f;
        u.fogColor[2] = avg[2] * 0.95f;
    }

    u.renderParams[0] = 1.0f;                             // manual exposure multiplier (on top of auto exposure)
    u.renderParams[1] = rc.bloomStrength * bloomScale_;   // bloom strength
    u.renderParams[2] = static_cast<float>(renderExtent_.width);
    u.renderParams[3] = static_cast<float>(renderExtent_.height);
    u.cullParams[0] = static_cast<float>(scene_.clusters.size());
    u.cullParams[1] = static_cast<float>(scene_.clusters.size());
    u.cullParams[2] = static_cast<float>(renderExtent_.height) / (2.0f * std::tan(kFovY * 0.5f));
    u.cullParams[3] = minClusterPixels_;  // minimum projected radius in pixels
    u.postParams[0] = static_cast<float>(preRotation_);
    u.postParams[1] = static_cast<float>(swapchainExtent_.width);
    u.postParams[2] = static_cast<float>(swapchainExtent_.height);
    u.postParams[3] = rc.emissiveIntensity;  // emissive intensity
    u.shadowParams[0] = 1.0f / static_cast<float>(shadowSize_);
    u.shadowParams[1] = shadowWorldTexel_ * 1.35f;
    u.shadowParams[2] = 0.00003f;
    u.shadowParams[3] = 1.8f;
    u.extraParams[0] = swapchainNeedsSrgbEncode_ ? 1.0f : 0.0f;
    u.extraParams[1] = 0.25f;  // vignette
    u.extraParams[2] = 1.08f;  // saturation
    u.extraParams[3] = 1.04f;  // contrast
    u.exposureParams[0] = static_cast<float>(dt);
    u.exposureParams[1] = rc.exposureKey;  // key value
    u.exposureParams[2] = rc.minExposure;
    u.exposureParams[3] = rc.maxExposure;
    u.exposureParams2[0] = 1.2f;  // adaptation speed
    u.exposureParams2[1] = 1.0f;  // auto exposure on
    u.exposureParams2[2] = pose.indoor ? 1.0f : 0.0f;
    u.exposureParams2[3] = rc.translucency;
    u.occlusionParams[0] = static_cast<float>(hizExtent_.width);
    u.occlusionParams[1] = static_cast<float>(hizExtent_.height);
    u.occlusionParams[2] = static_cast<float>(hizMipCount_);
    u.occlusionParams[3] = (occlusionCulling_ && hizValid_) ? 1.0f : 0.0f;
    float effAoIntensity = rc.aoIntensity * aoIntensityScale_;
    u.aoParams[0] = (aoEnabled_ && aoValid_ && effAoIntensity > 0.0f) ? 1.0f : 0.0f;
    u.aoParams[1] = rc.aoRadius;
    u.aoParams[2] = effAoIntensity;
    u.aoParams[3] = aoDebug_ ? 1.0f : 0.0f;
    uint32_t lightCount = static_cast<uint32_t>(pointLights_.size() / 2);
    for (uint32_t i = 0; i < lightCount; ++i) {
        memcpy(&u.lights[i][0], &pointLights_[i * 2], sizeof(Vec4));
        memcpy(&u.lights[i][4], &pointLights_[i * 2 + 1], sizeof(Vec4));
    }
    uint32_t featureFlags = (ssrEnabled_ ? 1u : 0u) | (contactShadowsEnabled_ ? 2u : 0u) |
                            (volumetricFogEnabled_ ? 4u : 0u) | (shadowPcfEnabled_ ? 8u : 0u);
    u.lightCount[0] = lightCount;
    u.lightCount[1] = featureFlags;
    memcpy(frame.uniforms.mapped, &u, sizeof(u));
}

void Renderer::readBackStats(FrameResources& frame, uint32_t frameIndex) {
    const auto* counts = static_cast<const uint32_t*>(frame.counts.mapped);
    for (int i = 0; i < 3; ++i) stats_.visibleClusters[i] = counts[i];
    stats_.visibleTriangles = counts[3];
    stats_.taa = taaEnabled_;
    stats_.shadowSize = shadowSize_;
    stats_.totalClusters = static_cast<uint32_t>(scene_.clusters.size());
    stats_.totalTriangles = scene_.triangleCount;
    if (frame.timestampsValid && timestampPool_) {
        uint64_t ts[2];
        if (vkGetQueryPoolResults(ctx_.device, timestampPool_, frameIndex * 2, 2, sizeof(ts), ts, sizeof(uint64_t),
                                  VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            float ms = static_cast<float>(ts[1] - ts[0]) * ctx_.properties.limits.timestampPeriod * 1e-6f;
            stats_.gpuMs = stats_.gpuMs * 0.9f + ms * 0.1f;
        }
    }
    stats_.exposure = static_cast<const float*>(exposureBuffer_.mapped)[0];
}

void Renderer::recordLoading(VkCommandBuffer cmd, uint32_t imageIndex, double time) {
    ScopedCmdLabel label(cmd, "Loading Screen (fullscreen.vert + loading.frag)", DebugColors::RenderPass);
    VkImage image = swapchainImages_[imageIndex];
    imageBarrier(cmd, image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    color.imageView = swapchainViews_[imageIndex];
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, swapchainExtent_};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &color;
    vkCmdBeginRendering(cmd, &ri);
    VkViewport vp{0, 0, static_cast<float>(swapchainExtent_.width), static_cast<float>(swapchainExtent_.height), 0, 1};
    VkRect2D sc{{0, 0}, swapchainExtent_};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, loadingPipeline_);
    LoadingPush push{};
    push.outputSize[0] = static_cast<float>(swapchainExtent_.width);
    push.outputSize[1] = static_cast<float>(swapchainExtent_.height);
    push.progress = scene_.progress();
    push.rotation = static_cast<float>(preRotation_);
    push.time = static_cast<float>(std::fmod(time, 1000.0));
    push.srgbEncode = swapchainNeedsSrgbEncode_ ? 1.0f : 0.0f;
    vkCmdPushConstants(cmd, pipelineLayout_, kAllStages, 0, sizeof(push), &push);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRendering(cmd);
    imageBarrier(cmd, image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_NONE, 0);
}

void Renderer::recordScene(VkCommandBuffer cmd, FrameResources& frame, uint32_t imageIndex) {
    const uint32_t capacity = static_cast<uint32_t>(scene_.clusters.size());
    const VkDeviceSize bucketStride = static_cast<VkDeviceSize>(capacity) * sizeof(DrawCommand);
    VkViewport renderVp{0, 0, static_cast<float>(renderExtent_.width), static_cast<float>(renderExtent_.height), 0, 1};
    VkRect2D renderSc{{0, 0}, renderExtent_};

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1, &frame.set, 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_, 0, 1, &frame.set, 0, nullptr);

    // 0. Camera-fitted directional shadow pass (recorded when the camera moves beyond the guard band).
    if (shadowDirty_) {
        shadowDirty_ = false;
        ScopedCmdLabel label(cmd, "Camera-Fitted Shadow Map (shadow.vert + shadow.frag)", DebugColors::RenderPass);
        imageBarrier(cmd, shadowMap_.image, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                     VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
        VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depth.imageView = shadowMap_.view;
        depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth.clearValue.depthStencil = {1.0f, 0};
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = {{0, 0}, {shadowSize_, shadowSize_}};
        ri.layerCount = 1;
        ri.pDepthAttachment = &depth;
        vkCmdBeginRendering(cmd, &ri);
        VkViewport vp{0, 0, static_cast<float>(shadowSize_), static_cast<float>(shadowSize_), 0, 1};
        VkRect2D sc{{0, 0}, {shadowSize_, shadowSize_}};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdPushConstants(cmd, pipelineLayout_, kAllStages, 0, sizeof(Mat4), &shadowViewProj_);
        VkDeviceSize vbOffset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &scene_.vertexBuffer.buffer, &vbOffset);
        vkCmdBindIndexBuffer(cmd, scene_.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);
        const VkDeviceSize slotOffset = static_cast<VkDeviceSize>(frameIndex_) * capacity * sizeof(DrawCommand);
        if (shadowOpaqueCount_ > 0) {
            ScopedCmdLabel l(cmd, "Shadow Opaque (shadow.vert)", DebugColors::Draw);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowOpaquePipeline_);
            vkCmdDrawIndexedIndirect(cmd, shadowDraws_.buffer, slotOffset, shadowOpaqueCount_, sizeof(DrawCommand));
        }
        if (shadowMaskCount_ > 0) {
            ScopedCmdLabel l(cmd, "Shadow Alpha Mask (shadow.vert + shadow.frag)", DebugColors::Draw);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowMaskPipeline_);
            vkCmdDrawIndexedIndirect(cmd, shadowDraws_.buffer,
                                     slotOffset + shadowOpaqueCount_ * sizeof(DrawCommand), shadowMaskCount_,
                                     sizeof(DrawCommand));
        }
        vkCmdEndRendering(cmd);
        imageBarrier(cmd, shadowMap_.image, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }

    // 1. GPU culling -> indirect draw lists.
    {
        ScopedCmdLabel label(cmd, "GPU Cluster Culling (cull.comp)", DebugColors::Transfer);
        vkCmdFillBuffer(cmd, frame.counts.buffer, 0, 16, 0);
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                      VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cullPipeline_);
        vkCmdDispatch(cmd, 1, 1, 1);
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
    }

    // 2. Main HDR pass (jittered for TAA, or 4x MSAA resolved on-tile).
    {
        ScopedCmdLabel label(cmd, "Main HDR Pass", DebugColors::RenderPass);
        if (msaaColor_.image) {
            imageBarrier(cmd, msaaColor_.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        }
        if (msaaDepth_.image) {
            imageBarrier(cmd, msaaDepth_.image, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                         VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                         VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                         VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                             VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
        }
        GpuImage& mainColorTarget = taaEnabled_ ? hdrResolve_ : taaOutput_[frameIndex_];
        imageBarrier(cmd, mainColorTarget.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        imageBarrier(cmd, depthResolve_.image, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                         VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                         VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

        VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        color.clearValue.color = {{0, 0, 0, 0}};
        color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        if (msaaColor_.image) {
            color.imageView = msaaColor_.view;
            color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            color.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            color.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
            color.resolveImageView = mainColorTarget.view;
            color.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        } else {
            color.imageView = mainColorTarget.view;
            color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        }
        VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depth.imageView = msaaDepth_.image ? msaaDepth_.view : depthResolve_.view;
        depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        // Without MSAA the depth is kept for Hi-Z and TAA reprojection.
        depth.storeOp = msaaDepth_.image ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE;
        depth.clearValue.depthStencil = {0.0f, 0};  // reversed Z
        if (msaaDepth_.image && buildHiz_) {
            // MIN resolve keeps the farthest sample: a conservative depth for occlusion culling.
            depth.resolveMode = VK_RESOLVE_MODE_MIN_BIT;
            depth.resolveImageView = depthResolve_.view;
            depth.resolveImageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        }
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = renderSc;
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &color;
        ri.pDepthAttachment = &depth;
        vkCmdBeginRendering(cmd, &ri);
        vkCmdSetViewport(cmd, 0, 1, &renderVp);
        vkCmdSetScissor(cmd, 0, 1, &renderSc);
        VkDeviceSize vbOffset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &scene_.vertexBuffer.buffer, &vbOffset);
        vkCmdBindIndexBuffer(cmd, scene_.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);

        {
            ScopedCmdLabel l(cmd, "Opaque (scene.vert + scene.frag)", DebugColors::Draw);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, opaquePipeline_);
            vkCmdDrawIndexedIndirectCount(cmd, frame.draws.buffer, 0, frame.counts.buffer, 0, capacity,
                                          sizeof(DrawCommand));
        }
        {
            ScopedCmdLabel l(cmd, "Alpha Tested (scene.vert + scene.frag)", DebugColors::Draw);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, maskPipeline_);
            vkCmdDrawIndexedIndirectCount(cmd, frame.draws.buffer, bucketStride, frame.counts.buffer, 4, capacity,
                                          sizeof(DrawCommand));
        }
        {
            ScopedCmdLabel l(cmd, "Sky (fullscreen.vert + sky.frag)", DebugColors::Draw);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyPipeline_);
            vkCmdDraw(cmd, 3, 1, 0, 0);
        }
        {
            ScopedCmdLabel l(cmd, "Transparent (scene.vert + scene.frag)", DebugColors::Draw);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, blendPipeline_);
            vkCmdDrawIndexedIndirectCount(cmd, frame.draws.buffer, bucketStride * 2, frame.counts.buffer, 8,
                                          capacity, sizeof(DrawCommand));
        }
        vkCmdEndRendering(cmd);
        imageBarrier(cmd, mainColorTarget.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        if (!taaEnabled_) {
            taaHistoryValid_ = true;
        }
        imageBarrier(cmd, depthResolve_.image, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }

    // 3. Hi-Z pyramid from this frame's depth, used to occlusion-cull next frame.
    if (buildHiz_) {
        ScopedCmdLabel label(cmd, "Hi-Z Pyramid Build (hiz.comp)", DebugColors::Transfer);
        // This frame's culling has read the pyramid; make the rewrite wait for it.
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hizPipeline_);
        for (uint32_t i = 0; i < hizMipCount_; ++i) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hizLayout_, 1, 1, &hizSets_[i], 0, nullptr);
            int32_t push[4];
            if (i == 0) {
                push[0] = static_cast<int32_t>(renderExtent_.width);
                push[1] = static_cast<int32_t>(renderExtent_.height);
            } else {
                push[0] = static_cast<int32_t>(std::max(1u, hizExtent_.width >> (i - 1)));
                push[1] = static_cast<int32_t>(std::max(1u, hizExtent_.height >> (i - 1)));
            }
            push[2] = static_cast<int32_t>(std::max(1u, hizExtent_.width >> i));
            push[3] = static_cast<int32_t>(std::max(1u, hizExtent_.height >> i));
            vkCmdPushConstants(cmd, hizLayout_, kAllStages, 0, sizeof(push), push);
            vkCmdDispatch(cmd, (push[2] + 7) / 8, (push[3] + 7) / 8, 1);
            memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
        hizValid_ = true;
        // Re-bind the global set for the compute work that follows.
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1, &frame.set, 0, nullptr);
    }

    // 4. Screen-space AO from this frame's depth (consumed, reprojected, by the next frame's shading).
    if (aoEnabled_) {
        ScopedCmdLabel label(cmd, "Screen-Space AO (ao.comp + ao_blur.comp)", DebugColors::Transfer);
        // aoRaw_ was read by the previous frame's blur and ao_[slot] by the frame before last's shading.
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                      VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        int32_t push[4] = {static_cast<int32_t>(renderExtent_.width), static_cast<int32_t>(renderExtent_.height),
                           static_cast<int32_t>(aoExtent_.width), static_cast<int32_t>(aoExtent_.height)};
        const uint32_t gx = (aoExtent_.width + 7) / 8, gy = (aoExtent_.height + 7) / 8;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, aoLayout_, 0, 1, &frame.set, 0, nullptr);
        vkCmdPushConstants(cmd, aoLayout_, kAllStages, 0, sizeof(push), push);
        {
            ScopedCmdLabel l(cmd, "SSAO Compute (ao.comp)", DebugColors::Transfer);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, aoPipeline_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, aoLayout_, 1, 1, &aoSets_[frameIndex_ * 2], 0,
                                    nullptr);
            vkCmdDispatch(cmd, gx, gy, 1);
        }
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        {
            ScopedCmdLabel l(cmd, "SSAO Blur (ao_blur.comp)", DebugColors::Transfer);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, aoBlurPipeline_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, aoLayout_, 1, 1, &aoSets_[frameIndex_ * 2 + 1],
                                    0, nullptr);
            vkCmdDispatch(cmd, gx, gy, 1);
        }
        // The next frame's scene shading samples the result.
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        aoValid_ = true;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1, &frame.set, 0, nullptr);
    }

    // 5. Temporal anti-aliasing resolve into this frame slot's history image.
    if (taaEnabled_) {
        ScopedCmdLabel label(cmd, "TAA Resolve (fullscreen.vert + taa.frag)", DebugColors::RenderPass);
        GpuImage& out = taaOutput_[frameIndex_];
        imageBarrier(cmd, out.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        VkRenderingAttachmentInfo a{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        a.imageView = out.view;
        a.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        a.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingInfo r{VK_STRUCTURE_TYPE_RENDERING_INFO};
        r.renderArea = renderSc;
        r.layerCount = 1;
        r.colorAttachmentCount = 1;
        r.pColorAttachments = &a;
        vkCmdBeginRendering(cmd, &r);
        vkCmdSetViewport(cmd, 0, 1, &renderVp);
        vkCmdSetScissor(cmd, 0, 1, &renderSc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, taaPipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, taaLayout_, 1, 1, &taaSets_[frameIndex_], 0,
                                nullptr);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRendering(cmd);
        imageBarrier(cmd, out.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        taaHistoryValid_ = true;
    }

    // 6. Auto exposure.
    {
        ScopedCmdLabel label(cmd, "Auto Exposure (exposure.comp)", DebugColors::Transfer);
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, exposurePipeline_);
        vkCmdDispatch(cmd, 1, 1, 1);
        memoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    }

    // 7. Bloom: progressive downsample then additive upsample.
    {
        ScopedCmdLabel label(cmd, "Bloom (bloom_down.frag + bloom_up.frag)", DebugColors::RenderPass);
        auto beginPass = [&](uint32_t mip, bool load) {
            VkRenderingAttachmentInfo a{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            a.imageView = bloomViews_[mip];
            a.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            a.loadOp = load ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            VkRenderingInfo r{VK_STRUCTURE_TYPE_RENDERING_INFO};
            r.renderArea = {{0, 0}, bloomExtents_[mip]};
            r.layerCount = 1;
            r.colorAttachmentCount = 1;
            r.pColorAttachments = &a;
            vkCmdBeginRendering(cmd, &r);
            VkViewport v{0, 0, static_cast<float>(bloomExtents_[mip].width),
                         static_cast<float>(bloomExtents_[mip].height), 0, 1};
            VkRect2D s{{0, 0}, bloomExtents_[mip]};
            vkCmdSetViewport(cmd, 0, 1, &v);
            vkCmdSetScissor(cmd, 0, 1, &s);
        };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, bloomDownPipeline_);
        for (uint32_t i = 0; i < kBloomLevels; ++i) {
            imageBarrier(cmd, bloom_.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, i,
                         1);
            beginPass(i, false);
            VkDescriptorSet srcSet = i == 0 ? bloomSourceSets_[frameIndex_] : bloomSourceSets_[i + 1];
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, postLayout_, 1, 1, &srcSet, 0, nullptr);
            BloomPush p{};
            VkExtent2D src = i == 0 ? renderExtent_ : bloomExtents_[i - 1];
            p.texel[0] = 1.0f / static_cast<float>(src.width);
            p.texel[1] = 1.0f / static_cast<float>(src.height);
            p.a = i == 0 ? 1.0f : 0.0f;
            p.b = 1.0f;  // threshold (in pre-exposure HDR units)
            vkCmdPushConstants(cmd, postLayout_, kAllStages, 0, sizeof(p), &p);
            vkCmdDraw(cmd, 3, 1, 0, 0);
            vkCmdEndRendering(cmd);
            imageBarrier(cmd, bloom_.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, i, 1);
        }
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, bloomUpPipeline_);
        for (uint32_t i = kBloomLevels - 1; i >= 1; --i) {
            imageBarrier(cmd, bloom_.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, i - 1, 1);
            beginPass(i - 1, true);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, postLayout_, 1, 1,
                                    &bloomSourceSets_[i + 2], 0, nullptr);
            BloomPush p{};
            p.texel[0] = 1.0f / static_cast<float>(bloomExtents_[i].width);
            p.texel[1] = 1.0f / static_cast<float>(bloomExtents_[i].height);
            p.a = 1.0f;  // radius
            p.b = 1.0f;  // weight
            vkCmdPushConstants(cmd, postLayout_, kAllStages, 0, sizeof(p), &p);
            vkCmdDraw(cmd, 3, 1, 0, 0);
            vkCmdEndRendering(cmd);
            imageBarrier(cmd, bloom_.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, i - 1, 1);
        }
    }

    // 8. Tone map into the swapchain image (with pre-rotation).
    {
        ScopedCmdLabel label(cmd, "Tonemap + Present (fullscreen.vert + tonemap.frag)", DebugColors::RenderPass);
        VkImage image = swapchainImages_[imageIndex];
        imageBarrier(cmd, image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        color.imageView = swapchainViews_[imageIndex];
        color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = {{0, 0}, swapchainExtent_};
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &color;
        vkCmdBeginRendering(cmd, &ri);
        VkViewport vp{0, 0, static_cast<float>(swapchainExtent_.width), static_cast<float>(swapchainExtent_.height), 0,
                      1};
        VkRect2D sc{{0, 0}, swapchainExtent_};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, tonemapPipeline_);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRendering(cmd);
        imageBarrier(cmd, image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_NONE, 0);
    }
}

void Renderer::applyGraphicsSettings(const GraphicsSettings& s) {
    float newScale = std::clamp(s.renderScale, 0.40f, 1.0f);
    if (std::fabs(newScale - renderScale_) > 0.005f) {
        renderScale_ = newScale;
        renderTargetsDirty_ = true;
    }
    if (s.taaEnabled != taaEnabled_) {
        taaEnabled_ = s.taaEnabled;
        taaHistoryValid_ = false;
    }
    uint32_t newShadowSize = (s.shadowSize <= 2048) ? 2048 : ((s.shadowSize <= 4096) ? 4096 : 8192);
    if (newShadowSize != shadowSize_) {
        shadowSize_ = newShadowSize;
        shadowMapDirtySize_ = true;
    }
    minClusterPixels_ = std::clamp(s.minClusterPixels, 0.05f, 4.0f);
    bloomScale_ = std::clamp(s.bloomScale, 0.0f, 2.5f);
    aoIntensityScale_ = std::clamp(s.aoIntensityScale, 0.0f, 2.5f);
    ssrEnabled_ = s.ssrEnabled;
    contactShadowsEnabled_ = s.contactShadowsEnabled;
    volumetricFogEnabled_ = s.volumetricFogEnabled;
    if (aoEnabled_ != s.aoEnabled) {
        aoEnabled_ = s.aoEnabled;
        if (!aoEnabled_) aoValid_ = false;
    }
    shadowPcfEnabled_ = s.shadowPcfEnabled;
    occlusionCulling_ = s.occlusionCulling;
}

void Renderer::drawFrame(double timeSeconds, double dtSeconds) {
    if (!initialized_ || !swapchain_) return;

    // Asset streaming happens between loading-screen frames.
    if (!currentAssetDir_.empty()) {
        if (!scene_.loaded() && !scene_.failed()) {
            scene_.loadStep();
        } else if (scene_.loaded() && !shadowReady_) {
            if (!buildShadowMap()) {
                initFailed_ = true;
                return;
            }
        }
    }
    if (swapchainDirty_) {
        vkDeviceWaitIdle(ctx_.device);
        destroySwapchain();
        if (!createSwapchain()) return;
        renderTargetsDirty_ = false;
    } else if (renderTargetsDirty_) {
        vkDeviceWaitIdle(ctx_.device);
        destroyRenderTargets();
        if (!createRenderTargets()) return;
        renderTargetsDirty_ = false;
    }
    if (shadowMapDirtySize_) {
        shadowMapDirtySize_ = false;
        if (scene_.loaded()) {
            vkDeviceWaitIdle(ctx_.device);
            ctx_.destroyImage(shadowMap_);
            ctx_.destroyBuffer(shadowDraws_);
            shadowReady_ = false;
            if (!buildShadowMap()) {
                initFailed_ = true;
                return;
            }
        }
    }

    FrameResources& frame = frames_[frameIndex_];
    vkWaitForFences(ctx_.device, 1, &frame.fence, VK_TRUE, UINT64_MAX);
    bool sceneReady = isLoaded();
    if (frame.submitted && sceneReady) readBackStats(frame, frameIndex_);

    uint32_t imageIndex = 0;
    VkResult res = vkAcquireNextImageKHR(ctx_.device, swapchain_, UINT64_MAX, frame.acquired, VK_NULL_HANDLE,
                                         &imageIndex);
    if (res == VK_ERROR_OUT_OF_DATE_KHR) {
        swapchainDirty_ = true;
        return;
    }
    if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR) {
        LOGE("vkAcquireNextImageKHR failed (%d)", res);
        return;
    }
    vkResetFences(ctx_.device, 1, &frame.fence);

    VkCommandBuffer cmd = frame.cmd;
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    if (timestampPool_) {
        vkCmdResetQueryPool(cmd, timestampPool_, frameIndex_ * 2, 2);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, timestampPool_, frameIndex_ * 2);
    }
    if (sceneReady) {
        updateUniforms(frame, timeSeconds, dtSeconds);
        std::string frameName = "Frame [" + currentAssetDir_;
        if (!stats_.cameraSegmentName.empty()) {
            frameName += " - " + stats_.cameraSegmentName;
        }
        frameName += "]";
        ScopedCmdLabel frameLabel(cmd, frameName.c_str(), DebugColors::Frame);
        recordScene(cmd, frame, imageIndex);
    } else {
        std::string loadName = currentAssetDir_.empty()
                                   ? "Loading Frame"
                                   : ("Loading Frame [" + currentAssetDir_ + " - " + scene_.status() + "]");
        ScopedCmdLabel frameLabel(cmd, loadName.c_str(), DebugColors::Frame);
        recordLoading(cmd, imageIndex, timeSeconds);
    }
    if (timestampPool_) {
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, timestampPool_, frameIndex_ * 2 + 1);
    }
    vkEndCommandBuffer(cmd);
    frame.timestampsValid = timestampPool_ != VK_NULL_HANDLE;

    VkSemaphoreSubmitInfo waitInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    waitInfo.semaphore = frame.acquired;
    waitInfo.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSemaphoreSubmitInfo signalInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    signalInfo.semaphore = renderDone_[imageIndex];
    signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkCommandBufferSubmitInfo cmdInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cmdInfo.commandBuffer = cmd;
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.waitSemaphoreInfoCount = 1;
    si.pWaitSemaphoreInfos = &waitInfo;
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &cmdInfo;
    si.signalSemaphoreInfoCount = 1;
    si.pSignalSemaphoreInfos = &signalInfo;
    res = vkQueueSubmit2(ctx_.queue, 1, &si, frame.fence);
    if (res != VK_SUCCESS) {
        LOGE("vkQueueSubmit2 failed (%d)", res);
        return;
    }
    frame.submitted = true;

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &renderDone_[imageIndex];
    pi.swapchainCount = 1;
    pi.pSwapchains = &swapchain_;
    pi.pImageIndices = &imageIndex;
    res = vkQueuePresentKHR(ctx_.queue, &pi);
    if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR) swapchainDirty_ = true;

    frameIndex_ = (frameIndex_ + 1) % kFramesInFlight;
    frameCounter_++;

    fpsFrames_++;
    wallClock_ += dtSeconds;
    if (wallClock_ - lastFpsTime_ >= 1.0) {
        stats_.fps = static_cast<float>(fpsFrames_ / (wallClock_ - lastFpsTime_));
        fpsFrames_ = 0;
        lastFpsTime_ = wallClock_;
    }
}

void Renderer::destroy() {
    if (!ctx_.device) return;
    vkDeviceWaitIdle(ctx_.device);
    destroySwapchain();
    if (surface_) vkDestroySurfaceKHR(ctx_.instance, surface_, nullptr);
    surface_ = VK_NULL_HANDLE;
    scene_.destroy();
    VkDevice dev = ctx_.device;
    for (auto& f : frames_) {
        ctx_.destroyBuffer(f.uniforms);
        ctx_.destroyBuffer(f.draws);
        ctx_.destroyBuffer(f.counts);
        if (f.fence) vkDestroyFence(dev, f.fence, nullptr);
        if (f.acquired) vkDestroySemaphore(dev, f.acquired, nullptr);
        if (f.pool) vkDestroyCommandPool(dev, f.pool, nullptr);
        f = FrameResources{};
    }
    ctx_.destroyImage(shadowMap_);
    ctx_.destroyBuffer(exposureBuffer_);
    ctx_.destroyBuffer(shadowDraws_);
    for (VkPipeline p : {cullPipeline_, exposurePipeline_, hizPipeline_, taaPipeline_, opaquePipeline_, maskPipeline_, blendPipeline_, skyPipeline_,
                         shadowOpaquePipeline_, shadowMaskPipeline_, bloomDownPipeline_, bloomUpPipeline_,
                         tonemapPipeline_, loadingPipeline_, aoPipeline_, aoBlurPipeline_}) {
        if (p) vkDestroyPipeline(dev, p, nullptr);
    }
    for (VkSampler s : {materialSampler_, linearClampSampler_, envSampler_, shadowSampler_, nearestSampler_}) {
        if (s) vkDestroySampler(dev, s, nullptr);
    }
    if (timestampPool_) vkDestroyQueryPool(dev, timestampPool_, nullptr);
    if (descriptorPool_) vkDestroyDescriptorPool(dev, descriptorPool_, nullptr);
    if (pipelineLayout_) vkDestroyPipelineLayout(dev, pipelineLayout_, nullptr);
    if (postLayout_) vkDestroyPipelineLayout(dev, postLayout_, nullptr);
    if (hizLayout_) vkDestroyPipelineLayout(dev, hizLayout_, nullptr);
    if (taaLayout_) vkDestroyPipelineLayout(dev, taaLayout_, nullptr);
    if (aoLayout_) vkDestroyPipelineLayout(dev, aoLayout_, nullptr);
    if (aoSetLayout_) vkDestroyDescriptorSetLayout(dev, aoSetLayout_, nullptr);
    if (taaSetLayout_) vkDestroyDescriptorSetLayout(dev, taaSetLayout_, nullptr);
    if (hizSetLayout_) vkDestroyDescriptorSetLayout(dev, hizSetLayout_, nullptr);
    if (globalSetLayout_) vkDestroyDescriptorSetLayout(dev, globalSetLayout_, nullptr);
    if (textureSetLayout_) vkDestroyDescriptorSetLayout(dev, textureSetLayout_, nullptr);
    ctx_.destroy();
    initialized_ = false;
}
