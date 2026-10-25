#pragma once

#include <android/asset_manager.h>

#include <string>
#include <vector>

#include "math_util.h"
#include "scene_format.h"
#include "vk_context.h"

// GPU-side mirror of a cluster (std430, 48 bytes).
struct GpuCluster {
    float aabbMin[3];
    uint32_t materialIndex;
    float aabbMax[3];
    uint32_t bucket;  // 0 opaque, 1 alpha-mask, 2 blend
    uint32_t firstIndex;
    uint32_t indexCount;
    int32_t vertexOffset;
    uint32_t pad;
};

// GPU-side material (std430, 64 bytes).
struct GpuMaterial {
    float baseColorFactor[4];
    float emissiveFactor[3];
    float alphaCutoff;
    uint32_t baseColorTexture;
    uint32_t normalTexture;
    uint32_t ormTexture;
    uint32_t emissiveTexture;
    uint32_t alphaMode;
    uint32_t flags;
    float roughnessFactor;
    float metallicFactor;
};

struct EnvironmentInfo {
    Vec3 sunDirection{0, 1, 0};  // towards the sun, in environment-map space
    Vec3 sunIrradiance{1, 1, 1};
    float sh[9][3] = {};
};

struct RenderConfig {
    float emissiveIntensity = 4.0f;
    float bloomStrength = 0.06f;
    float exposureKey = 0.16f;
    float minExposure = 0.15f;
    float maxExposure = 3.0f;
    float fogDensity = 0.0045f;
    float fogHeightFalloff = 0.08f;
    float fogBaseHeight = 0.0f;
    float fogMaxOpacity = 0.60f;
    Vec3 fogColor{0.0f, 0.0f, 0.0f};
    float sunAzimuthOffset = -0.35f;
    float sunIntensityScale = 2.2f;
    float pointLightScale = 0.030f;
    float translucency = 1.0f;  // strength of back-lit transmission through kMaterialTranslucent materials
    float aoRadius = 0.8f;      // screen-space AO search radius (m)
    float aoIntensity = 1.0f;   // screen-space AO exponent (0 disables)
};

constexpr uint32_t kBucketOpaque = 0;
constexpr uint32_t kBucketMask = 1;
constexpr uint32_t kBucketBlend = 2;
constexpr uint32_t kBucketCount = 3;

class Scene {
public:
    void begin(VulkanContext* ctx, AAssetManager* assets, const std::string& assetDir);
    // Performs a slice of the loading work. Returns true once everything is resident on the GPU.
    bool loadStep();
    [[nodiscard]] float progress() const;
    [[nodiscard]] const std::string& status() const { return status_; }
    [[nodiscard]] bool loaded() const { return stage_ == Stage::Done; }
    [[nodiscard]] bool failed() const { return failed_; }
    [[nodiscard]] const std::string& assetDir() const { return assetDir_; }
    void destroy();

    GpuBuffer vertexBuffer;
    GpuBuffer indexBuffer;
    GpuBuffer clusterBuffer;
    GpuBuffer materialBuffer;
    std::vector<GpuImage> textures;  // bindless texture array (index matches material references)
    GpuImage sky;
    GpuImage envSpecular;
    EnvironmentInfo environment;
    RenderConfig renderConfig;
    std::string manifestJson;

    std::vector<GpuCluster> clusters;
    std::vector<bistro::MaterialRecord> materials;
    std::vector<bistro::CameraKey> cameraKeys;
    std::vector<bistro::PointLightRecord> pointLights;
    bistro::SceneHeader header{};
    uint64_t triangleCount = 0;

private:
    enum class Stage { Geometry, Textures, Environment, Done };

    bool loadManifest();
    bool loadGeometry();
    bool loadTexture(const std::string& path, GpuImage& out, const char* debugName);
    bool createFallbackTextures();
    bool loadEnvironment();

    VulkanContext* ctx_ = nullptr;
    AAssetManager* assets_ = nullptr;
    std::string assetDir_;
    Stage stage_ = Stage::Geometry;
    std::vector<bistro::TextureRecord> textureRecords_;
    size_t nextTexture_ = 0;
    uint32_t fallbackBase_ = 0;
    std::string status_ = "Starting";
    bool failed_ = false;
};
