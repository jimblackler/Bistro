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

constexpr uint32_t kBucketOpaque = 0;
constexpr uint32_t kBucketMask = 1;
constexpr uint32_t kBucketBlend = 2;
constexpr uint32_t kBucketCount = 3;

class Scene {
public:
    void begin(VulkanContext* ctx, AAssetManager* assets);
    // Performs a slice of the loading work. Returns true once everything is resident on the GPU.
    bool loadStep();
    [[nodiscard]] float progress() const;
    [[nodiscard]] const std::string& status() const { return status_; }
    [[nodiscard]] bool loaded() const { return stage_ == Stage::Done; }
    [[nodiscard]] bool failed() const { return failed_; }
    void destroy();

    GpuBuffer vertexBuffer;
    GpuBuffer indexBuffer;
    GpuBuffer clusterBuffer;
    GpuBuffer materialBuffer;
    std::vector<GpuImage> textures;  // bindless texture array (index matches material references)
    GpuImage sky;
    GpuImage envSpecular;
    EnvironmentInfo environment;

    std::vector<GpuCluster> clusters;
    std::vector<bistro::MaterialRecord> materials;
    std::vector<bistro::CameraKey> cameraKeys;
    std::vector<bistro::PointLightRecord> pointLights;
    bistro::SceneHeader header{};
    uint64_t triangleCount = 0;

private:
    enum class Stage { Geometry, Textures, Environment, Done };

    bool loadGeometry();
    bool loadTexture(const std::string& path, GpuImage& out, const char* debugName);
    bool createFallbackTextures();
    bool loadEnvironment();

    VulkanContext* ctx_ = nullptr;
    AAssetManager* assets_ = nullptr;
    Stage stage_ = Stage::Geometry;
    std::vector<bistro::TextureRecord> textureRecords_;
    size_t nextTexture_ = 0;
    uint32_t fallbackBase_ = 0;
    std::string status_ = "Starting";
    bool failed_ = false;
};
