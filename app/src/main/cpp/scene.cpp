#include "scene.h"

#include <android/log.h>

#include <chrono>
#include <cstdlib>
#include <cstring>

#include "vulkan_debug.h"

#define LOG_TAG "BistroScene"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

class AssetFile {
public:
    AssetFile(AAssetManager* mgr, const std::string& path) {
        asset_ = AAssetManager_open(mgr, path.c_str(), AASSET_MODE_BUFFER);
        if (asset_) {
            data_ = static_cast<const uint8_t*>(AAsset_getBuffer(asset_));
            size_ = static_cast<size_t>(AAsset_getLength64(asset_));
        }
    }
    ~AssetFile() {
        if (asset_) AAsset_close(asset_);
    }
    AssetFile(const AssetFile&) = delete;
    AssetFile& operator=(const AssetFile&) = delete;

    [[nodiscard]] const uint8_t* data() const { return data_; }
    [[nodiscard]] size_t size() const { return size_; }
    [[nodiscard]] bool valid() const { return data_ != nullptr; }

private:
    AAsset* asset_ = nullptr;
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
};

VkFormat astcFormat(uint32_t bw, uint32_t bh, bool srgb) {
    if (bw == 4 && bh == 4) return srgb ? VK_FORMAT_ASTC_4x4_SRGB_BLOCK : VK_FORMAT_ASTC_4x4_UNORM_BLOCK;
    if (bw == 5 && bh == 5) return srgb ? VK_FORMAT_ASTC_5x5_SRGB_BLOCK : VK_FORMAT_ASTC_5x5_UNORM_BLOCK;
    if (bw == 6 && bh == 6) return srgb ? VK_FORMAT_ASTC_6x6_SRGB_BLOCK : VK_FORMAT_ASTC_6x6_UNORM_BLOCK;
    if (bw == 8 && bh == 8) return srgb ? VK_FORMAT_ASTC_8x8_SRGB_BLOCK : VK_FORMAT_ASTC_8x8_UNORM_BLOCK;
    return VK_FORMAT_UNDEFINED;
}

double nowMs() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

void transitionForUpload(VulkanContext* ctx, const std::vector<GpuImage*>& images, bool toTransfer) {
    VkCommandBuffer cmd = ctx->beginOneShot();
    for (GpuImage* img : images) {
        if (toTransfer) {
            imageBarrier(cmd, img->image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_NONE, 0,
                         VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        } else {
            imageBarrier(cmd, img->image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COPY_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    }
    ctx->endOneShot(cmd);
}

float parseJsonFloat(const std::string& json, const char* key, float fallback) {
    std::string needle = std::string("\"") + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return fallback;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return fallback;
    const char* start = json.c_str() + pos + 1;
    char* end = nullptr;
    float v = std::strtof(start, &end);
    return end != start ? v : fallback;
}

Vec3 parseJsonVec3(const std::string& json, const char* key, Vec3 fallback) {
    std::string needle = std::string("\"") + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return fallback;
    pos = json.find('[', pos + needle.size());
    if (pos == std::string::npos) return fallback;
    const char* p = json.c_str() + pos + 1;
    char* next = nullptr;
    float x = std::strtof(p, &next);
    if (next == p) return fallback;
    while (*next && (*next == ',' || *next == ' ')) next++;
    p = next;
    float y = std::strtof(p, &next);
    if (next == p) return fallback;
    while (*next && (*next == ',' || *next == ' ')) next++;
    p = next;
    float z = std::strtof(p, &next);
    if (next == p) return fallback;
    return Vec3{x, y, z};
}

}  // namespace

void Scene::begin(VulkanContext* ctx, AAssetManager* assets, const std::string& assetDir) {
    ctx_ = ctx;
    assets_ = assets;
    assetDir_ = assetDir;
    stage_ = Stage::Geometry;
    nextTexture_ = 0;
    failed_ = false;
    status_ = "Loading geometry";
    loadManifest();
}

bool Scene::loadManifest() {
    renderConfig = RenderConfig{};
    manifestJson.clear();
    AssetFile file(assets_, assetDir_ + "/manifest.json");
    if (!file.valid()) return false;
    manifestJson.assign(reinterpret_cast<const char*>(file.data()), file.size());
    size_t rcPos = manifestJson.find("\"renderConfig\"");
    if (rcPos != std::string::npos) {
        std::string sub = manifestJson.substr(rcPos);
        renderConfig.emissiveIntensity = parseJsonFloat(sub, "emissiveIntensity", renderConfig.emissiveIntensity);
        renderConfig.bloomStrength = parseJsonFloat(sub, "bloomStrength", renderConfig.bloomStrength);
        renderConfig.exposureKey = parseJsonFloat(sub, "exposureKey", renderConfig.exposureKey);
        renderConfig.minExposure = parseJsonFloat(sub, "minExposure", renderConfig.minExposure);
        renderConfig.maxExposure = parseJsonFloat(sub, "maxExposure", renderConfig.maxExposure);
        renderConfig.fogDensity = parseJsonFloat(sub, "fogDensity", renderConfig.fogDensity);
        renderConfig.fogHeightFalloff = parseJsonFloat(sub, "fogHeightFalloff", renderConfig.fogHeightFalloff);
        renderConfig.fogBaseHeight = parseJsonFloat(sub, "fogBaseHeight", renderConfig.fogBaseHeight);
        renderConfig.fogMaxOpacity = parseJsonFloat(sub, "fogMaxOpacity", renderConfig.fogMaxOpacity);
        renderConfig.fogColor = parseJsonVec3(sub, "fogColor", renderConfig.fogColor);
        renderConfig.sunAzimuthOffset = parseJsonFloat(sub, "sunAzimuthOffset", renderConfig.sunAzimuthOffset);
        renderConfig.sunIntensityScale = parseJsonFloat(sub, "sunIntensityScale", renderConfig.sunIntensityScale);
        renderConfig.pointLightScale = parseJsonFloat(sub, "pointLightScale", renderConfig.pointLightScale);
    }
    return true;
}

float Scene::progress() const {
    switch (stage_) {
        case Stage::Geometry: return 0.0f;
        case Stage::Textures:
            return 0.25f + 0.70f * (textureRecords_.empty() ? 1.0f
                                                             : static_cast<float>(nextTexture_) / textureRecords_.size());
        case Stage::Environment: return 0.95f;
        case Stage::Done: return 1.0f;
    }
    return 0.0f;
}

bool Scene::loadStep() {
    if (failed_) return false;
    switch (stage_) {
        case Stage::Geometry:
            if (!loadGeometry()) {
                failed_ = true;
                status_ = "Failed to load scene geometry";
                return false;
            }
            stage_ = Stage::Textures;
            status_ = "Loading textures";
            return false;
        case Stage::Textures: {
            double start = nowMs();
            while (nextTexture_ < textureRecords_.size() && nowMs() - start < 40.0) {
                const auto& rec = textureRecords_[nextTexture_];
                std::string path = assetDir_ + "/textures/" + rec.name + ".tex";
                if (!loadTexture(path, textures[nextTexture_], rec.name)) {
                    LOGE("Texture %s failed to load; using fallback", rec.name);
                }
                nextTexture_++;
            }
            if (nextTexture_ >= textureRecords_.size()) {
                stage_ = Stage::Environment;
                status_ = "Loading environment";
            }
            return false;
        }
        case Stage::Environment:
            if (!loadEnvironment()) {
                failed_ = true;
                status_ = "Failed to load environment";
                return false;
            }
            stage_ = Stage::Done;
            status_ = "Ready";
            LOGI("Scene (%s) loaded: %.1f MB of GPU memory allocated", assetDir_.c_str(),
                 ctx_->allocatedBytes / 1048576.0);
            return true;
        case Stage::Done:
            return true;
    }
    return false;
}

bool Scene::loadGeometry() {
    double t0 = nowMs();
    std::string scenePath = assetDir_ + "/scene.bin";
    AssetFile file(assets_, scenePath);
    if (!file.valid() || file.size() < sizeof(bistro::SceneHeader)) {
        LOGE("Unable to open %s", scenePath.c_str());
        return false;
    }
    const uint8_t* p = file.data();
    memcpy(&header, p, sizeof(header));
    if (memcmp(header.magic, bistro::kSceneMagic, 4) != 0 || header.version != bistro::kSceneVersion) {
        LOGE("scene.bin has wrong magic/version");
        return false;
    }
    p += sizeof(header);
    textureRecords_.resize(header.textureCount);
    memcpy(textureRecords_.data(), p, header.textureCount * sizeof(bistro::TextureRecord));
    p += header.textureCount * sizeof(bistro::TextureRecord);
    materials.resize(header.materialCount);
    memcpy(materials.data(), p, header.materialCount * sizeof(bistro::MaterialRecord));
    p += header.materialCount * sizeof(bistro::MaterialRecord);
    std::vector<bistro::ClusterRecord> records(header.clusterCount);
    memcpy(records.data(), p, header.clusterCount * sizeof(bistro::ClusterRecord));
    p += header.clusterCount * sizeof(bistro::ClusterRecord);
    cameraKeys.resize(header.cameraKeyCount);
    memcpy(cameraKeys.data(), p, header.cameraKeyCount * sizeof(bistro::CameraKey));
    p += header.cameraKeyCount * sizeof(bistro::CameraKey);
    pointLights.resize(header.pointLightCount);
    memcpy(pointLights.data(), p, header.pointLightCount * sizeof(bistro::PointLightRecord));
    p += header.pointLightCount * sizeof(bistro::PointLightRecord);
    const uint8_t* indexData = p;
    p += header.indexCount * sizeof(uint16_t);
    const uint8_t* vertexData = p;
    p += static_cast<size_t>(header.vertexCount) * sizeof(bistro::GpuVertex);
    if (static_cast<size_t>(p - file.data()) > file.size()) {
        LOGE("scene.bin truncated");
        return false;
    }

    VkDeviceSize vbSize = static_cast<VkDeviceSize>(header.vertexCount) * sizeof(bistro::GpuVertex);
    VkDeviceSize ibSize = static_cast<VkDeviceSize>(header.indexCount) * sizeof(uint16_t);
    if (!ctx_->createBuffer(vbSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, "Scene Vertex Buffer") ||
        !ctx_->createBuffer(ibSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, "Scene Index Buffer")) {
        return false;
    }
    ctx_->uploadBuffer(vertexBuffer, 0, vertexData, vbSize);
    ctx_->uploadBuffer(indexBuffer, 0, indexData, ibSize);

    // Texture table: scene textures followed by 4 fallback textures (one per kind).
    textures.resize(header.textureCount + 4);
    fallbackBase_ = header.textureCount;

    auto resolve = [&](int32_t index, uint32_t kind) {
        return index >= 0 ? static_cast<uint32_t>(index) : fallbackBase_ + kind;
    };
    std::vector<GpuMaterial> gpuMaterials(materials.size());
    for (size_t i = 0; i < materials.size(); ++i) {
        const auto& m = materials[i];
        GpuMaterial& g = gpuMaterials[i];
        memcpy(g.baseColorFactor, m.baseColorFactor, sizeof(g.baseColorFactor));
        memcpy(g.emissiveFactor, m.emissiveFactor, sizeof(g.emissiveFactor));
        g.alphaCutoff = m.alphaCutoff;
        g.baseColorTexture = resolve(m.baseColorTexture, bistro::kTextureBaseColor);
        g.normalTexture = resolve(m.normalTexture, bistro::kTextureNormal);
        g.ormTexture = resolve(m.ormTexture, bistro::kTextureOrm);
        g.emissiveTexture = resolve(m.emissiveTexture, bistro::kTextureEmissive);
        g.alphaMode = m.alphaMode;
        g.flags = m.flags | (m.ormTexture >= 0 ? 4u : 0u);  // bit 2: has an ORM texture
        g.roughnessFactor = m.roughnessFactor;
        g.metallicFactor = m.metallicFactor;
    }

    clusters.resize(records.size());
    triangleCount = 0;
    for (size_t i = 0; i < records.size(); ++i) {
        const auto& r = records[i];
        GpuCluster& c = clusters[i];
        memcpy(c.aabbMin, r.aabbMin, sizeof(c.aabbMin));
        memcpy(c.aabbMax, r.aabbMax, sizeof(c.aabbMax));
        c.materialIndex = r.materialIndex;
        uint32_t mode = materials[r.materialIndex].alphaMode;
        c.bucket = mode == bistro::kAlphaBlend ? kBucketBlend : (mode == bistro::kAlphaMask ? kBucketMask : kBucketOpaque);
        c.firstIndex = r.firstIndex;
        c.indexCount = r.indexCount;
        c.vertexOffset = r.vertexOffset;
        triangleCount += r.indexCount / 3;
    }

    if (!ctx_->createBuffer(clusters.size() * sizeof(GpuCluster),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, clusterBuffer, "Cluster Buffer") ||
        !ctx_->createBuffer(gpuMaterials.size() * sizeof(GpuMaterial),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, materialBuffer, "Material Buffer")) {
        return false;
    }
    ctx_->uploadBuffer(clusterBuffer, 0, clusters.data(), clusters.size() * sizeof(GpuCluster));
    ctx_->uploadBuffer(materialBuffer, 0, gpuMaterials.data(), gpuMaterials.size() * sizeof(GpuMaterial));
    ctx_->flushUploads();

    if (!createFallbackTextures()) return false;
    LOGI("Geometry (%s) loaded in %.0f ms: %u vertices, %llu triangles, %u clusters, %u materials, %u textures",
         assetDir_.c_str(), nowMs() - t0, header.vertexCount, static_cast<unsigned long long>(triangleCount),
         header.clusterCount, header.materialCount, header.textureCount);
    return true;
}

bool Scene::createFallbackTextures() {
    const uint8_t pixels[4][4] = {
        {255, 255, 255, 255},  // base colour: white
        {128, 128, 128, 128},  // normal: flat (X in RGB, Y in A)
        {255, 255, 255, 255},  // ORM: neutral (material factors provide roughness/metalness)
        {0, 0, 0, 255},        // emissive: black
    };
    const VkFormat formats[4] = {VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM,
                                 VK_FORMAT_R8G8B8A8_SRGB};
    std::vector<GpuImage*> images;
    for (int i = 0; i < 4; ++i) {
        ImageDesc d;
        d.format = formats[i];
        d.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        d.name = "Fallback Texture";
        if (!ctx_->createImage(d, textures[fallbackBase_ + i])) return false;
        images.push_back(&textures[fallbackBase_ + i]);
    }
    transitionForUpload(ctx_, images, true);
    for (int i = 0; i < 4; ++i) ctx_->uploadImageLevel(textures[fallbackBase_ + i], 0, 1, 1, pixels[i], 4);
    ctx_->flushUploads();
    transitionForUpload(ctx_, images, false);
    return true;
}

bool Scene::loadTexture(const std::string& path, GpuImage& out, const char* debugName) {
    AssetFile file(assets_, path);
    if (!file.valid() || file.size() < sizeof(bistro::TextureHeader)) {
        LOGE("Missing texture %s", path.c_str());
        return false;
    }
    bistro::TextureHeader th;
    memcpy(&th, file.data(), sizeof(th));
    if (memcmp(th.magic, bistro::kTextureMagic, 4) != 0) {
        LOGE("Bad texture header %s", path.c_str());
        return false;
    }
    std::vector<bistro::TextureLevel> levels(th.mipCount);
    memcpy(levels.data(), file.data() + sizeof(th), th.mipCount * sizeof(bistro::TextureLevel));

    VkFormat format = th.format == bistro::kFormatRgba16f ? VK_FORMAT_R16G16B16A16_SFLOAT
                                                           : astcFormat(th.blockWidth, th.blockHeight, th.srgb != 0);
    if (format == VK_FORMAT_UNDEFINED) {
        LOGE("Unsupported texture format in %s", path.c_str());
        return false;
    }
    ImageDesc d;
    d.width = th.width;
    d.height = th.height;
    d.mipLevels = th.mipCount;
    d.format = format;
    d.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    d.name = debugName;
    if (!ctx_->createImage(d, out)) return false;

    transitionForUpload(ctx_, {&out}, true);
    for (uint32_t i = 0; i < th.mipCount; ++i) {
        uint32_t w = std::max(1u, th.width >> i);
        uint32_t h = std::max(1u, th.height >> i);
        if (levels[i].offset + levels[i].size > file.size()) {
            LOGE("Texture %s level %u out of range", path.c_str(), i);
            break;
        }
        ctx_->uploadImageLevel(out, i, w, h, file.data() + levels[i].offset, levels[i].size);
    }
    ctx_->flushUploads();
    transitionForUpload(ctx_, {&out}, false);
    return true;
}

bool Scene::loadEnvironment() {
    if (!loadTexture(assetDir_ + "/sky.tex", sky, "Sky Equirect") ||
        !loadTexture(assetDir_ + "/env_specular.tex", envSpecular, "Prefiltered Specular Environment")) {
        return false;
    }
    AssetFile file(assets_, assetDir_ + "/environment.bin");
    if (!file.valid() || file.size() < 4 + 33 * sizeof(float) || memcmp(file.data(), "BENV", 4) != 0) {
        LOGE("environment.bin missing or invalid");
        return false;
    }
    const auto* f = reinterpret_cast<const float*>(file.data() + 4);
    environment.sunDirection = normalize(Vec3{f[0], f[1], f[2]});
    environment.sunIrradiance = Vec3{f[3], f[4], f[5]};
    for (int i = 0; i < 9; ++i)
        for (int c = 0; c < 3; ++c) environment.sh[i][c] = f[6 + i * 3 + c];
    return true;
}

void Scene::destroy() {
    if (!ctx_) return;
    ctx_->destroyBuffer(vertexBuffer);
    ctx_->destroyBuffer(indexBuffer);
    ctx_->destroyBuffer(clusterBuffer);
    ctx_->destroyBuffer(materialBuffer);
    for (auto& t : textures) ctx_->destroyImage(t);
    textures.clear();
    ctx_->destroyImage(sky);
    ctx_->destroyImage(envSpecular);
    clusters.clear();
    materials.clear();
    cameraKeys.clear();
    pointLights.clear();
    textureRecords_.clear();
}
