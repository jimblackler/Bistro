#pragma once

// Binary layout of the converted Bistro scene (scene.bin) and texture files (*.tex).
// Shared between the offline converter (tools/scene_converter) and the Android renderer.
//
// scene.bin layout (all little-endian, tightly packed in this order):
//   SceneHeader
//   TextureRecord[textureCount]
//   MaterialRecord[materialCount]
//   ClusterRecord[clusterCount]
//   CameraKey[cameraKeyCount]
//   PointLightRecord[pointLightCount]
//   uint16_t indices[indexCount]      (cluster-local, add ClusterRecord::vertexOffset)
//   GpuVertex vertices[vertexCount]

#include <cstdint>

namespace bistro {

constexpr char kSceneMagic[4] = {'B', 'S', 'T', 'R'};
constexpr uint32_t kSceneVersion = 3;

enum TextureKind : uint32_t {
    kTextureBaseColor = 0,  // sRGB colour + alpha (opacity)
    kTextureNormal = 1,     // tangent-space normal, X in RGB / Y in A (ASTC normal mode)
    kTextureOrm = 2,        // R (unused occlusion), G roughness, B metalness (linear)
    kTextureEmissive = 3,   // sRGB emissive colour
};

enum AlphaMode : uint32_t {
    kAlphaOpaque = 0,
    kAlphaMask = 1,
    kAlphaBlend = 2,
};

enum MaterialFlags : uint32_t {
    kMaterialDoubleSided = 1u << 0,
    kMaterialEmissive = 1u << 1,
};

struct SceneHeader {
    char magic[4];
    uint32_t version;
    uint32_t vertexCount;
    uint32_t indexCount;
    uint32_t clusterCount;
    uint32_t materialCount;
    uint32_t textureCount;
    uint32_t cameraKeyCount;
    float boundsMin[3];
    float boundsMax[3];
    float sunDirection[3];  // unit vector pointing towards the sun (from the FBX directional light)
    float sunColor[3];
    uint32_t pointLightCount;
};

struct TextureRecord {
    char name[96];  // file stem; the texture lives at textures/<name>.tex
    uint32_t kind;  // TextureKind
};

struct MaterialRecord {
    char name[96];
    int32_t baseColorTexture;  // index into the texture table or -1
    int32_t normalTexture;
    int32_t ormTexture;
    int32_t emissiveTexture;
    float baseColorFactor[4];
    float emissiveFactor[3];
    float alphaCutoff;
    uint32_t alphaMode;  // AlphaMode
    uint32_t flags;      // MaterialFlags
    float roughnessFactor;  // multiplied with the ORM texture's G channel
    float metallicFactor;   // multiplied with the ORM texture's B channel
};

struct ClusterRecord {
    uint32_t firstIndex;
    uint32_t indexCount;
    int32_t vertexOffset;
    uint32_t materialIndex;
    float aabbMin[3];
    float aabbMax[3];
};

struct CameraKey {
    float time;
    float position[3];
    float forward[3];
};

struct PointLightRecord {
    float position[3];
    float range;
    float color[3];  // as authored in the FBX (arbitrary units)
    float pad;
};

struct GpuVertex {
    float position[3];  // R32G32B32_SFLOAT, world space metres
    int16_t normal[2];  // R16G16_SNORM, octahedral encoded
    int8_t tangent[4];  // R8G8B8A8_SNORM, xyz tangent + w bitangent sign
    float uv[2];        // R32G32_SFLOAT
    uint8_t extra[4];   // R8G8B8A8_UNORM: x = baked sky visibility, yzw reserved
};
static_assert(sizeof(GpuVertex) == 32, "GpuVertex must be tightly packed");

// ---- Texture container (*.tex) ----
//   TextureHeader, TextureLevel[mipCount], payload
constexpr char kTextureMagic[4] = {'B', 'T', 'E', 'X'};

enum TextureFormat : uint32_t {
    kFormatAstcLdr = 0,   // ASTC LDR blocks (blockWidth x blockHeight), see 'srgb'
    kFormatRgba16f = 1,   // uncompressed half-float RGBA (environment maps)
};

struct TextureHeader {
    char magic[4];
    uint32_t format;  // TextureFormat
    uint32_t width;
    uint32_t height;
    uint32_t mipCount;
    uint32_t blockWidth;
    uint32_t blockHeight;
    uint32_t srgb;
};

struct TextureLevel {
    uint32_t offset;  // from start of file
    uint32_t size;
};

}  // namespace bistro
