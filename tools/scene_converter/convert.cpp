// Bistro scene converter.
//
// Loads the Amazon Lumberyard Bistro FBX files with Assimp, flattens the node hierarchy into world
// space, merges geometry per material and splits it into spatially-coherent clusters suitable for
// GPU-driven culling. The result is written to a compact binary file (scene.bin) that the Android
// app can upload to the GPU with minimal processing.
//
// Usage: bistro_convert <out_dir> <work_dir> <alpha_stats.txt> <exterior.fbx> [interior.fbx ...]

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../app/src/main/cpp/scene_format.h"
#include "sky_visibility.h"

namespace {

constexpr uint32_t kMaxClusterTriangles = 2048;
constexpr float kMaxClusterExtent = 12.0f;  // metres

struct Vec3 {
    float x = 0, y = 0, z = 0;
};

Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 cross(Vec3 a, Vec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
Vec3 normalize(Vec3 v) {
    float l = std::sqrt(dot(v, v));
    return l > 1e-12f ? v * (1.0f / l) : Vec3{0, 1, 0};
}
float comp(Vec3 v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }

struct FullVertex {
    Vec3 pos;
    Vec3 normal;
    Vec3 tangent;
    float tangentSign = 1.0f;
    float uv[2] = {0, 0};
    float skyVisibility = 1.0f;
};

struct MaterialInfo {
    std::string name;
    std::string baseColor, normal, orm, emissive;  // texture names (no extension)
    float baseColorFactor[4] = {1, 1, 1, 1};
    float emissiveFactor[3] = {0, 0, 0};
    float roughnessFactor = 1.0f;
    float metallicFactor = 1.0f;
    bool hasRoughnessFactor = false;
    bool hasMetallicFactor = false;
    bool doubleSided = false;
};

// All geometry belonging to one material, in world space.
struct MaterialGeometry {
    std::vector<FullVertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<int> triScores;
};

struct TextureStats {
    float cutout = 0;       // fraction of texels with alpha < 0.5
    float translucent = 0;  // fraction with 0.05 < alpha < 0.95
    float mean[3] = {0, 0, 0};
    float maxStdDev = 0;    // largest per-channel standard deviation (RGB)
};

struct AlphaStats {
    std::unordered_map<std::string, float> cutoutFraction;
    std::unordered_map<std::string, float> translucentFraction;
    std::unordered_map<std::string, TextureStats> all;
    std::unordered_map<std::string, std::string> canonicalName;
};

std::string toLowerStr(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string textureNameFromPath(const aiString& p, const AlphaStats& stats) {
    std::string s = p.C_Str();
    std::replace(s.begin(), s.end(), '\\', '/');
    size_t slash = s.find_last_of('/');
    if (slash != std::string::npos) s = s.substr(slash + 1);
    size_t dot = s.find_last_of('.');
    if (dot != std::string::npos) s = s.substr(0, dot);
    auto it = stats.canonicalName.find(toLowerStr(s));
    if (it != stats.canonicalName.end()) return it->second;
    return s;
}

bool containsCI(const std::string& hay, const char* needle) {
    std::string h = hay, n = needle;
    std::transform(h.begin(), h.end(), h.begin(), ::tolower);
    std::transform(n.begin(), n.end(), n.begin(), ::tolower);
    return h.find(n) != std::string::npos;
}

void octEncode(Vec3 n, int16_t out[2]) {
    float l1 = std::fabs(n.x) + std::fabs(n.y) + std::fabs(n.z);
    float x = n.x / l1, y = n.y / l1;
    if (n.z < 0) {
        float ox = (1.0f - std::fabs(y)) * (x >= 0 ? 1.0f : -1.0f);
        float oy = (1.0f - std::fabs(x)) * (y >= 0 ? 1.0f : -1.0f);
        x = ox;
        y = oy;
    }
    out[0] = static_cast<int16_t>(std::lround(std::clamp(x, -1.0f, 1.0f) * 32767.0f));
    out[1] = static_cast<int16_t>(std::lround(std::clamp(y, -1.0f, 1.0f) * 32767.0f));
}

int8_t toSnorm8(float v) { return static_cast<int8_t>(std::lround(std::clamp(v, -1.0f, 1.0f) * 127.0f)); }

struct ConvertConfig {
    int aoRays = 48;
    float aoMaxDist = 80.0f;
    float aoMinVis = 0.0f;
    float aoMinArea = 0.0f;
    float weldGrid = 0.0f;
    float foliageWeldGrid = 0.0f;
    bool foldConstantBaseColor = false;
    bool addCurtainGlass = true;
    bool deduplicateWithinScene = false;
    std::vector<std::string> skipNodeSubstrings;
    // Thin, light-transmitting materials (leaves, grass, curtains, awnings): a material is flagged
    // kMaterialTranslucent if its name contains any include substring and no exclude substring.
    std::vector<std::string> translucentInclude;
    std::vector<std::string> translucentExclude;
    bool hasSunOverride = false;
    float sunDir[3] = {0.3f, 0.8f, 0.2f};
    float sunColor[3] = {1.0f, 1.0f, 1.0f};
    std::vector<bistro::PointLightRecord> customPointLights;
};

float srgbToLinear(float c) {
    c = std::clamp(c, 0.0f, 1.0f);
    return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

class Converter {
public:
    Converter(AlphaStats stats, std::string workDir, ConvertConfig cfg)
        : stats_(std::move(stats)), workDir_(std::move(workDir)), cfg_(std::move(cfg)) {
        if (cfg_.hasSunOverride) {
            std::memcpy(sunDir_, cfg_.sunDir, sizeof(sunDir_));
            std::memcpy(sunColor_, cfg_.sunColor, sizeof(sunColor_));
        }
        pointLights_ = cfg_.customPointLights;
    }

    bool addScene(const std::string& path, bool isPrimary) {
        auto importer = std::make_unique<Assimp::Importer>();
        importer->SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_POINT | aiPrimitiveType_LINE);
        std::string lowerPath = toLowerStr(path);
        bool isGltf = (lowerPath.size() >= 5 && lowerPath.substr(lowerPath.size() - 5) == ".gltf") ||
                      (lowerPath.size() >= 4 && lowerPath.substr(lowerPath.size() - 4) == ".glb");
        unsigned flags = aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace |
                         aiProcess_JoinIdenticalVertices | aiProcess_ImproveCacheLocality |
                         aiProcess_SortByPType | aiProcess_FindInvalidData;
        if (!isGltf) flags |= aiProcess_FlipUVs;
        const aiScene* scene = importer->ReadFile(path, flags);
        if (!scene) {
            fprintf(stderr, "Failed to load %s: %s\n", path.c_str(), importer->GetErrorString());
            return false;
        }
        printf("Loaded %s: %u meshes, %u materials\n", path.c_str(), scene->mNumMeshes, scene->mNumMaterials);

        std::vector<uint32_t> materialRemap(scene->mNumMaterials);
        for (unsigned i = 0; i < scene->mNumMaterials; ++i) materialRemap[i] = registerMaterial(scene->mMaterials[i]);

        aiMatrix4x4 rootTransform;
        float sceneScale = 1.0f;
        if (isPrimary) {
            if (scene->mRootNode->mNumChildren == 1 && scene->mRootNode->mChildren[0]->mNumMeshes == 0) {
                float s = scene->mRootNode->mChildren[0]->mTransformation.a1 / 0.01f;
                if (s > 0.1f) primaryScale_ = s;
            }
        } else if (scene->mRootNode->mNumChildren > 1) {
            // BistroInterior.fbx has no top-level wrapper node, so its child nodes use 0.01 (1.0x)
            // whereas BistroExterior.fbx wraps its children in a 0.016 (1.6x) root node.
            sceneScale = primaryScale_;
            rootTransform.a1 = rootTransform.b2 = rootTransform.c3 = sceneScale;
        }

        size_t skipped = 0, added = 0;
        walk(scene, scene->mRootNode, rootTransform, materialRemap, isPrimary, skipped, added);
        printf("  instances added: %zu, skipped as duplicates: %zu (scale %.2f)\n", added, skipped, sceneScale);

        if (isPrimary && !cfg_.hasSunOverride) extractLightsAndCamera(scene);
        if (cfg_.customPointLights.empty()) extractPointLights(scene, rootTransform, sceneScale);
        importers_.push_back(std::move(importer));
        return true;
    }

    void addCurtainWindowGlass() {
        if (!cfg_.addCurtainGlass) return;
        auto itCurtain = materialByName_.find("MASTER_Curtains");
        auto itGlass = materialByName_.find("MASTER_Glass_Exterior");
        if (itCurtain == materialByName_.end() || itGlass == materialByName_.end()) return;
        uint32_t curtainMat = itCurtain->second;
        uint32_t glassMat = itGlass->second;

        // Collect existing exterior glass triangle centroids so we don't double-up panes on Paris_Building_01.
        std::vector<Vec3> existingGlass;
        for (size_t m = 0; m < materials_.size(); ++m) {
            if (classifyAlpha(materials_[m]) != bistro::kAlphaBlend) continue;
            const MaterialGeometry& g = geometry_[m];
            for (size_t t = 0; t + 2 < g.indices.size(); t += 3) {
                Vec3 c = (g.vertices[g.indices[t]].pos + g.vertices[g.indices[t + 1]].pos +
                          g.vertices[g.indices[t + 2]].pos) *
                         (1.0f / 3.0f);
                existingGlass.push_back(c);
            }
        }

        const MaterialGeometry& src = geometry_[curtainMat];
        MaterialGeometry& dst = geometry_[glassMat];
        size_t addedTris = 0;
        for (size_t t = 0; t + 2 < src.indices.size(); t += 3) {
            const FullVertex& v0 = src.vertices[src.indices[t]];
            const FullVertex& v1 = src.vertices[src.indices[t + 1]];
            const FullVertex& v2 = src.vertices[src.indices[t + 2]];
            Vec3 c = (v0.pos + v1.pos + v2.pos) * (1.0f / 3.0f);
            bool hasPane = false;
            for (const Vec3& eg : existingGlass) {
                Vec3 d = c - eg;
                if (dot(d, d) < 0.35f * 0.35f) {
                    hasPane = true;
                    break;
                }
            }
            if (hasPane) continue;
            uint32_t base = static_cast<uint32_t>(dst.vertices.size());
            dst.vertices.push_back(v0);
            dst.vertices.push_back(v1);
            dst.vertices.push_back(v2);
            dst.indices.push_back(base);
            dst.indices.push_back(base + 1);
            dst.indices.push_back(base + 2);
            addedTris++;
        }

        // Also add exterior glass panes across the 6 ground-floor shopfront display windows
        // (Boulangerie, Tie Shop, and Librairie) that were left unglazed in BistroExterior.fbx.
        struct StorefrontPane {
            float x0, z0, x1, z1, y0, y1;
            float nx, nz;
        };
        const StorefrontPane kStorefrontPanes[] = {
            // Boulangerie right & left display windows
            {4.50f, -17.54f, 6.83f, -20.42f, 1.99f, 4.40f, -0.78f, -0.62f},
            {10.30f, -24.73f, 12.63f, -27.61f, 1.99f, 4.40f, -0.78f, -0.62f},
            // Librairie right & left display windows
            {16.03f, -32.73f, 17.53f, -34.24f, 1.24f, 4.45f, -0.71f, -0.70f},
            {18.52f, -35.88f, 20.16f, -38.41f, 1.24f, 4.45f, -0.84f, -0.54f},
            // Tie Shop right & left display windows
            {4.52f, -42.48f, 6.48f, -44.82f, 1.52f, 3.70f, 0.77f, 0.64f},
            {7.35f, -45.65f, 11.80f, -51.40f, 1.52f, 3.70f, 0.79f, 0.61f},
        };
        for (const auto& sp : kStorefrontPanes) {
            Vec3 n = normalize({sp.nx, 0.0f, sp.nz});
            Vec3 t = normalize({-n.z, 0.0f, n.x});
            FullVertex q[4]{};
            q[0].pos = {sp.x0, sp.y0, sp.z0};
            q[1].pos = {sp.x1, sp.y0, sp.z1};
            q[2].pos = {sp.x1, sp.y1, sp.z1};
            q[3].pos = {sp.x0, sp.y1, sp.z0};
            for (int i = 0; i < 4; ++i) {
                q[i].normal = n;
                q[i].tangent = t;
                q[i].tangentSign = 1.0f;
                q[i].uv[0] = (i == 1 || i == 2) ? 1.0f : 0.0f;
                q[i].uv[1] = (i >= 2) ? 1.0f : 0.0f;
            }
            uint32_t base = static_cast<uint32_t>(dst.vertices.size());
            for (int i = 0; i < 4; ++i) dst.vertices.push_back(q[i]);
            dst.indices.push_back(base + 0);
            dst.indices.push_back(base + 1);
            dst.indices.push_back(base + 2);
            dst.indices.push_back(base + 0);
            dst.indices.push_back(base + 2);
            dst.indices.push_back(base + 3);
            addedTris += 2;
        }
        printf("  added %zu window glass triangles over MASTER_Curtains and storefronts\n", addedTris);
    }

    void prepareGeometry() {
        if (cfg_.deduplicateWithinScene) {
            deduplicateTrianglesAndOffsetDecals();
        } else if (cfg_.weldGrid > 0.0f || cfg_.foliageWeldGrid > 0.0f) {
            for (uint32_t m = 0; m < materials_.size(); ++m) {
                MaterialGeometry& geo = geometry_[m];
                std::vector<int32_t> remap(geo.vertices.size(), -1);
                std::vector<FullVertex> compactVerts;
                compactVerts.reserve(geo.vertices.size());
                for (uint32_t& vi : geo.indices) {
                    if (remap[vi] < 0) {
                        remap[vi] = static_cast<int32_t>(compactVerts.size());
                        compactVerts.push_back(geo.vertices[vi]);
                    }
                    vi = static_cast<uint32_t>(remap[vi]);
                }
                geo.vertices = std::move(compactVerts);
            }
        }
        addCurtainWindowGlass();
    }

    void deduplicateTrianglesAndOffsetDecals() {
        struct QPos {
            int32_t x, y, z;
            bool operator<(const QPos& o) const { return std::tie(x, y, z) < std::tie(o.x, o.y, o.z); }
            bool operator==(const QPos& o) const { return x == o.x && y == o.y && z == o.z; }
        };
        struct TriKey {
            QPos a, b, c;
            bool operator<(const TriKey& o) const { return std::tie(a, b, c) < std::tie(o.a, o.b, o.c); }
        };
        auto quant = [](const Vec3& v) -> QPos {
            return {static_cast<int32_t>(std::lround(v.x * 2000.0f)),
                    static_cast<int32_t>(std::lround(v.y * 2000.0f)),
                    static_cast<int32_t>(std::lround(v.z * 2000.0f))};
        };

        // 1. Deduplicate coplanar triangles that share the exact same 3D vertex positions (within 0.5 mm).
        // Cinema 4D FBX exports both the base mesh and separate *_Polygon_Selection_* nodes at identical
        // coordinates; keep only the highest-priority triangle at each 3D location.
        struct Winner {
            int score;
            uint32_t mat;
            uint32_t triIdx;
        };
        std::map<TriKey, Winner> winners;
        for (uint32_t m = 0; m < materials_.size(); ++m) {
            const MaterialGeometry& geo = geometry_[m];
            uint32_t triCount = static_cast<uint32_t>(geo.indices.size() / 3);
            for (uint32_t t = 0; t < triCount; ++t) {
                QPos p0 = quant(geo.vertices[geo.indices[t * 3]].pos);
                QPos p1 = quant(geo.vertices[geo.indices[t * 3 + 1]].pos);
                QPos p2 = quant(geo.vertices[geo.indices[t * 3 + 2]].pos);
                if (p0 == p1 || p1 == p2 || p0 == p2) continue;
                QPos arr[3] = {p0, p1, p2};
                std::sort(arr, arr + 3);
                TriKey key{arr[0], arr[1], arr[2]};
                int sc = t < geo.triScores.size() ? geo.triScores[t] : 10;
                auto it = winners.find(key);
                if (it == winners.end() || sc > it->second.score) {
                    winners[key] = {sc, m, t};
                }
            }
        }

        size_t removedExact = 0;
        for (uint32_t m = 0; m < materials_.size(); ++m) {
            MaterialGeometry& geo = geometry_[m];
            uint32_t triCount = static_cast<uint32_t>(geo.indices.size() / 3);
            std::vector<uint32_t> newIndices;
            newIndices.reserve(geo.indices.size());
            for (uint32_t t = 0; t < triCount; ++t) {
                QPos p0 = quant(geo.vertices[geo.indices[t * 3]].pos);
                QPos p1 = quant(geo.vertices[geo.indices[t * 3 + 1]].pos);
                QPos p2 = quant(geo.vertices[geo.indices[t * 3 + 2]].pos);
                if (p0 == p1 || p1 == p2 || p0 == p2) continue;
                QPos arr[3] = {p0, p1, p2};
                std::sort(arr, arr + 3);
                TriKey key{arr[0], arr[1], arr[2]};
                auto it = winners.find(key);
                if (it != winners.end() && (it->second.mat != m || it->second.triIdx != t)) {
                    removedExact++;
                    continue;
                }
                newIndices.push_back(geo.indices[t * 3]);
                newIndices.push_back(geo.indices[t * 3 + 1]);
                newIndices.push_back(geo.indices[t * 3 + 2]);
            }
            geo.indices = std::move(newIndices);
        }

        // 2. Offset alpha-masked decal planes along the underlying opaque hull's outward normal so that
        // flush or slightly recessed decals (-1 mm..+1 mm) never Z-fight with the opaque hull.
        struct HullTri {
            Vec3 a, n, center;
        };
        std::vector<HullTri> hullTris;
        for (uint32_t m = 0; m < materials_.size(); ++m) {
            if (classifyAlpha(materials_[m]) != bistro::kAlphaOpaque) continue;
            const MaterialGeometry& geo = geometry_[m];
            for (size_t i = 0; i + 2 < geo.indices.size(); i += 3) {
                const FullVertex& v0 = geo.vertices[geo.indices[i]];
                const FullVertex& v1 = geo.vertices[geo.indices[i + 1]];
                const FullVertex& v2 = geo.vertices[geo.indices[i + 2]];
                Vec3 n = normalize(v0.normal + v1.normal + v2.normal);
                Vec3 c = (v0.pos + v1.pos + v2.pos) * (1.0f / 3.0f);
                hullTris.push_back({v0.pos, n, c});
            }
        }

        size_t offsetDecalTris = 0;
        for (uint32_t m = 0; m < materials_.size(); ++m) {
            if (classifyAlpha(materials_[m]) != bistro::kAlphaMask) continue;
            MaterialGeometry& geo = geometry_[m];
            std::vector<FullVertex> newVerts;
            std::vector<uint32_t> newInds;
            std::vector<int32_t> unshiftedRemap(geo.vertices.size(), -1);
            for (size_t i = 0; i + 2 < geo.indices.size(); i += 3) {
                uint32_t i0 = geo.indices[i], i1 = geo.indices[i + 1], i2 = geo.indices[i + 2];
                Vec3 a = geo.vertices[i0].pos, b = geo.vertices[i1].pos, c = geo.vertices[i2].pos;
                Vec3 gn = normalize(cross(b - a, c - a));
                Vec3 ctr = (a + b + c) * (1.0f / 3.0f);
                float bestAbsDist = 1e9f;
                float bestSignedDist = 0.0f;
                Vec3 bestHullN = gn;
                for (const auto& ht : hullTris) {
                    Vec3 d = ctr - ht.center;
                    if (dot(d, d) > 0.30f * 0.30f) continue;
                    if (std::fabs(dot(gn, ht.n)) < 0.92f) continue;
                    float sdist = dot(ctr - ht.a, ht.n);
                    if (std::fabs(sdist) < bestAbsDist) {
                        bestAbsDist = std::fabs(sdist);
                        bestSignedDist = sdist;
                        bestHullN = ht.n;
                    }
                }
                if (bestAbsDist < 0.003f) {
                    // Place the decal 1.5 mm outside the underlying hull surface.
                    float shift = 0.0015f - bestSignedDist;
                    Vec3 delta = bestHullN * shift;
                    uint32_t base = static_cast<uint32_t>(newVerts.size());
                    FullVertex v0 = geo.vertices[i0], v1 = geo.vertices[i1], v2 = geo.vertices[i2];
                    v0.pos = v0.pos + delta;
                    v1.pos = v1.pos + delta;
                    v2.pos = v2.pos + delta;
                    newVerts.push_back(v0);
                    newVerts.push_back(v1);
                    newVerts.push_back(v2);
                    newInds.push_back(base);
                    newInds.push_back(base + 1);
                    newInds.push_back(base + 2);
                    offsetDecalTris++;
                } else {
                    for (uint32_t vi : {i0, i1, i2}) {
                        if (unshiftedRemap[vi] < 0) {
                            unshiftedRemap[vi] = static_cast<int32_t>(newVerts.size());
                            newVerts.push_back(geo.vertices[vi]);
                        }
                        newInds.push_back(static_cast<uint32_t>(unshiftedRemap[vi]));
                    }
                }
            }
            geo.vertices = std::move(newVerts);
            geo.indices = std::move(newInds);
        }

        // 3. Compact unreferenced vertices from deduplicated materials so AO baking only rays active vertices.
        for (uint32_t m = 0; m < materials_.size(); ++m) {
            MaterialGeometry& geo = geometry_[m];
            std::vector<int32_t> remap(geo.vertices.size(), -1);
            std::vector<FullVertex> compactVerts;
            compactVerts.reserve(geo.vertices.size());
            for (uint32_t& vi : geo.indices) {
                if (remap[vi] < 0) {
                    remap[vi] = static_cast<int32_t>(compactVerts.size());
                    compactVerts.push_back(geo.vertices[vi]);
                }
                vi = static_cast<uint32_t>(remap[vi]);
            }
            geo.vertices = std::move(compactVerts);
        }
        printf("  deduplicated %zu coplanar duplicate triangles; offset %zu decal triangles clear of hull\n",
               removedExact, offsetDecalTris);
    }

    void bakeSkyVisibility(int rays, float maxDistance) {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<skyvis::Triangle> tris;
        const float minArea2 = cfg_.aoMinArea * 2.0f;
        for (size_t m = 0; m < materials_.size(); ++m) {
            if (classifyAlpha(materials_[m]) == bistro::kAlphaBlend) continue;  // glass lets skylight in
            const MaterialGeometry& geo = geometry_[m];
            for (size_t t = 0; t + 2 < geo.indices.size(); t += 3) {
                Vec3 a = geo.vertices[geo.indices[t]].pos, b = geo.vertices[geo.indices[t + 1]].pos,
                     c = geo.vertices[geo.indices[t + 2]].pos;
                Vec3 e1 = b - a, e2 = c - a;
                if (minArea2 > 0.0f) {
                    Vec3 cr = cross(e1, e2);
                    if (dot(cr, cr) < minArea2 * minArea2) continue;
                }
                tris.push_back({{a.x, a.y, a.z}, {e1.x, e1.y, e1.z}, {e2.x, e2.y, e2.z}});
            }
        }
        size_t triCount = tris.size();
        skyvis::Bvh bvh;
        bvh.build(std::move(tris));
        auto t1 = std::chrono::steady_clock::now();
        printf("  sky visibility: BVH over %zu triangles (%zu nodes) built in %.1fs\n", triCount, bvh.nodeCount(),
               std::chrono::duration<double>(t1 - t0).count());
        std::vector<skyvis::Float3> positions, normals;
        for (auto& geo : geometry_) {
            for (auto& v : geo.vertices) {
                positions.push_back({v.pos.x, v.pos.y, v.pos.z});
                normals.push_back({v.normal.x, v.normal.y, v.normal.z});
            }
        }
        std::vector<float> vis;
        skyvis::computeVisibility(bvh, positions, normals, vis, rays, maxDistance);
        size_t k = 0;
        double sum = 0;
        for (auto& geo : geometry_) {
            for (auto& v : geo.vertices) {
                float rawVis = vis[k++];
                v.skyVisibility = cfg_.aoMinVis + (1.0f - cfg_.aoMinVis) * rawVis;
                sum += v.skyVisibility;
            }
        }
        if (getenv("BISTRO_AO_DEBUG")) {
            for (size_t m = 0; m < materials_.size(); ++m) {
                double ms = 0, ny = 0;
                for (auto& v : geometry_[m].vertices) {
                    ms += v.skyVisibility;
                    ny += v.normal.y;
                }
                size_t n = geometry_[m].vertices.size();
                if (n > 20000 || materials_[m].name.find("Pavement") != std::string::npos)
                    printf("   %-55s verts %8zu  vis %.3f  mean normal.y %.2f\n", materials_[m].name.c_str(), n,
                           ms / n, ny / n);
            }
        }
        auto t2 = std::chrono::steady_clock::now();
        printf("  sky visibility: %zu vertices x %d rays in %.1fs (mean visibility %.2f)\n", positions.size(), rays,
               std::chrono::duration<double>(t2 - t1).count(), positions.empty() ? 0.0 : sum / positions.size());
    }

    bool write(const std::string& outDir) {
        std::vector<bistro::GpuVertex> vertices;
        std::vector<uint16_t> indices;
        std::vector<bistro::ClusterRecord> clusters;

        for (uint32_t m = 0; m < materials_.size(); ++m) {
            MaterialGeometry& geo = geometry_[m];
            if (geo.indices.empty()) continue;
            uint32_t triCount = static_cast<uint32_t>(geo.indices.size() / 3);
            std::vector<uint32_t> tris(triCount);
            std::iota(tris.begin(), tris.end(), 0);
            std::vector<Vec3> centroids(triCount);
            for (uint32_t t = 0; t < triCount; ++t) {
                Vec3 a = geo.vertices[geo.indices[t * 3]].pos;
                Vec3 b = geo.vertices[geo.indices[t * 3 + 1]].pos;
                Vec3 c = geo.vertices[geo.indices[t * 3 + 2]].pos;
                centroids[t] = (a + b + c) * (1.0f / 3.0f);
            }
            std::vector<std::pair<uint32_t, uint32_t>> leaves;  // ranges into tris
            split(geo, centroids, tris, 0, triCount, leaves);

            std::vector<int32_t> remap(geo.vertices.size(), -1);
            for (auto [begin, end] : leaves) {
                // Restore the original (cache-optimised) triangle order within the leaf.
                std::sort(tris.begin() + begin, tris.begin() + end);
                bistro::ClusterRecord rec{};
                rec.firstIndex = static_cast<uint32_t>(indices.size());
                rec.vertexOffset = static_cast<int32_t>(vertices.size());
                rec.materialIndex = m;
                Vec3 mn{FLT_MAX, FLT_MAX, FLT_MAX}, mx{-FLT_MAX, -FLT_MAX, -FLT_MAX};
                std::vector<uint32_t> touched;
                for (uint32_t i = begin; i < end; ++i) {
                    uint32_t t = tris[i];
                    for (int k = 0; k < 3; ++k) {
                        uint32_t vi = geo.indices[t * 3 + k];
                        if (remap[vi] < 0) {
                            remap[vi] = static_cast<int32_t>(vertices.size()) - rec.vertexOffset;
                            touched.push_back(vi);
                            vertices.push_back(pack(geo.vertices[vi]));
                            Vec3 p = geo.vertices[vi].pos;
                            mn = {std::min(mn.x, p.x), std::min(mn.y, p.y), std::min(mn.z, p.z)};
                            mx = {std::max(mx.x, p.x), std::max(mx.y, p.y), std::max(mx.z, p.z)};
                        }
                        if (remap[vi] > 65535) {
                            fprintf(stderr, "Cluster exceeded 16-bit index range\n");
                            return false;
                        }
                        indices.push_back(static_cast<uint16_t>(remap[vi]));
                    }
                }
                for (uint32_t vi : touched) remap[vi] = -1;
                rec.indexCount = static_cast<uint32_t>(indices.size()) - rec.firstIndex;
                rec.aabbMin[0] = mn.x; rec.aabbMin[1] = mn.y; rec.aabbMin[2] = mn.z;
                rec.aabbMax[0] = mx.x; rec.aabbMax[1] = mx.y; rec.aabbMax[2] = mx.z;
                clusters.push_back(rec);
            }
        }

        // Pad the index stream so that the vertex data that follows stays 4-byte aligned.
        if (indices.size() % 2) indices.push_back(0);

        bistro::SceneHeader header{};
        std::memcpy(header.magic, bistro::kSceneMagic, 4);
        header.version = bistro::kSceneVersion;
        header.vertexCount = static_cast<uint32_t>(vertices.size());
        header.indexCount = static_cast<uint32_t>(indices.size());
        header.clusterCount = static_cast<uint32_t>(clusters.size());
        header.materialCount = static_cast<uint32_t>(materials_.size());
        header.textureCount = static_cast<uint32_t>(textures_.size());
        Vec3 mn{FLT_MAX, FLT_MAX, FLT_MAX}, mx{-FLT_MAX, -FLT_MAX, -FLT_MAX};
        for (auto& c : clusters) {
            mn = {std::min(mn.x, c.aabbMin[0]), std::min(mn.y, c.aabbMin[1]), std::min(mn.z, c.aabbMin[2])};
            mx = {std::max(mx.x, c.aabbMax[0]), std::max(mx.y, c.aabbMax[1]), std::max(mx.z, c.aabbMax[2])};
        }
        header.boundsMin[0] = mn.x; header.boundsMin[1] = mn.y; header.boundsMin[2] = mn.z;
        header.boundsMax[0] = mx.x; header.boundsMax[1] = mx.y; header.boundsMax[2] = mx.z;
        std::memcpy(header.sunDirection, sunDir_, sizeof(sunDir_));
        std::memcpy(header.sunColor, sunColor_, sizeof(sunColor_));

        std::vector<bistro::MaterialRecord> matRecords;
        size_t maskCount = 0, blendCount = 0, foldedOrm = 0, foldedBase = 0, translucentCount = 0;
        for (auto& m : materials_) {
            bistro::MaterialRecord r{};
            r.baseColorTexture = -1;
            r.normalTexture = textureIndex(m.normal, bistro::kTextureNormal);
            r.roughnessFactor = m.hasRoughnessFactor ? m.roughnessFactor : (m.orm.empty() ? 0.75f : 1.0f);
            r.metallicFactor = m.hasMetallicFactor ? m.metallicFactor : (m.orm.empty() ? 0.0f : 1.0f);
            r.ormTexture = -1;
            if (!m.orm.empty()) {
                auto st = stats_.all.find(m.orm);
                if (st != stats_.all.end() && st->second.maxStdDev < 0.02f) {
                    // Constant texture: fold into material factors and skip the texture fetch entirely.
                    r.roughnessFactor = st->second.mean[1];
                    r.metallicFactor = st->second.mean[2];
                    if (st->second.mean[0] > 0.95f && st->second.mean[1] > 0.95f && st->second.mean[2] > 0.95f) {
                        // Plain white placeholder map: use a neutral dielectric instead of rough metal.
                        r.roughnessFactor = 0.8f;
                        r.metallicFactor = 0.0f;
                    }
                    foldedOrm++;
                } else {
                    r.ormTexture = textureIndex(m.orm, bistro::kTextureOrm);
                }
            }
            r.emissiveTexture = -1;
            if (!m.emissive.empty()) {
                auto est = stats_.all.find(m.emissive);
                bool allBlackEmissive = est != stats_.all.end() && est->second.maxStdDev < 0.01f &&
                                        est->second.mean[0] < 0.01f && est->second.mean[1] < 0.01f &&
                                        est->second.mean[2] < 0.01f;
                if (!allBlackEmissive) {
                    r.emissiveTexture = textureIndex(m.emissive, bistro::kTextureEmissive);
                }
            }
            std::memcpy(r.baseColorFactor, m.baseColorFactor, sizeof(r.baseColorFactor));
            std::memcpy(r.emissiveFactor, m.emissiveFactor, sizeof(r.emissiveFactor));
            r.alphaCutoff = 0.5f;
            r.alphaMode = classifyAlpha(m);
            r.flags = 0;
            if (m.doubleSided || containsCI(m.name, "DoubleSided") || containsCI(m.name, "TwoSided") ||
                r.alphaMode != bistro::kAlphaOpaque) {
                r.flags |= bistro::kMaterialDoubleSided;
            }
            if (isTranslucentMaterial(m.name) && r.alphaMode != bistro::kAlphaBlend) {
                r.flags |= bistro::kMaterialTranslucent;
                translucentCount++;
            }
            if (r.emissiveTexture >= 0) {
                r.flags |= bistro::kMaterialEmissive;
                r.emissiveFactor[0] = r.emissiveFactor[1] = r.emissiveFactor[2] = 1.0f;
            } else if (m.name == "light_bulb") {
                r.flags |= bistro::kMaterialEmissive;
                r.emissiveFactor[0] = 2.5f;
                r.emissiveFactor[1] = 1.8f;
                r.emissiveFactor[2] = 0.9f;
                r.baseColorFactor[0] = 1.0f;
                r.baseColorFactor[1] = 0.9f;
                r.baseColorFactor[2] = 0.7f;
                r.baseColorFactor[3] = 1.0f;
            } else if (m.name == "River") {
                r.roughnessFactor = 0.06f;
                r.metallicFactor = 0.0f;
            }
            if (r.alphaMode == bistro::kAlphaBlend) {
                r.baseColorTexture = textureIndex(m.baseColor, bistro::kTextureBaseColor);
                // Ensure all glass surfaces (windows, bottles, glassware, display cases) are mirror-smooth
                // rather than inheriting the 0.737 placeholder roughness from constant 16x16 Specular maps.
                r.roughnessFactor = containsCI(m.name, "Dirty") ? 0.08f : 0.04f;
                r.metallicFactor = 0.0f;
                // The Bistro glass was authored for path tracing (specular transmission) and its base colour
                // textures are fully opaque. Give it a raster-friendly opacity so windows are see-through.
                auto st = stats_.all.find(m.baseColor);
                bool textureHasAlpha =
                    st != stats_.all.end() && (st->second.translucent > 0.01f || st->second.cutout > 0.01f);
                if (!textureHasAlpha) {
                    float alpha = 0.14f;
                    if (containsCI(m.name, "Dirty")) alpha = 0.24f;
                    else if (containsCI(m.name, "Emissive")) alpha = 0.6f;
                    else if (containsCI(m.name, "Bottle") || containsCI(m.name, "Wine")) alpha = 0.42f;
                    r.baseColorFactor[3] = alpha;
                }
            } else if (containsCI(m.name, "Frosted") || containsCI(m.name, "Frozen")) {
                r.baseColorTexture = textureIndex(m.baseColor, bistro::kTextureBaseColor);
                // Opaque frosted glass on unbacked exterior doors/transoms: dark tinted glossy glass.
                r.roughnessFactor = 0.08f;
                r.metallicFactor = 0.0f;
                r.baseColorFactor[0] = 0.22f;
                r.baseColorFactor[1] = 0.25f;
                r.baseColorFactor[2] = 0.27f;
            } else if (r.alphaMode == bistro::kAlphaOpaque && !(r.flags & bistro::kMaterialEmissive)) {
                auto st = stats_.all.find(m.baseColor);
                if (st != stats_.all.end() && st->second.maxStdDev < 0.01f &&
                    st->second.mean[0] < 0.01f && st->second.mean[1] < 0.01f && st->second.mean[2] < 0.01f) {
                    // Replace 16x16 all-zero black placeholder BaseColor textures with realistic colours.
                    r.baseColorTexture = -1;
                    if (containsCI(m.name, "Rollup_Door")) {
                        r.baseColorFactor[0] = 0.36f;
                        r.baseColorFactor[1] = 0.39f;
                        r.baseColorFactor[2] = 0.42f;
                        r.roughnessFactor = 0.45f;
                        r.metallicFactor = 0.70f;
                    } else if (containsCI(m.name, "Toffee")) {
                        r.baseColorFactor[0] = 0.55f;
                        r.baseColorFactor[1] = 0.32f;
                        r.baseColorFactor[2] = 0.14f;
                        r.roughnessFactor = 0.35f;
                    } else {
                        r.baseColorFactor[0] = 0.24f;
                        r.baseColorFactor[1] = 0.24f;
                        r.baseColorFactor[2] = 0.26f;
                    }
                    foldedBase++;
                } else if (cfg_.foldConstantBaseColor && st != stats_.all.end() && st->second.maxStdDev < 0.005f &&
                           st->second.cutout < 0.001f && st->second.translucent < 0.001f) {
                    r.baseColorTexture = -1;
                    r.baseColorFactor[0] = srgbToLinear(st->second.mean[0]);
                    r.baseColorFactor[1] = srgbToLinear(st->second.mean[1]);
                    r.baseColorFactor[2] = srgbToLinear(st->second.mean[2]);
                    r.baseColorFactor[3] = 1.0f;
                    foldedBase++;
                } else {
                    r.baseColorTexture = textureIndex(m.baseColor, bistro::kTextureBaseColor);
                }
            } else {
                r.baseColorTexture = textureIndex(m.baseColor, bistro::kTextureBaseColor);
            }
            if (r.alphaMode == bistro::kAlphaMask) maskCount++;
            if (r.alphaMode == bistro::kAlphaBlend) blendCount++;
            std::snprintf(r.name, sizeof(r.name), "%s", m.name.c_str());
            matRecords.push_back(r);
        }
        // Texture indices may have grown while building material records.
        header.textureCount = static_cast<uint32_t>(textures_.size());
        std::vector<bistro::TextureRecord> texRecords;
        for (auto& t : textures_) {
            bistro::TextureRecord r{};
            std::snprintf(r.name, sizeof(r.name), "%s", t.first.c_str());
            r.kind = t.second;
            texRecords.push_back(r);
        }
        header.cameraKeyCount = static_cast<uint32_t>(cameraKeys_.size());
        header.pointLightCount = static_cast<uint32_t>(pointLights_.size());

        std::string path = outDir + "/scene.bin";
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            fprintf(stderr, "Cannot write %s\n", path.c_str());
            return false;
        }
        out.write(reinterpret_cast<const char*>(&header), sizeof(header));
        out.write(reinterpret_cast<const char*>(texRecords.data()), texRecords.size() * sizeof(bistro::TextureRecord));
        out.write(reinterpret_cast<const char*>(matRecords.data()), matRecords.size() * sizeof(bistro::MaterialRecord));
        out.write(reinterpret_cast<const char*>(clusters.data()), clusters.size() * sizeof(bistro::ClusterRecord));
        out.write(reinterpret_cast<const char*>(cameraKeys_.data()), cameraKeys_.size() * sizeof(bistro::CameraKey));
        out.write(reinterpret_cast<const char*>(pointLights_.data()),
                  pointLights_.size() * sizeof(bistro::PointLightRecord));
        out.write(reinterpret_cast<const char*>(indices.data()), indices.size() * sizeof(uint16_t));
        out.write(reinterpret_cast<const char*>(vertices.data()), vertices.size() * sizeof(bistro::GpuVertex));
        out.close();

        size_t triangles = 0;
        for (auto& c : clusters) triangles += c.indexCount / 3;
        printf("  folded %zu constant ORM textures and %zu constant BaseColor textures into material factors\n",
               foldedOrm, foldedBase);
        printf("Wrote %s\n  vertices %zu, triangles %zu, clusters %zu, materials %zu (mask %zu, blend %zu, translucent %zu), textures %zu\n",
               path.c_str(), vertices.size(), triangles, clusters.size(), materials_.size(), maskCount, blendCount,
               translucentCount, textures_.size());
        printf("  bounds (%.2f %.2f %.2f) - (%.2f %.2f %.2f)\n", mn.x, mn.y, mn.z, mx.x, mx.y, mx.z);

        writeTextureList();
        writeHeightMap(mn, mx);
        return true;
    }

private:
    uint32_t registerMaterial(const aiMaterial* mat) {
        aiString name;
        mat->Get(AI_MATKEY_NAME, name);
        std::string key = name.C_Str();
        auto it = materialByName_.find(key);
        if (it != materialByName_.end()) return it->second;

        MaterialInfo info;
        info.name = key;
        aiString p;
        if (mat->GetTexture(aiTextureType_DIFFUSE, 0, &p) == AI_SUCCESS ||
            mat->GetTexture(aiTextureType_BASE_COLOR, 0, &p) == AI_SUCCESS) {
            info.baseColor = textureNameFromPath(p, stats_);
        }
        if (mat->GetTexture(aiTextureType_NORMALS, 0, &p) == AI_SUCCESS) {
            info.normal = textureNameFromPath(p, stats_);
        }
        if (mat->GetTexture(aiTextureType_SPECULAR, 0, &p) == AI_SUCCESS ||
            mat->GetTexture(aiTextureType_UNKNOWN, 0, &p) == AI_SUCCESS ||
            mat->GetTexture(aiTextureType_METALNESS, 0, &p) == AI_SUCCESS ||
            mat->GetTexture(aiTextureType_DIFFUSE_ROUGHNESS, 0, &p) == AI_SUCCESS) {
            info.orm = textureNameFromPath(p, stats_);
        }
        if (mat->GetTexture(aiTextureType_EMISSIVE, 0, &p) == AI_SUCCESS) {
            info.emissive = textureNameFromPath(p, stats_);
        }
        aiColor4D baseCol;
        if (mat->Get(AI_MATKEY_BASE_COLOR, baseCol) == AI_SUCCESS) {
            info.baseColorFactor[0] = baseCol.r;
            info.baseColorFactor[1] = baseCol.g;
            info.baseColorFactor[2] = baseCol.b;
            info.baseColorFactor[3] = baseCol.a;
        } else if (info.baseColor.empty() && mat->Get(AI_MATKEY_COLOR_DIFFUSE, baseCol) == AI_SUCCESS) {
            info.baseColorFactor[0] = baseCol.r;
            info.baseColorFactor[1] = baseCol.g;
            info.baseColorFactor[2] = baseCol.b;
        }
        float rf = 1.0f;
        if (mat->Get(AI_MATKEY_ROUGHNESS_FACTOR, rf) == AI_SUCCESS) {
            info.roughnessFactor = rf;
            info.hasRoughnessFactor = true;
        }
        float mf = 1.0f;
        if (mat->Get(AI_MATKEY_METALLIC_FACTOR, mf) == AI_SUCCESS) {
            info.metallicFactor = mf;
            info.hasMetallicFactor = true;
        }
        int twoSided = 0;
        if (mat->Get(AI_MATKEY_TWOSIDED, twoSided) == AI_SUCCESS && twoSided != 0) {
            info.doubleSided = true;
        }
        uint32_t index = static_cast<uint32_t>(materials_.size());
        materials_.push_back(info);
        geometry_.emplace_back();
        materialByName_[key] = index;
        return index;
    }

    void walk(const aiScene* scene, const aiNode* node, const aiMatrix4x4& parent,
              const std::vector<uint32_t>& materialRemap, bool isPrimary, size_t& skipped, size_t& added) {
        std::string nodeName = node->mName.C_Str();
        for (const auto& sub : cfg_.skipNodeSubstrings) {
            if (containsCI(nodeName, sub.c_str())) {
                skipped++;
                return;
            }
        }
        aiMatrix4x4 world = parent * node->mTransformation;
        if (node->mNumMeshes > 0) {
            // Bounds of this node instance in world space, used to detect duplicated geometry between the
            // exterior and interior files (the interior file contains part of the bistro building shell).
            Vec3 mn{FLT_MAX, FLT_MAX, FLT_MAX}, mx{-FLT_MAX, -FLT_MAX, -FLT_MAX};
            size_t nodeTris = 0;
            for (unsigned i = 0; i < node->mNumMeshes; ++i) {
                const aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
                nodeTris += mesh->mNumFaces;
                for (unsigned v = 0; v < mesh->mNumVertices; ++v) {
                    aiVector3D p = world * mesh->mVertices[v];
                    mn = {std::min(mn.x, p.x), std::min(mn.y, p.y), std::min(mn.z, p.z)};
                    mx = {std::max(mx.x, p.x), std::max(mx.y, p.y), std::max(mx.z, p.z)};
                }
            }
            bool duplicate = false;
            if (!isPrimary || cfg_.deduplicateWithinScene) {
                float tol = isPrimary ? 1e-3f : 0.10f;
                for (const auto& [pmn, pmx, ptris] : primaryBounds_) {
                    if (isPrimary && ptris != nodeTris) continue;
                    Vec3 dmn = pmn - mn, dmx = pmx - mx;
                    if (std::fabs(dmn.x) + std::fabs(dmn.y) + std::fabs(dmn.z) + std::fabs(dmx.x) +
                            std::fabs(dmx.y) + std::fabs(dmx.z) < tol) {
                        duplicate = true;
                        break;
                    }
                }
            }
            if (isPrimary && !duplicate) {
                primaryBounds_.push_back({mn, mx, nodeTris});
            }
            if (duplicate) {
                skipped++;
            } else {
                added++;
                bool isPolySel = containsCI(nodeName, "Polygon_Selection");
                for (unsigned i = 0; i < node->mNumMeshes; ++i) {
                    const aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
                    uint32_t matIdx = materialRemap[mesh->mMaterialIndex];
                    const MaterialInfo& mi = materials_[matIdx];
                    int score = 10;
                    if (mi.name == "Default_Material" || mi.name == "lambert1") {
                        score = 0;
                    } else if (!mi.emissive.empty()) {
                        score = 50;
                    } else if (isPolySel && nodeTris < 1000) {
                        score = 40;
                    } else if (!isPolySel && !mi.baseColor.empty()) {
                        score = 30;
                    } else if (!mi.baseColor.empty()) {
                        score = 20;
                    }
                    addMesh(mesh, world, matIdx, score);
                }
            }
        }
        for (unsigned c = 0; c < node->mNumChildren; ++c)
            walk(scene, node->mChildren[c], world, materialRemap, isPrimary, skipped, added);
    }

    bool isTranslucentMaterial(const std::string& name) const {
        bool included = false;
        for (const auto& s : cfg_.translucentInclude) {
            if (containsCI(name, s.c_str())) { included = true; break; }
        }
        if (!included) return false;
        for (const auto& s : cfg_.translucentExclude) {
            if (containsCI(name, s.c_str())) return false;
        }
        return true;
    }

    void addMesh(const aiMesh* mesh, const aiMatrix4x4& world, uint32_t material, int score) {
        if (!(mesh->mPrimitiveTypes & aiPrimitiveType_TRIANGLE) || !mesh->HasNormals()) return;
        MaterialGeometry& geo = geometry_[material];
        const MaterialInfo& mi = materials_[material];
        aiMatrix3x3 normalMat = aiMatrix3x3(world);
        normalMat.Inverse().Transpose();
        aiMatrix3x3 linear = aiMatrix3x3(world);
        bool flip = linear.Determinant() < 0;

        bool isFoliageOrCloth = containsCI(mi.name, "Ivy") || containsCI(mi.name, "Leaf") ||
                                containsCI(mi.name, "Leaves") || containsCI(mi.name, "Curtain") ||
                                containsCI(mi.name, "Lianas") || containsCI(mi.name, "Branches") ||
                                classifyAlpha(mi) == bistro::kAlphaMask;
        float grid = isFoliageOrCloth && cfg_.foliageWeldGrid > 0.0f ? cfg_.foliageWeldGrid : cfg_.weldGrid;

        struct WeldKey {
            int32_t x, y, z, u, v, n;
            bool operator==(const WeldKey& o) const {
                return x == o.x && y == o.y && z == o.z && u == o.u && v == o.v && n == o.n;
            }
        };
        struct WeldHash {
            size_t operator()(const WeldKey& k) const {
                size_t h = static_cast<size_t>(k.x) * 73856093u ^ static_cast<size_t>(k.y) * 19349663u ^
                           static_cast<size_t>(k.z) * 83492791u;
                h ^= (static_cast<size_t>(k.u) * 2654435761u) + (static_cast<size_t>(k.v) * 40503u) +
                     static_cast<size_t>(k.n);
                return h;
            }
        };
        std::unordered_map<WeldKey, uint32_t, WeldHash> welded;
        std::vector<uint32_t> vRemap(mesh->mNumVertices);
        if (grid > 0.0f) welded.reserve(mesh->mNumVertices / 2 + 16);

        uint32_t base = static_cast<uint32_t>(geo.vertices.size());
        for (unsigned v = 0; v < mesh->mNumVertices; ++v) {
            FullVertex fv;
            aiVector3D p = world * mesh->mVertices[v];
            fv.pos = {p.x, p.y, p.z};
            aiVector3D n = normalMat * mesh->mNormals[v];
            fv.normal = normalize({n.x, n.y, n.z});
            if (mesh->HasTangentsAndBitangents()) {
                aiVector3D t = linear * mesh->mTangents[v];
                aiVector3D b = linear * mesh->mBitangents[v];
                Vec3 tv{t.x, t.y, t.z};
                // Gram-Schmidt orthogonalise against the normal.
                tv = tv - fv.normal * dot(fv.normal, tv);
                if (dot(tv, tv) < 1e-12f || !std::isfinite(tv.x)) {
                    Vec3 axis = std::fabs(fv.normal.y) < 0.99f ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
                    tv = cross(axis, fv.normal);
                }
                fv.tangent = normalize(tv);
                Vec3 bv{b.x, b.y, b.z};
                fv.tangentSign = dot(cross(fv.normal, fv.tangent), bv) < 0.0f ? -1.0f : 1.0f;
            } else {
                Vec3 axis = std::fabs(fv.normal.y) < 0.99f ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
                fv.tangent = normalize(cross(axis, fv.normal));
            }
            if (mesh->HasTextureCoords(0)) {
                fv.uv[0] = mesh->mTextureCoords[0][v].x;
                fv.uv[1] = mesh->mTextureCoords[0][v].y;
            }
            if (grid > 0.0f) {
                int32_t qx = static_cast<int32_t>(std::lround(fv.pos.x / grid));
                int32_t qy = static_cast<int32_t>(std::lround(fv.pos.y / grid));
                int32_t qz = static_cast<int32_t>(std::lround(fv.pos.z / grid));
                int32_t qu = isFoliageOrCloth ? 0 : static_cast<int32_t>(std::lround(fv.uv[0] * 4.0f));
                int32_t qv = isFoliageOrCloth ? 0 : static_cast<int32_t>(std::lround(fv.uv[1] * 4.0f));
                int32_t qn = isFoliageOrCloth
                                 ? 0
                                 : ((fv.normal.x >= 0 ? 1 : 0) | (fv.normal.y >= 0 ? 2 : 0) | (fv.normal.z >= 0 ? 4 : 0));
                WeldKey wk{qx, qy, qz, qu, qv, qn};
                auto [it, inserted] = welded.emplace(wk, static_cast<uint32_t>(geo.vertices.size()));
                if (inserted) {
                    geo.vertices.push_back(fv);
                }
                vRemap[v] = it->second;
            } else {
                vRemap[v] = base + v;
                geo.vertices.push_back(fv);
            }
        }
        for (unsigned f = 0; f < mesh->mNumFaces; ++f) {
            const aiFace& face = mesh->mFaces[f];
            if (face.mNumIndices != 3) continue;
            uint32_t a = vRemap[face.mIndices[0]], b = vRemap[face.mIndices[1]], c = vRemap[face.mIndices[2]];
            if (a == b || b == c || a == c) continue;
            if (flip) std::swap(b, c);
            geo.indices.push_back(a);
            geo.indices.push_back(b);
            geo.indices.push_back(c);
            geo.triScores.push_back(score);
        }
    }

    void split(const MaterialGeometry& geo, const std::vector<Vec3>& centroids, std::vector<uint32_t>& tris,
               uint32_t begin, uint32_t end, std::vector<std::pair<uint32_t, uint32_t>>& leaves) {
        Vec3 mn{FLT_MAX, FLT_MAX, FLT_MAX}, mx{-FLT_MAX, -FLT_MAX, -FLT_MAX};
        for (uint32_t i = begin; i < end; ++i) {
            for (int k = 0; k < 3; ++k) {
                Vec3 p = geo.vertices[geo.indices[tris[i] * 3 + k]].pos;
                mn = {std::min(mn.x, p.x), std::min(mn.y, p.y), std::min(mn.z, p.z)};
                mx = {std::max(mx.x, p.x), std::max(mx.y, p.y), std::max(mx.z, p.z)};
            }
        }
        Vec3 ext = mx - mn;
        float maxExt = std::max(ext.x, std::max(ext.y, ext.z));
        uint32_t count = end - begin;
        if (count <= kMaxClusterTriangles && (maxExt <= kMaxClusterExtent || count <= 64)) {
            leaves.emplace_back(begin, end);
            return;
        }
        // Split on the longest axis of the centroid bounds at the median.
        Vec3 cmn{FLT_MAX, FLT_MAX, FLT_MAX}, cmx{-FLT_MAX, -FLT_MAX, -FLT_MAX};
        for (uint32_t i = begin; i < end; ++i) {
            Vec3 c = centroids[tris[i]];
            cmn = {std::min(cmn.x, c.x), std::min(cmn.y, c.y), std::min(cmn.z, c.z)};
            cmx = {std::max(cmx.x, c.x), std::max(cmx.y, c.y), std::max(cmx.z, c.z)};
        }
        Vec3 cext = cmx - cmn;
        int axis = cext.x > cext.y ? (cext.x > cext.z ? 0 : 2) : (cext.y > cext.z ? 1 : 2);
        uint32_t mid = begin + count / 2;
        std::nth_element(tris.begin() + begin, tris.begin() + mid, tris.begin() + end,
                         [&](uint32_t a, uint32_t b) { return comp(centroids[a], axis) < comp(centroids[b], axis); });
        split(geo, centroids, tris, begin, mid, leaves);
        split(geo, centroids, tris, mid, end, leaves);
    }

    static bistro::GpuVertex pack(const FullVertex& v) {
        bistro::GpuVertex g{};
        g.position[0] = v.pos.x;
        g.position[1] = v.pos.y;
        g.position[2] = v.pos.z;
        octEncode(v.normal, g.normal);
        g.tangent[0] = toSnorm8(v.tangent.x);
        g.tangent[1] = toSnorm8(v.tangent.y);
        g.tangent[2] = toSnorm8(v.tangent.z);
        g.tangent[3] = v.tangentSign < 0 ? -127 : 127;
        g.uv[0] = v.uv[0];
        g.uv[1] = v.uv[1];
        g.extra[0] = static_cast<uint8_t>(std::lround(std::clamp(v.skyVisibility, 0.0f, 1.0f) * 255.0f));
        g.extra[1] = g.extra[2] = g.extra[3] = 255;
        return g;
    }

    int32_t textureIndex(const std::string& name, uint32_t kind) {
        if (name.empty()) return -1;
        for (size_t i = 0; i < textures_.size(); ++i) {
            if (textures_[i].first == name) return static_cast<int32_t>(i);
        }
        textures_.emplace_back(name, kind);
        return static_cast<int32_t>(textures_.size() - 1);
    }

    uint32_t classifyAlpha(const MaterialInfo& m) const {
        if (containsCI(m.name, "Frosted") || containsCI(m.name, "Frozen")) return bistro::kAlphaOpaque;
        if (containsCI(m.name, "Glass") || containsCI(m.name, "Water") || containsCI(m.name, "Ice") ||
            containsCI(m.name, "Wine") || containsCI(m.name, "Beer")) {
            // Opaque labels/bottles with "Wine" in the name but no translucency should stay opaque.
            auto it = stats_.translucentFraction.find(m.baseColor);
            auto cut = stats_.cutoutFraction.find(m.baseColor);
            bool hasAlpha = (it != stats_.translucentFraction.end() && it->second > 0.01f) ||
                            (cut != stats_.cutoutFraction.end() && cut->second > 0.01f);
            if (containsCI(m.name, "Glass") || hasAlpha) return bistro::kAlphaBlend;
        }
        auto it = stats_.cutoutFraction.find(m.baseColor);
        if (it != stats_.cutoutFraction.end() && it->second > 0.002f) return bistro::kAlphaMask;
        return bistro::kAlphaOpaque;
    }

    void extractLightsAndCamera(const aiScene* scene) {
        for (unsigned i = 0; i < scene->mNumLights; ++i) {
            const aiLight* light = scene->mLights[i];
            if (light->mType != aiLightSource_DIRECTIONAL) continue;
            const aiNode* node = scene->mRootNode->FindNode(light->mName);
            aiMatrix4x4 world;
            for (const aiNode* n = node; n; n = n->mParent) world = n->mTransformation * world;
            aiVector3D d = aiMatrix3x3(world) * light->mDirection;
            d.Normalize();
            // Store the direction *towards* the light.
            sunDir_[0] = -d.x;
            sunDir_[1] = -d.y;
            sunDir_[2] = -d.z;
            sunColor_[0] = light->mColorDiffuse.r;
            sunColor_[1] = light->mColorDiffuse.g;
            sunColor_[2] = light->mColorDiffuse.b;
            printf("  directional light '%s': towards-light (%.3f %.3f %.3f) colour (%.1f %.1f %.1f)\n",
                   light->mName.C_Str(), sunDir_[0], sunDir_[1], sunDir_[2], sunColor_[0], sunColor_[1],
                   sunColor_[2]);
        }
        // Sample the embedded camera fly-through animation (used as a reference when authoring paths).
        if (scene->mNumAnimations == 0 || scene->mNumCameras == 0) return;
        const aiAnimation* anim = scene->mAnimations[0];
        const aiNodeAnim* tr = nullptr;
        const aiNodeAnim* rot = nullptr;
        for (unsigned c = 0; c < anim->mNumChannels; ++c) {
            std::string n = anim->mChannels[c]->mNodeName.C_Str();
            if (n.find("Camera") == 0 && n.find("Translation") != std::string::npos) tr = anim->mChannels[c];
            if (n.find("Camera") == 0 && n.find("Rotation") != std::string::npos) rot = anim->mChannels[c];
        }
        if (!tr) return;
        const aiNode* trNode = scene->mRootNode->FindNode(tr->mNodeName);
        aiMatrix4x4 parentWorld;
        for (const aiNode* n = trNode ? trNode->mParent : nullptr; n; n = n->mParent)
            parentWorld = n->mTransformation * parentWorld;
        for (unsigned k = 0; k < tr->mNumPositionKeys; ++k) {
            bistro::CameraKey key{};
            key.time = static_cast<float>(tr->mPositionKeys[k].mTime / anim->mTicksPerSecond);
            aiVector3D p = parentWorld * tr->mPositionKeys[k].mValue;
            key.position[0] = p.x;
            key.position[1] = p.y;
            key.position[2] = p.z;
            if (rot && k < rot->mNumRotationKeys) {
                aiMatrix3x3 r = rot->mRotationKeys[k].mValue.GetMatrix();
                // Assimp/FBX cameras look down +X in their local frame.
                aiVector3D fwd = aiMatrix3x3(parentWorld) * (r * aiVector3D(1, 0, 0));
                fwd.Normalize();
                key.forward[0] = fwd.x;
                key.forward[1] = fwd.y;
                key.forward[2] = fwd.z;
            }
            printf("  camera key t=%.2f pos (%.2f %.2f %.2f) fwd (%.2f %.2f %.2f)\n", key.time, key.position[0],
                   key.position[1], key.position[2], key.forward[0], key.forward[1], key.forward[2]);
            cameraKeys_.push_back(key);
        }
    }

    void extractPointLights(const aiScene* scene, const aiMatrix4x4& rootTransform, float sceneScale) {
        for (unsigned i = 0; i < scene->mNumLights; ++i) {
            const aiLight* light = scene->mLights[i];
            if (light->mType != aiLightSource_POINT && light->mType != aiLightSource_SPOT) continue;
            const aiNode* node = scene->mRootNode->FindNode(light->mName);
            aiMatrix4x4 world;
            for (const aiNode* n = node; n; n = n->mParent) world = n->mTransformation * world;
            world = rootTransform * world;
            aiVector3D p = world * light->mPosition;
            bistro::PointLightRecord r{};
            r.position[0] = p.x;
            r.position[1] = p.y;
            r.position[2] = p.z;
            r.range = 9.0f * sceneScale;
            r.color[0] = light->mColorDiffuse.r;
            r.color[1] = light->mColorDiffuse.g;
            r.color[2] = light->mColorDiffuse.b;
            printf("  point light '%s' at (%.2f %.2f %.2f) colour (%.1f %.1f %.1f)\n", light->mName.C_Str(), p.x, p.y,
                   p.z, r.color[0], r.color[1], r.color[2]);
            pointLights_.push_back(r);
        }
    }

    void writeTextureList() {
        std::ofstream out(workDir_ + "/texture_list.txt");
        for (auto& t : textures_) out << t.first << " " << t.second << "\n";
    }

    // Writes top-down obstacle maps (PGM), one per 1 m height band from 0 to 16 m, used by
    // tools/check_path.py / relax_path.py to keep the fly-through path clear of walls and props.
    void writeHeightMap(Vec3 mn, Vec3 mx) {
        const float cell = 0.25f;
        constexpr int kBands = 16;
        const float yBase = mn.y < -0.5f ? std::floor(mn.y) : 0.0f;
        int w = static_cast<int>((mx.x - mn.x) / cell) + 1;
        int h = static_cast<int>((mx.z - mn.z) / cell) + 1;
        std::vector<uint8_t> grid(static_cast<size_t>(w) * h * kBands, 0);
        auto mark = [&](const Vec3& p) {
            int cx = static_cast<int>((p.x - mn.x) / cell), cz = static_cast<int>((p.z - mn.z) / cell);
            int band = static_cast<int>(std::floor(p.y - yBase));
            if (cx >= 0 && cz >= 0 && cx < w && cz < h && band >= 0 && band < kBands)
                grid[(static_cast<size_t>(band) * h + cz) * w + cx] = 255;
        };
        for (size_t m = 0; m < geometry_.size(); ++m) {
            const auto& geo = geometry_[m];
            for (size_t t = 0; t + 2 < geo.indices.size(); t += 3) {
                const Vec3& a = geo.vertices[geo.indices[t]].pos;
                const Vec3& b = geo.vertices[geo.indices[t + 1]].pos;
                const Vec3& c = geo.vertices[geo.indices[t + 2]].pos;
                if (std::max({a.y, b.y, c.y}) < yBase || std::min({a.y, b.y, c.y}) > yBase + kBands) continue;
                // Sample the triangle densely enough to hit every cell it covers.
                float len = std::max({length3(a - b), length3(b - c), length3(c - a)});
                int steps = std::min(600, static_cast<int>(len / (cell * 0.5f)) + 1);
                for (int i = 0; i <= steps; ++i) {
                    for (int j = 0; j <= steps - i; ++j) {
                        float u = static_cast<float>(i) / steps, v = static_cast<float>(j) / steps;
                        mark(a + (b - a) * u + (c - a) * v);
                    }
                }
            }
        }
        std::ofstream out(workDir_ + "/obstacles.pgm", std::ios::binary);
        out << "P5\n" << w << " " << h * kBands << "\n255\n";
        out.write(reinterpret_cast<const char*>(grid.data()), static_cast<std::streamsize>(grid.size()));
        printf("  obstacle map %dx%d x %d bands (yBase %.1f), origin (%.2f, %.2f) cell %.2f\n", w, h, kBands, yBase,
               mn.x, mn.z, cell);
    }

    static float length3(const Vec3& v) { return std::sqrt(dot(v, v)); }

    AlphaStats stats_;
    std::string workDir_;
    ConvertConfig cfg_;
    std::vector<std::unique_ptr<Assimp::Importer>> importers_;
    std::vector<MaterialInfo> materials_;
    std::vector<MaterialGeometry> geometry_;
    std::map<std::string, uint32_t> materialByName_;
    std::vector<std::pair<std::string, uint32_t>> textures_;
    std::vector<std::tuple<Vec3, Vec3, size_t>> primaryBounds_;
    float primaryScale_ = 1.0f;
    std::vector<bistro::CameraKey> cameraKeys_;
    std::vector<bistro::PointLightRecord> pointLights_;
    float sunDir_[3] = {0.3f, 0.8f, 0.2f};
    float sunColor_[3] = {1, 1, 1};
};

AlphaStats loadAlphaStats(const std::string& path) {
    AlphaStats stats;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        // Parse the 6 trailing numeric fields from the right so texture names with spaces work cleanly.
        size_t pos = line.size();
        for (int i = 0; i < 6; ++i) {
            while (pos > 0 && std::isspace(static_cast<unsigned char>(line[pos - 1]))) pos--;
            while (pos > 0 && !std::isspace(static_cast<unsigned char>(line[pos - 1]))) pos--;
        }
        size_t nameEnd = pos;
        while (nameEnd > 0 && std::isspace(static_cast<unsigned char>(line[nameEnd - 1]))) nameEnd--;
        if (nameEnd == 0) continue;
        std::string name = line.substr(0, nameEnd);
        std::istringstream iss(line.substr(pos));
        TextureStats t;
        if (iss >> t.cutout >> t.translucent >> t.mean[0] >> t.mean[1] >> t.mean[2] >> t.maxStdDev) {
            stats.cutoutFraction[name] = t.cutout;
            stats.translucentFraction[name] = t.translucent;
            stats.all[name] = t;
            stats.canonicalName[toLowerStr(name)] = name;
        }
    }
    printf("Loaded alpha stats for %zu textures\n", stats.cutoutFraction.size());
    return stats;
}

ConvertConfig loadConvertConfig(const std::string& workDir) {
    ConvertConfig cfg;
    std::ifstream in(workDir + "/convert_config.txt");
    std::string key;
    while (in >> key) {
        if (key == "ao_rays") in >> cfg.aoRays;
        else if (key == "ao_max_dist") in >> cfg.aoMaxDist;
        else if (key == "ao_min_vis") in >> cfg.aoMinVis;
        else if (key == "ao_min_area") in >> cfg.aoMinArea;
        else if (key == "weld_grid") in >> cfg.weldGrid;
        else if (key == "foliage_weld_grid") in >> cfg.foliageWeldGrid;
        else if (key == "fold_constant_basecolor") { int v = 0; in >> v; cfg.foldConstantBaseColor = v != 0; }
        else if (key == "add_curtain_glass") { int v = 1; in >> v; cfg.addCurtainGlass = v != 0; }
        else if (key == "deduplicate_within_scene") { int v = 0; in >> v; cfg.deduplicateWithinScene = v != 0; }
        else if (key == "skip_node") { std::string s; in >> s; cfg.skipNodeSubstrings.push_back(s); }
        else if (key == "translucent_material") { std::string s; in >> s; cfg.translucentInclude.push_back(s); }
        else if (key == "translucent_exclude") { std::string s; in >> s; cfg.translucentExclude.push_back(s); }
        else if (key == "sun_override") {
            cfg.hasSunOverride = true;
            in >> cfg.sunDir[0] >> cfg.sunDir[1] >> cfg.sunDir[2] >> cfg.sunColor[0] >> cfg.sunColor[1] >> cfg.sunColor[2];
            Vec3 d = normalize({cfg.sunDir[0], cfg.sunDir[1], cfg.sunDir[2]});
            cfg.sunDir[0] = d.x; cfg.sunDir[1] = d.y; cfg.sunDir[2] = d.z;
        } else if (key == "point_light") {
            bistro::PointLightRecord pl{};
            in >> pl.position[0] >> pl.position[1] >> pl.position[2] >> pl.range >> pl.color[0] >> pl.color[1] >> pl.color[2];
            cfg.customPointLights.push_back(pl);
        }
    }
    return cfg;
}

}  // namespace

#ifdef __APPLE__
#include <dlfcn.h>
#include <mach-o/dyld.h>

extern "C" int run_astc(const char* bundlePath, unsigned long entryOff, int argc, const char** argv) {
    static void* handle = nullptr;
    static int (*entry)(int, const char**) = nullptr;
    if (!entry) {
        handle = dlopen(bundlePath, RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            fprintf(stderr, "dlopen(%s) failed: %s\n", bundlePath, dlerror());
            return -1;
        }
        uint32_t count = _dyld_image_count();
        for (uint32_t i = 0; i < count; ++i) {
            const char* name = _dyld_get_image_name(i);
            if (name && std::strstr(name, "astcenc.bundle")) {
                uintptr_t base = reinterpret_cast<uintptr_t>(_dyld_get_image_header(i));
                entry = reinterpret_cast<int (*)(int, const char**)>(base + entryOff);
                break;
            }
        }
    }
    if (!entry) return -2;
    return entry(argc, argv);
}
#endif

extern "C" int bistro_convert_main(int argc, const char** argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <out_dir> <work_dir> <alpha_stats.txt> <primary.fbx> [secondary.fbx ...]\n",
                argv[0]);
        return 1;
    }
    ConvertConfig cfg = loadConvertConfig(argv[2]);
    Converter converter(loadAlphaStats(argv[3]), argv[2], cfg);
    for (int i = 4; i < argc; ++i) {
        if (!converter.addScene(argv[i], i == 4)) return 1;
    }
    converter.prepareGeometry();
    const char* rays = getenv("BISTRO_AO_RAYS");
    int rayCount = rays ? atoi(rays) : cfg.aoRays;
    if (rayCount > 0) converter.bakeSkyVisibility(rayCount, cfg.aoMaxDist);
    return converter.write(argv[1]) ? 0 : 1;
}

int main(int argc, char** argv) {
    return bistro_convert_main(argc, const_cast<const char**>(argv));
}
