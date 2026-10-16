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
};

// All geometry belonging to one material, in world space.
struct MaterialGeometry {
    std::vector<FullVertex> vertices;
    std::vector<uint32_t> indices;
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
};

std::string textureNameFromPath(const aiString& p) {
    std::string s = p.C_Str();
    std::replace(s.begin(), s.end(), '\\', '/');
    size_t slash = s.find_last_of('/');
    if (slash != std::string::npos) s = s.substr(slash + 1);
    size_t dot = s.find_last_of('.');
    if (dot != std::string::npos) s = s.substr(0, dot);
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

class Converter {
public:
    Converter(AlphaStats stats, std::string workDir) : stats_(std::move(stats)), workDir_(std::move(workDir)) {}

    bool addScene(const std::string& path, bool isPrimary) {
        auto importer = std::make_unique<Assimp::Importer>();
        importer->SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_POINT | aiPrimitiveType_LINE);
        const unsigned flags = aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace |
                               aiProcess_JoinIdenticalVertices | aiProcess_ImproveCacheLocality |
                               aiProcess_SortByPType | aiProcess_FlipUVs | aiProcess_FindInvalidData;
        const aiScene* scene = importer->ReadFile(path, flags);
        if (!scene) {
            fprintf(stderr, "Failed to load %s: %s\n", path.c_str(), importer->GetErrorString());
            return false;
        }
        printf("Loaded %s: %u meshes, %u materials\n", path.c_str(), scene->mNumMeshes, scene->mNumMaterials);

        std::vector<uint32_t> materialRemap(scene->mNumMaterials);
        for (unsigned i = 0; i < scene->mNumMaterials; ++i) materialRemap[i] = registerMaterial(scene->mMaterials[i]);

        size_t skipped = 0, added = 0;
        walk(scene, scene->mRootNode, aiMatrix4x4(), materialRemap, isPrimary, skipped, added);
        printf("  instances added: %zu, skipped as duplicates: %zu\n", added, skipped);

        if (isPrimary) extractLightsAndCamera(scene);
        extractPointLights(scene);
        importers_.push_back(std::move(importer));
        return true;
    }

    void bakeSkyVisibility(int rays, float maxDistance) {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<skyvis::Triangle> tris;
        for (size_t m = 0; m < materials_.size(); ++m) {
            if (classifyAlpha(materials_[m]) == bistro::kAlphaBlend) continue;  // glass lets skylight in
            const MaterialGeometry& geo = geometry_[m];
            for (size_t t = 0; t + 2 < geo.indices.size(); t += 3) {
                Vec3 a = geo.vertices[geo.indices[t]].pos, b = geo.vertices[geo.indices[t + 1]].pos,
                     c = geo.vertices[geo.indices[t + 2]].pos;
                tris.push_back({{a.x, a.y, a.z}, {b.x - a.x, b.y - a.y, b.z - a.z}, {c.x - a.x, c.y - a.y, c.z - a.z}});
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
                v.skyVisibility = vis[k++];
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
        size_t maskCount = 0, blendCount = 0, foldedOrm = 0;
        for (auto& m : materials_) {
            bistro::MaterialRecord r{};
            r.baseColorTexture = textureIndex(m.baseColor, bistro::kTextureBaseColor);
            r.normalTexture = textureIndex(m.normal, bistro::kTextureNormal);
            r.roughnessFactor = 1.0f;
            r.metallicFactor = 1.0f;
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
            r.emissiveTexture = textureIndex(m.emissive, bistro::kTextureEmissive);
            std::memcpy(r.baseColorFactor, m.baseColorFactor, sizeof(r.baseColorFactor));
            std::memcpy(r.emissiveFactor, m.emissiveFactor, sizeof(r.emissiveFactor));
            r.alphaCutoff = 0.5f;
            r.alphaMode = classifyAlpha(m);
            r.flags = 0;
            if (containsCI(m.name, "DoubleSided") || r.alphaMode != bistro::kAlphaOpaque) r.flags |= bistro::kMaterialDoubleSided;
            if (r.emissiveTexture >= 0) {
                r.flags |= bistro::kMaterialEmissive;
                r.emissiveFactor[0] = r.emissiveFactor[1] = r.emissiveFactor[2] = 1.0f;
            }
            if (r.alphaMode == bistro::kAlphaBlend) {
                // The Bistro glass was authored for path tracing (specular transmission) and its base colour
                // textures are fully opaque. Give it a raster-friendly opacity so windows are see-through.
                auto st = stats_.all.find(m.baseColor);
                bool textureHasAlpha =
                    st != stats_.all.end() && (st->second.translucent > 0.01f || st->second.cutout > 0.01f);
                if (!textureHasAlpha) {
                    float alpha = 0.16f;
                    if (containsCI(m.name, "Frosted") || containsCI(m.name, "Frozen")) alpha = 0.7f;
                    else if (containsCI(m.name, "Dirty")) alpha = 0.3f;
                    else if (containsCI(m.name, "Emissive")) alpha = 0.6f;
                    else if (containsCI(m.name, "Bottle") || containsCI(m.name, "Wine")) alpha = 0.45f;
                    r.baseColorFactor[3] = alpha;
                }
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
        printf("  folded %zu constant ORM textures into material factors\n", foldedOrm);
        printf("Wrote %s\n  vertices %zu, triangles %zu, clusters %zu, materials %zu (mask %zu, blend %zu), textures %zu\n",
               path.c_str(), vertices.size(), triangles, clusters.size(), materials_.size(), maskCount, blendCount,
               textures_.size());
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
        if (mat->GetTexture(aiTextureType_DIFFUSE, 0, &p) == AI_SUCCESS) info.baseColor = textureNameFromPath(p);
        if (mat->GetTexture(aiTextureType_NORMALS, 0, &p) == AI_SUCCESS) info.normal = textureNameFromPath(p);
        if (mat->GetTexture(aiTextureType_SPECULAR, 0, &p) == AI_SUCCESS) info.orm = textureNameFromPath(p);
        if (mat->GetTexture(aiTextureType_EMISSIVE, 0, &p) == AI_SUCCESS) info.emissive = textureNameFromPath(p);
        aiColor4D diffuse;
        if (info.baseColor.empty() && mat->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse) == AI_SUCCESS) {
            info.baseColorFactor[0] = diffuse.r;
            info.baseColorFactor[1] = diffuse.g;
            info.baseColorFactor[2] = diffuse.b;
        }
        uint32_t index = static_cast<uint32_t>(materials_.size());
        materials_.push_back(info);
        geometry_.emplace_back();
        materialByName_[key] = index;
        return index;
    }

    void walk(const aiScene* scene, const aiNode* node, const aiMatrix4x4& parent,
              const std::vector<uint32_t>& materialRemap, bool isPrimary, size_t& skipped, size_t& added) {
        aiMatrix4x4 world = parent * node->mTransformation;
        if (node->mNumMeshes > 0) {
            std::string nodeName = node->mName.C_Str();
            // Bounds of this node instance in world space, used to detect duplicated geometry between the
            // exterior and interior files (the interior file contains part of the bistro building shell).
            Vec3 mn{FLT_MAX, FLT_MAX, FLT_MAX}, mx{-FLT_MAX, -FLT_MAX, -FLT_MAX};
            for (unsigned i = 0; i < node->mNumMeshes; ++i) {
                const aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
                for (unsigned v = 0; v < mesh->mNumVertices; ++v) {
                    aiVector3D p = world * mesh->mVertices[v];
                    mn = {std::min(mn.x, p.x), std::min(mn.y, p.y), std::min(mn.z, p.z)};
                    mx = {std::max(mx.x, p.x), std::max(mx.y, p.y), std::max(mx.z, p.z)};
                }
            }
            bool duplicate = false;
            if (isPrimary) {
                primaryNodes_.insert({nodeName, {mn, mx}});
            } else {
                auto range = primaryNodes_.equal_range(nodeName);
                for (auto it = range.first; it != range.second; ++it) {
                    Vec3 dmn = it->second.first - mn, dmx = it->second.second - mx;
                    if (std::fabs(dmn.x) + std::fabs(dmn.y) + std::fabs(dmn.z) + std::fabs(dmx.x) +
                            std::fabs(dmx.y) + std::fabs(dmx.z) < 0.05f) {
                        duplicate = true;
                        break;
                    }
                }
            }
            if (duplicate) {
                skipped++;
            } else {
                added++;
                for (unsigned i = 0; i < node->mNumMeshes; ++i) {
                    const aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
                    addMesh(mesh, world, materialRemap[mesh->mMaterialIndex]);
                }
            }
        }
        for (unsigned c = 0; c < node->mNumChildren; ++c)
            walk(scene, node->mChildren[c], world, materialRemap, isPrimary, skipped, added);
    }

    void addMesh(const aiMesh* mesh, const aiMatrix4x4& world, uint32_t material) {
        if (!(mesh->mPrimitiveTypes & aiPrimitiveType_TRIANGLE) || !mesh->HasNormals()) return;
        MaterialGeometry& geo = geometry_[material];
        aiMatrix3x3 normalMat = aiMatrix3x3(world);
        normalMat.Inverse().Transpose();
        aiMatrix3x3 linear = aiMatrix3x3(world);
        bool flip = linear.Determinant() < 0;

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
            geo.vertices.push_back(fv);
        }
        for (unsigned f = 0; f < mesh->mNumFaces; ++f) {
            const aiFace& face = mesh->mFaces[f];
            if (face.mNumIndices != 3) continue;
            uint32_t a = face.mIndices[0], b = face.mIndices[1], c = face.mIndices[2];
            if (a == b || b == c || a == c) continue;
            if (flip) std::swap(b, c);
            geo.indices.push_back(base + a);
            geo.indices.push_back(base + b);
            geo.indices.push_back(base + c);
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

    void extractPointLights(const aiScene* scene) {
        for (unsigned i = 0; i < scene->mNumLights; ++i) {
            const aiLight* light = scene->mLights[i];
            if (light->mType != aiLightSource_POINT && light->mType != aiLightSource_SPOT) continue;
            const aiNode* node = scene->mRootNode->FindNode(light->mName);
            aiMatrix4x4 world;
            for (const aiNode* n = node; n; n = n->mParent) world = n->mTransformation * world;
            aiVector3D p = world * light->mPosition;
            bistro::PointLightRecord r{};
            r.position[0] = p.x;
            r.position[1] = p.y;
            r.position[2] = p.z;
            r.range = 9.0f;
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
        int w = static_cast<int>((mx.x - mn.x) / cell) + 1;
        int h = static_cast<int>((mx.z - mn.z) / cell) + 1;
        std::vector<uint8_t> grid(static_cast<size_t>(w) * h * kBands, 0);
        auto mark = [&](const Vec3& p) {
            int cx = static_cast<int>((p.x - mn.x) / cell), cz = static_cast<int>((p.z - mn.z) / cell);
            int band = static_cast<int>(std::floor(p.y));
            if (cx >= 0 && cz >= 0 && cx < w && cz < h && band >= 0 && band < kBands)
                grid[(static_cast<size_t>(band) * h + cz) * w + cx] = 255;
        };
        for (size_t m = 0; m < geometry_.size(); ++m) {
            const auto& geo = geometry_[m];
            for (size_t t = 0; t + 2 < geo.indices.size(); t += 3) {
                const Vec3& a = geo.vertices[geo.indices[t]].pos;
                const Vec3& b = geo.vertices[geo.indices[t + 1]].pos;
                const Vec3& c = geo.vertices[geo.indices[t + 2]].pos;
                if (std::max({a.y, b.y, c.y}) < 0.0f || std::min({a.y, b.y, c.y}) > kBands) continue;
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
        printf("  obstacle map %dx%d x %d bands, origin (%.2f, %.2f) cell %.2f\n", w, h, kBands, mn.x, mn.z, cell);
    }

    static float length3(const Vec3& v) { return std::sqrt(dot(v, v)); }

    AlphaStats stats_;
    std::string workDir_;
    std::vector<std::unique_ptr<Assimp::Importer>> importers_;
    std::vector<MaterialInfo> materials_;
    std::vector<MaterialGeometry> geometry_;
    std::map<std::string, uint32_t> materialByName_;
    std::vector<std::pair<std::string, uint32_t>> textures_;
    std::multimap<std::string, std::pair<Vec3, Vec3>> primaryNodes_;
    std::vector<bistro::CameraKey> cameraKeys_;
    std::vector<bistro::PointLightRecord> pointLights_;
    float sunDir_[3] = {0.3f, 0.8f, 0.2f};
    float sunColor_[3] = {1, 1, 1};
};

AlphaStats loadAlphaStats(const std::string& path) {
    AlphaStats stats;
    std::ifstream in(path);
    std::string name;
    TextureStats t;
    while (in >> name >> t.cutout >> t.translucent >> t.mean[0] >> t.mean[1] >> t.mean[2] >> t.maxStdDev) {
        stats.cutoutFraction[name] = t.cutout;
        stats.translucentFraction[name] = t.translucent;
        stats.all[name] = t;
    }
    printf("Loaded alpha stats for %zu textures\n", stats.cutoutFraction.size());
    return stats;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <out_dir> <work_dir> <alpha_stats.txt> <primary.fbx> [secondary.fbx ...]\n",
                argv[0]);
        return 1;
    }
    Converter converter(loadAlphaStats(argv[3]), argv[2]);
    for (int i = 4; i < argc; ++i) {
        if (!converter.addScene(argv[i], i == 4)) return 1;
    }
    const char* rays = getenv("BISTRO_AO_RAYS");
    int rayCount = rays ? atoi(rays) : 48;
    if (rayCount > 0) converter.bakeSkyVisibility(rayCount, 80.0f);
    return converter.write(argv[1]) ? 0 : 1;
}
