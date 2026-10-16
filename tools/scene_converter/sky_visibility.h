#pragma once

// Offline per-vertex sky visibility ("ambient occlusion") baking.
//
// Builds a BVH over all occluding triangles of the scene and, for every vertex, casts cosine-weighted
// rays over the hemisphere around its normal. The fraction of rays that escape (no hit within
// maxDistance) is stored per vertex and used at runtime to occlude sky/ambient lighting. This gives
// streets, alleys, awnings and the bistro interior the large-scale occlusion a real-time renderer on
// a phone could not afford to compute dynamically.

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

namespace skyvis {

struct Float3 {
    float x, y, z;
};

inline Float3 sub(Float3 a, Float3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Float3 crossp(Float3 a, Float3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline float dotp(Float3 a, Float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

struct Triangle {
    Float3 v0, e1, e2;
};

struct Node {
    float mn[3];
    uint32_t leftOrFirst;  // inner: index of left child (right = left + 1); leaf: first triangle
    float mx[3];
    uint32_t count;  // 0 for inner nodes
};

class Bvh {
public:
    void build(std::vector<Triangle> tris) {
        tris_ = std::move(tris);
        const size_t n = tris_.size();
        centroids_.resize(n);
        bmin_.resize(n);
        bmax_.resize(n);
        order_.resize(n);
        for (size_t i = 0; i < n; ++i) {
            const Triangle& t = tris_[i];
            Float3 a = t.v0, b{t.v0.x + t.e1.x, t.v0.y + t.e1.y, t.v0.z + t.e1.z},
                   c{t.v0.x + t.e2.x, t.v0.y + t.e2.y, t.v0.z + t.e2.z};
            bmin_[i] = {std::min({a.x, b.x, c.x}), std::min({a.y, b.y, c.y}), std::min({a.z, b.z, c.z})};
            bmax_[i] = {std::max({a.x, b.x, c.x}), std::max({a.y, b.y, c.y}), std::max({a.z, b.z, c.z})};
            centroids_[i] = {(a.x + b.x + c.x) / 3, (a.y + b.y + c.y) / 3, (a.z + b.z + c.z) / 3};
            order_[i] = static_cast<uint32_t>(i);
        }
        nodes_.clear();
        nodes_.reserve(n * 2 / kLeafSize + 16);
        nodes_.push_back({});
        buildRecursive(0, 0, static_cast<uint32_t>(n));
        // Reorder triangles to leaf order for cache-friendly traversal.
        std::vector<Triangle> sorted(n);
        for (size_t i = 0; i < n; ++i) sorted[i] = tris_[order_[i]];
        tris_ = std::move(sorted);
        centroids_.clear();
        bmin_.clear();
        bmax_.clear();
        order_.clear();
    }

    // Any-hit occlusion query.
    bool occluded(Float3 o, Float3 d, float tMax, float tMin = 1e-4f) const {
        Float3 inv{1.0f / (std::fabs(d.x) > 1e-12f ? d.x : 1e-12f), 1.0f / (std::fabs(d.y) > 1e-12f ? d.y : 1e-12f),
                   1.0f / (std::fabs(d.z) > 1e-12f ? d.z : 1e-12f)};
        uint32_t stack[64];
        int sp = 0;
        stack[sp++] = 0;
        while (sp > 0) {
            const Node& node = nodes_[stack[--sp]];
            if (!hitBox(node, o, inv, tMax)) continue;
            if (node.count > 0) {
                for (uint32_t i = 0; i < node.count; ++i) {
                    if (hitTriangle(tris_[node.leftOrFirst + i], o, d, tMin, tMax)) return true;
                }
            } else if (sp < 62) {
                stack[sp++] = node.leftOrFirst;
                stack[sp++] = node.leftOrFirst + 1;
            }
        }
        return false;
    }

    size_t nodeCount() const { return nodes_.size(); }

private:
    static constexpr uint32_t kLeafSize = 4;

    void buildRecursive(uint32_t nodeIndex, uint32_t first, uint32_t count) {
        float mn[3] = {FLT_MAX, FLT_MAX, FLT_MAX}, mx[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
        float cmn[3] = {FLT_MAX, FLT_MAX, FLT_MAX}, cmx[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
        for (uint32_t i = first; i < first + count; ++i) {
            uint32_t t = order_[i];
            const float* bmn = &bmin_[t].x;
            const float* bmx = &bmax_[t].x;
            const float* c = &centroids_[t].x;
            for (int k = 0; k < 3; ++k) {
                mn[k] = std::min(mn[k], bmn[k]);
                mx[k] = std::max(mx[k], bmx[k]);
                cmn[k] = std::min(cmn[k], c[k]);
                cmx[k] = std::max(cmx[k], c[k]);
            }
        }
        Node& node = nodes_[nodeIndex];
        for (int k = 0; k < 3; ++k) {
            node.mn[k] = mn[k];
            node.mx[k] = mx[k];
        }
        int axis = 0;
        float ext[3] = {cmx[0] - cmn[0], cmx[1] - cmn[1], cmx[2] - cmn[2]};
        if (ext[1] > ext[axis]) axis = 1;
        if (ext[2] > ext[axis]) axis = 2;
        if (count <= kLeafSize || ext[axis] <= 1e-6f) {
            node.leftOrFirst = first;
            node.count = count;
            return;
        }
        uint32_t mid = first + count / 2;
        std::nth_element(order_.begin() + first, order_.begin() + mid, order_.begin() + first + count,
                         [&](uint32_t a, uint32_t b) { return (&centroids_[a].x)[axis] < (&centroids_[b].x)[axis]; });
        uint32_t left = static_cast<uint32_t>(nodes_.size());
        nodes_.push_back({});
        nodes_.push_back({});
        nodes_[nodeIndex].leftOrFirst = left;
        nodes_[nodeIndex].count = 0;
        buildRecursive(left, first, mid - first);
        buildRecursive(left + 1, mid, first + count - mid);
    }

    static bool hitBox(const Node& n, Float3 o, Float3 inv, float tMax) {
        float tx1 = (n.mn[0] - o.x) * inv.x, tx2 = (n.mx[0] - o.x) * inv.x;
        float tmin = std::min(tx1, tx2), tmax = std::max(tx1, tx2);
        float ty1 = (n.mn[1] - o.y) * inv.y, ty2 = (n.mx[1] - o.y) * inv.y;
        tmin = std::max(tmin, std::min(ty1, ty2));
        tmax = std::min(tmax, std::max(ty1, ty2));
        float tz1 = (n.mn[2] - o.z) * inv.z, tz2 = (n.mx[2] - o.z) * inv.z;
        tmin = std::max(tmin, std::min(tz1, tz2));
        tmax = std::min(tmax, std::max(tz1, tz2));
        return tmax >= std::max(tmin, 0.0f) && tmin < tMax;
    }

    static bool hitTriangle(const Triangle& t, Float3 o, Float3 d, float tMin, float tMax) {
        Float3 p = crossp(d, t.e2);
        float det = dotp(t.e1, p);
        if (std::fabs(det) < 1e-10f) return false;
        float invDet = 1.0f / det;
        Float3 s = sub(o, t.v0);
        float u = dotp(s, p) * invDet;
        if (u < 0.0f || u > 1.0f) return false;
        Float3 q = crossp(s, t.e1);
        float v = dotp(d, q) * invDet;
        if (v < 0.0f || u + v > 1.0f) return false;
        float dist = dotp(t.e2, q) * invDet;
        return dist > tMin && dist < tMax;
    }

    std::vector<Triangle> tris_;
    std::vector<Node> nodes_;
    std::vector<Float3> centroids_, bmin_, bmax_;
    std::vector<uint32_t> order_;
};

// Computes sky visibility for each (position, normal) pair using cosine-weighted hemisphere rays.
inline void computeVisibility(const Bvh& bvh, const std::vector<Float3>& positions,
                              const std::vector<Float3>& normals, std::vector<float>& out, int raysPerVertex,
                              float maxDistance) {
    out.assign(positions.size(), 1.0f);
    // Fixed cosine-weighted direction set (Hammersley), randomly rotated per vertex.
    std::vector<Float3> local(raysPerVertex);
    for (int i = 0; i < raysPerVertex; ++i) {
        float u1 = (i + 0.5f) / raysPerVertex;
        uint32_t b = static_cast<uint32_t>(i);
        b = (b << 16) | (b >> 16);
        b = ((b & 0x55555555u) << 1) | ((b & 0xAAAAAAAAu) >> 1);
        b = ((b & 0x33333333u) << 2) | ((b & 0xCCCCCCCCu) >> 2);
        b = ((b & 0x0F0F0F0Fu) << 4) | ((b & 0xF0F0F0F0u) >> 4);
        b = ((b & 0x00FF00FFu) << 8) | ((b & 0xFF00FF00u) >> 8);
        float u2 = static_cast<float>(b) * 2.3283064365386963e-10f;
        float r = std::sqrt(u1);
        float phi = 6.2831853f * u2;
        local[i] = {r * std::cos(phi), r * std::sin(phi), std::sqrt(std::max(0.0f, 1.0f - u1))};
    }
    std::atomic<size_t> next{0};
    const size_t total = positions.size();
    unsigned threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> pool;
    std::atomic<size_t> done{0};
    for (unsigned t = 0; t < threads; ++t) {
        pool.emplace_back([&]() {
            constexpr size_t kChunk = 4096;
            for (;;) {
                size_t begin = next.fetch_add(kChunk);
                if (begin >= total) break;
                size_t end = std::min(total, begin + kChunk);
                for (size_t i = begin; i < end; ++i) {
                    Float3 n = normals[i];
                    Float3 up = std::fabs(n.y) < 0.999f ? Float3{0, 1, 0} : Float3{1, 0, 0};
                    Float3 tx = crossp(up, n);
                    float l = std::sqrt(dotp(tx, tx));
                    tx = {tx.x / l, tx.y / l, tx.z / l};
                    Float3 ty = crossp(n, tx);
                    // Per-vertex rotation angle from a hash of the index.
                    uint32_t h = static_cast<uint32_t>(i) * 2654435761u;
                    float ang = (h >> 8) * (6.2831853f / 16777216.0f);
                    float ca = std::cos(ang), sa = std::sin(ang);
                    // Small offset plus a minimum hit distance: the Bistro has stacked decal layers
                    // (wet patches, leaves) a few centimetres above the pavement that must not occlude it.
                    Float3 o{positions[i].x + n.x * 0.03f, positions[i].y + n.y * 0.03f,
                             positions[i].z + n.z * 0.03f};
                    int open = 0;
                    for (int r = 0; r < raysPerVertex; ++r) {
                        float lx = local[r].x * ca - local[r].y * sa;
                        float ly = local[r].x * sa + local[r].y * ca;
                        float lz = local[r].z;
                        Float3 d{tx.x * lx + ty.x * ly + n.x * lz, tx.y * lx + ty.y * ly + n.y * lz,
                                 tx.z * lx + ty.z * ly + n.z * lz};
                        if (!bvh.occluded(o, d, maxDistance, 0.12f)) open++;
                    }
                    out[i] = static_cast<float>(open) / raysPerVertex;
                }
                done.fetch_add(end - begin);
            }
        });
    }
    for (auto& th : pool) th.join();
}

}  // namespace skyvis
