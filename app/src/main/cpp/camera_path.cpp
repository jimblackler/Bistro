#include "camera_path.h"

#include <algorithm>
#include <cmath>

namespace {

// Keyframes (world space, metres) for the closed camera loop. Each keyframe specifies both the
// camera eye position and a look-at target so that through turnarounds the camera looks back down
// the street or across the bistro terrace rather than toward the unmodeled outer edges of the asset.
struct Keyframe {
    Vec3 eye;
    Vec3 target;
};

const Keyframe kKeyframes[] = {
    // Plaza: sweep from the north side of the plaza down toward the archway and street.
    {{61.0f, 5.5f, 30.0f}, {38.0f, 4.0f, 27.5f}},
    {{50.0f, 4.3f, 31.0f}, {26.0f, 3.4f, 23.0f}},
    {{40.0f, 3.8f, 28.5f}, {16.0f, 3.0f, 19.0f}},
    {{28.0f, 3.4f, 23.5f}, {6.0f, 2.8f, 14.5f}},
    {{16.0f, 3.1f, 19.5f}, {-5.0f, 2.8f, 11.5f}},
    {{6.5f, 3.0f, 14.5f}, {-10.0f, 3.0f, 6.0f}},
    {{-5.0f, 3.0f, 11.5f}, {-14.0f, 3.2f, -4.0f}},
    {{-11.0f, 3.1f, 5.5f}, {-12.0f, 3.4f, -10.0f}},
    // Low pass up the long north street, looking up the street between the facades.
    {{-14.0f, 3.3f, -6.0f}, {-4.0f, 3.6f, -18.0f}},
    {{-5.0f, 3.5f, -16.0f}, {6.0f, 3.8f, -29.0f}},
    {{4.0f, 3.6f, -27.0f}, {14.0f, 4.0f, -40.0f}},
    {{12.5f, 3.7f, -37.5f}, {21.0f, 4.2f, -51.0f}},
    // As we reach the north end, pan smoothly across the east facade and back down the street
    // (avoiding looking past the west facade where the asset ends at z = -55).
    {{18.5f, 3.9f, -46.5f}, {25.0f, 4.5f, -55.0f}},
    {{21.0f, 4.2f, -51.5f}, {23.0f, 4.6f, -42.0f}},
    {{19.5f, 4.5f, -48.5f}, {10.0f, 4.2f, -35.0f}},
    // Return pass down the long street at first-floor height, looking south toward the bistro corner.
    {{12.5f, 4.8f, -39.0f}, {1.0f, 4.0f, -24.0f}},
    {{4.5f, 5.0f, -28.0f}, {-8.0f, 3.8f, -13.0f}},
    {{-3.5f, 4.8f, -18.0f}, {-13.0f, 3.6f, -4.0f}},
    {{-14.0f, 4.4f, -7.0f}, {-11.0f, 3.5f, 6.0f}},
    // Round the bistro corner and glide east along the terrace toward the archway and plaza.
    {{-12.0f, 3.8f, 1.5f}, {-2.0f, 3.5f, 13.0f}},
    {{-7.5f, 4.0f, 10.5f}, {10.0f, 4.0f, 18.0f}},
    {{5.5f, 4.6f, 16.5f}, {24.0f, 4.5f, 23.0f}},
    {{16.5f, 5.2f, 19.5f}, {36.0f, 5.0f, 26.5f}},
    {{28.0f, 5.8f, 24.0f}, {48.0f, 5.5f, 29.0f}},
    {{40.0f, 6.4f, 27.5f}, {62.0f, 6.0f, 32.0f}},
    {{52.0f, 6.8f, 28.0f}, {72.0f, 6.2f, 36.0f}},
    // Circle around the plaza tree while keeping the camera aimed inward at the tree and plaza facades.
    {{66.0f, 7.2f, 22.0f}, {72.0f, 5.5f, 38.0f}},
    {{79.0f, 7.4f, 24.0f}, {68.0f, 5.2f, 38.0f}},
    {{83.5f, 7.0f, 31.0f}, {62.0f, 4.8f, 34.0f}},
    {{74.5f, 6.2f, 28.0f}, {48.0f, 4.2f, 29.5f}},
};
constexpr int kNumPoints = sizeof(kKeyframes) / sizeof(kKeyframes[0]);
constexpr int kSamplesPerSegment = 64;

// Centripetal Catmull-Rom (alpha = 0.5) between p1 and p2.
Vec3 catmullRom(const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3, float t) {
    auto knot = [](float ti, const Vec3& a, const Vec3& b) {
        return ti + std::sqrt(std::max(length(b - a), 1e-4f));
    };
    float t0 = 0.0f;
    float t1 = knot(t0, p0, p1);
    float t2 = knot(t1, p1, p2);
    float t3 = knot(t2, p2, p3);
    float u = t1 + (t2 - t1) * t;
    Vec3 a1 = p0 * ((t1 - u) / (t1 - t0)) + p1 * ((u - t0) / (t1 - t0));
    Vec3 a2 = p1 * ((t2 - u) / (t2 - t1)) + p2 * ((u - t1) / (t2 - t1));
    Vec3 a3 = p2 * ((t3 - u) / (t3 - t2)) + p3 * ((u - t2) / (t3 - t2));
    Vec3 b1 = a1 * ((t2 - u) / (t2 - t0)) + a2 * ((u - t0) / (t2 - t0));
    Vec3 b2 = a2 * ((t3 - u) / (t3 - t1)) + a3 * ((u - t1) / (t3 - t1));
    return b1 * ((t2 - u) / (t2 - t1)) + b2 * ((u - t1) / (t2 - t1));
}

}  // namespace

CameraPath::CameraPath() {
    points_.resize(kNumPoints);
    targets_.resize(kNumPoints);
    for (int i = 0; i < kNumPoints; ++i) {
        points_[i] = kKeyframes[i].eye;
        targets_[i] = kKeyframes[i].target;
    }
    const int n = static_cast<int>(points_.size());
    samplePosition_.reserve(n * kSamplesPerSegment + 1);
    sampleTarget_.reserve(n * kSamplesPerSegment + 1);
    auto targetSpline = [&](int segment, float t) {
        auto p = [&](int i) { return targets_[((i % n) + n) % n]; };
        return catmullRom(p(segment - 1), p(segment), p(segment + 1), p(segment + 2), t);
    };
    for (int seg = 0; seg < n; ++seg) {
        for (int i = 0; i < kSamplesPerSegment; ++i) {
            float u = static_cast<float>(i) / kSamplesPerSegment;
            samplePosition_.push_back(splinePoint(seg, u));
            sampleTarget_.push_back(targetSpline(seg, u));
        }
    }
    samplePosition_.push_back(samplePosition_.front());
    sampleTarget_.push_back(sampleTarget_.front());

    sampleDistance_.resize(samplePosition_.size());
    sampleDistance_[0] = 0.0f;
    for (size_t i = 1; i < samplePosition_.size(); ++i) {
        sampleDistance_[i] = sampleDistance_[i - 1] + length(samplePosition_[i] - samplePosition_[i - 1]);
    }
    totalLength_ = sampleDistance_.back();

    // Time parameterisation: ease off the speed through tighter turns.
    sampleTime_.resize(samplePosition_.size());
    sampleTime_[0] = 0.0f;
    for (size_t i = 1; i < samplePosition_.size(); ++i) {
        float s = 0.5f * (sampleDistance_[i] + sampleDistance_[i - 1]);
        float ds = sampleDistance_[i] - sampleDistance_[i - 1];
        sampleTime_[i] = sampleTime_[i - 1] + ds / (speed_ * speedFactorAtDistance(s));
    }
    totalTime_ = sampleTime_.back();
}

Vec3 CameraPath::splinePoint(int segment, float t) const {
    const int n = static_cast<int>(points_.size());
    auto p = [&](int i) { return points_[((i % n) + n) % n]; };
    return catmullRom(p(segment - 1), p(segment), p(segment + 1), p(segment + 2), t);
}

Vec3 CameraPath::positionAtDistance(float s) const {
    s = std::fmod(s, totalLength_);
    if (s < 0) s += totalLength_;
    auto it = std::upper_bound(sampleDistance_.begin(), sampleDistance_.end(), s);
    size_t i = std::clamp<size_t>(static_cast<size_t>(it - sampleDistance_.begin()), 1, sampleDistance_.size() - 1);
    float d0 = sampleDistance_[i - 1], d1 = sampleDistance_[i];
    float f = d1 > d0 ? (s - d0) / (d1 - d0) : 0.0f;
    return lerp(samplePosition_[i - 1], samplePosition_[i], f);
}

Vec3 CameraPath::targetAtDistance(float s) const {
    s = std::fmod(s, totalLength_);
    if (s < 0) s += totalLength_;
    auto it = std::upper_bound(sampleDistance_.begin(), sampleDistance_.end(), s);
    size_t i = std::clamp<size_t>(static_cast<size_t>(it - sampleDistance_.begin()), 1, sampleDistance_.size() - 1);
    float d0 = sampleDistance_[i - 1], d1 = sampleDistance_[i];
    float f = d1 > d0 ? (s - d0) / (d1 - d0) : 0.0f;
    return lerp(sampleTarget_[i - 1], sampleTarget_[i], f);
}

float CameraPath::speedFactorAtDistance(float s) const {
    // Heading change over a 6 m window approximates curvature.
    Vec3 a = positionAtDistance(s - 3.0f);
    Vec3 b = positionAtDistance(s);
    Vec3 c = positionAtDistance(s + 3.0f);
    Vec3 d0 = normalize(Vec3{b.x - a.x, 0, b.z - a.z});
    Vec3 d1 = normalize(Vec3{c.x - b.x, 0, c.z - b.z});
    float turn = std::acos(std::clamp(dot(d0, d1), -1.0f, 1.0f));
    return std::clamp(1.0f / (1.0f + turn * 2.5f), 0.45f, 1.0f);
}

CameraPath::Pose CameraPath::evaluate(double timeSeconds) const {
    double t = std::fmod(timeSeconds, static_cast<double>(totalTime_));
    if (t < 0) t += totalTime_;
    auto it = std::upper_bound(sampleTime_.begin(), sampleTime_.end(), static_cast<float>(t));
    size_t i = std::clamp<size_t>(static_cast<size_t>(it - sampleTime_.begin()), 1, sampleTime_.size() - 1);
    float t0 = sampleTime_[i - 1], t1 = sampleTime_[i];
    float f = t1 > t0 ? (static_cast<float>(t) - t0) / (t1 - t0) : 0.0f;
    float s = sampleDistance_[i - 1] + (sampleDistance_[i] - sampleDistance_[i - 1]) * f;

    Pose pose;
    pose.eye = positionAtDistance(s);
    // Smooth the target over a 6 m window along the path for silky camera panning.
    Vec3 target{0, 0, 0};
    float wsum = 0.0f;
    for (int k = -2; k <= 2; ++k) {
        float w = 1.0f / (1.0f + 0.4f * std::abs(k));
        target += targetAtDistance(s + 1.5f * static_cast<float>(k)) * w;
        wsum += w;
    }
    pose.target = target * (1.0f / wsum);
    return pose;
}
