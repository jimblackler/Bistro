#include "camera_path.h"

#include <algorithm>
#include <cmath>

namespace {

// Exterior keyframes (world space, metres) for the first fly-through loop. Each keyframe specifies
// both the camera eye position and a look-at target so that through turnarounds the camera looks back
// down the street or across the bistro terrace rather than toward the unmodeled outer edges of the asset.
const CameraPath::Keyframe kExteriorKeyframes[] = {
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
constexpr int kNumExteriorPoints = sizeof(kExteriorKeyframes) / sizeof(kExteriorKeyframes[0]);

// Restaurant interior keyframes (world space, metres) for the second fly-through loop.
// Glides through the bistro dining room, past the window booths, into the back wine alcove,
// along the zinc bar counter (espresso machine, beer taps, liquor shelves, cash register),
// and back through the centre aisle of set tables.
const CameraPath::Keyframe kInteriorKeyframes[] = {
    // Near the front corner entrance, looking across the dining room toward the bar counter.
    {{0.4f, 2.0f, -1.8f}, {6.8f, 1.8f, -2.2f}},
    // Glide north along the west window tables, looking at the set tables and bar.
    {{1.5f, 2.0f, -2.9f}, {6.5f, 1.8f, -3.5f}},
    {{2.7f, 2.05f, -4.5f}, {5.8f, 1.9f, -6.8f}},
    // Peer into the north dining room / wine alcove, panning across the wine racks and paintings.
    {{3.6f, 2.05f, -5.3f}, {6.2f, 2.0f, -8.5f}},
    {{4.3f, 2.05f, -5.0f}, {6.8f, 1.9f, -6.2f}},
    // Turn south-east toward the curved end of the zinc bar counter and espresso machine.
    {{4.7f, 2.02f, -4.1f}, {7.4f, 1.85f, -3.0f}},
    {{4.8f, 2.0f, -3.2f}, {7.9f, 1.85f, -2.4f}},
    // Cruise parallel to the bar stools, looking closely at the beer taps, bottles, and cash register.
    {{5.6f, 1.95f, -2.3f}, {8.5f, 1.85f, -2.0f}},
    {{7.0f, 1.95f, -1.3f}, {9.8f, 1.85f, -1.4f}},
    // Reach the east end of the bar and turn back toward the main dining room and sunlit windows.
    {{8.2f, 2.0f, -0.5f}, {5.0f, 1.7f, 0.8f}},
    {{6.6f, 2.0f, -0.1f}, {2.8f, 1.6f, 0.5f}},
    // Float west down the main dining aisle between the bistro tables and ceiling fans.
    {{4.8f, 2.0f, -1.0f}, {0.8f, 1.65f, -0.5f}},
    {{2.8f, 2.0f, -1.7f}, {0.2f, 1.75f, -2.2f}},
    {{1.2f, 2.0f, -1.4f}, {5.5f, 1.8f, -2.0f}},
};
constexpr int kNumInteriorPoints = sizeof(kInteriorKeyframes) / sizeof(kInteriorKeyframes[0]);

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
    buildLoop(loops_[0], kExteriorKeyframes, kNumExteriorPoints, 2.4f, 1.5f);
    buildLoop(loops_[1], kInteriorKeyframes, kNumInteriorPoints, 0.75f, 0.45f);
}

void CameraPath::buildLoop(Loop& loop, const Keyframe* keyframes, int count, float speed, float smoothWindow) {
    loop.speed = speed;
    loop.smoothWindow = smoothWindow;
    loop.points.resize(count);
    loop.targets.resize(count);
    for (int i = 0; i < count; ++i) {
        loop.points[i] = keyframes[i].eye;
        loop.targets[i] = keyframes[i].target;
    }
    const int n = count;
    loop.samplePosition.reserve(n * kSamplesPerSegment + 1);
    loop.sampleTarget.reserve(n * kSamplesPerSegment + 1);
    auto pointSpline = [&](int segment, float t) {
        auto p = [&](int i) { return loop.points[((i % n) + n) % n]; };
        return catmullRom(p(segment - 1), p(segment), p(segment + 1), p(segment + 2), t);
    };
    auto targetSpline = [&](int segment, float t) {
        auto p = [&](int i) { return loop.targets[((i % n) + n) % n]; };
        return catmullRom(p(segment - 1), p(segment), p(segment + 1), p(segment + 2), t);
    };
    for (int seg = 0; seg < n; ++seg) {
        for (int i = 0; i < kSamplesPerSegment; ++i) {
            float u = static_cast<float>(i) / kSamplesPerSegment;
            loop.samplePosition.push_back(pointSpline(seg, u));
            loop.sampleTarget.push_back(targetSpline(seg, u));
        }
    }
    loop.samplePosition.push_back(loop.samplePosition.front());
    loop.sampleTarget.push_back(loop.sampleTarget.front());

    loop.sampleDistance.resize(loop.samplePosition.size());
    loop.sampleDistance[0] = 0.0f;
    for (size_t i = 1; i < loop.samplePosition.size(); ++i) {
        loop.sampleDistance[i] = loop.sampleDistance[i - 1] + length(loop.samplePosition[i] - loop.samplePosition[i - 1]);
    }
    loop.totalLength = loop.sampleDistance.back();

    // Time parameterisation: ease off the speed through tighter turns.
    loop.sampleTime.resize(loop.samplePosition.size());
    loop.sampleTime[0] = 0.0f;
    for (size_t i = 1; i < loop.samplePosition.size(); ++i) {
        float s = 0.5f * (loop.sampleDistance[i] + loop.sampleDistance[i - 1]);
        float ds = loop.sampleDistance[i] - loop.sampleDistance[i - 1];
        loop.sampleTime[i] = loop.sampleTime[i - 1] + ds / (loop.speed * speedFactorAtDistance(loop, s));
    }
    loop.totalTime = loop.sampleTime.back();
}

Vec3 CameraPath::positionAtDistance(const Loop& loop, float s) {
    s = std::fmod(s, loop.totalLength);
    if (s < 0) s += loop.totalLength;
    auto it = std::upper_bound(loop.sampleDistance.begin(), loop.sampleDistance.end(), s);
    size_t i = std::clamp<size_t>(static_cast<size_t>(it - loop.sampleDistance.begin()), 1, loop.sampleDistance.size() - 1);
    float d0 = loop.sampleDistance[i - 1], d1 = loop.sampleDistance[i];
    float f = d1 > d0 ? (s - d0) / (d1 - d0) : 0.0f;
    return lerp(loop.samplePosition[i - 1], loop.samplePosition[i], f);
}

Vec3 CameraPath::targetAtDistance(const Loop& loop, float s) {
    s = std::fmod(s, loop.totalLength);
    if (s < 0) s += loop.totalLength;
    auto it = std::upper_bound(loop.sampleDistance.begin(), loop.sampleDistance.end(), s);
    size_t i = std::clamp<size_t>(static_cast<size_t>(it - loop.sampleDistance.begin()), 1, loop.sampleDistance.size() - 1);
    float d0 = loop.sampleDistance[i - 1], d1 = loop.sampleDistance[i];
    float f = d1 > d0 ? (s - d0) / (d1 - d0) : 0.0f;
    return lerp(loop.sampleTarget[i - 1], loop.sampleTarget[i], f);
}

float CameraPath::speedFactorAtDistance(const Loop& loop, float s) {
    float w = loop.smoothWindow * 2.0f;
    Vec3 a = positionAtDistance(loop, s - w);
    Vec3 b = positionAtDistance(loop, s);
    Vec3 c = positionAtDistance(loop, s + w);
    Vec3 d0 = normalize(Vec3{b.x - a.x, 0, b.z - a.z});
    Vec3 d1 = normalize(Vec3{c.x - b.x, 0, c.z - b.z});
    float turn = std::acos(std::clamp(dot(d0, d1), -1.0f, 1.0f));
    return std::clamp(1.0f / (1.0f + turn * 2.5f), 0.45f, 1.0f);
}

CameraPath::Pose CameraPath::evaluateLoop(const Loop& loop, double t, int segment) {
    t = std::fmod(t, static_cast<double>(loop.totalTime));
    if (t < 0) t += loop.totalTime;
    auto it = std::upper_bound(loop.sampleTime.begin(), loop.sampleTime.end(), static_cast<float>(t));
    size_t i = std::clamp<size_t>(static_cast<size_t>(it - loop.sampleTime.begin()), 1, loop.sampleTime.size() - 1);
    float t0 = loop.sampleTime[i - 1], t1 = loop.sampleTime[i];
    float f = t1 > t0 ? (static_cast<float>(t) - t0) / (t1 - t0) : 0.0f;
    float s = loop.sampleDistance[i - 1] + (loop.sampleDistance[i] - loop.sampleDistance[i - 1]) * f;

    Pose pose;
    pose.eye = positionAtDistance(loop, s);
    pose.segment = segment;
    Vec3 target{0, 0, 0};
    float wsum = 0.0f;
    for (int k = -2; k <= 2; ++k) {
        float w = 1.0f / (1.0f + 0.4f * std::abs(k));
        target += targetAtDistance(loop, s + loop.smoothWindow * static_cast<float>(k)) * w;
        wsum += w;
    }
    pose.target = target * (1.0f / wsum);
    return pose;
}

CameraPath::Pose CameraPath::evaluate(double timeSeconds) const {
    double total = static_cast<double>(loops_[0].totalTime + loops_[1].totalTime);
    double t = std::fmod(timeSeconds, total);
    if (t < 0) t += total;
    if (t < static_cast<double>(loops_[0].totalTime)) {
        return evaluateLoop(loops_[0], t, 0);
    }
    return evaluateLoop(loops_[1], t - static_cast<double>(loops_[0].totalTime), 1);
}
