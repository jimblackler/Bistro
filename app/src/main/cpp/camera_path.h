#pragma once

#include <string>
#include <vector>

#include "math_util.h"

// Evaluates a sequence of closed, smooth camera loops loaded from an environment's manifest.json.
// Each loop is a centripetal Catmull-Rom spline re-parameterised by arc length with curvature easing.
class CameraPath {
public:
    struct Pose {
        Vec3 eye{0.0f, 2.0f, 0.0f};
        Vec3 target{0.0f, 2.0f, -1.0f};
        int segment = 0;
        bool indoor = false;
        bool cut = false;
    };

    struct Keyframe {
        Vec3 eye;
        Vec3 target;
    };

    CameraPath() = default;
    bool loadFromManifest(const std::string& manifestJson);

    [[nodiscard]] Pose evaluate(double timeSeconds) const;
    [[nodiscard]] float totalLength() const;
    [[nodiscard]] float loopDuration() const;
    [[nodiscard]] int loopCount() const { return static_cast<int>(loops_.size()); }
    [[nodiscard]] float segmentDuration(int segment) const;
    [[nodiscard]] float segmentStartTime(int segment) const;
    [[nodiscard]] const std::string& loopName(int segment) const;

private:
    struct Loop {
        std::string name;
        bool indoor = false;
        std::vector<Vec3> points;
        std::vector<Vec3> targets;
        std::vector<float> sampleDistance;  // cumulative arc length per sample
        std::vector<Vec3> samplePosition;
        std::vector<Vec3> sampleTarget;
        std::vector<float> sampleTime;      // cumulative time (seconds) per sample, with curvature easing
        float totalLength = 0.0f;
        float totalTime = 0.0f;
        float speed = 2.4f;
        float smoothWindow = 1.0f;
    };

    static void buildLoop(Loop& loop, const std::vector<Keyframe>& keyframes, float speed, float smoothWindow);
    [[nodiscard]] static Vec3 positionAtDistance(const Loop& loop, float s);
    [[nodiscard]] static Vec3 targetAtDistance(const Loop& loop, float s);
    [[nodiscard]] static float speedFactorAtDistance(const Loop& loop, float s);
    [[nodiscard]] static Pose evaluateLoop(const Loop& loop, double t, int segment);

    std::vector<Loop> loops_;
};
