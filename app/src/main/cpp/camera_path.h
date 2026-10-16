#pragma once

#include <vector>

#include "math_util.h"

// Two closed, smooth camera paths that alternate continuously:
//   0: Exterior fly-through (plaza, street, bistro terrace)
//   1: Restaurant interior fly-through (bar counter, wine shelves, espresso machine, dining tables)
// Each loop is a centripetal Catmull-Rom spline re-parameterised by arc length with curvature easing.
class CameraPath {
public:
    struct Pose {
        Vec3 eye;
        Vec3 target;
        int segment = 0;       // 0 = Exterior, 1 = Interior
        bool cut = false;      // true when switching between Exterior and Interior
    };

    struct Keyframe {
        Vec3 eye;
        Vec3 target;
    };

    CameraPath();
    [[nodiscard]] Pose evaluate(double timeSeconds) const;
    [[nodiscard]] float totalLength() const { return loops_[0].totalLength + loops_[1].totalLength; }
    [[nodiscard]] float exteriorDuration() const { return loops_[0].totalTime; }
    [[nodiscard]] float interiorDuration() const { return loops_[1].totalTime; }
    [[nodiscard]] float loopDuration() const { return loops_[0].totalTime + loops_[1].totalTime; }

private:
    struct Loop {
        std::vector<Vec3> points;
        std::vector<Vec3> targets;
        std::vector<float> sampleDistance;  // cumulative arc length per sample
        std::vector<Vec3> samplePosition;
        std::vector<Vec3> sampleTarget;
        std::vector<float> sampleTime;      // cumulative time (seconds) per sample, with curvature easing
        float totalLength = 0.0f;
        float totalTime = 0.0f;
        float speed = 2.4f;
        float smoothWindow = 1.5f;
    };

    static void buildLoop(Loop& loop, const Keyframe* keyframes, int count, float speed, float smoothWindow);
    [[nodiscard]] static Vec3 positionAtDistance(const Loop& loop, float s);
    [[nodiscard]] static Vec3 targetAtDistance(const Loop& loop, float s);
    [[nodiscard]] static float speedFactorAtDistance(const Loop& loop, float s);
    [[nodiscard]] static Pose evaluateLoop(const Loop& loop, double t, int segment);

    Loop loops_[2];
};
