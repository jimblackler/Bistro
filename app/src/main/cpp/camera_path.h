#pragma once

#include <vector>

#include "math_util.h"

// A closed, smooth camera path: a centripetal Catmull-Rom spline through hand-placed control points,
// re-parameterised by arc length so the camera moves at a steady speed. The view direction looks
// ahead along the same spline, giving smooth turns through the streets of the Bistro.
class CameraPath {
public:
    struct Pose {
        Vec3 eye;
        Vec3 target;
    };

    CameraPath();
    [[nodiscard]] Pose evaluate(double timeSeconds) const;
    [[nodiscard]] float totalLength() const { return totalLength_; }
    [[nodiscard]] float speed() const { return speed_; }
    [[nodiscard]] float loopDuration() const { return totalTime_; }

private:
    [[nodiscard]] Vec3 splinePoint(int segment, float t) const;
    [[nodiscard]] Vec3 positionAtDistance(float s) const;
    [[nodiscard]] Vec3 targetAtDistance(float s) const;
    [[nodiscard]] float speedFactorAtDistance(float s) const;

    std::vector<Vec3> points_;
    std::vector<Vec3> targets_;
    std::vector<float> sampleDistance_;  // cumulative arc length per sample
    std::vector<Vec3> samplePosition_;
    std::vector<Vec3> sampleTarget_;
    std::vector<float> sampleTime_;      // cumulative time (seconds) per sample, with curvature easing
    float totalLength_ = 0.0f;
    float totalTime_ = 0.0f;
    float speed_ = 2.4f;  // metres per second on straights
};
