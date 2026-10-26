#include "camera_path.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>

namespace {

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

size_t findMatchingBracket(const std::string& s, size_t openPos, char openChar, char closeChar) {
    int depth = 0;
    bool inStr = false;
    for (size_t i = openPos; i < s.size(); ++i) {
        char c = s[i];
        if (c == '"' && (i == 0 || s[i - 1] != '\\')) inStr = !inStr;
        if (inStr) continue;
        if (c == openChar) depth++;
        else if (c == closeChar) {
            depth--;
            if (depth == 0) return i;
        }
    }
    return std::string::npos;
}

std::string extractJsonString(const std::string& obj, const char* key, const char* fallback = "") {
    std::string needle = std::string("\"") + key + "\"";
    size_t pos = obj.find(needle);
    if (pos == std::string::npos) return fallback;
    pos = obj.find(':', pos + needle.size());
    if (pos == std::string::npos) return fallback;
    pos = obj.find('"', pos + 1);
    if (pos == std::string::npos) return fallback;
    size_t end = obj.find('"', pos + 1);
    if (end == std::string::npos) return fallback;
    return obj.substr(pos + 1, end - pos - 1);
}

float extractJsonFloat(const std::string& obj, const char* key, float fallback) {
    std::string needle = std::string("\"") + key + "\"";
    size_t pos = obj.find(needle);
    if (pos == std::string::npos) return fallback;
    pos = obj.find(':', pos + needle.size());
    if (pos == std::string::npos) return fallback;
    const char* start = obj.c_str() + pos + 1;
    char* endPtr = nullptr;
    float v = std::strtof(start, &endPtr);
    return endPtr != start ? v : fallback;
}

bool extractJsonBool(const std::string& obj, const char* key, bool fallback) {
    std::string needle = std::string("\"") + key + "\"";
    size_t pos = obj.find(needle);
    if (pos == std::string::npos) return fallback;
    pos = obj.find(':', pos + needle.size());
    if (pos == std::string::npos) return fallback;
    while (pos + 1 < obj.size() && std::isspace(static_cast<unsigned char>(obj[pos + 1]))) pos++;
    if (obj.compare(pos + 1, 4, "true") == 0) return true;
    if (obj.compare(pos + 1, 5, "false") == 0) return false;
    return fallback;
}

std::vector<float> extractAllFloats(const std::string& s) {
    std::vector<float> out;
    const char* p = s.c_str();
    const char* end = p + s.size();
    while (p < end) {
        if (*p == '-' || *p == '+' || (*p >= '0' && *p <= '9')) {
            char* next = nullptr;
            float v = std::strtof(p, &next);
            if (next != p) {
                out.push_back(v);
                p = next;
                continue;
            }
        }
        ++p;
    }
    return out;
}

}  // namespace

bool CameraPath::loadFromManifest(const std::string& manifestJson) {
    loops_.clear();
    size_t loopsKey = manifestJson.find("\"cameraLoops\"");
    if (loopsKey == std::string::npos) return false;
    size_t arrStart = manifestJson.find('[', loopsKey);
    if (arrStart == std::string::npos) return false;
    size_t arrEnd = findMatchingBracket(manifestJson, arrStart, '[', ']');
    if (arrEnd == std::string::npos) return false;

    size_t cur = arrStart + 1;
    while (cur < arrEnd) {
        size_t objStart = manifestJson.find('{', cur);
        if (objStart == std::string::npos || objStart >= arrEnd) break;
        size_t objEnd = findMatchingBracket(manifestJson, objStart, '{', '}');
        if (objEnd == std::string::npos || objEnd > arrEnd) break;
        std::string obj = manifestJson.substr(objStart, objEnd - objStart + 1);

        std::string name = extractJsonString(obj, "name", "Fly-through");
        float speed = extractJsonFloat(obj, "speed", 2.4f);
        float smoothWindow = extractJsonFloat(obj, "smoothWindow", 1.0f);
        bool indoor = extractJsonBool(obj, "indoor", false);

        size_t kfKey = obj.find("\"keyframes\"");
        if (kfKey != std::string::npos) {
            size_t kfStart = obj.find('[', kfKey);
            size_t kfEnd = kfStart != std::string::npos ? findMatchingBracket(obj, kfStart, '[', ']') : std::string::npos;
            if (kfStart != std::string::npos && kfEnd != std::string::npos) {
                std::vector<float> nums = extractAllFloats(obj.substr(kfStart, kfEnd - kfStart + 1));
                std::vector<Keyframe> kfs;
                for (size_t i = 0; i + 5 < nums.size(); i += 6) {
                    kfs.push_back({Vec3{nums[i], nums[i + 1], nums[i + 2]},
                                   Vec3{nums[i + 3], nums[i + 4], nums[i + 5]}});
                }
                if (kfs.size() >= 2) {
                    Loop loop;
                    loop.name = name;
                    loop.indoor = indoor;
                    buildLoop(loop, kfs, speed, smoothWindow);
                    loops_.push_back(std::move(loop));
                }
            }
        }
        cur = objEnd + 1;
    }
    return !loops_.empty();
}

void CameraPath::buildLoop(Loop& loop, const std::vector<Keyframe>& keyframes, float speed, float smoothWindow) {
    const int count = static_cast<int>(keyframes.size());
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
    pose.indoor = loop.indoor;
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

float CameraPath::totalLength() const {
    float sum = 0.0f;
    for (const auto& l : loops_) sum += l.totalLength;
    return sum;
}

float CameraPath::loopDuration() const {
    float sum = 0.0f;
    for (const auto& l : loops_) sum += l.totalTime;
    return sum;
}

float CameraPath::segmentDuration(int segment) const {
    if (segment < 0 || segment >= static_cast<int>(loops_.size())) return 0.0f;
    return loops_[segment].totalTime;
}

float CameraPath::segmentStartTime(int segment) const {
    float sum = 0.0f;
    for (int i = 0; i < segment && i < static_cast<int>(loops_.size()); ++i) {
        sum += loops_[i].totalTime;
    }
    return sum;
}

const std::string& CameraPath::loopName(int segment) const {
    static const std::string kEmpty;
    if (segment < 0 || segment >= static_cast<int>(loops_.size())) return kEmpty;
    return loops_[segment].name;
}

bool CameraPath::loopLightBounds(int segment, const Mat4& lightView, float lookAheadDist, float outMin[2],
                                 float outMax[2]) const {
    if (segment < 0 || segment >= static_cast<int>(loops_.size())) return false;
    const Loop& loop = loops_[segment];
    if (loop.samplePosition.empty()) return false;

    outMin[0] = outMin[1] = 1e30f;
    outMax[0] = outMax[1] = -1e30f;
    auto accum = [&](const Vec3& p) {
        Vec4 v = lightView * Vec4{p.x, p.y, p.z, 1.0f};
        outMin[0] = std::min(outMin[0], v.x);
        outMax[0] = std::max(outMax[0], v.x);
        outMin[1] = std::min(outMin[1], v.y);
        outMax[1] = std::max(outMax[1], v.y);
    };

    const size_t n = std::min(loop.samplePosition.size(), loop.sampleTarget.size());
    for (size_t i = 0; i < n; ++i) {
        const Vec3& eye = loop.samplePosition[i];
        const Vec3& tgt = loop.sampleTarget[i];
        Vec3 fwd = normalize(tgt - eye);
        accum(eye);
        accum(tgt);
        accum(eye + fwd * lookAheadDist);
    }
    return outMin[0] <= outMax[0] && outMin[1] <= outMax[1];
}

CameraPath::Pose CameraPath::evaluate(double timeSeconds) const {
    if (loops_.empty()) return Pose{};
    double total = static_cast<double>(loopDuration());
    if (total <= 0.0) return Pose{};
    double t = std::fmod(timeSeconds, total);
    if (t < 0) t += total;
    double accum = 0.0;
    for (size_t i = 0; i < loops_.size(); ++i) {
        double dur = static_cast<double>(loops_[i].totalTime);
        if (t < accum + dur || i + 1 == loops_.size()) {
            return evaluateLoop(loops_[i], t - accum, static_cast<int>(i));
        }
        accum += dur;
    }
    return evaluateLoop(loops_.back(), 0.0, static_cast<int>(loops_.size() - 1));
}
