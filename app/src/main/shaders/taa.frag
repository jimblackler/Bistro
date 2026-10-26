#version 460
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

// Temporal anti-aliasing resolve. The scene is rendered with a sub-pixel jittered projection; each pixel
// is reprojected into the previous frame using the depth buffer (static scene, so camera motion is the
// only motion), the history is clamped to the current neighbourhood in YCoCg space and blended.
layout(set = 1, binding = 0) uniform sampler2D currentColor;
layout(set = 1, binding = 1) uniform sampler2D historyColor;
layout(set = 1, binding = 2) uniform sampler2D sceneDepth;

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

vec3 toYCoCg(vec3 c) {
    return vec3(dot(c, vec3(0.25, 0.5, 0.25)), dot(c, vec3(0.5, 0.0, -0.5)), dot(c, vec3(-0.25, 0.5, -0.25)));
}
vec3 fromYCoCg(vec3 c) { return vec3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z); }

// Compress HDR before filtering so bright highlights don't dominate (reversible).
vec3 compress(vec3 c) { return c / (1.0 + luminance(c)); }
vec3 decompress(vec3 c) { return c / max(1.0 - luminance(c), 1e-4); }

// 5-tap Catmull-Rom history fetch (sharper than bilinear, avoids progressive blurring).
vec3 sampleHistory(vec2 uv, vec2 texSize) {
    vec2 samplePos = uv * texSize;
    vec2 tc1 = floor(samplePos - 0.5) + 0.5;
    vec2 f = samplePos - tc1;
    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    vec2 w3 = f * f * (-0.5 + 0.5 * f);
    vec2 w12 = w1 + w2;
    vec2 tc0 = (tc1 - 1.0) / texSize;
    vec2 tc3 = (tc1 + 2.0) / texSize;
    vec2 tc12 = (tc1 + w2 / w12) / texSize;
    vec3 r = textureLod(historyColor, vec2(tc12.x, tc0.y), 0.0).rgb * (w12.x * w0.y) +
             textureLod(historyColor, vec2(tc0.x, tc12.y), 0.0).rgb * (w0.x * w12.y) +
             textureLod(historyColor, vec2(tc12.x, tc12.y), 0.0).rgb * (w12.x * w12.y) +
             textureLod(historyColor, vec2(tc3.x, tc12.y), 0.0).rgb * (w3.x * w12.y) +
             textureLod(historyColor, vec2(tc12.x, tc3.y), 0.0).rgb * (w12.x * w3.y);
    float wsum = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    return max(r / wsum, vec3(0.0));
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 size = textureSize(currentColor, 0);
    vec2 texSize = vec2(size);

    // Neighbourhood statistics and nearest depth (dilates edges so silhouettes reproject correctly).
    vec3 cur = vec3(0.0);
    vec3 curTent = vec3(0.0);
    vec3 m1 = vec3(0.0), m2 = vec3(0.0);
    vec3 mn = vec3(1e9), mx = vec3(-1e9);
    float nearestDepth = 0.0;
    float farthestDepth = 1.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            ivec2 q = clamp(p + ivec2(x, y), ivec2(0), size - 1);
            vec3 c = toYCoCg(compress(texelFetch(currentColor, q, 0).rgb));
            if (x == 0 && y == 0) cur = c;
            float tw = (x == 0 && y == 0) ? 0.25 : ((x == 0 || y == 0) ? 0.125 : 0.0625);
            curTent += c * tw;
            m1 += c;
            m2 += c * c;
            mn = min(mn, c);
            mx = max(mx, c);
            float d = texelFetch(sceneDepth, q, 0).r;  // reversed-Z
            nearestDepth = max(nearestDepth, d);
            farthestDepth = min(farthestDepth, d);
        }
    }

    if (frame.taaParams.w < 0.5) {
        outColor = vec4(decompress(fromYCoCg(cur)), 1.0);
        return;
    }

    // Reproject unjittered pixel center into the previous frame's unjittered clip space.
    vec2 ndc = inUV * 2.0 - 1.0;
    vec4 prevClip = frame.reprojection * vec4(ndc, nearestDepth, 1.0);
    vec2 prevUV = prevClip.xy / prevClip.w * 0.5 + 0.5;
    if (prevClip.w <= 0.0 || any(lessThan(prevUV, vec2(0.0))) || any(greaterThan(prevUV, vec2(1.0)))) {
        outColor = vec4(decompress(fromYCoCg(cur)), 1.0);
        return;
    }

    // Variance clipping, intersected with the min/max box (wider when slow so 8x Halton jitter
    // accumulates cleanly on sub-pixel grilles, thin lines, distant mullions, and alpha-cutout foliage).
    vec3 mean = m1 / 9.0;
    vec3 sigma = sqrt(max(m2 / 9.0 - mean * mean, vec3(0.0)));
    // In reversed-Z (depth = zNear / z), depthJump equals deltaZ / zFar. Lower the threshold so
    // distant architectural depth discontinuities (balcony railings, recessed windows, lampposts)
    // are recognized as silhouettes alongside local high-contrast sub-pixel edges.
    float depthJump = (nearestDepth - farthestDepth) / max(nearestDepth, 1e-5);
    float silhouette = smoothstep(0.005, 0.05, depthJump);
    float subpixelContrast = smoothstep(0.015, 0.09, sigma.x);
    float edgeFactor = max(silhouette, subpixelContrast);

    // Blend Catmull-Rom toward bilinear on high-contrast sub-pixel edges and silhouettes so negative
    // bicubic lobes never ring or crawl across distant architectural details.
    vec3 histBicubic = sampleHistory(prevUV, texSize);
    vec3 histBilinear = textureLod(historyColor, prevUV, 0.0).rgb;
    vec3 hist = toYCoCg(compress(mix(histBicubic, histBilinear, edgeFactor * 0.75)));
    vec2 velocityPx = (inUV - prevUV) * texSize;
    float speed = length(velocityPx);

    // Soften single-pixel jitter steps on high-contrast sub-pixel edges and silhouettes.
    float tentMix = max(subpixelContrast * 0.65, silhouette * 0.70);
    cur = mix(cur, curTent, tentMix);
    float gamma = mix(1.75, 1.35, clamp(speed / 2.0, 0.0, 1.0)) + edgeFactor * 0.65;
    float slowFactor = 1.0 - clamp(speed / 2.0, 0.0, 1.0);
    vec3 slack = vec3(edgeFactor * (0.025 + slowFactor * 0.055), 0.012, 0.012);
    vec3 boxMin = max(mn - slack, mean - gamma * sigma);
    vec3 boxMax = min(mx + slack, mean + gamma * sigma);
    vec3 center = 0.5 * (boxMax + boxMin);
    vec3 extents = max(0.5 * (boxMax - boxMin), vec3(1e-5));
    vec3 offset = hist - center;
    vec3 ts = abs(offset / extents);
    float t = max(ts.x, max(ts.y, ts.z));
    if (t > 1.0) {
        vec3 clippedHist = center + offset / t;
        float minClip = mix(0.22, 0.12, edgeFactor);
        float maxClip = mix(0.95, 0.65, edgeFactor * slowFactor);
        float clipStrength = mix(minClip, maxClip, clamp(speed / 3.0, 0.0, 1.0));
        hist = mix(hist, clippedHist, clipStrength);
    }

    // Blend: higher history weight on slow high-frequency edges, more current frame on fast motion.
    float baseFeedback = mix(frame.taaParams.z, min(frame.taaParams.z + 0.03, 0.94), edgeFactor * slowFactor);
    float feedback = mix(baseFeedback, baseFeedback - 0.07, clamp(speed / 8.0, 0.0, 1.0));
    vec3 result = mix(cur, hist, feedback);
    outColor = vec4(decompress(fromYCoCg(result)), 1.0);
}
