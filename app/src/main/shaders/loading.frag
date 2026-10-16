#version 460

// Loading screen: dark vignette background with a progress bar, honouring swapchain pre-rotation.
layout(push_constant) uniform Push {
    vec2 outputSize;
    float progress;
    float rotation;
    float time;
    float srgbEncode;
} pc;

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

vec2 logicalUV(vec2 uv) {
    int rot = int(pc.rotation + 0.5);
    if (rot == 1) return vec2(uv.y, 1.0 - uv.x);
    if (rot == 2) return vec2(1.0 - uv.x, 1.0 - uv.y);
    if (rot == 3) return vec2(1.0 - uv.y, uv.x);
    return uv;
}

void main() {
    vec2 uv = logicalUV(gl_FragCoord.xy / pc.outputSize);
    vec2 logicalSize = (int(pc.rotation + 0.5) % 2 == 1) ? pc.outputSize.yx : pc.outputSize;
    vec2 px = uv * logicalSize;

    vec2 v = uv - 0.5;
    vec3 color = mix(vec3(0.035, 0.030, 0.028), vec3(0.010, 0.009, 0.008), dot(v, v) * 2.5);

    // Progress bar centred near the bottom third.
    float barWidth = logicalSize.x * 0.4;
    float barHeight = max(4.0, logicalSize.y * 0.006);
    vec2 barMin = vec2((logicalSize.x - barWidth) * 0.5, logicalSize.y * 0.72);
    vec2 local = px - barMin;
    if (local.x >= 0.0 && local.x <= barWidth && local.y >= 0.0 && local.y <= barHeight) {
        float filled = step(local.x, barWidth * pc.progress);
        float shimmer = 0.5 + 0.5 * sin(local.x * 0.05 - pc.time * 4.0);
        vec3 fillColor = mix(vec3(0.85, 0.55, 0.25), vec3(1.0, 0.8, 0.5), shimmer * 0.4);
        color = mix(vec3(0.12, 0.11, 0.10), fillColor, filled);
    }
    if (pc.srgbEncode > 0.5) color = pow(color, vec3(1.0 / 2.2));
    outColor = vec4(color, 1.0);
}
