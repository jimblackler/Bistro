#version 460

// Bloom downsample (13-tap filter from Jimenez, "Next Generation Post Processing in Call of Duty").
// The first pass applies a soft threshold and a Karis average to suppress fireflies.
layout(set = 1, binding = 0) uniform sampler2D srcTexture;

layout(push_constant) uniform Push {
    vec2 srcTexelSize;
    float firstPass;
    float threshold;
} pc;

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

float karisWeight(vec3 c) { return 1.0 / (1.0 + dot(c, vec3(0.2126, 0.7152, 0.0722))); }

vec3 prefilter(vec3 c) {
    float brightness = max(c.r, max(c.g, c.b));
    float knee = pc.threshold * 0.5;
    float soft = clamp(brightness - pc.threshold + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee + 1e-5);
    float contribution = max(soft, brightness - pc.threshold) / max(brightness, 1e-5);
    return c * contribution;
}

void main() {
    vec2 t = pc.srcTexelSize;
    vec3 a = texture(srcTexture, inUV + t * vec2(-2, -2)).rgb;
    vec3 b = texture(srcTexture, inUV + t * vec2(0, -2)).rgb;
    vec3 c = texture(srcTexture, inUV + t * vec2(2, -2)).rgb;
    vec3 d = texture(srcTexture, inUV + t * vec2(-2, 0)).rgb;
    vec3 e = texture(srcTexture, inUV).rgb;
    vec3 f = texture(srcTexture, inUV + t * vec2(2, 0)).rgb;
    vec3 g = texture(srcTexture, inUV + t * vec2(-2, 2)).rgb;
    vec3 h = texture(srcTexture, inUV + t * vec2(0, 2)).rgb;
    vec3 i = texture(srcTexture, inUV + t * vec2(2, 2)).rgb;
    vec3 j = texture(srcTexture, inUV + t * vec2(-1, -1)).rgb;
    vec3 k = texture(srcTexture, inUV + t * vec2(1, -1)).rgb;
    vec3 l = texture(srcTexture, inUV + t * vec2(-1, 1)).rgb;
    vec3 m = texture(srcTexture, inUV + t * vec2(1, 1)).rgb;

    vec3 result;
    if (pc.firstPass > 0.5) {
        vec3 g0 = (j + k + l + m) * 0.25;
        vec3 g1 = (a + b + d + e) * 0.25;
        vec3 g2 = (b + c + e + f) * 0.25;
        vec3 g3 = (d + e + g + h) * 0.25;
        vec3 g4 = (e + f + h + i) * 0.25;
        float w0 = karisWeight(g0) * 0.5, w1 = karisWeight(g1) * 0.125, w2 = karisWeight(g2) * 0.125;
        float w3 = karisWeight(g3) * 0.125, w4 = karisWeight(g4) * 0.125;
        result = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
        result = prefilter(result);
    } else {
        result = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
    }
    outColor = vec4(max(result, vec3(0.0)), 1.0);
}
