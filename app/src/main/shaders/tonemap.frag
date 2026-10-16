#version 460
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

// Final composite: bloom, exposure, filmic tone mapping, grading and swapchain pre-rotation.
layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

// ACES fitted curve (Stephen Hill) including the sRGB->ACEScg->sRGB matrices.
vec3 acesFitted(vec3 color) {
    const mat3 inputMat = mat3(0.59719, 0.07600, 0.02840, 0.35458, 0.90834, 0.13383, 0.04823, 0.01566, 0.83777);
    const mat3 outputMat = mat3(1.60475, -0.10208, -0.00327, -0.53108, 1.10813, -0.07276, -0.07367, -0.00605, 1.07602);
    color = inputMat * color;
    vec3 a = color * (color + 0.0245786) - 0.000090537;
    vec3 b = color * (0.983729 * color + 0.4329510) + 0.238081;
    color = a / b;
    return clamp(outputMat * color, 0.0, 1.0);
}

vec3 linearToSrgb(vec3 c) {
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
}

vec2 logicalUV(vec2 uv) {
    // Undo the swapchain pre-transform so the image appears upright without compositor rotation.
    int rot = int(frame.postParams.x + 0.5);
    if (rot == 1) return vec2(uv.y, 1.0 - uv.x);
    if (rot == 2) return vec2(1.0 - uv.x, 1.0 - uv.y);
    if (rot == 3) return vec2(1.0 - uv.y, uv.x);
    return uv;
}

void main() {
    vec2 outUV = gl_FragCoord.xy / frame.postParams.yz;
    vec2 uv = logicalUV(outUV);

    vec3 hdr = texture(hdrTexture, uv).rgb;
    vec3 bloom = texture(bloomTexture, uv).rgb;
    vec3 color = hdr + bloom * frame.renderParams.y;
    color *= frame.renderParams.x * adapt.exposure;

    color = acesFitted(color);

    // Mild grading: saturation and contrast around mid grey.
    float l = luminance(color);
    color = mix(vec3(l), color, frame.extraParams.z);
    color = clamp((color - 0.18) * frame.extraParams.w + 0.18, 0.0, 1.0);

    // Vignette.
    vec2 v = uv - 0.5;
    color *= 1.0 - frame.extraParams.y * dot(v, v) * 1.6;

    if (frame.extraParams.x > 0.5) color = linearToSrgb(color);
    // Triangular dither to hide banding in the sky gradients.
    float n = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453) +
              fract(sin(dot(gl_FragCoord.xy, vec2(39.3468, 11.135))) * 24634.6345) - 1.0;
    color += n / 255.0 * (frame.extraParams.x > 0.5 ? 1.0 : 0.0);
    outColor = vec4(color, 1.0);
}
