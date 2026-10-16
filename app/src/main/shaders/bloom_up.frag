#version 460

// Bloom upsample: 3x3 tent filter, additively blended onto the next larger mip.
layout(set = 1, binding = 0) uniform sampler2D srcTexture;

layout(push_constant) uniform Push {
    vec2 srcTexelSize;
    float radius;
    float weight;
} pc;

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

void main() {
    vec2 t = pc.srcTexelSize * pc.radius;
    vec3 s = texture(srcTexture, inUV).rgb * 4.0;
    s += (texture(srcTexture, inUV + vec2(-t.x, 0)).rgb + texture(srcTexture, inUV + vec2(t.x, 0)).rgb +
          texture(srcTexture, inUV + vec2(0, -t.y)).rgb + texture(srcTexture, inUV + vec2(0, t.y)).rgb) * 2.0;
    s += texture(srcTexture, inUV + vec2(-t.x, -t.y)).rgb + texture(srcTexture, inUV + vec2(t.x, -t.y)).rgb +
         texture(srcTexture, inUV + vec2(-t.x, t.y)).rgb + texture(srcTexture, inUV + vec2(t.x, t.y)).rgb;
    outColor = vec4(s / 16.0 * pc.weight, 1.0);
}
