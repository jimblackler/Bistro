#version 460
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

// Depth-only pass for the (static) sun shadow map.
layout(location = 0) in vec3 inPosition;
layout(location = 3) in vec2 inUV;

layout(push_constant) uniform Push {
    mat4 lightViewProj;
} pc;

layout(location = 0) out vec2 outUV;
layout(location = 1) flat out uint outMaterial;

void main() {
    outUV = inUV;
    outMaterial = clusters[gl_InstanceIndex].materialIndex;
    gl_Position = pc.lightViewProj * vec4(inPosition, 1.0);
}
