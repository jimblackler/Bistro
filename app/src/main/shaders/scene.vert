#version 460
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inNormal;    // octahedral
layout(location = 2) in vec4 inTangent;   // xyz + bitangent sign
layout(location = 3) in vec2 inUV;
layout(location = 4) in vec4 inExtra;     // x = baked sky visibility

// Varyings are kept small (and half precision where possible): on tile-based GPUs every vertex output
// is written to and read back from the parameter buffer. World position is reconstructed from depth.
layout(location = 0) out mediump vec3 outNormal;
layout(location = 1) out mediump vec4 outTangent;
layout(location = 2) out vec2 outUV;
layout(location = 3) flat out uint outMaterial;
layout(location = 4) out mediump float outSkyVisibility;

void main() {
    // firstInstance of each indirect draw carries the cluster index.
    Cluster cl = clusters[gl_InstanceIndex];
    outMaterial = cl.materialIndex;
    outNormal = octDecode(inNormal);
    outTangent = vec4(inTangent.xyz, inTangent.w < 0.0 ? -1.0 : 1.0);
    outUV = inUV;
    outSkyVisibility = inExtra.x;
    vec4 clip = frame.viewProj * vec4(inPosition, 1.0);
    if (cl.bucket == 1u) {
        // Keep alpha-tested foliage on the unjittered grid so sub-pixel leaves and canopy cutouts
        // do not binary-flip in and out of 1x pixel centers every frame.
        clip.xy -= frame.taaParams.xy * clip.w;
    }
    gl_Position = clip;
}
