#version 460
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

// Alpha-tested shadow casters (foliage etc).
layout(location = 0) in vec2 inUV;
layout(location = 1) flat in uint inMaterial;

void main() {
    Material m = materials[inMaterial];
    float alpha = texture(textures[nonuniformEXT(m.baseColorTexture)], inUV).a * m.baseColorFactor.a;
    if (alpha < m.alphaCutoff) discard;
}
