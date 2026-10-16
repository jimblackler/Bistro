// Shared declarations for all Bistro shaders. Must match FrameUniforms / GpuCluster / GpuMaterial in C++.
#extension GL_EXT_nonuniform_qualifier : require
precision highp float;
precision highp int;

const float PI = 3.14159265359;

struct Cluster {
    vec3 aabbMin;
    uint materialIndex;
    vec3 aabbMax;
    uint bucket;
    uint firstIndex;
    uint indexCount;
    int vertexOffset;
    uint pad;
};

struct Material {
    vec4 baseColorFactor;
    vec3 emissiveFactor;
    float alphaCutoff;
    uint baseColorTexture;
    uint normalTexture;
    uint ormTexture;
    uint emissiveTexture;
    uint alphaMode;
    uint flags;
    float roughnessFactor;
    float metallicFactor;
};

struct PointLight {
    vec4 positionRadius;  // xyz position, w = range
    vec4 color;           // rgb intensity
};

const uint MAX_POINT_LIGHTS = 8u;

layout(set = 0, binding = 0, std140) uniform FrameUniforms {
    mat4 viewProj;
    mat4 invViewProj;
    mat4 shadowViewProj;
    mat4 prevViewProj;
    mat4 reprojection;    // current (unjittered) clip -> previous frame clip, for TAA
    vec4 cameraPos;       // xyz, w = time (seconds)
    vec4 sunDirection;    // xyz towards the sun (world space)
    vec4 sunColor;        // rgb sun irradiance, w = cos(sun angular radius)
    vec4 envRotation;     // x = cos(yaw), y = sin(yaw), z = sky intensity, w = IBL intensity
    vec4 sh[9];           // SH9 irradiance (already convolved with the cosine lobe)
    vec4 frustumPlanes[6];
    vec4 fogParams;       // x = density, y = height falloff, z = base height, w = max opacity
    vec4 fogColor;        // rgb in-scatter colour
    vec4 renderParams;    // x = exposure, y = bloom strength, z = render width, w = render height
    vec4 cullParams;      // x = cluster count, y = draw capacity per bucket, z = projection scale (px), w = min px
    vec4 postParams;      // x = pre-rotation (0..3), y = output width, z = output height, w = emissive intensity
    vec4 shadowParams;    // x = texel size (uv), y = normal bias (m), z = depth bias, w = filter radius (texels)
    vec4 extraParams;     // x = sRGB output needs manual encode, y = vignette, z = saturation, w = contrast
    vec4 exposureParams;  // x = frame dt (s), y = key (target mid grey), z = min exposure, w = max exposure
    vec4 exposureParams2; // x = adaptation speed (1/s), y = auto exposure enabled, z/w unused
    vec4 occlusionParams; // x = Hi-Z width, y = Hi-Z height, z = Hi-Z mip count, w = occlusion culling enabled
    vec4 taaParams;       // xy = projection jitter (NDC), z = history feedback, w = history valid
    PointLight lights[MAX_POINT_LIGHTS];
    uvec4 lightCount;
} frame;

layout(set = 0, binding = 1, std430) readonly buffer ClusterBuffer { Cluster clusters[]; };
layout(set = 0, binding = 2, std430) readonly buffer MaterialBuffer { Material materials[]; };
layout(set = 0, binding = 5) uniform sampler2DShadow shadowMap;
layout(set = 0, binding = 6) uniform sampler2D skyTexture;
layout(set = 0, binding = 7) uniform sampler2D envSpecular;
layout(set = 0, binding = 8) uniform sampler2D hdrTexture;
layout(set = 0, binding = 9) uniform sampler2D bloomTexture;
layout(set = 0, binding = 11, std430) buffer ExposureBuffer {
    float exposure;   // adapted exposure multiplier
    float avgLogLum;  // measured log2 luminance
    float pad0;
    float pad1;
} adapt;
layout(set = 0, binding = 13) uniform sampler2D hizTexture;  // previous frame's Hi-Z pyramid
layout(set = 0, binding = 15) uniform sampler2D textures[];

vec3 octDecode(vec2 e) {
    vec3 v = vec3(e, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0.0) {
        vec2 s = vec2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
        v.xy = (1.0 - abs(v.yx)) * s;
    }
    return normalize(v);
}

// World direction -> environment map direction (rotation about Y by -yaw).
vec3 worldToEnv(vec3 d) {
    float c = frame.envRotation.x, s = frame.envRotation.y;
    return vec3(c * d.x + s * d.z, d.y, -s * d.x + c * d.z);
}

vec2 equirectUV(vec3 d) {
    return vec2(atan(d.x, -d.z) * (0.5 / PI) + 0.5, acos(clamp(d.y, -1.0, 1.0)) / PI);
}

vec3 shIrradiance(vec3 n) {
    // n in environment space
    vec3 r = frame.sh[0].rgb * 0.282095
           + frame.sh[1].rgb * 0.488603 * n.y
           + frame.sh[2].rgb * 0.488603 * n.z
           + frame.sh[3].rgb * 0.488603 * n.x
           + frame.sh[4].rgb * 1.092548 * n.x * n.y
           + frame.sh[5].rgb * 1.092548 * n.y * n.z
           + frame.sh[6].rgb * 0.315392 * (3.0 * n.z * n.z - 1.0)
           + frame.sh[7].rgb * 1.092548 * n.x * n.z
           + frame.sh[8].rgb * 0.546274 * (n.x * n.x - n.y * n.y);
    return max(r, vec3(0.0));
}

float luminance(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
