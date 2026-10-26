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
    vec4 exposureParams2; // x = adaptation speed (1/s), y = auto exposure enabled, z = camera indoors, w = translucency
    vec4 occlusionParams; // x = Hi-Z width, y = Hi-Z height, z = Hi-Z mip count, w = occlusion culling enabled
    vec4 taaParams;       // xy = projection jitter (NDC), z = history feedback, w = history valid
    vec4 aoParams;        // x = previous frame's SSAO valid, y = radius (m), z = intensity exponent, w = debug view
    PointLight lights[MAX_POINT_LIGHTS];
    uvec4 lightCount;     // x = point light count, y = feature flags (bit0=SSR, bit1=SSCS, bit2=GodRays, bit3=ShadowPCF)
} frame;

layout(set = 0, binding = 1, std430) readonly buffer ClusterBuffer { Cluster clusters[]; };
layout(set = 0, binding = 2, std430) readonly buffer MaterialBuffer { Material materials[]; };
layout(set = 0, binding = 5) uniform sampler2DShadow shadowMap;
layout(set = 0, binding = 6) uniform sampler2D skyTexture;
layout(set = 0, binding = 7) uniform sampler2D envSpecular;
layout(set = 0, binding = 8) uniform sampler2D hdrTexture;
layout(set = 0, binding = 9) uniform sampler2D bloomTexture;
// Previous frame's half-res SSAO (R = ambient visibility, G = view depth), reprojected in scene.frag.
layout(set = 0, binding = 10) uniform sampler2D aoTexture;
layout(set = 0, binding = 11, std430) buffer ExposureBuffer {
    float exposure;   // adapted exposure multiplier
    float avgLogLum;  // measured log2 luminance
    float pad0;
    float pad1;
} adapt;
layout(set = 0, binding = 13) uniform sampler2D hizTexture;      // previous frame's Hi-Z pyramid
layout(set = 0, binding = 14) uniform sampler2D prevHdrTexture;  // previous frame's resolved HDR colour (for SSR)
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

float henyeyGreenstein(float cosTheta, float g) {
    float g2 = g * g;
    float denom = max(1.0 + g2 - 2.0 * g * cosTheta, 1e-3);
    return (1.0 - g2) / (4.0 * PI * denom * sqrt(denom));
}

// Evaluates anisotropic Mie + Rayleigh height fog and marches 16 jittered steps through
// the camera-fitted directional shadow map to accumulate volumetric sun god-rays (crepuscular rays).
vec3 evaluateVolumetricFog(vec3 worldPos, vec3 rayDir, float dist, vec2 fragCoord, out float outTransmittance,
                           out vec3 outSunShafts) {
    float endHeightTerm = exp(-max(worldPos.y - frame.fogParams.z, 0.0) * frame.fogParams.y);
    bool cameraIndoors = frame.exposureParams2.z > 0.5;
    bool godRaysEnabled = (frame.lightCount.y & 4u) != 0u;
    if (cameraIndoors || !godRaysEnabled) {
        float fog = min(1.0 - exp(-dist * frame.fogParams.x * endHeightTerm), frame.fogParams.w);
        outTransmittance = 1.0 - fog;
        outSunShafts = vec3(0.0);
        return frame.fogColor.rgb * fog;
    }

    vec3 camPos = frame.cameraPos.xyz;
    vec3 L = frame.sunDirection.xyz;
    float cosTheta = clamp(dot(rayDir, L), -1.0, 1.0);

    // March the near-to-mid atmosphere (0.4m to 28m) through the camera-fitted directional shadow map.
    const float startDist = 0.40;
    const float maxMarchDist = 28.0;
    float marchDist = min(dist, maxMarchDist);

    float sunShaftVis = 1.0;
    float avgHeightTerm = endHeightTerm;
    if (marchDist > startDist) {
        vec3 rayStart = camPos + rayDir * startDist;
        vec3 rayEnd = camPos + rayDir * marchDist;
        vec3 startShadow = (frame.shadowViewProj * vec4(rayStart, 1.0)).xyz;
        vec3 endShadow = (frame.shadowViewProj * vec4(rayEnd, 1.0)).xyz;
        startShadow.xy = startShadow.xy * 0.5 + 0.5;
        endShadow.xy = endShadow.xy * 0.5 + 0.5;

        // Evaluate Interleaved Gradient Noise at integer pixel coordinates and add the TAA Halton phase
        // outside the outer fract() so the spatial blue-noise spectrum is never aliased into stripes.
        float ign = fract(52.9829189 * fract(dot(floor(fragCoord), vec2(0.06711056, 0.00583715))));
        float taaPhase = (frame.taaParams.x * frame.renderParams.z + frame.taaParams.y * frame.renderParams.w * 1.6180339) * 0.5;
        float jitter = fract(ign + taaPhase);

        float accumSun = 0.0;
        float accumWeight = 0.0;
        float heightSum = 0.0;
        const int STEPS = 16;
        for (int i = 0; i < STEPS; ++i) {
            float u = (float(i) + jitter) * (1.0 / float(STEPS));
            float sampleY = mix(rayStart.y, rayEnd.y, u);
            float localDensity = exp(-max(sampleY - frame.fogParams.z, 0.0) * frame.fogParams.y);
            float sampleDist = mix(startDist, marchDist, u);
            float distWeight = exp(-sampleDist * 0.035);
            float w = localDensity * distWeight;

            vec3 sc = mix(startShadow, endShadow, u);
            float edge = min(min(sc.x, 1.0 - sc.x), min(sc.y, 1.0 - sc.y));
            float inBounds = smoothstep(0.0, 0.06, edge) * step(0.0, sc.z) * step(sc.z, 1.0);
            float shadowSample = 1.0;
            if (inBounds > 0.0) {
                shadowSample = textureLod(shadowMap, vec3(sc.xy, clamp(sc.z, 0.0, 1.0)), 0.0);
                shadowSample = mix(1.0, shadowSample, inBounds);
            }
            accumSun += shadowSample * w;
            accumWeight += w;
            heightSum += localDensity;
        }
        sunShaftVis = accumWeight > 1e-5 ? (accumSun / accumWeight) : 1.0;
        avgHeightTerm = heightSum * (1.0 / float(STEPS));
    }

    float pathHeightTerm = mix(avgHeightTerm, endHeightTerm, clamp((dist - maxMarchDist) / 40.0, 0.0, 0.65));
    float fog = min(1.0 - exp(-dist * frame.fogParams.x * pathHeightTerm), frame.fogParams.w);
    outTransmittance = 1.0 - fog;

    // Dual-lobe Henyey-Greenstein Mie phase function (g = 0.45 broad haze + g = 0.70 forward solar aureole),
    // weighted toward the sunward hemisphere so crepuscular rays bloom naturally when looking toward the sun.
    float sunward = smoothstep(-0.15, 0.85, cosTheta);
    float mieBroad = henyeyGreenstein(cosTheta, 0.45);
    float mieForward = henyeyGreenstein(cosTheta, 0.70);
    float miePhase = min(mix(mieBroad, mieForward, 0.55) * (0.20 + 0.80 * sunward), 0.75);
    float rayleighPhase = 1.0 + 0.16 * cosTheta * cosTheta;

    // Ambient sky fog is slightly cooler in shadowed air and warmer in sunlit air.
    vec3 shadowFogTint = vec3(0.90, 0.94, 1.00) * 0.88;
    vec3 litFogTint = vec3(1.03, 1.00, 0.96);
    vec3 ambientFogColor = frame.fogColor.rgb * mix(shadowFogTint, litFogTint, sunShaftVis) * rayleighPhase;
    vec3 mieAerialGlow = frame.sunColor.rgb * vec3(1.06, 0.94, 0.80) * (miePhase * (0.20 + 0.80 * sunShaftVis));
    vec3 baseInscatter = (ambientFogColor + mieAerialGlow * 0.30) * fog;

    // Volumetric crepuscular sun shafts (god-rays) along the marched near-to-mid ray segment.
    float effDensity = frame.fogParams.x * 1.25;
    float shaftOpticalDepth = 1.0 - exp(-max(marchDist - startDist, 0.0) * effDensity * avgHeightTerm);
    outSunShafts = frame.sunColor.rgb * vec3(1.08, 0.95, 0.78) * (shaftOpticalDepth * miePhase * sunShaftVis * 0.95);
    return baseInscatter;
}
