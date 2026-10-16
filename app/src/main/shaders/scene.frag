#version 460
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

// Physically based shading for the Bistro materials (GGX metal/rough, ORM packed textures).
// ALPHA_MODE: 0 = opaque, 1 = alpha tested, 2 = alpha blended (glass).
// ALPHA_TO_COVERAGE: sharpen alpha for MSAA alpha-to-coverage instead of a hard alpha test.
layout(constant_id = 0) const uint ALPHA_MODE = 0u;
layout(constant_id = 1) const bool ALPHA_TO_COVERAGE = false;

layout(location = 0) in mediump vec3 inNormal;
layout(location = 1) in mediump vec4 inTangent;
layout(location = 2) in vec2 inUV;
layout(location = 3) flat in uint inMaterial;
layout(location = 4) in mediump float inSkyVisibility;

layout(location = 0) out vec4 outColor;

float D_GGX(float NdotH, float a) {
    float a2 = a * a;
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

float V_SmithGGXCorrelated(float NdotV, float NdotL, float a) {
    float a2 = a * a;
    float gv = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    float gl = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-5);
}

vec3 F_Schlick(vec3 f0, float VdotH) {
    float f = pow(1.0 - VdotH, 5.0);
    return f0 + (1.0 - f0) * f;
}

// Karis' analytic approximation of the split-sum environment BRDF (mobile friendly).
vec3 envBRDFApprox(vec3 f0, float roughness, float NdotV) {
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NdotV)) * r.x + r.y;
    vec2 ab = vec2(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + ab.y;
}

float sampleShadow(vec3 worldPos, vec3 geomNormal, float NdotL) {
    float slope = clamp(sqrt(1.0 - NdotL * NdotL) / max(NdotL, 0.05), 0.0, 10.0);
    vec3 p = worldPos + geomNormal * frame.shadowParams.y * (1.0 + slope);
    vec4 sc = frame.shadowViewProj * vec4(p, 1.0);
    vec2 uv = sc.xy * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return 1.0;
    float ref = sc.z - frame.shadowParams.z;
    float texel = frame.shadowParams.x;
    float r = frame.shadowParams.w;
    // 8-tap rotated Poisson disc PCF on top of hardware bilinear comparison; the per-pixel rotation
    // noise is integrated away by TAA.
    const vec2 poisson[8] = vec2[](
        vec2(-0.326, -0.406), vec2(-0.840, -0.074), vec2(-0.696, 0.457), vec2(-0.203, 0.621),
        vec2(0.962, -0.195), vec2(0.473, -0.480), vec2(0.519, 0.767), vec2(0.185, -0.893));
    // Interleaved gradient noise, varied per frame so TAA accumulates different rotations.
    float angle = fract(52.9829189 * fract(dot(gl_FragCoord.xy + frame.cameraPos.w * 37.0, vec2(0.06711056, 0.00583715)))) * 6.2831853;
    float ca = cos(angle), sa = sin(angle);
    mat2 rot = mat2(ca, sa, -sa, ca);
    float sum = 0.0;
    for (int i = 0; i < 8; ++i) {
        vec2 o = rot * poisson[i] * r * texel;
        sum += texture(shadowMap, vec3(uv + o, ref));
    }
    return sum * 0.125;
}

vec3 sampleEnvSpecular(vec3 R, float roughness) {
    float lod = roughness * 5.0;
    return textureLod(envSpecular, equirectUV(worldToEnv(R)), lod).rgb;
}

void main() {
    Material m = materials[inMaterial];
    vec4 base = texture(textures[nonuniformEXT(m.baseColorTexture)], inUV) * m.baseColorFactor;

    if (ALPHA_MODE == 1u) {
        if (ALPHA_TO_COVERAGE) {
            // Sharpen alpha around the cutoff so alpha-to-coverage produces crisp, anti-aliased edges.
            base.a = (base.a - m.alphaCutoff) / max(fwidth(base.a), 1e-4) + 0.5;
            if (base.a <= 0.0) discard;
        } else if (base.a < m.alphaCutoff) {
            discard;
        }
    }

    // Reconstruct the world position from the (jittered) depth.
    vec2 ndcXY = gl_FragCoord.xy / frame.renderParams.zw * 2.0 - 1.0;
    vec4 wp = frame.invViewProj * vec4(ndcXY, gl_FragCoord.z, 1.0);
    vec3 inWorldPos = wp.xyz / wp.w;

    vec3 geomN = normalize(inNormal);
    if (!gl_FrontFacing) geomN = -geomN;
    vec3 T = normalize(inTangent.xyz - geomN * dot(geomN, inTangent.xyz));
    vec3 B = cross(geomN, T) * inTangent.w;

    vec4 nt = texture(textures[nonuniformEXT(m.normalTexture)], inUV);
    vec2 nxy = vec2(nt.r, nt.a) * 2.0 - 1.0;
    float nz = sqrt(clamp(1.0 - dot(nxy, nxy), 0.0, 1.0));
    vec3 N = normalize(T * nxy.x + B * nxy.y + geomN * nz);

    // The Bistro's "Specular" maps pack roughness (G) and metalness (B); R is unused (always 0).
    vec3 orm = (m.flags & 4u) != 0u ? texture(textures[nonuniformEXT(m.ormTexture)], inUV).rgb : vec3(1.0);
    float ao = 1.0;
    float roughness = clamp(orm.g * m.roughnessFactor, 0.045, 1.0);
    float metallic = clamp(orm.b * m.metallicFactor, 0.0, 1.0);
    float a = roughness * roughness;

    vec3 V = normalize(frame.cameraPos.xyz - inWorldPos);
    float NdotV = clamp(dot(N, V), 1e-4, 1.0);
    vec3 diffuseColor = base.rgb * (1.0 - metallic);
    vec3 f0 = mix(vec3(0.04), base.rgb, metallic);

    // --- Sun (directional light with static shadow map) ---
    vec3 L = frame.sunDirection.xyz;
    float NdotL = clamp(dot(N, L), 0.0, 1.0);
    float geomNdotL = dot(geomN, L);
    vec3 color = vec3(0.0);
    if (NdotL > 0.0 && geomNdotL > -0.05) {
        float shadow = sampleShadow(inWorldPos, geomN, clamp(geomNdotL, 0.0, 1.0));
        vec3 H = normalize(V + L);
        float NdotH = clamp(dot(N, H), 0.0, 1.0);
        float VdotH = clamp(dot(V, H), 0.0, 1.0);
        vec3 F = F_Schlick(f0, VdotH);
        vec3 spec = D_GGX(NdotH, a) * V_SmithGGXCorrelated(NdotV, NdotL, a) * F;
        vec3 diff = diffuseColor * (1.0 / PI) * (1.0 - F);
        color += (diff + spec) * frame.sunColor.rgb * NdotL * shadow;
    }

    // --- Point lights (interior lamps) ---
    for (uint i = 0u; i < frame.lightCount.x; ++i) {
        PointLight pl = frame.lights[i];
        vec3 toLight = pl.positionRadius.xyz - inWorldPos;
        float d2 = dot(toLight, toLight);
        float range = pl.positionRadius.w;
        if (d2 > range * range) continue;
        vec3 Lp = toLight * inversesqrt(d2);
        float nl = clamp(dot(N, Lp), 0.0, 1.0);
        if (nl <= 0.0) continue;
        float falloff = clamp(1.0 - pow(d2 / (range * range), 2.0), 0.0, 1.0);
        falloff = falloff * falloff / max(d2, 0.01);
        vec3 H = normalize(V + Lp);
        vec3 F = F_Schlick(f0, clamp(dot(V, H), 0.0, 1.0));
        vec3 spec = D_GGX(clamp(dot(N, H), 0.0, 1.0), a) * V_SmithGGXCorrelated(NdotV, nl, a) * F;
        color += (diffuseColor * (1.0 / PI) * (1.0 - F) + spec) * pl.color.rgb * nl * falloff;
    }

    // --- Image based ambient lighting, occluded by baked sky visibility and material AO ---
    float skyVis = inSkyVisibility;
    float diffuseOcclusion = ao * skyVis;
    vec3 irradiance = shIrradiance(worldToEnv(N)) * frame.envRotation.w;
    // Light bounced off the sunlit surroundings keeps occluded areas from going completely black.
    vec3 bounce = frame.sunColor.rgb * 0.035 * (0.6 + 0.4 * skyVis);
    color += diffuseColor * (irradiance * diffuseOcclusion + bounce * ao) * (1.0 / PI);

    vec3 R = reflect(-V, N);
    float specOcclusion = clamp(pow(NdotV + diffuseOcclusion, exp2(-16.0 * roughness - 1.0)) - 1.0 + diffuseOcclusion,
                                0.0, 1.0);
    vec3 envSpec = sampleEnvSpecular(R, roughness) * frame.envRotation.w;
    color += envSpec * envBRDFApprox(f0, roughness, NdotV) * specOcclusion;

    // --- Emissive ---
    if ((m.flags & 2u) != 0u) {
        vec3 e = texture(textures[nonuniformEXT(m.emissiveTexture)], inUV).rgb;
        color += e * m.emissiveFactor * frame.postParams.w;
    }

    // --- Aerial perspective / height fog ---
    float dist = length(frame.cameraPos.xyz - inWorldPos);
    float heightTerm = exp(-max(inWorldPos.y - frame.fogParams.z, 0.0) * frame.fogParams.y);
    float fog = min(1.0 - exp(-dist * frame.fogParams.x * heightTerm), frame.fogParams.w);

    if (ALPHA_MODE == 2u) {
        // Glass: premultiplied alpha. Reflections are added on top of the transmitted background.
        float alpha = clamp(base.a, 0.0, 1.0);
        vec3 reflection = envSpec * envBRDFApprox(f0, roughness, NdotV) * specOcclusion;
        vec3 body = (color - reflection) * alpha + reflection;
        body = mix(body, frame.fogColor.rgb * alpha, fog);
        float fresnelAlpha = clamp(alpha + luminance(envBRDFApprox(vec3(0.04), roughness, NdotV)), 0.0, 1.0);
        outColor = vec4(body, fresnelAlpha);
    } else {
        color = mix(color, frame.fogColor.rgb, fog);
        outColor = vec4(color, ALPHA_MODE == 1u ? base.a : 1.0);
    }
}
