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
    // Interleaved gradient noise, varied per frame on jittered geometry so TAA accumulates different rotations.
    float timeSeed = ALPHA_MODE == 1u ? 0.0 : frame.cameraPos.w * 37.0;
    float angle = fract(52.9829189 * fract(dot(gl_FragCoord.xy + timeSeed, vec2(0.06711056, 0.00583715)))) * 6.2831853;
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

// Screen-space ambient occlusion computed from the previous frame's depth (ao.comp), reprojected to this
// fragment. The stored view depth rejects samples from a different surface (disocclusions, edges), which
// then fall back to unoccluded.
float sampleSSAO(vec3 worldPos) {
    if (frame.aoParams.x < 0.5) return 1.0;
    vec4 pc = frame.prevViewProj * vec4(worldPos, 1.0);
    if (pc.w <= 0.0) return 1.0;
    vec2 uv = pc.xy / pc.w * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return 1.0;
    vec2 a = textureLod(aoTexture, uv, 0.0).rg;
    float rel = abs(a.g - pc.w) / pc.w;
    return mix(a.r, 1.0, smoothstep(0.03, 0.08, rel));
}

void main() {
    // Compute all screen-space derivatives in uniform control flow before any discard or dynamic branch.
    vec2 dUVdx = dFdx(inUV);
    vec2 dUVdy = dFdy(inUV);
    float skyVisGrad = fwidth(inSkyVisibility);
    vec3 dGeomNdx = dFdx(inNormal);
    vec3 dGeomNdy = dFdy(inNormal);

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

    // Reconstruct the world position from the depth (accounting for unjittered ALPHA_MODE == 1u).
    vec2 ndcXY = gl_FragCoord.xy / frame.renderParams.zw * 2.0 - 1.0;
    if (ALPHA_MODE == 1u) ndcXY += frame.taaParams.xy;
    vec4 wp = frame.invViewProj * vec4(ndcXY, gl_FragCoord.z, 1.0);
    vec3 inWorldPos = wp.xyz / wp.w;

    vec3 geomN = normalize(inNormal);
    if (!gl_FrontFacing) geomN = -geomN;
    vec3 Traw = inTangent.xyz - geomN * dot(geomN, inTangent.xyz);
    vec3 T = Traw * inversesqrt(max(dot(Traw, Traw), 1e-8));
    vec3 B = cross(geomN, T) * inTangent.w;

    vec4 nt = texture(textures[nonuniformEXT(m.normalTexture)], inUV);
    vec2 nxy = vec2(nt.r, nt.a) * 2.0 - 1.0;
    float uvFootprint = sqrt(max(dot(dUVdx, dUVdx) + dot(dUVdy, dUVdy), 0.0));
    if (ALPHA_MODE == 1u) {
        // Fade high-frequency normal perturbation on distant alpha-tested leaves to prevent sub-pixel shimmer.
        nxy *= 1.0 - smoothstep(0.015, 0.08, uvFootprint) * 0.75;
    }
    float nz = sqrt(clamp(1.0 - dot(nxy, nxy), 0.0, 1.0));
    vec3 N = normalize(T * nxy.x + B * nxy.y + geomN * nz);

    float louverOcc = 1.0;
    float louverNormalVar = 0.0;
    if ((m.flags & 16u) != 0u) {
        float phi = inUV.x;
        float stepMeters = abs(inUV.y);
        bool highContrast = inUV.y > 0.0;
        float freq = sqrt(max(dUVdx.x * dUVdx.x + dUVdy.x * dUVdy.x, 0.0));
        float stepScale = smoothstep(0.002, 0.006, stepMeters);
        float fClamped = min(freq, 0.5);
        float sinc = fClamped > 1e-4 ? sin(fClamped * PI) / (fClamped * PI) : 1.0;
        float bandLimit = sinc * smoothstep(0.32, 0.08, freq) * stepScale;
        float wave = cos(phi * (2.0 * PI));
        float grooveMin = highContrast ? 0.08 : 0.45;
        float meanOcc = 0.5 * (grooveMin + 1.0);
        louverOcc = mix(meanOcc, mix(grooveMin, 1.0, 0.5 + 0.5 * wave), bandLimit);
        float tiltAmp = (highContrast ? 0.16 : 0.08) * bandLimit;
        N = normalize(N + T * (sin(phi * (2.0 * PI)) * tiltAmp));
        louverNormalVar = (1.0 - bandLimit) * (highContrast ? 0.04 : 0.015);
    }

    // The Bistro's "Specular" maps pack roughness (G) and metalness (B); R is unused (always 0).
    vec3 orm = (m.flags & 4u) != 0u ? texture(textures[nonuniformEXT(m.ormTexture)], inUV).rgb : vec3(1.0);
    float ao = ALPHA_MODE == 2u ? 1.0 : sampleSSAO(inWorldPos);
    if (ALPHA_MODE == 1u) {
        // Half-res depth-reconstructed SSAO fluctuates across porous alpha-cutout leaves; rely primarily on baked skyVis.
        ao = mix(ao, 1.0, 0.65);
    }
    float skyVis = mix(inSkyVisibility, 1.0, smoothstep(0.25, 0.85, skyVisGrad));
    float diffuseOcclusion = ao * skyVis;
    float directLightOcclusion = mix(1.0, diffuseOcclusion, 0.65);
    float roughness = clamp(orm.g * m.roughnessFactor, 0.04, 1.0);
    if (ALPHA_MODE == 1u) {
        roughness = max(roughness, mix(0.04, 0.55, smoothstep(0.015, 0.08, uvFootprint)));
    }
    float metallic = clamp(orm.b * m.metallicFactor, 0.0, 1.0);
    // Geometric and normal-map specular anti-aliasing (Kaplanyan 2016): broaden GGX roughness by the
    // screen-space normal variance so sub-pixel grilles, bevels, and normal grooves don't alias.
    float normalVar = min(dot(dGeomNdx, dGeomNdx) + dot(dGeomNdy, dGeomNdy) +
                          dot(nxy, nxy) * min(uvFootprint * 8.0, 0.25) + louverNormalVar, 0.25);
    float aOrig = roughness * roughness;
    float a = clamp(sqrt(aOrig * aOrig + normalVar * 0.5), 0.0016, 1.0);
    roughness = sqrt(a);

    vec3 V = normalize(frame.cameraPos.xyz - inWorldPos);
    float NdotV = clamp(dot(N, V), 1e-4, 1.0);
    vec3 diffuseColor = base.rgb * (1.0 - metallic) * louverOcc;
    vec3 f0 = mix(vec3(0.04), base.rgb, metallic) * mix(0.20, 1.0, louverOcc);

    // --- Thin-surface translucency setup (leaves, grass, curtains, awnings) ---
    // Light passing through a thin leaf or fabric is filtered by the material on the way in and out, so
    // the transmitted colour is a more saturated version of the albedo.
    float translucency = (m.flags & 8u) != 0u ? frame.exposureParams2.w : 0.0;
    vec3 transTint = vec3(0.0);
    if (translucency > 0.0) {
        float maxC = max(diffuseColor.r, max(diffuseColor.g, max(diffuseColor.b, 1e-3)));
        transTint = diffuseColor * mix(vec3(1.0), diffuseColor / maxC, 0.6) * (translucency / PI);
    }

    // --- Sun (directional light with static shadow map) ---
    vec3 L = frame.sunDirection.xyz;
    float NdotL = clamp(dot(N, L), 0.0, 1.0);
    float geomNdotL = dot(geomN, L);
    vec3 color = vec3(0.0);
    vec3 specDirect = vec3(0.0);
    if (NdotL > 0.0 && geomNdotL > -0.05) {
        float shadow = sampleShadow(inWorldPos, geomN, clamp(geomNdotL, 0.0, 1.0));
        vec3 H = normalize(V + L);
        float NdotH = clamp(dot(N, H), 0.0, 1.0);
        float VdotH = clamp(dot(V, H), 0.0, 1.0);
        vec3 F = F_Schlick(f0, VdotH);
        vec3 spec = D_GGX(NdotH, a) * V_SmithGGXCorrelated(NdotV, NdotL, a) * F;
        vec3 diff = diffuseColor * (1.0 / PI) * (1.0 - F);
        vec3 sunRadiance = frame.sunColor.rgb * NdotL * shadow;
        color += (diff + spec) * sunRadiance;
        specDirect += spec * sunRadiance;
    }
    if (translucency > 0.0 && geomNdotL < 0.05) {
        // Back-lit side: wrapped transmission plus a forward-scattering lobe that makes leaves and
        // curtains glow when the camera looks towards the sun through them. The shadow lookup is
        // taken from the sun-facing side so leaves deep inside a canopy stay dark.
        float backNdotL = clamp((0.25 - dot(N, L)) * 0.8, 0.0, 1.0);
        if (backNdotL > 0.0) {
            float shadowT = sampleShadow(inWorldPos, -geomN, clamp(-geomNdotL, 0.0, 1.0));
            float VdotL = clamp(-dot(V, L), 0.0, 1.0);
            float VdotL2 = VdotL * VdotL;
            float forward = VdotL2 * VdotL2 * VdotL2;
            color += transTint * frame.sunColor.rgb * ((0.35 + 2.5 * forward) * backNdotL * shadowT);
        }
    }

    // --- Point lights (interior lamps) ---
    for (uint i = 0u; i < frame.lightCount.x; ++i) {
        PointLight pl = frame.lights[i];
        vec3 toLight = pl.positionRadius.xyz - inWorldPos;
        float d2 = dot(toLight, toLight);
        float range = pl.positionRadius.w;
        if (d2 > range * range) continue;
        vec3 Lp = toLight * inversesqrt(d2);
        float nlRaw = dot(N, Lp);
        float falloff = clamp(1.0 - pow(d2 / (range * range), 2.0), 0.0, 1.0);
        falloff = falloff * falloff / max(d2, 0.01);
        if (translucency > 0.0 && nlRaw < 0.25) {
            float bt = clamp((0.25 - nlRaw) * 0.8, 0.0, 1.0);
            float vl = clamp(-dot(V, Lp), 0.0, 1.0);
            float vl2 = vl * vl;
            color += transTint * pl.color.rgb * (falloff * bt * (0.35 + 2.5 * vl2 * vl2 * vl2));
        }
        float nl = clamp(nlRaw, 0.0, 1.0);
        if (nl <= 0.0) continue;
        vec3 H = normalize(V + Lp);
        vec3 F = F_Schlick(f0, clamp(dot(V, H), 0.0, 1.0));
        vec3 spec = D_GGX(clamp(dot(N, H), 0.0, 1.0), a) * V_SmithGGXCorrelated(NdotV, nl, a) * F;
        vec3 lightRadiance = pl.color.rgb * nl * falloff * directLightOcclusion;
        color += (diffuseColor * (1.0 / PI) * (1.0 - F) + spec) * lightRadiance;
        specDirect += spec * lightRadiance;
    }

    // --- Image based ambient lighting, occluded by baked sky visibility and screen-space AO ---
    vec3 irradiance = shIrradiance(worldToEnv(N)) * frame.envRotation.w;
    // Light bounced off the sunlit surroundings keeps occluded areas from going completely black.
    vec3 bounce = frame.sunColor.rgb * 0.035 * (0.6 + 0.4 * skyVis);
    color += diffuseColor * (irradiance * diffuseOcclusion + bounce * ao) * (1.0 / PI);
    if (translucency > 0.0) {
        // Skylight diffusing through from the far side of the leaf / fabric.
        vec3 backIrradiance = shIrradiance(worldToEnv(-N)) * frame.envRotation.w;
        color += transTint * backIrradiance * (0.35 * diffuseOcclusion);
    }

    vec3 R = reflect(-V, N);
    float specOcclusion = clamp(pow(NdotV + diffuseOcclusion, exp2(-16.0 * roughness - 1.0)) - 1.0 + diffuseOcclusion,
                                0.0, 1.0);
    if (ALPHA_MODE == 2u) {
        bool cameraIndoors = frame.exposureParams2.z > 0.5;
        specOcclusion = cameraIndoors ? min(specOcclusion, 0.12) : clamp(max(specOcclusion, 0.25), 0.0, 0.55);
    }
    vec3 envSpec = sampleEnvSpecular(R, roughness) * frame.envRotation.w;
    color += envSpec * envBRDFApprox(f0, roughness, NdotV) * specOcclusion;

    // --- Emissive ---
    if ((m.flags & 2u) != 0u) {
        vec3 e = texture(textures[nonuniformEXT(m.emissiveTexture)], inUV, 0.5).rgb;
        color += e * m.emissiveFactor * frame.postParams.w;
    }

    // --- Aerial perspective / height fog ---
    float dist = length(frame.cameraPos.xyz - inWorldPos);
    float heightTerm = exp(-max(inWorldPos.y - frame.fogParams.z, 0.0) * frame.fogParams.y);
    float fog = min(1.0 - exp(-dist * frame.fogParams.x * heightTerm), frame.fogParams.w);

    if (ALPHA_MODE == 2u) {
        // Glass: premultiplied alpha. Physical dielectric Fresnel reflections are added on top of
        // the transmitted background without skybox over-brightening or normal warping.
        float alpha = clamp(base.a, 0.0, 1.0);
        vec3 F_env = envBRDFApprox(vec3(0.04), roughness, NdotV);
        vec3 envRefl = envSpec * F_env * specOcclusion;
        vec3 clampedSpecDirect = specDirect / (1.0 + luminance(specDirect) * 0.25);
        if (!gl_FrontFacing) {
            envRefl *= 0.35;
            clampedSpecDirect *= 0.35;
        }
        vec3 reflection = envRefl + clampedSpecDirect * 0.45;
        vec3 diffOnly = max(color - specDirect - envSpec * envBRDFApprox(f0, roughness, NdotV) * specOcclusion, vec3(0.0));
        vec3 body = diffOnly * alpha + reflection;
        body = mix(body, frame.fogColor.rgb * alpha, fog);
        float fresnelAlpha = clamp(alpha + luminance(F_env), 0.0, 1.0);
        outColor = vec4(body, fresnelAlpha);
    } else {
        color = mix(color, frame.fogColor.rgb, fog);
        if (frame.aoParams.w > 0.5) color = vec3(ao * 0.5);  // `debug.bistro.aodebug`: screen-space AO only
        outColor = vec4(color, ALPHA_MODE == 1u ? base.a : 1.0);
    }
}
