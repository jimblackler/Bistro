#version 460
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

// Physically based shading for the Bistro materials (GGX metal/rough, ORM packed textures).
// ALPHA_MODE: 0 = opaque, 1 = alpha tested, 2 = alpha blended (glass).
// ALPHA_TO_COVERAGE: sharpen alpha for MSAA alpha-to-coverage instead of a hard alpha test.
layout(constant_id = 0) const uint ALPHA_MODE = 0u;
layout(constant_id = 1) const bool ALPHA_TO_COVERAGE = false;

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec4 inTangent;
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
vec2 envBRDFScaleBias(float roughness, float NdotV) {
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NdotV)) * r.x + r.y;
    return clamp(vec2(-1.04, 1.04) * a004 + r.zw, vec2(0.0), vec2(1.0));
}

vec3 envBRDFApprox(vec3 f0, float roughness, float NdotV) {
    vec2 ab = envBRDFScaleBias(roughness, NdotV);
    return f0 * ab.x + ab.y;
}

float sampleShadow(vec3 worldPos, vec3 geomNormal, float NdotL) {
    float slope = clamp(sqrt(1.0 - NdotL * NdotL) / max(NdotL, 0.12), 0.0, 4.0);
    vec3 p = worldPos + geomNormal * frame.shadowParams.y * (1.0 + slope);
    vec4 sc = frame.shadowViewProj * vec4(p, 1.0);
    vec2 uv = sc.xy * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) || sc.z <= 0.0 || sc.z >= 1.0) return 1.0;
    vec2 border = smoothstep(vec2(0.0), vec2(0.08), uv) * (vec2(1.0) - smoothstep(vec2(0.92), vec2(1.0), uv));
    float zFade = smoothstep(0.0, 0.03, sc.z) * (1.0 - smoothstep(0.97, 1.0, sc.z));
    float borderWeight = border.x * border.y * zFade;
    if (borderWeight <= 0.001) return 1.0;
    float ref = sc.z - frame.shadowParams.z;
    if ((frame.lightCount.y & 8u) == 0u) {
        float tap = texture(shadowMap, vec3(uv, ref));
        return mix(1.0, tap, borderWeight);
    }
    float texel = frame.shadowParams.x;
    float r = frame.shadowParams.w;
    // 8-tap rotated Poisson disc PCF on top of hardware bilinear comparison; spatial interleaved
    // gradient noise avoids temporal shadow-edge boiling when TAA clips history during motion.
    const vec2 poisson[8] = vec2[](
        vec2(-0.326, -0.406), vec2(-0.840, -0.074), vec2(-0.696, 0.457), vec2(-0.203, 0.621),
        vec2(0.962, -0.195), vec2(0.473, -0.480), vec2(0.519, 0.767), vec2(0.185, -0.893));
    float angle = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715)))) * 6.2831853;
    float ca = cos(angle), sa = sin(angle);
    mat2 rot = mat2(ca, sa, -sa, ca);
    float sum = 0.0;
    for (int i = 0; i < 8; ++i) {
        vec2 o = rot * poisson[i] * r * texel;
        sum += texture(shadowMap, vec3(uv + o, ref));
    }
    return mix(1.0, sum * 0.125, borderWeight);
}

// 8-step Screen-Space Contact Shadow (SSCS) raymarch against the full-resolution depth buffer
// (hizTexture mip 0). Fills in millimeter-to-decimeter contact shadows (table cutlery, wine glasses,
// cobble crevices, foliage roots, and stone carvings) inside the shadow map's normal-bias window.
float sampleContactShadow(vec3 worldPos, vec3 geomN, vec3 lightDir, float geomNdotL, float dist) {
    if ((frame.lightCount.y & 2u) == 0u || frame.occlusionParams.z < 0.5 || dist > 32.0 || geomNdotL <= 0.04) return 1.0;
    float distFade = (1.0 - smoothstep(20.0, 32.0, dist)) * smoothstep(0.04, 0.12, geomNdotL);
    float grazingScale = clamp(1.0 / max(geomNdotL, 0.18), 1.0, 3.5);

    vec3 p0 = worldPos + geomN * max(0.0025, dist * 0.0006) * grazingScale;
    float rayLen = clamp(0.10 + dist * 0.009, 0.12, 0.34);

    vec4 clip0 = frame.prevViewProj * vec4(p0, 1.0);
    vec4 clip1 = frame.prevViewProj * vec4(p0 + lightDir * rayLen, 1.0);
    if (clip0.w <= 0.12 || clip1.w <= 0.12) return 1.0;

    vec3 ndc0 = clip0.xyz / clip0.w;
    vec3 ndc1 = clip1.xyz / clip1.w;
    vec2 uv0 = ndc0.xy * 0.5 + 0.5;
    vec2 uv1 = ndc1.xy * 0.5 + 0.5;
    if (any(lessThan(uv0, vec2(0.002))) || any(greaterThan(uv0, vec2(0.998)))) return 1.0;

    vec2 dUV = uv1 - uv0;
    float dZ = ndc1.z - ndc0.z;
    float screenLenPx = length(dUV * frame.occlusionParams.xy);
    if (screenLenPx < 1.2) return 1.0;
    float lenFade = smoothstep(1.2, 3.2, screenLenPx);

    float jitter = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    const int kSteps = 8;
    float occlusion = 0.0;

    for (int i = 0; i < kSteps; ++i) {
        float s = (float(i) + 0.25 + jitter * 0.70) / float(kSteps);
        // Quadratic bias concentrates the first steps within a few millimeters of the contact seam.
        float t = mix(s, s * s, 0.42);
        vec2 uv = uv0 + dUV * t;
        if (any(lessThan(uv, vec2(0.001))) || any(greaterThan(uv, vec2(0.999)))) break;

        float rayZ = ndc0.z + dZ * t;
        if (rayZ <= 1e-5) break;

        float sceneZ = textureLod(hizTexture, uv, 0.0).r;
        if (sceneZ > rayZ) {
            // Convert reversed-Z difference into linear view-space depth thickness in meters:
            // d_ray - d_scene = d_ray * (sceneZ - rayZ) / sceneZ.
            float deltaMeters = clip0.w * (sceneZ - rayZ) / max(sceneZ, 1e-5);
            float minSlop = max(0.002 + dist * 0.0005, 0.003 * t) * grazingScale;
            float maxThick = rayLen * (0.28 + 0.65 * t);
            if (deltaMeters > minSlop && deltaMeters < maxThick) {
                float stepFalloff = 1.0 - smoothstep(0.15, 1.0, t) * 0.65;
                float thickFalloff = 1.0 - smoothstep(maxThick * 0.55, maxThick, deltaMeters);
                occlusion = max(occlusion, stepFalloff * thickFalloff);
            }
        }
    }

    vec2 border = smoothstep(vec2(0.01), vec2(0.06), uv0) * (vec2(1.0) - smoothstep(vec2(0.94), vec2(0.99), uv0));
    float edgeFade = border.x * border.y * lenFade;
    return 1.0 - clamp(occlusion * distFade * edgeFade * 0.88, 0.0, 0.88);
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

// Screen-space reflection raymarcher against the previous frame's Hi-Z depth pyramid (hizTexture)
// and TAA-resolved HDR colour buffer (prevHdrTexture).
// Returns vec4(reflectedColor, confidence) where confidence in [0, 1] blends SSR over the IBL fallback.
vec4 traceSSR(vec3 worldPos, vec3 geomN, vec3 R, vec3 V, float dist, float roughness, float maxRelThickness, bool isWater) {
    if ((frame.lightCount.y & 1u) == 0u || frame.taaParams.w < 0.5 || frame.occlusionParams.z < 1.5 || roughness > 0.36) return vec4(0.0);
    float distFade = isWater ? (1.0 - smoothstep(55.0, 80.0, dist)) : (1.0 - smoothstep(16.0, 28.0, dist));
    if (distFade <= 0.001) return vec4(0.0);
    float retroFade = 1.0 - smoothstep(0.20, 0.62, dot(R, V));
    if (retroFade <= 0.001) return vec4(0.0);

    vec3 p0 = worldPos + geomN * max(0.025, dist * 0.003);
    vec4 clip0 = frame.prevViewProj * vec4(p0, 1.0);
    if (clip0.w <= 0.15) return vec4(0.0);
    vec3 ndc0 = clip0.xyz / clip0.w;
    vec2 uv0 = ndc0.xy * 0.5 + 0.5;
    float z0 = ndc0.z;  // reversed-Z: z = zNear / viewZ
    if (any(lessThan(uv0, vec2(0.002))) || any(greaterThan(uv0, vec2(0.998))) || z0 <= 1e-5) return vec4(0.0);

    vec4 clipDir = frame.prevViewProj * vec4(R, 0.0);
    float maxRayLen = isWater ? min(max(dist * 1.6, 18.0), 75.0) : min(max(dist * 1.1, 8.0), 22.0);
    if (clip0.w + clipDir.w * maxRayLen < 0.25) {
        if (clipDir.w >= -1e-4) return vec4(0.0);
        maxRayLen = (0.25 - clip0.w) / clipDir.w;
    }
    if (maxRayLen <= 0.08) return vec4(0.0);

    vec4 clip1 = clip0 + clipDir * maxRayLen;
    vec3 ndc1 = clip1.xyz / clip1.w;
    vec2 uv1 = ndc1.xy * 0.5 + 0.5;
    float z1 = ndc1.z;

    vec2 dUV = uv1 - uv0;
    float dZ = z1 - z0;
    float tClip = 1.0;
    if (dUV.x > 1e-5) tClip = min(tClip, (0.998 - uv0.x) / dUV.x);
    else if (dUV.x < -1e-5) tClip = min(tClip, (0.002 - uv0.x) / dUV.x);
    if (dUV.y > 1e-5) tClip = min(tClip, (0.998 - uv0.y) / dUV.y);
    else if (dUV.y < -1e-5) tClip = min(tClip, (0.002 - uv0.y) / dUV.y);
    tClip = clamp(tClip, 0.0, 1.0);
    dUV *= tClip;
    dZ *= tClip;

    float screenLenPx = length(dUV * frame.occlusionParams.xy);
    if (screenLenPx < 2.5) return vec4(0.0);
    float lenFade = smoothstep(2.5, 6.0, screenLenPx);

    // Cap relative thickness on non-water surfaces so mid-distance walls don't accept 3-5m thick occluders.
    float effMaxRelThick = isWater ? maxRelThickness : min(maxRelThickness, 1.4 / max(dist, 4.0));

    float jitter = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    const int kSteps = 16;
    float sPrev = max(1.5 / screenLenPx, 0.004);
    float hitS = -1.0;
    float hitSceneZ = 0.0;
    float hitRelThick = 0.0;

    for (int i = 1; i <= kSteps; ++i) {
        float fi = (float(i) - 0.65 + jitter * 0.65) / float(kSteps);
        // Quadratic progression concentrates small steps near the surface contact and reaches far across the screen.
        float s = clamp(mix(fi, fi * fi, 0.45), sPrev + 0.002, 1.0);
        vec2 uv = uv0 + dUV * s;
        float rayZ = z0 + dZ * s;
        if (rayZ <= 1e-5) break;

        // Coarse Hi-Z test at mip 1 first; if the ray is behind the coarse minimum depth, test mip 0.
        float coarseZ = textureLod(hizTexture, uv, 1.0).r;
        if (rayZ < coarseZ) {
            float fineZ = textureLod(hizTexture, uv, 0.0).r;
            if (fineZ > 1e-5 && rayZ < fineZ) {
                // 3-step binary search between sPrev (in front) and s (behind) to pinpoint the surface crossing.
                float a = sPrev;
                float b = s;
                float bestZ = fineZ;
                for (int r = 0; r < 3; ++r) {
                    float mid = 0.5 * (a + b);
                    vec2 mUV = uv0 + dUV * mid;
                    float mRayZ = z0 + dZ * mid;
                    float mSceneZ = textureLod(hizTexture, mUV, 0.0).r;
                    if (mSceneZ > 1e-5 && mRayZ < mSceneZ) {
                        b = mid;
                        bestZ = mSceneZ;
                    } else {
                        a = mid;
                    }
                }
                float bRayZ = z0 + dZ * b;
                // In reversed-Z (z = zNear / depth), (sceneZ - rayZ) / sceneZ equals (depthRay - depthScene) / depthRay.
                float relThick = (bestZ - bRayZ) / max(bestZ, 1e-5);
                if (relThick >= -0.003 && relThick < effMaxRelThick) {
                    hitS = b;
                    hitSceneZ = bestZ;
                    hitRelThick = max(relThick, 0.0);
                    break;
                }
            }
        }
        sPrev = s;
    }

    if (hitS < 0.0 || hitSceneZ <= 1e-5) return vec4(0.0);

    vec2 hitUV = uv0 + dUV * hitS;
    vec2 border = smoothstep(vec2(0.01), vec2(0.11), hitUV) * (vec2(1.0) - smoothstep(vec2(0.89), vec2(0.99), hitUV));
    float edgeFade = border.x * border.y * lenFade * distFade;
    float thickFade = 1.0 - smoothstep(effMaxRelThick * 0.35, effMaxRelThick, hitRelThick);
    float rayFade = 1.0 - smoothstep(0.82, 1.0, hitS);
    float roughFade = 1.0 - smoothstep(0.10, 0.36, roughness);
    float confidence = clamp(edgeFade * thickFade * rayFade * roughFade * retroFade, 0.0, 1.0);
    if (confidence <= 0.005) return vec4(0.0);

    vec3 ssrColor = textureLod(prevHdrTexture, hitUV, 0.0).rgb;
    // Softly compress extreme specular peaks so secondary reflections stay temporally stable.
    ssrColor /= 1.0 + luminance(ssrColor) * 0.15;
    return vec4(ssrColor, confidence);
}

// Multi-layered procedural flowing water normal for the Jungle Ruins river.
// Evaluates world-space swells, cross-ripples, and distance-filtered capillary wavelets.
vec3 waterWaveNormal(vec2 xz, float time, float dist, out float outWaveRoughness) {
    // Subtle spatial domain warp so swells curve naturally along the riverbed.
    vec2 warp = vec2(sin(xz.x * 0.22 + xz.y * 0.17 + time * 0.35),
                     cos(xz.x * 0.19 - xz.y * 0.25 - time * 0.28)) * 0.45;
    vec2 p = xz + warp;

    // Layer 1: primary downstream swells (wavelengths ~2.5m - 4.0m).
    float p1 = dot(p, vec2(1.15, 0.85)) - time * 1.35;
    float p2 = dot(p, vec2(0.90, -1.25)) - time * 1.05;
    vec2 grad1 = vec2(1.15, 0.85) * cos(p1) * 0.028 + vec2(0.90, -1.25) * cos(p2) * 0.022;

    // Layer 2: medium crossing river ripples (wavelengths ~0.7m - 1.1m).
    float medAtten = exp(-dist * 0.012);
    float p3 = dot(p, vec2(3.1, 2.2)) - time * 2.4;
    float p4 = dot(p, vec2(-2.4, 3.5)) + time * 1.9;
    float p5 = dot(p, vec2(4.2, -1.6)) - time * 2.8;
    vec2 grad2 = (vec2(3.1, 2.2) * cos(p3) * 0.018 +
                  vec2(-2.4, 3.5) * cos(p4) * 0.015 +
                  vec2(4.2, -1.6) * cos(p5) * 0.012) * medAtten;

    // Layer 3: fine capillary wavelets (wavelengths ~0.22m - 0.35m), filtered out with distance into roughness.
    float fineAtten = exp(-dist * 0.038);
    float p6 = dot(xz, vec2(9.5, 6.2)) - time * 4.2;
    float p7 = dot(xz, vec2(-7.8, 10.4)) + time * 3.7;
    vec2 grad3 = (vec2(9.5, 6.2) * cos(p6) * 0.008 +
                  vec2(-7.8, 10.4) * cos(p7) * 0.007) * fineAtten;

    // Convert filtered capillary slope energy into GGX roughness (LEADR/Toksvig filtering).
    outWaveRoughness = mix(0.09, 0.025, fineAtten * medAtten);
    vec2 g = grad1 + grad2 + grad3;
    return normalize(vec3(-g.x, 1.0, -g.y));
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
    vec4 nt = texture(textures[nonuniformEXT(m.normalTexture)], inUV);
    vec2 nxy = vec2(nt.r, nt.a) * 2.0 - 1.0;
    vec2 dNxydx = dFdx(nxy);
    vec2 dNxydy = dFdy(nxy);

    bool alphaDiscarded = false;
    if (ALPHA_MODE == 1u) {
        if (ALPHA_TO_COVERAGE) {
            // Sharpen alpha around the cutoff so alpha-to-coverage produces crisp, anti-aliased edges.
            base.a = (base.a - m.alphaCutoff) / max(fwidth(base.a), 1e-4) + 0.5;
            alphaDiscarded = (base.a <= 0.0);
        } else {
            alphaDiscarded = (base.a < m.alphaCutoff);
        }
    }

    // Reconstruct the world position from the depth (accounting for unjittered ALPHA_MODE == 1u).
    vec2 ndcXY = gl_FragCoord.xy / frame.renderParams.zw * 2.0 - 1.0;
    if (ALPHA_MODE == 1u) ndcXY += frame.taaParams.xy;
    vec4 wp = frame.invViewProj * vec4(ndcXY, gl_FragCoord.z, 1.0);
    vec3 inWorldPos = wp.xyz / wp.w;
    float dist = length(frame.cameraPos.xyz - inWorldPos);

    vec3 geomN = normalize(inNormal);
    if (!gl_FrontFacing) geomN = -geomN;
    vec3 Traw = inTangent.xyz - geomN * dot(geomN, inTangent.xyz);
    vec3 T = Traw * inversesqrt(max(dot(Traw, Traw), 1e-8));
    vec3 B = cross(geomN, T) * inTangent.w;

    bool isFrostedGlass = (m.flags & 32u) != 0u;
    bool isWater = (m.flags & 64u) != 0u;

    float uvFootprint = sqrt(max(dot(dUVdx, dUVdx) + dot(dUVdy, dUVdy), 0.0));
    if (ALPHA_MODE == 1u) {
        // Fade high-frequency normal perturbation on distant alpha-tested leaves to prevent sub-pixel shimmer.
        nxy *= 1.0 - smoothstep(0.015, 0.08, uvFootprint) * 0.75;
    } else if (ALPHA_MODE == 0u && !isWater) {
        // Attenuate high-frequency normal-map slopes on distant cobblestones, brick, and architectural trim.
        nxy *= 1.0 - smoothstep(0.015, 0.10, uvFootprint) * 0.65;
    }
    float nz = sqrt(clamp(1.0 - dot(nxy, nxy), 0.0, 1.0));
    vec3 N = normalize(T * nxy.x + B * nxy.y + geomN * nz);

    float waterWaveRough = 0.04;
    if (isWater) {
        N = waterWaveNormal(inWorldPos.xz, frame.cameraPos.w, dist, waterWaveRough);
    }

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
    float ao = (ALPHA_MODE == 2u || isWater) ? 1.0 : sampleSSAO(inWorldPos);
    if (ALPHA_MODE == 1u) {
        // Half-res depth-reconstructed SSAO fluctuates across porous alpha-cutout leaves; rely primarily on baked skyVis.
        ao = mix(ao, 1.0, 0.65);
    }
    float skyVis = mix(inSkyVisibility, 1.0, smoothstep(0.25, 0.85, skyVisGrad));
    float diffuseOcclusion = ao * skyVis;
    float directLightOcclusion = mix(1.0, diffuseOcclusion, 0.65);
    float roughness = isWater ? waterWaveRough : clamp(orm.g * m.roughnessFactor, 0.04, 1.0);
    float geomVar = dot(dGeomNdx, dGeomNdx) + dot(dGeomNdy, dGeomNdy);
    if (ALPHA_MODE == 1u) {
        roughness = max(roughness, mix(0.04, 0.55, smoothstep(0.015, 0.08, uvFootprint)));
    } else if (ALPHA_MODE == 0u && !isWater) {
        // Prevent sub-pixel specular fireflies on distant architectural trim, wrought-iron railings, and cobblestones.
        float distRoughFloor = mix(0.04, 0.36, smoothstep(0.018, 0.11, uvFootprint));
        float edgeRoughFloor = mix(0.04, 0.40, smoothstep(0.004, 0.05, geomVar));
        roughness = max(roughness, max(distRoughFloor, edgeRoughFloor));
    }
    float metallic = isWater ? 0.0 : clamp(orm.b * m.metallicFactor, 0.0, 1.0);
    // Geometric and normal-map specular anti-aliasing (Kaplanyan 2016): broaden GGX roughness by the
    // screen-space normal variance so sub-pixel grilles, bevels, and normal grooves don't alias.
    float nxyVar = dot(dNxydx, dNxydx) + dot(dNxydy, dNxydy) + dot(nxy, nxy) * min(uvFootprint * 6.0, 0.20);
    float normalVar = isWater ? 0.0 : min(geomVar * 1.5 + nxyVar * 0.75 + louverNormalVar, 0.30);
    float aOrig = roughness * roughness;
    float a = clamp(sqrt(aOrig * aOrig + normalVar * 0.5), 0.0009, 1.0);
    roughness = sqrt(a);

    vec3 V = normalize(frame.cameraPos.xyz - inWorldPos);
    float NdotV = clamp(dot(N, V), 1e-4, 1.0);
    vec3 diffuseColor = base.rgb * (1.0 - metallic) * louverOcc;
    if (isWater) {
        // Deep emerald-teal river absorption blended with warm shallow riverbed scattering at steep angles.
        vec3 deepWater = vec3(0.025, 0.095, 0.080);
        vec3 shallowBed = base.rgb * vec3(0.42, 0.62, 0.48);
        diffuseColor = mix(deepWater, shallowBed, pow(NdotV, 1.6));
    }
    vec3 f0 = isWater ? vec3(0.02) : (mix(vec3(0.04), base.rgb, metallic) * mix(0.20, 1.0, louverOcc));

    // Multi-scattering GGX energy compensation (Turquin 2019 / Fdez-Aguera 2019):
    // Restores the 35-50% energy lost by single-scattering Smith-GGX on rough metals and dielectrics.
    vec2 brdfAB = envBRDFScaleBias(roughness, NdotV);
    float Ess = clamp(brdfAB.x + brdfAB.y, 0.18, 1.0);
    float Ems = 1.0 - Ess;
    vec3 multiScatterDirect = vec3(1.0) + f0 * ((1.0 - Ess) / Ess);
    vec3 Fss = f0 * brdfAB.x + brdfAB.y;
    vec3 Favg = f0 + (vec3(1.0) - f0) * (1.0 / 21.0);
    vec3 Fms = Fss * Favg * Ems / max(vec3(1.0) - Favg * Ems, vec3(1e-4));
    vec3 iblSpecBRDF = Fss + Fms * 0.55;

    // --- Thin-surface translucency setup (leaves, grass, curtains, awnings) ---
    // Light passing through a thin leaf or fabric is filtered by the material on the way in and out, so
    // the transmitted colour is a more saturated version of the albedo.
    float translucency = (m.flags & 8u) != 0u ? frame.exposureParams2.w : 0.0;
    vec3 transTint = vec3(0.0);
    if (translucency > 0.0) {
        float maxC = max(diffuseColor.r, max(diffuseColor.g, max(diffuseColor.b, 1e-3)));
        transTint = diffuseColor * mix(vec3(1.0), diffuseColor / maxC, 0.6) * (translucency / PI);
    }

    // --- Sun (directional light with camera-fitted shadow map + screen-space contact shadows) ---
    vec3 L = frame.sunDirection.xyz;
    float NdotL = clamp(dot(N, L), 0.0, 1.0);
    float geomNdotL = dot(geomN, L);
    vec3 color = vec3(0.0);
    vec3 specDirect = vec3(0.0);
    if (NdotL > 0.0 && geomNdotL > -0.05) {
        float shadow = sampleShadow(inWorldPos, geomN, clamp(geomNdotL, 0.0, 1.0));
        if (shadow > 0.01 && geomNdotL > 0.04 && !isWater) {
            float contact = sampleContactShadow(inWorldPos, geomN, L, geomNdotL, dist);
            if (ALPHA_MODE == 1u) contact = mix(contact, 1.0, 0.55);
            shadow *= contact;
        }
        vec3 H = normalize(V + L);
        float NdotH = clamp(dot(N, H), 0.0, 1.0);
        float VdotH = clamp(dot(V, H), 0.0, 1.0);
        vec3 F = F_Schlick(f0, VdotH);
        vec3 spec = D_GGX(NdotH, a) * V_SmithGGXCorrelated(NdotV, NdotL, a) * F * multiScatterDirect;
        if (isWater) {
            // Soften extreme single-pixel sun glints on water while keeping sparkling highlights.
            spec /= 1.0 + luminance(spec) * 0.08;
        }
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
    vec3 pointColor = vec3(0.0);
    vec3 pointSpec = vec3(0.0);
    vec3 dominantLp = vec3(0.0);
    float dominantWeight = 0.0;
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
        vec3 spec = D_GGX(clamp(dot(N, H), 0.0, 1.0), a) * V_SmithGGXCorrelated(NdotV, nl, a) * F * multiScatterDirect;
        vec3 lightRadiance = pl.color.rgb * nl * falloff * directLightOcclusion;
        float w = luminance(lightRadiance);
        if (w > dominantWeight) {
            dominantWeight = w;
            dominantLp = Lp;
        }
        pointColor += (diffuseColor * (1.0 / PI) * (1.0 - F) + spec) * lightRadiance;
        pointSpec += spec * lightRadiance;
    }
    if (dominantWeight > 1e-4 && ALPHA_MODE == 0u && !isWater && !isFrostedGlass) {
        float ptGeomNL = dot(geomN, dominantLp);
        if (ptGeomNL > 0.05) {
            float ptContact = sampleContactShadow(inWorldPos, geomN, dominantLp, ptGeomNL, dist);
            pointColor *= mix(1.0, ptContact, 0.80);
            pointSpec *= mix(1.0, ptContact, 0.80);
        }
    }
    color += pointColor;
    specDirect += pointSpec;

    // --- Image based ambient lighting, occluded by baked sky visibility and screen-space AO ---
    vec3 irradiance = shIrradiance(worldToEnv(N)) * frame.envRotation.w;
    // Light bounced off the sunlit surroundings keeps occluded areas from going completely black.
    vec3 bounce = frame.sunColor.rgb * 0.035 * (0.6 + 0.4 * skyVis);
    float effDiffuseOcc = isFrostedGlass ? max(diffuseOcclusion, 0.10) : (isWater ? max(skyVis, 0.45) : diffuseOcclusion);
    vec3 waterFresnel = isWater ? F_Schlick(vec3(0.02), NdotV) : vec3(0.0);
    vec3 ambientIrr = irradiance * effDiffuseOcc + bounce * ao;
    vec3 kd = isWater ? (vec3(1.0) - waterFresnel) : clamp(vec3(1.0) - Fss * 0.5, vec3(0.0), vec3(1.0));
    color += (diffuseColor * kd + Fms * 0.45) * ambientIrr * (1.0 / PI);
    if (isFrostedGlass) {
        // Subtle diffused interior/alcove backlight so recessed frosted partitions aren't coal-black.
        vec3 backIrr = shIrradiance(worldToEnv(-geomN)) * frame.envRotation.w;
        color += diffuseColor * (backIrr * max(skyVis, 0.06) + bounce * 0.35) * (0.25 / PI);
    } else if (translucency > 0.0) {
        // Skylight diffusing through from the far side of the leaf / fabric.
        vec3 backIrradiance = shIrradiance(worldToEnv(-N)) * frame.envRotation.w;
        color += transTint * backIrradiance * (0.35 * diffuseOcclusion);
    }

    vec3 R = reflect(-V, N);
    float specOcclusion = isWater
        ? clamp(0.35 + 0.65 * skyVis, 0.0, 1.0)
        : clamp(pow(NdotV + diffuseOcclusion, exp2(-16.0 * roughness - 1.0)) - 1.0 + diffuseOcclusion, 0.0, 1.0);
    vec3 envSpec = sampleEnvSpecular(R, roughness) * frame.envRotation.w;
    if (ALPHA_MODE == 0u && !isFrostedGlass && (isWater || roughness < 0.36)) {
        float maxThick = isWater ? 0.22 : 0.10;
        vec4 ssr = traceSSR(inWorldPos, geomN, R, V, dist, roughness, maxThick, isWater);
        if (isWater) {
            // Compress overly bright skybox peaks on water and blend in Hi-Z SSR of riverbank trees & pyramid.
            vec3 waterEnv = envSpec / (1.0 + luminance(envSpec) * 0.25) * specOcclusion;
            vec3 waterRefl = mix(waterEnv, ssr.rgb, ssr.a * 0.90);
            color += waterRefl * max(waterFresnel, iblSpecBRDF);
        } else {
            vec3 reflSpec = mix(envSpec * specOcclusion, ssr.rgb, ssr.a * 0.75);
            color += reflSpec * iblSpecBRDF;
        }
    } else {
        color += envSpec * iblSpecBRDF * specOcclusion;
    }

    // --- Emissive ---
    if ((m.flags & 2u) != 0u) {
        vec3 e = texture(textures[nonuniformEXT(m.emissiveTexture)], inUV, 0.5).rgb;
        color += e * m.emissiveFactor * frame.postParams.w;
    }

    // --- Aerial perspective, anisotropic Mie fog & volumetric sun god-rays (crepuscular rays) ---
    float fogTransmittance = 1.0;
    vec3 sunShafts = vec3(0.0);
    vec3 fogInscatter = evaluateVolumetricFog(inWorldPos, -V, dist, gl_FragCoord.xy, fogTransmittance, sunShafts);

    if (ALPHA_MODE == 2u || isFrostedGlass) {
        bool cameraIndoors = frame.exposureParams2.z > 0.5;
        bool isBottle = !isFrostedGlass && (m.baseColorFactor.b < m.baseColorFactor.r);
        bool isGlassware = !isFrostedGlass && (isBottle || (length(dGeomNdx) + length(dGeomNdy) > 0.012));

        // Flat architectural windows and the smooth outer pane of frosted glass reflect along the
        // unperturbed geometric normal geomN so reflections never warp like a funhouse mirror.
        vec3 N_refl = isGlassware ? N : geomN;
        vec3 R_glass = reflect(-V, N_refl);
        float reflNdotV = clamp(dot(N_refl, V), 1e-4, 1.0);

        float glassRough = isGlassware ? clamp(roughness, 0.04, 0.10) : (isFrostedGlass ? 0.12 : max(roughness, 0.10));
        vec3 rawEnv = sampleEnvSpecular(R_glass, glassRough) * frame.envRotation.w;
        // Compress bright skybox peaks so reflections never blow out to pure white.
        vec3 tonedEnv = rawEnv / (1.0 + luminance(rawEnv) * 0.65);

        // Scale outdoor environment reflections by baked sky visibility so shaded awnings and indoor
        // glassware don't glow with unoccluded daylight sky irradiance.
        vec3 diffOnly = max(color - specDirect - envSpec * iblSpecBRDF * specOcclusion, vec3(0.0));
        float outdoorVis = cameraIndoors ? (skyVis * 0.30) : clamp(skyVis * 2.4 + 0.18, 0.18, 0.90);
        vec3 indoorRoomEnv = bounce * vec3(1.35, 1.12, 0.88) + diffOnly * 1.4;
        vec3 glassEnv = tonedEnv * outdoorVis + indoorRoomEnv * (1.0 - clamp(skyVis * 2.5, 0.0, 1.0));

        // Blend Hi-Z Screen-Space Reflections of street facades, cobblestones, and cafe furniture into glass.
        vec4 ssrGlass = traceSSR(inWorldPos, geomN, R_glass, V, dist, glassRough, 0.10, false);
        vec3 tonedSSR = ssrGlass.rgb / (1.0 + luminance(ssrGlass.rgb) * 0.35);
        glassEnv = mix(glassEnv, tonedSSR, ssrGlass.a * (cameraIndoors ? 0.45 : 0.65));

        if (isFrostedGlass) {
            // Smooth outer plate-glass clearcoat reflection over the frosted glass body.
            vec3 F_clearcoat = mix(vec3(0.05), vec3(0.28), pow(1.0 - reflNdotV, 2.8));
            if (cameraIndoors) F_clearcoat *= 0.35;
            vec3 frostedColor = color + glassEnv * F_clearcoat;
            vec3 frostedShafts = sunShafts / (1.0 + luminance(frostedColor) * 0.25);
            frostedColor = frostedColor * fogTransmittance + fogInscatter + frostedShafts;
            outColor = vec4(frostedColor, 1.0);
        } else {
            float alpha = isGlassware
                ? (isBottle ? 0.26 : 0.04)
                : clamp(base.a, 0.04, 0.10);

            vec3 F_glass = isGlassware
                ? mix(vec3(0.07), vec3(0.62), pow(1.0 - reflNdotV, 2.3))
                : (cameraIndoors
                       ? mix(vec3(0.03), vec3(0.18), pow(1.0 - reflNdotV, 3.0))
                       : mix(vec3(0.08), vec3(0.44), pow(1.0 - reflNdotV, 2.5)));

            if (!gl_FrontFacing) {
                float backScale = isGlassware ? 0.50 : 0.55;
                F_glass *= backScale;
                specDirect *= backScale;
            }
            vec3 clampedSpecDirect = specDirect / (1.0 + luminance(specDirect) * 0.22);
            vec3 reflection = glassEnv * F_glass + clampedSpecDirect;
            vec3 bottleFloor = isBottle ? (diffuseColor * bounce * 0.45) : vec3(0.0);
            float tintWeight = isBottle ? 0.85 : (base.a > 0.35 ? 0.85 : 0.15);
            vec3 body = max(diffOnly, bottleFloor) * (alpha * tintWeight) * (1.0 - F_glass) + reflection;
            float fresnelAlpha = clamp(alpha * tintWeight + luminance(F_glass) * 0.85, 0.04, 0.78);
            body = body * fogTransmittance + (fogInscatter + sunShafts * 0.5) * fresnelAlpha;
            outColor = vec4(body, fresnelAlpha);
        }
    } else {
        vec3 shaftContrib = sunShafts / (1.0 + luminance(color) * 0.25);
        color = color * fogTransmittance + fogInscatter + shaftContrib;
        if (frame.aoParams.w > 0.5) color = vec3(ao * 0.5);  // `debug.bistro.aodebug`: screen-space AO only
        // Defer OpTerminateInvocation (`discard`) until after all non-uniform descriptor waterfall loops
        // and divergent lighting branches so PowerVR USC SIMD wave execution masks are never corrupted.
        if (ALPHA_MODE == 1u && alphaDiscarded) {
            discard;
        }
        outColor = vec4(color, ALPHA_MODE == 1u ? base.a : 1.0);
    }
}
