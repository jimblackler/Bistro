#version 460
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

// Sky: HDR environment map plus an analytic sun disc, drawn behind all geometry (depth == far).
layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

void main() {
    vec2 ndc = inUV * 2.0 - 1.0;
    // Reversed infinite projection: depth 1 is the near plane, depth 0.5 lies at twice the near
    // distance. Both points are finite, so their difference gives the view ray.
    vec4 p0 = frame.invViewProj * vec4(ndc, 1.0, 1.0);
    vec4 p1 = frame.invViewProj * vec4(ndc, 0.5, 1.0);
    vec3 dir = normalize(p1.xyz / p1.w - p0.xyz / p0.w);

    vec3 color = textureLod(skyTexture, equirectUV(worldToEnv(dir)), 0.0).rgb * abs(frame.envRotation.z);

    // Sun disc with a soft limb and a gentle glow.
    float cosAngle = dot(dir, frame.sunDirection.xyz);
    float cosRadius = frame.sunColor.w;
    float disc = smoothstep(cosRadius - 0.00002, cosRadius + 0.00002, cosAngle);
    float solidAngle = 2.0 * PI * (1.0 - cosRadius);
    color += frame.sunColor.rgb / solidAngle * disc * 0.02;
    color += frame.sunColor.rgb * pow(max(cosAngle, 0.0), 800.0) * 0.6;

    // Haze toward the horizon to match the aerial perspective applied to geometry.
    float horizon = pow(1.0 - clamp(abs(dir.y), 0.0, 1.0), 6.0);
    color = mix(color, frame.fogColor.rgb, horizon * frame.fogParams.w * 0.6);
    if (frame.envRotation.z < 0.0) {
        // Debug (`debug.bistro.skydebug`): magenta sky to reveal holes in the geometry.
        float l = dot(color, vec3(0.2126, 0.7152, 0.0722));
        color = vec3(l * 1.5, 0.0, l * 1.5);
    }
    outColor = vec4(color, 1.0);
}
