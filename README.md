# Bistro Fly-Through

A C++ / Vulkan Android app that flies a camera endlessly through the
[Amazon Lumberyard Bistro](https://developer.nvidia.com/orca/amazon-lumberyard-bistro) scene
(~3.9 million triangles, exterior and interior), rendered in real time on a phone.

Based on the structure of the `VulkanVenture` project (Kotlin activity + native Vulkan renderer with
SPIR-V embedded at build time), rebuilt as a GPU-driven scene renderer.

See [ATTRIBUTION.md](ATTRIBUTION.md) for the asset credits and licence (CC BY 4.0).

## Rendering features

- **Vulkan 1.3**: dynamic rendering, synchronization2, `vkQueueSubmit2`.
- **GPU-driven culling**: the scene is split offline into ~6.4k spatial clusters. A compute shader
  frustum-, size- and **Hi-Z occlusion-culls** them every frame (using the previous frame's
  depth pyramid) and compacts survivors into indirect command lists drawn with
  `vkCmdDrawIndexedIndirectCount`.
- **Bindless materials**: all textures live in one descriptor-indexed array; materials and clusters
  are fetched from storage buffers using `gl_InstanceIndex`.
- **ASTC textures** with offline-generated, sRGB-correct, coverage-preserving mip chains.
- **PBR shading** (GGX metal/rough) with normal maps, a high-resolution **static sun shadow map**
  (8192², Poisson PCF), interior point lights and emissives.
- **Image-based lighting** from the bundled HDR sky: SH9 diffuse irradiance and a GGX pre-filtered
  specular map, plus the sky itself and an analytic sun disc.
- **Baked sky visibility**: per-vertex ambient occlusion ray-traced offline (BVH, 48 rays/vertex).
- **4× MSAA** resolved on-tile (transient attachments), **alpha-to-coverage** foliage.
- **HDR post**: auto-exposure (compute), bloom (dual filter), ACES tone mapping, grading, vignette.
- **Swapchain pre-rotation** to avoid compositor rotation on Android.
- **Two-stage camera fly-through**: alternates between a street-level exterior tour around the
  boulevard (~181 s) and a close-up interior tour inside the restaurant dining room, wine alcove,
  and zinc bar counter (~49 s), resetting temporal history cleanly on each cut.
- Dedicated native render thread; asset streaming with a loading screen.

## Building

### 1. Assets

The converted assets (`app/src/main/assets/bistro`) are generated from the original download:

```bash
brew install assimp
python3 -m venv .venv && .venv/bin/pip install numpy pillow
# astcenc: download from https://github.com/ARM-software/astc-encoder/releases into tools/bin/
.venv/bin/python tools/build_assets.py --download
```

This downloads `Bistro_v5_2.zip` (~850 MB) into `source_assets/`, builds the converter
(`tools/scene_converter`), bakes ambient occlusion, converts the textures to ASTC and pre-processes
the environment map. It takes a couple of minutes on a modern laptop.

### 2. App

```bash
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Requires a device with Vulkan 1.3, ASTC and `drawIndirectCount` (tested on a Pixel 10).

## Debugging aids

- Tap the screen to toggle the overlay.
- `adb shell am start -n com.android.demo.bistro/.MainActivity --ef startTime 42 --ez freeze true`
  holds the camera at a point on the path; `--es camera "x,y,z,tx,ty,tz"` places it freely.
- `tools/capture.py OUT t1 t2 …` screenshots several points along the path.
- `tools/bench.sh <time> [scale=0.6] [msaa=1] [occlusion=0]` measures GPU time with tuning overrides
  (via `debug.bistro.*` system properties).
