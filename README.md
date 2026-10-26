# Bistro & Multi-Scene Vulkan Fly-Through

A real-time C++20 / Vulkan 1.3 Android application that flies a camera through four complex 3D showcase environments—led by the [Amazon Lumberyard Bistro](https://developer.nvidia.com/orca/amazon-lumberyard-bistro) (~3.9 million triangles, exterior and interior)—rendered on mobile GPUs using a fully GPU-driven pipeline.

See [ATTRIBUTION.md](ATTRIBUTION.md) for asset credits and licences (CC BY 4.0 / CC0).

---

## Included Environments

All environments are defined declaratively in `environments/<id>/environment.json` and compiled offline into `app/src/main/assets/environments/<id>/`:

1. **[Amazon Lumberyard Bistro](https://developer.nvidia.com/orca/amazon-lumberyard-bistro) (`bistro`)** — *Parisian Street & Restaurant Interior*
   - Merges `BistroExterior.fbx` and `BistroInterior.fbx` from the [NVIDIA ORCA Amazon Lumberyard Bistro](https://developer.nvidia.com/orca/amazon-lumberyard-bistro) archive (~3.9M triangles, ~6.4k spatial clusters) with [`san_giuseppe_bridge_4k.hdr`](https://polyhaven.com/a/san_giuseppe_bridge) image-based lighting.
   - Alternates between a street-level **Exterior** tour along the boulevard (~90 s) and a close-up **Restaurant Interior** tour through the dining room, wine alcove, and zinc bar counter (~46 s).
   - **License**: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) — see the [NVIDIA ORCA Bistro page](https://developer.nvidia.com/orca/amazon-lumberyard-bistro) and bundled [`LICENSE.txt`](app/src/main/assets/environments/bistro/LICENSE.txt); HDR sky map licensed [CC0 (Poly Haven)](https://polyhaven.com/license).
2. **[Beeple Zero-Day](https://developer.nvidia.com/orca/beeple-zero-day) (`beeple`)** — *Measure Seven · Mechanical Music Room*
   - Renders `MEASURE_SEVEN_COLORED_LIGHTS.fbx` from the [NVIDIA ORCA Beeple Zero-Day](https://developer.nvidia.com/orca/beeple-zero-day) archive with procedural multi-lobe studio HDR lighting, coloured cyan/amber point lights, and emissive light panels.
   - **License**: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) — by Mike Winkelmann (Beeple) and Kate Anderson; see the [NVIDIA ORCA Beeple Zero-Day page](https://developer.nvidia.com/orca/beeple-zero-day) and bundled [`LICENSE.txt`](app/src/main/assets/environments/beeple/LICENSE.txt).
3. **[Intel Sponza](https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-processing-research/samples.html) (`sponza`)** — *Atrium · Curtains, Ivy & Cypress*
   - Combines [Intel GPU Research Samples'](https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-processing-research/samples.html) updated Sponza (`NewSponza_Main_glTF_003.gltf`, `NewSponza_Curtains_glTF.gltf`, `NewSponza_IvyGrowth_glTF.gltf`, and `NewSponza_CypressTree_glTF.gltf`) with 22 warm colonnade/gallery point lights and [`kloppenheim_05_4k.hdr`](https://polyhaven.com/a/kloppenheim_05) IBL across a two-tier atrium and upper gallery fly-through.
   - **License**: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) — commissioned by Frank Meinl and sponsored by Anton Kaplanyan (with the Sponza Addon Package Crew); see the [Intel GPU Research Samples library](https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-processing-research/samples.html) and bundled [`LICENSE.txt`](app/src/main/assets/environments/sponza/LICENSE.txt); HDR sky map licensed [CC0 (Poly Haven)](https://polyhaven.com/license).
4. **[Jungle Ruins](https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-processing-research/samples.html) (`jungle`)** — *Mayan Pyramid · Banyan & River Clearing*
   - Assembles [Intel GPU Research Samples'](https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-processing-research/samples.html) `JungleRuins` USD scene (via `tools/build_jungle_gltf.py`) with [`kloppenheim_06_puresky_8k.hdr`](https://polyhaven.com/a/kloppenheim_06_puresky) sky/IBL along a river clearing, banyan tree, and pyramid temple fly-through.
   - **License**: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) — created by Cristiano Siqueira, Charlene Teets, Sebastian Herholz, Anton Sochenov, and Anton Kaplanyan; see the [Intel GPU Research Samples library](https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-processing-research/samples.html) and bundled [`LICENSE.txt`](app/src/main/assets/environments/jungle/LICENSE.txt); HDR sky map licensed [CC0 (Poly Haven)](https://polyhaven.com/license).

---

## Rendering Features

- **Vulkan 1.3 Core**: Dynamic rendering (`vkCmdBeginRendering`), `VK_KHR_synchronization2`, `vkQueueSubmit2`, and a dedicated native render thread with asynchronous asset streaming and a procedural loading screen.
- **2-Phase GPU-Driven Culling (`cull.comp`, `hiz.comp`)**:
  - Scenes are partitioned offline into spatially coherent clusters ($\le 2048$ triangles, $\le 12\text{ m}$ extent, 16-bit cluster-local indices).
  - **Phase 1** frustum-, sub-pixel size-, and Hi-Z occlusion-culls clusters against the previous frame's depth pyramid, draws visible occluders via `vkCmdDrawIndexedIndirectCount`, and builds an **Early Hi-Z** mip pyramid from the current frame's depth buffer.
  - **Phase 2** re-tests clusters rejected by stale history against the current frame's Early Hi-Z pyramid and draws newly disoccluded geometry with zero frame-lag popping.
- **Bindless Materials & ASTC Textures**:
  - All textures reside in a single descriptor-indexed array (`binding = 4`); materials and cluster metadata are fetched from storage buffers via `gl_InstanceIndex`.
  - Offline sRGB-correct box mip chains with alpha-coverage preservation for foliage cutouts (`ASTC_6x6_SRGB`), reconstructed 3D normal maps encoded with `astcenc -normal` (`ASTC_6x6_UNORM`), and folded constant ORM/base-colour maps.
- **Physically Based Shading (`scene.frag`)**:
  - Metallic-roughness GGX direct and image-based lighting (L2 `SH9` diffuse irradiance + 6-mip GGX importance-sampled specular environment map + analytic sun disc).
  - **Multi-Scattering GGX Energy Compensation** (Turquin / Fdez-Aguera) so rough metals and dielectrics preserve energy across multiple microfacet bounces.
  - **Specular Geometric & Normal-Map Anti-Aliasing** (Toksvig / Kaplanyan) to suppress sub-pixel specular shimmering on detailed metalwork and cobblestones.
  - **Two-Sided Subsurface Foliage & Fabric Translucency** for sunlit leaves, curtains, and awnings, plus exterior street wetness glazing and interior window-pane parallax cubemap depth.
- **Loop-Fitted Directional Shadows & Screen-Space Contact Shadows (`shadow.vert`, `scene.frag`)**:
  - Orthographic sun shadow map (`2048²` / `4096²` / `8192²`, `D16_UNORM`) fitted across the entire closed camera loop (`CameraPath::loopLightBounds`) so shadow texels remain locked in world space without mid-flight refit snapping (falling back to camera-frustum fitting in free-camera mode).
  - 8-tap interleaved-gradient rotated Vogel disk PCF with receiver-plane depth bias (`dFdx`/`dFdy`) and normal offset bias.
  - 8-step screen-space depth ray-marched **Contact Shadows** (`sampleContactShadow`) capturing fine geometric crevices below shadow-map texel resolution.
- **Screen-Space Ambient Occlusion (`ao.comp`, `ao_blur.comp`) & Baked Sky Visibility**:
  - Half-resolution horizon-based GTAO compute pass sampling the Hi-Z depth pyramid, filtered by a depth/normal-aware 5×5 separable bilateral blur.
  - Combined via a micro/macro `min()` blend with offline BVH ray-traced per-vertex sky visibility (`tools/scene_converter/sky_visibility.h`), driving both diffuse AO and Lagarde/de Rousiers specular occlusion (`computeSpecularAO`).
- **Screen-Space Glossy Reflections (`traceSSR` in `scene.frag`)**:
  - Ray-marches the Hi-Z depth pyramid (`hizTex`) and reprojects previous-frame HDR radiance (`prevHdrTex`) on smooth surfaces (`roughness < 0.36`), fading smoothly across screen edges, thickness, roughness, and distance into pre-filtered IBL.
- **Volumetric Sun God-Rays & Anisotropic Mie Fog (`computeVolumetricFog` in `scene.frag`)**:
  - 8-step interleaved-gradient jittered ray march through the directional shadow map accumulating shadowed crepuscular rays (Henyey-Greenstein phase function, $g = 0.62$) and height-falloff atmospheric haze.
- **Temporal Anti-Aliasing (`taa.frag`) & HDR Post-Processing (`exposure.comp`, `bloom_*.frag`, `tonemap.frag`)**:
  - Halton(2,3) 8-phase sub-pixel jitter, Catmull-Rom 5-tap history sampling, YCoCg 3×3 variance clipping (`1.25σ`), perceptual luminance weighting, and contrast-adaptive sharpening (CAS).
  - Compute auto-exposure (`exposure.comp`), 5-mip dual-filter HDR bloom with 13-tap Karis firefly suppression (`bloom_down.frag` / `bloom_up.frag`), ACES filmic tone mapping (`tonemap.frag`), vignette, and Android swapchain pre-rotation (`preTransform`).
- **Vulkan Debug Markers (`VK_EXT_debug_utils`)**:
  - Every Vulkan object (buffers, `.tex` images/views, samplers, shader modules, pipelines, descriptor sets, command buffers, semaphores, fences) and render/compute pass region is labeled for inspection in RenderDoc and Android GPU Inspector.

---

## Interactive UI & Graphics Settings

- **Start Screen**: Paginated environment cards (`◀` / `▶`) with live 3D background rendering and a 5-second auto-start countdown.
- **In-Flight HUD**: Tap anywhere on the 3D view to toggle the HUD overlay, cycle environments with `◀` / `▶`, or return to the start screen via `☰ Menu`.
- **Graphics & Performance Panel (`⚙ Graphics`)**:
  - **Quality Presets**: One-tap *Low*, *Medium*, *High* (default), and *Ultra* profiles.
  - **Freeze / Resume Camera (`⏸ Freeze` / `▶ Resume`)**: Pause camera motion in place to compare visual fidelity and FPS across settings on the exact same frame.
  - **Render Scale (`50%` – `100%`)**: Dynamically scales internal HDR, depth, AO, TAA, and Hi-Z attachments while keeping presentation at native swapchain resolution.
  - **Directional Shadow Map (`2K` / `4K` / `8K`)**: Dynamically reallocates the sun shadow map (`2048²`, `4096²`, `8192²`).
  - **Feature Toggles**: Individually toggle *SSR Reflections*, *Contact Shadows*, *Volumetric Rays*, *Screen-Space AO*, *8-Tap Shadow PCF*, *Temporal AA*, and *Hi-Z Occlusion*.
  - **Scalable Sliders**: Adjust *Small-Cluster Culling* (`0.1 px` – `3.0 px`), *AO Intensity* (`0%` – `200%`), and *Bloom Intensity* (`0%` – `200%`).

---

## Building

### 1. Runtime Assets

The offline pipeline in `tools/build_assets.py` converts source archives under `source_assets/` into mobile runtime assets under `app/src/main/assets/environments/<id>/`:

```bash
brew install assimp
python3 -m venv .venv && .venv/bin/pip install numpy pillow
# astcenc: download from https://github.com/ARM-software/astc-encoder/releases into tools/bin/astcenc
.venv/bin/python tools/build_assets.py --download
```

Useful flags for `tools/build_assets.py`:
- `--download`: Automatically downloads and extracts `source_assets/Bistro_v5_2.zip` and `source_assets/ZeroDay_v1.zip` if not yet present.
- `--env <id>`: Rebuild a single environment (`bistro`, `beeple`, `sponza`, or `jungle`).
- `--max-size <N>`: Maximum texture dimension (defaults to `1024`).
- `--jobs <N>`: Number of parallel worker processes for texture analysis and ASTC encoding.
- `--skip-textures`: Rebuild `scene.bin`, `environment.bin`, and `manifest.json` without re-checking `.tex` files.
- `--force`: Force regeneration of all outputs even when cached outputs are newer than inputs.

### 2. Android Application

Ensure `JAVA_HOME` points to a modern JDK (such as Android Studio's bundled JBR 21), then build and install:

```bash
export JAVA_HOME=/Users/jimblackler/Library/Java/JavaVirtualMachines/jbr-21.0.9/Contents/Home
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

All GLSL shaders in `app/src/main/shaders/` are compiled automatically by CMake using the NDK's `glslc` (`--target-env=vulkan1.3 -O`) into `shaders/generated_shaders.h`.

Requires an Android device supporting Vulkan 1.3, LDR ASTC texture compression, and `drawIndirectCount`.

---

## Debugging & Inspection Tools

### ADB Intent Extras

Launch directly into a specific environment, camera pose, lighting override, or quality preset via `adb shell am start`:

```bash
# Jump to a specific environment and timestamp (in seconds), optionally freezing the camera:
adb shell am start -n com.android.demo.bistro/.MainActivity --es env bistro --ef startTime 42 --ez freeze true

# Place the camera at an arbitrary world-space eye and look-at target ("x,y,z,tx,ty,tz"):
adb shell am start -n com.android.demo.bistro/.MainActivity --es env sponza --es camera "6.8,1.75,-1.95,-1.5,3.2,0.5"

# Launch with a quality preset (low|medium|high|ultra) and open the Graphics & Performance panel:
adb shell am start -n com.android.demo.bistro/.MainActivity --es env bistro --es preset ultra --ez graphicsPanel true

# Override lighting / AO scales for look-dev:
adb shell am start -n com.android.demo.bistro/.MainActivity --es env beeple --ef sunAzimuth 0.5 --ef sunScale 1.2 --ef pointScale 1.0 --ef aoScale 1.0
```

### Offline Scripts (`tools/`)

- `.venv/bin/python tools/check_path.py [out.png]`: Validates camera splines against the 16-band 3D obstacle grid (`build_assets/<id>/obstacles.pgm`) and renders a top-down clearance map.
- `.venv/bin/python tools/path_map.py` & `tools/relax_path.py`: Visualizes and relaxes camera waypoints away from obstacle geometry.
- `.venv/bin/python tools/capture.py <out_dir> <t1> <t2> …`: Captures device screenshots at specified timestamps along the camera path.
- `.venv/bin/python tools/contact_sheet.py <out.png> <img1.png> …`: Montages captured screenshots into a labeled contact sheet.
- `tools/bench.sh <time> [scale] [msaa] [occlusion]`: Benchmarks GPU frame times on an attached device using `debug.bistro.*` system properties.
