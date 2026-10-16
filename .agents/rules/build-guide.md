---
trigger: always_on
description: Instructions for building the Bistro project, converting source_assets into runtime assets, and running mandatory verification checks
---

# Agent Build & Asset Conversion Guide for Bistro

## Overview

Bistro is a C++20 / Vulkan 1.3 Android application that renders the **Amazon Lumberyard Bistro** scene (`https://developer.nvidia.com/orca/amazon-lumberyard-bistro`). Because raw FBX geometry and desktop DDS (`BC1`/`BC3`/`BC5`) textures cannot be used directly on mobile GPUs, an offline pipeline under `tools/` converts `source_assets/` into mobile-ready runtime assets under `app/src/main/assets/bistro/`.

---

## 1. Converting `source_assets` into Runtime Assets

### Prerequisites

1. **Homebrew Assimp** (used by the native C++ FBX converter):
   ```bash
   brew install assimp
   ```
2. **Python Virtual Environment** (`.venv` in repository root with `numpy` and `pillow`):
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install numpy pillow
   ```
3. **ARM `astcenc` Binary**:
   - Located at `tools/bin/astcenc` (or available on `PATH`).
   - If missing, download the macOS universal binary from `https://github.com/ARM-software/astc-encoder/releases` and place `astcenc` in `tools/bin/astcenc` (`chmod +x tools/bin/astcenc`).
4. **Source Assets (`source_assets/Bistro_v5_2/`)**:
   - Contains `BistroExterior.fbx`, `BistroInterior.fbx`, `BistroInterior_Wine.fbx`, `Textures/*.dds`, and `san_giuseppe_bridge_4k.hdr`.
   - Passing `--download` to `tools/build_assets.py` automatically downloads `source_assets/Bistro_v5_2.zip` from `https://developer.nvidia.com/bistro` and extracts it if `source_assets/Bistro_v5_2/` is not yet present.

### Running the Full Asset Pipeline

Run `tools/build_assets.py` using the repository's `.venv`:

```bash
.venv/bin/python tools/build_assets.py --download
```

Useful flags:
- `--download`: Downloads and extracts `source_assets/Bistro_v5_2.zip` if not already present.
- `--max-size <N>`: Maximum texture dimension (defaults to `1024`).
- `--jobs <N>`: Number of parallel worker processes for texture analysis and ASTC encoding (defaults to CPU count).
- `--skip-textures`: Rebuilds `scene.bin` and `environment.bin` without re-checking `.tex` files.
- Environment variable `BISTRO_AO_RAYS=<N>`: Number of hemisphere rays per vertex for offline sky-visibility baking (defaults to `48`; set `BISTRO_AO_RAYS=0` only for quick geometry debugging).

### What Each Stage of `tools/build_assets.py` Does

1. **Texture Alpha & Constant-Map Analysis (`build_assets/texture_stats.txt`)**
   - Scans `source_assets/Bistro_v5_2/Textures/*_BaseColor.dds` and `*_Specular.dds` to measure alpha cutout/translucency fractions (for material `AlphaMode` classification) and RGB mean/std-dev (so constant 16×16 ORM maps can be folded directly into scalar `roughnessFactor` and `metallicFactor` on the material).

2. **Native C++ Scene Converter (`tools/scene_converter/convert.cpp` + `sky_visibility.h`)**
   - Automatically compiled by `tools/build_assets.py` into `build_assets/bistro_convert` via:
     ```bash
     clang++ -std=c++17 -O2 -Wall -I$(brew --prefix)/include tools/scene_converter/convert.cpp -o build_assets/bistro_convert -L$(brew --prefix)/lib -lassimp
     ```
   - Loads `BistroExterior.fbx` (primary) and `BistroInterior.fbx` (secondary), deduplicating overlapping building shell nodes by name and world-space AABB.
   - Bakes per-vertex ambient sky visibility (`sky_visibility.h`) using a custom SAH BVH ray tracer (48 cosine-weighted upper-hemisphere rays per vertex, skipping glass triangles).
   - Splits geometry per material into spatially coherent clusters ($\le 2048$ triangles, $\le 12\text{ m}$ extent, 16-bit cluster-local indices) for GPU compute culling (`cull.comp`).
   - Packs vertices into 32-byte `bistro::GpuVertex` records (`app/src/main/cpp/scene_format.h`) and writes:
     - `app/src/main/assets/bistro/scene.bin` (geometry, materials, clusters, lights, camera keys)
     - `build_assets/texture_list.txt` (referenced textures and their `TextureKind`)
     - `build_assets/obstacles.pgm` (16-band 3D obstacle grid used by camera path tools)

3. **DDS to ASTC Texture Compression (`app/src/main/assets/bistro/textures/*.tex`)**
   - Converts every referenced `.dds` texture to custom `.tex` containers (`BTEX` header + full mip chain of raw ASTC blocks):
     - Base colour (`kTextureBaseColor`): `ASTC_6x6_SRGB`, sRGB-linearized box mip chain with alpha-coverage preservation for masked foliage.
     - Normal maps (`kTextureNormal`): Reconstructs $Z$ from 2-channel `BC5`, filters and renormalizes 3D normals per mip, encodes with `astcenc -cl -normal` (`ASTC_6x6_UNORM`, reading `.rgb` as $X$ and `.a` as $Y$ in `scene.frag`).
     - ORM maps (`kTextureOrm`): `ASTC_8x8_UNORM` at half max-dimension (only for non-constant ORM textures).
     - Emissive maps (`kTextureEmissive`): `ASTC_6x6_SRGB` at half max-dimension.
   - Existing `.tex` files newer than their source `.dds` are cached automatically.

4. **HDR Environment Pre-processing (`tools/hdr_io.py`)**
   - Reads `source_assets/Bistro_v5_2/san_giuseppe_bridge_4k.hdr`, extracts the sun disc direction and irradiance, and generates:
     - `app/src/main/assets/bistro/sky.tex` (2048×1024 `RGBA16F` equirectangular sky with mip chain)
     - `app/src/main/assets/bistro/env_specular.tex` (256×128 6-mip GGX importance-sampled specular IBL)
     - `app/src/main/assets/bistro/environment.bin` (`BENV` header + sun direction/irradiance + L2 `SH9` diffuse irradiance coefficients)
   - Copies `LICENSE.txt` and `README.txt` to `app/src/main/assets/bistro/BISTRO_LICENSE.txt` and `BISTRO_README.txt`.

### Camera Path & Inspection Tools (`tools/`)

- `.venv/bin/python tools/check_path.py [out.png]`: Checks the spline in `app/src/main/cpp/camera_path.cpp` against the 16-band obstacle map (`build_assets/obstacles.pgm`) and renders a top-down clearance map.
- `.venv/bin/python tools/relax_path.py`: Nudges rough route waypoints away from walls using `build_assets/obstacles.pgm`.
- `.venv/bin/python tools/capture.py <out_dir> <t1> <t2> ...`: Launches the app on a connected Android device at specific timestamps along the camera path and captures screenshots.
- `.venv/bin/python tools/contact_sheet.py <out.png> <img1.png> ...`: Montages captured screenshots into a labeled contact sheet.
- `tools/bench.sh <time> [scale] [msaa] [occlusion]`: Benchmarks GPU frame times on an attached device using `debug.bistro.*` system properties.

---

## 2. Building the Android Application & Shaders

### Environment Setup

Always ensure `JAVA_HOME` points to a modern JDK (e.g. JetBrains Runtime 21 bundled with Android Studio) so `jlink` and AGP run cleanly:

```bash
export JAVA_HOME=/Users/jimblackler/Library/Java/JavaVirtualMachines/jbr-21.0.9/Contents/Home
```

### Shader Compilation

GLSL shaders reside in `app/src/main/shaders/` (`common.glsl`, `cull.comp`, `hiz.comp`, `scene.vert`, `scene.frag`, `shadow.vert`, `shadow.frag`, `fullscreen.vert`, `sky.frag`, `taa.frag`, `exposure.comp`, `bloom_down.frag`, `bloom_up.frag`, `tonemap.frag`, `loading.frag`).
- They are compiled automatically during the CMake build (`app/src/main/cpp/CMakeLists.txt`) using the NDK's `glslc` (`--target-env=vulkan1.3 -O`) into SPIR-V uint32 arrays embedded in `shaders/generated_shaders.h`.
- You do **not** need to run `glslc` manually; `./gradlew assembleDebug` tracks all shader dependencies.

---

## 3. Mandatory Verification After Changes

Whenever you modify any code, shaders, resources, or build scripts in this repository, you **must** run the following verification checks before completing your task:

### 1. Regenerate Runtime Assets (if converter/format changed)
If you change `tools/scene_converter/*`, `tools/build_assets.py`, or `app/src/main/cpp/scene_format.h`, regenerate the runtime assets first:

```bash
.venv/bin/python tools/build_assets.py
```

### 2. Run Android Lint
Verify that no lint errors or warnings are introduced:

```bash
JAVA_HOME=/Users/jimblackler/Library/Java/JavaVirtualMachines/jbr-21.0.9/Contents/Home ./gradlew lintDebug
```

### 3. Verify Build
Verify that Kotlin, GLSL shaders, and C++20 native code compile cleanly:

```bash
JAVA_HOME=/Users/jimblackler/Library/Java/JavaVirtualMachines/jbr-21.0.9/Contents/Home ./gradlew assembleDebug
```

All checks must exit with code 0 and report `BUILD SUCCESSFUL` without unresolved lint errors.
