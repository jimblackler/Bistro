# Attribution

## Amazon Lumberyard Bistro

This application renders the **Amazon Lumberyard Bistro** scene, obtained from the NVIDIA
Open Research Content Archive (ORCA):

> Amazon Lumberyard Bistro, Open Research Content Archive (ORCA)
> Amazon Lumberyard, July 2017
> <http://developer.nvidia.com/orca/amazon-lumberyard-bistro>

```bibtex
@misc{ORCAAmazonBistro,
   title = {Amazon Lumberyard Bistro, Open Research Content Archive (ORCA)},
   author = {Amazon Lumberyard},
   year = {2017},
   month = {July},
   note = {\small \texttt{http://developer.nvidia.com/orca/amazon-lumberyard-bistro}},
   url = {http://developer.nvidia.com/orca/amazon-lumberyard-bistro}
}
```

The scene is licensed under the
[Creative Commons Attribution 4.0 International license (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
The full license text shipped with the asset is included in the app as
`app/src/main/assets/environments/bistro/LICENSE.txt` (and the original README as `README.txt`).

### Changes made to the original material

As required by CC BY 4.0, the modifications made for this app are indicated here. The original
`Bistro_v5_2.zip` (FBX + DDS) is converted by `tools/build_assets.py`:

- `BistroExterior.fbx` and `BistroInterior.fbx` are merged into a single scene, flattened into world
  space, re-indexed and split into spatial clusters for GPU culling; vertex attributes are quantised.
- All DDS (BC1/BC3/BC5) textures are decoded, downscaled to at most 1024 px, given new mip chains and
  re-encoded as ASTC for mobile GPUs. Normal maps are re-encoded in ASTC normal-map mode.
- Material transparency modes are derived from the texture alpha channels and material names.
- The camera fly-through path is new, loosely based on the camera animation embedded in the FBX.

The original, unmodified scene can be downloaded from the ORCA page above.

## Beeple Zero-Day

This application also renders the **Beeple Zero-Day** scene (`MEASURE_SEVEN_COLORED_LIGHTS.fbx`),
obtained from the NVIDIA Open Research Content Archive (ORCA):

> Zero-Day, Open Research Content Archive (ORCA)
> Mike Winkelmann and Kate Anderson, November 2019
> <https://developer.nvidia.com/orca/beeple-zero-day>

```bibtex
@misc{ZeroDay,
   title = {Zero-Day, Open Research Content Archive (ORCA)},
   author = {Mike Winkelmann and Kate Anderson},
   year = {2019},
   month = {November},
   note = {\small \texttt{https://developer.nvidia.com/orca/beeple-zero-day}},
   url = {https://developer.nvidia.com/orca/beeple-zero-day}
}
```

The scene is licensed under the
[Creative Commons Attribution 4.0 International license (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
The full license text shipped with the asset is included in the app as
`app/src/main/assets/environments/beeple/LICENSE.txt` (and the original README as `README.txt`).

### Changes made to the original material

As required by CC BY 4.0, the modifications made for this app are indicated here. The original
`ZeroDay_v1.zip` (FBX + DDS) is converted by `tools/build_assets.py`:

- `MEASURE_SEVEN/MEASURE_SEVEN_COLORED_LIGHTS.fbx` is flattened into world space, re-indexed and
  split into spatial clusters for GPU culling; vertex attributes and local ambient visibility are
  quantised into `app/src/main/assets/environments/beeple/scene.bin`.
- All referenced DDS (BC1/BC3/BC5) textures are decoded, downscaled to at most 1024 px, given mip
  chains and re-encoded as ASTC for mobile GPUs, with constant 16x16 maps folded into material factors.
- A custom camera fly-through path and studio HDR lighting environment are authored for real-time
  mobile playback.

The original, unmodified scene can be downloaded from the ORCA page above.

## Intel Sponza (2022) & Add-ons

This application also renders the **Intel Sponza** scene (`NewSponza_Main_glTF_003.gltf`,
`NewSponza_Curtains_glTF.gltf`, `NewSponza_IvyGrowth_glTF.gltf`, and `NewSponza_CypressTree_glTF.gltf`),
obtained from the [Intel GPU Research Samples Library](https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-processing-research/samples.html):

> Intel Sample Library — Sponza 2022 Scene & Add-on Packages
> Commissioned by Frank Meinl, sponsored by Anton Kaplanyan; Add-on Package Crew: Katica Putica, Cristiano Siqueira, Timothy Heath, Justin Prazen, Sebastian Herholz, Bruce Cherniak, and Anton Kaplanyan (2022)
> <https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-processing-research/samples.html>

```bibtex
@misc{sponza22,
   author = {Frank Meinl and Anton Kaplanyan},
   year = {2022},
   note = {https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-processing-research/samples.html},
   title = {Intel Sample Library}
}
```

The scene is licensed under the
[Creative Commons Attribution 4.0 International license (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
(for personal and educational use, with limited commercial use for marketing and print purposes as noted in the upstream README).
The full license text shipped with the asset is included in the app as
[`app/src/main/assets/environments/sponza/LICENSE.txt`](app/src/main/assets/environments/sponza/LICENSE.txt)
(and the upstream README as [`README.txt`](app/src/main/assets/environments/sponza/README.txt)).

### Changes made to the original material

- The base Sponza glTF and the Curtains, Ivy Growth, and Cypress Tree add-on glTF packages are merged into a single world-space scene (`app/src/main/assets/environments/sponza/scene.bin`), split into spatial clusters for GPU culling, and baked with per-vertex ambient sky visibility.
- All textures are downscaled to at most 1024 px, given new mip chains, and encoded as ASTC for mobile GPUs.
- A custom camera fly-through path and 22 warm colonnade/gallery point lights are authored for real-time mobile playback.

## Jungle Ruins

This application also renders the **Jungle Ruins** scene (`JungleRuins`), obtained from the
[Intel GPU Research Samples Library](https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-processing-research/samples.html):

> Intel Sample Library — Jungle Ruins 1.0b
> Cristiano Siqueira, Charlene Teets, Sebastian Herholz, Anton Sochenov, and Anton Kaplanyan (2024)
> <https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-processing-research/samples.html>

```bibtex
@misc{jungleruins,
   author = {Cristiano Siqueira and Charlene Teets and Sebastian Herholz and Anton Sochenov and Anton Kaplanyan},
   year = {2024},
   note = {https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-processing-research/samples.html},
   title = {Intel Sample Library}
}
```

The scene is licensed under the
[Creative Commons Attribution 4.0 International license (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
The full license text shipped with the asset is included in the app as
[`app/src/main/assets/environments/jungle/LICENSE.txt`](app/src/main/assets/environments/jungle/LICENSE.txt)
(and the upstream README as [`README.txt`](app/src/main/assets/environments/jungle/README.txt)).

### Changes made to the original material

- The upstream USD scene is assembled via `tools/build_jungle_gltf.py`, flattened into world space (`app/src/main/assets/environments/jungle/scene.bin`), split into spatial clusters for GPU culling, and baked with per-vertex ambient sky visibility.
- All textures are downscaled to at most 1024 px, given new mip chains, and encoded as ASTC for mobile GPUs.
- A custom camera fly-through path and animated water-wave shading are authored for real-time mobile playback.

## Environment maps

The HDR environment maps used for image-based lighting originate from [Poly Haven](https://polyhaven.com/)
(formerly HDRI Haven) and are licensed under [CC0](https://polyhaven.com/license):

- [`san_giuseppe_bridge_4k.hdr`](https://polyhaven.com/a/san_giuseppe_bridge) (shipped with the Bistro download; used in **Amazon Lumberyard Bistro**)
- [`kloppenheim_05_4k.hdr`](https://polyhaven.com/a/kloppenheim_05) (used in **Intel Sponza**)
- [`kloppenheim_06_puresky_8k.hdr`](https://polyhaven.com/a/kloppenheim_06_puresky) (used in **Jungle Ruins**)

Each HDR map is converted offline into an equirectangular sky texture (`sky.tex`), a 6-mip GGX pre-filtered specular environment map (`env_specular.tex`), and L2 spherical-harmonic irradiance coefficients (`environment.bin`).
