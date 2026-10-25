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

## Environment map

The Bistro exterior HDR environment map `san_giuseppe_bridge_4k.hdr` ships with the Bistro download and
originates from [Poly Haven](https://polyhaven.com/a/san_giuseppe_bridge) (formerly HDRI Haven),
licensed CC0. It is converted into a sky texture, a pre-filtered specular environment map and
spherical-harmonic irradiance coefficients.
