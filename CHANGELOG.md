# Changelog

## [Unreleased]

### Added

### Fixed

### Changed

### Removed

## [0.4.0] - 2025-02-25

### Added

- Spatial radiance caching
- Ray visibility flags for lights
- Clouds fluttering
- Multi-threaded pipeline initialization
- Direct loading of YCoCg textures
- AgX tonemapping
- Extended API to pass device/instance/command buffer

### Fixed

- Unnormalized emissive base color
- AMD artifacts with DirectX
- Incorrect handling of Vulkan initialization failure
- Missing UNet filter memory reallocation on resize
- Sun importance sampling issue
- Clouds CPU/GPU mismatch
- Envmap sampling seam
- Glowing corners with radiance caching enabled
- Unsynchronized access to cpu features struct
- Custom HLSL cross-compiler is used for DirectX backend
- Denoising artifacts with coop matrix enabled

### Changed

- Procedural sky is made pixel-perfect
- Skymap generation is moved to GPU
- Light BVH is quantized
- Matrix multiplication uses tiled/blocked approach
- SPIRV reflection data is extracted manually
- Cross-platform cooperative matrix is used insted of NV-specific
- Cornel box is used in samples

## [0.3.0] - 2023-12-03

### Added

- Light tree for hierarchical NEE and direct intersections
- PMJ sampling
- Ray type visibility masks (diffuse, glossy etc.)
- Gauss and Blackman-Harris image filters
- Multithreaded interface for scene Finalize
- Texture compression for CPU backends
- Direct loading of BC-compressed images
- TSAN tests on CI
- Path-space regularization
- Option to flip normalmap Y channel (DirectX convention)
- Ability to create single-sided mesh lights
- Physical sky sample
- Build version string

### Fixed

- Indirect clamp not affecting unshadowed lights
- Clamping not preserving color hue
- Glossy material desaturation an glazing angles
- Flipped front/back materials on GPU
- Not implemented RemoveMeshInstance, RemoveLight, RemoveMesh
- Incorrect scaled mesh lights intensity
- Incorrect mix node resolution in SIMD mode
- Incorrect NEE for textured triangle lights
- Incorrect MIS at the last bounce
- Incorrect far clip plane
- Incorrect transmission below 1.0
- Incorrect pitch black values with filmic tonemap
- Incorrect clamping of direct light intersections
- Triangular artifacts on AMD
- Crash on out of memory fallback to CPU RAM
- Fireflies with some HDRI images
- Flipped tangent basis on flipped UV islands
- Crash on hitting transparency limit (CPU only)
- Broken DOF RNG (GPU only)
- Incorrect alpha of 0.8 in samples
- Meshlights triangles memory leak
- Adreno issues with new driver
- Non-reproduceable GPU test runs

### Changed

- Improved physical sky (multiple scattering, clouds, moon, stars)
- Improved area light sampling
- Improved adaptive sampling
- BC4/BC5 compression uses SSE acceleration
- Texture compression uses NEON acceleration
- AUX buffers output is enabled permanently
- Bundled GPU shaders compressed using deflate
- More flexible interface for specifying vertex attributes
- Sphere light is allowed to have zero radius
- Bounded sampling is used for VNDF
- Mix node texture is allowed to be SRGB
- HWRT BLAS build happens per mesh instead of all at once
- 15-seconds time limit is used for README images

### Removed

- LinearAlloc
- Tent image filter

## [0.2.0] - 2023-08-05

### Added

- DirectX 12 backend
- DNN denoising (using weights from OIDN)
- GPU ray sorting
- Better memory allocator for GPU backends
- Physical sky (as baked env texture)
- Pipeline HWRT invocation (Vulkan only, disabled for now)
- Detailed denoising tests, basic samples

### Fixed

- Incorrect camera far clipping
- High memory consumption with Vulkan
- Bright pixels burn-in with filmic tonemapping
- Env map seam (GPU only)
- Incorrect directional light caustics
- Crash on weird resolutions (e.g. 513x513)
- Negative color values (GPU only)

### Changed

- Texture filtering is done stochastically
- Tests are invoked in parallel
- VNDF sampling is simplified (using spherical caps method)

## [0.1.0] - 2023-05-01

### Added

- NLM denoising with additional alpbedo and normal buffers
- Filmic tonemapping (using 3d LUTs)
- Adaptive sampling
- Direct/indirect light clamping
- AUX buffers output (albedo, normals and depth)
- Mesh debug names for logging
- Access to untonemapped pixels
- Ability to query available GPUs
- First performance tests on CI
- This CHANGELOG file

### Fixed

- Artifacts on integrated GPUs
- 32-bit ARM compilation
- Incorrect GPU region (non-fullscreen) invocation
- Lockstep execution mode (needed for debugging)
- Incorrect GPU timestamps units

### Changed

- Scene loading functions are made thread-safe
- CPU renderers are unified into single templated class
- Macos build uses universal executable on CI (instead of per arch)
- Manual unity build replaced with build-in CMake machanism
- Enums/bitmasks are made typesafe
- Tests are sped up to finish under 5 minutes



[Unreleased]: https://github.com/sergcpp/Ray/compare/v0.4.0...master
[0.4.0]: https://github.com/sergcpp/Ray/releases/v0.4.0
[0.3.0]: https://github.com/sergcpp/Ray/releases/v0.3.0
[0.2.0]: https://github.com/sergcpp/Ray/releases/v0.2.0
[0.1.0]: https://github.com/sergcpp/Ray/releases/v0.1.0
