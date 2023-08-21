# Changelog

## [Unreleased]

### Added

### Fixed

### Changed

### Removed

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



[Unreleased]: https://gitlab.com/sergcpp/Ray/-/compare/v0.2.0...master
[0.2.0]: https://gitlab.com/sergcpp/Ray/-/releases/v0.2.0
[0.1.0]: https://gitlab.com/sergcpp/Ray/-/releases/v0.1.0
