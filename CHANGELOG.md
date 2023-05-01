# Changelog

## [Unreleased]

### Added
### Fixed
### Changed
### Removed

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



[Unreleased]: https://gitlab.com/sergcpp/Ray/-/compare/v0.1.0...master
[0.1.0]: https://gitlab.com/sergcpp/Ray/-/releases/v0.1.0
