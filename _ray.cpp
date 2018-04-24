
#include "RendererFactory.cpp"
#include "SceneBase.cpp"

#include "internal/BVHSplit.cpp"
#include "internal/TextureSplitter.cpp"

#include "internal/Core.cpp"

#include "internal/CoreRef.cpp"
#include "internal/FramebufferRef.cpp"
#include "internal/RendererRef.cpp"
#include "internal/SceneRef.cpp"
#include "internal/TextureAtlasRef.cpp"
#include "internal/TextureUtilsRef.cpp"

#include "internal/RendererOCL.cpp"
#include "internal/SceneOCL.cpp"
#include "internal/TextureAtlasOCL.cpp"

// TODO:
// make camera fov work
// add deletion functions for OpenCL backend
// try again with spatial splits or remove unnecessary indirection
// add tests for intersection
// add validation tests (use Cycles)
// add android build
// add neon support

// DONE:
// add shading to cpu implementations
// catch up CPU backends
// fix precision issues
// add deletion functions for CPU backends
// macro tree for OpenCL
// split shade kernel
// simple textures
// texture atlas
// rethink 'shapes' in mesh description
// ray differentials
// sky and sun colors
// make render process incremental