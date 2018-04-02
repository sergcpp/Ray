
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

/*#define NS ref2
#include "internal/CoreSIMD.cpp"
#include "internal/RendererSIMD.cpp"
#undef NS*/

/*#define NS sse2
#define USE_SSE
#include "internal/CoreSIMD.cpp"
#include "internal/RendererSIMD.cpp"
#undef USE_SSE
#undef NS*/

/*#define NS avx
#define USE_AVX
#include "internal/CoreSIMD.cpp"
#undef USE_AVX
#undef NS*/

/*#define NS ref3
#include "internal/CoreSIMD2.h"
#include "internal/RendererSIMD2.h"
#undef NS*/

//#include "internal/CoreAVX.cpp"
//#include "internal/RendererAVX.cpp"

#include "internal/RendererOCL.cpp"
#include "internal/SceneOCL.cpp"
#include "internal/TextureAtlasOCL.cpp"

// TODO:
// catch up CPU backends
// fix precision issues
// make camera fov work
// add deletion functions for OpenCL backend
// try again with spatial splits or remove unnecessary indirection
// add tests for intersection
// add android build
// add neon support
// add shading to cpu implementations

// DONE:
// add deletion functions for CPU backends
// macro tree for OpenCL
// split shade kernel
// simple textures
// texture atlas
// rethink 'shapes' in mesh description
// ray differentials
// sky and sun colors
// make render process incremental