
#include "RendererBase.cpp"
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

#if defined(__ARM_NEON__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
//#include "internal/RendererNEON.cpp"
#endif

#if !defined(__ANDROID__) && !defined(DISABLE_OCL)
//#include "internal/RendererOCL.cpp"
//#include "internal/SceneOCL.cpp"
//#include "internal/TextureAtlasOCL.cpp"
#endif

