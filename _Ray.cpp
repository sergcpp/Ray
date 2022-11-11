
#include "Config.h"

#include "RendererBase.cpp"
#include "RendererFactory.cpp"
#include "SceneBase.cpp"

#include "internal/BVHSplit.cpp"
#include "internal/TextureSplitter.cpp"

#include "internal/Core.cpp"

#include "internal/CoreRef.cpp"
#include "internal/FramebufferRef.cpp"
#ifdef ENABLE_REF_IMPL
#include "internal/RendererRef.cpp"
#endif
#include "internal/SceneRef.cpp"
#include "internal/TextureStorageRef.cpp"
#include "internal/TextureUtilsRef.cpp"
#include "internal/Time.cpp"
#include "internal/Utils.cpp"

