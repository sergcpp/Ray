
#if !defined(DISABLE_GPU)
#include "internal/Vk/VKExt.cpp"
#include "internal/Vk/AccStructure.cpp"
#include "internal/Vk/Buffer.cpp"
#include "internal/Vk/Context.cpp"
#include "internal/Vk/DescriptorPool.cpp"
#include "internal/Vk/DrawCall.cpp"
#include "internal/Vk/Fence.cpp"
#include "internal/Vk/LinearAlloc.cpp"
#include "internal/Vk/MemoryAllocator.cpp"
#include "internal/Vk/Pipeline.cpp"
#include "internal/Vk/Program.cpp"
#include "internal/Vk/RenderPass.cpp"
#include "internal/Vk/Resource.cpp"
#include "internal/Vk/Sampler.cpp"
#include "internal/Vk/Shader.cpp"
#include "internal/Vk/Texture.cpp"
#include "internal/Vk/TextureAtlas.cpp"
#include "internal/Vk/Utils.cpp"
#include "internal/Vk/VertexInput.cpp"

#include "internal/RendererVK.cpp"
#include "internal/RendererVK_kernels.cpp"
#include "internal/SceneVK.cpp"

#include "third-party/SPIRV-Reflect/spirv_reflect.c"
#endif
