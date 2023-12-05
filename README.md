# Ray [![pipeline status](https://gitlab.com/sergcpp/Ray/badges/master/pipeline.svg)](https://gitlab.com/sergcpp/Ray/-/commits/master)

Embeddable cross-platform semi-realtime physically based renderer.

Features:

- Unidirectional pathtracing with NEE and skyportals
- Principled BSDF, filmic tonemapping, physical sky
- Automatic texture compression
- CPU backend accelerated using SSE/AVX/NEON extensions
- GPU backends (Vulkan, DirectX 12) with optional HW raytracing
- DNN denoising (manual port of OpenImageDenoise UNet), accelerated using VK_NV_cooperative_matrix if available
- Compatible with Windows (including ARM), Linux, and macOS

<details>
  <summary>Images</summary>

  - Rendered with time limit of 15 seconds using: <https://github.com/sergcpp/RayDemo>
  - Links to the original scenes:  \
    https://benedikt-bitterli.me/resources/  \
    https://www.blender.org/download/demo-files/  \
    https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-research/samples.html  \
    https://developer.nvidia.com/orca/amazon-lumberyard-bistro \
    https://wirewheelsclub.com/models/1965-ford-mustang-fastback \
    https://evermotion.org/shop/show_product/scene-1-ai43-archinteriors-for-blender/14564 \
    https://www.blendermarket.com/products/blender-eevee-modern-villa

  <div>
    <div float="left" >
      <img src="images/scene11.jpg" width="44.7%" />
      <img src="images/scene12.jpg" width="44.7%" />
    </div>
    <div float="left" >
      <img src="images/scene6.jpg" width="47.0%" />
      <img src="images/scene5.jpg" width="42.4%" />
    </div>
    <div float="left" >
      <img src="images/scene9.jpg" width="44.7%" />
      <img src="images/scene7.jpg" width="44.7%" />
    </div>
    <div float="left" >
      <img src="images/scene10.jpg" width="46.95%" />
      <img src="images/scene8.jpg" width="42.45%" />
    </div>
    <div float="left" >
      <img src="images/scene13.jpg" width="44.7%" />
      <img src="images/scene14.jpg" width="44.7%" />
    </div>
    <div float="left" >
      <img src="images/scene3.jpg" width="26.55%" />
      <img src="images/scene4.jpg" width="62.9%" />
    </div>
    <div float="left" >
      <img src="images/scene16.jpg" width="60.0%" />
      <img src="images/scene1.jpg" width="29.4%" />
    </div>
    <div float="left" >
      <img src="images/scene2.jpg" width="29.9%" />
      <img src="images/scene15.jpg" width="59.5%" />
    </div>
    <div float="left" >
      <img src="images/scene17.jpg" width="89.8%" />
    </div>
  </div>
</details>

<details>
  <summary>Usage</summary>

  ## Installation
The intended use is to add it as a submodule to an existing project:

```console
git submodule add https://github.com/sergcpp/Ray.git
```

Then in CMakeLists.txt file:

```cmake
add_subdirectory(Ray)
```

But also standalone samples can be compiled and run:

### Windows

```console
git clone https://github.com/sergcpp/Ray.git
cd Ray
mkdir build && cd build/
cmake ..
msbuild ALL_BUILD.vcxproj /p:Configuration=Release
```

### Linux/MacOS

```console
git clone https://github.com/sergcpp/Ray.git
cd Ray
mkdir build && cd build/
cmake .. -DCMAKE_BUILD_TYPE=Release && make
```

## Usage

### Image rendering

```c++
#include <Ray/Ray.h>

int main() {
    const int IMG_W = 256, IMG_H = 256;
    const int SAMPLE_COUNT = 64;

    // Initial frame resolution, can be changed later
    Ray::settings_t s;
    s.w = IMG_W;
    s.h = IMG_H;

    // Additional Ray::eRendererType parameter can be passed (Vulkan GPU renderer created by default)
    Ray::RendererBase *renderer = Ray::CreateRenderer(s, &Ray::g_stdout_log);

    // Each renderer has its own storage implementation (RAM, GPU-RAM),
    // so renderer itself should create scene object
    Ray::SceneBase *scene = renderer->CreateScene();

    // Setup environment
    Ray::environment_desc_t env_desc;
    env_desc.env_col[0] = env_desc.env_col[1] = env_desc.env_col[2] = 0.0f;
    scene->SetEnvironment(env_desc);

    // Add diffuse material
    Ray::shading_node_desc_t mat_desc1;
    mat_desc1.type = Ray::eShadingNode::Diffuse;

    const Ray::MaterialHandle mat1 = scene->AddMaterial(mat_desc1);

    // Add emissive materials
    Ray::shading_node_desc_t mat_desc2;
    mat_desc2.type = Ray::eShadingNode::Emissive;
    mat_desc2.strength = 4.0f;
    mat_desc2.base_color[0] = 1.0f;
    mat_desc2.base_color[1] = 0.0f;
    mat_desc2.base_color[2] = 0.0f;
    mat_desc2.multiple_importance = true; // Use NEE for this lightsource

    const Ray::MaterialHandle mat2 = scene->AddMaterial(mat_desc2);

    mat_desc2.base_color[0] = 0.0f;
    mat_desc2.base_color[1] = 1.0f;
    const Ray::MaterialHandle mat3 = scene->AddMaterial(mat_desc2);

    mat_desc2.base_color[1] = 0.0f;
    mat_desc2.base_color[2] = 1.0f;
    const Ray::MaterialHandle mat4 = scene->AddMaterial(mat_desc2);

    // Setup test mesh
    // Attribute layout is controlled by Ray::eVertexLayout enums
    // Is this example(PxyzNxyzTuv): position(3 floats), normal(3 floats), tex_coord(2 floats)
    // clang-format off
    const float attrs[] = { -1.0f, 0.0f, -1.0f,     0.0f, 1.0f, 0.0f,   1.0f, 0.0f,
                            1.0f, 0.0f, -1.0f,      0.0f, 1.0f, 0.0f,   0.0f, 0.0f,
                            1.0f, 0.0f, 1.0f,       0.0f, 1.0f, 0.0f,   0.0f, 1.0f,
                            -1.0f, 0.0f, 1.0f,      0.0f, 1.0f, 0.0f,   1.0f, 1.0f,

                            -1.0f, 0.5f, -1.0f,     0.0f, 1.0f, 0.0f,   1.0f, 1.0f,
                            -0.33f, 0.5f, -1.0f,    0.0f, 1.0f, 0.0f,   1.0f, 1.0f,
                            -0.33f, 0.0f, -1.0f,    0.0f, 1.0f, 0.0f,   1.0f, 1.0f,
                            0.33f, 0.5f, -1.0f,     0.0f, 1.0f, 0.0f,   0.0f, 0.0f,
                            0.33f, 0.0f, -1.0f,     0.0f, 1.0f, 0.0f,   0.0f, 0.0f,
                            1.0f, 0.5f, -1.0f,      0.0f, 1.0f, 0.0f,   0.0f, 0.0f };
    const uint32_t indices[] = { 0, 2, 1, 0, 3, 2,
                                 0, 5, 4, 6, 5, 0,
                                 5, 6, 7, 7, 6, 8,
                                 7, 8, 9, 8, 1, 9 };
    // clang-format on

    Ray::mesh_desc_t mesh_desc;
    mesh_desc.prim_type = Ray::ePrimType::TriangleList;
    mesh_desc.vtx_positions = {attrs, 0, 8};
    mesh_desc.vtx_normals = {attrs, 3, 8};
    mesh_desc.vtx_uvs = {attrs, 6, 8};
    mesh_desc.vtx_indices = indices;

    // Setup material groups
    const Ray::mat_group_desc_t groups[] = {
        {mat1, 0, 6},
        {mat2, 6, 6},
        {mat3, 12, 6},
        {mat4, 18, 6},
    };
    mesh_desc.groups = groups;

    Ray::MeshHandle mesh1 = scene->AddMesh(mesh_desc);

    // Instantiate mesh
    const float xform[] = {1.0f, 0.0f, 0.0f, 0.0f, //
                           0.0f, 1.0f, 0.0f, 0.0f, //
                           0.0f, 0.0f, 1.0f, 0.0f, //
                           0.0f, 0.0f, 0.0f, 1.0f};
    scene->AddMeshInstance(mesh1, xform);

    // Add camera
    const float view_origin[] = {2.0f, 2.0f, 2.0f};
    const float view_dir[] = {-0.577f, -0.577f, -0.577f};

    Ray::camera_desc_t cam_desc;
    cam_desc.type = Ray::eCamType::Persp;
    memcpy(&cam_desc.origin[0], &view_origin[0], 3 * sizeof(float));
    memcpy(&cam_desc.fwd[0], &view_dir[0], 3 * sizeof(float));
    cam_desc.fov = 45.0f;
    cam_desc.gamma = 2.2f;

    const Ray::CameraHandle cam = scene->AddCamera(cam_desc);
    scene->set_current_cam(cam);

    scene->Finalize();

    // Create region contex for frame, setup to use whole frame
    auto region = Ray::RegionContext{{0, 0, IMG_W, IMG_H}};

    // Render image
    for (int i = 0; i < SAMPLE_COUNT; i++) {
        // Each call performs one iteration, blocks until finished
        renderer->RenderScene(scene, region);
        printf("Renderered %i samples\n", i);
    }
    printf("Done\n");

    // Get rendered image pixels in 32-bit floating point RGBA format
    const Ray::color_data_rgba_t pixels = renderer->get_pixels_ref();

    for (int y = 0; y < IMG_H; y++) {
        for (int x = 0; x < IMG_W; x++) {
            int i = y * pixels.pitch + x;
            const Ray::color_rgba_t &p = pixels.ptr[i];

            float red = p.v[0];
            float green = p.v[1];
            float blue = p.v[2];
            float alpha = p.v[3];

            // ...
            // Save pixels or convert to desired format
            // ...
        }
    }

    delete scene;
    delete renderer;
}
```

![Screenshot](images/img1.jpg)

### Multithreading

With CPU backends it is safe to call RenderScene from different threads for non-overlaping image regions:

```c++
...
    if (Ray::RendererSupportsMultithreading(renderer->type())) {
        // Split image into 4 regions
        Ray::RegionContext regions[] = { Ray::RegionContext{ { 0,       0,       IMG_W/2, IMG_H/2 } },
                                         Ray::RegionContext{ { IMG_W/2, 0,       IMG_W/2, IMG_H/2 } },
                                         Ray::RegionContext{ { 0,       IMG_H/2, IMG_W/2, IMG_H/2 } },
                                         Ray::RegionContext{ { IMG_W/2, IMG_H/2, IMG_W/2, IMG_H/2 } } };

        #pragma omp parallel for
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < SAMPLE_COUNT; j++) {
                renderer->RenderScene(scene, regions[i]);
            }
        }
    }
...
```

### Denoising

The image can be denoised either with UNet (slower) or NLM filter (faster).

```c++
...
  if (EnableHighQualityDenoising) {
      // Initialize neural denoiser
      Ray::unet_filter_properties_t unet_props;
      renderer->InitUNetFilter(true, unet_props);

      for (int pass = 0; pass < unet_props.pass_count; ++pass) {
          renderer->DenoiseImage(pass, region);
      }
  } else {
      // Run simple NLM filter
      renderer->DenoiseImage(region);
  }
...
```

See [samples](samples) folder for more.
</details>


