#include "test_common.h"

#include "../internal/TextureAtlasRef.h"

#include <memory>
#include <random>

void test_atlas() {
    {   // Test two types of atlas
        const int AtlasRes = 8192;
        Ray::Ref::TextureAtlasTiled atlas_linear = { AtlasRes, AtlasRes },
                                    atlas_tiled = { AtlasRes, AtlasRes };

        const int TextureRes = 4096;
        auto test_pixels = std::unique_ptr<Ray::pixel_color8_t[]>{ new Ray::pixel_color8_t[TextureRes * TextureRes * 4] };

        {   // Fill test pixels
            std::uniform_int_distribution<int> dist(0, 255);
            std::mt19937 gen(42);

            for (int j = 0; j < TextureRes; j++) {
                for (int i = 0; i < TextureRes; i++) {
                    test_pixels[j * TextureRes + i].r = static_cast<uint8_t>(dist(gen));
                    test_pixels[j * TextureRes + i].g = static_cast<uint8_t>(dist(gen));
                    test_pixels[j * TextureRes + i].b = static_cast<uint8_t>(dist(gen));
                    test_pixels[j * TextureRes + i].a = static_cast<uint8_t>(dist(gen));
                }
            }
        }

        int res[2] = { TextureRes, TextureRes }, pos_linear[2], pos_tiled[2];
        require(atlas_linear.Allocate(test_pixels.get(), res, pos_linear) == 0);
        require(atlas_tiled.Allocate(test_pixels.get(), res, pos_tiled) == 0);

        for (int y = 0; y < TextureRes; y++) {
            for (int x = 0; x < TextureRes; x++) {
                const Ray::pixel_color8_t sampled_color1 = atlas_linear.Get(0, pos_linear[0] + x + 1, pos_linear[1] + y + 1),
                                          sampled_color2 = atlas_tiled.Get(0, pos_tiled[0] + x + 1, pos_tiled[1] + y + 1);

                const Ray::pixel_color8_t &test_color = test_pixels[y * TextureRes + x];
                require(sampled_color1.r == test_color.r);
                require(sampled_color1.g == test_color.g);
                require(sampled_color1.b == test_color.b);
                require(sampled_color1.a == test_color.a);

                require(sampled_color2.r == test_color.r);
                require(sampled_color2.g == test_color.g);
                require(sampled_color2.b == test_color.b);
                require(sampled_color2.a == test_color.a);
            }
        }
    }
}