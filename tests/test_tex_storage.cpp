#include "test_common.h"

#include "../internal/TextureStorageCPU.h"

#include <memory>
#include <random>

void test_tex_storage() {
    printf("Test tex_storage        | ");

    { // Test three uncompressed storage layouts
        Ray::Cpu::TexStorageLinear<uint8_t, 4> storage_linear;
        Ray::Cpu::TexStorageTiled<uint8_t, 4> storage_tiled;
        Ray::Cpu::TexStorageSwizzled<uint8_t, 4> storage_swizzled;

        const int TextureRes = 4093;
        std::vector<Ray::color_t<uint8_t, 4>> test_pixels(TextureRes * TextureRes);

        { // Fill test pixels
            std::uniform_int_distribution<int> dist(0, 255);
            std::mt19937 gen(42);

            for (int j = 0; j < TextureRes; j++) {
                for (int i = 0; i < TextureRes; i++) {
                    test_pixels[j * TextureRes + i].v[0] = static_cast<uint8_t>(dist(gen));
                    test_pixels[j * TextureRes + i].v[1] = static_cast<uint8_t>(dist(gen));
                    test_pixels[j * TextureRes + i].v[2] = static_cast<uint8_t>(dist(gen));
                    test_pixels[j * TextureRes + i].v[3] = static_cast<uint8_t>(dist(gen));
                }
            }
        }

        const int res[2] = {TextureRes, TextureRes};
        require_fatal(storage_linear.Allocate(test_pixels, res, false) == 0);
        require_fatal(storage_tiled.Allocate(test_pixels, res, false) == 0);
        require_fatal(storage_swizzled.Allocate(test_pixels, res, false) == 0);

        for (int y = 0; y < TextureRes; ++y) {
            for (int x = 0; x < TextureRes; ++x) {
                const Ray::color_t<uint8_t, 4> sampled_color1 = storage_linear.Get(0, x, y, 0),
                                               sampled_color2 = storage_tiled.Get(0, x, y, 0),
                                               sampled_color3 = storage_swizzled.Get(0, x, y, 0);

                const Ray::color_t<uint8_t, 4> &test_color = test_pixels[y * TextureRes + x];
                require_fatal(sampled_color1.v[0] == test_color.v[0]);
                require_fatal(sampled_color1.v[1] == test_color.v[1]);
                require_fatal(sampled_color1.v[2] == test_color.v[2]);
                require_fatal(sampled_color1.v[3] == test_color.v[3]);

                require_fatal(sampled_color2.v[0] == test_color.v[0]);
                require_fatal(sampled_color2.v[1] == test_color.v[1]);
                require_fatal(sampled_color2.v[2] == test_color.v[2]);
                require_fatal(sampled_color2.v[3] == test_color.v[3]);

                require(sampled_color3.v[0] == test_color.v[0]);
                require(sampled_color3.v[1] == test_color.v[1]);
                require(sampled_color3.v[2] == test_color.v[2]);
                require(sampled_color3.v[3] == test_color.v[3]);
            }
        }
    }

    { // Test compressed storages
        Ray::Cpu::TexStorageBCn<1> storage_bc4;
        Ray::Cpu::TexStorageBCn<2> storage_bc5;

        const int TextureRes = 4093;
        std::vector<Ray::color_t<uint8_t, 1>> test_pixels_r(TextureRes * TextureRes);
        std::vector<Ray::color_t<uint8_t, 2>> test_pixels_rg(TextureRes * TextureRes);

        // Fill test pixels
        for (int j = 0; j < TextureRes; j++) {
            for (int i = 0; i < TextureRes; i++) {
                test_pixels_r[j * TextureRes + i].v[0] = static_cast<uint8_t>((i + j) % 255);

                test_pixels_rg[j * TextureRes + i].v[0] = static_cast<uint8_t>((i + j) % 255);
                test_pixels_rg[j * TextureRes + i].v[1] = static_cast<uint8_t>((i + j) % 255);
            }
        }

        const int res[2] = {TextureRes, TextureRes};
        require_fatal(storage_bc4.Allocate(test_pixels_r, res, false) == 0);
        require_fatal(storage_bc5.Allocate(test_pixels_rg, res, false) == 0);

        for (int y = 0; y < TextureRes; ++y) {
            for (int x = 0; x < TextureRes; ++x) {
                const Ray::color_t<uint8_t, 1> sampled_color1 = storage_bc4.Get(0, x, y, 0);
                const Ray::color_t<uint8_t, 2> sampled_color2 = storage_bc5.Get(0, x, y, 0);

                const Ray::color_t<uint8_t, 1> &test_color1 = test_pixels_r[y * TextureRes + x];
                const Ray::color_t<uint8_t, 2> &test_color2 = test_pixels_rg[y * TextureRes + x];
                
                require_fatal(std::abs(int(sampled_color1.v[0]) - test_color1.v[0]) < 8);

                require_fatal(std::abs(int(sampled_color2.v[0]) - test_color2.v[0]) < 8);
                require_fatal(std::abs(int(sampled_color2.v[1]) - test_color2.v[1]) < 8);
            }
        }
    }

    printf("OK\n");
}
