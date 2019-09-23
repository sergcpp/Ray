
#include <cstdio>

void test_atlas();
void test_simd();
void test_primary_ray_gen();
void test_mesh_lights();
void test_texture();

int main() {
    test_atlas();
    test_simd();
    test_primary_ray_gen();
#ifndef _DEBUG
    test_mesh_lights();
    test_texture();
#endif
    puts("OK");
}

