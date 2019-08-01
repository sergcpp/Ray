
#include <cstdio>

void test_atlas();
void test_simd();
void test_primary_ray_gen();
void test_mesh_lights();

int main() {
    test_atlas();
#if 0
    test_simd();
    test_primary_ray_gen();
#ifndef _DEBUG
    test_mesh_lights();
#endif
#endif

    puts("OK");
}

