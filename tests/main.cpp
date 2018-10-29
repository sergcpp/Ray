
#include <cstdio>

void test_simd();
void test_primary_ray_gen();
void test_mesh_lights();

int main() {
    test_simd();
    test_primary_ray_gen();
#ifndef _DEBUG
    test_mesh_lights();
#endif

    puts("OK");
}

