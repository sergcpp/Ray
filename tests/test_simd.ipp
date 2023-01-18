using namespace Ray::NS;

{
    std::cout << "Test simd_fvec4  (" << (simd_fvec4::is_native() ? "hard" : "soft") << ") | ";

    simd_fvec4 v1, v2 = {42.0f}, v3 = {1.0f, 2.0f, 3.0f, 4.0f};

    require(v2[0] == 42.0f);
    require(v2[1] == 42.0f);
    require(v2[2] == 42.0f);
    require(v2[3] == 42.0f);

    require(v3[0] == 1.0f);
    require(v3[1] == 2.0f);
    require(v3[2] == 3.0f);
    require(v3[3] == 4.0f);

    require(v2.get<0>() == 42.0f);
    require(v2.get<1>() == 42.0f);
    require(v2.get<2>() == 42.0f);
    require(v2.get<3>() == 42.0f);

    require(v3.get<0>() == 1.0f);
    require(v3.get<1>() == 2.0f);
    require(v3.get<2>() == 3.0f);
    require(v3.get<3>() == 4.0f);

    simd_fvec4 v4(v2), v5 = v3;

    require(v4[0] == 42.0f);
    require(v4[1] == 42.0f);
    require(v4[2] == 42.0f);
    require(v4[3] == 42.0f);

    require(v5[0] == 1.0f);
    require(v5[1] == 2.0f);
    require(v5[2] == 3.0f);
    require(v5[3] == 4.0f);

    v1 = v5;

    require(v1[0] == 1.0f);
    require(v1[1] == 2.0f);
    require(v1[2] == 3.0f);
    require(v1[3] == 4.0f);

    float unaligned_array[] = {0.0f, 2.0f, 30.0f, 14.0f};
    alignas(alignof(simd_fvec4)) float aligned_array[] = {0.0f, 2.0f, 30.0f, 14.0f};

    auto v7 = simd_fvec4{&unaligned_array[0]}, v8 = simd_fvec4{&aligned_array[0], simd_mem_aligned};

    require(v7[0] == 0.0f);
    require(v7[1] == 2.0f);
    require(v7[2] == 30.0f);
    require(v7[3] == 14.0f);

    require(v8[0] == 0.0f);
    require(v8[1] == 2.0f);
    require(v8[2] == 30.0f);
    require(v8[3] == 14.0f);

    v5.copy_to(&unaligned_array[0]);
    v1.copy_to(&aligned_array[0], simd_mem_aligned);

    require(unaligned_array[0] == 1.0f);
    require(unaligned_array[1] == 2.0f);
    require(unaligned_array[2] == 3.0f);
    require(unaligned_array[3] == 4.0f);

    require(aligned_array[0] == 1.0f);
    require(aligned_array[1] == 2.0f);
    require(aligned_array[2] == 3.0f);
    require(aligned_array[3] == 4.0f);

    v1 = {1.0f, 2.0f, 3.0f, 4.0f};
    v2 = {4.0f, 5.0f, 6.0f, 7.0f};

    v3 = v1 + v2;
    v4 = v1 - v2;
    v5 = v1 * v2;
    auto v6 = v1 / v2;
    auto v66 = -v1;

    require(v3[0] == Approx(5));
    require(v3[1] == Approx(7));
    require(v3[2] == Approx(9));
    require(v3[3] == Approx(11));

    require(v4[0] == Approx(-3));
    require(v4[1] == Approx(-3));
    require(v4[2] == Approx(-3));
    require(v4[3] == Approx(-3));

    require(v5[0] == Approx(4));
    require(v5[1] == Approx(10));
    require(v5[2] == Approx(18));
    require(v5[3] == Approx(28));

    require(v6[0] == Approx(0.25));
    require(v6[1] == Approx(0.4));
    require(v6[2] == Approx(0.5));
    require(v6[3] == Approx(0.57142));

    require(v66[0] == Approx(-1.0));
    require(v66[1] == Approx(-2.0));
    require(v66[2] == Approx(-3.0));
    require(v66[3] == Approx(-4.0));

    v5 = sqrt(v5);

    require(v5[0] == Approx(2));
    require(v5[1] == Approx(3.1623));
    require(v5[2] == Approx(4.2426));
    require(v5[3] == Approx(5.2915));

    simd_fvec4 v55 = fract(v5);

    require(v55[0] == Approx(0));
    require(v55[1] == Approx(0.1623));
    require(v55[2] == Approx(0.2426));
    require(v55[3] == Approx(0.2915));

    simd_fvec4 v9 = {3.0f, 6.0f, 7.0f, 6.0f};

    auto v10 = simd_cast(v2 < v9);

    require(v10[0] == 0);
    require(v10[1] == -1);
    require(v10[2] == -1);
    require(v10[3] == 0);

    auto v11 = simd_cast(v2 > v9);

    require(v11[0] == -1);
    require(v11[1] == 0);
    require(v11[2] == 0);
    require(v11[3] == -1);

    static const float gather_source[] = {0, 42.0f, 0, 0, 12.0f, 0, 0, 0, 11.0f, 0, 0, 0, 0, 0, 0, 23.0f, 0, 32.0f};

    const simd_ivec4 v12i = {-1, 2, 6, 13};
    const simd_fvec4 v12 = gather<1>(gather_source + 2, v12i);

    require(v12[0] == Approx(42));
    require(v12[1] == Approx(12));
    require(v12[2] == Approx(11));
    require(v12[3] == Approx(23));

    const simd_ivec4 v13i = {-1, 6, 7, 6};
    const simd_fvec4 v13 = gather<2 /* Scale */>(gather_source + 3, v13i);

    require(v13[0] == Approx(42));
    require(v13[1] == Approx(23));
    require(v13[2] == Approx(32));
    require(v13[3] == Approx(23));

    const simd_fvec4 v14 = {42.0f, 0, 24.0f, 0};
    simd_fvec4 v15 = {0, 12.0f, 0, 0};

    v15 |= v14;

    require(v15[0] == 42.0f);
    require(v15[1] == 12.0f);
    require(v15[2] == 24.0f);
    require(v15[3] == 0);

    std::cout << "OK" << std::endl;
}

{
    std::cout << "Test simd_ivec4  (" << (simd_ivec4::is_native() ? "hard" : "soft") << ") | ";

    simd_ivec4 v1, v2 = {42}, v3 = {1, 2, 3, 4};

    require(v2[0] == 42);
    require(v2[1] == 42);
    require(v2[2] == 42);
    require(v2[3] == 42);

    require(v3[0] == 1);
    require(v3[1] == 2);
    require(v3[2] == 3);
    require(v3[3] == 4);

    require(v2.get<0>() == 42);
    require(v2.get<1>() == 42);
    require(v2.get<2>() == 42);
    require(v2.get<3>() == 42);

    require(v3.get<0>() == 1);
    require(v3.get<1>() == 2);
    require(v3.get<2>() == 3);
    require(v3.get<3>() == 4);

    simd_ivec4 v4(v2), v5 = v3;

    require(v4[0] == 42);
    require(v4[1] == 42);
    require(v4[2] == 42);
    require(v4[3] == 42);

    require(v5[0] == 1);
    require(v5[1] == 2);
    require(v5[2] == 3);
    require(v5[3] == 4);

    v1 = v5;

    require(v1[0] == 1);
    require(v1[1] == 2);
    require(v1[2] == 3);
    require(v1[3] == 4);

    int unaligned_array[] = {0, 2, 30, 14};
    alignas(alignof(simd_ivec4)) int aligned_array[] = {0, 2, 30, 14};

    auto v7 = simd_ivec4{&unaligned_array[0]}, v8 = simd_ivec4{&aligned_array[0], simd_mem_aligned};

    require(v7[0] == 0);
    require(v7[1] == 2);
    require(v7[2] == 30);
    require(v7[3] == 14);

    require(v8[0] == 0);
    require(v8[1] == 2);
    require(v8[2] == 30);
    require(v8[3] == 14);

    v5.copy_to(&unaligned_array[0]);
    v1.copy_to(&aligned_array[0], simd_mem_aligned);

    require(unaligned_array[0] == 1);
    require(unaligned_array[1] == 2);
    require(unaligned_array[2] == 3);
    require(unaligned_array[3] == 4);

    require(aligned_array[0] == 1);
    require(aligned_array[1] == 2);
    require(aligned_array[2] == 3);
    require(aligned_array[3] == 4);

    v1 = {1, 2, 3, 4};
    v2 = {4, 5, 6, 7};

    v3 = v1 + v2;
    v4 = v1 - v2;
    v5 = v1 * v2;
    auto v6 = v1 / v2;
    auto v66 = -v1;

    require(v3[0] == 5);
    require(v3[1] == 7);
    require(v3[2] == 9);
    require(v3[3] == 11);

    require(v4[0] == -3);
    require(v4[1] == -3);
    require(v4[2] == -3);
    require(v4[3] == -3);

    require(v5[0] == 4);
    require(v5[1] == 10);
    require(v5[2] == 18);
    require(v5[3] == 28);

    require(v6[0] == 0);
    require(v6[1] == 0);
    require(v6[2] == 0);
    require(v6[3] == 0);

    require(v66[0] == -1);
    require(v66[1] == -2);
    require(v66[2] == -3);
    require(v66[3] == -4);

    require(!v3.all_zeros());
    require(v6.all_zeros());

    static const int gather_source[] = {0, 42, 0, 0, 12, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 23, 0, 32};

    const simd_ivec4 v9i = {-1, 2, 6, 13};
    const simd_ivec4 v9 = gather<1>(gather_source + 2, v9i);

    require(v9[0] == 42);
    require(v9[1] == 12);
    require(v9[2] == 11);
    require(v9[3] == 23);

    const simd_ivec4 v10i = {-1, 6, 7, 6};
    const simd_ivec4 v10 = gather<2 /* Scale */>(gather_source + 3, v10i);

    require(v10[0] == 42);
    require(v10[1] == 23);
    require(v10[2] == 32);
    require(v10[3] == 23);

    const simd_ivec4 v11 = {-1, 0, -1, 0};
    simd_ivec4 v12 = {0, -1, 0, 0};

    v12 |= v11;

    require(v12[0] == -1);
    require(v12[1] == -1);
    require(v12[2] == -1);
    require(v12[3] == 0);

    const simd_ivec4 v13 = {-1, 0, -1, 0};
    simd_ivec4 v14 = {0, -1, 0, 0};

    v14 &= v13;

    require(v14[0] == 0);
    require(v14[1] == 0);
    require(v14[2] == 0);
    require(v14[3] == 0);

    const simd_ivec4 v15 = {-2147483647, 1, -42, 42};
    const simd_ivec4 v16 = srai(v15, 31);
    require((v16 != simd_ivec4{-1, 0, -1, 0}).all_zeros());

    std::cout << "OK" << std::endl;
}

{
    std::cout << "Test simd_fvec8  (" << (simd_fvec8::is_native() ? "hard" : "soft") << ") | ";

    simd_fvec8 v1, v2 = {42.0f}, v3 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    require(v2[0] == 42.0f);
    require(v2[1] == 42.0f);
    require(v2[2] == 42.0f);
    require(v2[3] == 42.0f);
    require(v2[4] == 42.0f);
    require(v2[5] == 42.0f);
    require(v2[6] == 42.0f);
    require(v2[7] == 42.0f);

    require(v3[0] == 1.0f);
    require(v3[1] == 2.0f);
    require(v3[2] == 3.0f);
    require(v3[3] == 4.0f);
    require(v3[4] == 5.0f);
    require(v3[5] == 6.0f);
    require(v3[6] == 7.0f);
    require(v3[7] == 8.0f);

    require(v2.get<0>() == 42.0f);
    require(v2.get<1>() == 42.0f);
    require(v2.get<2>() == 42.0f);
    require(v2.get<3>() == 42.0f);
    require(v2.get<4>() == 42.0f);
    require(v2.get<5>() == 42.0f);
    require(v2.get<6>() == 42.0f);
    require(v2.get<7>() == 42.0f);

    require(v3.get<0>() == 1.0f);
    require(v3.get<1>() == 2.0f);
    require(v3.get<2>() == 3.0f);
    require(v3.get<3>() == 4.0f);
    require(v3.get<4>() == 5.0f);
    require(v3.get<5>() == 6.0f);
    require(v3.get<6>() == 7.0f);
    require(v3.get<7>() == 8.0f);

    simd_fvec8 v4(v2), v5 = v3;

    require(v4[0] == 42.0f);
    require(v4[1] == 42.0f);
    require(v4[2] == 42.0f);
    require(v4[3] == 42.0f);
    require(v4[4] == 42.0f);
    require(v4[5] == 42.0f);
    require(v4[6] == 42.0f);
    require(v4[7] == 42.0f);

    require(v5[0] == 1.0f);
    require(v5[1] == 2.0f);
    require(v5[2] == 3.0f);
    require(v5[3] == 4.0f);
    require(v5[4] == 5.0f);
    require(v5[5] == 6.0f);
    require(v5[6] == 7.0f);
    require(v5[7] == 8.0f);

    v1 = v5;

    require(v1[0] == 1.0f);
    require(v1[1] == 2.0f);
    require(v1[2] == 3.0f);
    require(v1[3] == 4.0f);
    require(v1[4] == 5.0f);
    require(v1[5] == 6.0f);
    require(v1[6] == 7.0f);
    require(v1[7] == 8.0f);

    v1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 4.0f, 3.0f, 2.0f};
    v2 = {4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 10.0f, 12.0f, 1.0f};

    v3 = v1 + v2;
    v4 = v1 - v2;
    v5 = v1 * v2;
    auto v6 = v1 / v2;
    auto v66 = -v1;

    require(v3[0] == Approx(5));
    require(v3[1] == Approx(7));
    require(v3[2] == Approx(9));
    require(v3[3] == Approx(11));
    require(v3[4] == Approx(13));
    require(v3[5] == Approx(14));
    require(v3[6] == Approx(15));
    require(v3[7] == Approx(3));

    require(v4[0] == Approx(-3));
    require(v4[1] == Approx(-3));
    require(v4[2] == Approx(-3));
    require(v4[3] == Approx(-3));
    require(v4[4] == Approx(-3));
    require(v4[5] == Approx(-6));
    require(v4[6] == Approx(-9));
    require(v4[7] == Approx(1));

    require(v5[0] == Approx(4));
    require(v5[1] == Approx(10));
    require(v5[2] == Approx(18));
    require(v5[3] == Approx(28));
    require(v5[4] == Approx(40));
    require(v5[5] == Approx(40));
    require(v5[6] == Approx(36));
    require(v5[7] == Approx(2));

    require(v6[0] == Approx(0.25));
    require(v6[1] == Approx(0.4));
    require(v6[2] == Approx(0.5));
    require(v6[3] == Approx(0.57142));
    require(v6[4] == Approx(0.625));
    require(v6[5] == Approx(0.4));
    require(v6[6] == Approx(0.25));
    require(v6[7] == Approx(2.0));

    require(v66[0] == Approx(-1.0));
    require(v66[1] == Approx(-2.0));
    require(v66[2] == Approx(-3.0));
    require(v66[3] == Approx(-4.0));
    require(v66[4] == Approx(-5.0));
    require(v66[5] == Approx(-4.0));
    require(v66[6] == Approx(-3.0));
    require(v66[7] == Approx(-2.0));

    v5 = sqrt(v5);

    require(v5[0] == Approx(2));
    require(v5[1] == Approx(3.1623));
    require(v5[2] == Approx(4.2426));
    require(v5[3] == Approx(5.2915));
    require(v5[4] == Approx(6.3246));
    require(v5[5] == Approx(6.3246));
    require(v5[6] == Approx(6));
    require(v5[7] == Approx(1.4142));

    simd_fvec8 v55 = fract(v5);

    require(v55[0] == Approx(0));
    require(v55[1] == Approx(0.1623));
    require(v55[2] == Approx(0.2426));
    require(v55[3] == Approx(0.2915));
    require(v55[4] == Approx(0.3246));
    require(v55[5] == Approx(0.3246));
    require(v55[6] == Approx(0));
    require(v55[7] == Approx(0.4142));

    simd_fvec8 v9 = {3.0f, 6.0f, 7.0f, 6.0f, 2.0f, 12.0f, 18.0f, 0.0f};

    auto v10 = simd_cast(v2 < v9);

    require(v10[0] == 0);
    require(v10[1] == -1);
    require(v10[2] == -1);
    require(v10[3] == 0);
    require(v10[4] == 0);
    require(v10[5] == -1);
    require(v10[6] == -1);
    require(v10[7] == 0);

    auto v11 = simd_cast(v2 > v9);

    require(v11[0] == -1);
    require(v11[1] == 0);
    require(v11[2] == 0);
    require(v11[3] == -1);
    require(v11[4] == -1);
    require(v11[5] == 0);
    require(v11[6] == 0);
    require(v11[7] == -1);

    static const float gather_source[] = {0, 42.0f, 0, 0, 12.0f, 0, 0, 0, 11.0f, 0, 0, 0, 0, 0, 0, 23.0f, 0, 32.0f,
                                          0, 42.0f, 0, 0, 12.0f, 0, 0, 0, 11.0f, 0, 0, 0, 0, 0, 0, 23.0f, 0, 32.0f};

    const simd_ivec8 v12i = {-1, 2, 6, 13, -1, 2, 6, 13};
    const simd_fvec8 v12 = gather<1>(gather_source + 2, v12i);

    require(v12[0] == Approx(42));
    require(v12[1] == Approx(12));
    require(v12[2] == Approx(11));
    require(v12[3] == Approx(23));
    require(v12[4] == Approx(42));
    require(v12[5] == Approx(12));
    require(v12[6] == Approx(11));
    require(v12[7] == Approx(23));

    const simd_ivec8 v13i = {-1, 6, 7, 6, -1, 6, 7, 6};
    const simd_fvec8 v13 = gather<2 /* Scale */>(gather_source + 3, v13i);

    require(v13[0] == Approx(42));
    require(v13[1] == Approx(23));
    require(v13[2] == Approx(32));
    require(v13[3] == Approx(23));
    require(v13[4] == Approx(42));
    require(v13[5] == Approx(23));
    require(v13[6] == Approx(32));
    require(v13[7] == Approx(23));

    const simd_fvec8 v14 = {42.0f, 0, 24.0f, 0, 42.0f, 0, 24.0f, 0};
    simd_fvec8 v15 = {0, 12.0f, 0, 0, 0, 12.0f, 0, 0};

    v15 |= v14;

    require(v15[0] == 42.0f);
    require(v15[1] == 12.0f);
    require(v15[2] == 24.0f);
    require(v15[3] == 0);
    require(v15[4] == 42.0f);
    require(v15[5] == 12.0f);
    require(v15[6] == 24.0f);
    require(v15[7] == 0);

    std::cout << "OK" << std::endl;
}

{
    std::cout << "Test simd_ivec8  (" << (simd_ivec8::is_native() ? "hard" : "soft") << ") | ";

    simd_ivec8 v1, v2 = {42}, v3 = {1, 2, 3, 4, 5, 6, 7, 8};

    require(v2[0] == 42);
    require(v2[1] == 42);
    require(v2[2] == 42);
    require(v2[3] == 42);
    require(v2[4] == 42);
    require(v2[5] == 42);
    require(v2[6] == 42);
    require(v2[7] == 42);

    require(v3[0] == 1);
    require(v3[1] == 2);
    require(v3[2] == 3);
    require(v3[3] == 4);
    require(v3[4] == 5);
    require(v3[5] == 6);
    require(v3[6] == 7);
    require(v3[7] == 8);

    require(v2.get<0>() == 42);
    require(v2.get<1>() == 42);
    require(v2.get<2>() == 42);
    require(v2.get<3>() == 42);
    require(v2.get<4>() == 42);
    require(v2.get<5>() == 42);
    require(v2.get<6>() == 42);
    require(v2.get<7>() == 42);

    require(v3.get<0>() == 1);
    require(v3.get<1>() == 2);
    require(v3.get<2>() == 3);
    require(v3.get<3>() == 4);
    require(v3.get<4>() == 5);
    require(v3.get<5>() == 6);
    require(v3.get<6>() == 7);
    require(v3.get<7>() == 8);

    simd_ivec8 v4(v2), v5 = v3;

    require(v4[0] == 42);
    require(v4[1] == 42);
    require(v4[2] == 42);
    require(v4[3] == 42);
    require(v4[4] == 42);
    require(v4[5] == 42);
    require(v4[6] == 42);
    require(v4[7] == 42);

    require(v5[0] == 1);
    require(v5[1] == 2);
    require(v5[2] == 3);
    require(v5[3] == 4);
    require(v5[4] == 5);
    require(v5[5] == 6);
    require(v5[6] == 7);
    require(v5[7] == 8);

    v1 = v5;

    require(v1[0] == 1);
    require(v1[1] == 2);
    require(v1[2] == 3);
    require(v1[3] == 4);
    require(v1[4] == 5);
    require(v1[5] == 6);
    require(v1[6] == 7);
    require(v1[7] == 8);

    v1 = {1, 2, 3, 4, 5, 4, 3, 2};
    v2 = {4, 5, 6, 7, 8, 10, 12, 1};

    v3 = v1 + v2;
    v4 = v1 - v2;
    v5 = v1 * v2;
    auto v6 = v1 / v2;
    auto v66 = -v1;

    require(v3[0] == 5);
    require(v3[1] == 7);
    require(v3[2] == 9);
    require(v3[3] == 11);
    require(v3[4] == 13);
    require(v3[5] == 14);
    require(v3[6] == 15);
    require(v3[7] == 3);

    require(v4[0] == -3);
    require(v4[1] == -3);
    require(v4[2] == -3);
    require(v4[3] == -3);
    require(v4[4] == -3);
    require(v4[5] == -6);
    require(v4[6] == -9);
    require(v4[7] == 1);

    require(v5[0] == 4);
    require(v5[1] == 10);
    require(v5[2] == 18);
    require(v5[3] == 28);
    require(v5[4] == 40);
    require(v5[5] == 40);
    require(v5[6] == 36);
    require(v5[7] == 2);

    require(v6[0] == 0);
    require(v6[1] == 0);
    require(v6[2] == 0);
    require(v6[3] == 0);
    require(v6[4] == 0);
    require(v6[5] == 0);
    require(v6[6] == 0);
    require(v6[7] == 2);

    require(v66[0] == -1);
    require(v66[1] == -2);
    require(v66[2] == -3);
    require(v66[3] == -4);
    require(v66[4] == -5);
    require(v66[5] == -4);
    require(v66[6] == -3);
    require(v66[7] == -2);

    static const int gather_source[] = {0, 42, 0, 0, 12, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 23, 0, 32,
                                        0, 42, 0, 0, 12, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 23, 0, 32};

    const simd_ivec8 v9i = {-1, 2, 6, 13, -1, 2, 6, 13};
    const simd_ivec8 v9 = gather<1>(gather_source + 2, v9i);

    require(v9[0] == 42);
    require(v9[1] == 12);
    require(v9[2] == 11);
    require(v9[3] == 23);
    require(v9[4] == 42);
    require(v9[5] == 12);
    require(v9[6] == 11);
    require(v9[7] == 23);

    const simd_ivec8 v10i = {-1, 6, 7, 6, -1, 6, 7, 6};
    const simd_ivec8 v10 = gather<2 /* Scale */>(gather_source + 3, v10i);

    require(v10[0] == 42);
    require(v10[1] == 23);
    require(v10[2] == 32);
    require(v10[3] == 23);
    require(v10[4] == 42);
    require(v10[5] == 23);
    require(v10[6] == 32);
    require(v10[7] == 23);

    const simd_ivec8 v11 = {-1, 0, -1, 0, -1, 0, -1, 0};
    simd_ivec8 v12 = {0, -1, 0, 0, 0, -1, 0, 0};

    v12 |= v11;

    require(v12[0] == -1);
    require(v12[1] == -1);
    require(v12[2] == -1);
    require(v12[3] == 0);
    require(v12[4] == -1);
    require(v12[5] == -1);
    require(v12[6] == -1);
    require(v12[7] == 0);

    const simd_ivec8 v13 = {-1, 0, -1, 0, -1, 0, -1, 0};
    simd_ivec8 v14 = {0, -1, 0, 0, 0, -1, 0, 0};

    v14 &= v13;

    require(v14[0] == 0);
    require(v14[1] == 0);
    require(v14[2] == 0);
    require(v14[3] == 0);
    require(v14[4] == 0);
    require(v14[5] == 0);
    require(v14[6] == 0);
    require(v14[7] == 0);

    const simd_ivec8 v15 = {-2147483647, 1, -42, 42, -2147483647, 1, -42, 42};
    const simd_ivec8 v16 = srai(v15, 31);
    require((v16 != simd_ivec8{-1, 0, -1, 0, -1, 0, -1, 0}).all_zeros());

    std::cout << "OK" << std::endl;
}

//////////////////////////////////////////////////

{
    std::cout << "Test simd_fvec16 (" << (simd_fvec16::is_native() ? "hard" : "soft") << ") | ";

    simd_fvec16 v1, v2 = {42.0f}, v3 = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                                        9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};

    require(v2[0] == 42.0f);
    require(v2[1] == 42.0f);
    require(v2[2] == 42.0f);
    require(v2[3] == 42.0f);
    require(v2[4] == 42.0f);
    require(v2[5] == 42.0f);
    require(v2[6] == 42.0f);
    require(v2[7] == 42.0f);
    require(v2[8] == 42.0f);
    require(v2[9] == 42.0f);
    require(v2[10] == 42.0f);
    require(v2[11] == 42.0f);
    require(v2[12] == 42.0f);
    require(v2[13] == 42.0f);
    require(v2[14] == 42.0f);
    require(v2[15] == 42.0f);

    require(v3[0] == 1.0f);
    require(v3[1] == 2.0f);
    require(v3[2] == 3.0f);
    require(v3[3] == 4.0f);
    require(v3[4] == 5.0f);
    require(v3[5] == 6.0f);
    require(v3[6] == 7.0f);
    require(v3[7] == 8.0f);
    require(v3[8] == 9.0f);
    require(v3[9] == 10.0f);
    require(v3[10] == 11.0f);
    require(v3[11] == 12.0f);
    require(v3[12] == 13.0f);
    require(v3[13] == 14.0f);
    require(v3[14] == 15.0f);
    require(v3[15] == 16.0f);

    require(v2.get<0>() == 42.0f);
    require(v2.get<1>() == 42.0f);
    require(v2.get<2>() == 42.0f);
    require(v2.get<3>() == 42.0f);
    require(v2.get<4>() == 42.0f);
    require(v2.get<5>() == 42.0f);
    require(v2.get<6>() == 42.0f);
    require(v2.get<7>() == 42.0f);
    require(v2.get<8>() == 42.0f);
    require(v2.get<9>() == 42.0f);
    require(v2.get<10>() == 42.0f);
    require(v2.get<11>() == 42.0f);
    require(v2.get<12>() == 42.0f);
    require(v2.get<13>() == 42.0f);
    require(v2.get<14>() == 42.0f);
    require(v2.get<15>() == 42.0f);

    require(v3.get<0>() == 1.0f);
    require(v3.get<1>() == 2.0f);
    require(v3.get<2>() == 3.0f);
    require(v3.get<3>() == 4.0f);
    require(v3.get<4>() == 5.0f);
    require(v3.get<5>() == 6.0f);
    require(v3.get<6>() == 7.0f);
    require(v3.get<7>() == 8.0f);
    require(v3.get<8>() == 9.0f);
    require(v3.get<9>() == 10.0f);
    require(v3.get<10>() == 11.0f);
    require(v3.get<11>() == 12.0f);
    require(v3.get<12>() == 13.0f);
    require(v3.get<13>() == 14.0f);
    require(v3.get<14>() == 15.0f);
    require(v3.get<15>() == 16.0f);

    simd_fvec16 v4(v2), v5 = v3;

    require(v4[0] == 42.0f);
    require(v4[1] == 42.0f);
    require(v4[2] == 42.0f);
    require(v4[3] == 42.0f);
    require(v4[4] == 42.0f);
    require(v4[5] == 42.0f);
    require(v4[6] == 42.0f);
    require(v4[7] == 42.0f);
    require(v4[8] == 42.0f);
    require(v4[9] == 42.0f);
    require(v4[10] == 42.0f);
    require(v4[11] == 42.0f);
    require(v4[12] == 42.0f);
    require(v4[13] == 42.0f);
    require(v4[14] == 42.0f);
    require(v4[15] == 42.0f);

    require(v5[0] == 1.0f);
    require(v5[1] == 2.0f);
    require(v5[2] == 3.0f);
    require(v5[3] == 4.0f);
    require(v5[4] == 5.0f);
    require(v5[5] == 6.0f);
    require(v5[6] == 7.0f);
    require(v5[7] == 8.0f);
    require(v5[8] == 9.0f);
    require(v5[9] == 10.0f);
    require(v5[10] == 11.0f);
    require(v5[11] == 12.0f);
    require(v5[12] == 13.0f);
    require(v5[13] == 14.0f);
    require(v5[14] == 15.0f);
    require(v5[15] == 16.0f);

    v1 = v5;

    require(v1[0] == 1.0f);
    require(v1[1] == 2.0f);
    require(v1[2] == 3.0f);
    require(v1[3] == 4.0f);
    require(v1[4] == 5.0f);
    require(v1[5] == 6.0f);
    require(v1[6] == 7.0f);
    require(v1[7] == 8.0f);
    require(v1[8] == 9.0f);
    require(v1[9] == 10.0f);
    require(v1[10] == 11.0f);
    require(v1[11] == 12.0f);
    require(v1[12] == 13.0f);
    require(v1[13] == 14.0f);
    require(v1[14] == 15.0f);
    require(v1[15] == 16.0f);

    v1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 4.0f, 3.0f, 2.0f};
    v2 = {4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 10.0f, 12.0f, 1.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 10.0f, 12.0f, 1.0f};

    v3 = v1 + v2;
    v4 = v1 - v2;
    v5 = v1 * v2;
    auto v6 = v1 / v2;
    auto v66 = -v1;

    require(v3[0] == Approx(5));
    require(v3[1] == Approx(7));
    require(v3[2] == Approx(9));
    require(v3[3] == Approx(11));
    require(v3[4] == Approx(13));
    require(v3[5] == Approx(14));
    require(v3[6] == Approx(15));
    require(v3[7] == Approx(3));
    require(v3[8] == Approx(5));
    require(v3[9] == Approx(7));
    require(v3[10] == Approx(9));
    require(v3[11] == Approx(11));
    require(v3[12] == Approx(13));
    require(v3[13] == Approx(14));
    require(v3[14] == Approx(15));
    require(v3[15] == Approx(3));

    require(v4[0] == Approx(-3));
    require(v4[1] == Approx(-3));
    require(v4[2] == Approx(-3));
    require(v4[3] == Approx(-3));
    require(v4[4] == Approx(-3));
    require(v4[5] == Approx(-6));
    require(v4[6] == Approx(-9));
    require(v4[7] == Approx(1));
    require(v4[8] == Approx(-3));
    require(v4[9] == Approx(-3));
    require(v4[10] == Approx(-3));
    require(v4[11] == Approx(-3));
    require(v4[12] == Approx(-3));
    require(v4[13] == Approx(-6));
    require(v4[14] == Approx(-9));
    require(v4[15] == Approx(1));

    require(v5[0] == Approx(4));
    require(v5[1] == Approx(10));
    require(v5[2] == Approx(18));
    require(v5[3] == Approx(28));
    require(v5[4] == Approx(40));
    require(v5[5] == Approx(40));
    require(v5[6] == Approx(36));
    require(v5[7] == Approx(2));
    require(v5[8] == Approx(4));
    require(v5[9] == Approx(10));
    require(v5[10] == Approx(18));
    require(v5[11] == Approx(28));
    require(v5[12] == Approx(40));
    require(v5[13] == Approx(40));
    require(v5[14] == Approx(36));
    require(v5[15] == Approx(2));

    require(v6[0] == Approx(0.25));
    require(v6[1] == Approx(0.4));
    require(v6[2] == Approx(0.5));
    require(v6[3] == Approx(0.57142));
    require(v6[4] == Approx(0.625));
    require(v6[5] == Approx(0.4));
    require(v6[6] == Approx(0.25));
    require(v6[7] == Approx(2.0));
    require(v6[8] == Approx(0.25));
    require(v6[9] == Approx(0.4));
    require(v6[10] == Approx(0.5));
    require(v6[11] == Approx(0.57142));
    require(v6[12] == Approx(0.625));
    require(v6[13] == Approx(0.4));
    require(v6[14] == Approx(0.25));
    require(v6[15] == Approx(2.0));

    require(v66[0] == Approx(-1.0));
    require(v66[1] == Approx(-2.0));
    require(v66[2] == Approx(-3.0));
    require(v66[3] == Approx(-4.0));
    require(v66[4] == Approx(-5.0));
    require(v66[5] == Approx(-4.0));
    require(v66[6] == Approx(-3.0));
    require(v66[7] == Approx(-2.0));
    require(v66[8] == Approx(-1.0));
    require(v66[9] == Approx(-2.0));
    require(v66[10] == Approx(-3.0));
    require(v66[11] == Approx(-4.0));
    require(v66[12] == Approx(-5.0));
    require(v66[13] == Approx(-4.0));
    require(v66[14] == Approx(-3.0));
    require(v66[15] == Approx(-2.0));

    v5 = sqrt(v5);

    require(v5[0] == Approx(2));
    require(v5[1] == Approx(3.1623));
    require(v5[2] == Approx(4.2426));
    require(v5[3] == Approx(5.2915));
    require(v5[4] == Approx(6.3246));
    require(v5[5] == Approx(6.3246));
    require(v5[6] == Approx(6));
    require(v5[7] == Approx(1.4142));
    require(v5[8] == Approx(2));
    require(v5[9] == Approx(3.1623));
    require(v5[10] == Approx(4.2426));
    require(v5[11] == Approx(5.2915));
    require(v5[12] == Approx(6.3246));
    require(v5[13] == Approx(6.3246));
    require(v5[14] == Approx(6));
    require(v5[15] == Approx(1.4142));

    simd_fvec16 v55 = fract(v5);

    require(v55[0] == Approx(0));
    require(v55[1] == Approx(0.1623));
    require(v55[2] == Approx(0.2426));
    require(v55[3] == Approx(0.2915));
    require(v55[4] == Approx(0.3246));
    require(v55[5] == Approx(0.3246));
    require(v55[6] == Approx(0));
    require(v55[7] == Approx(0.4142));
    require(v55[8] == Approx(0));
    require(v55[9] == Approx(0.1623));
    require(v55[10] == Approx(0.2426));
    require(v55[11] == Approx(0.2915));
    require(v55[12] == Approx(0.3246));
    require(v55[13] == Approx(0.3246));
    require(v55[14] == Approx(0));
    require(v55[15] == Approx(0.4142));

    simd_fvec16 v9 = {3.0f, 6.0f, 7.0f, 6.0f, 2.0f, 12.0f, 18.0f, 0.0f,
                      3.0f, 6.0f, 7.0f, 6.0f, 2.0f, 12.0f, 18.0f, 0.0f};

    auto v10 = simd_cast(v2 < v9);

    require(v10[0] == 0);
    require(v10[1] == -1);
    require(v10[2] == -1);
    require(v10[3] == 0);
    require(v10[4] == 0);
    require(v10[5] == -1);
    require(v10[6] == -1);
    require(v10[7] == 0);

    auto v11 = simd_cast(v2 > v9);

    require(v11[0] == -1);
    require(v11[1] == 0);
    require(v11[2] == 0);
    require(v11[3] == -1);
    require(v11[4] == -1);
    require(v11[5] == 0);
    require(v11[6] == 0);
    require(v11[7] == -1);

    static const float gather_source[] = {0, 42.0f, 0, 0, 12.0f, 0, 0, 0, 11.0f, 0, 0, 0, 0, 0, 0, 23.0f, 0, 32.0f,
                                          0, 42.0f, 0, 0, 12.0f, 0, 0, 0, 11.0f, 0, 0, 0, 0, 0, 0, 23.0f, 0, 32.0f,
                                          0, 42.0f, 0, 0, 12.0f, 0, 0, 0, 11.0f, 0, 0, 0, 0, 0, 0, 23.0f, 0, 32.0f,
                                          0, 42.0f, 0, 0, 12.0f, 0, 0, 0, 11.0f, 0, 0, 0, 0, 0, 0, 23.0f, 0, 32.0f};

    const simd_ivec16 v12i = {-1, 2, 6, 13, -1, 2, 6, 13, -1, 2, 6, 13, -1, 2, 6, 13};
    const simd_fvec16 v12 = gather<1>(gather_source + 2, v12i);

    require(v12[0] == Approx(42));
    require(v12[1] == Approx(12));
    require(v12[2] == Approx(11));
    require(v12[3] == Approx(23));
    require(v12[4] == Approx(42));
    require(v12[5] == Approx(12));
    require(v12[6] == Approx(11));
    require(v12[7] == Approx(23));
    require(v12[8] == Approx(42));
    require(v12[9] == Approx(12));
    require(v12[10] == Approx(11));
    require(v12[11] == Approx(23));
    require(v12[12] == Approx(42));
    require(v12[13] == Approx(12));
    require(v12[14] == Approx(11));
    require(v12[15] == Approx(23));

    const simd_ivec16 v13i = {-1, 6, 7, 6, -1, 6, 7, 6, -1, 6, 7, 6, -1, 6, 7, 6};
    const simd_fvec16 v13 = gather<2 /* Scale */>(gather_source + 3, v13i);

    require(v13[0] == Approx(42));
    require(v13[1] == Approx(23));
    require(v13[2] == Approx(32));
    require(v13[3] == Approx(23));
    require(v13[4] == Approx(42));
    require(v13[5] == Approx(23));
    require(v13[6] == Approx(32));
    require(v13[7] == Approx(23));
    require(v13[8] == Approx(42));
    require(v13[9] == Approx(23));
    require(v13[10] == Approx(32));
    require(v13[11] == Approx(23));
    require(v13[12] == Approx(42));
    require(v13[13] == Approx(23));
    require(v13[14] == Approx(32));
    require(v13[15] == Approx(23));

    const simd_fvec16 v14 = {42.0f, 0, 24.0f, 0, 42.0f, 0, 24.0f, 0, 42.0f, 0, 24.0f, 0, 42.0f, 0, 24.0f, 0};
    simd_fvec16 v15 = {0, 12.0f, 0, 0, 0, 12.0f, 0, 0, 0, 12.0f, 0, 0, 0, 12.0f, 0, 0};

    v15 |= v14;

    require(v15[0] == 42.0f);
    require(v15[1] == 12.0f);
    require(v15[2] == 24.0f);
    require(v15[3] == 0);
    require(v15[4] == 42.0f);
    require(v15[5] == 12.0f);
    require(v15[6] == 24.0f);
    require(v15[7] == 0);
    require(v15[8] == 42.0f);
    require(v15[9] == 12.0f);
    require(v15[10] == 24.0f);
    require(v15[11] == 0);
    require(v15[12] == 42.0f);
    require(v15[13] == 12.0f);
    require(v15[14] == 24.0f);
    require(v15[15] == 0);

    std::cout << "OK" << std::endl;
}

{
    std::cout << "Test simd_ivec16 (" << (simd_ivec16::is_native() ? "hard" : "soft") << ") | ";

    simd_ivec16 v1, v2 = {42}, v3 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    require(v2[0] == 42);
    require(v2[1] == 42);
    require(v2[2] == 42);
    require(v2[3] == 42);
    require(v2[4] == 42);
    require(v2[5] == 42);
    require(v2[6] == 42);
    require(v2[7] == 42);
    require(v2[8] == 42);
    require(v2[9] == 42);
    require(v2[10] == 42);
    require(v2[11] == 42);
    require(v2[12] == 42);
    require(v2[13] == 42);
    require(v2[14] == 42);
    require(v2[15] == 42);

    require(v3[0] == 1);
    require(v3[1] == 2);
    require(v3[2] == 3);
    require(v3[3] == 4);
    require(v3[4] == 5);
    require(v3[5] == 6);
    require(v3[6] == 7);
    require(v3[7] == 8);
    require(v3[8] == 9);
    require(v3[9] == 10);
    require(v3[10] == 11);
    require(v3[11] == 12);
    require(v3[12] == 13);
    require(v3[13] == 14);
    require(v3[14] == 15);
    require(v3[15] == 16);

    require(v2.get<0>() == 42);
    require(v2.get<1>() == 42);
    require(v2.get<2>() == 42);
    require(v2.get<3>() == 42);
    require(v2.get<4>() == 42);
    require(v2.get<5>() == 42);
    require(v2.get<6>() == 42);
    require(v2.get<7>() == 42);
    require(v2.get<8>() == 42);
    require(v2.get<9>() == 42);
    require(v2.get<10>() == 42);
    require(v2.get<11>() == 42);
    require(v2.get<12>() == 42);
    require(v2.get<13>() == 42);
    require(v2.get<14>() == 42);
    require(v2.get<15>() == 42);

    require(v3.get<0>() == 1);
    require(v3.get<1>() == 2);
    require(v3.get<2>() == 3);
    require(v3.get<3>() == 4);
    require(v3.get<4>() == 5);
    require(v3.get<5>() == 6);
    require(v3.get<6>() == 7);
    require(v3.get<7>() == 8);
    require(v3.get<8>() == 9);
    require(v3.get<9>() == 10);
    require(v3.get<10>() == 11);
    require(v3.get<11>() == 12);
    require(v3.get<12>() == 13);
    require(v3.get<13>() == 14);
    require(v3.get<14>() == 15);
    require(v3.get<15>() == 16);

    simd_ivec16 v4(v2), v5 = v3;

    require(v4[0] == 42);
    require(v4[1] == 42);
    require(v4[2] == 42);
    require(v4[3] == 42);
    require(v4[4] == 42);
    require(v4[5] == 42);
    require(v4[6] == 42);
    require(v4[7] == 42);
    require(v4[8] == 42);
    require(v4[9] == 42);
    require(v4[10] == 42);
    require(v4[11] == 42);
    require(v4[12] == 42);
    require(v4[13] == 42);
    require(v4[14] == 42);
    require(v4[15] == 42);

    require(v5[0] == 1);
    require(v5[1] == 2);
    require(v5[2] == 3);
    require(v5[3] == 4);
    require(v5[4] == 5);
    require(v5[5] == 6);
    require(v5[6] == 7);
    require(v5[7] == 8);
    require(v5[8] == 9);
    require(v5[9] == 10);
    require(v5[10] == 11);
    require(v5[11] == 12);
    require(v5[12] == 13);
    require(v5[13] == 14);
    require(v5[14] == 15);
    require(v5[15] == 16);

    v1 = v5;

    require(v1[0] == 1);
    require(v1[1] == 2);
    require(v1[2] == 3);
    require(v1[3] == 4);
    require(v1[4] == 5);
    require(v1[5] == 6);
    require(v1[6] == 7);
    require(v1[7] == 8);
    require(v1[8] == 9);
    require(v1[9] == 10);
    require(v1[10] == 11);
    require(v1[11] == 12);
    require(v1[12] == 13);
    require(v1[13] == 14);
    require(v1[14] == 15);
    require(v1[15] == 16);

    v1 = {1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2};
    v2 = {4, 5, 6, 7, 8, 10, 12, 1, 4, 5, 6, 7, 8, 10, 12, 1};

    v3 = v1 + v2;
    v4 = v1 - v2;
    v5 = v1 * v2;
    auto v6 = v1 / v2;
    auto v66 = -v1;

    require(v3[0] == 5);
    require(v3[1] == 7);
    require(v3[2] == 9);
    require(v3[3] == 11);
    require(v3[4] == 13);
    require(v3[5] == 14);
    require(v3[6] == 15);
    require(v3[7] == 3);
    require(v3[8] == 5);
    require(v3[9] == 7);
    require(v3[10] == 9);
    require(v3[11] == 11);
    require(v3[12] == 13);
    require(v3[13] == 14);
    require(v3[14] == 15);
    require(v3[15] == 3);

    require(v4[0] == -3);
    require(v4[1] == -3);
    require(v4[2] == -3);
    require(v4[3] == -3);
    require(v4[4] == -3);
    require(v4[5] == -6);
    require(v4[6] == -9);
    require(v4[7] == 1);
    require(v4[8] == -3);
    require(v4[9] == -3);
    require(v4[10] == -3);
    require(v4[11] == -3);
    require(v4[12] == -3);
    require(v4[13] == -6);
    require(v4[14] == -9);
    require(v4[15] == 1);

    require(v5[0] == 4);
    require(v5[1] == 10);
    require(v5[2] == 18);
    require(v5[3] == 28);
    require(v5[4] == 40);
    require(v5[5] == 40);
    require(v5[6] == 36);
    require(v5[7] == 2);
    require(v5[8] == 4);
    require(v5[9] == 10);
    require(v5[10] == 18);
    require(v5[11] == 28);
    require(v5[12] == 40);
    require(v5[13] == 40);
    require(v5[14] == 36);
    require(v5[15] == 2);

    require(v6[0] == 0);
    require(v6[1] == 0);
    require(v6[2] == 0);
    require(v6[3] == 0);
    require(v6[4] == 0);
    require(v6[5] == 0);
    require(v6[6] == 0);
    require(v6[7] == 2);
    require(v6[8] == 0);
    require(v6[9] == 0);
    require(v6[10] == 0);
    require(v6[11] == 0);
    require(v6[12] == 0);
    require(v6[13] == 0);
    require(v6[14] == 0);
    require(v6[15] == 2);

    require(v66[0] == -1);
    require(v66[1] == -2);
    require(v66[2] == -3);
    require(v66[3] == -4);
    require(v66[4] == -5);
    require(v66[5] == -4);
    require(v66[6] == -3);
    require(v66[7] == -2);
    require(v66[8] == -1);
    require(v66[9] == -2);
    require(v66[10] == -3);
    require(v66[11] == -4);
    require(v66[12] == -5);
    require(v66[13] == -4);
    require(v66[14] == -3);
    require(v66[15] == -2);

    static const int gather_source[] = {0, 42, 0, 0, 12, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 23, 0, 32,
                                        0, 42, 0, 0, 12, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 23, 0, 32,
                                        0, 42, 0, 0, 12, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 23, 0, 32,
                                        0, 42, 0, 0, 12, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 23, 0, 32};

    const simd_ivec16 v9i = {-1, 2, 6, 13, -1, 2, 6, 13, -1, 2, 6, 13, -1, 2, 6, 13};
    const simd_ivec16 v9 = gather<1>(gather_source + 2, v9i);

    require(v9[0] == 42);
    require(v9[1] == 12);
    require(v9[2] == 11);
    require(v9[3] == 23);
    require(v9[4] == 42);
    require(v9[5] == 12);
    require(v9[6] == 11);
    require(v9[7] == 23);
    require(v9[8] == 42);
    require(v9[9] == 12);
    require(v9[10] == 11);
    require(v9[11] == 23);
    require(v9[12] == 42);
    require(v9[13] == 12);
    require(v9[14] == 11);
    require(v9[15] == 23);

    const simd_ivec16 v10i = {-1, 6, 7, 6, -1, 6, 7, 6, -1, 6, 7, 6, -1, 6, 7, 6};
    const simd_ivec16 v10 = gather<2 /* Scale */>(gather_source + 3, v10i);

    require(v10[0] == 42);
    require(v10[1] == 23);
    require(v10[2] == 32);
    require(v10[3] == 23);
    require(v10[4] == 42);
    require(v10[5] == 23);
    require(v10[6] == 32);
    require(v10[7] == 23);
    require(v10[8] == 42);
    require(v10[9] == 23);
    require(v10[10] == 32);
    require(v10[11] == 23);
    require(v10[12] == 42);
    require(v10[13] == 23);
    require(v10[14] == 32);
    require(v10[15] == 23);

    const simd_ivec16 v11 = {-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0};
    simd_ivec16 v12 = {0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0};

    v12 |= v11;

    require(v12[0] == -1);
    require(v12[1] == -1);
    require(v12[2] == -1);
    require(v12[3] == 0);
    require(v12[4] == -1);
    require(v12[5] == -1);
    require(v12[6] == -1);
    require(v12[7] == 0);
    require(v12[8] == -1);
    require(v12[9] == -1);
    require(v12[10] == -1);
    require(v12[11] == 0);
    require(v12[12] == -1);
    require(v12[13] == -1);
    require(v12[14] == -1);
    require(v12[15] == 0);

    const simd_ivec16 v13 = {-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0};
    simd_ivec16 v14 = {0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0};

    v14 &= v13;

    require(v14[0] == 0);
    require(v14[1] == 0);
    require(v14[2] == 0);
    require(v14[3] == 0);
    require(v14[4] == 0);
    require(v14[5] == 0);
    require(v14[6] == 0);
    require(v14[7] == 0);
    require(v14[8] == 0);
    require(v14[9] == 0);
    require(v14[10] == 0);
    require(v14[11] == 0);
    require(v14[12] == 0);
    require(v14[13] == 0);
    require(v14[14] == 0);
    require(v14[15] == 0);

    const simd_ivec16 v15 = {-2147483647, 1, -42, 42, -2147483647, 1, -42, 42,
                             -2147483647, 1, -42, 42, -2147483647, 1, -42, 42};
    const simd_ivec16 v16 = srai(v15, 31);
    require((v16 != simd_ivec16{-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0}).all_zeros());

    std::cout << "OK" << std::endl;
}