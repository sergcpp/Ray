using namespace Ray::NS;

{
    printf("Test fvec4  (%s)\t| ", fvec4::is_native() ? "hard" : "soft");

    fvec4 v1, v2 = {42.0f}, v3 = {1.0f, 2.0f, 3.0f, 4.0f};

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

    fvec4 v4(v2), v5 = v3;

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
    alignas(alignof(fvec4)) float aligned_array[] = {0.0f, 2.0f, 30.0f, 14.0f};

    auto v7 = fvec4{&unaligned_array[0]}, v8 = fvec4{&aligned_array[0], vector_aligned};

    require(v7[0] == 0.0f);
    require(v7[1] == 2.0f);
    require(v7[2] == 30.0f);
    require(v7[3] == 14.0f);

    require(v8[0] == 0.0f);
    require(v8[1] == 2.0f);
    require(v8[2] == 30.0f);
    require(v8[3] == 14.0f);

    v5.store_to(&unaligned_array[0]);
    v1.store_to(&aligned_array[0], vector_aligned);

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
    fvec4 v6 = v1 / v2;
    fvec4 v66 = -v1;
    fvec4 v666 = normalize(v1);
    float v1_len;
    fvec4 v6666 = normalize_len(v1, v1_len);

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

    require(v666[0] == Approx(0.182574183));
    require(v666[1] == Approx(0.365148365));
    require(v666[2] == Approx(0.547722518));
    require(v666[3] == Approx(0.730296731));

    require(v1_len == Approx(5.47722578));
    require(v6666[0] == Approx(0.182574183));
    require(v6666[1] == Approx(0.365148365));
    require(v6666[2] == Approx(0.547722518));
    require(v6666[3] == Approx(0.730296731));

    v5 = sqrt(v5);

    require(v5[0] == Approx(2));
    require(v5[1] == Approx(3.1623));
    require(v5[2] == Approx(4.2426));
    require(v5[3] == Approx(5.2915));

    fvec4 v55 = fract(v5);

    require(v55[0] == Approx(0));
    require(v55[1] == Approx(0.1623));
    require(v55[2] == Approx(0.2426));
    require(v55[3] == Approx(0.2915));

    fvec4 v9 = {3.0f, 6.0f, 7.0f, 6.0f};

    require(hsum(v9) == Approx(22.0f));

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

    static const float gather_source[] = {0, 42.0f, 0, 0, 12.0f, 0, 0, 0, 11.0f, 0, 0, 0, 0, 0, 0, 23.0f, 0, 0};

    const ivec4 v12i = {-1, 2, 6, 13};
    const fvec4 v12 = gather(gather_source + 2, v12i);
    const fvec4 v12_masked = gather(fvec4{69}, gather_source + 2, ivec4{-1, 0, -1, 0}, v12i);

    require(v12[0] == Approx(42));
    require(v12[1] == Approx(12));
    require(v12[2] == Approx(11));
    require(v12[3] == Approx(23));

    require(v12_masked[0] == Approx(42));
    require(v12_masked[1] == Approx(69));
    require(v12_masked[2] == Approx(11));
    require(v12_masked[3] == Approx(69));

    float scatter_destination[18] = {};
    scatter(scatter_destination + 2, v12i, v12);

    require(memcmp(gather_source, scatter_destination, sizeof(gather_source)) == 0);

    const ivec4 scatter_mask = {-1, 0, 0, -1};

    float masked_scatter_destination[] = {1, -1, 2, 3, -1, 4, 5, 6, -1, 7, 8, 9, 1, 2, 3, -1, 4, 0};
    static const float masked_scatter_expected[] = {1, 42.0f, 2, 3, -1, 4, 5, 6, -1, 7, 8, 9, 1, 2, 3, 23.0f, 4, 0};

    scatter(masked_scatter_destination + 2, scatter_mask, v12i, v12);

    require(memcmp(masked_scatter_destination, masked_scatter_expected, sizeof(masked_scatter_destination)) == 0);

    const fvec4 v14 = {42.0f, 0, 24.0f, 0};
    fvec4 v15 = {0, 12.0f, 0, 0};

    v15 |= v14;

    require(v15[0] == 42.0f);
    require(v15[1] == 12.0f);
    require(v15[2] == 24.0f);
    require(v15[3] == 0);

    const fvec4 v16 = {3, 1, 4, 1};
    const fvec4 v17 = inclusive_scan(v16);

    require(v17[0] == 3.0f);
    require(v17[1] == 4.0f);
    require(v17[2] == 8.0f);
    require(v17[3] == 9.0f);

    const ivec4 vmask = {-1, 0, 0, -1};

    fvec4 v18 = v3;
    where(vmask, v18) = v2;

    const fvec4 v19 = select(vmask, v2, v3);

    require(v18.get<0>() == 4.0f);
    require(v18.get<1>() == 7.0f);
    require(v18.get<2>() == 9.0f);
    require(v18.get<3>() == 7.0f);

    require(v19.get<0>() == 4.0f);
    require(v19.get<1>() == 7.0f);
    require(v19.get<2>() == 9.0f);
    require(v19.get<3>() == 7.0f);

    printf("OK\n");
}

{
    printf("Test ivec4  (%s)\t| ", ivec4::is_native() ? "hard" : "soft");

    ivec4 v1, v2 = {42}, v3 = {1, 2, 3, 4};

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

    ivec4 v4(v2), v5 = v3;

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
    alignas(alignof(ivec4)) int aligned_array[] = {0, 2, 30, 14};

    auto v7 = ivec4{&unaligned_array[0]}, v8 = ivec4{&aligned_array[0], vector_aligned};

    require(v7[0] == 0);
    require(v7[1] == 2);
    require(v7[2] == 30);
    require(v7[3] == 14);

    require(v8[0] == 0);
    require(v8[1] == 2);
    require(v8[2] == 30);
    require(v8[3] == 14);

    v5.store_to(&unaligned_array[0]);
    v1.store_to(&aligned_array[0], vector_aligned);

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
    ivec4 v6 = v1 / v2;
    ivec4 v66 = -v1;

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

    static const int gather_source[] = {0, 42, 0, 0, 12, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 23, 0, 0};

    const ivec4 v9i = {-1, 2, 6, 13};
    const ivec4 v9 = gather(gather_source + 2, v9i);
    const ivec4 v9_masked = gather(ivec4{69}, gather_source + 2, ivec4{-1, 0, -1, 0}, v9i);

    require(v9[0] == 42);
    require(v9[1] == 12);
    require(v9[2] == 11);
    require(v9[3] == 23);

    require(v9_masked[0] == 42);
    require(v9_masked[1] == 69);
    require(v9_masked[2] == 11);
    require(v9_masked[3] == 69);

    ivec4 v9_ = {3, 6, 7, 6};
    require(hsum(v9_) == 22);

    int scatter_destination[18] = {};
    scatter(scatter_destination + 2, v9i, v9);

    require(memcmp(gather_source, scatter_destination, sizeof(gather_source)) == 0);

    const ivec4 scatter_mask = {-1, 0, 0, -1};

    int masked_scatter_destination[] = {1, -1, 2, 3, -1, 4, 5, 6, -1, 7, 8, 9, 1, 2, 3, -1, 4, 0};
    static const int masked_scatter_expected[] = {1, 42, 2, 3, -1, 4, 5, 6, -1, 7, 8, 9, 1, 2, 3, 23, 4, 0};

    scatter(masked_scatter_destination + 2, scatter_mask, v9i, v9);

    require(memcmp(masked_scatter_destination, masked_scatter_expected, sizeof(masked_scatter_destination)) == 0);

    const ivec4 v11 = {-1, 0, -1, 0};
    ivec4 v12 = {0, -1, 0, 0};

    v12 |= v11;

    require(v12[0] == -1);
    require(v12[1] == -1);
    require(v12[2] == -1);
    require(v12[3] == 0);

    const ivec4 v13 = {-1, 0, -1, 0};
    ivec4 v14 = {0, -1, 0, 0};

    v14 &= v13;

    require(v14[0] == 0);
    require(v14[1] == 0);
    require(v14[2] == 0);
    require(v14[3] == 0);

    const ivec4 v15 = {-2147483647, 1, -42, 42};
    const ivec4 v16 = srai(v15, 31);
    require((v16 != ivec4{-1, 0, -1, 0}).all_zeros());

    const ivec4 v17 = {3, 1, 4, 1};
    const ivec4 v18 = inclusive_scan(v17);

    require(v18[0] == 3);
    require(v18[1] == 4);
    require(v18[2] == 8);
    require(v18[3] == 9);

    const uvec4 vmask = {0xffffffff, 0, 0, 0xffffffff};

    ivec4 v19 = v3;
    where(vmask, v19) = v2;

    const ivec4 v20 = select(vmask, v2, v3);

    require(v19.get<0>() == 4);
    require(v19.get<1>() == 7);
    require(v19.get<2>() == 9);
    require(v19.get<3>() == 7);

    require(v20.get<0>() == 4);
    require(v20.get<1>() == 7);
    require(v20.get<2>() == 9);
    require(v20.get<3>() == 7);

    printf("OK\n");
}

{
    printf("Test uvec4  (%s)\t| ", uvec4::is_native() ? "hard" : "soft");

    uvec4 v1, v2 = {42}, v3 = {1, 2, 3, 4};

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

    uvec4 v4(v2), v5 = v3;

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

    unsigned unaligned_array[] = {0, 2, 30, 14};
    alignas(alignof(uvec4)) unsigned aligned_array[] = {0, 2, 30, 14};

    auto v7 = uvec4{&unaligned_array[0]}, v8 = uvec4{&aligned_array[0], vector_aligned};

    require(v7[0] == 0);
    require(v7[1] == 2);
    require(v7[2] == 30);
    require(v7[3] == 14);

    require(v8[0] == 0);
    require(v8[1] == 2);
    require(v8[2] == 30);
    require(v8[3] == 14);

    v5.store_to(&unaligned_array[0]);
    v1.store_to(&aligned_array[0], vector_aligned);

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

    require(!v3.all_zeros());
    require(v6.all_zeros());

    static const unsigned gather_source[] = {0, 42, 0, 0, 12, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 23, 0, 0};

    const ivec4 v9i = {-1, 2, 6, 13};
    const uvec4 v9 = gather(gather_source + 2, v9i);
    const uvec4 v9_masked = gather(uvec4{69}, gather_source + 2, ivec4{-1, 0, -1, 0}, v9i);

    require(v9[0] == 42);
    require(v9[1] == 12);
    require(v9[2] == 11);
    require(v9[3] == 23);

    require(v9_masked[0] == 42);
    require(v9_masked[1] == 69);
    require(v9_masked[2] == 11);
    require(v9_masked[3] == 69);

    uvec4 v9_ = {3, 6, 7, 6};
    require(hsum(v9_) == 22);

    unsigned scatter_destination[18] = {};
    scatter(scatter_destination + 2, v9i, v9);

    require(memcmp(gather_source, scatter_destination, sizeof(gather_source)) == 0);

    const ivec4 scatter_mask = {-1, 0, 0, -1};

    unsigned masked_scatter_destination[] = {1, 0xffffffff, 2, 3, 0xffffffff, 4, 5,          6, 0xffffffff,
                                             7, 8,          9, 1, 2,          3, 0xffffffff, 4, 0};
    static const unsigned masked_scatter_expected[] = {1, 42, 2, 3, 0xffffffff, 4, 5,  6, 0xffffffff,
                                                       7, 8,  9, 1, 2,          3, 23, 4, 0};

    scatter(masked_scatter_destination + 2, scatter_mask, v9i, v9);

    require(memcmp(masked_scatter_destination, masked_scatter_expected, sizeof(masked_scatter_destination)) == 0);

    const uvec4 v11 = {0xffffffff, 0, 0xffffffff, 0};
    uvec4 v12 = {0, 0xffffffff, 0, 0};

    v12 |= v11;

    require(v12[0] == 0xffffffff);
    require(v12[1] == 0xffffffff);
    require(v12[2] == 0xffffffff);
    require(v12[3] == 0);

    const uvec4 v13 = {0xffffffff, 0, 0xffffffff, 0};
    uvec4 v14 = {0, 0xffffffff, 0, 0};

    v14 &= v13;

    require(v14[0] == 0);
    require(v14[1] == 0);
    require(v14[2] == 0);
    require(v14[3] == 0);

    const uvec4 v17 = {3, 1, 4, 1};
    const uvec4 v18 = inclusive_scan(v17);

    require(v18[0] == 3);
    require(v18[1] == 4);
    require(v18[2] == 8);
    require(v18[3] == 9);

    const ivec4 vmask = {-1, 0, 0, -1};

    uvec4 v19 = v3;
    where(vmask, v19) = v2;

    const uvec4 v20 = select(vmask, v2, v3);

    require(v19.get<0>() == 4);
    require(v19.get<1>() == 7);
    require(v19.get<2>() == 9);
    require(v19.get<3>() == 7);

    require(v20.get<0>() == 4);
    require(v20.get<1>() == 7);
    require(v20.get<2>() == 9);
    require(v20.get<3>() == 7);

    printf("OK\n");
}

{
    printf("Test fvec8  (%s)\t| ", fvec8::is_native() ? "hard" : "soft");

    fvec8 v1, v2 = {42.0f}, v3 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

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

    fvec8 v4(v2), v5 = v3;

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
    fvec8 v6 = v1 / v2;
    fvec8 v66 = -v1;
    fvec8 v666 = normalize(v1);
    float v1_len;
    fvec8 v6666 = normalize_len(v1, v1_len);

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

    require(v666[0] == Approx(0.109108940));
    require(v666[1] == Approx(0.218217880));
    require(v666[2] == Approx(0.327326834));
    require(v666[3] == Approx(0.436435759));
    require(v666[4] == Approx(0.545544684));
    require(v666[5] == Approx(0.436435759));
    require(v666[6] == Approx(0.327326834));
    require(v666[7] == Approx(0.218217880));

    require(v1_len == Approx(9.16515160));
    require(v6666[0] == Approx(0.109108940));
    require(v6666[1] == Approx(0.218217880));
    require(v6666[2] == Approx(0.327326834));
    require(v6666[3] == Approx(0.436435759));
    require(v6666[4] == Approx(0.545544684));
    require(v6666[5] == Approx(0.436435759));
    require(v6666[6] == Approx(0.327326834));
    require(v6666[7] == Approx(0.218217880));

    v5 = sqrt(v5);

    require(v5[0] == Approx(2));
    require(v5[1] == Approx(3.1623));
    require(v5[2] == Approx(4.2426));
    require(v5[3] == Approx(5.2915));
    require(v5[4] == Approx(6.3246));
    require(v5[5] == Approx(6.3246));
    require(v5[6] == Approx(6));
    require(v5[7] == Approx(1.4142));

    fvec8 v55 = fract(v5);

    require(v55[0] == Approx(0));
    require(v55[1] == Approx(0.1623));
    require(v55[2] == Approx(0.2426));
    require(v55[3] == Approx(0.2915));
    require(v55[4] == Approx(0.3246));
    require(v55[5] == Approx(0.3246));
    require(v55[6] == Approx(0));
    require(v55[7] == Approx(0.4142));

    fvec8 v9 = {3.0f, 6.0f, 7.0f, 6.0f, 2.0f, 12.0f, 18.0f, 0.0f};
    require(hsum(v9) == Approx(54.0f));

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

    static const float gather_source[] = {0, 42.0f, 0, 0, 12.0f, 0, 0, 0, 11.0f, 0, 0, 0, 0, 0, 0, 23.0f, 0, 0,
                                          0, 42.0f, 0, 0, 12.0f, 0, 0, 0, 11.0f, 0, 0, 0, 0, 0, 0, 23.0f, 0, 0};

    const ivec8 v12i = {-1, 2, 6, 13, 17, 20, 24, 31};
    const fvec8 v12 = gather(gather_source + 2, v12i);
    const fvec8 v12_masked =
        gather(fvec8{69}, gather_source + 2, ivec8{-1, 0, -1, 0, -1, 0, -1, 0}, v12i);

    require(v12[0] == Approx(42));
    require(v12[1] == Approx(12));
    require(v12[2] == Approx(11));
    require(v12[3] == Approx(23));
    require(v12[4] == Approx(42));
    require(v12[5] == Approx(12));
    require(v12[6] == Approx(11));
    require(v12[7] == Approx(23));

    require(v12_masked[0] == Approx(42));
    require(v12_masked[1] == Approx(69));
    require(v12_masked[2] == Approx(11));
    require(v12_masked[3] == Approx(69));
    require(v12_masked[4] == Approx(42));
    require(v12_masked[5] == Approx(69));
    require(v12_masked[6] == Approx(11));
    require(v12_masked[7] == Approx(69));

    float scatter_destination[36] = {};
    scatter(scatter_destination + 2, v12i, v12);

    require(memcmp(gather_source, scatter_destination, sizeof(gather_source)) == 0);

    const ivec8 scatter_mask = {-1, 0, 0, -1, -1, 0, 0, -1};

    float masked_scatter_destination[] = {1, -1, 2, 3, -1, 4, 5, 6, -1, 7, 8, 9, 1, 2, 3, -1, 4, 0,
                                          1, -1, 2, 3, -1, 4, 5, 6, -1, 7, 8, 9, 1, 2, 3, -1, 4, 0};
    static const float masked_scatter_expected[] = {1, 42.0f, 2, 3, -1, 4, 5, 6, -1, 7, 8, 9, 1, 2, 3, 23.0f, 4, 0,
                                                    1, 42.0f, 2, 3, -1, 4, 5, 6, -1, 7, 8, 9, 1, 2, 3, 23.0f, 4, 0};

    scatter(masked_scatter_destination + 2, scatter_mask, v12i, v12);

    require(memcmp(masked_scatter_destination, masked_scatter_expected, sizeof(masked_scatter_destination)) == 0);

    const fvec8 v14 = {42.0f, 0, 24.0f, 0, 42.0f, 0, 24.0f, 0};
    fvec8 v15 = {0, 12.0f, 0, 0, 0, 12.0f, 0, 0};

    v15 |= v14;

    require(v15[0] == 42.0f);
    require(v15[1] == 12.0f);
    require(v15[2] == 24.0f);
    require(v15[3] == 0);
    require(v15[4] == 42.0f);
    require(v15[5] == 12.0f);
    require(v15[6] == 24.0f);
    require(v15[7] == 0);

    const fvec8 v16 = {3, 1, 4, 1, 3, 1, 4, 1};
    const fvec8 v17 = inclusive_scan(v16);

    require(v17[0] == 3.0f);
    require(v17[1] == 4.0f);
    require(v17[2] == 8.0f);
    require(v17[3] == 9.0f);
    require(v17[4] == 12.0f);
    require(v17[5] == 13.0f);
    require(v17[6] == 17.0f);
    require(v17[7] == 18.0f);

    const ivec8 vmask = {-1, 0, 0, -1, -1, 0, 0, -1};

    fvec8 v18 = v3;
    where(vmask, v18) = v2;

    const fvec8 v19 = select(vmask, v2, v3);

    require(v18.get<0>() == 4.0f);
    require(v18.get<1>() == 7.0f);
    require(v18.get<2>() == 9.0f);
    require(v18.get<3>() == 7.0f);
    require(v18.get<4>() == 8.0f);
    require(v18.get<5>() == 14.0f);
    require(v18.get<6>() == 15.0f);
    require(v18.get<7>() == 1.0f);

    require(v19.get<0>() == 4.0f);
    require(v19.get<1>() == 7.0f);
    require(v19.get<2>() == 9.0f);
    require(v19.get<3>() == 7.0f);
    require(v19.get<4>() == 8.0f);
    require(v19.get<5>() == 14.0f);
    require(v19.get<6>() == 15.0f);
    require(v19.get<7>() == 1.0f);

    printf("OK\n");
}

{
    printf("Test ivec8  (%s)\t| ", ivec8::is_native() ? "hard" : "soft");

    ivec8 v1, v2 = {42}, v3 = {1, 2, 3, 4, 5, 6, 7, 8};

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

    ivec8 v4(v2), v5 = v3;

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
    ivec8 v6 = v1 / v2;
    ivec8 v66 = -v1;

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

    static const int gather_source[] = {0, 42, 0, 0, 12, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 23, 0, 0,
                                        0, 42, 0, 0, 12, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 23, 0, 0};

    const ivec8 v9i = {-1, 2, 6, 13, 17, 20, 24, 31};
    const ivec8 v9 = gather(gather_source + 2, v9i);
    const ivec8 v9_masked = gather(ivec8{69}, gather_source + 2, ivec8{-1, 0, -1, 0, -1, 0, -1, 0}, v9i);

    require(v9[0] == 42);
    require(v9[1] == 12);
    require(v9[2] == 11);
    require(v9[3] == 23);
    require(v9[4] == 42);
    require(v9[5] == 12);
    require(v9[6] == 11);
    require(v9[7] == 23);

    require(v9_masked[0] == 42);
    require(v9_masked[1] == 69);
    require(v9_masked[2] == 11);
    require(v9_masked[3] == 69);
    require(v9_masked[4] == 42);
    require(v9_masked[5] == 69);
    require(v9_masked[6] == 11);
    require(v9_masked[7] == 69);

    ivec8 v9_ = {3, 6, 7, 6, 2, 12, 18, 0};
    require(hsum(v9_) == 54);

    int scatter_destination[36] = {};
    scatter(scatter_destination + 2, v9i, v9);

    require(memcmp(gather_source, scatter_destination, sizeof(gather_source)) == 0);

    const ivec8 scatter_mask = {-1, 0, 0, -1, -1, 0, 0, -1};

    int masked_scatter_destination[] = {1, -1, 2, 3, -1, 4, 5, 6, -1, 7, 8, 9, 1, 2, 3, -1, 4, 0,
                                        1, -1, 2, 3, -1, 4, 5, 6, -1, 7, 8, 9, 1, 2, 3, -1, 4, 0};
    static const int masked_scatter_expected[] = {1, 42, 2, 3, -1, 4, 5, 6, -1, 7, 8, 9, 1, 2, 3, 23, 4, 0,
                                                  1, 42, 2, 3, -1, 4, 5, 6, -1, 7, 8, 9, 1, 2, 3, 23, 4, 0};

    scatter(masked_scatter_destination + 2, scatter_mask, v9i, v9);

    require(memcmp(masked_scatter_destination, masked_scatter_expected, sizeof(masked_scatter_destination)) == 0);

    const ivec8 v11 = {-1, 0, -1, 0, -1, 0, -1, 0};
    ivec8 v12 = {0, -1, 0, 0, 0, -1, 0, 0};

    v12 |= v11;

    require(v12[0] == -1);
    require(v12[1] == -1);
    require(v12[2] == -1);
    require(v12[3] == 0);
    require(v12[4] == -1);
    require(v12[5] == -1);
    require(v12[6] == -1);
    require(v12[7] == 0);

    const ivec8 v13 = {-1, 0, -1, 0, -1, 0, -1, 0};
    ivec8 v14 = {0, -1, 0, 0, 0, -1, 0, 0};

    v14 &= v13;

    require(v14[0] == 0);
    require(v14[1] == 0);
    require(v14[2] == 0);
    require(v14[3] == 0);
    require(v14[4] == 0);
    require(v14[5] == 0);
    require(v14[6] == 0);
    require(v14[7] == 0);

    const ivec8 v15 = {-2147483647, 1, -42, 42, -2147483647, 1, -42, 42};
    const ivec8 v16 = srai(v15, 31);
    require((v16 != ivec8{-1, 0, -1, 0, -1, 0, -1, 0}).all_zeros());

    const ivec8 v17 = {3, 1, 4, 1, 3, 1, 4, 1};
    const ivec8 v18 = inclusive_scan(v17);

    require(v18[0] == 3);
    require(v18[1] == 4);
    require(v18[2] == 8);
    require(v18[3] == 9);
    require(v18[4] == 12);
    require(v18[5] == 13);
    require(v18[6] == 17);
    require(v18[7] == 18);

    const uvec8 vmask = {0xffffffff, 0, 0, 0xffffffff, 0xffffffff, 0, 0, 0xffffffff};

    ivec8 v19 = v3;
    where(vmask, v19) = v2;

    const ivec8 v20 = select(vmask, v2, v3);

    require(v19.get<0>() == 4);
    require(v19.get<1>() == 7);
    require(v19.get<2>() == 9);
    require(v19.get<3>() == 7);
    require(v19.get<4>() == 8);
    require(v19.get<5>() == 14);
    require(v19.get<6>() == 15);
    require(v19.get<7>() == 1);

    require(v20.get<0>() == 4);
    require(v20.get<1>() == 7);
    require(v20.get<2>() == 9);
    require(v20.get<3>() == 7);
    require(v20.get<4>() == 8);
    require(v20.get<5>() == 14);
    require(v20.get<6>() == 15);
    require(v20.get<7>() == 1);

    printf("OK\n");
}

{
    printf("Test uvec8  (%s)\t| ", uvec8::is_native() ? "hard" : "soft");

    uvec8 v1, v2 = {42}, v3 = {1, 2, 3, 4, 5, 6, 7, 8};

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

    uvec8 v4(v2), v5 = v3;

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

    static const unsigned gather_source[] = {0, 42, 0, 0, 12, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 23, 0, 0,
                                             0, 42, 0, 0, 12, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 23, 0, 0};

    const ivec8 v9i = {-1, 2, 6, 13, 17, 20, 24, 31};
    const uvec8 v9 = gather(gather_source + 2, v9i);
    const uvec8 v9_masked = gather(uvec8{69}, gather_source + 2, ivec8{-1, 0, -1, 0, -1, 0, -1, 0}, v9i);

    require(v9[0] == 42);
    require(v9[1] == 12);
    require(v9[2] == 11);
    require(v9[3] == 23);
    require(v9[4] == 42);
    require(v9[5] == 12);
    require(v9[6] == 11);
    require(v9[7] == 23);

    require(v9_masked[0] == 42);
    require(v9_masked[1] == 69);
    require(v9_masked[2] == 11);
    require(v9_masked[3] == 69);
    require(v9_masked[4] == 42);
    require(v9_masked[5] == 69);
    require(v9_masked[6] == 11);
    require(v9_masked[7] == 69);

    uvec8 v9_ = {3, 6, 7, 6, 2, 12, 18, 0};
    require(hsum(v9_) == 54);

    unsigned scatter_destination[36] = {};
    scatter(scatter_destination + 2, v9i, v9);

    require(memcmp(gather_source, scatter_destination, sizeof(gather_source)) == 0);

    const ivec8 scatter_mask = {-1, 0, 0, -1, -1, 0, 0, -1};

    unsigned masked_scatter_destination[] = {
        1, 0xffffffff, 2, 3, 0xffffffff, 4, 5, 6, 0xffffffff, 7, 8, 9, 1, 2, 3, 0xffffffff, 4, 0,
        1, 0xffffffff, 2, 3, 0xffffffff, 4, 5, 6, 0xffffffff, 7, 8, 9, 1, 2, 3, 0xffffffff, 4, 0};
    static const int masked_scatter_expected[] = {1, 42, 2, 3, -1, 4, 5, 6, -1, 7, 8, 9, 1, 2, 3, 23, 4, 0,
                                                  1, 42, 2, 3, -1, 4, 5, 6, -1, 7, 8, 9, 1, 2, 3, 23, 4, 0};

    scatter(masked_scatter_destination + 2, scatter_mask, v9i, v9);

    require(memcmp(masked_scatter_destination, masked_scatter_expected, sizeof(masked_scatter_destination)) == 0);

    const uvec8 v11 = {0xffffffff, 0, 0xffffffff, 0, 0xffffffff, 0, 0xffffffff, 0};
    uvec8 v12 = {0, 0xffffffff, 0, 0, 0, 0xffffffff, 0, 0};

    v12 |= v11;

    require(v12[0] == -1);
    require(v12[1] == -1);
    require(v12[2] == -1);
    require(v12[3] == 0);
    require(v12[4] == -1);
    require(v12[5] == -1);
    require(v12[6] == -1);
    require(v12[7] == 0);

    const uvec8 v13 = {0xffffffff, 0, 0xffffffff, 0, 0xffffffff, 0, 0xffffffff, 0};
    uvec8 v14 = {0, 0xffffffff, 0, 0, 0, 0xffffffff, 0, 0};

    v14 &= v13;

    require(v14[0] == 0);
    require(v14[1] == 0);
    require(v14[2] == 0);
    require(v14[3] == 0);
    require(v14[4] == 0);
    require(v14[5] == 0);
    require(v14[6] == 0);
    require(v14[7] == 0);

    const uvec8 v17 = {3, 1, 4, 1, 3, 1, 4, 1};
    const uvec8 v18 = inclusive_scan(v17);

    require(v18[0] == 3);
    require(v18[1] == 4);
    require(v18[2] == 8);
    require(v18[3] == 9);
    require(v18[4] == 12);
    require(v18[5] == 13);
    require(v18[6] == 17);
    require(v18[7] == 18);

    const ivec8 vmask = {-1, 0, 0, -1, -1, 0, 0, -1};

    uvec8 v19 = v3;
    where(vmask, v19) = v2;

    const uvec8 v20 = select(vmask, v2, v3);

    require(v19.get<0>() == 4);
    require(v19.get<1>() == 7);
    require(v19.get<2>() == 9);
    require(v19.get<3>() == 7);
    require(v19.get<4>() == 8);
    require(v19.get<5>() == 14);
    require(v19.get<6>() == 15);
    require(v19.get<7>() == 1);

    require(v20.get<0>() == 4);
    require(v20.get<1>() == 7);
    require(v20.get<2>() == 9);
    require(v20.get<3>() == 7);
    require(v20.get<4>() == 8);
    require(v20.get<5>() == 14);
    require(v20.get<6>() == 15);
    require(v20.get<7>() == 1);

    printf("OK\n");
}

//////////////////////////////////////////////////

{
    printf("Test fvec16 (%s)\t| ", fvec16::is_native() ? "hard" : "soft");

    fvec16 v1, v2 = {42.0f}, v3 = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
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

    fvec16 v4(v2), v5 = v3;

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
    fvec16 v6 = v1 / v2;
    fvec16 v66 = -v1;
    fvec16 v666 = normalize(v1);
    float v1_len;
    fvec16 v6666 = normalize_len(v1, v1_len);

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

    require(v666[0] == Approx(0.0771516785));
    require(v666[1] == Approx(0.154303357));
    require(v666[2] == Approx(0.231455028));
    require(v666[3] == Approx(0.308606714));
    require(v666[4] == Approx(0.385758370));
    require(v666[5] == Approx(0.308606714));
    require(v666[6] == Approx(0.231455028));
    require(v666[7] == Approx(0.154303357));
    require(v666[8] == Approx(0.0771516785));
    require(v666[9] == Approx(0.154303357));
    require(v666[10] == Approx(0.231455028));
    require(v666[11] == Approx(0.308606714));
    require(v666[12] == Approx(0.385758370));
    require(v666[13] == Approx(0.308606714));
    require(v666[14] == Approx(0.231455028));
    require(v666[15] == Approx(0.154303357));

    require(v1_len == Approx(12.9614811));
    require(v6666[0] == Approx(0.0771516785));
    require(v6666[1] == Approx(0.154303357));
    require(v6666[2] == Approx(0.231455028));
    require(v6666[3] == Approx(0.308606714));
    require(v6666[4] == Approx(0.385758370));
    require(v6666[5] == Approx(0.308606714));
    require(v6666[6] == Approx(0.231455028));
    require(v6666[7] == Approx(0.154303357));
    require(v6666[8] == Approx(0.0771516785));
    require(v6666[9] == Approx(0.154303357));
    require(v6666[10] == Approx(0.231455028));
    require(v6666[11] == Approx(0.308606714));
    require(v6666[12] == Approx(0.385758370));
    require(v6666[13] == Approx(0.308606714));
    require(v6666[14] == Approx(0.231455028));
    require(v6666[15] == Approx(0.154303357));

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

    fvec16 v55 = fract(v5);

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

    fvec16 v9 = {3.0f, 6.0f, 7.0f, 6.0f, 2.0f, 12.0f, 18.0f, 0.0f,
                      3.0f, 6.0f, 7.0f, 6.0f, 2.0f, 12.0f, 18.0f, 0.0f};
    require(hsum(v9) == Approx(108.0f));

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

    static const float gather_source[] = {0, 42.0f, 0, 0, 12.0f, 0, 0, 0, 11.0f, 0, 0, 0, 0, 0, 0, 23.0f, 0, 0,
                                          0, 42.0f, 0, 0, 12.0f, 0, 0, 0, 11.0f, 0, 0, 0, 0, 0, 0, 23.0f, 0, 0,
                                          0, 42.0f, 0, 0, 12.0f, 0, 0, 0, 11.0f, 0, 0, 0, 0, 0, 0, 23.0f, 0, 0,
                                          0, 42.0f, 0, 0, 12.0f, 0, 0, 0, 11.0f, 0, 0, 0, 0, 0, 0, 23.0f, 0, 0};

    const ivec16 v12i = {-1, 2, 6, 13, 17, 20, 24, 31, 35, 38, 42, 49, 53, 56, 60, 67};
    const fvec16 v12 = gather(gather_source + 2, v12i);
    const fvec16 v12_masked = gather(fvec16{69}, gather_source + 2,
                                          ivec16{-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0}, v12i);

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

    require(v12_masked[0] == Approx(42));
    require(v12_masked[1] == Approx(69));
    require(v12_masked[2] == Approx(11));
    require(v12_masked[3] == Approx(69));
    require(v12_masked[4] == Approx(42));
    require(v12_masked[5] == Approx(69));
    require(v12_masked[6] == Approx(11));
    require(v12_masked[7] == Approx(69));
    require(v12_masked[8] == Approx(42));
    require(v12_masked[9] == Approx(69));
    require(v12_masked[10] == Approx(11));
    require(v12_masked[11] == Approx(69));
    require(v12_masked[12] == Approx(42));
    require(v12_masked[13] == Approx(69));
    require(v12_masked[14] == Approx(11));
    require(v12_masked[15] == Approx(69));

    float scatter_destination[72] = {};
    scatter(scatter_destination + 2, v12i, v12);

    require(memcmp(gather_source, scatter_destination, sizeof(gather_source)) == 0);

    const ivec16 scatter_mask = {-1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1};

    float masked_scatter_destination[] = {1, -1, 2, 3, -1, 4, 5, 6, -1, 7, 8, 9, 1, 2, 3, -1, 4, 0,
                                          1, -1, 2, 3, -1, 4, 5, 6, -1, 7, 8, 9, 1, 2, 3, -1, 4, 0,
                                          1, -1, 2, 3, -1, 4, 5, 6, -1, 7, 8, 9, 1, 2, 3, -1, 4, 0,
                                          1, -1, 2, 3, -1, 4, 5, 6, -1, 7, 8, 9, 1, 2, 3, -1, 4, 0};
    static const float masked_scatter_expected[] = {1, 42.0f, 2, 3, -1, 4, 5, 6, -1, 7, 8, 9, 1, 2, 3, 23.0f, 4, 0,
                                                    1, 42.0f, 2, 3, -1, 4, 5, 6, -1, 7, 8, 9, 1, 2, 3, 23.0f, 4, 0,
                                                    1, 42.0f, 2, 3, -1, 4, 5, 6, -1, 7, 8, 9, 1, 2, 3, 23.0f, 4, 0,
                                                    1, 42.0f, 2, 3, -1, 4, 5, 6, -1, 7, 8, 9, 1, 2, 3, 23.0f, 4, 0};

    scatter(masked_scatter_destination + 2, scatter_mask, v12i, v12);

    require(memcmp(masked_scatter_destination, masked_scatter_expected, sizeof(masked_scatter_destination)) == 0);

    const fvec16 v14 = {42.0f, 0, 24.0f, 0, 42.0f, 0, 24.0f, 0, 42.0f, 0, 24.0f, 0, 42.0f, 0, 24.0f, 0};
    fvec16 v15 = {0, 12.0f, 0, 0, 0, 12.0f, 0, 0, 0, 12.0f, 0, 0, 0, 12.0f, 0, 0};

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

    const fvec16 v16 = {3, 1, 4, 1, 3, 1, 4, 1, 3, 1, 4, 1, 3, 1, 4, 1};
    const fvec16 v17 = inclusive_scan(v16);

    require(v17[0] == 3.0f);
    require(v17[1] == 4.0f);
    require(v17[2] == 8.0f);
    require(v17[3] == 9.0f);
    require(v17[4] == 12.0f);
    require(v17[5] == 13.0f);
    require(v17[6] == 17.0f);
    require(v17[7] == 18.0f);
    require(v17[8] == 21.0f);
    require(v17[9] == 22.0f);
    require(v17[10] == 26.0f);
    require(v17[11] == 27.0f);
    require(v17[12] == 30.0f);
    require(v17[13] == 31.0f);
    require(v17[14] == 35.0f);
    require(v17[15] == 36.0f);

    const ivec16 vmask = {-1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1};

    fvec16 v18 = v3;
    where(vmask, v18) = v2;

    const fvec16 v19 = select(vmask, v2, v3);

    require(v18.get<0>() == 4.0f);
    require(v18.get<1>() == 7.0f);
    require(v18.get<2>() == 9.0f);
    require(v18.get<3>() == 7.0f);
    require(v18.get<4>() == 8.0f);
    require(v18.get<5>() == 14.0f);
    require(v18.get<6>() == 15.0f);
    require(v18.get<7>() == 1.0f);
    require(v18.get<8>() == 4.0f);
    require(v18.get<9>() == 7.0f);
    require(v18.get<10>() == 9.0f);
    require(v18.get<11>() == 7.0f);
    require(v18.get<12>() == 8.0f);
    require(v18.get<13>() == 14.0f);
    require(v18.get<14>() == 15.0f);
    require(v18.get<15>() == 1.0f);

    require(v19.get<0>() == 4.0f);
    require(v19.get<1>() == 7.0f);
    require(v19.get<2>() == 9.0f);
    require(v19.get<3>() == 7.0f);
    require(v19.get<4>() == 8.0f);
    require(v19.get<5>() == 14.0f);
    require(v19.get<6>() == 15.0f);
    require(v19.get<7>() == 1.0f);
    require(v19.get<8>() == 4.0f);
    require(v19.get<9>() == 7.0f);
    require(v19.get<10>() == 9.0f);
    require(v19.get<11>() == 7.0f);
    require(v19.get<12>() == 8.0f);
    require(v19.get<13>() == 14.0f);
    require(v19.get<14>() == 15.0f);
    require(v19.get<15>() == 1.0f);

    printf("OK\n");
}

{
    printf("Test ivec16 (%s)\t| ", ivec16::is_native() ? "hard" : "soft");

    ivec16 v1, v2 = {42}, v3 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

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

    ivec16 v4(v2), v5 = v3;

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
    ivec16 v6 = v1 / v2;
    ivec16 v66 = -v1;

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

    static const int gather_source[] = {0, 42, 0, 0, 12, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 23, 0, 0,
                                        0, 42, 0, 0, 12, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 23, 0, 0,
                                        0, 42, 0, 0, 12, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 23, 0, 0,
                                        0, 42, 0, 0, 12, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 23, 0, 0};

    const ivec16 v9i = {-1, 2, 6, 13, 17, 20, 24, 31, 35, 38, 42, 49, 53, 56, 60, 67};
    const ivec16 v9 = gather(gather_source + 2, v9i);
    const ivec16 v9_masked = gather(ivec16{69}, gather_source + 2,
                                         ivec16{-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0}, v9i);

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

    require(v9_masked[0] == 42);
    require(v9_masked[1] == 69);
    require(v9_masked[2] == 11);
    require(v9_masked[3] == 69);
    require(v9_masked[4] == 42);
    require(v9_masked[5] == 69);
    require(v9_masked[6] == 11);
    require(v9_masked[7] == 69);
    require(v9_masked[8] == 42);
    require(v9_masked[9] == 69);
    require(v9_masked[10] == 11);
    require(v9_masked[11] == 69);
    require(v9_masked[12] == 42);
    require(v9_masked[13] == 69);
    require(v9_masked[14] == 11);
    require(v9_masked[15] == 69);

    fvec16 v9_ = {3, 6, 7, 6, 2, 12, 18, 0, 3, 6, 7, 6, 2, 12, 18, 0};
    require(hsum(v9_) == 108);

    int scatter_destination[72] = {};
    scatter(scatter_destination + 2, v9i, v9);

    require(memcmp(gather_source, scatter_destination, sizeof(gather_source)) == 0);

    const ivec16 scatter_mask = {-1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1};

    int masked_scatter_destination[] = {1, -1, 2, 3, -1, 4, 5, 6, -1, 7, 8, 9, 1, 2, 3, -1, 4, 0,
                                        1, -1, 2, 3, -1, 4, 5, 6, -1, 7, 8, 9, 1, 2, 3, -1, 4, 0,
                                        1, -1, 2, 3, -1, 4, 5, 6, -1, 7, 8, 9, 1, 2, 3, -1, 4, 0,
                                        1, -1, 2, 3, -1, 4, 5, 6, -1, 7, 8, 9, 1, 2, 3, -1, 4, 0};
    static const int masked_scatter_expected[] = {1, 42, 2, 3, -1, 4, 5, 6, -1, 7, 8, 9, 1, 2, 3, 23, 4, 0,
                                                  1, 42, 2, 3, -1, 4, 5, 6, -1, 7, 8, 9, 1, 2, 3, 23, 4, 0,
                                                  1, 42, 2, 3, -1, 4, 5, 6, -1, 7, 8, 9, 1, 2, 3, 23, 4, 0,
                                                  1, 42, 2, 3, -1, 4, 5, 6, -1, 7, 8, 9, 1, 2, 3, 23, 4, 0};

    scatter(masked_scatter_destination + 2, scatter_mask, v9i, v9);

    require(memcmp(masked_scatter_destination, masked_scatter_expected, sizeof(masked_scatter_destination)) == 0);

    const ivec16 v11 = {-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0};
    ivec16 v12 = {0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0};

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

    const ivec16 v13 = {-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0};
    ivec16 v14 = {0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0};

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

    const ivec16 v15 = {-2147483647, 1, -42, 42, -2147483647, 1, -42, 42,
                             -2147483647, 1, -42, 42, -2147483647, 1, -42, 42};
    const ivec16 v16 = srai(v15, 31);
    require((v16 != ivec16{-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0}).all_zeros());

    const ivec16 v17 = {3, 1, 4, 1, 3, 1, 4, 1, 3, 1, 4, 1, 3, 1, 4, 1};
    const ivec16 v18 = inclusive_scan(v17);

    require(v18[0] == 3);
    require(v18[1] == 4);
    require(v18[2] == 8);
    require(v18[3] == 9);
    require(v18[4] == 12);
    require(v18[5] == 13);
    require(v18[6] == 17);
    require(v18[7] == 18);
    require(v18[8] == 21);
    require(v18[9] == 22);
    require(v18[10] == 26);
    require(v18[11] == 27);
    require(v18[12] == 30);
    require(v18[13] == 31);
    require(v18[14] == 35);
    require(v18[15] == 36);

    const uvec16 vmask = {0xffffffff, 0, 0, 0xffffffff, 0xffffffff, 0, 0, 0xffffffff,
                               0xffffffff, 0, 0, 0xffffffff, 0xffffffff, 0, 0, 0xffffffff};

    ivec16 v19 = v3;
    where(vmask, v19) = v2;

    const ivec16 v20 = select(vmask, v2, v3);

    require(v19.get<0>() == 4);
    require(v19.get<1>() == 7);
    require(v19.get<2>() == 9);
    require(v19.get<3>() == 7);
    require(v19.get<4>() == 8);
    require(v19.get<5>() == 14);
    require(v19.get<6>() == 15);
    require(v19.get<7>() == 1);
    require(v19.get<8>() == 4);
    require(v19.get<9>() == 7);
    require(v19.get<10>() == 9);
    require(v19.get<11>() == 7);
    require(v19.get<12>() == 8);
    require(v19.get<13>() == 14);
    require(v19.get<14>() == 15);
    require(v19.get<15>() == 1);

    require(v20.get<0>() == 4);
    require(v20.get<1>() == 7);
    require(v20.get<2>() == 9);
    require(v20.get<3>() == 7);
    require(v20.get<4>() == 8);
    require(v20.get<5>() == 14);
    require(v20.get<6>() == 15);
    require(v20.get<7>() == 1);
    require(v20.get<8>() == 4);
    require(v20.get<9>() == 7);
    require(v20.get<10>() == 9);
    require(v20.get<11>() == 7);
    require(v20.get<12>() == 8);
    require(v20.get<13>() == 14);
    require(v20.get<14>() == 15);
    require(v20.get<15>() == 1);

    printf("OK\n");
}

{
    printf("Test uvec16 (%s)\t| ", uvec16::is_native() ? "hard" : "soft");

    uvec16 v1, v2 = {42}, v3 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

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

    uvec16 v4(v2), v5 = v3;

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

    static const unsigned gather_source[] = {0, 42, 0, 0, 12, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 23, 0, 0,
                                             0, 42, 0, 0, 12, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 23, 0, 0,
                                             0, 42, 0, 0, 12, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 23, 0, 0,
                                             0, 42, 0, 0, 12, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 23, 0, 0};

    const ivec16 v9i = {-1, 2, 6, 13, 17, 20, 24, 31, 35, 38, 42, 49, 53, 56, 60, 67};
    const uvec16 v9 = gather(gather_source + 2, v9i);
    const uvec16 v9_masked = gather(uvec16{69}, gather_source + 2,
                                         ivec16{-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0}, v9i);

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

    require(v9_masked[0] == 42);
    require(v9_masked[1] == 69);
    require(v9_masked[2] == 11);
    require(v9_masked[3] == 69);
    require(v9_masked[4] == 42);
    require(v9_masked[5] == 69);
    require(v9_masked[6] == 11);
    require(v9_masked[7] == 69);
    require(v9_masked[8] == 42);
    require(v9_masked[9] == 69);
    require(v9_masked[10] == 11);
    require(v9_masked[11] == 69);
    require(v9_masked[12] == 42);
    require(v9_masked[13] == 69);
    require(v9_masked[14] == 11);
    require(v9_masked[15] == 69);

    fvec16 v9_ = {3, 6, 7, 6, 2, 12, 18, 0, 3, 6, 7, 6, 2, 12, 18, 0};
    require(hsum(v9_) == 108);

    unsigned scatter_destination[72] = {};
    scatter(scatter_destination + 2, v9i, v9);

    require(memcmp(gather_source, scatter_destination, sizeof(gather_source)) == 0);

    const ivec16 scatter_mask = {-1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1};

    unsigned masked_scatter_destination[] = {
        1, 0xffffffff, 2, 3, 0xffffffff, 4, 5, 6, 0xffffffff, 7, 8, 9, 1, 2, 3, 0xffffffff, 4, 0,
        1, 0xffffffff, 2, 3, 0xffffffff, 4, 5, 6, 0xffffffff, 7, 8, 9, 1, 2, 3, 0xffffffff, 4, 0,
        1, 0xffffffff, 2, 3, 0xffffffff, 4, 5, 6, 0xffffffff, 7, 8, 9, 1, 2, 3, 0xffffffff, 4, 0,
        1, 0xffffffff, 2, 3, 0xffffffff, 4, 5, 6, 0xffffffff, 7, 8, 9, 1, 2, 3, 0xffffffff, 4, 0};
    static const unsigned masked_scatter_expected[] = {
        1, 42, 2, 3, 0xffffffff, 4, 5, 6, 0xffffffff, 7, 8, 9, 1, 2, 3, 23, 4, 0,
        1, 42, 2, 3, 0xffffffff, 4, 5, 6, 0xffffffff, 7, 8, 9, 1, 2, 3, 23, 4, 0,
        1, 42, 2, 3, 0xffffffff, 4, 5, 6, 0xffffffff, 7, 8, 9, 1, 2, 3, 23, 4, 0,
        1, 42, 2, 3, 0xffffffff, 4, 5, 6, 0xffffffff, 7, 8, 9, 1, 2, 3, 23, 4, 0};

    scatter(masked_scatter_destination + 2, scatter_mask, v9i, v9);

    require(memcmp(masked_scatter_destination, masked_scatter_expected, sizeof(masked_scatter_destination)) == 0);

    const uvec16 v11 = {0xffffffff, 0, 0xffffffff, 0, 0xffffffff, 0, 0xffffffff, 0,
                             0xffffffff, 0, 0xffffffff, 0, 0xffffffff, 0, 0xffffffff, 0};
    uvec16 v12 = {0, 0xffffffff, 0, 0, 0, 0xffffffff, 0, 0, 0, 0xffffffff, 0, 0, 0, 0xffffffff, 0, 0};

    v12 |= v11;

    require(v12[0] == 0xffffffff);
    require(v12[1] == 0xffffffff);
    require(v12[2] == 0xffffffff);
    require(v12[3] == 0);
    require(v12[4] == 0xffffffff);
    require(v12[5] == 0xffffffff);
    require(v12[6] == 0xffffffff);
    require(v12[7] == 0);
    require(v12[8] == 0xffffffff);
    require(v12[9] == 0xffffffff);
    require(v12[10] == 0xffffffff);
    require(v12[11] == 0);
    require(v12[12] == 0xffffffff);
    require(v12[13] == 0xffffffff);
    require(v12[14] == 0xffffffff);
    require(v12[15] == 0);

    const uvec16 v13 = {0xffffffff, 0, 0xffffffff, 0, 0xffffffff, 0, 0xffffffff, 0,
                             0xffffffff, 0, 0xffffffff, 0, 0xffffffff, 0, 0xffffffff, 0};
    uvec16 v14 = {0, 0xffffffff, 0, 0, 0, 0xffffffff, 0, 0, 0, 0xffffffff, 0, 0, 0, 0xffffffff, 0, 0};

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

    const uvec16 v17 = {3, 1, 4, 1, 3, 1, 4, 1, 3, 1, 4, 1, 3, 1, 4, 1};
    const uvec16 v18 = inclusive_scan(v17);

    require(v18[0] == 3);
    require(v18[1] == 4);
    require(v18[2] == 8);
    require(v18[3] == 9);
    require(v18[4] == 12);
    require(v18[5] == 13);
    require(v18[6] == 17);
    require(v18[7] == 18);
    require(v18[8] == 21);
    require(v18[9] == 22);
    require(v18[10] == 26);
    require(v18[11] == 27);
    require(v18[12] == 30);
    require(v18[13] == 31);
    require(v18[14] == 35);
    require(v18[15] == 36);

    const ivec16 vmask = {-1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1};

    uvec16 v19 = v3;
    where(vmask, v19) = v2;

    const uvec16 v20 = select(vmask, v2, v3);

    require(v19.get<0>() == 4);
    require(v19.get<1>() == 7);
    require(v19.get<2>() == 9);
    require(v19.get<3>() == 7);
    require(v19.get<4>() == 8);
    require(v19.get<5>() == 14);
    require(v19.get<6>() == 15);
    require(v19.get<7>() == 1);
    require(v19.get<8>() == 4);
    require(v19.get<9>() == 7);
    require(v19.get<10>() == 9);
    require(v19.get<11>() == 7);
    require(v19.get<12>() == 8);
    require(v19.get<13>() == 14);
    require(v19.get<14>() == 15);
    require(v19.get<15>() == 1);

    require(v20.get<0>() == 4);
    require(v20.get<1>() == 7);
    require(v20.get<2>() == 9);
    require(v20.get<3>() == 7);
    require(v20.get<4>() == 8);
    require(v20.get<5>() == 14);
    require(v20.get<6>() == 15);
    require(v20.get<7>() == 1);
    require(v20.get<8>() == 4);
    require(v20.get<9>() == 7);
    require(v20.get<10>() == 9);
    require(v20.get<11>() == 7);
    require(v20.get<12>() == 8);
    require(v20.get<13>() == 14);
    require(v20.get<14>() == 15);
    require(v20.get<15>() == 1);

    printf("OK\n");
}