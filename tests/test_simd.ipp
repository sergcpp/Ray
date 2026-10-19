using namespace Ray::NS;

{
    printf("Test fvec4    (%s)     | ", fvec4::is_native() ? "hard" : "soft");

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
    fvec4 v66666 = fmadd(v1, v2, v3);
    fvec4 v666666 = fmsub(v1, v2, v3);

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

    require(v66666[0] == Approx(9.0));
    require(v66666[1] == Approx(17.0));
    require(v66666[2] == Approx(27.0));
    require(v66666[3] == Approx(39.0));

    require(v666666[0] == Approx(-1.0));
    require(v666666[1] == Approx(3.0));
    require(v666666[2] == Approx(9.0));
    require(v666666[3] == Approx(17.0));

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

    auto v11le = simd_cast(v2 <= v9);

    require(v11le[0] == 0);
    require(v11le[1] == -1);
    require(v11le[2] == -1);
    require(v11le[3] == 0);

    auto v11ge = simd_cast(v2 >= v9);

    require(v11ge[0] == -1);
    require(v11ge[1] == 0);
    require(v11ge[2] == 0);
    require(v11ge[3] == -1);

    auto v11eq = simd_cast(v2 == v9);

    require(v11eq[0] == 0);
    require(v11eq[1] == 0);
    require(v11eq[2] == 0);
    require(v11eq[3] == 0);

    auto v11ne = simd_cast(v2 != v9);

    require(v11ne[0] == -1);
    require(v11ne[1] == -1);
    require(v11ne[2] == -1);
    require(v11ne[3] == -1);

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

    const fvec4 v20 = copysign(v3, fvec4{-1, 2, -3, -4});

    require(v20[0] == Approx(-5));
    require(v20[1] == Approx(7));
    require(v20[2] == Approx(-9));
    require(v20[3] == Approx(-11));

    const fvec4 vw2 = {10.0f, 20.0f, 30.0f, 40.0f};
    const ivec4 vwmask = {-1, 0, 0, -1};

    fvec4 vw_add = {1.0f, 2.0f, 3.0f, 4.0f};
    where(vwmask, vw_add) += vw2;
    require(vw_add[0] == Approx(11.0f));
    require(vw_add[1] == Approx(2.0f));
    require(vw_add[2] == Approx(3.0f));
    require(vw_add[3] == Approx(44.0f));

    fvec4 vw_sub = {1.0f, 2.0f, 3.0f, 4.0f};
    where(vwmask, vw_sub) -= vw2;
    require(vw_sub[0] == Approx(-9.0f));
    require(vw_sub[1] == Approx(2.0f));
    require(vw_sub[2] == Approx(3.0f));
    require(vw_sub[3] == Approx(-36.0f));

    fvec4 vw_mul = {1.0f, 2.0f, 3.0f, 4.0f};
    where(vwmask, vw_mul) *= vw2;
    require(vw_mul[0] == Approx(10.0f));
    require(vw_mul[1] == Approx(2.0f));
    require(vw_mul[2] == Approx(3.0f));
    require(vw_mul[3] == Approx(160.0f));

    fvec4 vw_div = {1.0f, 2.0f, 3.0f, 4.0f};
    where(vwmask, vw_div) /= vw2;
    require(vw_div[0] == Approx(0.1f));
    require(vw_div[1] == Approx(2.0f));
    require(vw_div[2] == Approx(3.0f));
    require(vw_div[3] == Approx(0.1f));

    const fvec4 vmix1 = {0.0f, 2.0f, 4.0f, 6.0f};
    const fvec4 vmix2 = {4.0f, 6.0f, 8.0f, 10.0f};

    const fvec4 vmix_0 = mix(vmix1, vmix2, 0.0f);
    require(vmix_0[0] == Approx(0.0f));
    require(vmix_0[1] == Approx(2.0f));
    require(vmix_0[2] == Approx(4.0f));
    require(vmix_0[3] == Approx(6.0f));

    const fvec4 vmix_1 = mix(vmix1, vmix2, 1.0f);
    require(vmix_1[0] == Approx(4.0f));
    require(vmix_1[1] == Approx(6.0f));
    require(vmix_1[2] == Approx(8.0f));
    require(vmix_1[3] == Approx(10.0f));

    const fvec4 vmix_h = mix(vmix1, vmix2, 0.5f);
    require(vmix_h[0] == Approx(2.0f));
    require(vmix_h[1] == Approx(4.0f));
    require(vmix_h[2] == Approx(6.0f));
    require(vmix_h[3] == Approx(8.0f));

    const fvec4 vconv_f = {1.9f, -2.1f, 3.5f, -4.9f};

    const ivec4 vconv_fi = (ivec4)vconv_f;
    require(vconv_fi[0] == 1);
    require(vconv_fi[1] == -2);
    require(vconv_fi[2] == 3);
    require(vconv_fi[3] == -4);

    const fvec4 vconv_fu_src = {1.9f, 2.1f, 3.5f, 4.9f};
    const uvec4 vconv_fu = (uvec4)vconv_fu_src;
    require(vconv_fu[0] == 1);
    require(vconv_fu[1] == 2);
    require(vconv_fu[2] == 3);
    require(vconv_fu[3] == 4);

    const fvec4 vmin_a = {3.0f, 1.0f, 4.0f, 1.0f};
    const fvec4 vmin_b = {2.0f, 3.0f, 2.0f, 5.0f};
    const fvec4 vmin_r = min(vmin_a, vmin_b);
    const fvec4 vmax_r = max(vmin_a, vmin_b);

    require(vmin_r[0] == 2.0f);
    require(vmin_r[1] == 1.0f);
    require(vmin_r[2] == 2.0f);
    require(vmin_r[3] == 1.0f);
    require(vmax_r[0] == 3.0f);
    require(vmax_r[1] == 3.0f);
    require(vmax_r[2] == 4.0f);
    require(vmax_r[3] == 5.0f);

    // Half-float load/store: SSE-class x86 converts via fp16.h; aarch64 NEON via ARMv8 FCVTL
    // (ARMv7/soft fall back to fp16.h). Identical expectations on every arch.
    const unsigned short hb0[4] = {0x0000, 0x3C00, 0xBC00, 0x3E00};
    const fvec4 hv0 = fvec4::load_f16(hb0);
    require(hv0[0] == 0.0f);
    require(hv0[1] == 1.0f);
    require(hv0[2] == -1.0f);
    require(hv0[3] == 1.5f);

    const unsigned short hb1[4] = {0x7BFF, 0x0400, 0x0001, 0x8000};
    const fvec4 hv1 = fvec4::load_f16(hb1);
    require(hv1[0] == 65504.0f);               // max normal fp16
    require(hv1[1] == 6.103515625e-5f);        // min normal fp16
    require(hv1[2] == 5.9604644775390625e-8f); // min denormal fp16
    require(hv1[3] == 0.0f && std::signbit(hv1[3]));

    const unsigned short hb2[4] = {0x7C00, 0xFC00, 0x7E00, 0x8400};
    const fvec4 hv2 = fvec4::load_f16(hb2);
    require(std::isinf(hv2[0]) && hv2[0] > 0.0f);
    require(std::isinf(hv2[1]) && hv2[1] < 0.0f);
    require(std::isnan(hv2[2]));
    require(hv2[3] == -6.103515625e-5f);

    const float hvals[4] = {0.0f, -0.0f, 0.1f, 1.00048828125f};
    unsigned short hstore[4];
    fvec4{hvals}.store_f16(hstore);
    require(hstore[0] == 0x0000);
    require(hstore[1] == 0x8000);
    require(hstore[2] == 0x2E66); // RNE of 0.1f
    require(hstore[3] == 0x3C00); // 1 + 2^-11: exactly halfway -> ties to even

    // Exhaustive check against an independent ldexp-based reference decoder (bit-exact: finite
    // fp16 is exact in fp32). exp = 0x1F groups skipped - hw conversion quiets sNaNs (unmasked).
    auto f16_ref = [](const unsigned short h) -> float {
        const int e = (h >> 10) & 0x1F;
        const float m = float(h & 0x3FF);
        const float mag = (e == 0) ? m * std::ldexp(1.0f, -24) : (1.0f + m / 1024.0f) * std::ldexp(1.0f, e - 15);
        return (h & 0x8000) ? -mag : mag;
    };
    for (unsigned int base = 0; base < 0x10000u; base += 4) {
        if ((base & 0x7C00) == 0x7C00)
            continue;
        unsigned short hh[4];
        for (int k = 0; k < 4; ++k)
            hh[k] = uint16_t(base + k);
        const fvec4 hv = fvec4::load_f16(hh);
        for (int k = 0; k < 4; ++k)
            require(hv[k] == f16_ref(hh[k]));
    }

    // fp16 -> fp32 -> fp16 must be an identity over the finite range (exp = 0x1F groups skipped as above)
    for (unsigned int base = 0; base < 0x10000u; base += 4) {
        if ((base & 0x7C00) == 0x7C00)
            continue;
        unsigned short hh[4], ho[4];
        for (int k = 0; k < 4; ++k)
            hh[k] = uint16_t(base + k);
        fvec4::load_f16(hh).store_f16(ho);
        for (int k = 0; k < 4; ++k)
            require(ho[k] == hh[k]);
    }

    printf("OK\n");
}
{
    printf("Test ivec4    (%s)     | ", ivec4::is_native() ? "hard" : "soft");

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

    const ivec4 vsrli_i = {-1, -2, (int)0x80000000, 16};
    const ivec4 vsrli_r = srli(vsrli_i, 1);
    require(vsrli_r[0] == 0x7fffffff);
    require(vsrli_r[1] == 0x7fffffff);
    require(vsrli_r[2] == 0x40000000);
    require(vsrli_r[3] == 8);

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

    const ivec4 vwmask_i = {-1, 0, 0, -1};

    ivec4 vw_or = {0x0f, 0x0f, 0x0f, 0x0f};
    where(vwmask_i, vw_or) |= ivec4{0xf0, 0xf0, 0xf0, 0xf0};
    require(vw_or[0] == 0xff);
    require(vw_or[1] == 0x0f);
    require(vw_or[2] == 0x0f);
    require(vw_or[3] == 0xff);

    ivec4 vw_and = {0xff, 0xff, 0xff, 0xff};
    where(vwmask_i, vw_and) &= ivec4{0x0f, 0x0f, 0x0f, 0x0f};
    require(vw_and[0] == 0x0f);
    require(vw_and[1] == 0xff);
    require(vw_and[2] == 0xff);
    require(vw_and[3] == 0x0f);

    ivec4 vw_xor = {0xff, 0xff, 0xff, 0xff};
    where(vwmask_i, vw_xor) ^= ivec4{0x0f, 0x0f, 0x0f, 0x0f};
    require(vw_xor[0] == 0xf0);
    require(vw_xor[1] == 0xff);
    require(vw_xor[2] == 0xff);
    require(vw_xor[3] == 0xf0);

    // 0x3f800000 = 1.0f, 0x40000000 = 2.0f, 0xbf800000 = -1.0f, 0xc0000000 = -2.0f
    const ivec4 vsc_i = {0x3f800000, 0x40000000, (int)0xbf800000, (int)0xc0000000};
    const fvec4 vsc_if = simd_cast(vsc_i);
    require(vsc_if[0] == Approx(1.0f));
    require(vsc_if[1] == Approx(2.0f));
    require(vsc_if[2] == Approx(-1.0f));
    require(vsc_if[3] == Approx(-2.0f));

    const ivec4 vconv_i = {1, -2, 2147483647, -2147483647};
    const fvec4 vconv_if = (fvec4)vconv_i;
    require(vconv_if[0] == Approx(1.0f));
    require(vconv_if[1] == Approx(-2.0f));
    require(vconv_if[2] == Approx(2147483647.0f));
    require(vconv_if[3] == Approx(-2147483647.0f));

    const ivec4 vmin_a = {3, -1, 4, -5};
    const ivec4 vmin_b = {-2, 3, 2, 1};
    const ivec4 vmin_r = min(vmin_a, vmin_b);
    const ivec4 vmax_r = max(vmin_a, vmin_b);

    require(vmin_r[0] == -2);
    require(vmin_r[1] == -1);
    require(vmin_r[2] == 2);
    require(vmin_r[3] == -5);
    require(vmax_r[0] == 3);
    require(vmax_r[1] == 3);
    require(vmax_r[2] == 4);
    require(vmax_r[3] == 1);

    printf("OK\n");
}
{
    printf("Test uvec4    (%s)     | ", uvec4::is_native() ? "hard" : "soft");

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

    // 0x3f800000 = 1.0f, 0xbf800000 = -1.0f, 0x40000000 = 2.0f, 0xc0000000 = -2.0f
    const uvec4 vsc_u = {0x3f800000u, 0xbf800000u, 0x40000000u, 0xc0000000u};
    const fvec4 vsc_uf = simd_cast(vsc_u);
    require(vsc_uf[0] == Approx(1.0f));
    require(vsc_uf[1] == Approx(-1.0f));
    require(vsc_uf[2] == Approx(2.0f));
    require(vsc_uf[3] == Approx(-2.0f));

    const uvec4 vconv_u = {1, 0x80000000, 0xffffffff, 42};
    const fvec4 vconv_uf = (fvec4)vconv_u;
    require(vconv_uf[0] == Approx(1.0f));
    require(vconv_uf[1] == Approx(2147483648.0f));
    require(vconv_uf[2] == Approx(4294967295.0f));
    require(vconv_uf[3] == Approx(42.0f));

    const uvec4 vmin_a = {3, 0xffffffff, 4, 1};
    const uvec4 vmin_b = {2, 3, 0xffffffff, 5};
    const uvec4 vmin_r = min(vmin_a, vmin_b);
    const uvec4 vmax_r = max(vmin_a, vmin_b);

    require(vmin_r[0] == 2);
    require(vmin_r[1] == 3);
    require(vmin_r[2] == 4);
    require(vmin_r[3] == 1);
    require(vmax_r[0] == 3);
    require(vmax_r[1] == 0xffffffff);
    require(vmax_r[2] == 0xffffffff);
    require(vmax_r[3] == 5);

    const uvec4 vsrli_u = {0xffffffffu, 0x80000000u, 16u, 0u};
    const uvec4 vsrli_r = srli(vsrli_u, 4u);
    require(vsrli_r[0] == 0x0fffffffu);
    require(vsrli_r[1] == 0x08000000u);
    require(vsrli_r[2] == 1u);
    require(vsrli_r[3] == 0u);

    printf("OK\n");
}
{
    printf("Test fvec8    (%s)     | ", fvec8::is_native() ? "hard" : "soft");

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
    fvec8 v66666 = fmadd(v1, v2, v3);
    fvec8 v666666 = fmsub(v1, v2, v3);

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

    require(v66666[0] == Approx(9.0));
    require(v66666[1] == Approx(17.0));
    require(v66666[2] == Approx(27.0));
    require(v66666[3] == Approx(39.0));
    require(v66666[4] == Approx(53.0));
    require(v66666[5] == Approx(54.0));
    require(v66666[6] == Approx(51.0));
    require(v66666[7] == Approx(5.0));

    require(v666666[0] == Approx(-1.0));
    require(v666666[1] == Approx(3.0));
    require(v666666[2] == Approx(9.0));
    require(v666666[3] == Approx(17.0));
    require(v666666[4] == Approx(27.0));
    require(v666666[5] == Approx(26.0));
    require(v666666[6] == Approx(21.0));
    require(v666666[7] == Approx(-1.0));

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

    auto v11le = simd_cast(v2 <= v9);

    require(v11le[0] == 0);
    require(v11le[1] == -1);
    require(v11le[2] == -1);
    require(v11le[3] == 0);
    require(v11le[4] == 0);
    require(v11le[5] == -1);
    require(v11le[6] == -1);
    require(v11le[7] == 0);

    auto v11ge = simd_cast(v2 >= v9);

    require(v11ge[0] == -1);
    require(v11ge[1] == 0);
    require(v11ge[2] == 0);
    require(v11ge[3] == -1);
    require(v11ge[4] == -1);
    require(v11ge[5] == 0);
    require(v11ge[6] == 0);
    require(v11ge[7] == -1);

    auto v11eq = simd_cast(v2 == v9);

    require(v11eq[0] == 0);
    require(v11eq[1] == 0);
    require(v11eq[2] == 0);
    require(v11eq[3] == 0);
    require(v11eq[4] == 0);
    require(v11eq[5] == 0);
    require(v11eq[6] == 0);
    require(v11eq[7] == 0);

    auto v11ne = simd_cast(v2 != v9);

    require(v11ne[0] == -1);
    require(v11ne[1] == -1);
    require(v11ne[2] == -1);
    require(v11ne[3] == -1);
    require(v11ne[4] == -1);
    require(v11ne[5] == -1);
    require(v11ne[6] == -1);
    require(v11ne[7] == -1);

    static const float gather_source[] = {0, 42.0f, 0, 0, 12.0f, 0, 0, 0, 11.0f, 0, 0, 0, 0, 0, 0, 23.0f, 0, 0,
                                          0, 42.0f, 0, 0, 12.0f, 0, 0, 0, 11.0f, 0, 0, 0, 0, 0, 0, 23.0f, 0, 0};

    const ivec8 v12i = {-1, 2, 6, 13, 17, 20, 24, 31};
    const fvec8 v12 = gather(gather_source + 2, v12i);
    const fvec8 v12_masked = gather(fvec8{69}, gather_source + 2, ivec8{-1, 0, -1, 0, -1, 0, -1, 0}, v12i);

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

    const fvec8 v20 = copysign(v3, fvec8{-1, 2, -3, -4, -1, 2, -3, -4});

    require(v20[0] == Approx(-5));
    require(v20[1] == Approx(7));
    require(v20[2] == Approx(-9));
    require(v20[3] == Approx(-11));
    require(v20[4] == Approx(-13));
    require(v20[5] == Approx(14));
    require(v20[6] == Approx(-15));
    require(v20[7] == Approx(-3));

    const fvec8 vmin_a = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f, 6.0f};
    const fvec8 vmin_b = {2.0f, 3.0f, 2.0f, 5.0f, 9.0f, 5.0f, 8.0f, 2.0f};
    const fvec8 vmin_r = min(vmin_a, vmin_b);
    const fvec8 vmax_r = max(vmin_a, vmin_b);

    require(vmin_r[0] == 2.0f);
    require(vmin_r[1] == 1.0f);
    require(vmin_r[2] == 2.0f);
    require(vmin_r[3] == 1.0f);
    require(vmin_r[4] == 5.0f);
    require(vmin_r[5] == 5.0f);
    require(vmin_r[6] == 2.0f);
    require(vmin_r[7] == 2.0f);
    require(vmax_r[0] == 3.0f);
    require(vmax_r[1] == 3.0f);
    require(vmax_r[2] == 4.0f);
    require(vmax_r[3] == 5.0f);
    require(vmax_r[4] == 9.0f);
    require(vmax_r[5] == 9.0f);
    require(vmax_r[6] == 8.0f);
    require(vmax_r[7] == 6.0f);

    // Half-float load/store: F16C on AVX2/AVX512; software through fp16.h for the soft fvec8
    // (SSE policies) and for AVX1. Identical expectations on every arch.
    const unsigned short hb0[8] = {0x0000, 0x3C00, 0xBC00, 0x3E00, 0x7BFF, 0x0400, 0x0001, 0x0002};
    const fvec8 hv0 = fvec8::load_f16(hb0);
    require(hv0[0] == 0.0f);
    require(hv0[1] == 1.0f);
    require(hv0[2] == -1.0f);
    require(hv0[3] == 1.5f);
    require(hv0[4] == 65504.0f);               // max normal fp16
    require(hv0[5] == 6.103515625e-5f);        // min normal fp16
    require(hv0[6] == 5.9604644775390625e-8f); // min denormal fp16
    require(hv0[7] == 1.1920928955078125e-7f); // 2x min denormal

    const unsigned short hb1[8] = {0x7C00, 0xFC00, 0x7E00, 0x8000, 0x8400, 0xC100, 0x4248, 0x2E66};
    const fvec8 hv1 = fvec8::load_f16(hb1);
    require(std::isinf(hv1[0]) && hv1[0] > 0.0f);
    require(std::isinf(hv1[1]) && hv1[1] < 0.0f);
    require(std::isnan(hv1[2]));
    require(hv1[3] == 0.0f && std::signbit(hv1[3]));
    require(hv1[4] == -6.103515625e-5f);
    require(hv1[5] == -2.5f);
    require(hv1[6] == 3.140625f);
    require(hv1[7] == 0.0999755859375f);

    const float hvals[8] = {0.0f, -0.0f, 0.1f, 1.00048828125f, 3.14159265f, -2.5f, 65000.0f, 70000.0f};
    unsigned short hstore[8];
    fvec8{hvals}.store_f16(hstore);
    require(hstore[0] == 0x0000);
    require(hstore[1] == 0x8000);
    require(hstore[2] == 0x2E66); // RNE of 0.1f
    require(hstore[3] == 0x3C00); // 1 + 2^-11: exactly halfway -> ties to even
    require(hstore[4] == 0x4248); // nearest below pi
    require(hstore[5] == 0xC100); // -2.5 exact
    require(hstore[6] == 0x7BEF); // RNE of 65000 within the max normal range
    require(hstore[7] == 0x7C00); // overflow -> infinity

    // fp16 -> fp32 -> fp16 identity over the finite range (exp = 0x1F groups skipped, see fvec4)
    for (unsigned int base = 0; base < 0x10000u; base += 8) {
        if ((base & 0x7C00) == 0x7C00)
            continue;
        unsigned short hh[8], ho[8];
        for (int k = 0; k < 8; ++k)
            hh[k] = uint16_t(base + k);
        fvec8::load_f16(hh).store_f16(ho);
        for (int k = 0; k < 8; ++k)
            require(ho[k] == hh[k]);
    }

    printf("OK\n");
}
{
    printf("Test ivec8    (%s)     | ", ivec8::is_native() ? "hard" : "soft");

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

    const ivec8 vsrli_i = {-1, -2, (int)0x80000000, 16, -1, -2, (int)0x80000000, 16};
    const ivec8 vsrli_r = srli(vsrli_i, 1);
    require(vsrli_r[0] == 0x7fffffff);
    require(vsrli_r[1] == 0x7fffffff);
    require(vsrli_r[2] == 0x40000000);
    require(vsrli_r[3] == 8);
    require(vsrli_r[4] == 0x7fffffff);
    require(vsrli_r[5] == 0x7fffffff);
    require(vsrli_r[6] == 0x40000000);
    require(vsrli_r[7] == 8);

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

    const ivec8 vmin_a = {3, -1, 4, -5, 9, 2, -6, 5};
    const ivec8 vmin_b = {-2, 3, 2, 1, -9, 5, 3, 5};
    const ivec8 vmin_r = min(vmin_a, vmin_b);
    const ivec8 vmax_r = max(vmin_a, vmin_b);

    require(vmin_r[0] == -2);
    require(vmin_r[1] == -1);
    require(vmin_r[2] == 2);
    require(vmin_r[3] == -5);
    require(vmin_r[4] == -9);
    require(vmin_r[5] == 2);
    require(vmin_r[6] == -6);
    require(vmin_r[7] == 5);
    require(vmax_r[0] == 3);
    require(vmax_r[1] == 3);
    require(vmax_r[2] == 4);
    require(vmax_r[3] == 1);
    require(vmax_r[4] == 9);
    require(vmax_r[5] == 5);
    require(vmax_r[6] == 3);
    require(vmax_r[7] == 5);

    printf("OK\n");
}
{
    printf("Test uvec8    (%s)     | ", uvec8::is_native() ? "hard" : "soft");

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

    const uvec8 vmin_a = {3, 0xffffffff, 4, 1, 5, 0x80000001, 2, 6};
    const uvec8 vmin_b = {2, 3, 0xffffffff, 5, 9, 5, 0x80000000, 2};
    const uvec8 vmin_r = min(vmin_a, vmin_b);
    const uvec8 vmax_r = max(vmin_a, vmin_b);

    require(vmin_r[0] == 2);
    require(vmin_r[1] == 3);
    require(vmin_r[2] == 4);
    require(vmin_r[3] == 1);
    require(vmin_r[4] == 5);
    require(vmin_r[5] == 5);
    require(vmin_r[6] == 2);
    require(vmin_r[7] == 2);
    require(vmax_r[0] == 3);
    require(vmax_r[1] == 0xffffffff);
    require(vmax_r[2] == 0xffffffff);
    require(vmax_r[3] == 5);
    require(vmax_r[4] == 9);
    require(vmax_r[5] == 0x80000001);
    require(vmax_r[6] == 0x80000000);
    require(vmax_r[7] == 6);

    const uvec8 vsrli_u = {0xffffffffu, 0x80000000u, 16u, 0u, 0xffffffffu, 0x80000000u, 16u, 0u};
    const uvec8 vsrli_r = srli(vsrli_u, 4u);
    require(vsrli_r[0] == 0x0fffffffu);
    require(vsrli_r[1] == 0x08000000u);
    require(vsrli_r[2] == 1u);
    require(vsrli_r[3] == 0u);
    require(vsrli_r[4] == 0x0fffffffu);
    require(vsrli_r[5] == 0x08000000u);
    require(vsrli_r[6] == 1u);
    require(vsrli_r[7] == 0u);

    printf("OK\n");
}
{
    printf("Test fvec16   (%s)     | ", fvec16::is_native() ? "hard" : "soft");

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
    fvec16 v66666 = fmadd(v1, v2, v3);
    fvec16 v666666 = fmsub(v1, v2, v3);

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

    require(v66666[0] == Approx(9.0));
    require(v66666[1] == Approx(17.0));
    require(v66666[2] == Approx(27.0));
    require(v66666[3] == Approx(39.0));
    require(v66666[4] == Approx(53.0));
    require(v66666[5] == Approx(54.0));
    require(v66666[6] == Approx(51.0));
    require(v66666[7] == Approx(5.0));
    require(v66666[8] == Approx(9.0));
    require(v66666[9] == Approx(17.0));
    require(v66666[10] == Approx(27.0));
    require(v66666[11] == Approx(39.0));
    require(v66666[12] == Approx(53.0));
    require(v66666[13] == Approx(54.0));
    require(v66666[14] == Approx(51.0));
    require(v66666[15] == Approx(5.0));

    require(v666666[0] == Approx(-1.0));
    require(v666666[1] == Approx(3.0));
    require(v666666[2] == Approx(9.0));
    require(v666666[3] == Approx(17.0));
    require(v666666[4] == Approx(27.0));
    require(v666666[5] == Approx(26.0));
    require(v666666[6] == Approx(21.0));
    require(v666666[7] == Approx(-1.0));
    require(v666666[8] == Approx(-1.0));
    require(v666666[9] == Approx(3.0));
    require(v666666[10] == Approx(9.0));
    require(v666666[11] == Approx(17.0));
    require(v666666[12] == Approx(27.0));
    require(v666666[13] == Approx(26.0));
    require(v666666[14] == Approx(21.0));
    require(v666666[15] == Approx(-1.0));

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

    fvec16 v9 = {3.0f, 6.0f, 7.0f, 6.0f, 2.0f, 12.0f, 18.0f, 0.0f, 3.0f, 6.0f, 7.0f, 6.0f, 2.0f, 12.0f, 18.0f, 0.0f};
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

    auto v11le = simd_cast(v2 <= v9);

    require(v11le[0] == 0);
    require(v11le[1] == -1);
    require(v11le[2] == -1);
    require(v11le[3] == 0);
    require(v11le[4] == 0);
    require(v11le[5] == -1);
    require(v11le[6] == -1);
    require(v11le[7] == 0);

    auto v11ge = simd_cast(v2 >= v9);

    require(v11ge[0] == -1);
    require(v11ge[1] == 0);
    require(v11ge[2] == 0);
    require(v11ge[3] == -1);
    require(v11ge[4] == -1);
    require(v11ge[5] == 0);
    require(v11ge[6] == 0);
    require(v11ge[7] == -1);

    auto v11eq = simd_cast(v2 == v9);

    require(v11eq[0] == 0);
    require(v11eq[1] == 0);
    require(v11eq[2] == 0);
    require(v11eq[3] == 0);
    require(v11eq[4] == 0);
    require(v11eq[5] == 0);
    require(v11eq[6] == 0);
    require(v11eq[7] == 0);

    auto v11ne = simd_cast(v2 != v9);

    require(v11ne[0] == -1);
    require(v11ne[1] == -1);
    require(v11ne[2] == -1);
    require(v11ne[3] == -1);
    require(v11ne[4] == -1);
    require(v11ne[5] == -1);
    require(v11ne[6] == -1);
    require(v11ne[7] == -1);

    static const float gather_source[] = {0, 42.0f, 0, 0, 12.0f, 0, 0, 0, 11.0f, 0, 0, 0, 0, 0, 0, 23.0f, 0, 0,
                                          0, 42.0f, 0, 0, 12.0f, 0, 0, 0, 11.0f, 0, 0, 0, 0, 0, 0, 23.0f, 0, 0,
                                          0, 42.0f, 0, 0, 12.0f, 0, 0, 0, 11.0f, 0, 0, 0, 0, 0, 0, 23.0f, 0, 0,
                                          0, 42.0f, 0, 0, 12.0f, 0, 0, 0, 11.0f, 0, 0, 0, 0, 0, 0, 23.0f, 0, 0};

    const ivec16 v12i = {-1, 2, 6, 13, 17, 20, 24, 31, 35, 38, 42, 49, 53, 56, 60, 67};
    const fvec16 v12 = gather(gather_source + 2, v12i);
    const fvec16 v12_masked =
        gather(fvec16{69}, gather_source + 2, ivec16{-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0}, v12i);

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

    const fvec16 v20 = copysign(v3, fvec16{-1, 2, -3, -4, -1, 2, -3, -4, -1, 2, -3, -4, -1, 2, -3, -4});

    require(v20[0] == Approx(-5));
    require(v20[1] == Approx(7));
    require(v20[2] == Approx(-9));
    require(v20[3] == Approx(-11));
    require(v20[4] == Approx(-13));
    require(v20[5] == Approx(14));
    require(v20[6] == Approx(-15));
    require(v20[7] == Approx(-3));
    require(v20[8] == Approx(-5));
    require(v20[9] == Approx(7));
    require(v20[10] == Approx(-9));
    require(v20[11] == Approx(-11));
    require(v20[12] == Approx(-13));
    require(v20[13] == Approx(14));
    require(v20[14] == Approx(-15));
    require(v20[15] == Approx(-3));

    const fvec16 vmin_a = {3, 1, 4, 1, 5, 9, 2, 6, 3, 1, 4, 1, 5, 9, 2, 6};
    const fvec16 vmin_b = {2, 3, 2, 5, 9, 5, 8, 2, 2, 3, 2, 5, 9, 5, 8, 2};
    const fvec16 vmin_r = min(vmin_a, vmin_b);
    const fvec16 vmax_r = max(vmin_a, vmin_b);

    require(vmin_r[0] == 2.0f);
    require(vmin_r[1] == 1.0f);
    require(vmin_r[2] == 2.0f);
    require(vmin_r[3] == 1.0f);
    require(vmin_r[4] == 5.0f);
    require(vmin_r[5] == 5.0f);
    require(vmin_r[6] == 2.0f);
    require(vmin_r[7] == 2.0f);
    require(vmin_r[8] == 2.0f);
    require(vmin_r[9] == 1.0f);
    require(vmin_r[10] == 2.0f);
    require(vmin_r[11] == 1.0f);
    require(vmin_r[12] == 5.0f);
    require(vmin_r[13] == 5.0f);
    require(vmin_r[14] == 2.0f);
    require(vmin_r[15] == 2.0f);
    require(vmax_r[0] == 3.0f);
    require(vmax_r[1] == 3.0f);
    require(vmax_r[2] == 4.0f);
    require(vmax_r[3] == 5.0f);
    require(vmax_r[4] == 9.0f);
    require(vmax_r[5] == 9.0f);
    require(vmax_r[6] == 8.0f);
    require(vmax_r[7] == 6.0f);
    require(vmax_r[8] == 3.0f);
    require(vmax_r[9] == 3.0f);
    require(vmax_r[10] == 4.0f);
    require(vmax_r[11] == 5.0f);
    require(vmax_r[12] == 9.0f);
    require(vmax_r[13] == 9.0f);
    require(vmax_r[14] == 8.0f);
    require(vmax_r[15] == 6.0f);

    // Half-float load/store: F16C on AVX512; software through fp16.h for the soft fvec16 (SSE,
    // AVX1 and AVX2 policies). Identical expectations on every arch.
    const unsigned short hb0[16] = {0x0000, 0x3C00, 0xBC00, 0x3E00, 0x7BFF, 0x0400, 0x0001, 0x0002,
                                    0x7C00, 0xFC00, 0x7E00, 0x8000, 0x8400, 0xC100, 0x4248, 0x2E66};
    const fvec16 hv = fvec16::load_f16(hb0);
    require(hv[0] == 0.0f);
    require(hv[1] == 1.0f);
    require(hv[2] == -1.0f);
    require(hv[3] == 1.5f);
    require(hv[4] == 65504.0f);               // max normal fp16
    require(hv[5] == 6.103515625e-5f);        // min normal fp16
    require(hv[6] == 5.9604644775390625e-8f); // min denormal fp16
    require(hv[7] == 1.1920928955078125e-7f); // 2x min denormal
    require(std::isinf(hv[8]) && hv[8] > 0.0f);
    require(std::isinf(hv[9]) && hv[9] < 0.0f);
    require(std::isnan(hv[10]));
    require(hv[11] == 0.0f && std::signbit(hv[11]));
    require(hv[12] == -6.103515625e-5f);
    require(hv[13] == -2.5f);
    require(hv[14] == 3.140625f);
    require(hv[15] == 0.0999755859375f);

    const float hvals[16] = {0.0f,    -0.0f,       0.1f,      1.00048828125f, 3.14159265f, -2.5f, 65000.f,
                             70000.f, 65520.0f,    -65520.0f, 1e-8f,          -1e-8f,      6e-8f, 3.814697265625e-6f,
                             -0.3f,   2.998046875f};
    unsigned short hstore[16];
    fvec16{hvals}.store_f16(hstore);
    require(hstore[0] == 0x0000);
    require(hstore[1] == 0x8000);
    require(hstore[2] == 0x2E66);  // RNE of 0.1f
    require(hstore[3] == 0x3C00);  // 1 + 2^-11: exactly halfway -> ties to even
    require(hstore[4] == 0x4248);  // nearest below pi
    require(hstore[5] == 0xC100);  // -2.5 exact
    require(hstore[6] == 0x7BEF);  // RNE of 65000 within the max normal range
    require(hstore[7] == 0x7C00);  // overflow -> infinity
    require(hstore[8] == 0x7C00);  // 65520: exactly halfway -> ties to even -> inf
    require(hstore[9] == 0xFC00);  // negative counterpart
    require(hstore[10] == 0x0000); // underflow to zero
    require(hstore[11] == 0x8000); // underflow to zero, sign kept
    require(hstore[12] == 0x0001); // fp16 min denormal
    require(hstore[13] == 0x0040); // 2^-18, exact, in the denormal range
    require(hstore[14] == 0xB4CD); // RNE of -0.3f
    require(hstore[15] == 0x41FF); // exact: (1 + 511/1024) * 2

    // fp16 -> fp32 -> fp16 must be an identity over the finite range (the exp = 0x1F groups are
    // skipped as in the fvec4 half-float test)
    for (unsigned int base = 0; base < 0x10000u; base += 16) {
        if ((base & 0x7C00) == 0x7C00)
            continue;
        unsigned short hh[16], ho[16];
        for (int k = 0; k < 16; ++k)
            hh[k] = uint16_t(base + k);
        fvec16::load_f16(hh).store_f16(ho);
        for (int k = 0; k < 16; ++k)
            require(ho[k] == hh[k]);
    }

    printf("OK\n");
}
{
    printf("Test ivec16   (%s)     | ", ivec16::is_native() ? "hard" : "soft");

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
    const ivec16 v9_masked =
        gather(ivec16{69}, gather_source + 2, ivec16{-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0}, v9i);

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

    const ivec16 vsrli_i = {-1, -2, (int)0x80000000, 16, -1, -2, (int)0x80000000, 16,
                            -1, -2, (int)0x80000000, 16, -1, -2, (int)0x80000000, 16};
    const ivec16 vsrli_r = srli(vsrli_i, 1);
    require(vsrli_r[0] == 0x7fffffff);
    require(vsrli_r[1] == 0x7fffffff);
    require(vsrli_r[2] == 0x40000000);
    require(vsrli_r[3] == 8);
    require(vsrli_r[4] == 0x7fffffff);
    require(vsrli_r[5] == 0x7fffffff);
    require(vsrli_r[6] == 0x40000000);
    require(vsrli_r[7] == 8);

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

    const ivec16 vmin_a = {3, -1, 4, -5, 9, 2, -6, 5, 3, -1, 4, -5, 9, 2, -6, 5};
    const ivec16 vmin_b = {-2, 3, 2, 1, -9, 5, 3, 5, -2, 3, 2, 1, -9, 5, 3, 5};
    const ivec16 vmin_r = min(vmin_a, vmin_b);
    const ivec16 vmax_r = max(vmin_a, vmin_b);

    require(vmin_r[0] == -2);
    require(vmin_r[1] == -1);
    require(vmin_r[2] == 2);
    require(vmin_r[3] == -5);
    require(vmin_r[4] == -9);
    require(vmin_r[5] == 2);
    require(vmin_r[6] == -6);
    require(vmin_r[7] == 5);
    require(vmin_r[8] == -2);
    require(vmin_r[9] == -1);
    require(vmin_r[10] == 2);
    require(vmin_r[11] == -5);
    require(vmin_r[12] == -9);
    require(vmin_r[13] == 2);
    require(vmin_r[14] == -6);
    require(vmin_r[15] == 5);
    require(vmax_r[0] == 3);
    require(vmax_r[1] == 3);
    require(vmax_r[2] == 4);
    require(vmax_r[3] == 1);
    require(vmax_r[4] == 9);
    require(vmax_r[5] == 5);
    require(vmax_r[6] == 3);
    require(vmax_r[7] == 5);
    require(vmax_r[8] == 3);
    require(vmax_r[9] == 3);
    require(vmax_r[10] == 4);
    require(vmax_r[11] == 1);
    require(vmax_r[12] == 9);
    require(vmax_r[13] == 5);
    require(vmax_r[14] == 3);
    require(vmax_r[15] == 5);

    printf("OK\n");
}
{
    printf("Test uvec16   (%s)     | ", uvec16::is_native() ? "hard" : "soft");

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
    const uvec16 v9_masked =
        gather(uvec16{69}, gather_source + 2, ivec16{-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0}, v9i);

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

    const uvec16 vmin_a = {3, 0xffffffff, 4, 1, 5, 0x80000001, 2, 6, 3, 0xffffffff, 4, 1, 5, 0x80000001, 2, 6};
    const uvec16 vmin_b = {2, 3, 0xffffffff, 5, 9, 5, 0x80000000, 2, 2, 3, 0xffffffff, 5, 9, 5, 0x80000000, 2};
    const uvec16 vmin_r = min(vmin_a, vmin_b);
    const uvec16 vmax_r = max(vmin_a, vmin_b);

    require(vmin_r[0] == 2);
    require(vmin_r[1] == 3);
    require(vmin_r[2] == 4);
    require(vmin_r[3] == 1);
    require(vmin_r[4] == 5);
    require(vmin_r[5] == 5);
    require(vmin_r[6] == 2);
    require(vmin_r[7] == 2);
    require(vmin_r[8] == 2);
    require(vmin_r[9] == 3);
    require(vmin_r[10] == 4);
    require(vmin_r[11] == 1);
    require(vmin_r[12] == 5);
    require(vmin_r[13] == 5);
    require(vmin_r[14] == 2);
    require(vmin_r[15] == 2);
    require(vmax_r[0] == 3);
    require(vmax_r[1] == 0xffffffff);
    require(vmax_r[2] == 0xffffffff);
    require(vmax_r[3] == 5);
    require(vmax_r[4] == 9);
    require(vmax_r[5] == 0x80000001);
    require(vmax_r[6] == 0x80000000);
    require(vmax_r[7] == 6);
    require(vmax_r[8] == 3);
    require(vmax_r[9] == 0xffffffff);
    require(vmax_r[10] == 0xffffffff);
    require(vmax_r[11] == 5);
    require(vmax_r[12] == 9);
    require(vmax_r[13] == 0x80000001);
    require(vmax_r[14] == 0x80000000);
    require(vmax_r[15] == 6);

    const uvec16 vsrli_u = {0xffffffffu, 0x80000000u, 16u, 0u, 0xffffffffu, 0x80000000u, 16u, 0u,
                            0xffffffffu, 0x80000000u, 16u, 0u, 0xffffffffu, 0x80000000u, 16u, 0u};
    const uvec16 vsrli_r = srli(vsrli_u, 4u);
    require(vsrli_r[0] == 0x0fffffffu);
    require(vsrli_r[1] == 0x08000000u);
    require(vsrli_r[2] == 1u);
    require(vsrli_r[3] == 0u);
    require(vsrli_r[4] == 0x0fffffffu);
    require(vsrli_r[5] == 0x08000000u);
    require(vsrli_r[6] == 1u);
    require(vsrli_r[7] == 0u);

    printf("OK\n");
}
{
    printf("Test f16vec8  (%s)     | ", f16vec8::is_native() ? "hard" : "soft");

    // v2={4,5,6,7,8,10,11,12}, v9={3,5,7,6,9,12,11,11}
    // equal at lanes 1 (5==5) and 6 (11==11)
    f16vec8 v1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    f16vec8 v2 = {4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 10.0f, 11.0f, 12.0f};

    require((float)v1[0] == Approx(1));
    require((float)v1[7] == Approx(8));
    require((float)v2[0] == Approx(4));
    require((float)v2[7] == Approx(12));

    const f16vec8 v_sum = v1 + v2;
    const f16vec8 v_diff = v1 - v2;
    const f16vec8 v_prod = v1 * v2;
    const f16vec8 v_quot = v1 / v2;
    const f16vec8 v_neg = -v1;
    const f16vec8 v_fma = fmadd(v1, v2, v_sum);
    const f16vec8 v_fms = fmsub(v1, v2, v_sum);

    // v1+v2 = {5,7,9,11,13,16,18,20}
    require((float)v_sum[0] == Approx(5));
    require((float)v_sum[1] == Approx(7));
    require((float)v_sum[2] == Approx(9));
    require((float)v_sum[3] == Approx(11));
    require((float)v_sum[4] == Approx(13));
    require((float)v_sum[5] == Approx(16));
    require((float)v_sum[6] == Approx(18));
    require((float)v_sum[7] == Approx(20));

    // v1-v2 = {-3,-3,-3,-3,-3,-4,-4,-4}
    require((float)v_diff[0] == Approx(-3));
    require((float)v_diff[4] == Approx(-3));
    require((float)v_diff[5] == Approx(-4));
    require((float)v_diff[7] == Approx(-4));

    // v1*v2 = {4,10,18,28,40,60,77,96}
    require((float)v_prod[0] == Approx(4));
    require((float)v_prod[1] == Approx(10));
    require((float)v_prod[2] == Approx(18));
    require((float)v_prod[3] == Approx(28));
    require((float)v_prod[4] == Approx(40));
    require((float)v_prod[5] == Approx(60));
    require((float)v_prod[6] == Approx(77));
    require((float)v_prod[7] == Approx(96));

    // v1/v2 = {0.25,0.4,0.5,0.5714,0.625,0.6,0.6364,0.6667}
    require((float)v_quot[0] == Approx(0.25));
    require((float)v_quot[1] == Approx(0.4));
    require((float)v_quot[2] == Approx(0.5));
    require((float)v_quot[3] == Approx(0.5714));
    require((float)v_quot[4] == Approx(0.625));
    require((float)v_quot[5] == Approx(0.6));
    require((float)v_quot[6] == Approx(0.6364));
    require((float)v_quot[7] == Approx(0.6667));

    require((float)v_neg[0] == Approx(-1));
    require((float)v_neg[7] == Approx(-8));

    // fmadd(v1,v2,v_sum) = v1*v2+(v1+v2) = {9,17,27,39,53,76,95,116}
    require((float)v_fma[0] == Approx(9));
    require((float)v_fma[1] == Approx(17));
    require((float)v_fma[2] == Approx(27));
    require((float)v_fma[3] == Approx(39));
    require((float)v_fma[4] == Approx(53));
    require((float)v_fma[5] == Approx(76));
    require((float)v_fma[6] == Approx(95));
    require((float)v_fma[7] == Approx(116));

    // fmsub(v1,v2,v_sum) = v1*v2-(v1+v2) = {-1,3,9,17,27,44,59,76}
    require((float)v_fms[0] == Approx(-1));
    require((float)v_fms[1] == Approx(3));
    require((float)v_fms[2] == Approx(9));
    require((float)v_fms[3] == Approx(17));
    require((float)v_fms[4] == Approx(27));
    require((float)v_fms[5] == Approx(44));
    require((float)v_fms[6] == Approx(59));
    require((float)v_fms[7] == Approx(76));

    const f16vec8 vsq8 = {4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f, 4.0f, 1.0f};
    const f16vec8 vsqrt8 = sqrt(vsq8);

    require((float)vsqrt8[0] == Approx(2));
    require((float)vsqrt8[1] == Approx(3));
    require((float)vsqrt8[2] == Approx(4));
    require((float)vsqrt8[3] == Approx(5));
    require((float)vsqrt8[4] == Approx(6));
    require((float)vsqrt8[5] == Approx(7));
    require((float)vsqrt8[6] == Approx(2));
    require((float)vsqrt8[7] == Approx(1));

    const f16vec8 v9 = {3.0f, 5.0f, 7.0f, 6.0f, 9.0f, 12.0f, 11.0f, 11.0f};

    // hsum = 3+5+7+6+9+12+11+11 = 64
    require((float)hsum(v9) == Approx(64));

    // v2={4,5,6,7,8,10,11,12} vs v9={3,5,7,6,9,12,11,11}; where() writes 2.0f on masked lanes
    f16vec8 vt;

    vt = f16vec8{1.0f};
    where(v2 < v9, vt) = f16vec8{2.0f};
    require((float)vt[0] == Approx(1));
    require((float)vt[1] == Approx(1));
    require((float)vt[2] == Approx(2));
    require((float)vt[3] == Approx(1));
    require((float)vt[4] == Approx(2));
    require((float)vt[5] == Approx(2));
    require((float)vt[6] == Approx(1));
    require((float)vt[7] == Approx(1));

    vt = f16vec8{1.0f};
    where(v2 > v9, vt) = f16vec8{2.0f};
    require((float)vt[0] == Approx(2));
    require((float)vt[1] == Approx(1));
    require((float)vt[2] == Approx(1));
    require((float)vt[3] == Approx(2));
    require((float)vt[4] == Approx(1));
    require((float)vt[5] == Approx(1));
    require((float)vt[6] == Approx(1));
    require((float)vt[7] == Approx(2));

    vt = f16vec8{1.0f};
    where(v2 <= v9, vt) = f16vec8{2.0f};
    require((float)vt[0] == Approx(1));
    require((float)vt[1] == Approx(2));
    require((float)vt[2] == Approx(2));
    require((float)vt[3] == Approx(1));
    require((float)vt[4] == Approx(2));
    require((float)vt[5] == Approx(2));
    require((float)vt[6] == Approx(2));
    require((float)vt[7] == Approx(1));

    vt = f16vec8{1.0f};
    where(v2 >= v9, vt) = f16vec8{2.0f};
    require((float)vt[0] == Approx(2));
    require((float)vt[1] == Approx(2));
    require((float)vt[2] == Approx(1));
    require((float)vt[3] == Approx(2));
    require((float)vt[4] == Approx(1));
    require((float)vt[5] == Approx(1));
    require((float)vt[6] == Approx(2));
    require((float)vt[7] == Approx(2));

    vt = f16vec8{1.0f};
    where(v2 == v9, vt) = f16vec8{2.0f};
    require((float)vt[0] == Approx(1));
    require((float)vt[1] == Approx(2));
    require((float)vt[2] == Approx(1));
    require((float)vt[3] == Approx(1));
    require((float)vt[4] == Approx(1));
    require((float)vt[5] == Approx(1));
    require((float)vt[6] == Approx(2));
    require((float)vt[7] == Approx(1));

    vt = f16vec8{1.0f};
    where(v2 != v9, vt) = f16vec8{2.0f};
    require((float)vt[0] == Approx(2));
    require((float)vt[1] == Approx(1));
    require((float)vt[2] == Approx(2));
    require((float)vt[3] == Approx(2));
    require((float)vt[4] == Approx(2));
    require((float)vt[5] == Approx(2));
    require((float)vt[6] == Approx(1));
    require((float)vt[7] == Approx(2));

    // mask={F,F,T,F,T,T,F,F}: v_sum={5,7,9,11,13,16,18,20}, v2={4,5,6,7,8,10,11,12}
    // where true use v2, where false keep v_sum -> {5,7,6,11,8,10,18,20}
    const f16vec8 vmask8 = v2 < v9;
    f16vec8 v18 = v_sum;
    where(vmask8, v18) = v2;
    require((float)v18[0] == Approx(5));
    require((float)v18[1] == Approx(7));
    require((float)v18[2] == Approx(6));
    require((float)v18[3] == Approx(11));
    require((float)v18[4] == Approx(8));
    require((float)v18[5] == Approx(10));
    require((float)v18[6] == Approx(18));
    require((float)v18[7] == Approx(20));

    const f16vec8 v19 = select(vmask8, v2, v_sum);
    require((float)v19[0] == Approx(5));
    require((float)v19[1] == Approx(7));
    require((float)v19[2] == Approx(6));
    require((float)v19[3] == Approx(11));
    require((float)v19[4] == Approx(8));
    require((float)v19[5] == Approx(10));
    require((float)v19[6] == Approx(18));
    require((float)v19[7] == Approx(20));

    const f16vec8 vwmask8 = v2 < v9;
    const f16vec8 vw2_8 = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};

    f16vec8 vw_add8 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    where(vwmask8, vw_add8) += vw2_8;
    require((float)vw_add8[0] == Approx(1));
    require((float)vw_add8[1] == Approx(2));
    require((float)vw_add8[2] == Approx(33));
    require((float)vw_add8[3] == Approx(4));
    require((float)vw_add8[4] == Approx(55));
    require((float)vw_add8[5] == Approx(66));
    require((float)vw_add8[6] == Approx(7));
    require((float)vw_add8[7] == Approx(8));

    f16vec8 vw_sub8 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    where(vwmask8, vw_sub8) -= vw2_8;
    require((float)vw_sub8[0] == Approx(1));
    require((float)vw_sub8[1] == Approx(2));
    require((float)vw_sub8[2] == Approx(-27));
    require((float)vw_sub8[3] == Approx(4));
    require((float)vw_sub8[4] == Approx(-45));
    require((float)vw_sub8[5] == Approx(-54));
    require((float)vw_sub8[6] == Approx(7));
    require((float)vw_sub8[7] == Approx(8));

    f16vec8 vw_mul8 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    where(vwmask8, vw_mul8) *= vw2_8;
    require((float)vw_mul8[0] == Approx(1));
    require((float)vw_mul8[1] == Approx(2));
    require((float)vw_mul8[2] == Approx(90));
    require((float)vw_mul8[3] == Approx(4));
    require((float)vw_mul8[4] == Approx(250));
    require((float)vw_mul8[5] == Approx(360));
    require((float)vw_mul8[6] == Approx(7));
    require((float)vw_mul8[7] == Approx(8));

    f16vec8 vw_div8 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    where(vwmask8, vw_div8) /= vw2_8;
    require((float)vw_div8[0] == Approx(1));
    require((float)vw_div8[1] == Approx(2));
    require((float)vw_div8[2] == Approx(0.1));
    require((float)vw_div8[3] == Approx(4));
    require((float)vw_div8[4] == Approx(0.1));
    require((float)vw_div8[5] == Approx(0.1));
    require((float)vw_div8[6] == Approx(7));
    require((float)vw_div8[7] == Approx(8));

    const f16vec8 vcps8_v = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const f16vec8 vcps8_sign = {-1.0f, 2.0f, -3.0f, -4.0f, 5.0f, -6.0f, -7.0f, 8.0f};
    const f16vec8 vcps8_r = copysign(vcps8_v, vcps8_sign);
    require((float)vcps8_r[0] == Approx(-1));
    require((float)vcps8_r[1] == Approx(2));
    require((float)vcps8_r[2] == Approx(-3));
    require((float)vcps8_r[3] == Approx(-4));
    require((float)vcps8_r[4] == Approx(5));
    require((float)vcps8_r[5] == Approx(-6));
    require((float)vcps8_r[6] == Approx(-7));
    require((float)vcps8_r[7] == Approx(8));

    const f16vec8 vmin8_a = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f, 6.0f};
    const f16vec8 vmin8_b = {2.0f, 3.0f, 2.0f, 5.0f, 3.0f, 7.0f, 4.0f, 5.0f};
    const f16vec8 vmin8_r = min(vmin8_a, vmin8_b);
    const f16vec8 vmax8_r = max(vmin8_a, vmin8_b);

    require((float)vmin8_r[0] == Approx(2));
    require((float)vmin8_r[1] == Approx(1));
    require((float)vmin8_r[2] == Approx(2));
    require((float)vmin8_r[3] == Approx(1));
    require((float)vmin8_r[4] == Approx(3));
    require((float)vmin8_r[5] == Approx(7));
    require((float)vmin8_r[6] == Approx(2));
    require((float)vmin8_r[7] == Approx(5));
    require((float)vmax8_r[0] == Approx(3));
    require((float)vmax8_r[1] == Approx(3));
    require((float)vmax8_r[2] == Approx(4));
    require((float)vmax8_r[3] == Approx(5));
    require((float)vmax8_r[4] == Approx(5));
    require((float)vmax8_r[5] == Approx(9));
    require((float)vmax8_r[6] == Approx(4));
    require((float)vmax8_r[7] == Approx(6));

    printf("OK\n");
}
{
    printf("Test f16vec16 (%s)     | ", f16vec16::is_native() ? "hard" : "soft");

    // same pattern as f16vec8, repeated for both halves
    f16vec16 v1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    f16vec16 v2 = {4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 10.0f, 11.0f, 12.0f,
                   4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 10.0f, 11.0f, 12.0f};

    require((float)v1[0] == Approx(1));
    require((float)v1[15] == Approx(8));
    require((float)v2[0] == Approx(4));
    require((float)v2[15] == Approx(12));

    const f16vec16 v_sum = v1 + v2;
    const f16vec16 v_diff = v1 - v2;
    const f16vec16 v_prod = v1 * v2;
    const f16vec16 v_quot = v1 / v2;
    const f16vec16 v_neg = -v1;
    const f16vec16 v_fma = fmadd(v1, v2, v_sum);
    const f16vec16 v_fms = fmsub(v1, v2, v_sum);

    // v1+v2 = {5,7,9,11,13,16,18,20} x2
    require((float)v_sum[0] == Approx(5));
    require((float)v_sum[1] == Approx(7));
    require((float)v_sum[2] == Approx(9));
    require((float)v_sum[3] == Approx(11));
    require((float)v_sum[4] == Approx(13));
    require((float)v_sum[5] == Approx(16));
    require((float)v_sum[6] == Approx(18));
    require((float)v_sum[7] == Approx(20));
    require((float)v_sum[8] == Approx(5));
    require((float)v_sum[9] == Approx(7));
    require((float)v_sum[10] == Approx(9));
    require((float)v_sum[11] == Approx(11));
    require((float)v_sum[12] == Approx(13));
    require((float)v_sum[13] == Approx(16));
    require((float)v_sum[14] == Approx(18));
    require((float)v_sum[15] == Approx(20));

    // v1-v2 = {-3,-3,-3,-3,-3,-4,-4,-4} x2
    require((float)v_diff[0] == Approx(-3));
    require((float)v_diff[4] == Approx(-3));
    require((float)v_diff[5] == Approx(-4));
    require((float)v_diff[7] == Approx(-4));
    require((float)v_diff[8] == Approx(-3));
    require((float)v_diff[12] == Approx(-3));
    require((float)v_diff[13] == Approx(-4));
    require((float)v_diff[15] == Approx(-4));

    // v1*v2 = {4,10,18,28,40,60,77,96} x2
    require((float)v_prod[0] == Approx(4));
    require((float)v_prod[1] == Approx(10));
    require((float)v_prod[2] == Approx(18));
    require((float)v_prod[3] == Approx(28));
    require((float)v_prod[4] == Approx(40));
    require((float)v_prod[5] == Approx(60));
    require((float)v_prod[6] == Approx(77));
    require((float)v_prod[7] == Approx(96));
    require((float)v_prod[8] == Approx(4));
    require((float)v_prod[12] == Approx(40));
    require((float)v_prod[15] == Approx(96));

    // v1/v2 = {0.25,0.4,0.5,0.5714,0.625,0.6,0.6364,0.6667} x2
    require((float)v_quot[0] == Approx(0.25));
    require((float)v_quot[1] == Approx(0.4));
    require((float)v_quot[2] == Approx(0.5));
    require((float)v_quot[3] == Approx(0.5714));
    require((float)v_quot[4] == Approx(0.625));
    require((float)v_quot[5] == Approx(0.6));
    require((float)v_quot[6] == Approx(0.6364));
    require((float)v_quot[7] == Approx(0.6667));
    require((float)v_quot[8] == Approx(0.25));
    require((float)v_quot[15] == Approx(0.6667));

    require((float)v_neg[0] == Approx(-1));
    require((float)v_neg[15] == Approx(-8));

    // fmadd = {9,17,27,39,53,76,95,116} x2
    require((float)v_fma[0] == Approx(9));
    require((float)v_fma[1] == Approx(17));
    require((float)v_fma[2] == Approx(27));
    require((float)v_fma[3] == Approx(39));
    require((float)v_fma[4] == Approx(53));
    require((float)v_fma[5] == Approx(76));
    require((float)v_fma[6] == Approx(95));
    require((float)v_fma[7] == Approx(116));
    require((float)v_fma[8] == Approx(9));
    require((float)v_fma[15] == Approx(116));

    // fmsub = {-1,3,9,17,27,44,59,76} x2
    require((float)v_fms[0] == Approx(-1));
    require((float)v_fms[1] == Approx(3));
    require((float)v_fms[2] == Approx(9));
    require((float)v_fms[3] == Approx(17));
    require((float)v_fms[4] == Approx(27));
    require((float)v_fms[5] == Approx(44));
    require((float)v_fms[6] == Approx(59));
    require((float)v_fms[7] == Approx(76));
    require((float)v_fms[8] == Approx(-1));
    require((float)v_fms[15] == Approx(76));

    const f16vec16 vsq16 = {4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f, 4.0f, 1.0f,
                            4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f, 4.0f, 1.0f};
    const f16vec16 vsqrt16 = sqrt(vsq16);

    require((float)vsqrt16[0] == Approx(2));
    require((float)vsqrt16[1] == Approx(3));
    require((float)vsqrt16[2] == Approx(4));
    require((float)vsqrt16[3] == Approx(5));
    require((float)vsqrt16[4] == Approx(6));
    require((float)vsqrt16[5] == Approx(7));
    require((float)vsqrt16[6] == Approx(2));
    require((float)vsqrt16[7] == Approx(1));
    require((float)vsqrt16[8] == Approx(2));
    require((float)vsqrt16[15] == Approx(1));

    const f16vec16 v9 = {3.0f, 5.0f, 7.0f, 6.0f, 9.0f, 12.0f, 11.0f, 11.0f,
                         3.0f, 5.0f, 7.0f, 6.0f, 9.0f, 12.0f, 11.0f, 11.0f};

    // hsum = 64 * 2 = 128
    require((float)hsum(v9) == Approx(128));

    // comparisons: same masks as the f16vec8 section, mirrored into the upper half
    f16vec16 vt;

    vt = f16vec16{1.0f};
    where(v2 < v9, vt) = f16vec16{2.0f};
    require((float)vt[0] == Approx(1));
    require((float)vt[1] == Approx(1));
    require((float)vt[2] == Approx(2));
    require((float)vt[3] == Approx(1));
    require((float)vt[4] == Approx(2));
    require((float)vt[5] == Approx(2));
    require((float)vt[6] == Approx(1));
    require((float)vt[7] == Approx(1));
    require((float)vt[8] == Approx(1));
    require((float)vt[9] == Approx(1));
    require((float)vt[10] == Approx(2));
    require((float)vt[11] == Approx(1));
    require((float)vt[12] == Approx(2));
    require((float)vt[13] == Approx(2));
    require((float)vt[14] == Approx(1));
    require((float)vt[15] == Approx(1));

    vt = f16vec16{1.0f};
    where(v2 > v9, vt) = f16vec16{2.0f};
    require((float)vt[0] == Approx(2));
    require((float)vt[1] == Approx(1));
    require((float)vt[2] == Approx(1));
    require((float)vt[3] == Approx(2));
    require((float)vt[4] == Approx(1));
    require((float)vt[5] == Approx(1));
    require((float)vt[6] == Approx(1));
    require((float)vt[7] == Approx(2));
    require((float)vt[8] == Approx(2));
    require((float)vt[11] == Approx(2));
    require((float)vt[14] == Approx(1));
    require((float)vt[15] == Approx(2));

    vt = f16vec16{1.0f};
    where(v2 <= v9, vt) = f16vec16{2.0f};
    require((float)vt[0] == Approx(1));
    require((float)vt[1] == Approx(2));
    require((float)vt[2] == Approx(2));
    require((float)vt[3] == Approx(1));
    require((float)vt[4] == Approx(2));
    require((float)vt[5] == Approx(2));
    require((float)vt[6] == Approx(2));
    require((float)vt[7] == Approx(1));
    require((float)vt[8] == Approx(1));
    require((float)vt[14] == Approx(2));
    require((float)vt[15] == Approx(1));

    vt = f16vec16{1.0f};
    where(v2 >= v9, vt) = f16vec16{2.0f};
    require((float)vt[0] == Approx(2));
    require((float)vt[1] == Approx(2));
    require((float)vt[2] == Approx(1));
    require((float)vt[3] == Approx(2));
    require((float)vt[4] == Approx(1));
    require((float)vt[5] == Approx(1));
    require((float)vt[6] == Approx(2));
    require((float)vt[7] == Approx(2));
    require((float)vt[8] == Approx(2));
    require((float)vt[14] == Approx(2));
    require((float)vt[15] == Approx(2));

    vt = f16vec16{1.0f};
    where(v2 == v9, vt) = f16vec16{2.0f};
    require((float)vt[0] == Approx(1));
    require((float)vt[1] == Approx(2));
    require((float)vt[2] == Approx(1));
    require((float)vt[3] == Approx(1));
    require((float)vt[4] == Approx(1));
    require((float)vt[5] == Approx(1));
    require((float)vt[6] == Approx(2));
    require((float)vt[7] == Approx(1));
    require((float)vt[8] == Approx(1));
    require((float)vt[9] == Approx(2));
    require((float)vt[13] == Approx(1));
    require((float)vt[14] == Approx(2));
    require((float)vt[15] == Approx(1));

    vt = f16vec16{1.0f};
    where(v2 != v9, vt) = f16vec16{2.0f};
    require((float)vt[0] == Approx(2));
    require((float)vt[1] == Approx(1));
    require((float)vt[2] == Approx(2));
    require((float)vt[3] == Approx(2));
    require((float)vt[4] == Approx(2));
    require((float)vt[5] == Approx(2));
    require((float)vt[6] == Approx(1));
    require((float)vt[7] == Approx(2));
    require((float)vt[8] == Approx(2));
    require((float)vt[9] == Approx(1));
    require((float)vt[13] == Approx(2));
    require((float)vt[14] == Approx(1));
    require((float)vt[15] == Approx(2));

    // mask={F,F,T,F,T,T,F,F}x2: v_sum->v2 where true -> {5,7,6,11,8,10,18,20}x2
    const f16vec16 vmask16 = v2 < v9;
    f16vec16 v18 = v_sum;
    where(vmask16, v18) = v2;
    require((float)v18[0] == Approx(5));
    require((float)v18[1] == Approx(7));
    require((float)v18[2] == Approx(6));
    require((float)v18[3] == Approx(11));
    require((float)v18[4] == Approx(8));
    require((float)v18[5] == Approx(10));
    require((float)v18[6] == Approx(18));
    require((float)v18[7] == Approx(20));
    require((float)v18[8] == Approx(5));
    require((float)v18[13] == Approx(10));
    require((float)v18[14] == Approx(18));
    require((float)v18[15] == Approx(20));

    const f16vec16 v19 = select(vmask16, v2, v_sum);
    require((float)v19[0] == Approx(5));
    require((float)v19[2] == Approx(6));
    require((float)v19[5] == Approx(10));
    require((float)v19[8] == Approx(5));
    require((float)v19[10] == Approx(6));
    require((float)v19[13] == Approx(10));
    require((float)v19[15] == Approx(20));

    const f16vec16 vwmask16 = v2 < v9;
    const f16vec16 vw2_16 = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f,
                             10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};

    f16vec16 vw_add16 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    where(vwmask16, vw_add16) += vw2_16;
    require((float)vw_add16[0] == Approx(1));
    require((float)vw_add16[2] == Approx(33));
    require((float)vw_add16[4] == Approx(55));
    require((float)vw_add16[5] == Approx(66));
    require((float)vw_add16[7] == Approx(8));
    require((float)vw_add16[8] == Approx(1));
    require((float)vw_add16[10] == Approx(33));
    require((float)vw_add16[13] == Approx(66));
    require((float)vw_add16[15] == Approx(8));

    f16vec16 vw_sub16 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    where(vwmask16, vw_sub16) -= vw2_16;
    require((float)vw_sub16[0] == Approx(1));
    require((float)vw_sub16[2] == Approx(-27));
    require((float)vw_sub16[4] == Approx(-45));
    require((float)vw_sub16[5] == Approx(-54));
    require((float)vw_sub16[10] == Approx(-27));
    require((float)vw_sub16[13] == Approx(-54));

    f16vec16 vw_mul16 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    where(vwmask16, vw_mul16) *= vw2_16;
    require((float)vw_mul16[0] == Approx(1));
    require((float)vw_mul16[2] == Approx(90));
    require((float)vw_mul16[4] == Approx(250));
    require((float)vw_mul16[5] == Approx(360));
    require((float)vw_mul16[10] == Approx(90));
    require((float)vw_mul16[13] == Approx(360));

    f16vec16 vw_div16 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    where(vwmask16, vw_div16) /= vw2_16;
    require((float)vw_div16[0] == Approx(1));
    require((float)vw_div16[2] == Approx(0.1));
    require((float)vw_div16[4] == Approx(0.1));
    require((float)vw_div16[5] == Approx(0.1));
    require((float)vw_div16[7] == Approx(8));
    require((float)vw_div16[10] == Approx(0.1));
    require((float)vw_div16[13] == Approx(0.1));

    const f16vec16 vcps16_v = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                               1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const f16vec16 vcps16_sign = {-1.0f, 2.0f, -3.0f, -4.0f, 5.0f, -6.0f, -7.0f, 8.0f,
                                  -1.0f, 2.0f, -3.0f, -4.0f, 5.0f, -6.0f, -7.0f, 8.0f};
    const f16vec16 vcps16_r = copysign(vcps16_v, vcps16_sign);
    require((float)vcps16_r[0] == Approx(-1));
    require((float)vcps16_r[1] == Approx(2));
    require((float)vcps16_r[2] == Approx(-3));
    require((float)vcps16_r[4] == Approx(5));
    require((float)vcps16_r[5] == Approx(-6));
    require((float)vcps16_r[8] == Approx(-1));
    require((float)vcps16_r[15] == Approx(8));

    const f16vec16 vmin16_a = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f, 6.0f,
                               3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f, 6.0f};
    const f16vec16 vmin16_b = {2.0f, 3.0f, 2.0f, 5.0f, 3.0f, 7.0f, 4.0f, 5.0f,
                               2.0f, 3.0f, 2.0f, 5.0f, 3.0f, 7.0f, 4.0f, 5.0f};
    const f16vec16 vmin16_r = min(vmin16_a, vmin16_b);
    const f16vec16 vmax16_r = max(vmin16_a, vmin16_b);

    require((float)vmin16_r[0] == Approx(2));
    require((float)vmin16_r[1] == Approx(1));
    require((float)vmin16_r[2] == Approx(2));
    require((float)vmin16_r[3] == Approx(1));
    require((float)vmin16_r[4] == Approx(3));
    require((float)vmin16_r[5] == Approx(7));
    require((float)vmin16_r[6] == Approx(2));
    require((float)vmin16_r[7] == Approx(5));
    require((float)vmin16_r[8] == Approx(2));
    require((float)vmin16_r[15] == Approx(5));
    require((float)vmax16_r[0] == Approx(3));
    require((float)vmax16_r[5] == Approx(9));
    require((float)vmax16_r[7] == Approx(6));
    require((float)vmax16_r[15] == Approx(6));

    printf("OK\n");
}
{
    printf("Test f16vec32 (%s)     | ", f16vec32::is_native() ? "hard" : "soft");

    // v1[i] = i+1 in {1..32}, v2[i] = 32-i in {1..32}
    Ray::float16 src1[32], src2[32];
    for (int i = 0; i < 32; ++i) {
        src1[i] = float(i + 1);
        src2[i] = float(32 - i);
    }

    const f16vec32 v1{&src1[0]};
    const f16vec32 v2{&src2[0]};

    for (int i = 0; i < 32; ++i) {
        require((float)v1[i] == Approx(float(i + 1)));
        require((float)v2[i] == Approx(float(32 - i)));
    }

    // hsum: 1+2+...+32 = 528
    require(float(v1.hsum()) == Approx(528));

    // arithmetic: v1[i]+v2[i] = (i+1)+(32-i) = 33 for all lanes
    const f16vec32 v_sum = v1 + v2;
    for (int i = 0; i < 32; ++i) {
        require((float)v_sum[i] == Approx(33));
    }

    const f16vec32 v_neg = -v1;
    for (int i = 0; i < 32; ++i) {
        require((float)v_neg[i] == Approx(-float(i + 1)));
    }

    // sqrt of broadcast perfect square
    const f16vec32 vsq32 = {4.0f};
    const f16vec32 vsqrt32 = sqrt(vsq32);
    require((float)vsqrt32[0] == Approx(2));
    require((float)vsqrt32[31] == Approx(2));

    // fmadd: v1*v2 + v_sum = (i+1)*(32-i) + 33
    const f16vec32 v_fma = fmadd(v1, v2, v_sum);
    for (int i = 0; i < 32; ++i) {
        require((float)v_fma[i] == Approx(float(i + 1) * float(32 - i) + 33.0f));
    }

    // comparisons: v1 < v2 iff i+1 < 32-i iff i < 15.5, so true for i in {0..15}
    f16vec32 vt{1.0f};
    where(v1 < v2, vt) = f16vec32{2.0f};
    for (int i = 0; i < 32; ++i) {
        require((float)vt[i] == Approx(float(i + 1) < float(32 - i) ? 2.0f : 1.0f));
    }

    const f16vec32 vmask32 = v1 < v2;
    const f16vec32 v19 = select(vmask32, v1, v2);
    for (int i = 0; i < 32; ++i) {
        float a = float(i + 1), b = float(32 - i);
        require((float)v19[i] == Approx(a < b ? a : b));
    }

    // min/max
    const f16vec32 vmin32 = min(v1, v2);
    const f16vec32 vmax32 = max(v1, v2);
    for (int i = 0; i < 32; ++i) {
        float a = float(i + 1), b = float(32 - i);
        require((float)vmin32[i] == Approx(a < b ? a : b));
        require((float)vmax32[i] == Approx(a > b ? a : b));
    }

    // copysign: apply sign of v_neg (all negative) to broadcast 3.0f -> all -3
    const f16vec32 vcps_r = copysign(f16vec32{3.0f}, v_neg);
    require((float)vcps_r[0] == Approx(-3));
    require((float)vcps_r[31] == Approx(-3));

    printf("OK\n");
}
