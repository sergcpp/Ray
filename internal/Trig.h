#pragma once

namespace Ray::NS {
//
// Polynomial approximation of cosine
// Max error : 6.987e-07
//
template <int S> fvec<S> portable_cos(fvec<S> a) {
    // Normalize angle to [0, 1] range, where 1 corresponds to 2*PI
    a = fract(abs(a) * 0.15915494309189535f);

    // Select between ranges [0; 0.25), [0.25; 0.75), [0.75; 1.0]
    fvec<S> selector[3] = {0.0f, 0.0f, 0.0f};
    where(a < 0.25f, selector[0]) = 1.0f;
    where(a >= 0.75f, selector[2]) = 1.0f;
    selector[1] = -1.0f + selector[0] + selector[2];

    // Center around ranges
    fvec<S> arg[3] = {a, a - 0.5f, a - 1.0f};
    // Squared value gives better precision
    UNROLLED_FOR(i, 3, { arg[i] *= arg[i]; })

    // Evaluate 5th-degree polynome
    fvec<S> res[3];
    UNROLLED_FOR(i, 3, {
        res[i] = fmadd(-25.0407296503853054f, arg[i], 60.1524123580209817f);
        res[i] = fmsub(res[i], arg[i], 85.4539888046442542f);
        res[i] = fmadd(res[i], arg[i], 64.9393549651994562f);
        res[i] = fmsub(res[i], arg[i], 19.7392086060579359f);
        res[i] = fmadd(res[i], arg[i], 0.9999999998415476f);
    })

    // Combine contributions based on selector to get final value
    return res[0] * selector[0] + res[1] * selector[1] + res[2] * selector[2];
}

//
// Polynomial approximation of cosine
// Max error : 8.482e-07
//
template <int S> fvec<S> portable_sin(fvec<S> a) {
    // Normalize angle to [0, 1] range, where 1 corresponds to 2*PI
    a = fract(abs(a - 1.5707963267948966f) * 0.15915494309189535f);

    // Select between ranges [0; 0.25), [0.25; 0.75), [0.75; 1.0]
    fvec<S> selector[3] = {0.0f, 0.0f, 0.0f};
    where(a < 0.25f, selector[0]) = 1.0f;
    where(a >= 0.75f, selector[2]) = 1.0f;
    selector[1] = -1.0f + selector[0] + selector[2];

    // Center around ranges
    fvec<S> arg[3] = {a, a - 0.5f, a - 1.0f};
    // Squared value gives better precision
    UNROLLED_FOR(i, 3, { arg[i] *= arg[i]; })

    // Evaluate 5th-degree polynome
    fvec<S> res[3];
    UNROLLED_FOR(i, 3, {
        res[i] = fmadd(-25.0407296503853054f, arg[i], 60.1524123580209817f);
        res[i] = fmsub(res[i], arg[i], 85.4539888046442542f);
        res[i] = fmadd(res[i], arg[i], 64.9393549651994562f);
        res[i] = fmsub(res[i], arg[i], 19.7392086060579359f);
        res[i] = fmadd(res[i], arg[i], 0.9999999998415476f);
    })

    // Combine contributions based on selector to get final value
    return res[0] * selector[0] + res[1] * selector[1] + res[2] * selector[2];
}

//
// Combined approximation of sine/cosine
// Max error : 8.482e-07
//
template <int S> void portable_sincos(fvec<S> a, fvec<S> &out_sin, fvec<S> &out_cos) {
    // Normalize angle to [0, 1] range, where 1 corresponds to 2*PI
    const fvec<S> a_cos = fract(abs(a) * 0.15915494309189535f);
    const fvec<S> a_sin = fract(abs(a - 1.5707963267948966f) * 0.15915494309189535f);

    // Select between ranges [0; 0.25), [0.25; 0.75), [0.75; 1.0]
    fvec<S> selector_cos[3] = {0.0f, 0.0f, 0.0f}, selector_sin[3] = {0.0f, 0.0f, 0.0f};
    where(a_cos < 0.25f, selector_cos[0]) = 1.0f;
    where(a_cos >= 0.75f, selector_cos[2]) = 1.0f;
    selector_cos[1] = -1.0f + selector_cos[0] + selector_cos[2];
    where(a_sin < 0.25f, selector_sin[0]) = 1.0f;
    where(a_sin >= 0.75f, selector_sin[2]) = 1.0f;
    selector_sin[1] = -1.0f + selector_sin[0] + selector_sin[2];

    // Center around ranges
    fvec<S> arg_cos[3] = {a_cos, a_cos - 0.5f, a_cos - 1.0f}, arg_sin[3] = {a_sin, a_sin - 0.5f, a_sin - 1.0f};
    // Squared value gives better precision
    UNROLLED_FOR(i, 3, {
        arg_cos[i] *= arg_cos[i];
        arg_sin[i] *= arg_sin[i];
    })

    // Evaluate 5th-degree polynome
    fvec<S> res_cos[3], res_sin[3];
    UNROLLED_FOR(i, 3, {
        res_cos[i] = fmadd(-25.0407296503853054f, arg_cos[i], 60.1524123580209817f);
        res_sin[i] = fmadd(-25.0407296503853054f, arg_sin[i], 60.1524123580209817f);
        res_cos[i] = fmsub(res_cos[i], arg_cos[i], 85.4539888046442542f);
        res_sin[i] = fmsub(res_sin[i], arg_sin[i], 85.4539888046442542f);
        res_cos[i] = fmadd(res_cos[i], arg_cos[i], 64.9393549651994562f);
        res_sin[i] = fmadd(res_sin[i], arg_sin[i], 64.9393549651994562f);
        res_cos[i] = fmsub(res_cos[i], arg_cos[i], 19.7392086060579359f);
        res_sin[i] = fmsub(res_sin[i], arg_sin[i], 19.7392086060579359f);
        res_cos[i] = fmadd(res_cos[i], arg_cos[i], 0.9999999998415476f);
        res_sin[i] = fmadd(res_sin[i], arg_sin[i], 0.9999999998415476f);
    })

    // Combine contributions based on selector to get final value
    out_sin = res_sin[0] * selector_sin[0] + res_sin[1] * selector_sin[1] + res_sin[2] * selector_sin[2];
    out_cos = res_cos[0] * selector_cos[0] + res_cos[1] * selector_cos[1] + res_cos[2] * selector_cos[2];
}
} // namespace Ray::NS