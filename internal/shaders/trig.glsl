#ifndef TRIG_GLSL
#define TRIG_GLSL

//
// Polynomial approximation of cosine
// Max error : 6.987e-07
//
float portable_cos(float a) {
    // Normalize angle to [0, 1] range, where 1 corresponds to 2*PI
    a = fract(abs(a) * 0.15915494309189535);

    // Select between ranges [0; 0.25), [0.25; 0.75), [0.75; 1.0]
    vec3 selector = vec3(0.0);
    selector.x = float(a < 0.25);
    selector.z = float(a >= 0.75);
    selector.y = -1.0 + dot(selector, vec3(1.0, 0.0, 1.0));

    // Center around ranges
    vec3 arg = vec3(a) - vec3(0.0, 0.5, 1.0);
    // Squared value gives better precision
    arg *= arg;

    // Evaluate 5th-degree polynome
    vec3 res = fma(vec3(-25.0407296503853054), arg, vec3(60.1524123580209817));
    res = fma(res, arg, vec3(-85.4539888046442542));
    res = fma(res, arg, vec3(64.9393549651994562));
    res = fma(res, arg, vec3(-19.7392086060579359));
    res = fma(res, arg, vec3(0.9999999998415476));

    // Combine contributions based on selector to get final value
    return dot(res, selector);
}

//
// Polynomial approximation of sine
// Max error : 8.482e-07
//
float portable_sin(float a) {
    // Normalize angle to [0, 1] range, where 1 corresponds to 2*PI
    a = fract(abs(a - 1.5707963267948966) * 0.15915494309189535);

    // Select between ranges [0; 0.25), [0.25; 0.75), [0.75; 1.0]
    vec3 selector = vec3(0.0);
    selector.x = float(a < 0.25);
    selector.z = float(a >= 0.75);
    selector.y = -1.0 + dot(selector, vec3(1.0, 0.0, 1.0));

    // Center around ranges
    vec3 arg = vec3(a) - vec3(0.0, 0.5, 1.0);
    // Squared value gives better precision
    arg *= arg;

    // Evaluate 5th-degree polynome
    vec3 res = fma(vec3(-25.0407296503853054), arg, vec3(60.1524123580209817));
    res = fma(res, arg, vec3(-85.4539888046442542));
    res = fma(res, arg, vec3(64.9393549651994562));
    res = fma(res, arg, vec3(-19.7392086060579359));
    res = fma(res, arg, vec3(0.9999999998415476));

    // Combine contributions based on selector to get final value
    return dot(res, selector);
}

//
// Combined approximation of sine/cosine
// Max error : 8.482e-07
//
vec2 portable_sincos(const float a) {
    // Normalize angle to [0, 1] range, where 1 corresponds to 2*PI
    const float a_cos = fract(abs(a) * 0.15915494309189535);
    const float a_sin = fract(abs(a - 1.5707963267948966) * 0.15915494309189535);

    // Select between ranges [0; 0.25), [0.25; 0.75), [0.75; 1.0]
    vec3 selector_cos = vec3(0.0), selector_sin = vec3(0.0);
    selector_cos.x = float(a_cos < 0.25);
    selector_cos.z = float(a_cos >= 0.75);
    selector_cos.y = -1.0 + dot(selector_cos, vec3(1, 0, 1));
    selector_sin.x = float(a_sin < 0.25);
    selector_sin.z = float(a_sin >= 0.75);
    selector_sin.y = -1.0 + dot(selector_sin, vec3(1, 0, 1));

    // Center around ranges
    vec3 arg_cos = vec3(a_cos) - vec3(0.0, 0.5, 1.0);
    vec3 arg_sin = vec3(a_sin) - vec3(0.0, 0.5, 1.0);
    // Squared value gives better precision
    arg_cos *= arg_cos;
    arg_sin *= arg_sin;

    // Evaluate 5th-degree polynome
    vec3 res_cos = fma(vec3(-25.0407296503853054), arg_cos, vec3(60.1524123580209817));
    vec3 res_sin = fma(vec3(-25.0407296503853054), arg_sin, vec3(60.1524123580209817));
    res_cos = fma(res_cos, arg_cos, vec3(-85.4539888046442542));
    res_sin = fma(res_sin, arg_sin, vec3(-85.4539888046442542));
    res_cos = fma(res_cos, arg_cos, vec3(64.9393549651994562));
    res_sin = fma(res_sin, arg_sin, vec3(64.9393549651994562));
    res_cos = fma(res_cos, arg_cos, vec3(-19.7392086060579359));
    res_sin = fma(res_sin, arg_sin, vec3(-19.7392086060579359));
    res_cos = fma(res_cos, arg_cos, vec3(0.9999999998415476));
    res_sin = fma(res_sin, arg_sin, vec3(0.9999999998415476));

    // Combine contributions based on selector to get final value
    return vec2(dot(res_sin, selector_sin), dot(res_cos, selector_cos));
}

//
// asinf/acosf implemantation. Taken from apple libm source code
//

// Return arcsine(x) given that .57 < x
float asin_tail(const float x) {
    return (PI / 2) - ((x + 2.71745038) * x + 14.0375338) * (0.00440413551 * ((x - 8.31223679) * x + 25.3978882)) *
                          sqrt(1 - x);
}

// Taken from apple libm source code
float portable_asinf(float x) {
    const bool negate = (x < 0.0);
    if (abs(x) > 0.57) {
        const float ret = asin_tail(abs(x));
        return negate ? -ret : ret;
    } else {
        const float x2 = x * x;
        return float(x + (0.0517513789 * ((x2 + 1.83372748) * x2 + 1.56678128)) * x *
                             (x2 * ((x2 - 1.48268414) * x2 + 2.05554748)));
    }
}

float acos_positive_tail(const float x) {
    return (((x + 2.71850395) * x + 14.7303705)) * (0.00393401226 * ((x - 8.60734272) * x + 27.0927486)) *
           sqrt(1 - x);
}

float acos_negative_tail(const float x) {
    return PI - (((x - 2.71850395) * x + 14.7303705)) * (0.00393401226 * ((x + 8.60734272) * x + 27.0927486)) *
                    sqrt(1 + x);
}

float portable_acosf(float x) {
    if (x < -0.62) {
        return acos_negative_tail(x);
    } else if (x <= 0.62) {
        const float x2 = x * x;
        return (PI / 2) - x -
               (0.0700945929 * x * ((x2 + 1.57144082) * x2 + 1.25210774)) *
                   (x2 * ((x2 - 1.53757966) * x2 + 1.89929986));
    } else {
        return acos_positive_tail(x);
    }
}

#endif // TRIG_GLSL
