#ifndef TRIG_GLSL
#define TRIG_GLSL

//
// sinf/cosf implemantation. Taken from cgfx documentation
//

float portable_cosf(const float a) {
    // C simulation gives a max absolute error of less than 1.8e-7
    const vec4 c0 = vec4(0.0,            0.5,            1.0,            0.0           );
    const vec4 c1 = vec4(0.25,          -9.0,            0.75,           0.159154943091);
    const vec4 c2 = vec4(24.9808039603, -24.9808039603, -60.1458091736,  60.1458091736 );
    const vec4 c3 = vec4(85.4537887573, -85.4537887573, -64.9393539429,  64.9393539429 );
    const vec4 c4 = vec4(19.7392082214, -19.7392082214, -1.0,            1.0           );

    vec3 r0, r1, r2;

    r1.x  = c1.w * a;                                 // normalize input
    r1.y  = fract(r1.x);                              // and extract fraction
    r2.x  = float(r1.y < c1.x);                       // range check: 0.0 to 0.25
    r2.yz = vec2(greaterThanEqual(r1.yy, c1.yz));     // range check: 0.75 to 1.0
    r2.y  = dot(r2, c4.zwz);                          // range check: 0.25 to 0.75
    r0    = c0.xyz - r1.yyy;                          // range centering
    r0    = r0 * r0;
    r1    = c2.xyx * r0 + c2.zwz;                     // start power series
    r1    =     r1 * r0 + c3.xyx;
    r1    =     r1 * r0 + c3.zwz;
    r1    =     r1 * r0 + c4.xyx;
    r1    =     r1 * r0 + c4.zwz;
    r0.x  = dot(r1, -r2);                             // range extract

    return r0.x;
}

float portable_sinf(const float a) {
    // C simulation gives a max absolute error of less than 1.8e-7
    const vec4 c0 = vec4(0.0,            0.5,            1.0,            0.0           );
    const vec4 c1 = vec4(0.25,          -9.0,            0.75,           0.159154943091);
    const vec4 c2 = vec4(24.9808039603, -24.9808039603, -60.1458091736,  60.1458091736 );
    const vec4 c3 = vec4(85.4537887573, -85.4537887573, -64.9393539429,  64.9393539429 );
    const vec4 c4 = vec4(19.7392082214, -19.7392082214, -1.0,            1.0           );

    vec3 r0, r1, r2;

    r1.x  = c1.w * a - c1.x;                          // only difference from cos!
    r1.y  = fract(r1.x);                              // and extract fraction
    r2.x  = float(r1.y < c1.x);                       // range check: 0.0 to 0.25
    r2.yz = vec2(greaterThanEqual(r1.yy, c1.yz));     // range check: 0.75 to 1.0
    r2.y  = dot(r2, c4.zwz);                          // range check: 0.25 to 0.75
    r0    = c0.xyz - r1.yyy;                          // range centering
    r0    = r0 * r0;
    r1    = c2.xyx * r0 + c2.zwz;                     // start power series
    r1    =     r1 * r0 + c3.xyx;
    r1    =     r1 * r0 + c3.zwz;
    r1    =     r1 * r0 + c4.xyx;
    r1    =     r1 * r0 + c4.zwz;
    r0.x  = dot(r1, -r2);                             // range extract

    return r0.x;
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
