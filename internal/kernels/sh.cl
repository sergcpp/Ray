R"(

__constant float SH_Y0 = 0.282094806f; // sqrt(1.0f / (4.0f * PI))
__constant float SH_Y1 = 0.488602519f; // sqrt(3.0f / (4.0f * PI))

__constant float SH_A0 = 0.886226952f; // PI / sqrt(4.0f * Pi)
__constant float SH_A1 = 1.02332675f;  // sqrt(PI / 3.0f)

__constant float SH_AY0 = 0.25f; // SH_A0 * SH_Y0
__constant float SH_AY1 = 0.5f;  // SH_A1 * SH_Y1

float4 SH_EvaluateL1(const float3 v) {
    return (float4)(SH_Y0, SH_Y1 * v.y, SH_Y1 * v.z, SH_Y1 * v.x);
}

float4 SH_ApplyDiffuseConvolutionL1(const float4 coeff) {
    return coeff * (float4)(SH_A0, SH_A1, SH_A1, SH_A1);
}

float4 SH_EvaluateDiffuseL1(const float3 v) {
    return (float4)(SH_AY0, SH_AY1 * v.y, SH_AY1 * v.z, SH_AY1 * v.x);
}

)"