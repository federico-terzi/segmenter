// Explicit IEEE binary16 round-to-nearest, ties-to-even. The legacy f32tof16
// instruction on the tested GPU truncated normalized image values downward.
// Do not rely on that instruction's rounding when matching CPU/ONNX FP16.
// NATIVE_HALF_RNE is an opt-in, adapter-qualified shortcut: it passed on the
// tested NVIDIA GPU but FAILED the input audit on AMD 780M. Portable is default.
#ifndef NATIVE_HALF_RNE
#define NATIVE_HALF_RNE 0
#endif
#if NATIVE_HALF_RNE
uint float_to_half_rne(float value){return f32tof16(float(float16_t(value)));}
float round_half_rne(float value){return float(float16_t(value));}
#else
uint float_to_half_rne(float value) {
    uint bits=asuint(value),sign=(bits>>16)&0x8000u;
    uint exponent=(bits>>23)&255u,mantissa=bits&0x7fffffu;
    if(exponent==255u)return sign|0x7c00u|(mantissa!=0u?0x200u:0u);
    if(exponent>=143u)return sign|0x7c00u;
    if(exponent<102u)return sign;
    if(exponent<=112u){
        uint significand=mantissa|0x800000u,shift=126u-exponent;
        uint rounded=significand+((1u<<(shift-1u))-1u)+((significand>>shift)&1u);
        return sign|(rounded>>shift);
    }
    uint rounded=mantissa+0xfffu+((mantissa>>13)&1u);
    return sign|(((exponent-112u)<<10)+(rounded>>13));
}
float round_half_rne(float value){
    uint bits=asuint(value),magnitude=bits&0x7fffffffu;
    // Most network activations are normal binary16 values: round their low
    // mantissa bits directly, without a legacy conversion instruction.
    [branch]if(magnitude>=0x38800000u && magnitude<0x477ff000u)
        return asfloat((bits+0xfffu+((bits>>13)&1u))&0xffffe000u);
    return f16tof32(float_to_half_rne(value));
}
#endif
