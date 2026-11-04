#include "fp16_round.hlsli"
ByteAddressBuffer Src:register(t0);
RWStructuredBuffer<uint> Out:register(u0);
cbuffer Dimensions:register(b0){uint Width;uint Height;}
[numthreads(16,8,1)]
void main(uint3 id:SV_DispatchThreadID){
    if(id.x>=Width || id.y>=Height)return;
    uint p=id.y*Width+id.x;float value=asfloat(Src.Load(p*4));
    Out[p]=float_to_half_rne(value)|(f32tof16(round_half_rne(value))<<16);
}
