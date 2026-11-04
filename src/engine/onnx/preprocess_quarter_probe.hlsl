// Original FP16 RGB normalization and Resize_3 (.25, half-pixel bilinear).
// Two adjacent low-resolution pixels per thread avoid half-word write races.
#include "fp16_round.hlsli"
ByteAddressBuffer Bgra : register(t0);
RWByteAddressBuffer Low : register(u0);
cbuffer Dimensions:register(b0){uint Width;uint Height;}
float q(float v){return round_half_rne(v);}
float3 q3(float3 v){return float3(q(v.x),q(v.y),q(v.z));}
#ifndef ROUNDING_VARIANT
#define ROUNDING_VARIANT 0
#endif
#ifndef PREPROCESS_SCALE
#define PREPROCESS_SCALE 4
#endif
float3 rgb(uint2 xy){uint v=Bgra.Load((xy.y*Width+xy.x)*4);return float3(q(float((v>>16)&255)/255.),q(float((v>>8)&255)/255.),q(float(v&255)/255.));}
float3 low(uint2 xy){uint2 p=xy*4+1;float3 a=rgb(p),b=rgb(p+uint2(1,0)),c=rgb(p+uint2(0,1)),d=rgb(p+1);
#if ROUNDING_VARIANT == 1
return (q3((a+b)*.5)+q3((c+d)*.5))*.5;
#elif ROUNDING_VARIANT == 2
return (q3((a+c)*.5)+q3((b+d)*.5))*.5;
#elif ROUNDING_VARIANT == 3
return q3(q3(a*.25+b*.25)+c*.25)+d*.25;
#else
return (a+b+c+d)*.25;
#endif
}
[numthreads(8,8,1)]
void main(uint3 id:SV_DispatchThreadID){
    uint2 xy=uint2(id.x*2,id.y);uint w=Width/PREPROCESS_SCALE,h=Height/PREPROCESS_SCALE;
    if(xy.x>=w || xy.y>=h)return;
#if PREPROCESS_SCALE == 1
    float3 a=rgb(xy),b=rgb(xy+uint2(1,0));
#else
    float3 a=low(xy),b=low(xy+uint2(1,0));
#endif
    uint p=xy.y*w+xy.x,n=w*h;
    [unroll]for(uint c=0;c<3;c++)Low.Store((c*n+p)*2,float_to_half_rne(a[c])|(float_to_half_rne(b[c])<<16));
}
