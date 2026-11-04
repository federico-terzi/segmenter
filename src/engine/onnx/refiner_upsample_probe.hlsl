// Ratio .25: original pytorch_half_pixel bilinear resize of alpha A and b,
// followed by the FP16-rounded fine-guidance expression and clipping.
#include "fp16_round.hlsli"
ByteAddressBuffer Src : register(t0);
StructuredBuffer<float> Coefficient : register(t1);
#ifndef PACKED_MASK
#define PACKED_MASK 0
#endif
#if PACKED_MASK
RWStructuredBuffer<uint> Alpha : register(u0);
#else
RWStructuredBuffer<float> Alpha : register(u0);
#endif
cbuffer Dimensions : register(b0) {uint Width;uint Height;}
float q(float v){return round_half_rne(v);}
float half_at(uint i){return f16tof32((Src.Load((i>>1)*4)>>((i&1)*16))&65535);}
float sample(uint c,int2 xy){uint w=Width/4,h=Height/4;xy=clamp(xy,int2(0,0),int2(w-1,h-1));return Coefficient[c*w*h+xy.y*w+xy.x];}
[numthreads(16,8,1)]
void main(uint3 id:SV_DispatchThreadID) {
    if(id.x>=Width || id.y>=Height)return;
    float2 coord=(float2(id.xy)+.5)*float2(Width/4,Height/4)/float2(Width,Height)-.5;
    int2 xy=int2(floor(coord));float2 f=coord-float2(xy);float a[2];
    [unroll]for(uint c=0;c<2;c++)a[c]=q(lerp(lerp(sample(c,xy),sample(c,xy+int2(1,0)),f.x),lerp(sample(c,xy+int2(0,1)),sample(c,xy+1),f.x),f.y));
    uint n=Width*Height,p=id.y*Width+id.x;
#if BGRA_SOURCE
    uint pixel=Src.Load(p*4);
    float gray=q((q(float(pixel&255)/255.)+q(float((pixel>>8)&255)/255.)+q(float((pixel>>16)&255)/255.))/3.);
#else
    float gray=q((half_at(p)+half_at(n+p)+half_at(2*n+p))/3.);
#endif
    float value=q(q(a[0]*gray)+a[1]);
#if PACKED_MASK
    uint byte_value=(uint)floor(clamp(value,0,1)*255.+.5);
    Alpha[p]=isfinite(value)?(byte_value*0x010101u|0xff000000u):0;
#else
    Alpha[p]=clamp(value,0,1);
#endif
}
