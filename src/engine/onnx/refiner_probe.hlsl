// RVM deep guided refinement at ratio=1. Research only, FP16 storage/rounding.
// Compile with USE_SHARED=0 or 1. Every learned box/MLP weight is retained.
#include "fp16_round.hlsli"
#ifndef USE_SHARED
#define USE_SHARED 1
#endif
#ifndef REFINER_SCALE
#define REFINER_SCALE 1
#endif
#ifndef OUTPUT_COEFFICIENTS
#define OUTPUT_COEFFICIENTS 0
#endif
ByteAddressBuffer Src : register(t0);
ByteAddressBuffer Hidden : register(t1);
ByteAddressBuffer Projection : register(t2);
StructuredBuffer<float> Weight : register(t3);
RWStructuredBuffer<float> Alpha : register(u0);
cbuffer Dimensions : register(b0) { uint FineWidth; uint FineHeight; }
#define Width (FineWidth/REFINER_SCALE)
#define Height (FineHeight/REFINER_SCALE)
float q(float x) { return round_half_rne(x); }
float4 q4(float4 x) { return float4(q(x.x),q(x.y),q(x.z),q(x.w)); }
float load_half(ByteAddressBuffer data, uint i) {
    uint word=data.Load((i>>1)*4);
    return f16tof32((word>>((i&1)*16))&65535);
}
void load_pixel(int2 xy,out float4 x,out float4 y) {
    x=0; y=0;
    if (xy.x>=0 && xy.y>=0 && xy.x<int(Width) && xy.y<int(Height)) {
        uint n=Width*Height, p=xy.y*Width+xy.x;
        x.xyz=float3(load_half(Src,p),load_half(Src,n+p),load_half(Src,2*n+p));
        x.w=q((x.x+x.y+x.z)/3.0);
        y=float4(load_half(Projection,p),load_half(Projection,n+p),load_half(Projection,2*n+p),load_half(Projection,3*n+p));
    }
}
#if USE_SHARED
groupshared float4 TileX[180];
groupshared float4 TileY[180];
#endif
[numthreads(16,8,1)]
void main(uint3 id:SV_DispatchThreadID,uint3 local:SV_GroupThreadID,uint3 group:SV_GroupID) {
#if USE_SHARED
    for(uint i=local.y*16+local.x;i<180;i+=128) {
        float4 x,y;
        load_pixel(int2(group.xy*uint2(16,8))+int2(i%18,i/18)-1,x,y);
        TileX[i]=x; TileY[i]=y;
    }
    GroupMemoryBarrierWithGroupSync();
#endif
    if(id.x>=Width || id.y>=Height) return;
    float4 mx=0,my=0,mxy=0,mxx=0;
    [unroll] for(uint j=0;j<9;j++) {
        float4 x,y;
#if USE_SHARED
        uint i=(local.y+j/3)*18+local.x+j%3;
        x=TileX[i]; y=TileY[i];
#else
        load_pixel(int2(id.xy)+int2(j%3,j/3)-1,x,y);
#endif
        float4 w=float4(Weight[j],Weight[9+j],Weight[18+j],Weight[27+j]);
        mx=mad(w,x,mx); my=mad(w,y,my);
        mxy=mad(w,q4(x*y),mxy); mxx=mad(w,q4(x*x),mxx);
    }
    mx=q4(mx); my=q4(my);
    float4 cov=q4(q4(mxy)-q4(mx*my));
    float4 var=q4(q4(mxx)-q4(mx*mx));
    float features[24], a[16], b[16];
    [unroll] for(uint c=0;c<4;c++) {features[c]=cov[c]; features[c+4]=var[c];}
    uint p=id.y*Width+id.x,n=Width*Height;
    [unroll] for(uint c=0;c<16;c++) features[c+8]=load_half(Hidden,c*n+p);
    [unroll] for(uint o=0;o<16;o++) {
        float value=Weight[420+o];
        [unroll] for(uint c=0;c<24;c++) value=mad(features[c],Weight[36+o*24+c],value);
        a[o]=max(0,q(value));
    }
    [unroll] for(uint o=0;o<16;o++) {
        float value=Weight[692+o];
        [unroll] for(uint c=0;c<16;c++) value=mad(a[c],Weight[436+o*16+c],value);
        b[o]=max(0,q(value));
    }
    float coefficient=Weight[724];
    [unroll] for(uint c=0;c<16;c++) coefficient=mad(b[c],Weight[708+c],coefficient);
    coefficient=q(coefficient);
    float bias=q(my.w-q(coefficient*mx.w));
#if OUTPUT_COEFFICIENTS
    Alpha[p]=coefficient;
    Alpha[n+p]=bias;
#else
    float4 fine,y_unused; load_pixel(int2(id.xy),fine,y_unused);
    Alpha[p]=clamp(q(q(coefficient*fine.w)+bias),0,1);
#endif
}
