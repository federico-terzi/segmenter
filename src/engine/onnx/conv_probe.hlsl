// Tiled dense RVM output convolution + ReLU. Model-specific research baseline.
// 35 input channels, 16 outputs, 3x3, stride 1, zero padding. FP32 accumulates.
ByteAddressBuffer Src : register(t0);
StructuredBuffer<float> Weight : register(t3);
RWStructuredBuffer<float> Out : register(u0);
cbuffer Dimensions : register(b0) {uint Width; uint Height;}
groupshared float Tile[4*180];
groupshared float Filter[16*4*9];
float load_half(uint i) {return f16tof32((Src.Load((i>>1)*4)>>((i&1)*16))&65535);}
[numthreads(16,8,1)]
void main(uint3 id:SV_DispatchThreadID,uint3 local:SV_GroupThreadID,uint3 group:SV_GroupID) {
    uint lane=local.y*16+local.x, n=Width*Height;
    float accum[16];
    [unroll] for(uint o=0;o<16;o++) accum[o]=Weight[5040+o];
    [loop] for(uint base=0;base<35;base+=4) {
        for(uint i=lane;i<720;i+=128) {
            uint c=base+i/180, t=i%180;
            int2 xy=int2(group.xy*uint2(16,8))+int2(t%18,t/18)-1;
            float value=0;
            if(c<35 && xy.x>=0 && xy.y>=0 && xy.x<int(Width) && xy.y<int(Height)) value=load_half(c*n+xy.y*Width+xy.x);
            Tile[i]=value;
        }
        for(uint i=lane;i<576;i+=128) {
            uint o=i/36,c=base+(i%36)/9,k=i%9;
            Filter[i]=c<35 ? Weight[o*35*9+c*9+k] : 0;
        }
        GroupMemoryBarrierWithGroupSync();
        [unroll] for(uint c=0;c<4;c++) {
            [unroll] for(uint k=0;k<9;k++) {
                float value=Tile[c*180+(local.y+k/3)*18+local.x+k%3];
                [unroll] for(uint o=0;o<16;o++) accum[o]=mad(value,Filter[o*36+c*9+k],accum[o]);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if(id.x<Width && id.y<Height) {
        [unroll] for(uint o=0;o<16;o++) Out[o*n+id.y*Width+id.x]=max(0,f16tof32(f32tof16(accum[o])));
    }
}
