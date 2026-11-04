// Research Conv_290: share one complete input halo, four output channels/thread.
// Deliberately bounded register pressure and just one group barrier.
ByteAddressBuffer Src : register(t0);
StructuredBuffer<float> Weight : register(t3);
RWStructuredBuffer<float> Out : register(u0);
cbuffer Dimensions : register(b0) {uint Width; uint Height;}
groupshared float Tile[35*180];
float half_at(uint i) {return f16tof32((Src.Load((i>>1)*4)>>((i&1)*16))&65535);}
[numthreads(16,8,1)]
void main(uint3 id:SV_DispatchThreadID,uint3 local:SV_GroupThreadID,uint3 group:SV_GroupID) {
    uint lane=local.y*16+local.x, n=Width*Height, o=group.z*4;
    for(uint i=lane;i<35*180;i+=128) {
        uint c=i/180,t=i%180;
        int2 xy=int2(group.xy*uint2(16,8))+int2(t%18,t/18)-1;
        float value=0;
        if(xy.x>=0 && xy.y>=0 && xy.x<int(Width) && xy.y<int(Height)) value=half_at(c*n+xy.y*Width+xy.x);
        Tile[i]=value;
    }
    GroupMemoryBarrierWithGroupSync();
    if(id.x>=Width || id.y>=Height) return;
    float4 sum=float4(Weight[5040+o],Weight[5041+o],Weight[5042+o],Weight[5043+o]);
    [loop] for(uint c=0;c<35;c++) {
        [unroll] for(uint k=0;k<9;k++) {
            float value=Tile[c*180+(local.y+k/3)*18+local.x+k%3];
            uint i=o*315+c*9+k;
            float4 w=float4(Weight[i],Weight[i+315],Weight[i+630],Weight[i+945]);
            sum=mad(value,w,sum);
        }
    }
    uint p=id.y*Width+id.x;
    [unroll] for(uint j=0;j<4;j++) Out[(o+j)*n+p]=max(0,f16tof32(f32tof16(sum[j])));
}
