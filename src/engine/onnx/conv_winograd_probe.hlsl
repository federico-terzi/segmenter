// Research F(2x2,3x3), Conv_290 + ReLU. FP32 transforms/accumulation.
// One group produces 32x16 pixels and four output channels. A four-channel
// shared tile bounds shared-memory use while keeping 128 threads per group.
ByteAddressBuffer Src : register(t0);
StructuredBuffer<float> Weight : register(t3);
RWStructuredBuffer<float> Out : register(u0);
cbuffer Dimensions : register(b0) {uint Width; uint Height;}
groupshared float Tile[4*612];
float half_at(uint i) {return f16tof32((Src.Load((i>>1)*4)>>((i&1)*16))&65535);}
[numthreads(16,8,1)]
void main(uint3 local:SV_GroupThreadID,uint3 group:SV_GroupID) {
    uint lane=local.y*16+local.x,n=Width*Height,o=group.z*4;
    uint2 pixel=group.xy*uint2(32,16)+local.xy*2;
    float4 m[16];
    [unroll] for(uint k=0;k<16;k++)m[k]=0;
    [loop] for(uint base=0;base<35;base+=4) {
        for(uint i=lane;i<4*612;i+=128) {
            uint c=base+i/612,t=i%612;
            int2 xy=int2(group.xy*uint2(32,16))+int2(t%34,t/34)-1;
            float value=0;
            if(c<35 && xy.x>=0 && xy.y>=0 && xy.x<int(Width) && xy.y<int(Height))value=half_at(c*n+xy.y*Width+xy.x);
            Tile[i]=value;
        }
        GroupMemoryBarrierWithGroupSync();
        [loop] for(uint c=base;c<min(base+4,35);c++) {
        float4 d[4],t[4];
        [unroll] for(uint y=0;y<4;y++) {
            uint i=(c-base)*612+(local.y*2+y)*34+local.x*2;
            d[y]=float4(Tile[i],Tile[i+1],Tile[i+2],Tile[i+3]);
        }
        t[0]=d[0]-d[2];t[1]=d[1]+d[2];t[2]=d[2]-d[1];t[3]=d[1]-d[3];
        [unroll] for(uint y=0;y<4;y++) {
            float4 v=float4(t[y].x-t[y].z,t[y].y+t[y].z,t[y].z-t[y].y,t[y].y-t[y].w);
            [unroll] for(uint x=0;x<4;x++) {
                uint k=y*4+x,i=(o*35+c)*16+k;
                float4 u=float4(Weight[i],Weight[i+560],Weight[i+1120],Weight[i+1680]);
                m[k]=mad(v[x],u,m[k]);
            }
        }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float4 a[4],b[4];
    [unroll] for(uint x=0;x<4;x++) {a[x]=m[x]+m[x+4]+m[x+8];b[x]=m[x+4]-m[x+8]-m[x+12];}
    float4 bias=float4(Weight[8960+o],Weight[8961+o],Weight[8962+o],Weight[8963+o]);
    float4 value[4];
    value[0]=a[0]+a[1]+a[2]+bias;value[1]=a[1]-a[2]-a[3]+bias;
    value[2]=b[0]+b[1]+b[2]+bias;value[3]=b[1]-b[2]-b[3]+bias;
    [unroll] for(uint k=0;k<4;k++) {
        uint2 p=pixel+uint2(k%2,k/2);
        if(p.x<Width && p.y<Height) {
            [unroll] for(uint j=0;j<4;j++)Out[(o+j)*n+p.y*Width+p.x]=max(0,f16tof32(f32tof16(value[k][j])));
        }
    }
}
