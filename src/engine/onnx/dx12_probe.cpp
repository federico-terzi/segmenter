// Research-only DirectX timing/shader harness. Uses the installed Windows SDK.
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <DirectML.h>
#include <wrl/client.h>
#include <array>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
using Microsoft::WRL::ComPtr;
static thread_local std::string last_error;
static void check(HRESULT hr) {
    if (FAILED(hr)) { char s[48]; sprintf_s(s, "DirectX HRESULT 0x%08x", unsigned(hr)); throw std::runtime_error(s); }
}
struct Probe {
    ComPtr<ID3D12Device> device;
    ComPtr<IDMLDevice> dml;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12QueryHeap> queries;
    ComPtr<ID3D12Resource> timestamps;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> first, last, shader, copy_output;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pso;
    ComPtr<ID3D12PipelineState> post_pso;
    ComPtr<ID3D12Resource> intermediate, fine_src;
    ComPtr<ID3D12Resource> bgra, bgra_upload, zero_upload;
    ComPtr<ID3D12PipelineState> prepare_pso;
    ComPtr<ID3D12GraphicsCommandList> prepare_cmd, recurrence_cmd, reset_cmd;
    std::vector<ComPtr<ID3D12Resource>> state_resources;
    std::vector<size_t> state_copy_sizes;
    void* mapped_bgra=nullptr;
    std::vector<ComPtr<ID3D12GraphicsCommandList>> ring_prepare,ring_copy;
    std::vector<ComPtr<ID3D12Resource>> ring_readback;
    bool read_batch_complete=false;
    bool temporal_encoder=false;
    std::vector<ComPtr<ID3D12GraphicsCommandList>> ring_shader,feature_slices;
    std::vector<ComPtr<ID3D12Resource>> feature_resources;
    std::vector<ComPtr<ID3D12Resource>> inputs;
    ComPtr<ID3D12Resource> output, readback;
    UINT64 frequency=0, fence_value=0;
    HANDLE event_handle=nullptr;
    UINT width=0, height=0, output_channels=1;
    UINT dispatch_depth=1;
    UINT group_width=16,group_height=8;
    D3D12_COMMAND_LIST_TYPE command_type=D3D12_COMMAND_LIST_TYPE_DIRECT;
    std::string shader_path;
    ~Probe() { if (queue && fence && event_handle) {try {wait();} catch(...) {}} if(mapped_bgra)bgra_upload->Unmap(0,nullptr); if (event_handle) CloseHandle(event_handle); }
    ComPtr<ID3D12Resource> buffer(UINT64 bytes, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES state, bool uav=false) {
        D3D12_HEAP_PROPERTIES hp{}; hp.Type=heap;
        D3D12_RESOURCE_DESC desc{}; desc.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width=(bytes+3)&~UINT64(3); desc.Height=1; desc.DepthOrArraySize=1; desc.MipLevels=1;
        desc.SampleDesc.Count=1; desc.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc.Flags=uav?D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS:D3D12_RESOURCE_FLAG_NONE;
        ComPtr<ID3D12Resource> result;
        check(device->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&desc,state,nullptr,IID_PPV_ARGS(&result)));
        return result;
    }
    ComPtr<ID3D12GraphicsCommandList> list() {
        ComPtr<ID3D12GraphicsCommandList> r;
        check(device->CreateCommandList(0,command_type,allocator.Get(),nullptr,IID_PPV_ARGS(&r)));
        return r;
    }
    void submit(ID3D12GraphicsCommandList* list) { ID3D12CommandList* lists[]={list}; queue->ExecuteCommandLists(1,lists); }
    void wait() {
        check(queue->Signal(fence.Get(),++fence_value));
        if (fence->GetCompletedValue()<fence_value) {
            check(fence->SetEventOnCompletion(fence_value,event_handle));
            if (WaitForSingleObject(event_handle,30000)!=WAIT_OBJECT_0) throw std::runtime_error("GPU fence timeout");
        }
        read_batch_complete=true;
    }
    static D3D12_RESOURCE_BARRIER transition(ID3D12Resource* r,D3D12_RESOURCE_STATES from,D3D12_RESOURCE_STATES to) {
        D3D12_RESOURCE_BARRIER b{}; b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition={r,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,from,to}; return b;
    }
    void init(int adapter_index) {
        if(GetEnvironmentVariableA("SEGMENTER_DX12_COMPUTE",nullptr,0))command_type=D3D12_COMMAND_LIST_TYPE_COMPUTE;
        ComPtr<IDXGIFactory4> factory; check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        ComPtr<IDXGIAdapter1> adapter; check(factory->EnumAdapters1(adapter_index,&adapter));
        DXGI_ADAPTER_DESC1 ad{}; check(adapter->GetDesc1(&ad));
        fwprintf(stderr,L"DX12 adapter=%s\n",ad.Description);
        check(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&device)));
        D3D12_COMMAND_QUEUE_DESC q{}; q.Type=command_type;
        check(device->CreateCommandQueue(&q,IID_PPV_ARGS(&queue)));
        check(DMLCreateDevice1(device.Get(),DML_CREATE_DEVICE_FLAG_NONE,DML_FEATURE_LEVEL_5_0,IID_PPV_ARGS(&dml)));
        check(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)));
        event_handle=CreateEventW(nullptr,FALSE,FALSE,nullptr);
        if (!event_handle) throw std::runtime_error("CreateEvent failed");
        check(queue->GetTimestampFrequency(&frequency));
        D3D12_QUERY_HEAP_DESC query{}; query.Type=D3D12_QUERY_HEAP_TYPE_TIMESTAMP; query.Count=2;
        check(device->CreateQueryHeap(&query,IID_PPV_ARGS(&queries)));
        timestamps=buffer(16,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
        check(device->CreateCommandAllocator(command_type,IID_PPV_ARGS(&allocator)));
        first=list(); first->EndQuery(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0); check(first->Close());
        last=list(); last->EndQuery(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,1);
        last->ResolveQueryData(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0,2,timestamps.Get(),0); check(last->Close());
    }
    double end() {
        submit(last.Get()); wait();
        UINT64* data=nullptr; D3D12_RANGE range{0,16}; check(timestamps->Map(0,&range,reinterpret_cast<void**>(&data)));
        auto ticks=data[1]-data[0]; D3D12_RANGE written{0,0}; timestamps->Unmap(0,&written);
        return double(ticks)*1000.0/double(frequency);
    }
    ComPtr<ID3D12Resource> upload(const void* data,size_t bytes) {
        auto staging=buffer(bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
        auto gpu=buffer(bytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_COPY_DEST);
        void* p=nullptr; D3D12_RANGE empty{0,0}; check(staging->Map(0,&empty,&p)); std::memcpy(p,data,bytes); staging->Unmap(0,nullptr);
        auto cmd=list(); cmd->CopyBufferRegion(gpu.Get(),0,staging.Get(),0,bytes);
        auto barrier=transition(gpu.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        cmd->ResourceBarrier(1,&barrier); check(cmd->Close()); submit(cmd.Get()); wait(); return gpu;
    }
    void setup(const char* code_path, const void* src,size_t src_bytes,const void* hid,size_t hid_bytes,
               const void* proj,size_t proj_bytes,const void* weights,size_t weight_bytes,UINT w,UINT h,UINT channels) {
        wait(); inputs.clear(); width=w; height=h; output_channels=channels; dispatch_depth=1; group_width=16; group_height=8;
        post_pso.Reset(); intermediate.Reset(); fine_src.Reset();
        if(mapped_bgra){bgra_upload->Unmap(0,nullptr);mapped_bgra=nullptr;}
        bgra.Reset(); bgra_upload.Reset(); zero_upload.Reset(); prepare_pso.Reset();
        prepare_cmd.Reset(); recurrence_cmd.Reset(); reset_cmd.Reset();
        state_resources.clear(); state_copy_sizes.clear();
        ring_prepare.clear(); ring_copy.clear(); ring_readback.clear();
        ring_shader.clear(); feature_slices.clear(); feature_resources.clear();
        temporal_encoder=false; read_batch_complete=false;
        if (shader_path!=code_path) {
        std::ifstream file(code_path,std::ios::binary);
        std::vector<char> code((std::istreambuf_iterator<char>(file)),{});
        if(code.empty()) throw std::runtime_error("empty shader bytecode");
        std::array<D3D12_ROOT_PARAMETER,6> params{};
        for(UINT i=0;i<4;i++) { params[i].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV; params[i].Descriptor.ShaderRegister=i; }
        params[4].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[5].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[5].Constants={0,0,2};
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters=UINT(params.size()); desc.pParameters=params.data();
        ComPtr<ID3DBlob> blob,errors; check(D3D12SerializeRootSignature(&desc,D3D_ROOT_SIGNATURE_VERSION_1,&blob,&errors));
        check(device->CreateRootSignature(0,blob->GetBufferPointer(),blob->GetBufferSize(),IID_PPV_ARGS(root.ReleaseAndGetAddressOf())));
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{}; pd.pRootSignature=root.Get(); pd.CS={code.data(),code.size()};
        check(device->CreateComputePipelineState(&pd,IID_PPV_ARGS(pso.ReleaseAndGetAddressOf())));
        shader_path=code_path;
        }
        inputs.push_back(upload(src,src_bytes)); inputs.push_back(upload(hid,hid_bytes));
        inputs.push_back(upload(proj,proj_bytes)); inputs.push_back(upload(weights,weight_bytes));
        output=buffer(UINT64(w)*h*channels*4,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,true);
        readback=buffer(UINT64(w)*h*channels*4,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
        copy_output=list();
        auto before=transition(output.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
        copy_output->ResourceBarrier(1,&before);
        copy_output->CopyBufferRegion(readback.Get(),0,output.Get(),0,UINT64(w)*h*channels*4);
        auto after=transition(output.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        copy_output->ResourceBarrier(1,&after); check(copy_output->Close());
        record_shader(false);
    }
    void record_shader(bool external,UINT fine_slot=0) {
        shader=list(); shader->SetComputeRootSignature(root.Get()); shader->SetPipelineState(pso.Get());
        if(external) for(UINT i=0;i<3;i++) {
            auto b=transition(inputs[i].Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            shader->ResourceBarrier(1,&b);
        }
        for(UINT i=0;i<4;i++) shader->SetComputeRootShaderResourceView(i,inputs[i]->GetGPUVirtualAddress());
        auto first_output=post_pso?intermediate.Get():output.Get();
        shader->SetComputeRootUnorderedAccessView(4,first_output->GetGPUVirtualAddress());
        UINT dims[]={width,height}; shader->SetComputeRoot32BitConstants(5,2,dims,0);
        shader->Dispatch((width+group_width-1)/group_width,(height+group_height-1)/group_height,dispatch_depth);
        if(post_pso) {
            auto t=transition(intermediate.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            shader->ResourceBarrier(1,&t);
            auto f=transition(fine_src.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            if(fine_src.Get()!=bgra.Get())shader->ResourceBarrier(1,&f);
            shader->SetPipelineState(post_pso.Get());
            shader->SetComputeRootShaderResourceView(0,fine_src->GetGPUVirtualAddress()+(temporal_encoder?UINT64(fine_slot)*width*height*4:0));
            shader->SetComputeRootShaderResourceView(1,intermediate->GetGPUVirtualAddress());
            shader->SetComputeRootUnorderedAccessView(4,output->GetGPUVirtualAddress());
            shader->Dispatch((width+15)/16,(height+7)/8,1);
            t=transition(intermediate.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            shader->ResourceBarrier(1,&t);
            f=transition(fine_src.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            if(fine_src.Get()!=bgra.Get())shader->ResourceBarrier(1,&f);
        }
        D3D12_RESOURCE_BARRIER b{}; b.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV; b.UAV.pResource=output.Get();
        shader->ResourceBarrier(1,&b);
        if(external) for(UINT i=0;i<3;i++) {
            auto t=transition(inputs[i].Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            shader->ResourceBarrier(1,&t);
        }
        append_state_copy(shader.Get());
        if(temporal_encoder){
            auto t=transition(output.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);shader->ResourceBarrier(1,&t);
            shader->CopyBufferRegion(ring_readback.at(fine_slot).Get(),0,output.Get(),0,UINT64(width)*height*4);
            t=transition(output.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);shader->ResourceBarrier(1,&t);
        }
        check(shader->Close());
    }
    void append_state_copy(ID3D12GraphicsCommandList* cmd){
        for(size_t i=0;i<state_copy_sizes.size();i++){
            auto in=state_resources[i*2].Get();auto out=state_resources[i*2+1].Get();
            auto a=transition(in,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_DEST);
            auto b=transition(out,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
            cmd->ResourceBarrier(1,&a);cmd->ResourceBarrier(1,&b);cmd->CopyBufferRegion(in,0,out,0,state_copy_sizes[i]);
            a=transition(in,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            b=transition(out,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);cmd->ResourceBarrier(1,&a);cmd->ResourceBarrier(1,&b);
        }
    }
    void attach(ID3D12Resource* src,ID3D12Resource* hid,ID3D12Resource* proj) {
        wait();
        inputs[0]=src; inputs[1]=hid; inputs[2]=proj;
        record_shader(true);
    }
    void post(const char* path,ID3D12Resource* fine) {
        wait();fine_src=fine;
        std::ifstream file(path,std::ios::binary);std::vector<char> code((std::istreambuf_iterator<char>(file)),{});
        if(code.empty())throw std::runtime_error("empty post shader");
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};pd.pRootSignature=root.Get();pd.CS={code.data(),code.size()};
        check(device->CreateComputePipelineState(&pd,IID_PPV_ARGS(post_pso.ReleaseAndGetAddressOf())));
        intermediate=buffer(UINT64(width)*height*4,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,true);
        group_width=64;group_height=32;record_shader(true);
        ring_shader.clear();
        if(temporal_encoder)for(UINT slot=0;slot<ring_prepare.size();slot++){record_shader(true,slot);ring_shader.push_back(shader);}
    }
    void prepare_setup(const char* path,ID3D12Resource* low,UINT scale,UINT slots,bool temporal) {
        if(scale!=1 && scale!=4)throw std::runtime_error("unsupported preprocessing scale");
        if(slots<1 || slots>16)throw std::runtime_error("invalid upload ring size");
        wait();if(width%8 || height<4)throw std::runtime_error("quarter input requires width divisible by 8");
        const UINT64 normalized_bytes=UINT64(width/scale)*(height/scale)*3*2*(temporal?slots:1);
        if(!low || low->GetDesc().Dimension!=D3D12_RESOURCE_DIMENSION_BUFFER || low->GetDesc().Width<normalized_bytes)
            throw std::runtime_error("preprocessing tensor is smaller than the requested shape");
        temporal_encoder=temporal;
        if(mapped_bgra){bgra_upload->Unmap(0,nullptr);mapped_bgra=nullptr;}
        bgra_upload=buffer(UINT64(width)*height*4*slots,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
        D3D12_RANGE empty{0,0};check(bgra_upload->Map(0,&empty,&mapped_bgra));
        bgra=buffer(UINT64(width)*height*4*(temporal?slots:1),D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        std::ifstream file(path,std::ios::binary);std::vector<char> code((std::istreambuf_iterator<char>(file)),{});
        if(code.empty())throw std::runtime_error("empty preprocessing shader");
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};pd.pRootSignature=root.Get();pd.CS={code.data(),code.size()};
        check(device->CreateComputePipelineState(&pd,IID_PPV_ARGS(prepare_pso.ReleaseAndGetAddressOf())));
        ring_prepare.clear();ring_copy.clear();ring_readback.clear();
        for(UINT slot=0;slot<slots;slot++){
        prepare_cmd=list();
        auto t=transition(bgra.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_DEST);prepare_cmd->ResourceBarrier(1,&t);
        UINT64 raw_offset=temporal?UINT64(slot)*width*height*4:0;
        UINT64 normalized_offset=temporal?UINT64(slot)*(width/scale)*(height/scale)*3*2:0;
        prepare_cmd->CopyBufferRegion(bgra.Get(),raw_offset,bgra_upload.Get(),UINT64(slot)*width*height*4,UINT64(width)*height*4);
        t=transition(bgra.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);prepare_cmd->ResourceBarrier(1,&t);
        prepare_cmd->SetComputeRootSignature(root.Get());prepare_cmd->SetPipelineState(prepare_pso.Get());
        prepare_cmd->SetComputeRootShaderResourceView(0,bgra->GetGPUVirtualAddress()+raw_offset);
        prepare_cmd->SetComputeRootUnorderedAccessView(4,low->GetGPUVirtualAddress()+normalized_offset);
        UINT dims[]={width,height};prepare_cmd->SetComputeRoot32BitConstants(5,2,dims,0);
        prepare_cmd->Dispatch((width/scale+15)/16,(height/scale+7)/8,1);
        D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV;b.UAV.pResource=low;prepare_cmd->ResourceBarrier(1,&b);
        check(prepare_cmd->Close());
        ring_prepare.push_back(prepare_cmd);
        auto dst=buffer(UINT64(width)*height*4,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
        auto cmd=list();t=transition(output.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);cmd->ResourceBarrier(1,&t);
        cmd->CopyBufferRegion(dst.Get(),0,output.Get(),0,UINT64(width)*height*4);
        t=transition(output.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);cmd->ResourceBarrier(1,&t);check(cmd->Close());
        ring_readback.push_back(dst);ring_copy.push_back(cmd);
        }
    }
    void prepare_frame(const void* data,size_t stride,UINT slot=0) {
        // The preceding readback fence guarantees the upload mapping is idle.
        if(!mapped_bgra || stride<size_t(width)*4 || slot>=ring_prepare.size())throw std::runtime_error("invalid BGRA upload");
        auto destination=static_cast<char*>(mapped_bgra)+size_t(slot)*width*height*4;
        if(stride==size_t(width)*4)std::memcpy(destination,data,size_t(width)*height*4);
        else for(UINT y=0;y<height;y++)std::memcpy(destination+size_t(y)*width*4,static_cast<const char*>(data)+size_t(y)*stride,size_t(width)*4);
        read_batch_complete=false;submit(ring_prepare[slot].Get());
    }
    void queue_read(UINT slot){if(slot>=ring_copy.size())throw std::runtime_error("invalid readback slot");read_batch_complete=false;submit(ring_copy[slot].Get());}
    void read_slot(UINT slot,void* data,size_t bytes){
        if(!read_batch_complete || slot>=ring_readback.size() || bytes!=size_t(width)*height*4)throw std::runtime_error("readback requires completed batch and correct size");
        void* mapped=nullptr;D3D12_RANGE range{0,bytes};check(ring_readback[slot]->Map(0,&range,&mapped));std::memcpy(data,mapped,bytes);D3D12_RANGE empty{0,0};ring_readback[slot]->Unmap(0,&empty);
    }
    void recurrence(void** ins,void** outs,const size_t* bytes,UINT count) {
        wait();state_resources.clear();state_copy_sizes.assign(bytes,bytes+count);recurrence_cmd=list();
        size_t max_bytes=4;for(UINT i=0;i<count;i++)max_bytes=std::max(max_bytes,bytes[i]);
        zero_upload=buffer(max_bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
        void* zero=nullptr;D3D12_RANGE empty{0,0};check(zero_upload->Map(0,&empty,&zero));std::memset(zero,0,max_bytes);zero_upload->Unmap(0,nullptr);
        for(UINT i=0;i<count;i++) {
            auto in=static_cast<ID3D12Resource*>(ins[i]);auto out=static_cast<ID3D12Resource*>(outs[i]);
            if(in==out || in->GetDesc().Width<bytes[i] || out->GetDesc().Width<bytes[i])throw std::runtime_error("invalid recurrent buffers");
            state_resources.emplace_back(in);state_resources.emplace_back(out);
            auto a=transition(in,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_DEST);
            auto b=transition(out,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
            recurrence_cmd->ResourceBarrier(1,&a);recurrence_cmd->ResourceBarrier(1,&b);
            recurrence_cmd->CopyBufferRegion(in,0,out,0,bytes[i]);
            a=transition(in,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            b=transition(out,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            recurrence_cmd->ResourceBarrier(1,&a);recurrence_cmd->ResourceBarrier(1,&b);
        }
        check(recurrence_cmd->Close());
        // One allocator may have only one command list recording at a time.
        reset_cmd=list();
        for(UINT i=0;i<count;i++){
            auto in=static_cast<ID3D12Resource*>(ins[i]);
            auto a=transition(in,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_DEST);
            reset_cmd->ResourceBarrier(1,&a);reset_cmd->CopyBufferRegion(in,0,zero_upload.Get(),0,bytes[i]);
            a=transition(in,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);reset_cmd->ResourceBarrier(1,&a);
        }
        check(reset_cmd->Close());submit(reset_cmd.Get());wait();
        if(temporal_encoder){ring_shader.clear();for(UINT slot=0;slot<ring_prepare.size();slot++){record_shader(true,slot);ring_shader.push_back(shader);}}
        else record_shader(true);
    }
    void feature_setup(void** sources,void** destinations,const size_t* bytes,UINT count,UINT slots){
        wait();feature_slices.clear();feature_resources.clear();
        for(UINT i=0;i<count;i++){
            auto src=static_cast<ID3D12Resource*>(sources[i]);auto dst=static_cast<ID3D12Resource*>(destinations[i]);
            if(src==dst || src->GetDesc().Width<bytes[i]*slots || dst->GetDesc().Width<bytes[i])throw std::runtime_error("invalid encoder feature slices");
            feature_resources.emplace_back(src);feature_resources.emplace_back(dst);
        }
        for(UINT slot=0;slot<slots;slot++){
            auto cmd=list();
            for(UINT i=0;i<count;i++){
                auto src=feature_resources[i*2].Get();auto dst=feature_resources[i*2+1].Get();
                auto a=transition(src,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
                auto b=transition(dst,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_DEST);
                cmd->ResourceBarrier(1,&a);cmd->ResourceBarrier(1,&b);cmd->CopyBufferRegion(dst,0,src,bytes[i]*slot,bytes[i]);
                a=transition(src,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                b=transition(dst,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                cmd->ResourceBarrier(1,&a);cmd->ResourceBarrier(1,&b);
            }
            check(cmd->Close());feature_slices.push_back(cmd);
        }
    }
    void read(float* destination,size_t count) {
        if(count!=size_t(width)*height*output_channels) throw std::runtime_error("wrong output size");
        submit(copy_output.Get()); wait();
        void* p=nullptr; D3D12_RANGE range{0,count*4}; check(readback->Map(0,&range,&p));
        std::memcpy(destination,p,count*4); D3D12_RANGE empty{0,0}; readback->Unmap(0,&empty);
    }
};
#define TRY_PROBE try
#define CATCH_PROBE catch(const std::exception& e) {last_error=e.what(); return -1;}
extern "C" {
const char* rvm_probe_error() { return last_error.c_str(); }
void* rvm_probe_create(int adapter) {
    auto p=new Probe;
    try {p->init(adapter); return p;} catch(const std::exception& e) {last_error=e.what(); delete p; return nullptr;}
}
void rvm_probe_destroy(void* p) {delete static_cast<Probe*>(p);}
void* rvm_probe_device(void* p) {return static_cast<Probe*>(p)->dml.Get();}
void* rvm_probe_queue(void* p) {return static_cast<Probe*>(p)->queue.Get();}
int rvm_probe_begin(void* p) {TRY_PROBE {auto c=static_cast<Probe*>(p); c->submit(c->first.Get()); return 0;} CATCH_PROBE}
int rvm_probe_end(void* p,double* ms) {TRY_PROBE {*ms=static_cast<Probe*>(p)->end(); return 0;} CATCH_PROBE}
int rvm_probe_setup(void* p,const char* code,const void* src,size_t src_size,const void* hid,size_t hid_size,
                    const void* proj,size_t proj_size,const void* weights,size_t weights_size,unsigned w,unsigned h,unsigned channels) {
    TRY_PROBE {static_cast<Probe*>(p)->setup(code,src,src_size,hid,hid_size,proj,proj_size,weights,weights_size,w,h,channels); return 0;} CATCH_PROBE
}
int rvm_probe_shader(void* p,unsigned repeats,double* ms) {
    TRY_PROBE {auto c=static_cast<Probe*>(p); c->submit(c->first.Get());
        for(unsigned i=0;i<repeats;i++) c->submit(c->shader.Get());
        *ms=c->end()/repeats; return 0;} CATCH_PROBE
}
int rvm_probe_read(void* p,float* out,size_t count) {TRY_PROBE {static_cast<Probe*>(p)->read(out,count); return 0;} CATCH_PROBE}
int rvm_probe_attach(void* p,void* src,void* hid,void* proj) {
    TRY_PROBE {static_cast<Probe*>(p)->attach(static_cast<ID3D12Resource*>(src),static_cast<ID3D12Resource*>(hid),static_cast<ID3D12Resource*>(proj)); return 0;} CATCH_PROBE
}
int rvm_probe_post(void* p,const char* path,void* fine) {
    TRY_PROBE {static_cast<Probe*>(p)->post(path,static_cast<ID3D12Resource*>(fine));return 0;} CATCH_PROBE
}
int rvm_probe_dispatch(void* p) {TRY_PROBE {auto c=static_cast<Probe*>(p); c->submit(c->shader.Get()); return 0;} CATCH_PROBE}
int rvm_probe_prepare_setup(void* p,const char* path,void* low,unsigned scale,unsigned slots,unsigned temporal){TRY_PROBE{static_cast<Probe*>(p)->prepare_setup(path,static_cast<ID3D12Resource*>(low),scale,slots,temporal!=0);return 0;}CATCH_PROBE}
int rvm_probe_prepare(void* p,const void* data,size_t stride){TRY_PROBE{static_cast<Probe*>(p)->prepare_frame(data,stride);return 0;}CATCH_PROBE}
int rvm_probe_prepare_slot(void* p,const void* data,size_t stride,unsigned slot){TRY_PROBE{static_cast<Probe*>(p)->prepare_frame(data,stride,slot);return 0;}CATCH_PROBE}
int rvm_probe_queue_read(void* p,unsigned slot){TRY_PROBE{static_cast<Probe*>(p)->queue_read(slot);return 0;}CATCH_PROBE}
int rvm_probe_wait(void* p){TRY_PROBE{static_cast<Probe*>(p)->wait();return 0;}CATCH_PROBE}
int rvm_probe_read_slot(void* p,unsigned slot,void* data,size_t bytes){TRY_PROBE{static_cast<Probe*>(p)->read_slot(slot,data,bytes);return 0;}CATCH_PROBE}
int rvm_probe_feature_setup(void* p,void** src,void** dst,const size_t* bytes,unsigned count,unsigned slots){TRY_PROBE{static_cast<Probe*>(p)->feature_setup(src,dst,bytes,count,slots);return 0;}CATCH_PROBE}
int rvm_probe_feature_slice(void* p,unsigned slot){TRY_PROBE{auto c=static_cast<Probe*>(p);c->submit(c->feature_slices.at(slot).Get());return 0;}CATCH_PROBE}
int rvm_probe_dispatch_slot(void* p,unsigned slot){TRY_PROBE{auto c=static_cast<Probe*>(p);c->submit(c->ring_shader.at(slot).Get());return 0;}CATCH_PROBE}
void* rvm_probe_bgra(void* p){return static_cast<Probe*>(p)->bgra.Get();}
int rvm_probe_recurrence(void* p,void** ins,void** outs,const size_t* bytes,unsigned count){TRY_PROBE{static_cast<Probe*>(p)->recurrence(ins,outs,bytes,count);return 0;}CATCH_PROBE}
int rvm_probe_reset_recurrence(void* p){TRY_PROBE{auto c=static_cast<Probe*>(p);c->submit(c->reset_cmd.Get());return 0;}CATCH_PROBE}
int rvm_probe_ping_pong(void* p){TRY_PROBE{auto c=static_cast<Probe*>(p);c->wait();c->state_copy_sizes.clear();
    c->ring_shader.clear();for(UINT slot=0;slot<c->ring_prepare.size();slot++){c->record_shader(true,slot);c->ring_shader.push_back(c->shader);}return 0;}CATCH_PROBE}
int rvm_probe_copy_tensor(void* p,void* tensor,void* data,size_t bytes){
    TRY_PROBE{auto c=static_cast<Probe*>(p);auto src=static_cast<ID3D12Resource*>(tensor);
        if(bytes>src->GetDesc().Width)throw std::runtime_error("diagnostic tensor too short");
        auto dst=c->buffer(bytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);auto cmd=c->list();
        auto b=Probe::transition(src,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);cmd->ResourceBarrier(1,&b);
        cmd->CopyBufferRegion(dst.Get(),0,src,0,bytes);
        b=Probe::transition(src,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);cmd->ResourceBarrier(1,&b);
        check(cmd->Close());c->submit(cmd.Get());c->wait();void* mapped=nullptr;D3D12_RANGE range{0,bytes};check(dst->Map(0,&range,&mapped));std::memcpy(data,mapped,bytes);D3D12_RANGE empty{0,0};dst->Unmap(0,&empty);return 0;
    }CATCH_PROBE
}
void rvm_probe_release(void* p) {if(p) static_cast<IUnknown*>(p)->Release();}
int rvm_probe_dispatch_depth(void* p,unsigned depth) {
    TRY_PROBE {auto c=static_cast<Probe*>(p); if(depth==0 || depth>65535) throw std::runtime_error("invalid dispatch depth");
        c->wait();c->dispatch_depth=depth;c->record_shader(false);return 0;} CATCH_PROBE
}
int rvm_probe_group_coverage(void* p,unsigned w,unsigned h) {
    TRY_PROBE {auto c=static_cast<Probe*>(p); if(w==0 || h==0 || w>1024 || h>1024) throw std::runtime_error("invalid group coverage");
        c->wait();c->group_width=w;c->group_height=h;c->record_shader(false);return 0;} CATCH_PROBE
}
}
