// Isolated research runtime; links only with research-ncnn and the pinned SDK.
#include "net.h"
#include "gpu.h"
#include "command.h"
#include "layer.h"
#include <array>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

static thread_local std::string ncnn_error;
static void nc_check(int status, const char* operation) {
    if (status) throw std::runtime_error(std::string(operation)+" failed: "+std::to_string(status));
}
struct NcnnProbe {
    ncnn::Net net;
    const ncnn::VulkanDevice* device=nullptr;
    ncnn::VkAllocator* blob=nullptr;
    ncnn::VkAllocator* staging=nullptr;
    std::array<ncnn::VkMat,4> recurrent;
    std::array<int,4> recurrent_outputs;
    std::array<int,4> recurrent_inputs;
    int source_input=-1;
    int width,height;
    bool diagnostic=false;
    std::string directory;
    int calls=0; double record_ms=0, wait_ms=0;
    ~NcnnProbe() {
        for(auto& r:recurrent) r.release();
        net.clear();
        if(device && blob) device->reclaim_blob_allocator(blob);
        if(device && staging) device->reclaim_staging_allocator(staging);
    }
    void init(const char* root,int w,int h,int adapter,unsigned flags) {
        width=w; height=h;
        diagnostic=(flags&16)!=0; directory=root;
        int count=ncnn::get_gpu_count();
        if(adapter<0 || adapter>=count) throw std::runtime_error("Vulkan adapter unavailable");
        device=ncnn::get_gpu_device(adapter);
        fprintf(stderr,"NCNN Vulkan adapter=%s\n",ncnn::get_gpu_info(adapter).device_name());
        net.opt.use_vulkan_compute=true;
        net.opt.num_threads=1;
        net.opt.use_fp16_packed=(flags&1)==0;
        net.opt.use_fp16_storage=(flags&1)==0;
        net.opt.use_fp16_arithmetic=(flags&2)!=0;
        net.opt.use_winograd_convolution=(flags&4)==0;
        net.opt.use_cooperative_matrix=(flags&8)==0;
        net.set_vulkan_device(adapter);
        blob=device->acquire_blob_allocator(); staging=device->acquire_staging_allocator();
        net.opt.blob_vkallocator=blob;
        net.opt.workspace_vkallocator=blob;
        net.opt.staging_vkallocator=staging;
        const char* basename=(flags&64)?"/optimized":"/rvm";
        nc_check(net.load_param((std::string(root)+basename+".param").c_str()),"load_param");
        nc_check(net.load_model((std::string(root)+basename+".bin").c_str()),"load_model");
        if(net.input_names().size()!=5) throw std::runtime_error("unexpected input count");
        const char* state_layers[]={"Add_277","Add_247","Add_217","Add_187"};
        const char* input_layers[]={"input_r1i","input_r2i","input_r3i","input_r4i"};
        for(const auto* l:net.layers()) if(l->name=="input_src") source_input=l->tops.at(0);
        if(source_input<0) throw std::runtime_error("missing source input layer");
        for(int i=0;i<4;i++) {
            recurrent_outputs[i]=-1;
            recurrent_inputs[i]=-1;
            for(const auto* l:net.layers()) if(l->name==state_layers[i]) recurrent_outputs[i]=l->tops.at(0);
            for(const auto* l:net.layers()) if(l->name==input_layers[i]) recurrent_inputs[i]=l->tops.at(0);
            if(recurrent_outputs[i]<0) throw std::runtime_error("missing recurrent output layer");
            if(recurrent_inputs[i]<0) throw std::runtime_error("missing recurrent input layer");
        }
        for(const auto* l:net.layers()) if(!l->support_vulkan && l->type!="Input")
            throw std::runtime_error("unexpected CPU fallback: "+l->type+" "+l->name);
        reset();
    }
    void reset() {
        ncnn::VkCompute cmd(device);
        const int channels[]={16,20,40,64};
        // Keep all upload source Mats alive until the queue has completed.
        std::array<ncnn::Mat,4> zero;
        for(int i=0;i<4;i++) {
            int d=1<<(i+1);
            zero[i]=ncnn::Mat((width+d-1)/d,(height+d-1)/d,channels[i]); zero[i].fill(0.f);
            recurrent[i].release();
            cmd.record_upload(zero[i],recurrent[i],net.opt);
        }
        nc_check(cmd.submit_and_wait(),"initialize recurrent state");
    }
    void run(const float* pixels,float* alpha) {
        auto begin=std::chrono::steady_clock::now();
        // The tested 480-wide inputs have naturally aligned planar strides.
        if((width*height)%4) throw std::runtime_error("probe requires aligned input planes");
        ncnn::Mat source(width,height,3,const_cast<float*>(pixels));
        ncnn::VkMat gpu_source,gpu_alpha;
        ncnn::Mat result;
        std::array<ncnn::VkMat,4> next;
        ncnn::VkCompute cmd(device);
        cmd.record_upload(source,gpu_source,net.opt);
        auto ex=net.create_extractor();
        if(diagnostic) ex.set_light_mode(false);
        nc_check(ex.input(source_input,gpu_source),"bind source");
        for(int i=0;i<4;i++) nc_check(ex.input(recurrent_inputs[i],recurrent[i]),"bind recurrent state");
        // Hold each recurrent output before later decoder layers consume it.
        // This also preserves states whose external aliases ncnnoptimize removes.
        for(int i=3;i>=0;i--) nc_check(ex.extract(recurrent_outputs[i],next[i],cmd),"extract recurrent state");
        nc_check(ex.extract("pha",gpu_alpha,cmd),"extract alpha");
        cmd.record_download(gpu_alpha,result,net.opt);
        std::vector<std::pair<std::string,ncnn::Mat>> debug;
        std::vector<ncnn::VkMat> debug_gpu;
        if(diagnostic) {
            const char* names[]={"389","393","397","403","420","464","594","601","611","623","640","659","676","695","712","731","745","748","751","752","760","761","765","769","772","775","777","779","pha"};
            for(const auto* name:names) {
                std::string blobname=name;
                auto exists=[&](const std::string& s){for(const auto& b:net.blobs()) if(b.name==s)return true; return false;};
                if(!exists(blobname)) blobname+="_fanout";
                if(!exists(blobname)) continue;
                debug_gpu.emplace_back(); debug.emplace_back(name,ncnn::Mat());
                nc_check(ex.extract(blobname.c_str(),debug_gpu.back(),cmd),"extract diagnostic");
                cmd.record_download(debug_gpu.back(),debug.back().second,net.opt);
            }
        }
        auto recorded=std::chrono::steady_clock::now();
        nc_check(cmd.submit_and_wait(),"complete frame");
        auto completed=std::chrono::steady_clock::now();
        calls++;
        if(calls>30) {
            record_ms+=std::chrono::duration<double,std::milli>(recorded-begin).count();
            wait_ms+=std::chrono::duration<double,std::milli>(completed-recorded).count();
        }
        if(calls==150) fprintf(stderr,"NCNN_NATIVE record_ms=%.4f submit_wait_ms=%.4f\n",record_ms/120,wait_ms/120);
        for(const auto& item:debug) {
            ncnn::Mat m;
            ncnn::convert_packing(item.second,m,1,net.opt);
            fprintf(stderr,"NCNN_DUMP %s dims=%d c=%d h=%d w=%d bits=%d pack=%d\n",item.first.c_str(),m.dims,m.c,m.h,m.w,m.elembits(),m.elempack);
            if(m.elembits()!=32 || m.elempack!=1) throw std::runtime_error("unexpected debug tensor layout");
            std::ofstream f(directory+"/ncnn-"+item.first+".f32",std::ios::binary);
            for(int c=0;c<m.c;c++) f.write(reinterpret_cast<const char*>(m.channel(c).data),size_t(m.w)*m.h*m.d*sizeof(float));
        }
        if(result.w!=width || result.h!=height || result.c!=1 || result.elembits()!=32)
            throw std::runtime_error("unexpected alpha layout");
        std::memcpy(alpha,result.data,size_t(width)*height*sizeof(float));
        recurrent=std::move(next);
    }
};
extern "C" {
const char* rvm_ncnn_error() {return ncnn_error.c_str();}
void* rvm_ncnn_create(const char* root,int w,int h,int adapter,unsigned flags) {
    try {auto p=std::make_unique<NcnnProbe>();p->init(root,w,h,adapter,flags);return p.release();}
    catch(const std::exception& e) {ncnn_error=e.what();return nullptr;}
}
void rvm_ncnn_destroy(void* p) {delete static_cast<NcnnProbe*>(p);}
int rvm_ncnn_reset(void* p) {try{static_cast<NcnnProbe*>(p)->reset();return 0;}catch(const std::exception& e){ncnn_error=e.what();return -1;}}
int rvm_ncnn_run(void* p,const float* rgb,float* alpha) {try{static_cast<NcnnProbe*>(p)->run(rgb,alpha);return 0;}catch(const std::exception& e){ncnn_error=e.what();return -1;}}
}
