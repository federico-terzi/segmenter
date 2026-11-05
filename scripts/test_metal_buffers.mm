// GPU regression: texture uploads, padded/odd frames, queue ordering, and leases
// surviving a resolution change and engine destruction. No third-party libraries.
#include "../src/engine/metal/native.h"
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            // Keep the graph shape identical to isolate texture/lease ownership.
            // The Rust GPU test separately compares one-frame and batched graphs.
            if (!std::getenv("SEGMENTER_METAL_BATCH")) setenv("SEGMENTER_METAL_BATCH", "8", 1);
            const char *model = argc > 1 ? argv[1] : "rvm_mobilenetv3_fp16.rvmmetal";
            char error[4096] = {};
            using Engine = std::unique_ptr<SegmenterRvmMetalContext, decltype(&segmenter_rvm_metal_destroy)>;
            Engine reference(segmenter_rvm_metal_create(model,error,sizeof(error)),segmenter_rvm_metal_destroy);
            Engine native(segmenter_rvm_metal_create(model,error,sizeof(error)),segmenter_rvm_metal_destroy);
            if (!reference || !native) throw std::runtime_error(error);
            struct Mask { void *lease; const uint8_t *pixels; size_t len; };
            std::vector<Mask> masks;
            std::vector<std::vector<uint8_t>> expected;
            size_t pending = 0;
            auto receive = [&] {
                Mask mask = {};
                if (segmenter_rvm_metal_receive_buffer(native.get(), &mask.lease, &mask.pixels, &mask.len,
                                                       error,sizeof(error))) throw std::runtime_error(error);
                masks.push_back(mask);
                --pending;
            };
            for (unsigned i=0;i<65;++i) {
                // Resume after a partial batch as well as after a size change.
                if (i==11 || i==31 || i==47) while(pending) receive();
                const unsigned width=i<31?171:128, height=i<31?129:96;
                NSDictionary *attributes=@{(id)kCVPixelBufferIOSurfacePropertiesKey:@{},
                    (id)kCVPixelBufferMetalCompatibilityKey:@YES};
                CVPixelBufferRef buffer=nullptr;
                if(CVPixelBufferCreate(kCFAllocatorDefault,width,height,kCVPixelFormatType_32BGRA,
                    (CFDictionaryRef)attributes,&buffer)!=kCVReturnSuccess) throw std::runtime_error("pixel buffer creation failed");
                CVPixelBufferLockBaseAddress(buffer,0);
                auto *data=static_cast<uint8_t *>(CVPixelBufferGetBaseAddress(buffer));
                const size_t stride=CVPixelBufferGetBytesPerRow(buffer), len=CVPixelBufferGetDataSize(buffer);
                std::memset(data,0xa5,len);
                for(unsigned y=0;y<height;++y) for(unsigned x=0;x<width;++x) {
                    auto *p=data+y*stride+x*4;
                    p[0]=static_cast<uint8_t>(x+i*17); p[1]=static_cast<uint8_t>(y+i*31);
                    p[2]=static_cast<uint8_t>(x+y); p[3]=255;
                }
                expected.emplace_back(size_t(width)*height*4);
                auto &output=expected.back();
                if(segmenter_rvm_metal_run(reference.get(),data,len,width,height,static_cast<uint32_t>(stride),0.25f,
                    output.data(),output.size(),error,sizeof(error))) throw std::runtime_error(error);
                if(segmenter_rvm_metal_submit_pixel_buffer(native.get(),buffer,0.25f,error,sizeof(error))) throw std::runtime_error(error);
                // Only the GPU owns this pixel buffer after submission returns.
                CVPixelBufferUnlockBaseAddress(buffer,0);
                CVPixelBufferRelease(buffer);
                if(++pending==segmenter_rvm_metal_queue_capacity(native.get())) receive();
            }
            while(pending) receive();
            reference.reset(); native.reset();
            if(masks.size()!=expected.size()) throw std::runtime_error("frame count mismatch");
            for(size_t i=0;i<masks.size();++i) {
                if(masks[i].len!=expected[i].size() || std::memcmp(masks[i].pixels,expected[i].data(),masks[i].len))
                    throw std::runtime_error("native texture/lease output mismatch");
            }
            std::thread release([&] { for(auto &mask:masks) segmenter_rvm_metal_release_buffer(mask.lease); });
            release.join();
            std::puts("PASS: full/partial batches, resumed recurrence, native uploads, and retained masks match synchronous output for 65 frames");
        } catch(const std::exception &error) {
            std::fprintf(stderr,"%s\n",error.what()); return 1;
        }
    }
}
