#include "native.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CoreVideo/CoreVideo.h>
#include <CommonCrypto/CommonDigest.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// MPSGraph can replace the underlying buffer while encoding. Keep every submitted
// buffer so an earlier failure cannot be hidden by a successful final mask buffer.
@interface SegmenterCommandBuffer : MPSCommandBuffer {
    NSMutableArray<id<MTLCommandBuffer>> *_submittedBuffers;
}
- (NSError *)waitForExecution;
@end

@implementation SegmenterCommandBuffer
- (void)commitAndContinue {
    if (_submittedBuffers == nil) _submittedBuffers = [[NSMutableArray alloc] init];
    [_submittedBuffers addObject:self.rootCommandBuffer];
    [super commitAndContinue];
}

- (NSError *)waitForExecution {
    // Also drain partially encoded work when an error unwinds submission.
    if (self.rootCommandBuffer.status < MTLCommandBufferStatusCommitted) [self commit];
    // All work has already been committed: these waits do not split GPU execution.
    [self.rootCommandBuffer waitUntilCompleted];
    NSError *error = nil;
    for (id<MTLCommandBuffer> buffer in _submittedBuffers) {
        [buffer waitUntilCompleted];
        if (error == nil) error = buffer.error;
    }
    return error != nil ? error : self.rootCommandBuffer.error;
}

- (void)dealloc {
    [_submittedBuffers release];
    [super dealloc];
}
@end

namespace {

constexpr char kMagic[8] = {'R', 'V', 'M', 'M', 'E', 'T', 'A', 'L'};
constexpr uint32_t kVersion = 2;
constexpr uint32_t kMissingValue = 0xFFFF'FFFFu;

enum DType : uint32_t {
    kDTypeF32 = 1,
    kDTypeF16 = 2,
    kDTypeI64 = 3,
};

enum Op : uint32_t {
    kOpConv = 1,
    kOpRelu = 2,
    kOpSigmoid = 3,
    kOpTanh = 4,
    kOpHardSigmoid = 5,
    kOpAdd = 6,
    kOpSub = 7,
    kOpMul = 8,
    kOpDiv = 9,
    kOpAveragePool = 10,
    kOpGlobalAveragePool = 11,
    kOpResize = 12,
    kOpConcat = 13,
    kOpSplit = 14,
    kOpSlice = 15,
    kOpShape = 16,
    kOpExpand = 17,
    kOpConstant = 18,
    kOpClip = 19,
    kOpReduceMean = 20,
};

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

void set_error(char *error, size_t error_len, const char *message) {
    if (error == nullptr || error_len == 0) {
        return;
    }
    std::snprintf(error, error_len, "%s", message == nullptr ? "unknown error" : message);
}

uint32_t read_u32(std::ifstream &file, const char *label) {
    uint32_t value = 0;
    file.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!file) {
        fail(std::string("failed to read ") + label);
    }
    return value;
}

uint64_t read_u64(std::ifstream &file, const char *label) {
    uint64_t value = 0;
    file.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!file) {
        fail(std::string("failed to read ") + label);
    }
    return value;
}

int64_t read_i64(std::ifstream &file, const char *label) {
    int64_t value = 0;
    file.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!file) {
        fail(std::string("failed to read ") + label);
    }
    return value;
}

float i64_to_float(int64_t value) {
    int32_t bits = static_cast<int32_t>(value);
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t float_to_half(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffu;

    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa = (mantissa | 0x800000u) >> (1 - exponent);
        return static_cast<uint16_t>(sign | ((mantissa + 0x1000u) >> 13));
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | ((mantissa + 0x1000u) >> 13));
}

float half_to_float(uint16_t value) {
    const uint32_t sign = (static_cast<uint32_t>(value & 0x8000u)) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                exponent -= 1;
            }
            mantissa &= 0x03ffu;
            exponent = exponent + (127 - 15);
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        exponent = exponent + (127 - 15);
        bits = sign | (exponent << 23) | (mantissa << 13);
    }

    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

MPSDataType mps_dtype(uint32_t dtype) {
    switch (dtype) {
    case kDTypeF32:
        return MPSDataTypeFloat32;
    case kDTypeF16:
        return MPSDataTypeFloat16;
    case kDTypeI64:
        return MPSDataTypeInt64;
    default:
        fail("unsupported tensor dtype in RVM Metal model");
    }
}

size_t dtype_size(uint32_t dtype) {
    switch (dtype) {
    case kDTypeF32:
        return sizeof(float);
    case kDTypeF16:
        return sizeof(uint16_t);
    case kDTypeI64:
        return sizeof(int64_t);
    default:
        fail("unsupported tensor dtype size in RVM Metal model");
    }
}

struct TensorDef {
    uint32_t id = 0;
    uint32_t dtype = 0;
    std::vector<uint32_t> shape;
    std::vector<uint8_t> data;
};

struct NodeOutput {
    uint32_t id = kMissingValue;
    std::vector<uint32_t> shape_hint;
};

struct NodeDef {
    uint32_t op = 0;
    std::vector<NodeOutput> outputs;
    std::vector<uint32_t> inputs;
    std::vector<uint32_t> tensor_refs;
    std::vector<int64_t> attrs;
};

struct ModelDef {
    uint32_t src_id = kMissingValue;
    uint32_t downsample_id = kMissingValue;
    uint32_t pha_id = kMissingValue;
    std::array<uint32_t, 4> recurrent_input_ids = {};
    std::array<uint32_t, 4> recurrent_output_ids = {};
    uint32_t model_dtype = kDTypeF16;
    std::unordered_map<uint32_t, TensorDef> tensors;
    std::vector<NodeDef> nodes;
};

std::vector<uint32_t> read_shape(std::ifstream &file, uint32_t rank) {
    std::array<uint32_t, 4> shape4 = {};
    for (uint32_t i = 0; i < 4; ++i) {
        shape4[i] = read_u32(file, "shape dimension");
    }
    if (rank > 4) {
        fail("RVM Metal model contained a tensor rank greater than four");
    }
    std::vector<uint32_t> shape;
    shape.reserve(rank);
    for (uint32_t i = 0; i < rank; ++i) {
        shape.push_back(shape4[i]);
    }
    return shape;
}

ModelDef load_model(const char *path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        fail(std::string("failed to open RVM Metal model: ") + path);
    }

    char magic[8] = {};
    file.read(magic, sizeof(magic));
    if (!file || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
        fail("invalid RVM Metal model magic");
    }
    const uint32_t version = read_u32(file, "version");
    if (version != kVersion) {
        fail("unsupported RVM Metal model version; regenerate it with scripts/export_rvm_metal_model.py");
    }

    ModelDef model;
    model.src_id = read_u32(file, "src id");
    model.downsample_id = read_u32(file, "downsample id");
    model.pha_id = read_u32(file, "pha id");
    model.recurrent_input_ids[0] = read_u32(file, "r1i id");
    model.recurrent_input_ids[1] = read_u32(file, "r2i id");
    model.recurrent_input_ids[2] = read_u32(file, "r3i id");
    model.recurrent_input_ids[3] = read_u32(file, "r4i id");
    model.recurrent_output_ids[0] = read_u32(file, "r1o id");
    model.recurrent_output_ids[1] = read_u32(file, "r2o id");
    model.recurrent_output_ids[2] = read_u32(file, "r3o id");
    model.recurrent_output_ids[3] = read_u32(file, "r4o id");
    model.model_dtype = read_u32(file, "model dtype");
    for (int i = 0; i < 4; ++i) {
        (void)read_u32(file, "reserved header field");
    }

    const uint32_t tensor_count = read_u32(file, "tensor count");
    model.tensors.reserve(tensor_count);
    for (uint32_t i = 0; i < tensor_count; ++i) {
        TensorDef tensor;
        tensor.id = read_u32(file, "tensor id");
        tensor.dtype = read_u32(file, "tensor dtype");
        const uint32_t rank = read_u32(file, "tensor rank");
        tensor.shape = read_shape(file, rank);
        const uint64_t byte_len = read_u64(file, "tensor byte length");
        tensor.data.resize(static_cast<size_t>(byte_len));
        if (byte_len > 0) {
            file.read(reinterpret_cast<char *>(tensor.data.data()), static_cast<std::streamsize>(byte_len));
            if (!file) {
                fail("failed to read tensor bytes");
            }
        }
        model.tensors.emplace(tensor.id, std::move(tensor));
    }

    const uint32_t node_count = read_u32(file, "node count");
    model.nodes.reserve(node_count);
    for (uint32_t i = 0; i < node_count; ++i) {
        NodeDef node;
        node.op = read_u32(file, "node op");
        const uint32_t output_count = read_u32(file, "node output count");
        node.outputs.reserve(output_count);
        for (uint32_t output_index = 0; output_index < output_count; ++output_index) {
            NodeOutput output;
            output.id = read_u32(file, "node output id");
            const uint32_t rank = read_u32(file, "node output rank");
            output.shape_hint = read_shape(file, rank);
            node.outputs.push_back(std::move(output));
        }
        const uint32_t input_count = read_u32(file, "node input count");
        node.inputs.reserve(input_count);
        for (uint32_t input_index = 0; input_index < input_count; ++input_index) {
            node.inputs.push_back(read_u32(file, "node input id"));
        }
        const uint32_t tensor_ref_count = read_u32(file, "node tensor ref count");
        node.tensor_refs.reserve(tensor_ref_count);
        for (uint32_t ref_index = 0; ref_index < tensor_ref_count; ++ref_index) {
            node.tensor_refs.push_back(read_u32(file, "node tensor ref id"));
        }
        const uint32_t attr_count = read_u32(file, "node attr count");
        node.attrs.reserve(attr_count);
        for (uint32_t attr_index = 0; attr_index < attr_count; ++attr_index) {
            node.attrs.push_back(read_i64(file, "node attr"));
        }
        model.nodes.push_back(std::move(node));
    }

    return model;
}

size_t element_count(const std::vector<uint32_t> &shape) {
    size_t count = 1;
    for (uint32_t dim : shape) {
        count *= static_cast<size_t>(dim);
    }
    return count;
}

MPSShape *make_shape(const std::vector<uint32_t> &shape) {
    NSMutableArray<NSNumber *> *array = [NSMutableArray arrayWithCapacity:shape.size()];
    for (uint32_t dim : shape) {
        [array addObject:@(dim)];
    }
    return array;
}

NSArray<NSNumber *> *make_numbers(const std::vector<int64_t> &values) {
    NSMutableArray<NSNumber *> *array = [NSMutableArray arrayWithCapacity:values.size()];
    for (int64_t value : values) {
        [array addObject:@(value)];
    }
    return array;
}

NSArray<NSNumber *> *make_numbers_u32(const std::vector<uint32_t> &values) {
    NSMutableArray<NSNumber *> *array = [NSMutableArray arrayWithCapacity:values.size()];
    for (uint32_t value : values) {
        [array addObject:@(value)];
    }
    return array;
}

int64_t normalize_axis(int64_t axis, size_t rank) {
    if (axis < 0) {
        axis += static_cast<int64_t>(rank);
    }
    if (axis < 0 || axis >= static_cast<int64_t>(rank)) {
        fail("RVM Metal graph used an axis outside tensor rank");
    }
    return axis;
}

std::vector<uint32_t> broadcast_shape(const std::vector<uint32_t> &lhs, const std::vector<uint32_t> &rhs) {
    const size_t rank = std::max(lhs.size(), rhs.size());
    std::vector<uint32_t> result(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        const uint32_t a = i < rank - lhs.size() ? 1 : lhs[i - (rank - lhs.size())];
        const uint32_t b = i < rank - rhs.size() ? 1 : rhs[i - (rank - rhs.size())];
        if (a != b && a != 1 && b != 1) {
            fail("RVM Metal graph encountered incompatible broadcast shapes");
        }
        result[i] = std::max(a, b);
    }
    return result;
}

std::vector<int64_t> tensor_i64_values(const TensorDef &tensor) {
    if (tensor.dtype != kDTypeI64) {
        fail("expected int64 tensor");
    }
    if (tensor.data.size() % sizeof(int64_t) != 0) {
        fail("invalid int64 tensor byte length");
    }
    std::vector<int64_t> values(tensor.data.size() / sizeof(int64_t));
    std::memcpy(values.data(), tensor.data.data(), tensor.data.size());
    return values;
}

std::vector<float> tensor_float_values(const TensorDef &tensor) {
    if (tensor.dtype == kDTypeF32) {
        if (tensor.data.size() % sizeof(float) != 0) {
            fail("invalid float32 tensor byte length");
        }
        std::vector<float> values(tensor.data.size() / sizeof(float));
        std::memcpy(values.data(), tensor.data.data(), tensor.data.size());
        return values;
    }
    if (tensor.dtype == kDTypeF16) {
        if (tensor.data.size() % sizeof(uint16_t) != 0) {
            fail("invalid float16 tensor byte length");
        }
        std::vector<float> values(tensor.data.size() / sizeof(uint16_t));
        const uint16_t *source = reinterpret_cast<const uint16_t *>(tensor.data.data());
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = half_to_float(source[i]);
        }
        return values;
    }
    fail("expected floating-point tensor");
}

struct Value {
    MPSGraphTensor *tensor = nil;
    MPSDataType dtype = MPSDataTypeFloat16;
    std::vector<uint32_t> shape;
    std::vector<int64_t> ints;
    std::vector<float> floats;

    bool has_tensor() const { return tensor != nil; }
    bool has_ints() const { return !ints.empty() || (!shape.empty() && dtype == MPSDataTypeInt64); }
    bool has_floats() const { return !floats.empty() || dtype == MPSDataTypeFloat32 || dtype == MPSDataTypeFloat16; }
};

constexpr size_t kMaxBatchSize = 16;

// Leases may outlive the engine (the encoder owns completed frames). The pool
// therefore owns its device and is shared independently of GraphRuntime.
struct MaskBufferPool {
    id<MTLDevice> device;
    size_t length;
    std::mutex mutex;
    std::vector<id<MTLBuffer>> free;
    MaskBufferPool(id<MTLDevice> d, size_t n): device([d retain]), length(n) { free.reserve(8); }
    ~MaskBufferPool() { for (auto b:free) [b release]; [device release]; }
    id<MTLBuffer> take() {
        std::lock_guard<std::mutex> lock(mutex);
        if (!free.empty()) { auto b=free.back(); free.pop_back(); return b; }
        auto b=[device newBufferWithLength:length options:MTLResourceStorageModeShared];
        if (!b) fail("failed to allocate Metal mask buffer");
        return b;
    }
    void put(id<MTLBuffer> b) {
        std::lock_guard<std::mutex> lock(mutex);
        if (free.size()<8) free.push_back(b); else [b release];
    }
};
struct MaskBufferLease {
    std::shared_ptr<MaskBufferPool> pool;
    id<MTLBuffer> buffer = nil;
    ~MaskBufferLease() { if (buffer) pool->put(buffer); }
};
struct FrameResources {
    id<MTLBuffer> bgra_buffer = nil;
    id<MTLBuffer> mask_buffer = nil;
    id<MTLBuffer> invalid_buffer = nil;
    SegmenterCommandBuffer *command = nil;
    SegmenterCommandBuffer *completion = nil;
    CVPixelBufferRef source_pixel_buffer = nullptr;
    CVMetalTextureRef source_texture = nullptr;
    void release_source() {
        if (source_texture) CFRelease(source_texture);
        if (source_pixel_buffer) CVPixelBufferRelease(source_pixel_buffer);
        source_texture = nullptr; source_pixel_buffer = nullptr;
    }
    ~FrameResources() {
        release_source();
        [command release];
        [completion release];
        [bgra_buffer release];
        [mask_buffer release];
        [invalid_buffer release];
    }
};

struct GraphRuntime {
    uint32_t width = 0;
    uint32_t height = 0;
    float downsample_ratio = 0.0f;
    MPSGraph *graph = nil;
    MPSGraphExecutable *executable = nil;
    NSString *cache_path = nil;
    std::array<NSArray<MPSGraphTensorData *> *, 2> executable_inputs = {};
    std::array<NSArray<MPSGraphTensorData *> *, 2> executable_results = {};
    MPSGraphTensor *src_feed = nil;
    std::array<MPSGraphTensor *, 4> recurrent_feeds = {};
    std::array<std::vector<uint32_t>, 4> recurrent_shapes = {};
    std::vector<MPSGraphTensor *> targets;
    std::vector<MPSDataType> target_dtypes;
    std::vector<std::vector<uint32_t>> target_shapes;
    std::vector<size_t> input_order;
    std::vector<size_t> output_order;
    id<MTLBuffer> input_buffer = nil;
    id<MTLBuffer> alpha_buffer = nil;
    // Only CPU-visible staging needs a ring. Input/alpha and recurrent buffers
    // are consumed strictly in submission order on the same Metal queue.
    std::deque<FrameResources> slots;
    uint32_t batch_size = 1;
    bool fused_refiner = false;
    size_t batch_pending = 0;
    std::vector<id<MTLBuffer>> scratch_states;
    std::shared_ptr<MaskBufferPool> mask_pool;
    size_t submitted = 0;
    size_t received = 0;
    // Alternate banks: recurrent inputs must not alias the outputs of the same frame.
    std::array<std::array<id<MTLBuffer>, 4>, 2> recurrent_buffers = {};
    size_t next = 0;

    ~GraphRuntime() {
        // An encode/read/worker error must not free buffers still used by the GPU.
        for (auto &slot : slots) {
            if (slot.command) [slot.command waitForExecution];
            if (slot.completion) [slot.completion waitForExecution];
        }
        for (auto buffer : scratch_states) [buffer release];
        for (size_t bank = 0; bank < 2; ++bank) {
            [executable_inputs[bank] release];
            [executable_results[bank] release];
            for (id<MTLBuffer> buffer : recurrent_buffers[bank]) {
                [buffer release];
            }
        }
        [input_buffer release];
        [alpha_buffer release];
        [graph release];
        [executable release];
        [cache_path release];
    }
};

} // namespace

struct SegmenterRvmMetalContext {
    ModelDef model;
    std::string model_digest;
    uint32_t batch_size = 8;
    uint32_t configured_batch_size = 0;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> command_queue = nil;
    MPSGraphDevice *graph_device = nil;
    id<MTLComputePipelineState> unpack_pipeline = nil;
    id<MTLComputePipelineState> unpack_texture_pipeline = nil;
    CVMetalTextureCacheRef texture_cache = nullptr;
    id<MTLComputePipelineState> mask_f16_pipeline = nil;
    id<MTLComputePipelineState> mask_f32_pipeline = nil;
    id<MTLComputePipelineState> refine_pipeline = nil;
    id<MTLBuffer> normalized_bytes = nil;
    std::unique_ptr<GraphRuntime> runtime;
    bool profile = std::getenv("SEGMENTER_METAL_PROFILE") != nullptr;
    size_t frames = 0;
    double setup_ms = 0, encode_ms = 0, wait_ms = 0, copy_ms = 0;

    ~SegmenterRvmMetalContext() {
        if (profile) std::fprintf(stderr, "METAL: frames=%zu setup=%.3fms encode=%.3fms wait=%.3fms copy=%.3fms\n",
            frames, setup_ms, encode_ms, wait_ms, copy_ms);
        runtime.reset();
        [unpack_pipeline release];
        [unpack_texture_pipeline release];
        if (texture_cache) CFRelease(texture_cache);
        [mask_f16_pipeline release];
        [mask_f32_pipeline release];
        [refine_pipeline release];
        [normalized_bytes release];
        [graph_device release];
        [command_queue release];
        [device release];
    }
};

namespace {

// The lookup table preserves Rust's f32 byte / 255.0 exactly, including for f32 models.
// Disable fast math below so clamping, non-finite validation, and rounding match the CPU path.
NSString *const kPixelKernels = @R"metal(
#include <metal_stdlib>
using namespace metal;
kernel void unpack_bgra(device const uchar4 *bgra [[buffer(0)]],
                        device float *rgb [[buffer(1)]],
                        constant float *normalized [[buffer(2)]],
                        constant uint &count [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    uchar4 p = bgra[i];
    rgb[i] = normalized[p.z];
    rgb[count + i] = normalized[p.y];
    rgb[count * 2 + i] = normalized[p.x];
}
kernel void unpack_bgra_f16(device const uchar4 *bgra [[buffer(0)]],
                            device half *rgb [[buffer(1)]],
                            constant float *normalized [[buffer(2)]],
                            constant uint &count [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    uchar4 p = bgra[i];
    rgb[i] = half(normalized[p.z]);
    rgb[count + i] = half(normalized[p.y]);
    rgb[count * 2 + i] = half(normalized[p.x]);
}
template<typename T>
void unpack_texture(texture2d<float, access::read> source, device T *rgb,
                    constant float *normalized, uint count, uint i) {
    if (i >= count) return;
    // Recover exact bytes from UNORM texels before the original f32->f16 LUT.
    uint4 p = uint4(round(source.read(uint2(i % source.get_width(), i / source.get_width())) * 255.0f));
    rgb[i] = T(normalized[p.x]);
    rgb[count+i] = T(normalized[p.y]);
    rgb[count*2+i] = T(normalized[p.z]);
}
kernel void unpack_texture_f16(texture2d<float, access::read> source [[texture(0)]],
    device half *rgb [[buffer(1)]], constant float *normalized [[buffer(2)]],
    constant uint &count [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    unpack_texture(source, rgb, normalized, count, i);
}
kernel void unpack_texture_f32(texture2d<float, access::read> source [[texture(0)]],
    device float *rgb [[buffer(1)]], constant float *normalized [[buffer(2)]],
    constant uint &count [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    unpack_texture(source, rgb, normalized, count, i);
}
template<typename T>
void write_mask(device const T *alpha, device uchar4 *mask, device atomic_uint *invalid, uint i) {
    float value = float(alpha[i]);
    if (!isfinite(value)) {
        atomic_store_explicit(invalid, 1u, memory_order_relaxed);
        value = 0.0f;
    }
    uchar a = uchar(round(clamp(value, 0.0f, 1.0f) * 255.0f));
    mask[i] = uchar4(a, a, a, 255);
}
kernel void mask_f16(device const half *alpha [[buffer(0)]], device uchar4 *mask [[buffer(1)]],
                     device atomic_uint *invalid [[buffer(2)]], constant uint &count [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < count) write_mask(alpha, mask, invalid, i);
}
kernel void mask_f32(device const float *alpha [[buffer(0)]], device uchar4 *mask [[buffer(1)]],
                     device atomic_uint *invalid [[buffer(2)]], constant uint &count [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < count) write_mask(alpha, mask, invalid, i);
}
template<typename T>
T interpolate(device const T *p, uint4 positions, float2 fraction) {
    float top = float(p[positions.x]) * (1.0f-fraction.x) + float(p[positions.y]) * fraction.x;
    float bottom = float(p[positions.z]) * (1.0f-fraction.x) + float(p[positions.w]) * fraction.x;
    return T(top * (1.0f-fraction.y) + bottom * fraction.y);
}
template<typename T>
void refine_mask(device const T *coefficients, device uchar4 *mask, device atomic_uint *invalid,
                 uint4 dims, device const T *rgb, uint i) {
    uint pixels = dims.x * dims.y;
    if (i >= pixels) return;
    float2 location = (float2(i % dims.x, i / dims.x) + 0.5f) * float2(dims.zw) / float2(dims.xy) - 0.5f;
    int2 low = int2(floor(location));
    float2 fraction = location - float2(low);
    int2 high = clamp(low + 1, int2(0), int2(dims.zw)-1);
    low = clamp(low, int2(0), int2(dims.zw)-1);
    uint4 positions = uint4(low.y*dims.z+low.x, low.y*dims.z+high.x,
                           high.y*dims.z+low.x, high.y*dims.z+high.x);
    T a = interpolate(coefficients, positions, fraction);
    T b = interpolate(coefficients + dims.z*dims.w, positions, fraction);
    T mean = T((float(rgb[i]) + float(rgb[pixels+i]) + float(rgb[2*pixels+i])) / 3.0f);
    T product = T(a * mean);
    float value = float(T(product + b));
    if (!isfinite(value)) {
        atomic_store_explicit(invalid, 1u, memory_order_relaxed);
        value = 0.0f;
    }
    uchar alpha = uchar(round(clamp(value, 0.0f, 1.0f)*255.0f));
    mask[i] = uchar4(alpha,alpha,alpha,255);
}
kernel void refine_f16(device const half *coefficients [[buffer(0)]], device uchar4 *mask [[buffer(1)]],
    device atomic_uint *invalid [[buffer(2)]], constant uint4 &dims [[buffer(3)]],
    device const half *rgb [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    refine_mask(coefficients,mask,invalid,dims,rgb,i);
}
kernel void refine_f32(device const float *coefficients [[buffer(0)]], device uchar4 *mask [[buffer(1)]],
    device atomic_uint *invalid [[buffer(2)]], constant uint4 &dims [[buffer(3)]],
    device const float *rgb [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    refine_mask(coefficients,mask,invalid,dims,rgb,i);
}
)metal";

void ensure_pixel_pipelines(SegmenterRvmMetalContext *context) {
    if (context->unpack_pipeline != nil) return;
    MTLCompileOptions *options = [[[MTLCompileOptions alloc] init] autorelease];
    options.fastMathEnabled = NO;
    NSError *error = nil;
    id<MTLLibrary> library = [[context->device newLibraryWithSource:kPixelKernels options:options error:&error] autorelease];
    if (library == nil) fail(std::string("failed to compile Metal pixel kernels: ") + [[error localizedDescription] UTF8String]);
    auto pipeline = [&](NSString *name) {
        id<MTLFunction> function = [[library newFunctionWithName:name] autorelease];
        id<MTLComputePipelineState> result = [context->device newComputePipelineStateWithFunction:function error:&error];
        if (result == nil) fail("failed to create Metal pixel pipeline");
        return [result autorelease];
    };
    id<MTLComputePipelineState> mask_f16 = pipeline(@"mask_f16");
    id<MTLComputePipelineState> mask_f32 = pipeline(@"mask_f32");
    id<MTLComputePipelineState> unpack = pipeline(context->model.model_dtype == kDTypeF16 ? @"unpack_bgra_f16" : @"unpack_bgra");
    id<MTLComputePipelineState> unpack_texture = pipeline(context->model.model_dtype == kDTypeF16 ? @"unpack_texture_f16" : @"unpack_texture_f32");
    id<MTLComputePipelineState> refine = pipeline(context->model.model_dtype == kDTypeF16 ? @"refine_f16" : @"refine_f32");
    std::array<float, 256> normalized;
    for (size_t i = 0; i < normalized.size(); ++i) normalized[i] = static_cast<float>(i) / 255.0f;
    id<MTLBuffer> table = [[context->device newBufferWithBytes:normalized.data()
        length:sizeof(normalized) options:MTLResourceStorageModeShared] autorelease];
    if (table == nil) fail("failed to allocate Metal normalization table");
    // Publish only after every resource exists; failed initialization can be retried safely.
    context->mask_f16_pipeline = [mask_f16 retain];
    context->mask_f32_pipeline = [mask_f32 retain];
    context->normalized_bytes = [table retain];
    context->unpack_pipeline = [unpack retain];
    context->unpack_texture_pipeline = [unpack_texture retain];
    context->refine_pipeline = [refine retain];
}

void encode_pixels(MPSCommandBuffer *command, id<MTLComputePipelineState> pipeline,
                   id<MTLBuffer> input, id<MTLBuffer> output, id<MTLBuffer> extra, uint32_t count,
                   size_t input_offset = 0, size_t output_offset = 0) {
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) fail("failed to create Metal pixel command encoder");
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:input_offset atIndex:0];
    [encoder setBuffer:output offset:output_offset atIndex:1];
    [encoder setBuffer:extra offset:0 atIndex:2];
    [encoder setBytes:&count length:sizeof(count) atIndex:3];
    NSUInteger threads = std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreadgroups:MTLSizeMake((count + threads - 1) / threads, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
}

struct BuildContext {
    SegmenterRvmMetalContext *context = nullptr;
    GraphRuntime *runtime = nullptr;
    std::unordered_map<uint32_t, Value> values;
    std::unordered_map<uint32_t, const NodeDef *> producers;

    Value &require_value(uint32_t id) {
        auto it = values.find(id);
        if (it == values.end()) {
            fail("RVM Metal graph referenced a value before it was defined");
        }
        return it->second;
    }

    const TensorDef &require_tensor(uint32_t id) const {
        auto it = context->model.tensors.find(id);
        if (it == context->model.tensors.end()) {
            fail("RVM Metal graph referenced a missing constant tensor");
        }
        return it->second;
    }

    Value tensor_constant(uint32_t tensor_id) {
        const TensorDef &def = require_tensor(tensor_id);
        Value value;
        value.shape = def.shape;
        value.dtype = mps_dtype(def.dtype);

        if (def.dtype == kDTypeI64) {
            value.ints = tensor_i64_values(def);
            return value;
        }

        if (def.shape.size() <= 1 || def.data.empty()) {
            value.floats = tensor_float_values(def);
            return value;
        }

        NSData *data = [NSData dataWithBytes:def.data.data() length:def.data.size()];
        value.tensor = [runtime->graph constantWithData:data shape:make_shape(def.shape) dataType:mps_dtype(def.dtype)];
        return value;
    }

    Value tensor_constant_for_graph(uint32_t tensor_id) {
        const TensorDef &def = require_tensor(tensor_id);
        Value value;
        value.shape = def.shape;
        value.dtype = mps_dtype(def.dtype);
        if (def.dtype == kDTypeI64) {
            value.ints = tensor_i64_values(def);
            return value;
        }
        if (def.shape.empty()) {
            const std::vector<float> floats = tensor_float_values(def);
            const double scalar = floats.empty() ? 0.0 : floats[0];
            value.tensor = [runtime->graph constantWithScalar:scalar dataType:mps_dtype(def.dtype)];
            return value;
        }
        NSData *data = [NSData dataWithBytes:def.data.data() length:def.data.size()];
        value.tensor = [runtime->graph constantWithData:data shape:make_shape(def.shape) dataType:mps_dtype(def.dtype)];
        return value;
    }

    MPSGraphTensor *ensure_tensor(Value &value) {
        if (value.tensor != nil) {
            return value.tensor;
        }
        if (!value.floats.empty() || value.shape.empty()) {
            std::vector<uint8_t> bytes;
            if (context->model.model_dtype == kDTypeF16) {
                bytes.resize(value.floats.size() * sizeof(uint16_t));
                uint16_t *target = reinterpret_cast<uint16_t *>(bytes.data());
                for (size_t i = 0; i < value.floats.size(); ++i) {
                    target[i] = float_to_half(value.floats[i]);
                }
                value.dtype = MPSDataTypeFloat16;
            } else {
                bytes.resize(value.floats.size() * sizeof(float));
                if (!value.floats.empty()) {
                    std::memcpy(bytes.data(), value.floats.data(), bytes.size());
                }
                value.dtype = MPSDataTypeFloat32;
            }
            if (value.shape.empty()) {
                const double scalar = value.floats.empty() ? 0.0 : value.floats[0];
                value.tensor = [runtime->graph constantWithScalar:scalar dataType:value.dtype];
            } else {
                NSData *data = [NSData dataWithBytes:bytes.data() length:bytes.size()];
                value.tensor = [runtime->graph constantWithData:data shape:make_shape(value.shape) dataType:value.dtype];
            }
            return value.tensor;
        }
        fail("RVM Metal graph expected a tensor but found an int shape value");
    }

    Value binary_tensor(const NodeDef &node, MPSGraphTensor *(^op)(MPSGraphTensor *, MPSGraphTensor *)) {
        if (node.inputs.size() + node.tensor_refs.size() != 2) {
            fail("RVM Metal binary op expected exactly two inputs");
        }
        std::vector<Value> owned;
        std::vector<Value *> args;
        for (uint32_t id : node.inputs) {
            args.push_back(&require_value(id));
        }
        for (uint32_t id : node.tensor_refs) {
            owned.push_back(tensor_constant(id));
            args.push_back(&owned.back());
        }

        if (!args[0]->has_tensor() && !args[1]->has_tensor()) {
            return binary_cpu(*args[0], *args[1], node.op);
        }

        Value value;
        value.shape = broadcast_shape(args[0]->shape, args[1]->shape);
        value.dtype = args[0]->dtype;
        value.tensor = op(ensure_tensor(*args[0]), ensure_tensor(*args[1]));
        return value;
    }

    Value binary_cpu(const Value &lhs, const Value &rhs, uint32_t op) {
        if (lhs.ints.empty() && rhs.ints.empty()) {
            const size_t count = std::max(lhs.floats.size(), rhs.floats.size());
            Value value;
            value.dtype = MPSDataTypeFloat32;
            value.shape = lhs.floats.size() >= rhs.floats.size() ? lhs.shape : rhs.shape;
            value.floats.resize(count);
            for (size_t i = 0; i < count; ++i) {
                const float a = lhs.floats.size() == 1 ? lhs.floats[0] : lhs.floats[i];
                const float b = rhs.floats.size() == 1 ? rhs.floats[0] : rhs.floats[i];
                switch (op) {
                case kOpAdd:
                    value.floats[i] = a + b;
                    break;
                case kOpSub:
                    value.floats[i] = a - b;
                    break;
                case kOpMul:
                    value.floats[i] = a * b;
                    break;
                case kOpDiv:
                    value.floats[i] = a / b;
                    break;
                default:
                    fail("unsupported CPU binary op");
                }
            }
            return value;
        }
        fail("unsupported int CPU binary op in RVM Metal graph");
    }

    int recurrent_index(uint32_t id) const {
        for (size_t i = 0; i < context->model.recurrent_input_ids.size(); ++i) {
            if (context->model.recurrent_input_ids[i] == id) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    Value recurrent_placeholder(uint32_t id, const std::vector<uint32_t> &shape) {
        const int index = recurrent_index(id);
        if (index < 0) {
            fail("RVM Metal graph referenced an undefined recurrent placeholder");
        }
        if (runtime->recurrent_feeds[index] == nil) {
            runtime->recurrent_feeds[index] = [runtime->graph placeholderWithShape:make_shape(shape)
                                                                          dataType:mps_dtype(context->model.model_dtype)
                                                                              name:nil];
            runtime->recurrent_shapes[index] = shape;
        } else if (runtime->recurrent_shapes[index] != shape) {
            fail("RVM Metal graph tried to use a recurrent state with two shapes");
        }
        Value value;
        value.tensor = runtime->recurrent_feeds[index];
        value.dtype = mps_dtype(context->model.model_dtype);
        value.shape = shape;
        values[id] = value;
        return value;
    }
};

std::vector<int64_t> value_ints(const Value &value) {
    if (!value.ints.empty()) {
        return value.ints;
    }
    fail("RVM Metal graph expected an int vector");
}

std::vector<float> value_floats(const Value &value) {
    if (!value.floats.empty()) {
        return value.floats;
    }
    fail("RVM Metal graph expected a float vector");
}

std::vector<uint32_t> ints_to_shape(const std::vector<int64_t> &values) {
    std::vector<uint32_t> shape;
    shape.reserve(values.size());
    for (int64_t value : values) {
        if (value <= 0 || value > UINT32_MAX) {
            fail("RVM Metal graph produced an invalid tensor shape");
        }
        shape.push_back(static_cast<uint32_t>(value));
    }
    return shape;
}

uint32_t pool_output_dim(uint32_t input, uint32_t kernel, uint32_t stride, uint32_t pad_before, uint32_t pad_after, bool ceil_mode) {
    const int64_t usable = static_cast<int64_t>(input) + pad_before + pad_after - kernel;
    if (ceil_mode) {
        return static_cast<uint32_t>(std::floor((usable + stride - 1) / static_cast<double>(stride)) + 1);
    }
    return static_cast<uint32_t>(std::floor(usable / static_cast<double>(stride)) + 1);
}

void set_output(BuildContext &build, const NodeDef &node, Value value) {
    if (node.outputs.size() != 1) {
        fail("RVM Metal op expected one output");
    }
    build.values[node.outputs[0].id] = std::move(value);
}

void build_conv(BuildContext &build, const NodeDef &node, Value *partition_source = nullptr,
                uint32_t channel_offset = 0, bool include_bias = true) {
    if (node.inputs.size() != 1 || node.tensor_refs.empty() || node.attrs.size() < 7) {
        fail("invalid Conv node in RVM Metal model");
    }
    Value &source = partition_source ? *partition_source : build.require_value(node.inputs[0]);
    const TensorDef &weights_def = build.require_tensor(node.tensor_refs[0]);
    if (source.shape.size() != 4 || weights_def.shape.size() != 4) {
        fail("RVM Metal Conv expected rank-4 tensors");
    }
    if (partition_source && (channel_offset > weights_def.shape[1] ||
        source.shape[1] > weights_def.shape[1] - channel_offset ||
        weights_def.data.size() != element_count(weights_def.shape) * dtype_size(weights_def.dtype))) {
        fail("invalid recurrent convolution weight partition");
    }
    Value weights = build.tensor_constant_for_graph(node.tensor_refs[0]);
    if (partition_source) {
        // Materialize the channel subset. A strided constant slice produced
        // incorrect convolution weights with MPSGraph's optimizer on macOS 26.4.
        auto shape = weights_def.shape;
        shape[1] = source.shape[1];
        const size_t row = shape[2] * shape[3] * dtype_size(weights_def.dtype);
        std::vector<uint8_t> data(shape[0] * shape[1] * row);
        for (size_t out = 0; out < shape[0]; ++out) {
            std::memcpy(data.data() + out * shape[1] * row,
                weights_def.data.data() + (out * weights_def.shape[1] + channel_offset) * row, shape[1] * row);
        }
        weights.tensor = [build.runtime->graph constantWithData:[NSData dataWithBytes:data.data() length:data.size()]
            shape:make_shape(shape) dataType:mps_dtype(weights_def.dtype)];
    }

    const uint32_t stride_x = static_cast<uint32_t>(node.attrs[0]);
    const uint32_t stride_y = static_cast<uint32_t>(node.attrs[1]);
    const uint32_t pad_left = static_cast<uint32_t>(node.attrs[2]);
    const uint32_t pad_top = static_cast<uint32_t>(node.attrs[3]);
    const uint32_t dilation_x = static_cast<uint32_t>(node.attrs[4]);
    const uint32_t dilation_y = static_cast<uint32_t>(node.attrs[5]);
    const uint32_t groups = static_cast<uint32_t>(node.attrs[6]);
    const uint32_t kernel_h = weights_def.shape[2];
    const uint32_t kernel_w = weights_def.shape[3];
    const uint32_t out_h = static_cast<uint32_t>(
        (static_cast<int64_t>(source.shape[2]) + pad_top + pad_top - dilation_y * (kernel_h - 1) - 1) / stride_y + 1);
    const uint32_t out_w = static_cast<uint32_t>(
        (static_cast<int64_t>(source.shape[3]) + pad_left + pad_left - dilation_x * (kernel_w - 1) - 1) / stride_x + 1);

    MPSGraphConvolution2DOpDescriptor *desc =
        [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:stride_x
                                                         strideInY:stride_y
                                                   dilationRateInX:dilation_x
                                                   dilationRateInY:dilation_y
                                                            groups:groups
                                                       paddingLeft:pad_left
                                                      paddingRight:pad_left
                                                        paddingTop:pad_top
                                                     paddingBottom:pad_top
                                                      paddingStyle:MPSGraphPaddingStyleExplicit
                                                        dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                                                     weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

    Value output;
    output.shape = {source.shape[0], weights_def.shape[0], out_h, out_w};
    output.dtype = source.dtype;
    MPSGraphTensor *conv_source = build.ensure_tensor(source);
    output.tensor = [build.runtime->graph convolution2DWithSourceTensor:conv_source
                                                          weightsTensor:weights.tensor
                                                             descriptor:desc
                                                                   name:nil];
    if (include_bias && node.tensor_refs.size() > 1) {
        Value bias = build.tensor_constant_for_graph(node.tensor_refs[1]);
        output.tensor = [build.runtime->graph additionWithPrimaryTensor:output.tensor
                                                        secondaryTensor:bias.tensor
                                                                   name:nil];
    }
    set_output(build, node, std::move(output));
}

void build_activation(BuildContext &build, const NodeDef &node) {
    Value &input = build.require_value(node.inputs[0]);
    Value output;
    output.shape = input.shape;
    output.dtype = input.dtype;
    switch (node.op) {
    case kOpRelu:
        output.tensor = [build.runtime->graph reLUWithTensor:build.ensure_tensor(input) name:nil];
        break;
    case kOpSigmoid:
        output.tensor = [build.runtime->graph sigmoidWithTensor:build.ensure_tensor(input) name:nil];
        break;
    case kOpTanh:
        output.tensor = [build.runtime->graph tanhWithTensor:build.ensure_tensor(input) name:nil];
        break;
    default:
        fail("unknown activation op");
    }
    set_output(build, node, std::move(output));
}

void build_hard_sigmoid(BuildContext &build, const NodeDef &node) {
    if (node.attrs.size() < 2) {
        fail("HardSigmoid node was missing attributes");
    }
    Value &input = build.require_value(node.inputs[0]);
    const MPSDataType dtype = input.dtype;
    MPSGraphTensor *alpha = [build.runtime->graph constantWithScalar:i64_to_float(node.attrs[0]) dataType:dtype];
    MPSGraphTensor *beta = [build.runtime->graph constantWithScalar:i64_to_float(node.attrs[1]) dataType:dtype];
    MPSGraphTensor *zero = [build.runtime->graph constantWithScalar:0.0 dataType:dtype];
    MPSGraphTensor *one = [build.runtime->graph constantWithScalar:1.0 dataType:dtype];
    MPSGraphTensor *scaled = [build.runtime->graph multiplicationWithPrimaryTensor:build.ensure_tensor(input)
                                                                  secondaryTensor:alpha
                                                                             name:nil];
    MPSGraphTensor *biased = [build.runtime->graph additionWithPrimaryTensor:scaled secondaryTensor:beta name:nil];
    Value output;
    output.shape = input.shape;
    output.dtype = dtype;
    output.tensor = [build.runtime->graph clampWithTensor:biased minValueTensor:zero maxValueTensor:one name:nil];
    set_output(build, node, std::move(output));
}

void build_pool(BuildContext &build, const NodeDef &node) {
    Value &input = build.require_value(node.inputs[0]);
    if (input.shape.size() != 4) {
        fail("pooling expected a rank-4 tensor");
    }

    uint32_t kernel_w = input.shape[3];
    uint32_t kernel_h = input.shape[2];
    uint32_t stride_x = input.shape[3];
    uint32_t stride_y = input.shape[2];
    uint32_t pad_left = 0;
    uint32_t pad_right = 0;
    uint32_t pad_top = 0;
    uint32_t pad_bottom = 0;
    bool ceil_mode = false;

    if (node.op == kOpAveragePool) {
        if (node.attrs.size() < 9) {
            fail("AveragePool node was missing attributes");
        }
        kernel_w = static_cast<uint32_t>(node.attrs[0]);
        kernel_h = static_cast<uint32_t>(node.attrs[1]);
        stride_x = static_cast<uint32_t>(node.attrs[2]);
        stride_y = static_cast<uint32_t>(node.attrs[3]);
        pad_left = static_cast<uint32_t>(node.attrs[4]);
        pad_right = static_cast<uint32_t>(node.attrs[5]);
        pad_top = static_cast<uint32_t>(node.attrs[6]);
        pad_bottom = static_cast<uint32_t>(node.attrs[7]);
        ceil_mode = node.attrs[8] != 0;
    }

    MPSGraphPooling2DOpDescriptor *desc =
        [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:kernel_w
                                                    kernelHeight:kernel_h
                                                       strideInX:stride_x
                                                       strideInY:stride_y
                                                 dilationRateInX:1
                                                 dilationRateInY:1
                                                     paddingLeft:pad_left
                                                    paddingRight:pad_right
                                                      paddingTop:pad_top
                                                   paddingBottom:pad_bottom
                                                    paddingStyle:MPSGraphPaddingStyleExplicit
                                                      dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
    desc.ceilMode = ceil_mode ? YES : NO;

    Value output;
    output.dtype = input.dtype;
    output.shape = {
        input.shape[0],
        input.shape[1],
        node.op == kOpGlobalAveragePool ? 1 : pool_output_dim(input.shape[2], kernel_h, stride_y, pad_top, pad_bottom, ceil_mode),
        node.op == kOpGlobalAveragePool ? 1 : pool_output_dim(input.shape[3], kernel_w, stride_x, pad_left, pad_right, ceil_mode),
    };
    output.tensor = [build.runtime->graph avgPooling2DWithSourceTensor:build.ensure_tensor(input)
                                                            descriptor:desc
                                                                  name:nil];
    set_output(build, node, std::move(output));
}

void build_resize(BuildContext &build, const NodeDef &node) {
    Value &input = build.require_value(node.inputs[0]);
    if (input.shape.size() != 4) {
        fail("Resize expected a rank-4 NCHW tensor");
    }

    std::vector<uint32_t> output_shape;
    if (node.inputs.size() >= 4) {
        output_shape = ints_to_shape(value_ints(build.require_value(node.inputs[3])));
    } else {
        std::vector<float> scales;
        if (node.inputs.size() >= 3) {
            scales = value_floats(build.require_value(node.inputs[2]));
        } else if (!node.tensor_refs.empty()) {
            scales = tensor_float_values(build.require_tensor(node.tensor_refs[0]));
        }
        if (scales.size() != 4) {
            fail("Resize node did not provide static scales or sizes");
        }
        output_shape.resize(4);
        for (size_t i = 0; i < 4; ++i) {
            output_shape[i] = std::max<uint32_t>(1, static_cast<uint32_t>(std::floor(input.shape[i] * scales[i])));
        }
    }
    if (output_shape.size() != 4) {
        fail("Resize output shape was not rank 4");
    }

    const bool linear = node.attrs.empty() || node.attrs[0] != 0;
    MPSShape *size = make_shape({output_shape[2], output_shape[3]});
    Value output;
    output.shape = output_shape;
    output.dtype = input.dtype;
    output.tensor = [build.runtime->graph resizeTensor:build.ensure_tensor(input)
                                                 size:size
                                                 mode:(linear ? MPSGraphResizeBilinear : MPSGraphResizeNearest)
                                         centerResult:YES
                                         alignCorners:NO
                                               layout:MPSGraphTensorNamedDataLayoutNCHW
                                                 name:nil];
    set_output(build, node, std::move(output));
}

void build_concat(BuildContext &build, const NodeDef &node) {
    if (node.attrs.empty()) {
        fail("Concat node was missing its axis");
    }
    std::vector<Value *> inputs;
    inputs.reserve(node.inputs.size());
    bool all_ints = true;
    bool all_floats = true;
    bool any_tensor = false;
    for (uint32_t id : node.inputs) {
        Value &value = build.require_value(id);
        inputs.push_back(&value);
        all_ints = all_ints && !value.ints.empty();
        all_floats = all_floats && !value.floats.empty();
        any_tensor = any_tensor || value.has_tensor();
    }

    if (!any_tensor && all_ints) {
        Value output;
        output.dtype = MPSDataTypeInt64;
        for (const Value *input : inputs) {
            output.ints.insert(output.ints.end(), input->ints.begin(), input->ints.end());
        }
        output.shape = {static_cast<uint32_t>(output.ints.size())};
        set_output(build, node, std::move(output));
        return;
    }

    if (!any_tensor && all_floats) {
        Value output;
        output.dtype = MPSDataTypeFloat32;
        for (const Value *input : inputs) {
            output.floats.insert(output.floats.end(), input->floats.begin(), input->floats.end());
        }
        output.shape = {static_cast<uint32_t>(output.floats.size())};
        set_output(build, node, std::move(output));
        return;
    }

    const int64_t axis = normalize_axis(node.attrs[0], inputs[0]->shape.size());
    NSMutableArray<MPSGraphTensor *> *array = [NSMutableArray arrayWithCapacity:inputs.size()];
    std::vector<uint32_t> shape = inputs[0]->shape;
    shape[axis] = 0;
    for (Value *input : inputs) {
        [array addObject:build.ensure_tensor(*input)];
        shape[axis] += input->shape[axis];
    }

    Value output;
    output.dtype = inputs[0]->dtype;
    output.shape = shape;
    output.tensor = [build.runtime->graph concatTensors:array dimension:axis name:nil];
    set_output(build, node, std::move(output));
}

// Move channel selection ahead of full-resolution pointwise operations and
// resizes. Matting only consumes alpha; upsampling the three foreground channels
// would otherwise do four times the work and allocate four times the memory.
MPSGraphTensor *select_channels(BuildContext &build, uint32_t id, uint32_t start, uint32_t count) {
    Value &value = build.require_value(id);
    MPSGraph *graph = build.runtime->graph;
    if (value.shape.size() != 4 || start > value.shape[1] || count > value.shape[1] - start) {
        fail("invalid alpha channel selection");
    }
    if (start == 0 && count == value.shape[1]) return build.ensure_tensor(value);
    auto producer = build.producers.find(id);
    if (producer != build.producers.end() && value.shape[2] == build.runtime->height &&
        value.shape[3] == build.runtime->width) {
        const NodeDef &node = *producer->second;
        const bool channel_broadcast = (node.op == kOpAdd || node.op == kOpMul) && std::all_of(node.inputs.begin(), node.inputs.end(), [&](uint32_t input) {
            const auto &shape = build.require_value(input).shape;
            return shape.size() < 3 || (shape.size() == 3 && shape[0] == 1) ||
                (shape.size() == 4 && (shape[1] == 1 || shape[1] == value.shape[1]));
        });
        if ((node.op == kOpAdd || node.op == kOpMul) && node.inputs.size() == 2 &&
            node.tensor_refs.empty() && channel_broadcast) {
            auto operand = [&](uint32_t input) {
                Value &v = build.require_value(input);
                return v.shape.size() == 4 && v.shape[1] == value.shape[1]
                    ? select_channels(build, input, start, count) : build.ensure_tensor(v);
            };
            MPSGraphTensor *lhs = operand(node.inputs[0]), *rhs = operand(node.inputs[1]);
            return node.op == kOpAdd ? [graph additionWithPrimaryTensor:lhs secondaryTensor:rhs name:nil]
                : [graph multiplicationWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
        }
        if (node.op == kOpResize) {
            return [graph resizeTensor:select_channels(build, node.inputs[0], start, count)
                size:make_shape({value.shape[2], value.shape[3]})
                mode:(node.attrs.empty() || node.attrs[0] != 0 ? MPSGraphResizeBilinear : MPSGraphResizeNearest)
                centerResult:YES alignCorners:NO layout:MPSGraphTensorNamedDataLayoutNCHW name:nil];
        }
        if (node.op == kOpConcat && normalize_axis(node.attrs[0], 4) == 1) {
            NSMutableArray *parts = [NSMutableArray array];
            uint32_t offset = 0;
            for (uint32_t input : node.inputs) {
                uint32_t channels = build.require_value(input).shape[1];
                uint32_t begin = std::max(start, offset), end = std::min(start + count, offset + channels);
                if (begin < end) [parts addObject:select_channels(build, input, begin - offset, end - begin)];
                offset += channels;
            }
            return parts.count == 1 ? parts[0] : [graph concatTensors:parts dimension:1 name:nil];
        }
    }
    return [graph sliceTensor:build.ensure_tensor(value) dimension:1 start:start length:count name:nil];
}

void build_split(BuildContext &build, const NodeDef &node) {
    if (node.outputs.empty() || node.inputs.size() != 1 || node.attrs.empty()) {
        fail("invalid Split node in RVM Metal model");
    }
    Value &input = build.require_value(node.inputs[0]);
    const int64_t axis = normalize_axis(node.attrs[0], input.shape.size());

    std::vector<uint32_t> sizes;
    if (node.attrs.size() > 1) {
        for (size_t i = 1; i < node.attrs.size(); ++i) {
            sizes.push_back(static_cast<uint32_t>(node.attrs[i]));
        }
    } else {
        if (input.shape[axis] % node.outputs.size() != 0) {
            fail("Split node required equal split but dimension was not divisible");
        }
        sizes.assign(node.outputs.size(), input.shape[axis] / static_cast<uint32_t>(node.outputs.size()));
    }

    uint64_t split_total = 0;
    for (uint32_t size : sizes) split_total += size;
    if (sizes.size() != node.outputs.size() || split_total != input.shape[axis]) fail("invalid Split sizes");

    if (axis == 1 && input.shape.size() == 4 && input.shape[2] == build.runtime->height &&
        input.shape[3] == build.runtime->width && !std::getenv("SEGMENTER_METAL_FULL_CHANNELS")) {
        uint32_t start = 0;
        for (size_t i = 0; i < node.outputs.size(); ++i) {
            Value output = input;
            output.shape[1] = sizes[i];
            output.tensor = select_channels(build, node.inputs[0], start, sizes[i]);
            build.values[node.outputs[i].id] = std::move(output);
            start += sizes[i];
        }
        return;
    }

    NSArray<NSNumber *> *split_sizes = make_numbers_u32(sizes);
    NSArray<MPSGraphTensor *> *split = [build.runtime->graph splitTensor:build.ensure_tensor(input)
                                                               splitSizes:split_sizes
                                                                     axis:axis
                                                                     name:nil];
    if ([split count] != node.outputs.size()) {
        fail("MPSGraph returned an unexpected Split output count");
    }

    for (size_t i = 0; i < node.outputs.size(); ++i) {
        Value output;
        output.dtype = input.dtype;
        output.shape = input.shape;
        output.shape[axis] = sizes[i];
        output.tensor = [split objectAtIndex:i];
        build.values[node.outputs[i].id] = std::move(output);
    }
}

void build_slice(BuildContext &build, const NodeDef &node) {
    if (node.inputs.size() < 3) {
        fail("Slice node was missing inputs");
    }
    Value &data = build.require_value(node.inputs[0]);
    const std::vector<int64_t> starts_raw = value_ints(build.require_value(node.inputs[1]));
    const std::vector<int64_t> ends_raw = value_ints(build.require_value(node.inputs[2]));
    const std::vector<int64_t> axes_raw = node.inputs.size() > 3 ? value_ints(build.require_value(node.inputs[3])) : std::vector<int64_t>{};
    const std::vector<int64_t> steps_raw = node.inputs.size() > 4 ? value_ints(build.require_value(node.inputs[4])) : std::vector<int64_t>{};

    if (!data.ints.empty()) {
        std::vector<int64_t> result;
        const size_t rank = data.ints.size();
        for (size_t i = 0; i < starts_raw.size(); ++i) {
            const int64_t axis = axes_raw.empty() ? static_cast<int64_t>(i) : normalize_axis(axes_raw[i], rank);
            if (axis != 0) {
                fail("RVM Metal only supports shape-vector slices along axis 0");
            }
            const int64_t step = steps_raw.empty() ? 1 : steps_raw[i];
            for (int64_t index = starts_raw[i]; index < ends_raw[i]; index += step) {
                result.push_back(data.ints[static_cast<size_t>(index)]);
            }
        }
        Value output;
        output.dtype = MPSDataTypeInt64;
        output.ints = std::move(result);
        output.shape = {static_cast<uint32_t>(output.ints.size())};
        set_output(build, node, std::move(output));
        return;
    }

    const size_t rank = data.shape.size();
    std::vector<int64_t> starts(rank, 0);
    std::vector<int64_t> ends(rank, 0);
    std::vector<int64_t> strides(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        ends[i] = data.shape[i];
    }
    for (size_t i = 0; i < starts_raw.size(); ++i) {
        const int64_t axis = axes_raw.empty() ? static_cast<int64_t>(i) : normalize_axis(axes_raw[i], rank);
        starts[axis] = starts_raw[i];
        ends[axis] = ends_raw[i];
        strides[axis] = steps_raw.empty() ? 1 : steps_raw[i];
    }

    Value output;
    output.dtype = data.dtype;
    output.shape = data.shape;
    for (size_t i = 0; i < starts_raw.size(); ++i) {
        const int64_t axis = axes_raw.empty() ? static_cast<int64_t>(i) : normalize_axis(axes_raw[i], rank);
        output.shape[axis] = static_cast<uint32_t>(std::max<int64_t>(0, (ends[axis] - starts[axis] + strides[axis] - 1) / strides[axis]));
    }
    output.tensor = [build.runtime->graph sliceTensor:build.ensure_tensor(data)
                                               starts:make_numbers(starts)
                                                 ends:make_numbers(ends)
                                              strides:make_numbers(strides)
                                                 name:nil];
    set_output(build, node, std::move(output));
}

void build_shape(BuildContext &build, const NodeDef &node) {
    Value &input = build.require_value(node.inputs[0]);
    Value output;
    output.dtype = MPSDataTypeInt64;
    output.shape = {static_cast<uint32_t>(input.shape.size())};
    output.ints.reserve(input.shape.size());
    for (uint32_t dim : input.shape) {
        output.ints.push_back(dim);
    }
    set_output(build, node, std::move(output));
}

void build_expand(BuildContext &build, const NodeDef &node) {
    if (node.inputs.size() != 2) {
        fail("Expand node expected two inputs");
    }
    const std::vector<uint32_t> target_shape = ints_to_shape(value_ints(build.require_value(node.inputs[1])));
    Value *input = nullptr;
    auto existing = build.values.find(node.inputs[0]);
    if (existing == build.values.end()) {
        Value created = build.recurrent_placeholder(node.inputs[0], target_shape);
        input = &build.values[node.inputs[0]];
        (void)created;
    } else {
        input = &existing->second;
    }

    Value output;
    output.dtype = input->dtype;
    output.shape = target_shape;
    if (input->shape == target_shape) {
        output.tensor = build.ensure_tensor(*input);
    } else {
        output.tensor = [build.runtime->graph broadcastTensor:build.ensure_tensor(*input)
                                                      toShape:make_shape(target_shape)
                                                         name:nil];
    }
    set_output(build, node, std::move(output));
}

void build_clip(BuildContext &build, const NodeDef &node) {
    if (node.inputs.size() != 1 || node.tensor_refs.size() != 2) {
        fail("Clip node expected one tensor input and min/max constants");
    }
    Value &input = build.require_value(node.inputs[0]);
    Value min = build.tensor_constant_for_graph(node.tensor_refs[0]);
    Value max = build.tensor_constant_for_graph(node.tensor_refs[1]);
    Value output;
    output.dtype = input.dtype;
    output.shape = input.shape;
    output.tensor = [build.runtime->graph clampWithTensor:build.ensure_tensor(input)
                                           minValueTensor:min.tensor
                                           maxValueTensor:max.tensor
                                                     name:nil];
    set_output(build, node, std::move(output));
}

void build_reduce_mean(BuildContext &build, const NodeDef &node) {
    if (node.attrs.size() < 2) {
        fail("ReduceMean node was missing attributes");
    }
    Value &input = build.require_value(node.inputs[0]);
    const bool keepdims = node.attrs[0] != 0;
    std::vector<int64_t> axes;
    for (size_t i = 1; i < node.attrs.size(); ++i) {
        axes.push_back(normalize_axis(node.attrs[i], input.shape.size()));
    }
    Value output;
    output.dtype = input.dtype;
    output.shape = input.shape;
    if (keepdims) {
        for (int64_t axis : axes) {
            output.shape[axis] = 1;
        }
    } else {
        fail("RVM Metal ReduceMean currently requires keepdims=1");
    }
    output.tensor = [build.runtime->graph meanOfTensor:build.ensure_tensor(input)
                                                  axes:make_numbers(axes)
                                                  name:nil];
    set_output(build, node, std::move(output));
}

void build_graph_node(BuildContext &build, const NodeDef &node) {
    switch (node.op) {
    case kOpConv:
        build_conv(build, node);
        break;
    case kOpRelu:
    case kOpSigmoid:
    case kOpTanh:
        build_activation(build, node);
        break;
    case kOpHardSigmoid:
        build_hard_sigmoid(build, node);
        break;
    case kOpAdd:
        set_output(build, node, build.binary_tensor(node, ^MPSGraphTensor *(MPSGraphTensor *a, MPSGraphTensor *b) {
            return [build.runtime->graph additionWithPrimaryTensor:a secondaryTensor:b name:nil];
        }));
        break;
    case kOpSub:
        set_output(build, node, build.binary_tensor(node, ^MPSGraphTensor *(MPSGraphTensor *a, MPSGraphTensor *b) {
            return [build.runtime->graph subtractionWithPrimaryTensor:a secondaryTensor:b name:nil];
        }));
        break;
    case kOpMul:
        set_output(build, node, build.binary_tensor(node, ^MPSGraphTensor *(MPSGraphTensor *a, MPSGraphTensor *b) {
            return [build.runtime->graph multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
        }));
        break;
    case kOpDiv:
        set_output(build, node, build.binary_tensor(node, ^MPSGraphTensor *(MPSGraphTensor *a, MPSGraphTensor *b) {
            return [build.runtime->graph divisionWithPrimaryTensor:a secondaryTensor:b name:nil];
        }));
        break;
    case kOpAveragePool:
    case kOpGlobalAveragePool:
        build_pool(build, node);
        break;
    case kOpResize:
        build_resize(build, node);
        break;
    case kOpConcat:
        build_concat(build, node);
        break;
    case kOpSplit:
        build_split(build, node);
        break;
    case kOpSlice:
        build_slice(build, node);
        break;
    case kOpShape:
        build_shape(build, node);
        break;
    case kOpExpand:
        build_expand(build, node);
        break;
    case kOpConstant:
        if (node.tensor_refs.size() != 1) {
            fail("Constant node expected one tensor payload");
        }
        set_output(build, node, build.tensor_constant(node.tensor_refs[0]));
        break;
    case kOpClip:
        build_clip(build, node);
        break;
    case kOpReduceMean:
        build_reduce_mean(build, node);
        break;
    default:
        fail("unsupported op in RVM Metal model");
    }
}

bool fuse_refiner(BuildContext &build, Value &output) {
    if (std::getenv("SEGMENTER_METAL_GRAPH_REFINER")) return false;
    auto producer = [&](uint32_t id, uint32_t op) -> const NodeDef * {
        auto found = build.producers.find(id);
        return found != build.producers.end() && found->second->op == op ? found->second : nullptr;
    };
    const auto *clip = producer(build.context->model.pha_id, kOpClip);
    if (!clip || clip->inputs.size()!=1 || clip->tensor_refs.size()!=2) return false;
    if (tensor_float_values(build.require_tensor(clip->tensor_refs[0])) != std::vector<float>{0} ||
        tensor_float_values(build.require_tensor(clip->tensor_refs[1])) != std::vector<float>{1}) return false;
    const auto *split = producer(clip->inputs[0], kOpSplit);
    if (!split || split->inputs.size()!=1 || split->outputs.size()!=2 ||
        split->outputs[1].id != clip->inputs[0] || split->attrs != std::vector<int64_t>({1,3,1})) return false;
    const auto *add = producer(split->inputs[0], kOpAdd);
    if (!add || add->inputs.size()!=2) return false;
    const auto *mul = producer(add->inputs[0], kOpMul);
    const auto *resize_b = producer(add->inputs[1], kOpResize);
    if (!mul || mul->inputs.size()!=2 || !resize_b) return false;
    const auto *resize_a = producer(mul->inputs[0], kOpResize);
    const auto *guide = producer(mul->inputs[1], kOpConcat);
    if (!resize_a || !guide || guide->inputs.size()!=2 || guide->attrs != std::vector<int64_t>{1} ||
        guide->inputs[0] != build.context->model.src_id) return false;
    const auto *mean = producer(guide->inputs[1], kOpReduceMean);
    if (!mean || mean->inputs != std::vector<uint32_t>{build.context->model.src_id} ||
        mean->attrs != std::vector<int64_t>({1,1})) return false;
    for (const auto *resize : {resize_a, resize_b}) {
        if (resize->inputs.empty() || (!resize->attrs.empty() && resize->attrs[0] == 0)) return false;
    }
    Value &a = build.require_value(resize_a->inputs[0]);
    Value &b = build.require_value(resize_b->inputs[0]);
    if (a.shape != b.shape || a.shape.size()!=4 || a.shape[1]!=4 || a.dtype!=b.dtype) return false;
    output = a;
    output.shape[1] = 2;
    output.tensor = [build.runtime->graph concatTensors:@[
        select_channels(build,resize_a->inputs[0],3,1),
        select_channels(build,resize_b->inputs[0],3,1)] dimension:1 name:nil];
    return true;
}

std::unique_ptr<GraphRuntime> build_runtime(SegmenterRvmMetalContext *context, uint32_t width, uint32_t height, float downsample_ratio) {
    std::unique_ptr<GraphRuntime> runtime(new GraphRuntime());
    runtime->batch_size = context->batch_size;
    runtime->slots.resize(std::max<uint32_t>(3, context->batch_size * 2));
    runtime->targets.resize(5);
    runtime->target_dtypes.resize(runtime->targets.size());
    runtime->target_shapes.resize(runtime->targets.size());
    runtime->width = width;
    runtime->height = height;
    runtime->downsample_ratio = downsample_ratio;
    runtime->graph = [[MPSGraph alloc] init];
    if (runtime->graph == nil) {
        fail("failed to create MPSGraph");
    }

    BuildContext build;
    build.context = context;
    build.runtime = runtime.get();
    for (const NodeDef &node : context->model.nodes) {
        for (const auto &output : node.outputs) build.producers[output.id] = &node;
    }

    MPSGraphTensor *src_feed = [runtime->graph placeholderWithShape:make_shape({context->batch_size, 3, height, width})
                                                           dataType:mps_dtype(context->model.model_dtype)
                                                               name:nil];
    runtime->src_feed = src_feed;
    Value src;
    src.shape = {context->batch_size, 3, height, width};
    src.dtype = mps_dtype(context->model.model_dtype);
    src.tensor = src_feed;
    build.values[context->model.src_id] = src;

    Value downsample;
    downsample.dtype = MPSDataTypeFloat32;
    downsample.shape = {1};
    downsample.floats = {downsample_ratio};
    build.values[context->model.downsample_id] = std::move(downsample);

    const uint32_t batch = context->batch_size;
    const auto &nodes = context->model.nodes;
    for (size_t node_index=0; node_index<nodes.size(); ++node_index) {
        const NodeDef &node = nodes[node_index];
        int recurrent = -1;
        for (uint32_t input : node.inputs) {
            const int index = build.recurrent_index(input);
            if (index >= 0) recurrent = index;
        }
        if (recurrent < 0) { build_graph_node(build,node); continue; }
        if (node.op != kOpExpand) fail("batching requires an explicit recurrent Expand");
        const uint32_t state_input = context->model.recurrent_input_ids[recurrent];
        const uint32_t state_output = context->model.recurrent_output_ids[recurrent];
        size_t last = node_index;
        while (last < nodes.size() && std::none_of(nodes[last].outputs.begin(),nodes[last].outputs.end(),
            [&](const NodeOutput &output){return output.id==state_output;})) ++last;
        if (last == nodes.size()) fail("recurrent output not found while batching");
        // W*[x,h] = Wx*x + Wh*h. Only x is independent across time: compute its
        // contribution as a batch while keeping h (and the reset gate) sequential.
        struct PartialConv { Value value; uint32_t state; uint32_t channels; };
        std::unordered_map<size_t, PartialConv> partial;
        if (batch > 1 && !std::getenv("SEGMENTER_METAL_FULL_GRU")) {
            for (size_t k = node_index; k <= last; ++k) {
                const auto &conv = nodes[k];
                if (conv.op != kOpConv || conv.attrs.size() < 7 || conv.attrs[6] != 1 || conv.inputs.size()!=1) continue;
                auto it = build.producers.find(conv.inputs[0]);
                if (it == build.producers.end() || it->second->op != kOpConcat || it->second->inputs.size() != 2) continue;
                const auto &concat = *it->second;
                if (concat.attrs.size()!=1 || normalize_axis(concat.attrs[0],4)!=1) continue;
                auto source = build.values.find(concat.inputs[0]);
                if (source == build.values.end()) continue;
                Value input = source->second;
                build_conv(build, conv, &input);
                partial.emplace(k, PartialConv{build.require_value(conv.outputs[0].id), concat.inputs[1], input.shape[1]});
            }
        }
        std::unordered_map<uint32_t,std::vector<Value>> sequences;
        Value previous;
        for (uint32_t frame=0;frame<batch;++frame) {
            BuildContext step;
            step.context=context; step.runtime=runtime.get(); step.producers=build.producers; step.values=build.values;
            for (auto &entry:step.values) {
                Value &value=entry.second;
                if (value.tensor && value.shape.size()==4 && value.shape[0]==batch) {
                    value.tensor=[runtime->graph sliceTensor:value.tensor dimension:0 start:frame length:1 name:nil];
                    value.shape[0]=1;
                }
            }
            // This shape vector describes one recurrent state, not the temporal batch.
            Value &shape=step.require_value(node.inputs[1]);
            if (shape.ints.size()!=4 || shape.ints[0]!=batch) fail("unexpected recurrent batch shape");
            shape.ints[0]=1;
            if (frame) step.values[state_input]=previous;
            for (size_t k=node_index;k<=last;++k) {
                auto part = partial.find(k);
                if (part == partial.end()) {
                    build_graph_node(step,nodes[k]);
                } else {
                    Value input = step.require_value(part->second.state);
                    build_conv(step, nodes[k], &input, part->second.channels, false);
                    Value &output = step.require_value(nodes[k].outputs[0].id);
                    auto *x = [runtime->graph sliceTensor:part->second.value.tensor dimension:0 start:frame length:1 name:nil];
                    output.tensor = [runtime->graph additionWithPrimaryTensor:output.tensor secondaryTensor:x name:nil];
                }
                for (const auto &out:nodes[k].outputs) sequences[out.id].push_back(step.require_value(out.id));
            }
            previous=step.require_value(state_output);
        }
        // CNN layers following the recurrence consume every frame together.
        for (auto &entry:sequences) {
            Value value=entry.second[0];
            if (value.tensor && value.shape.size()==4 && value.shape[0]==1) {
                NSMutableArray *parts=[NSMutableArray arrayWithCapacity:batch];
                for (auto &item:entry.second) [parts addObject:item.tensor];
                value.tensor=batch==1 ? parts[0] : [runtime->graph concatTensors:parts dimension:0 name:nil];
                value.shape[0]=batch;
            }
            build.values[entry.first]=std::move(value);
        }
        node_index=last;
    }

    Value pha = build.require_value(context->model.pha_id);
    runtime->fused_refiner = fuse_refiner(build, pha);
    runtime->targets[0] = pha.tensor;
    runtime->target_dtypes[0] = pha.dtype;
    runtime->target_shapes[0] = pha.shape;
    if (!runtime->fused_refiner && pha.shape != std::vector<uint32_t>({batch, 1, height, width})) {
        fail("RVM Metal alpha output shape did not match input frame");
    }
    if (pha.dtype != MPSDataTypeFloat16 && pha.dtype != MPSDataTypeFloat32) {
        fail("RVM Metal alpha output had an unsupported dtype");
    }

    for (size_t i = 0; i < 4; ++i) {
        Value &state = build.require_value(context->model.recurrent_output_ids[i]);
        runtime->targets[i+1] = state.tensor;
        runtime->target_dtypes[i+1] = state.dtype;
        runtime->target_shapes[i+1] = state.shape;
        if (!state.tensor || !runtime->recurrent_feeds[i] || runtime->recurrent_shapes[i].empty())
            fail("RVM Metal graph did not create all batched recurrent states");
    }

    return runtime;
}

id<MTLBuffer> new_zero_buffer(id<MTLDevice> device, const std::vector<uint32_t> &shape, MPSDataType dtype) {
    const uint32_t export_dtype = dtype == MPSDataTypeFloat32 ? kDTypeF32 : kDTypeF16;
    const size_t byte_len = element_count(shape) * dtype_size(export_dtype);
    id<MTLBuffer> buffer = [device newBufferWithLength:byte_len options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        fail("failed to allocate RVM Metal recurrent buffer");
    }
    std::memset([buffer contents], 0, byte_len);
    return buffer;
}

size_t mps_dtype_size(MPSDataType dtype) {
    switch (dtype) {
    case MPSDataTypeFloat32:
        return sizeof(float);
    case MPSDataTypeFloat16:
        return sizeof(uint16_t);
    case MPSDataTypeInt64:
        return sizeof(int64_t);
    default:
        fail("unsupported MPS dtype size");
    }
}

// Packages contain compiled operations and weights, but not frame/recurrent data.
// Key every input to specialization; the OS temporary cache may disappear at any
// time. Failure to read/write it simply falls back to compiling the graph.
NSString *runtime_cache_path(SegmenterRvmMetalContext *context, uint32_t width,
                             uint32_t height, float ratio) {
    if (std::getenv("SEGMENTER_METAL_NO_CACHE")) return nil;
    uint32_t ratio_bits;
    std::memcpy(&ratio_bits, &ratio, sizeof(ratio_bits));
    NSString *key = [NSString stringWithFormat:@"rvm-batch-v6-%s-%@-%llu-%u-%u-%08x-%u-%d-%d-%d-%d",
        context->model_digest.c_str(), NSProcessInfo.processInfo.operatingSystemVersionString,
        context->device.registryID, width, height, ratio_bits, context->batch_size,
        std::getenv("SEGMENTER_METAL_FULL_CHANNELS") != nullptr,
        std::getenv("SEGMENTER_METAL_FULL_GRU") != nullptr,
        std::getenv("SEGMENTER_METAL_PRECISE_MATH") != nullptr,
        std::getenv("SEGMENTER_METAL_GRAPH_REFINER") != nullptr];
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    NSData *bytes = [key dataUsingEncoding:NSUTF8StringEncoding];
    CC_SHA256(bytes.bytes, static_cast<CC_LONG>(bytes.length), digest);
    NSMutableString *hex = [NSMutableString stringWithCapacity:64];
    for (auto byte : digest) [hex appendFormat:@"%02x", byte];
    return [[NSTemporaryDirectory() stringByAppendingPathComponent:@"segmenter-metal-cache"]
        stringByAppendingPathComponent:hex];
}

std::unique_ptr<GraphRuntime> load_runtime_cache(SegmenterRvmMetalContext *context,
        NSString *path, uint32_t width, uint32_t height, float ratio) {
    if (!path) return nullptr;
    if (@available(macOS 14.0, *)) {
        @try {
            NSDictionary *info = [NSDictionary dictionaryWithContentsOfFile:[path stringByAppendingPathComponent:@"bindings.plist"]];
            if (!info) return nullptr;
            const size_t count = 5;
            NSArray *shapes = info[@"shapes"], *dtypes = info[@"dtypes"];
            if (![shapes isKindOfClass:NSArray.class] || shapes.count != count ||
                ![dtypes isKindOfClass:NSArray.class] || dtypes.count != count) return nullptr;
            auto runtime = std::make_unique<GraphRuntime>();
            runtime->width = width; runtime->height = height; runtime->downsample_ratio = ratio;
            runtime->batch_size = context->batch_size;
            runtime->fused_refiner = [info[@"fused_refiner"] boolValue];
            runtime->slots.resize(std::max<uint32_t>(3, context->batch_size * 2));
            runtime->targets.resize(count);
            for (size_t i = 0; i < count; ++i) {
                NSArray *shape = shapes[i];
                if (![shape isKindOfClass:NSArray.class] || shape.count != 4) return nullptr;
                std::vector<uint32_t> dimensions;
                size_t elements = 1;
                for (id n in shape) {
                    if (![n isKindOfClass:NSNumber.class] || [n longLongValue] <= 0 || [n unsignedLongLongValue] > UINT32_MAX) return nullptr;
                    elements *= [n unsignedIntValue];
                    if (elements > (1ull << 29)) return nullptr;
                    dimensions.push_back([n unsignedIntValue]);
                }
                if (![dtypes[i] isKindOfClass:NSNumber.class]) return nullptr;
                auto dtype = static_cast<MPSDataType>([dtypes[i] unsignedIntValue]);
                if (dtype != mps_dtype(context->model.model_dtype)) return nullptr;
                runtime->target_shapes.push_back(dimensions);
                runtime->target_dtypes.push_back(dtype);
                if (i == 0) {
                    if (dimensions[0] != context->batch_size || dimensions[1] != (runtime->fused_refiner ? 2 : 1)) return nullptr;
                    if (runtime->fused_refiner ? (dimensions[2] > height || dimensions[3] > width) :
                        (dimensions[2] != height || dimensions[3] != width)) return nullptr;
                } else if (i <= 4) {
                    if (dimensions[0] != context->batch_size) return nullptr;
                    dimensions[0] = 1;
                    runtime->recurrent_shapes[i-1] = dimensions;
                } else if (dimensions != runtime->recurrent_shapes[(i-1)%4]) return nullptr;
            }
            auto order = [](id array, size_t size, std::vector<size_t> &result) {
                if (![array isKindOfClass:NSArray.class] || [array count] != size) return false;
                std::vector<bool> seen(size, false);
                for (id item in array) {
                    if (![item isKindOfClass:NSNumber.class]) return false;
                    const size_t index = [item unsignedLongLongValue];
                    if (index >= size || seen[index]) return false;
                    seen[index] = true; result.push_back(index);
                }
                return true;
            };
            if (!order(info[@"inputs"], 5, runtime->input_order) ||
                !order(info[@"outputs"], count, runtime->output_order)) return nullptr;
            NSURL *url = [NSURL fileURLWithPath:[path stringByAppendingPathComponent:@"graph.mpsgraphpackage"]];
            if (![NSFileManager.defaultManager fileExistsAtPath:url.path]) return nullptr;
            auto *descriptor = [[[MPSGraphCompilationDescriptor alloc] init] autorelease];
            // Re-specialize the portable graph instead of reusing the serialized
            // level-1 GPU module (which crashes in Conv2DOpHandler on macOS 26.4).
            descriptor.optimizationLevel = MPSGraphOptimizationLevel0;
            if (@available(macOS 26.0, *)) {
                if (!std::getenv("SEGMENTER_METAL_PRECISE_MATH") && context->model.model_dtype == kDTypeF16)
                    descriptor.reducedPrecisionFastMath = MPSGraphReducedPrecisionFastMathAllowFP16Intermediates;
            }
            runtime->executable = [[MPSGraphExecutable alloc] initWithMPSGraphPackageAtURL:url compilationDescriptor:descriptor];
            if (!runtime->executable) return nullptr;
            return runtime;
        } @catch (NSException *) {
            return nullptr;
        }
    }
    return nullptr;
}

void save_runtime_cache(GraphRuntime *runtime, NSString *path) {
    if (!path) return;
    if (@available(macOS 14.0, *)) {
        // Publish the package and its binding metadata together. Concurrent first
        // runs have separate staging directories and can safely race to publish.
        NSString *staging = [path stringByAppendingFormat:@".%@", NSUUID.UUID.UUIDString];
        auto *files = NSFileManager.defaultManager;
        @try {
            if (![files createDirectoryAtPath:staging withIntermediateDirectories:YES attributes:nil error:nil]) return;
            NSURL *url = [NSURL fileURLWithPath:[staging stringByAppendingPathComponent:@"graph.mpsgraphpackage"]];
            auto *descriptor = [[[MPSGraphExecutableSerializationDescriptor alloc] init] autorelease];
            descriptor.minimumDeploymentTarget = @"14.0";
            [runtime->executable serializeToMPSGraphPackageAtURL:url descriptor:descriptor];
            NSMutableArray *shapes = [NSMutableArray array], *dtypes = [NSMutableArray array];
            NSMutableArray *inputs = [NSMutableArray array], *outputs = [NSMutableArray array];
            for (auto &shape : runtime->target_shapes) [shapes addObject:make_shape(shape)];
            for (auto dtype : runtime->target_dtypes) [dtypes addObject:@(dtype)];
            for (auto index : runtime->input_order) [inputs addObject:@(index)];
            for (auto index : runtime->output_order) [outputs addObject:@(index)];
            NSDictionary *info = @{@"shapes":shapes, @"dtypes":dtypes, @"inputs":inputs, @"outputs":outputs,
                @"fused_refiner":@(runtime->fused_refiner)};
            if ([info writeToFile:[staging stringByAppendingPathComponent:@"bindings.plist"] atomically:YES]) {
                [files moveItemAtPath:staging toPath:path error:nil];
            }
        } @catch (NSException *) {
            // Caching is optional, including on a full or read-only filesystem.
        }
        [files removeItemAtPath:staging error:nil];
    }
}

void compile_runtime(SegmenterRvmMetalContext *context, GraphRuntime *runtime) {
    NSArray *inputs = @[runtime->src_feed, runtime->recurrent_feeds[0], runtime->recurrent_feeds[1],
                        runtime->recurrent_feeds[2], runtime->recurrent_feeds[3]];
    NSArray *outputs = [NSArray arrayWithObjects:runtime->targets.data() count:runtime->targets.size()];
    NSMutableDictionary *types = [NSMutableDictionary dictionaryWithCapacity:5];
    for (MPSGraphTensor *tensor in inputs) {
        types[tensor] = [[[MPSGraphShapedType alloc] initWithShape:tensor.shape dataType:tensor.dataType] autorelease];
    }
    auto *descriptor = [[[MPSGraphCompilationDescriptor alloc] init] autorelease];
    if (@available(macOS 26.0, *)) {
        if ((!std::getenv("SEGMENTER_METAL_PRECISE_MATH") && context->model.model_dtype == kDTypeF16))
            descriptor.reducedPrecisionFastMath = MPSGraphReducedPrecisionFastMathAllowFP16Intermediates;
    }
    runtime->executable = [[runtime->graph compileWithDevice:context->graph_device feeds:types
        targetTensors:outputs targetOperations:nil compilationDescriptor:descriptor] retain];
    if (!runtime->executable) fail("failed to compile RVM Metal graph");
    for (MPSGraphTensor *tensor in runtime->executable.feedTensors) {
        const NSUInteger index = [inputs indexOfObjectIdenticalTo:tensor];
        if (index == NSNotFound) fail("unexpected compiled Metal input");
        runtime->input_order.push_back(index);
    }
    for (MPSGraphTensor *tensor in runtime->executable.targetTensors) {
        const NSUInteger index = [outputs indexOfObjectIdenticalTo:tensor];
        if (index == NSNotFound) fail("unexpected compiled Metal output");
        runtime->output_order.push_back(index);
    }
}

void ensure_runtime(SegmenterRvmMetalContext *context, uint32_t width, uint32_t height, float downsample_ratio) {
    if (context->runtime &&
        context->runtime->width == width &&
        context->runtime->height == height &&
        std::fabs(context->runtime->downsample_ratio - downsample_ratio) < 1.0e-6f) {
        return;
    }
    if (context->runtime && context->runtime->submitted != context->runtime->received) {
        fail("drain pending Metal frames before changing input dimensions or ratio");
    }

    if (!context->configured_batch_size) {
        // Bound working memory for 4K inputs and for high downsample ratios.
        const uint64_t pixels = static_cast<uint64_t>(width) * height;
        const uint64_t low_width = std::max<uint64_t>(1, std::floor(width * downsample_ratio));
        const uint64_t low_height = std::max<uint64_t>(1, std::floor(height * downsample_ratio));
        context->batch_size = static_cast<uint32_t>(std::max<uint64_t>(1,
            std::min<uint64_t>({8, 16588800 / pixels, 1036800 / (low_width * low_height)})));
    }
    NSString *cache_path = runtime_cache_path(context, width, height, downsample_ratio);
    auto runtime = load_runtime_cache(context, cache_path, width, height, downsample_ratio);
    if (!runtime) {
        runtime = build_runtime(context, width, height, downsample_ratio);
        compile_runtime(context, runtime.get());
        runtime->cache_path = [cache_path copy];
    }
    runtime->executable.options = MPSGraphOptionsNone;
    runtime->input_buffer = [context->device newBufferWithLength:static_cast<size_t>(width) * height * 3 * runtime->batch_size * dtype_size(context->model.model_dtype)
                                                       options:MTLResourceStorageModeShared];
    runtime->alpha_buffer = [context->device newBufferWithLength:element_count(runtime->target_shapes[0]) * mps_dtype_size(runtime->target_dtypes[0])
                                                       options:MTLResourceStorageModeShared];
    const size_t bgra_len = static_cast<size_t>(width) * height * 4;
    runtime->mask_pool = std::make_shared<MaskBufferPool>(context->device, bgra_len);
    for (auto &slot : runtime->slots) {
        slot.mask_buffer = runtime->mask_pool->take();
        slot.invalid_buffer = [context->device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        if (slot.mask_buffer == nil || slot.invalid_buffer == nil) {
            fail("failed to allocate RVM Metal staging buffers");
        }
    }
    if (runtime->input_buffer == nil || runtime->alpha_buffer == nil) {
        fail("failed to allocate RVM Metal input/output buffers");
    }
    for (size_t bank = 0; bank < 2; ++bank) {
        for (size_t i = 0; i < 4; ++i) {
            auto state_shape = runtime->target_shapes[i + 1];
            state_shape[0] = 1;
            if (runtime->recurrent_shapes[i] != state_shape ||
                runtime->target_dtypes[i + 1] != mps_dtype(context->model.model_dtype)) {
                fail("RVM Metal recurrent input/output shapes or types do not match");
            }
            runtime->recurrent_buffers[bank][i] = new_zero_buffer(
                context->device, runtime->recurrent_shapes[i], mps_dtype(context->model.model_dtype));
        }
    }
    std::memset(runtime->input_buffer.contents,0,runtime->input_buffer.length);
    for (size_t i=1;i<runtime->targets.size();++i) {
        runtime->scratch_states.push_back(new_zero_buffer(context->device,runtime->target_shapes[i],runtime->target_dtypes[i]));
    }
    auto tensor_data = [](id<MTLBuffer> buffer, const std::vector<uint32_t> &shape, MPSDataType dtype) {
        return [[[MPSGraphTensorData alloc] initWithMTLBuffer:buffer shape:make_shape(shape) dataType:dtype] autorelease];
    };
    for (size_t bank=0;bank<2;++bank) {
        NSMutableArray *feeds=[NSMutableArray arrayWithCapacity:5];
        [feeds addObject:tensor_data(runtime->input_buffer,{runtime->batch_size,3,height,width},mps_dtype(context->model.model_dtype))];
        for (size_t i=0;i<4;++i) [feeds addObject:tensor_data(runtime->recurrent_buffers[bank][i],runtime->recurrent_shapes[i],mps_dtype(context->model.model_dtype))];
        NSMutableArray *inputs=[NSMutableArray arrayWithCapacity:5];
        for (size_t index : runtime->input_order) [inputs addObject:feeds[index]];
        runtime->executable_inputs[bank]=[inputs copy];
        NSMutableArray *results=[NSMutableArray arrayWithCapacity:runtime->targets.size()];
        [results addObject:tensor_data(runtime->alpha_buffer,runtime->target_shapes[0],runtime->target_dtypes[0])];
        for (size_t i=0;i<4;++i) {
            const size_t index=1+i;
            id<MTLBuffer> buffer = runtime->scratch_states[i];
            [results addObject:tensor_data(buffer,runtime->target_shapes[index],runtime->target_dtypes[index])];
        }
        NSMutableArray *outputs=[NSMutableArray arrayWithCapacity:runtime->targets.size()];
        for (size_t index : runtime->output_order) [outputs addObject:results[index]];
        runtime->executable_results[bank] = [outputs copy];
    }
    context->runtime = std::move(runtime);
}

void flush_batch(SegmenterRvmMetalContext *context) {
    GraphRuntime *runtime=context->runtime.get();
    const size_t count=runtime->batch_pending;
    if (!count) return;
    const size_t first=runtime->submitted-count;
    SegmenterCommandBuffer *command = runtime->slots[first % runtime->slots.size()].command;
    for (size_t i=0;i<count;++i) runtime->slots[(first+i)%runtime->slots.size()].completion=[command retain];
    [runtime->executable encodeToCommandBuffer:command inputsArray:runtime->executable_inputs[runtime->next]
        resultsArray:runtime->executable_results[runtime->next] executionDescriptor:nil];
    id<MTLBlitCommandEncoder> copy = [command blitCommandEncoder];
    if (!copy) fail("failed to create recurrent state copy encoder");
    for (size_t i=0;i<4;++i) {
        // The padded tail may compute additional states, but only the last real
        // frame is allowed to feed a later batch, including after drain/resume.
        size_t bytes = element_count(runtime->recurrent_shapes[i]) * dtype_size(context->model.model_dtype);
        [copy copyFromBuffer:runtime->scratch_states[i] sourceOffset:(count-1)*bytes
            toBuffer:runtime->recurrent_buffers[runtime->next^1][i] destinationOffset:0 size:bytes];
    }
    [copy endEncoding];
    if (runtime->cache_path) {
        save_runtime_cache(runtime, runtime->cache_path);
        [runtime->cache_path release]; runtime->cache_path = nil;
    }
    const uint32_t pixels=runtime->width*runtime->height;
    const size_t stride=element_count(runtime->target_shapes[0])/runtime->batch_size*mps_dtype_size(runtime->target_dtypes[0]);
    auto pipeline=runtime->target_dtypes[0]==MPSDataTypeFloat16 ? context->mask_f16_pipeline : context->mask_f32_pipeline;
    for (size_t i=0;i<count;++i) {
        auto &slot=runtime->slots[(first+i)%runtime->slots.size()];
        if (runtime->fused_refiner) {
            const uint32_t dims[4] = {runtime->width,runtime->height,runtime->target_shapes[0][3],runtime->target_shapes[0][2]};
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (!encoder) fail("failed to create Metal refiner encoder");
            [encoder setComputePipelineState:context->refine_pipeline];
            [encoder setBuffer:runtime->alpha_buffer offset:i*stride atIndex:0];
            [encoder setBuffer:slot.mask_buffer offset:0 atIndex:1];
            [encoder setBuffer:slot.invalid_buffer offset:0 atIndex:2];
            [encoder setBytes:dims length:sizeof(dims) atIndex:3];
            [encoder setBuffer:runtime->input_buffer offset:i*static_cast<size_t>(pixels)*3*dtype_size(context->model.model_dtype) atIndex:4];
            const NSUInteger threads = std::min<NSUInteger>(256,context->refine_pipeline.maxTotalThreadsPerThreadgroup);
            [encoder dispatchThreadgroups:MTLSizeMake((pixels+threads-1)/threads,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
            [encoder endEncoding];
        } else encode_pixels(command,pipeline,runtime->alpha_buffer,slot.mask_buffer,slot.invalid_buffer,pixels,i*stride);
    }
    [command commit];
    runtime->next^=1;
    runtime->batch_pending=0;
}

} // namespace

extern "C" SegmenterRvmMetalContext *segmenter_rvm_metal_create(
    const char *model_path,
    char *error,
    size_t error_len) {
    @autoreleasepool {
        try {
            if (model_path == nullptr) {
                fail("model_path was null");
            }
            std::unique_ptr<SegmenterRvmMetalContext> context(new SegmenterRvmMetalContext());
            context->model = load_model(model_path);
            NSData *model_bytes = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:model_path]
                                                       options:NSDataReadingMappedIfSafe error:nil];
            if (!model_bytes || model_bytes.length > UINT32_MAX) fail("failed to fingerprint Metal model");
            unsigned char digest[CC_SHA256_DIGEST_LENGTH];
            CC_SHA256(model_bytes.bytes, static_cast<CC_LONG>(model_bytes.length), digest);
            char hex[CC_SHA256_DIGEST_LENGTH * 2 + 1];
            for (size_t i = 0; i < sizeof(digest); ++i) std::snprintf(hex + i*2, 3, "%02x", digest[i]);
            context->model_digest = hex;
            if (const char *value = std::getenv("SEGMENTER_METAL_BATCH")) {
                char *end = nullptr;
                const long batch = std::strtol(value, &end, 10);
                if (end == value || *end || batch < 1 || batch > static_cast<long>(kMaxBatchSize))
                    fail("Metal batch size must be an integer in 1..16");
                context->configured_batch_size = context->batch_size = static_cast<uint32_t>(batch);
            }
            context->device = MTLCreateSystemDefaultDevice();
            if (context->device == nil) {
                fail("failed to create Metal device");
            }
            context->command_queue = [context->device newCommandQueue];
            if (context->command_queue == nil) {
                fail("failed to create Metal command queue");
            }
            context->graph_device = [[MPSGraphDevice deviceWithMTLDevice:context->device] retain];
            if (context->graph_device == nil) {
                fail("failed to create MPSGraph device");
            }
            return context.release();
        } catch (const std::exception &ex) {
            set_error(error, error_len, ex.what());
            return nullptr;
        } catch (...) {
            set_error(error, error_len, "unknown RVM Metal initialization error");
            return nullptr;
        }
    }
}

static int submit_frame(
    SegmenterRvmMetalContext *context, const uint8_t *bgra, size_t input_len,
    uint32_t width, uint32_t height, uint32_t bytes_per_row, float downsample_ratio,
    char *error, size_t error_len, CVPixelBufferRef pixel_buffer) {
    @autoreleasepool {
        try {
            if (context == nullptr || bgra == nullptr) fail("null pointer passed to RVM Metal submit");
            const size_t count = static_cast<size_t>(width) * height;
            const size_t row_len = static_cast<size_t>(width) * 4;
            if (width == 0 || height == 0 || count > UINT32_MAX / 3 || bytes_per_row < row_len ||
                input_len < static_cast<size_t>(bytes_per_row) * height) {
                fail("RVM Metal BGRA dimensions or buffer length mismatch");
            }
            if (!std::isfinite(downsample_ratio) || downsample_ratio <= 0.0f || downsample_ratio > 1.0f) {
                fail("RVM Metal downsample ratio must be within (0, 1]");
            }
            using Clock = std::chrono::steady_clock;
            auto t0 = Clock::now();
            ensure_runtime(context, width, height, downsample_ratio);
            ensure_pixel_pipelines(context);
            auto t1 = Clock::now();
            GraphRuntime *runtime = context->runtime.get();
            if (runtime->submitted - runtime->received == runtime->slots.size()) fail("Metal frame queue is full");
            FrameResources &slot = runtime->slots[runtime->submitted % runtime->slots.size()];
            if (slot.command != nil || slot.completion != nil) fail("Metal frame slot is still in use");
            const size_t input_offset=runtime->batch_pending*count*3*dtype_size(context->model.model_dtype);
            if (slot.mask_buffer == nil) slot.mask_buffer = runtime->mask_pool->take();
            if (pixel_buffer) {
                if (!context->texture_cache && CVMetalTextureCacheCreate(kCFAllocatorDefault, nullptr,
                    context->device, nullptr, &context->texture_cache) != kCVReturnSuccess) fail("failed to create Metal texture cache");
                if (CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, context->texture_cache,
                    pixel_buffer, nullptr, MTLPixelFormatBGRA8Unorm, width, height, 0, &slot.source_texture) == kCVReturnSuccess) {
                    slot.source_pixel_buffer = CVPixelBufferRetain(pixel_buffer);
                }
                // A non-IOSurface decoder buffer can still use the staging path.
            }
            if (!slot.source_texture) {
                if (!slot.bgra_buffer) slot.bgra_buffer = [context->device newBufferWithLength:count * 4 options:MTLResourceStorageModeShared];
                if (!slot.bgra_buffer) fail("failed to allocate Metal upload buffer");
                uint8_t *upload = static_cast<uint8_t *>([slot.bgra_buffer contents]);
                if (bytes_per_row == row_len) {
                    std::memcpy(upload, bgra, count * 4);
                } else {
                    for (size_t y = 0; y < height; ++y) {
                        std::memcpy(upload + y * row_len, bgra + y * bytes_per_row, row_len);
                    }
                }
            }
            *static_cast<uint32_t *>([slot.invalid_buffer contents]) = 0;
            auto t2 = Clock::now();
            SegmenterCommandBuffer *command;
            if (runtime->batch_pending) {
                command = runtime->slots[(runtime->submitted - runtime->batch_pending) % runtime->slots.size()].command;
            } else {
                id<MTLCommandBuffer> root = [context->command_queue commandBuffer];
                if (root == nil) fail("failed to create Metal command buffer");
                command = [[[SegmenterCommandBuffer alloc] initWithCommandBuffer:root] autorelease];
            }
            if (command == nil) fail("failed to create Metal command buffer");
            // Retain before encoding: MPSGraph may commit partial command buffers.
            // Runtime teardown drains them even if a later encoding step throws.
            slot.command = [command retain];
            if (slot.source_texture) {
                id<MTLComputeCommandEncoder> e = [command computeCommandEncoder];
                if (!e) fail("failed to create native texture command encoder");
                auto pipeline = context->unpack_texture_pipeline;
                [e setComputePipelineState:pipeline];
                [e setTexture:CVMetalTextureGetTexture(slot.source_texture) atIndex:0];
                [e setBuffer:runtime->input_buffer offset:input_offset atIndex:1];
                [e setBuffer:context->normalized_bytes offset:0 atIndex:2];
                uint32_t pixels = static_cast<uint32_t>(count);
                [e setBytes:&pixels length:sizeof(pixels) atIndex:3];
                NSUInteger threads = std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup);
                [e dispatchThreadgroups:MTLSizeMake((count+threads-1)/threads,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
                [e endEncoding];
            } else encode_pixels(command, context->unpack_pipeline, slot.bgra_buffer,
                          runtime->input_buffer, context->normalized_bytes, static_cast<uint32_t>(count),0,input_offset);
            runtime->submitted++;
            runtime->batch_pending++;
            if (runtime->batch_pending==runtime->batch_size) flush_batch(context);
            if (context->profile) {
                auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b-a).count(); };
                context->setup_ms += ms(t0, t1);
                context->encode_ms += ms(t2, Clock::now());
                context->copy_ms += ms(t1, t2);
            }
            return 0;
        } catch (const std::exception &ex) {
            set_error(error, error_len, ex.what());
            return -1;
        } catch (...) {
            set_error(error, error_len, "unknown RVM Metal submit error");
            return -1;
        }
    }
}

extern "C" int segmenter_rvm_metal_submit(
    SegmenterRvmMetalContext *context, const uint8_t *bgra, size_t input_len,
    uint32_t width, uint32_t height, uint32_t bytes_per_row, float downsample_ratio,
    char *error, size_t error_len) {
    return submit_frame(context, bgra, input_len, width, height, bytes_per_row, downsample_ratio, error, error_len, nullptr);
}
extern "C" int segmenter_rvm_metal_submit_pixel_buffer(
    SegmenterRvmMetalContext *context, void *native_buffer, float downsample_ratio, char *error, size_t error_len) {
    CVPixelBufferRef buffer = static_cast<CVPixelBufferRef>(native_buffer);
    if (!buffer || CVPixelBufferGetPixelFormatType(buffer)!=kCVPixelFormatType_32BGRA || CVPixelBufferIsPlanar(buffer)) {
        set_error(error,error_len,"Metal requires a BGRA pixel buffer"); return -1;
    }
    return submit_frame(context, static_cast<const uint8_t *>(CVPixelBufferGetBaseAddress(buffer)),
        CVPixelBufferGetDataSize(buffer), static_cast<uint32_t>(CVPixelBufferGetWidth(buffer)),
        static_cast<uint32_t>(CVPixelBufferGetHeight(buffer)), static_cast<uint32_t>(CVPixelBufferGetBytesPerRow(buffer)),
        downsample_ratio,error,error_len,buffer);
}

static int receive_frame(
    SegmenterRvmMetalContext *context, uint8_t *mask, size_t mask_len,
    char *error, size_t error_len, void **lease, const uint8_t **data, size_t *length) {
    @autoreleasepool {
        try {
            if (!context || !context->runtime || (!mask && !lease)) fail("null pointer passed to RVM Metal receive");
            GraphRuntime *runtime = context->runtime.get();
            if (runtime->submitted == runtime->received) fail("Metal frame queue is empty");
            if (!lease && mask_len != static_cast<size_t>(runtime->width) * runtime->height * 4) fail("Metal mask length mismatch");
            FrameResources &slot = runtime->slots[runtime->received % runtime->slots.size()];
            using Clock = std::chrono::steady_clock;
            auto start = Clock::now();
            if (!slot.completion) flush_batch(context);
            NSError *command_error = [slot.command waitForExecution];
            NSError *batch_error = [slot.completion waitForExecution];
            if (!command_error) command_error=batch_error;
            auto completed = Clock::now();
            if (command_error != nil) fail(std::string("RVM Metal command failed: ") + [[command_error localizedDescription] UTF8String]);
            if (*static_cast<uint32_t *>([slot.invalid_buffer contents]) != 0) {
                fail("RVM Metal alpha output contained a non-finite value");
            }
            if (lease) {
                auto owned = std::make_unique<MaskBufferLease>();
                owned->pool = runtime->mask_pool;
                owned->buffer = slot.mask_buffer;
                *data = static_cast<const uint8_t *>([slot.mask_buffer contents]);
                *length = runtime->mask_pool->length;
                *lease = owned.release();
                slot.mask_buffer = nil;
            } else std::memcpy(mask, [slot.mask_buffer contents], mask_len);
            slot.release_source();
            [slot.command release];
            slot.command = nil;
            [slot.completion release];
            slot.completion = nil;
            runtime->received++;
            context->frames++;
            if (context->profile) {
                context->wait_ms += std::chrono::duration<double, std::milli>(completed-start).count();
                context->copy_ms += std::chrono::duration<double, std::milli>(Clock::now()-completed).count();
            }
            return 0;
        } catch (const std::exception &ex) {
            set_error(error, error_len, ex.what());
            return -1;
        } catch (...) {
            set_error(error, error_len, "unknown RVM Metal receive error");
            return -1;
        }
    }
}

extern "C" int segmenter_rvm_metal_receive(
    SegmenterRvmMetalContext *context, uint8_t *mask, size_t mask_len, char *error, size_t error_len) {
    return receive_frame(context,mask,mask_len,error,error_len,nullptr,nullptr,nullptr);
}
extern "C" int segmenter_rvm_metal_receive_buffer(
    SegmenterRvmMetalContext *context, void **lease, const uint8_t **data, size_t *length, char *error, size_t error_len) {
    if (!lease || !data || !length) { set_error(error,error_len,"null Metal receive output"); return -1; }
    *lease = nullptr; *data = nullptr; *length = 0;
    return receive_frame(context,nullptr,0,error,error_len,lease,data,length);
}
extern "C" void segmenter_rvm_metal_release_buffer(void *lease) {
    delete static_cast<MaskBufferLease *>(lease);
}

extern "C" int segmenter_rvm_metal_run(
    SegmenterRvmMetalContext *context, const uint8_t *bgra, size_t input_len,
    uint32_t width, uint32_t height, uint32_t bytes_per_row, float downsample_ratio,
    uint8_t *mask, size_t mask_len, char *error, size_t error_len) {
    if (!mask || mask_len != static_cast<size_t>(width) * height * 4 ||
        (context && context->runtime && context->runtime->submitted != context->runtime->received)) {
        set_error(error, error_len, "synchronous Metal run requires a valid mask and an empty queue");
        return -1;
    }
    // A synchronous caller cannot amortize a temporal batch. Choose one frame
    // before the first runtime is built; never reset an existing recurrent state.
    if (context && !context->runtime && !context->configured_batch_size) {
        context->configured_batch_size = context->batch_size = 1;
    }
    int result = segmenter_rvm_metal_submit(context, bgra, input_len, width, height, bytes_per_row,
                                          downsample_ratio, error, error_len);
    return result != 0 ? result : segmenter_rvm_metal_receive(context, mask, mask_len, error, error_len);
}

extern "C" void segmenter_rvm_metal_destroy(SegmenterRvmMetalContext *context) {
    delete context;
}

extern "C" size_t segmenter_rvm_metal_queue_capacity(SegmenterRvmMetalContext *context) {
    return context ? std::max<uint32_t>(3,context->batch_size*2) : 0;
}
