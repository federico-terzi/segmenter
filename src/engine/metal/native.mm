#include "native.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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

struct GraphRuntime {
    uint32_t width = 0;
    uint32_t height = 0;
    float downsample_ratio = 0.0f;
    MPSGraph *graph = nil;
    MPSGraphTensor *src_feed = nil;
    std::array<MPSGraphTensor *, 4> recurrent_feeds = {};
    std::array<std::vector<uint32_t>, 4> recurrent_shapes = {};
    std::array<MPSGraphTensor *, 5> targets = {};
    std::array<MPSDataType, 5> target_dtypes = {};
    std::array<std::vector<uint32_t>, 5> target_shapes = {};
    std::array<id<MTLBuffer>, 4> recurrent_buffers = {};

    ~GraphRuntime() {
        [graph release];
        for (id<MTLBuffer> buffer : recurrent_buffers) {
            [buffer release];
        }
    }
};

} // namespace

struct SegmenterRvmMetalContext {
    ModelDef model;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> command_queue = nil;
    MPSGraphDevice *graph_device = nil;
    std::unique_ptr<GraphRuntime> runtime;

    ~SegmenterRvmMetalContext() {
        runtime.reset();
        [graph_device release];
        [command_queue release];
        [device release];
    }
};

namespace {

struct BuildContext {
    SegmenterRvmMetalContext *context = nullptr;
    GraphRuntime *runtime = nullptr;
    std::unordered_map<uint32_t, Value> values;

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

void build_conv(BuildContext &build, const NodeDef &node) {
    if (node.inputs.size() != 1 || node.tensor_refs.empty() || node.attrs.size() < 7) {
        fail("invalid Conv node in RVM Metal model");
    }
    Value &source = build.require_value(node.inputs[0]);
    const TensorDef &weights_def = build.require_tensor(node.tensor_refs[0]);
    Value weights = build.tensor_constant_for_graph(node.tensor_refs[0]);

    if (source.shape.size() != 4 || weights_def.shape.size() != 4) {
        fail("RVM Metal Conv expected rank-4 tensors");
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
    output.tensor = [build.runtime->graph convolution2DWithSourceTensor:build.ensure_tensor(source)
                                                          weightsTensor:weights.tensor
                                                             descriptor:desc
                                                                   name:nil];
    if (node.tensor_refs.size() > 1) {
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

std::unique_ptr<GraphRuntime> build_runtime(SegmenterRvmMetalContext *context, uint32_t width, uint32_t height, float downsample_ratio) {
    std::unique_ptr<GraphRuntime> runtime(new GraphRuntime());
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

    MPSGraphTensor *src_feed = [runtime->graph placeholderWithShape:make_shape({1, 3, height, width})
                                                           dataType:MPSDataTypeFloat32
                                                               name:nil];
    runtime->src_feed = src_feed;
    Value src;
    src.shape = {1, 3, height, width};
    src.dtype = mps_dtype(context->model.model_dtype);
    src.tensor = context->model.model_dtype == kDTypeF16
        ? [runtime->graph castTensor:src_feed toType:MPSDataTypeFloat16 name:nil]
        : src_feed;
    build.values[context->model.src_id] = src;

    Value downsample;
    downsample.dtype = MPSDataTypeFloat32;
    downsample.shape = {1};
    downsample.floats = {downsample_ratio};
    build.values[context->model.downsample_id] = std::move(downsample);

    for (const NodeDef &node : context->model.nodes) {
        build_graph_node(build, node);
    }

    Value &pha = build.require_value(context->model.pha_id);
    runtime->targets[0] = pha.tensor;
    runtime->target_dtypes[0] = pha.dtype;
    runtime->target_shapes[0] = pha.shape;
    if (pha.shape.size() != 4 || pha.shape[2] != height || pha.shape[3] != width) {
        fail("RVM Metal alpha output shape did not match input frame");
    }

    for (size_t i = 0; i < 4; ++i) {
        Value &state = build.require_value(context->model.recurrent_output_ids[i]);
        runtime->targets[i + 1] = state.tensor;
        runtime->target_dtypes[i + 1] = state.dtype;
        runtime->target_shapes[i + 1] = state.shape;
        if (runtime->recurrent_feeds[i] == nil || runtime->recurrent_shapes[i].empty()) {
            fail("RVM Metal graph did not create all recurrent input placeholders");
        }
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

void ensure_runtime(SegmenterRvmMetalContext *context, uint32_t width, uint32_t height, float downsample_ratio) {
    if (context->runtime &&
        context->runtime->width == width &&
        context->runtime->height == height &&
        std::fabs(context->runtime->downsample_ratio - downsample_ratio) < 1.0e-6f) {
        return;
    }

    context->runtime = build_runtime(context, width, height, downsample_ratio);
    for (size_t i = 0; i < 4; ++i) {
        context->runtime->recurrent_buffers[i] = new_zero_buffer(
            context->device,
            context->runtime->recurrent_shapes[i],
            mps_dtype(context->model.model_dtype));
    }
}

void copy_alpha_to_f32(const void *source, MPSDataType dtype, size_t count, float *target) {
    if (dtype == MPSDataTypeFloat32) {
        std::memcpy(target, source, count * sizeof(float));
        return;
    }
    if (dtype == MPSDataTypeFloat16) {
        const uint16_t *half = reinterpret_cast<const uint16_t *>(source);
        for (size_t i = 0; i < count; ++i) {
            target[i] = half_to_float(half[i]);
        }
        return;
    }
    fail("RVM Metal alpha output had an unsupported dtype");
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
            context->device = [MTLCreateSystemDefaultDevice() retain];
            if (context->device == nil) {
                fail("failed to create Metal device");
            }
            context->command_queue = [[context->device newCommandQueue] retain];
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

extern "C" int segmenter_rvm_metal_run(
    SegmenterRvmMetalContext *context,
    const float *input_nchw,
    size_t input_len,
    uint32_t width,
    uint32_t height,
    float downsample_ratio,
    float *alpha_nchw,
    size_t alpha_len,
    char *error,
    size_t error_len) {
    @autoreleasepool {
        try {
            if (context == nullptr || input_nchw == nullptr || alpha_nchw == nullptr) {
                fail("null pointer passed to RVM Metal run");
            }
            const size_t expected_input = static_cast<size_t>(width) * height * 3;
            const size_t expected_alpha = static_cast<size_t>(width) * height;
            if (input_len != expected_input || alpha_len != expected_alpha) {
                fail("RVM Metal input/output length mismatch");
            }
            if (!std::isfinite(downsample_ratio) || downsample_ratio <= 0.0f || downsample_ratio > 1.0f) {
                fail("RVM Metal downsample ratio must be within (0, 1]");
            }

            ensure_runtime(context, width, height, downsample_ratio);
            GraphRuntime *runtime = context->runtime.get();

            NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = [NSMutableDictionary dictionaryWithCapacity:5];
            id<MTLBuffer> input_buffer = [context->device newBufferWithBytes:input_nchw
                                                                       length:input_len * sizeof(float)
                                                                      options:MTLResourceStorageModeShared];
            if (input_buffer == nil) {
                fail("failed to allocate RVM Metal input buffer");
            }
            MPSGraphTensorData *input_data = [[[MPSGraphTensorData alloc] initWithMTLBuffer:input_buffer
                                                                                      shape:make_shape({1, 3, height, width})
                                                                                   dataType:MPSDataTypeFloat32] autorelease];
            [feeds setObject:input_data forKey:runtime->src_feed];

            for (size_t i = 0; i < 4; ++i) {
                MPSGraphTensorData *state_data = [[[MPSGraphTensorData alloc] initWithMTLBuffer:runtime->recurrent_buffers[i]
                                                                                          shape:make_shape(runtime->recurrent_shapes[i])
                                                                                       dataType:mps_dtype(context->model.model_dtype)] autorelease];
                [feeds setObject:state_data forKey:runtime->recurrent_feeds[i]];
            }

            NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results = [NSMutableDictionary dictionaryWithCapacity:5];
            std::array<id<MTLBuffer>, 5> result_buffers = {};
            for (size_t i = 0; i < runtime->targets.size(); ++i) {
                const size_t byte_len = element_count(runtime->target_shapes[i]) * mps_dtype_size(runtime->target_dtypes[i]);
                result_buffers[i] = [context->device newBufferWithLength:byte_len options:MTLResourceStorageModeShared];
                if (result_buffers[i] == nil) {
                    fail("failed to allocate RVM Metal output buffer");
                }
                MPSGraphTensorData *output_data = [[[MPSGraphTensorData alloc] initWithMTLBuffer:result_buffers[i]
                                                                                           shape:make_shape(runtime->target_shapes[i])
                                                                                        dataType:runtime->target_dtypes[i]] autorelease];
                [results setObject:output_data forKey:runtime->targets[i]];
            }

            [runtime->graph runWithMTLCommandQueue:context->command_queue
                                             feeds:feeds
                                  targetOperations:nil
                                 resultsDictionary:results];

            copy_alpha_to_f32(
                [result_buffers[0] contents],
                runtime->target_dtypes[0],
                expected_alpha,
                alpha_nchw);

            for (size_t i = 0; i < 4; ++i) {
                [runtime->recurrent_buffers[i] release];
                runtime->recurrent_buffers[i] = result_buffers[i + 1];
                result_buffers[i + 1] = nil;
            }

            [result_buffers[0] release];
            [input_buffer release];
            return 0;
        } catch (const std::exception &ex) {
            set_error(error, error_len, ex.what());
            return -1;
        } catch (...) {
            set_error(error, error_len, "unknown RVM Metal runtime error");
            return -1;
        }
    }
}

extern "C" void segmenter_rvm_metal_destroy(SegmenterRvmMetalContext *context) {
    delete context;
}
