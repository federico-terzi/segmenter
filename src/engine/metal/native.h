#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SegmenterRvmMetalContext SegmenterRvmMetalContext;

SegmenterRvmMetalContext *segmenter_rvm_metal_create(
    const char *model_path,
    char *error,
    size_t error_len);

int segmenter_rvm_metal_run(
    SegmenterRvmMetalContext *context,
    const float *input_nchw,
    size_t input_len,
    uint32_t width,
    uint32_t height,
    float downsample_ratio,
    float *alpha_nchw,
    size_t alpha_len,
    char *error,
    size_t error_len);

void segmenter_rvm_metal_destroy(SegmenterRvmMetalContext *context);

#ifdef __cplusplus
}
#endif
