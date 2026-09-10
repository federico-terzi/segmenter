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
    const uint8_t *bgra,
    size_t input_len,
    uint32_t width,
    uint32_t height,
    uint32_t bytes_per_row,
    float downsample_ratio,
    uint8_t *mask,
    size_t mask_len,
    char *error,
    size_t error_len);

void segmenter_rvm_metal_destroy(SegmenterRvmMetalContext *context);

#ifdef __cplusplus
}
#endif
