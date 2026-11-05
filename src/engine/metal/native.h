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

size_t segmenter_rvm_metal_queue_capacity(SegmenterRvmMetalContext *context);

// At most queue_capacity frames may be outstanding. Receive returns the oldest mask.
// All frames must be drained before changing dimensions or downsample ratio.
int segmenter_rvm_metal_submit(
    SegmenterRvmMetalContext *context, const uint8_t *bgra, size_t input_len,
    uint32_t width, uint32_t height, uint32_t bytes_per_row, float downsample_ratio,
    char *error, size_t error_len);
int segmenter_rvm_metal_receive(
    SegmenterRvmMetalContext *context, uint8_t *mask, size_t mask_len,
    char *error, size_t error_len);

// The caller retains and read-locks the BGRA CVPixelBuffer while submitting.
// The backend retains it until inference completes.
int segmenter_rvm_metal_submit_pixel_buffer(SegmenterRvmMetalContext *context,
    void *pixel_buffer, float downsample_ratio, char *error, size_t error_len);
// Transfers immutable CPU-readable mask storage; release on any thread.
int segmenter_rvm_metal_receive_buffer(SegmenterRvmMetalContext *context,
    void **lease, const uint8_t **data, size_t *length, char *error, size_t error_len);
void segmenter_rvm_metal_release_buffer(void *lease);

#ifdef __cplusplus
}
#endif
