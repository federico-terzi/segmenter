#ifndef SEGMENTER_ENCODER_AVFOUNDATION_NATIVE_HPP
#define SEGMENTER_ENCODER_AVFOUNDATION_NATIVE_HPP

#include <cstdio>
#include <stdint.h>

#define SEGE_SUCCESS 0
#define SEGE_ERROR_INVALID_FORMAT 1
#define SEGE_ERROR_ASSET_WRITER_INITIALIZATION_FAILED 2
#define SEGE_ERROR_CANNOT_ADD_INPUT 3
#define SEGE_ERROR_NULL_INPUT 4
#define SEGE_ERROR_START_WRITING_FAILED 5
#define SEGE_ERROR_WRITER_INPUT_NOT_READY 6
#define SEGE_ERROR_PIXEL_BUFFER_CREATION_FAILED 7
#define SEGE_ERROR_FORMAT_DESCRIPTION_CREATION_FAILED 8
#define SEGE_ERROR_SAMPLE_BUFFER_CREATION_FAILED 9
#define SEGE_ERROR_COPY_SAMPLE_DATA_FAILED 10
#define SEGE_ERROR_WRITER_INPUT_APPEND_FAILED 11
#define SEGE_ERROR_ASSET_WRITER_FINALIZE_FAILED 12

#define SEGE_FORMAT_MP4 1
#define SEGE_FORMAT_MOV 2

typedef struct {
  char file_path[FILENAME_MAX];
  uint32_t format;
} SEGEncodeWriterOptions;

extern "C" void *sege_initialize_asset_writer(
    const SEGEncodeWriterOptions *options,
    int32_t *error_code);
extern "C" void sege_release_asset_writer(void *asset_writer);

typedef struct {
  int64_t value;
  int32_t timescale;
} SEGEncodeTime;

extern "C" uint32_t sege_start_asset_writer(void *asset_writer,
                                            const SEGEncodeTime *start_time);
extern "C" uint32_t sege_finalize_asset_writer(void *asset_writer);

#define SEGE_VIDEO_CODEC_H264 1

typedef struct {
  void *asset_writer;
  uint32_t video_codec;
  uint32_t video_width;
  uint32_t video_height;
} SEGEncodeWriterInputOptions;

extern "C" void *sege_initialize_asset_writer_input(
    const SEGEncodeWriterInputOptions *options,
    int32_t *error_code);
extern "C" void sege_release_asset_writer_input(void *asset_writer_input);
extern "C" uint32_t
sege_wait_for_asset_writer_input_ready(void *asset_writer_input);
extern "C" uint32_t sege_finalize_asset_writer_input(void *asset_writer_input);

typedef struct {
  const void *data;
  uint64_t size;
  uint32_t bytes_per_row;
} SEGEncodeVideoSamplePlane;

#define SEGE_VIDEO_SAMPLE_MAX_PLANES 1
#define SEGE_VIDEO_FORMAT_BGRA 1

typedef struct {
  uint32_t format;
  uint32_t width;
  uint32_t height;
  SEGEncodeVideoSamplePlane planes[SEGE_VIDEO_SAMPLE_MAX_PLANES];
  uint32_t planes_count;
  SEGEncodeTime pts;
} SEGEncodeVideoSample;

extern "C" uint32_t sege_send_video_sample(
    void *asset_writer,
    void *asset_writer_input,
    const SEGEncodeVideoSample *sample);

#endif
