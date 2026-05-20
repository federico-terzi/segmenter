#ifndef SEGMENTER_DECODER_AVFOUNDATION_NATIVE_HPP
#define SEGMENTER_DECODER_AVFOUNDATION_NATIVE_HPP

#include <cstdio>
#include <stdint.h>

#define SEGD_BUFFER_MAX_VIDEO_PLANES 1

typedef struct {
  int64_t value;
  int32_t timescale;
} SEGDecodeTime;

typedef struct {
  void *sample_buffer; // CMSampleBufferRef
  uint32_t width;
  uint32_t height;
  SEGDecodeTime pts;
} SEGDecodedSample;

typedef struct {
  uint32_t valid;
  uint32_t format;
  void *planes[SEGD_BUFFER_MAX_VIDEO_PLANES];
  uint32_t planes_size[SEGD_BUFFER_MAX_VIDEO_PLANES];
  uint32_t bytes_per_row[SEGD_BUFFER_MAX_VIDEO_PLANES];
} SEGDecodedSampleData;

#define SEGD_SUCCESS 0
#define SEGD_NOT_FOUND_ERROR 1
#define SEGD_INVALID_VIDEO_FORMAT_ERROR 2
#define SEGD_ADD_OUTPUT_ERROR 3
#define SEGD_START_ERROR 4

#define SEGD_VIDEO_FORMAT_BGRA 1

typedef struct {
  char file_path[FILENAME_MAX];
  uint32_t output_video_format;
  uint32_t max_dimension;
} SEGDecodeOptions;

extern "C" void *segd_initialize_asset(const SEGDecodeOptions *options,
                                       int32_t *error_code);
extern "C" void segd_release_asset(void *asset);
extern "C" uint32_t segd_get_asset_duration(void *asset,
                                            SEGDecodeTime *duration);

extern "C" void *segd_initialize_asset_reader(const SEGDecodeOptions *options,
                                              void *asset,
                                              int32_t *error_code);
extern "C" uint32_t segd_start_asset_reader(void *asset_reader);
extern "C" void segd_release_asset_reader(void *asset_reader);

#define SEGD_READ_SAMPLE_SUCCESS 0
#define SEGD_READ_SAMPLE_UNKNOWN 1
#define SEGD_READ_SAMPLE_COMPLETED 2
#define SEGD_READ_SAMPLE_FAILED 3
#define SEGD_READ_SAMPLE_CANCELLED 4
#define SEGD_READ_SAMPLE_NO_SAMPLE 5

extern "C" uint32_t segd_read_sample(void *asset_reader, void *track_output,
                                     SEGDecodedSample *sample);
extern "C" uint32_t segd_lock_sample(const SEGDecodedSample *sample);
extern "C" uint32_t segd_unlock_sample(const SEGDecodedSample *sample);
extern "C" SEGDecodedSampleData
segd_get_sample_data(const SEGDecodedSample *sample);
extern "C" void segd_release_sample(const SEGDecodedSample *sample);

extern "C" void *
segd_initialize_video_track_output(const SEGDecodeOptions *options,
                                   void *asset,
                                   void *asset_reader,
                                   int32_t *error_code);
extern "C" void segd_release_track_output(void *video_track_output);

#endif
