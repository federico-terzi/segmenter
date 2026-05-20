#include "./native.hpp"
#import <AVFoundation/AVFoundation.h>
#include <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>

void *sege_initialize_asset_writer(const SEGEncodeWriterOptions *options,
                                   int32_t *error_code) {
  NSURL *file_url = [NSURL
      fileURLWithPath:[NSString stringWithUTF8String:options->file_path]];

  AVFileType file_type = nil;
  if (options->format == SEGE_FORMAT_MP4) {
    file_type = AVFileTypeMPEG4;
  } else if (options->format == SEGE_FORMAT_MOV) {
    file_type = AVFileTypeQuickTimeMovie;
  } else {
    *error_code = SEGE_ERROR_INVALID_FORMAT;
    return NULL;
  }

  NSError *error = nil;
  AVAssetWriter *asset_writer = [[AVAssetWriter alloc] initWithURL:file_url
                                                          fileType:file_type
                                                             error:&error];
  [asset_writer retain];
  if (error) {
    *error_code = SEGE_ERROR_ASSET_WRITER_INITIALIZATION_FAILED;
    return NULL;
  }

  return asset_writer;
}

void sege_release_asset_writer(void *asset_writer) {
  [(AVAssetWriter *)asset_writer release];
}

uint32_t sege_start_asset_writer(void *asset_writer,
                                 const SEGEncodeTime *start_time) {
  AVAssetWriter *writer = (AVAssetWriter *)asset_writer;
  if (!writer) {
    return SEGE_ERROR_NULL_INPUT;
  }

  if (![writer startWriting]) {
    return SEGE_ERROR_START_WRITING_FAILED;
  }

  CMTime time = CMTimeMake(start_time->value, start_time->timescale);
  [writer startSessionAtSourceTime:time];
  return SEGE_SUCCESS;
}

uint32_t sege_finalize_asset_writer(void *asset_writer) {
  AVAssetWriter *writer = (AVAssetWriter *)asset_writer;
  if (!writer) {
    return SEGE_ERROR_NULL_INPUT;
  }

  dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
  [writer finishWritingWithCompletionHandler:^{
    dispatch_semaphore_signal(semaphore);
  }];
  dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);

  if (writer.status == AVAssetWriterStatusCompleted) {
    return SEGE_SUCCESS;
  }

  return SEGE_ERROR_ASSET_WRITER_FINALIZE_FAILED;
}

void *sege_initialize_asset_writer_input(
    const SEGEncodeWriterInputOptions *options,
    int32_t *error_code) {
  AVAssetWriter *asset_writer = (AVAssetWriter *)options->asset_writer;
  if (!asset_writer) {
    *error_code = SEGE_ERROR_NULL_INPUT;
    return NULL;
  }
  if (options->video_codec != SEGE_VIDEO_CODEC_H264) {
    *error_code = SEGE_ERROR_INVALID_FORMAT;
    return NULL;
  }

  NSDictionary *video_settings = @{
    AVVideoCodecKey : AVVideoCodecTypeH264,
    AVVideoWidthKey : @(options->video_width),
    AVVideoHeightKey : @(options->video_height)
  };

  AVAssetWriterInput *video_writer_input =
      [AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo
                                         outputSettings:video_settings];
  [video_writer_input retain];
  video_writer_input.expectsMediaDataInRealTime = YES;

  if (![asset_writer canAddInput:video_writer_input]) {
    *error_code = SEGE_ERROR_CANNOT_ADD_INPUT;
    [video_writer_input release];
    return NULL;
  }

  [asset_writer addInput:video_writer_input];
  return video_writer_input;
}

void sege_release_asset_writer_input(void *asset_writer_input) {
  [(AVAssetWriterInput *)asset_writer_input release];
}

uint32_t sege_wait_for_asset_writer_input_ready(void *asset_writer_input) {
  AVAssetWriterInput *writer_input = (AVAssetWriterInput *)asset_writer_input;
  if (!writer_input) {
    return SEGE_ERROR_NULL_INPUT;
  }

  while (![writer_input isReadyForMoreMediaData]) {
    [NSThread sleepForTimeInterval:0.01];
  }

  return SEGE_SUCCESS;
}

uint32_t sege_finalize_asset_writer_input(void *asset_writer_input) {
  AVAssetWriterInput *writer_input = (AVAssetWriterInput *)asset_writer_input;
  if (!writer_input) {
    return SEGE_ERROR_NULL_INPUT;
  }

  [writer_input markAsFinished];
  return SEGE_SUCCESS;
}

static uint32_t sege_copy_bgra_sample_data(CVPixelBufferRef pixel_buffer,
                                           const SEGEncodeVideoSample *sample) {
  if (sample->planes_count != 1 || sample->format != SEGE_VIDEO_FORMAT_BGRA) {
    return SEGE_ERROR_INVALID_FORMAT;
  }

  SEGEncodeVideoSamplePlane plane = sample->planes[0];
  uint8_t *input_data = (uint8_t *)plane.data;
  uint8_t *output_data = (uint8_t *)CVPixelBufferGetBaseAddress(pixel_buffer);
  uint32_t input_bytes_per_row = plane.bytes_per_row;
  uint32_t output_bytes_per_row =
      (uint32_t)CVPixelBufferGetBytesPerRow(pixel_buffer);
  uint32_t row_bytes = MIN(input_bytes_per_row, output_bytes_per_row);

  for (uint32_t y = 0; y < sample->height; y++) {
    uint8_t *input_row = input_data + y * input_bytes_per_row;
    uint8_t *output_row = output_data + y * output_bytes_per_row;
    memcpy(output_row, input_row, row_bytes);
  }

  return SEGE_SUCCESS;
}

uint32_t sege_send_video_sample(void *asset_writer,
                                void *asset_writer_input,
                                const SEGEncodeVideoSample *sample) {
  AVAssetWriter *writer = (AVAssetWriter *)asset_writer;
  AVAssetWriterInput *writer_input = (AVAssetWriterInput *)asset_writer_input;
  if (!writer || !writer_input) {
    return SEGE_ERROR_NULL_INPUT;
  }
  if (![writer_input isReadyForMoreMediaData]) {
    return SEGE_ERROR_WRITER_INPUT_NOT_READY;
  }
  if (sample->format != SEGE_VIDEO_FORMAT_BGRA) {
    return SEGE_ERROR_INVALID_FORMAT;
  }

  CVPixelBufferRef pixel_buffer = NULL;
  if (CVPixelBufferCreate(kCFAllocatorDefault, sample->width, sample->height,
                          kCVPixelFormatType_32BGRA, NULL,
                          &pixel_buffer) != 0) {
    return SEGE_ERROR_PIXEL_BUFFER_CREATION_FAILED;
  }

  CMFormatDescriptionRef format_desc = NULL;
  if (CMVideoFormatDescriptionCreateForImageBuffer(
          kCFAllocatorDefault, pixel_buffer, &format_desc) != 0) {
    CFRelease(pixel_buffer);
    return SEGE_ERROR_FORMAT_DESCRIPTION_CREATION_FAILED;
  }

  CMTime presentation_time =
      CMTimeMake(sample->pts.value, sample->pts.timescale);
  CMSampleTimingInfo info = {kCMTimeInvalid, presentation_time, kCMTimeInvalid};
  CMSampleBufferRef sample_buffer = NULL;
  if (CMSampleBufferCreateReadyWithImageBuffer(kCFAllocatorDefault,
                                               pixel_buffer, format_desc, &info,
                                               &sample_buffer) != 0) {
    CFRelease(format_desc);
    CFRelease(pixel_buffer);
    return SEGE_ERROR_SAMPLE_BUFFER_CREATION_FAILED;
  }

  CVPixelBufferLockBaseAddress(pixel_buffer, 0);
  uint32_t copy_result = sege_copy_bgra_sample_data(pixel_buffer, sample);
  CVPixelBufferUnlockBaseAddress(pixel_buffer, 0);
  if (copy_result != SEGE_SUCCESS) {
    CFRelease(sample_buffer);
    CFRelease(format_desc);
    CFRelease(pixel_buffer);
    return SEGE_ERROR_COPY_SAMPLE_DATA_FAILED;
  }

  BOOL append_result = [writer_input appendSampleBuffer:sample_buffer];
  CFRelease(sample_buffer);
  CFRelease(format_desc);
  CFRelease(pixel_buffer);

  if (!append_result) {
    return SEGE_ERROR_WRITER_INPUT_APPEND_FAILED;
  }

  return SEGE_SUCCESS;
}
