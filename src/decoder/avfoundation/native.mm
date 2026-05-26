#include "./native.hpp"
#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>
#include <math.h>

static bool segd_nearly_equal(CGFloat value, CGFloat expected) {
  return fabs((double)value - (double)expected) < 0.0001;
}

static bool segd_transform_matches(CGAffineTransform transform, CGFloat a,
                                   CGFloat b, CGFloat c, CGFloat d) {
  return segd_nearly_equal(transform.a, a) &&
         segd_nearly_equal(transform.b, b) &&
         segd_nearly_equal(transform.c, c) &&
         segd_nearly_equal(transform.d, d);
}

void *segd_initialize_asset(const SEGDecodeOptions *options,
                            int32_t *error_code) {
  (void)error_code;
  NSString *asset_path = [NSString stringWithUTF8String:options->file_path];
  NSURL *asset_url = [NSURL fileURLWithPath:asset_path];
  AVAsset *asset = [AVAsset assetWithURL:asset_url];
  [asset retain];
  return asset;
}

void segd_release_asset(void *asset) { [(AVAsset *)asset release]; }

uint32_t segd_get_asset_duration(void *asset, SEGDecodeTime *duration) {
  if (!asset || !duration) {
    return SEGD_NOT_FOUND_ERROR;
  }

  AVAsset *av_asset = (AVAsset *)asset;
  CMTime asset_duration = av_asset.duration;
  if (!CMTIME_IS_NUMERIC(asset_duration) || asset_duration.timescale <= 0) {
    return SEGD_NOT_FOUND_ERROR;
  }

  duration->value = asset_duration.value;
  duration->timescale = asset_duration.timescale;
  return SEGD_SUCCESS;
}

uint32_t segd_get_asset_rotation(void *asset, int32_t *rotation) {
  if (!asset || !rotation) {
    return SEGD_NOT_FOUND_ERROR;
  }

  AVAsset *av_asset = (AVAsset *)asset;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  AVAssetTrack *video_track =
      [[av_asset tracksWithMediaType:AVMediaTypeVideo] firstObject];
#pragma clang diagnostic pop
  if (!video_track) {
    return SEGD_NOT_FOUND_ERROR;
  }

  CGAffineTransform transform = [video_track preferredTransform];
  if (segd_transform_matches(transform, 1, 0, 0, 1)) {
    *rotation = 0;
  } else if (segd_transform_matches(transform, 0, 1, -1, 0)) {
    *rotation = 90;
  } else if (segd_transform_matches(transform, -1, 0, 0, -1)) {
    *rotation = 180;
  } else if (segd_transform_matches(transform, 0, -1, 1, 0)) {
    *rotation = -90;
  } else {
    return SEGD_NOT_FOUND_ERROR;
  }

  return SEGD_SUCCESS;
}

void *segd_initialize_asset_reader(const SEGDecodeOptions *options,
                                   void *asset,
                                   int32_t *error_code) {
  (void)options;
  AVAsset *av_asset = (AVAsset *)asset;
  NSError *error = nil;
  AVAssetReader *asset_reader =
      [[AVAssetReader alloc] initWithAsset:av_asset error:&error];

  if (error) {
    *error_code = (int32_t)error.code;
    return NULL;
  }

  return asset_reader;
}

uint32_t segd_start_asset_reader(void *asset_reader) {
  AVAssetReader *reader = (AVAssetReader *)asset_reader;
  if (![reader startReading]) {
    return SEGD_START_ERROR;
  }

  return SEGD_SUCCESS;
}

void segd_release_asset_reader(void *asset_reader) {
  AVAssetReader *reader = (AVAssetReader *)asset_reader;
  if (reader.status == AVAssetReaderStatusReading) {
    [reader cancelReading];
  }
  [reader release];
}

void *segd_initialize_video_track_output(const SEGDecodeOptions *options,
                                         void *asset,
                                         void *asset_reader,
                                         int32_t *error_code) {
  AVAsset *av_asset = (AVAsset *)asset;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  AVAssetTrack *video_track =
      [[av_asset tracksWithMediaType:AVMediaTypeVideo] firstObject];
#pragma clang diagnostic pop
  if (!video_track) {
    *error_code = SEGD_NOT_FOUND_ERROR;
    return NULL;
  }

  if (options->output_video_format != SEGD_VIDEO_FORMAT_BGRA) {
    *error_code = SEGD_INVALID_VIDEO_FORMAT_ERROR;
    return NULL;
  }

  CGSize original_size = [video_track naturalSize];
  CGFloat original_width = fabs(original_size.width);
  CGFloat original_height = fabs(original_size.height);
  CGFloat max_original_dimension = MAX(original_width, original_height);
  CGFloat downscale_factor = 1.0;
  if (options->max_dimension > 0) {
    downscale_factor =
        MAX(1.0, max_original_dimension / (CGFloat)options->max_dimension);
  }

  uint32_t output_width = (uint32_t)floor(original_width / downscale_factor);
  uint32_t output_height = (uint32_t)floor(original_height / downscale_factor);
  if (output_width == 0) {
    output_width = 1;
  }
  if (output_height == 0) {
    output_height = 1;
  }

  NSMutableDictionary *output_settings =
      [NSMutableDictionary dictionaryWithDictionary:@{
        (id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA),
        (id)kCVPixelBufferWidthKey : @(output_width),
        (id)kCVPixelBufferHeightKey : @(output_height)
      }];

  NSOperatingSystemVersion version =
      [[NSProcessInfo processInfo] operatingSystemVersion];
  if (version.majorVersion >= 11 &&
      [video_track hasMediaCharacteristic:
                       AVMediaCharacteristicContainsHDRVideo]) {
    output_settings[AVVideoColorPropertiesKey] = @{
      AVVideoColorPrimariesKey : AVVideoColorPrimaries_ITU_R_709_2,
      AVVideoTransferFunctionKey : AVVideoTransferFunction_ITU_R_709_2,
      AVVideoYCbCrMatrixKey : AVVideoYCbCrMatrix_ITU_R_709_2
    };
  }

  AVAssetReaderTrackOutput *track_output =
      [[AVAssetReaderTrackOutput alloc] initWithTrack:video_track
                                       outputSettings:output_settings];
  track_output.alwaysCopiesSampleData = NO;

  AVAssetReader *reader = (AVAssetReader *)asset_reader;
  if ([reader canAddOutput:track_output]) {
    [reader addOutput:track_output];
  } else {
    *error_code = SEGD_ADD_OUTPUT_ERROR;
    [track_output release];
    return NULL;
  }

  return track_output;
}

void segd_release_track_output(void *video_track_output) {
  [(AVAssetReaderTrackOutput *)video_track_output release];
}

uint32_t segd_read_sample(void *asset_reader,
                          void *track_output,
                          SEGDecodedSample *sample) {
  AVAssetReader *reader = (AVAssetReader *)asset_reader;

  if (reader.status == AVAssetReaderStatusUnknown) {
    return SEGD_READ_SAMPLE_UNKNOWN;
  }
  if (reader.status == AVAssetReaderStatusCompleted) {
    return SEGD_READ_SAMPLE_COMPLETED;
  }
  if (reader.status == AVAssetReaderStatusFailed) {
    return SEGD_READ_SAMPLE_FAILED;
  }
  if (reader.status == AVAssetReaderStatusCancelled) {
    return SEGD_READ_SAMPLE_CANCELLED;
  }

  AVAssetReaderTrackOutput *av_track_output =
      (AVAssetReaderTrackOutput *)track_output;
  CMSampleBufferRef sample_buffer = [av_track_output copyNextSampleBuffer];
  if (!sample_buffer) {
    return SEGD_READ_SAMPLE_NO_SAMPLE;
  }

  CVImageBufferRef image_buffer = CMSampleBufferGetImageBuffer(sample_buffer);
  CMTime presentation_time =
      CMSampleBufferGetPresentationTimeStamp(sample_buffer);

  sample->sample_buffer = sample_buffer;
  sample->width = (uint32_t)CVPixelBufferGetWidth(image_buffer);
  sample->height = (uint32_t)CVPixelBufferGetHeight(image_buffer);
  sample->pts.value = presentation_time.value;
  sample->pts.timescale = presentation_time.timescale;

  return SEGD_READ_SAMPLE_SUCCESS;
}

uint32_t segd_lock_sample(const SEGDecodedSample *sample) {
  CMSampleBufferRef sample_buffer = (CMSampleBufferRef)sample->sample_buffer;
  CVImageBufferRef image_buffer = CMSampleBufferGetImageBuffer(sample_buffer);
  CVPixelBufferLockBaseAddress(image_buffer, kCVPixelBufferLock_ReadOnly);
  return SEGD_SUCCESS;
}

uint32_t segd_unlock_sample(const SEGDecodedSample *sample) {
  CMSampleBufferRef sample_buffer = (CMSampleBufferRef)sample->sample_buffer;
  CVImageBufferRef image_buffer = CMSampleBufferGetImageBuffer(sample_buffer);
  CVPixelBufferUnlockBaseAddress(image_buffer, kCVPixelBufferLock_ReadOnly);
  return SEGD_SUCCESS;
}

SEGDecodedSampleData segd_get_sample_data(const SEGDecodedSample *sample) {
  CMSampleBufferRef sample_buffer = (CMSampleBufferRef)sample->sample_buffer;
  CVImageBufferRef image_buffer = CMSampleBufferGetImageBuffer(sample_buffer);

  SEGDecodedSampleData data = {};
  OSType pixel_format = CVPixelBufferGetPixelFormatType(image_buffer);
  if (pixel_format != kCVPixelFormatType_32BGRA ||
      CVPixelBufferIsPlanar(image_buffer)) {
    return data;
  }

  data.planes[0] = CVPixelBufferGetBaseAddress(image_buffer);
  data.planes_size[0] = (uint32_t)CVPixelBufferGetDataSize(image_buffer);
  data.bytes_per_row[0] = (uint32_t)CVPixelBufferGetBytesPerRow(image_buffer);
  data.format = SEGD_VIDEO_FORMAT_BGRA;
  data.valid = true;
  return data;
}

void segd_release_sample(const SEGDecodedSample *sample) {
  CMSampleBufferRef sample_buffer = (CMSampleBufferRef)sample->sample_buffer;
  if (sample_buffer) {
    CFRelease(sample_buffer);
  }
}
