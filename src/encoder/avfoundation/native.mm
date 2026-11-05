#include "./native.hpp"
#import <Accelerate/Accelerate.h>
#import <AVFoundation/AVFoundation.h>
#include <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <VideoToolbox/VideoToolbox.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <array>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

struct SegmenterWriterInput {
  AVAssetWriterInput *input;
  AVAssetWriter *writer; // Owned by the Rust writer, which outlives this input.
  uint32_t width = 0, height = 0;
  CVPixelBufferPoolRef pool = nullptr;
  CMVideoFormatDescriptionRef format = nullptr;
  std::array<uint8_t, 256> gray_luma = {};
  bool affine_luma = false;
  double wait_ms = 0, allocate_ms = 0, copy_ms = 0, append_ms = 0;
  bool profile = std::getenv("SEGMENTER_METAL_PROFILE") != nullptr;
  ~SegmenterWriterInput() {
    if (format) CFRelease(format);
    if (pool) CFRelease(pool);
    [input release];
  }
};

// Generate the LUT with the same system conversion as AVFoundation's BGRA
// path. A formula such as round(16 + gray * 219/255) differs for some byte
// values because vImage rounds its Rec.709 coefficients/intermediates.
static bool sege_make_gray_luma(std::array<uint8_t, 256> &table) {
  std::array<uint8_t, 256 * 2 * 4> bgra;
  std::array<uint8_t, 256 * 2> luma;
  std::array<uint8_t, 256> chroma;
  for (size_t i = 0; i < 512; ++i) {
    bgra[i*4] = bgra[i*4+1] = bgra[i*4+2] = static_cast<uint8_t>(i);
    bgra[i*4+3] = 255;
  }
  vImage_YpCbCrPixelRange range = {16, 128, 235, 240, 255, 0, 255, 0};
  vImage_ARGBToYpCbCr conversion;
  if (vImageConvert_ARGBToYpCbCr_GenerateConversion(kvImage_ARGBToYpCbCrMatrix_ITU_R_709_2,
      &range, &conversion, kvImageARGB8888, kvImage420Yp8_CbCr8, kvImageNoFlags)) return false;
  vImage_Buffer source = {bgra.data(), 2, 256, 1024};
  vImage_Buffer y = {luma.data(), 2, 256, 256};
  vImage_Buffer uv = {chroma.data(), 1, 128, 256};
  const uint8_t permute[4] = {3, 2, 1, 0};
  if (vImageConvert_ARGB8888To420Yp8_CbCr8(&source, &y, &uv, &conversion,
      permute, kvImageDoNotTile)) return false;
  // A grayscale fast path is valid only with neutral chroma.
  for (auto c : chroma) if (c != 128) return false;
  std::copy_n(luma.begin(), table.size(), table.begin());
  return true;
}

// This integer mapping reproduces vImage's grayscale coefficients on Apple
// Silicon. Verify every entry before using it; retain the LUT fallback if a
// future OS changes the conversion. This is not a rounded Rec.709 approximation.
static bool sege_gray_affine_supported(const std::array<uint8_t, 256> &table) {
  for (unsigned x = 0; x < 256; ++x)
    if (table[x] != 16 + ((x * 7035 + 4297) >> 13)) return false;
  return true;
}

static void sege_copy_gray_mask(CVPixelBufferRef buffer, const SEGEncodeVideoSample *sample,
                               const std::array<uint8_t, 256> &table, bool affine = false) {
  auto *luma = static_cast<uint8_t *>(CVPixelBufferGetBaseAddressOfPlane(buffer, 0));
  auto *chroma = static_cast<uint8_t *>(CVPixelBufferGetBaseAddressOfPlane(buffer, 1));
  const size_t y_stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0);
  memset(chroma, 128, CVPixelBufferGetBytesPerRowOfPlane(buffer, 1) * CVPixelBufferGetHeightOfPlane(buffer, 1));
#if defined(__aarch64__)
  // Four 64-byte table lookups cover the complete byte domain exactly. Out of
  // range indices produce zero, so only one lookup contributes to each lane.
  uint8x16x4_t lut[4];
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) lut[i].val[j] = vld1q_u8(table.data() + i*64 + j*16);
  }
#endif
  for (size_t row = 0; row < sample->height; ++row) {
    const auto *src = static_cast<const uint8_t *>(sample->planes[0].data) + row * sample->planes[0].bytes_per_row;
    auto *dst = luma + row * y_stride;
    size_t x = 0;
#if defined(__aarch64__)
    if (affine) {
      auto convert = [](uint16x8_t values) {
        const auto bias = vdupq_n_u32(4297 + 16 * 8192);
        return vmovn_u16(vcombine_u16(
            vshrn_n_u32(vmlal_n_u16(bias, vget_low_u16(values), 7035), 13),
            vshrn_n_u32(vmlal_n_u16(bias, vget_high_u16(values), 7035), 13)));
      };
      for (; x + 16 <= sample->width; x += 16) {
        const auto blue = vld4q_u8(src + x * 4).val[0];
        vst1q_u8(dst + x, vcombine_u8(convert(vmovl_u8(vget_low_u8(blue))),
                                    convert(vmovl_u8(vget_high_u8(blue)))));
      }
    }
    for (; x + 16 <= sample->width; x += 16) {
      uint8x16_t indices = vld4q_u8(src + x*4).val[0];
      uint8x16_t values = vqtbl4q_u8(lut[0], indices);
      for (size_t i = 1; i < 4; ++i) {
        indices = vsubq_u8(indices, vdupq_n_u8(64));
        values = vorrq_u8(values, vqtbl4q_u8(lut[i], indices));
      }
      vst1q_u8(dst + x, values);
    }
#endif
    for (; x < sample->width; ++x) dst[x] = table[src[x*4]];
  }
}

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
  if (error) {
    [asset_writer release];
    *error_code = SEGE_ERROR_ASSET_WRITER_INITIALIZATION_FAILED;
    return NULL;
  }

  return asset_writer;
}

void sege_release_asset_writer(void *asset_writer) {
  AVAssetWriter *writer = (AVAssetWriter *)asset_writer;
  // Early preparation can start writing before inference fails or is cancelled.
  if (writer.status == AVAssetWriterStatusWriting) [writer cancelWriting];
  [writer release];
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
  dispatch_release(semaphore);

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

  if (options->fast_h264) {
    // Constant quality preserves faint matte detail more efficiently than the
    // tested average-bitrate presets. Keep 8-bit High profile and P-frames;
    // quality 1.0 would select Apple's less portable High 4:4:4 lossless mode.
    NSMutableDictionary *compression = [@{
      AVVideoProfileLevelKey : AVVideoProfileLevelH264HighAutoLevel,
      AVVideoAllowFrameReorderingKey : @NO,
      (id)kVTCompressionPropertyKey_Quality : @0.83
    } mutableCopy];
    // AutoLevel needs the source FPS to choose a conforming H.264 level
    // (e.g. 4.2 for 1080p60). This never replaces or quantizes frame timestamps.
    if (std::isfinite(options->expected_frame_rate) && options->expected_frame_rate > 0) {
      compression[AVVideoExpectedSourceFrameRateKey] = @(options->expected_frame_rate);
    }
    video_settings = @{
      AVVideoCodecKey : AVVideoCodecTypeH264,
      AVVideoWidthKey : @(options->video_width),
      AVVideoHeightKey : @(options->video_height),
      AVVideoCompressionPropertiesKey : compression
    };
    [compression release];
  }

  AVAssetWriterInput *video_writer_input = nil;
  @try {
    video_writer_input = [AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo
        outputSettings:video_settings];
  } @catch (NSException *exception) {
    // Some OS/encoder versions may not accept optional compression properties.
    // Restore the original public AVFoundation settings rather than aborting.
    if (!options->fast_h264) {
      *error_code = SEGE_ERROR_CANNOT_ADD_INPUT;
      return nullptr;
    }
    std::fprintf(stderr, "INFO: H.264 fast preset unavailable; using original compression\n");
    @try {
      video_writer_input = [AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo
          outputSettings:@{AVVideoCodecKey: AVVideoCodecTypeH264,
              AVVideoWidthKey: @(options->video_width), AVVideoHeightKey: @(options->video_height)}];
    } @catch (NSException *fallback_exception) {
      *error_code = SEGE_ERROR_CANNOT_ADD_INPUT;
      return nullptr;
    }
  }
  if (!video_writer_input) {
    *error_code = SEGE_ERROR_CANNOT_ADD_INPUT;
    return nullptr;
  }
  [video_writer_input retain];
  if (options->source_timescale > 0) video_writer_input.mediaTimeScale = options->source_timescale;
  // Input is a file, so allow AVFoundation to schedule offline encoding.
  video_writer_input.expectsMediaDataInRealTime = NO;

  if (![asset_writer canAddInput:video_writer_input]) {
    *error_code = SEGE_ERROR_CANNOT_ADD_INPUT;
    [video_writer_input release];
    return NULL;
  }

  auto *state = new SegmenterWriterInput{video_writer_input, asset_writer};
  state->width = options->video_width;
  state->height = options->video_height;
  const bool gray = options->grayscale_mask && options->video_width % 2 == 0 &&
      options->video_height % 2 == 0 && sege_make_gray_luma(state->gray_luma);
  state->affine_luma = gray && sege_gray_affine_supported(state->gray_luma);
  // General BGRA and odd sizes keep the original conversion path. Native masks
  // use the exact vImage grayscale mapping, without repeated RGB/chroma math.
  NSMutableDictionary *attributes = [@{
    (id)kCVPixelBufferPixelFormatTypeKey : @(gray ? kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange : kCVPixelFormatType_32BGRA),
    (id)kCVPixelBufferWidthKey : @(options->video_width),
    (id)kCVPixelBufferHeightKey : @(options->video_height)
  } mutableCopy];
  if (gray) attributes[(id)kCVPixelBufferIOSurfacePropertiesKey] = @{};
  if (CVPixelBufferPoolCreate(kCFAllocatorDefault, nullptr,
                             (CFDictionaryRef)attributes, &state->pool) != kCVReturnSuccess) {
    [attributes release];
    *error_code = SEGE_ERROR_PIXEL_BUFFER_CREATION_FAILED;
    delete state;
    return nullptr;
  }
  [attributes release];
  [asset_writer addInput:video_writer_input];
  return state;
}

void sege_release_asset_writer_input(void *asset_writer_input) {
  delete static_cast<SegmenterWriterInput *>(asset_writer_input);
}

uint32_t sege_wait_for_asset_writer_input_ready(void *asset_writer_input) {
  auto start = std::chrono::steady_clock::now();
  auto *state = static_cast<SegmenterWriterInput *>(asset_writer_input);
  if (!state) {
    return SEGE_ERROR_NULL_INPUT;
  }

  while (![state->input isReadyForMoreMediaData]) {
    if (state->writer.status != AVAssetWriterStatusWriting) {
      return SEGE_ERROR_WRITER_INPUT_NOT_READY;
    }
    [NSThread sleepForTimeInterval:0.0001];
  }

  if (state->profile) state->wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

  return SEGE_SUCCESS;
}

uint32_t sege_finalize_asset_writer_input(void *asset_writer_input) {
  auto *state = static_cast<SegmenterWriterInput *>(asset_writer_input);
  if (!state) {
    return SEGE_ERROR_NULL_INPUT;
  }

  [state->input markAsFinished];
  if (state->profile) std::fprintf(stderr, "WRITER: wait=%.3fms allocate=%.3fms copy=%.3fms append=%.3fms\n", state->wait_ms, state->allocate_ms, state->copy_ms, state->append_ms);
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
  auto start = std::chrono::steady_clock::now();
  AVAssetWriter *writer = (AVAssetWriter *)asset_writer;
  auto *state = static_cast<SegmenterWriterInput *>(asset_writer_input);
  if (!writer || !state) {
    return SEGE_ERROR_NULL_INPUT;
  }
  AVAssetWriterInput *writer_input = state->input;
  if (![writer_input isReadyForMoreMediaData]) {
    return SEGE_ERROR_WRITER_INPUT_NOT_READY;
  }
  if (!sample || sample->format != SEGE_VIDEO_FORMAT_BGRA || sample->planes_count != 1) {
    return SEGE_ERROR_INVALID_FORMAT;
  }
  const uint64_t row_bytes = static_cast<uint64_t>(sample->width) * 4;
  if (sample->width != state->width || sample->height != state->height ||
      !sample->planes[0].data || sample->planes[0].bytes_per_row < row_bytes ||
      sample->planes[0].size < static_cast<uint64_t>(sample->planes[0].bytes_per_row) * sample->height) {
    return SEGE_ERROR_INVALID_FORMAT;
  }

  CVPixelBufferRef pixel_buffer = NULL;
  if (CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, state->pool, &pixel_buffer) != kCVReturnSuccess) {
    return SEGE_ERROR_PIXEL_BUFFER_CREATION_FAILED;
  }

  if (!state->format && CMVideoFormatDescriptionCreateForImageBuffer(
          kCFAllocatorDefault, pixel_buffer, &state->format) != 0) {
    CFRelease(pixel_buffer);
    return SEGE_ERROR_FORMAT_DESCRIPTION_CREATION_FAILED;
  }

  CMTime presentation_time =
      CMTimeMake(sample->pts.value, sample->pts.timescale);
  CMSampleTimingInfo info = {kCMTimeInvalid, presentation_time, kCMTimeInvalid};
  CMSampleBufferRef sample_buffer = NULL;
  if (CMSampleBufferCreateReadyWithImageBuffer(kCFAllocatorDefault,
                                               pixel_buffer, state->format, &info,
                                               &sample_buffer) != 0) {
    CFRelease(pixel_buffer);
    return SEGE_ERROR_SAMPLE_BUFFER_CREATION_FAILED;
  }

  auto allocated = std::chrono::steady_clock::now();
  if (CVPixelBufferLockBaseAddress(pixel_buffer, 0) != kCVReturnSuccess) {
    CFRelease(sample_buffer);
    CFRelease(pixel_buffer);
    return SEGE_ERROR_COPY_SAMPLE_DATA_FAILED;
  }
  uint32_t copy_result = SEGE_SUCCESS;
  if (CVPixelBufferIsPlanar(pixel_buffer)) {
    sege_copy_gray_mask(pixel_buffer, sample, state->gray_luma, state->affine_luma);
  } else {
    copy_result = sege_copy_bgra_sample_data(pixel_buffer, sample);
  }
  CVPixelBufferUnlockBaseAddress(pixel_buffer, 0);
  if (copy_result != SEGE_SUCCESS) {
    CFRelease(sample_buffer);
    CFRelease(pixel_buffer);
    return SEGE_ERROR_COPY_SAMPLE_DATA_FAILED;
  }

  auto copied = std::chrono::steady_clock::now();
  BOOL append_result = [writer_input appendSampleBuffer:sample_buffer];
  CFRelease(sample_buffer);
  CFRelease(pixel_buffer);
  if (state->profile) {
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b-a).count(); };
    state->allocate_ms += ms(start, allocated);
    state->copy_ms += ms(allocated, copied);
    state->append_ms += ms(copied, std::chrono::steady_clock::now());
  }

  if (!append_result) {
    return SEGE_ERROR_WRITER_INPUT_APPEND_FAILED;
  }

  return SEGE_SUCCESS;
}
