// Native H.264 round-trip, variable timestamps, and early teardown regression.
// Build like test_macos_encoder.mm, using only the system media frameworks.
#include "../src/encoder/avfoundation/native.mm"
#include <cassert>
#include <vector>

static int64_t timestamp(int frame) {
  if (frame < 7) return frame * 512;
  if (frame < 9) return 6 * 512 + frame - 6;
  return (frame - 2) * 512;
}

static void round_trip(uint32_t width, uint32_t height, bool gray, bool mp4, uint32_t fast_h264 = 0, double expected_fps = 0) {
  @autoreleasepool {
    NSString *path = [NSTemporaryDirectory() stringByAppendingPathComponent:
        [NSString stringWithFormat:@"segmenter-h264-%@.%@", NSUUID.UUID.UUIDString, mp4 ? @"mp4" : @"mov"]];
    SEGEncodeWriterOptions options = {};
    assert(strlen(path.fileSystemRepresentation) < sizeof(options.file_path));
    strcpy(options.file_path, path.fileSystemRepresentation);
    options.format = mp4 ? SEGE_FORMAT_MP4 : SEGE_FORMAT_MOV;
    int32_t error = 0;
    void *writer = sege_initialize_asset_writer(&options, &error);
    assert(writer);
    SEGEncodeWriterInputOptions input_options = {writer, SEGE_VIDEO_CODEC_H264, width, height, gray, 15360, fast_h264, expected_fps};
    void *input = sege_initialize_asset_writer_input(&input_options, &error);
    assert(input);
    SEGEncodeTime start = {0, 15360};
    assert(sege_start_asset_writer(writer, &start) == SEGE_SUCCESS);
    const size_t stride = width * 4 + 28;
    std::vector<uint8_t> pixels(stride * height, 0xa5);
    constexpr int frames = 17;
    for (int i = 0; i < frames; ++i) {
      @autoreleasepool {
        for (size_t y = 0; y < height; ++y) for (size_t x = 0; x < width; ++x) {
          auto *p = &pixels[y * stride + x * 4];
          p[0] = static_cast<uint8_t>(x * 37 + y * 19 + i * 67);
          p[1] = gray ? p[0] : 255 - p[0];
          p[2] = gray ? p[0] : 100;
          p[3] = 255;
        }
        SEGEncodeVideoSample sample = {};
        sample.width = width; sample.height = height;
        sample.format = SEGE_VIDEO_FORMAT_BGRA; sample.planes_count = 1;
        sample.planes[0] = {pixels.data(), pixels.size(), static_cast<uint32_t>(stride)};
        sample.pts = {timestamp(i), 15360};
        assert(sege_wait_for_asset_writer_input_ready(input) == SEGE_SUCCESS);
        // Invalid source storage must fail before reading beyond the allocation.
        if (i == 0) {
          auto invalid = sample;
          invalid.planes[0].size = 1;
          assert(sege_send_video_sample(writer, input, &invalid) == SEGE_ERROR_INVALID_FORMAT);
        }
        assert(sege_send_video_sample(writer, input, &sample) == SEGE_SUCCESS);
      }
    }
    assert(sege_finalize_asset_writer_input(input) == SEGE_SUCCESS);
    assert(sege_finalize_asset_writer(writer) == SEGE_SUCCESS);
    sege_release_asset_writer_input(input);
    sege_release_asset_writer(writer);

    AVURLAsset *asset = [AVURLAsset URLAssetWithURL:[NSURL fileURLWithPath:path] options:nil];
    AVAssetReader *reader = [[AVAssetReader alloc] initWithAsset:asset error:nil];
    AVAssetTrack *track = [asset tracksWithMediaType:AVMediaTypeVideo].firstObject;
    assert(track);
    CMVideoFormatDescriptionRef description = (CMVideoFormatDescriptionRef)track.formatDescriptions.firstObject;
    assert(CMFormatDescriptionGetMediaSubType(description) == kCMVideoCodecType_H264);
    const uint8_t *sps = nullptr; size_t sps_size = 0;
    assert(CMVideoFormatDescriptionGetH264ParameterSetAtIndex(description, 0, &sps, &sps_size, nullptr, nullptr) == noErr);
    // Baseline/Main/High are supported by the Windows Media Foundation decoder;
    // High 4:4:4 Predictive (244), used by Apple's lossless mode, is excluded.
    assert(sps_size >= 4 && (sps[1] == 66 || sps[1] == 77 || sps[1] == 100));
    if (width == 1920 && height == 1080 && expected_fps == 60 && fast_h264) {
      assert(sps[3] >= 42); // 1080p60 requires at least H.264 level 4.2.
    }
    // AVFoundation's existing H.264 path rounds odd dimensions down to even.
    // Odd input retains that path; the optimized mask path is even-size only.
    const uint32_t display_width = width & ~1u, display_height = height & ~1u;
    assert(track.naturalSize.width == display_width && track.naturalSize.height == display_height);
    // Request the display size, as the production reader does.
    AVAssetReaderTrackOutput *output = [[AVAssetReaderTrackOutput alloc] initWithTrack:track outputSettings:
        @{(id)kCVPixelBufferPixelFormatTypeKey : @((width % 2 || height % 2) ?
            kCVPixelFormatType_32BGRA : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange),
          (id)kCVPixelBufferWidthKey : @(display_width), (id)kCVPixelBufferHeightKey : @(display_height)}];
    [reader addOutput:output]; assert([reader startReading]);
    int count = 0;
    for (;;) {
      @autoreleasepool {
        CMSampleBufferRef sample = [output copyNextSampleBuffer];
        if (!sample) break;
        assert(CMTimeCompare(CMSampleBufferGetPresentationTimeStamp(sample), CMTimeMake(timestamp(count), 15360)) == 0);
        CVPixelBufferRef image = CMSampleBufferGetImageBuffer(sample);
        assert(CVPixelBufferGetWidth(image) == display_width && CVPixelBufferGetHeight(image) == display_height);
        ++count; CFRelease(sample);
      }
    }
    assert(count == frames && reader.status == AVAssetReaderStatusCompleted);
    [output release]; [reader release];
    assert([[NSFileManager defaultManager] removeItemAtPath:path error:nil]);
  }
}

int main() {
  round_trip(128, 96, true, false);
  round_trip(128, 96, true, true);
  round_trip(129, 97, true, false);
  round_trip(128, 96, false, false);
  round_trip(128, 96, true, false, 1);
  round_trip(128, 96, true, true, 1);
  round_trip(128, 96, true, true, 1, 30000.0 / 1001.0);
  round_trip(1920, 1080, true, true, 1, 60);
  @autoreleasepool {
    NSString *path = [NSTemporaryDirectory() stringByAppendingPathComponent:
        [NSString stringWithFormat:@"segmenter-abort-%@.mov", NSUUID.UUID.UUIDString]];
    SEGEncodeWriterOptions options = {};
    strcpy(options.file_path, path.fileSystemRepresentation); options.format = SEGE_FORMAT_MOV;
    int32_t error = 0;
    void *writer = sege_initialize_asset_writer(&options, &error); assert(writer);
    SEGEncodeWriterInputOptions opts = {writer, SEGE_VIDEO_CODEC_H264, 128, 96, 1, 30};
    void *input = sege_initialize_asset_writer_input(&opts, &error); assert(input);
    SEGEncodeTime start = {0, 30}; assert(sege_start_asset_writer(writer, &start) == SEGE_SUCCESS);
    sege_release_asset_writer_input(input); sege_release_asset_writer(writer);
    [[NSFileManager defaultManager] removeItemAtPath:path error:nil];
  }
  puts("PASS: H.264 MOV/MP4 round trips, FPS hints and level selection, sub-millisecond PTS, odd/padded input, general BGRA, and early teardown");
}
