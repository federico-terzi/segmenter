// Compare an H.264 export against uncompressed BGRA masks after the exact native
// video-range mapping. Optional benchmark_metal SHA metadata checks each PTS.
#include "../src/encoder/avfoundation/native.mm"
#include <fstream>
#include <vector>
#include <cmath>
#include <stdexcept>

static void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      require(argc == 3 || argc == 4, "usage: check_macos_mask_export MASKS.bgra VIDEO.mov [MASKS.sha256]");
      std::ifstream raw(argv[1], std::ios::binary), timestamps;
      require(raw.good(), "could not open raw masks");
      if (argc == 4) { timestamps.open(argv[3]); require(timestamps.good(), "could not open timestamps"); }
      AVURLAsset *asset = [AVURLAsset URLAssetWithURL:[NSURL fileURLWithPath:@(argv[2])] options:nil];
      AVAssetTrack *track = [asset tracksWithMediaType:AVMediaTypeVideo].firstObject;
      require(track != nil, "video track missing");
      const size_t width = track.naturalSize.width, height = track.naturalSize.height;
      require(width && height && width % 2 == 0 && height % 2 == 0, "comparison requires even dimensions");
      AVAssetReader *reader = [[[AVAssetReader alloc] initWithAsset:asset error:nil] autorelease];
      AVAssetReaderTrackOutput *output = [[[AVAssetReaderTrackOutput alloc] initWithTrack:track outputSettings:
          @{(id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange),
            (id)kCVPixelBufferWidthKey : @(width), (id)kCVPixelBufferHeightKey : @(height)}] autorelease];
      require([reader canAddOutput:output], "cannot add video output");
      [reader addOutput:output]; require([reader startReading], "cannot start video reader");
      std::array<uint8_t, 256> table; require(sege_make_gray_luma(table), "vImage mapping failed");
      std::vector<uint8_t> bgra(width * height * 4);
      uint64_t frames = 0, total = 0, absolute = 0, squared = 0, edges = 0, edge_error = 0;
      unsigned maximum = 0;
      for (;;) {
        @autoreleasepool {
          CMSampleBufferRef sample = [output copyNextSampleBuffer];
          if (!sample) break;
          raw.read(reinterpret_cast<char *>(bgra.data()), bgra.size());
          require(raw.gcount() == static_cast<std::streamsize>(bgra.size()), "raw masks ended before encoded frames");
          if (argc == 4) {
            uint64_t index, w, h; int64_t value; int32_t scale; std::string hash;
            require(bool(timestamps >> index >> w >> h >> value >> scale >> hash), "missing reference timestamp");
            require(index == frames + 1 && w == width && h == height, "reference dimensions or order differ");
            require(CMTimeCompare(CMSampleBufferGetPresentationTimeStamp(sample), CMTimeMake(value, scale)) == 0,
                    "encoded timestamp differs from reference");
          }
          CVPixelBufferRef pixel = CMSampleBufferGetImageBuffer(sample);
          require(CVPixelBufferLockBaseAddress(pixel, kCVPixelBufferLock_ReadOnly) == kCVReturnSuccess, "pixel lock failed");
          const auto *luma = static_cast<const uint8_t *>(CVPixelBufferGetBaseAddressOfPlane(pixel, 0));
          const auto *chroma = static_cast<const uint8_t *>(CVPixelBufferGetBaseAddressOfPlane(pixel, 1));
          const size_t stride = CVPixelBufferGetBytesPerRowOfPlane(pixel, 0);
          for (size_t y = 0; y < height; ++y) for (size_t x = 0; x < width; ++x) {
            const uint8_t *p = &bgra[(y * width + x) * 4];
            require(p[0] == p[1] && p[0] == p[2] && p[3] == 255, "raw mask is not opaque grayscale");
            unsigned error = std::abs(int(table[p[0]]) - int(luma[y * stride + x]));
            absolute += error; squared += error * error; maximum = std::max(maximum, error);
            if (p[0] > 0 && p[0] < 255) { edges++; edge_error += error; }
          }
          const size_t uv_stride = CVPixelBufferGetBytesPerRowOfPlane(pixel, 1);
          for (size_t y = 0; y < height / 2; ++y) for (size_t x = 0; x < width; ++x)
            require(chroma[y * uv_stride + x] == 128, "encoded mask chroma is not neutral");
          CVPixelBufferUnlockBaseAddress(pixel, kCVPixelBufferLock_ReadOnly); CFRelease(sample);
          frames++; total += width * height;
        }
      }
      require(reader.status == AVAssetReaderStatusCompleted && frames, "decoder did not finish successfully");
      require(raw.peek() == EOF, "raw masks have additional frames");
      if (argc == 4) { timestamps >> std::ws; require(timestamps.peek() == EOF, "extra reference timestamps"); }
      printf("{\"frames\":%llu,\"pixels\":%llu,\"exact_luma\":%s,\"maximum_luma_error\":%u,"
             "\"mean_luma_error\":%.9f,\"mean_edge_luma_error\":%.9f,\"neutral_chroma\":true,\"timestamps_checked\":%s,\"psnr_db\":",
             frames, total, squared ? "false" : "true", maximum, double(absolute) / total,
             edges ? double(edge_error) / edges : 0, argc == 4 ? "true" : "false");
      if (squared) printf("%.6f", 10 * log10(255.0 * 255 * total / squared)); else printf("null");
      puts("}");
    } catch (const std::exception &error) {
      fprintf(stderr, "%s\n", error.what()); return 1;
    }
  }
}
