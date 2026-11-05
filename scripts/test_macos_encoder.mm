// Compile standalone: includes the encoder to exercise its internal SIMD path.
#include "../src/encoder/avfoundation/native.mm"
#include <cassert>
#include <vector>

int main() {
  @autoreleasepool {
    std::array<uint8_t, 256> table;
    assert(sege_make_gray_luma(table));
    for (bool affine : {false, true}) {
      if (affine && !sege_gray_affine_supported(table)) continue;
      for (size_t width : {2, 18, 258, 640}) {
      const size_t height = 8, stride = width * 4 + 28;
      std::vector<uint8_t> bgra(stride * height, 0xa5);
      for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
          auto *p = &bgra[y*stride+x*4];
          p[0] = p[1] = p[2] = static_cast<uint8_t>(x + y * 97);
          p[3] = 255;
        }
      }
      CVPixelBufferRef output = nullptr;
      assert(CVPixelBufferCreate(kCFAllocatorDefault, width, height,
          kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, nullptr, &output) == kCVReturnSuccess);
      assert(CVPixelBufferLockBaseAddress(output, 0) == kCVReturnSuccess);
      SEGEncodeVideoSample sample = {};
      sample.format = SEGE_VIDEO_FORMAT_BGRA;
      sample.width = width;
      sample.height = height;
      sample.planes_count = 1;
      sample.planes[0] = {bgra.data(), bgra.size(), static_cast<uint32_t>(stride)};
      sege_copy_gray_mask(output, &sample, table, affine);

      std::vector<uint8_t> reference_y(width * height), reference_uv(width * height / 2);
      vImage_YpCbCrPixelRange range = {16, 128, 235, 240, 255, 0, 255, 0};
      vImage_ARGBToYpCbCr conversion;
      assert(vImageConvert_ARGBToYpCbCr_GenerateConversion(kvImage_ARGBToYpCbCrMatrix_ITU_R_709_2,
          &range, &conversion, kvImageARGB8888, kvImage420Yp8_CbCr8, kvImageNoFlags) == 0);
      vImage_Buffer source = {bgra.data(), height, width, stride};
      vImage_Buffer y = {reference_y.data(), height, width, width};
      vImage_Buffer uv = {reference_uv.data(), height/2, width/2, width};
      const uint8_t permute[4] = {3, 2, 1, 0};
      assert(vImageConvert_ARGB8888To420Yp8_CbCr8(&source, &y, &uv, &conversion, permute, kvImageDoNotTile) == 0);
      for (size_t plane = 0; plane < 2; ++plane) {
        const auto *actual = static_cast<const uint8_t *>(CVPixelBufferGetBaseAddressOfPlane(output, plane));
        const auto *reference = plane == 0 ? reference_y.data() : reference_uv.data();
        size_t rows = CVPixelBufferGetHeightOfPlane(output, plane);
        size_t row_bytes = CVPixelBufferGetBytesPerRowOfPlane(output, plane);
        for (size_t row = 0; row < rows; ++row) {
          assert(std::memcmp(actual + row * row_bytes, reference + row * width, width) == 0);
        }
      }
      CVPixelBufferUnlockBaseAddress(output, 0);
      CFRelease(output);
    }
    }
    std::puts("Grayscale LUT and affine conversion match vImage, including padded rows and vector tails.");
  }
}
