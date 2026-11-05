// Isolated native encoder timing. Input is prefaulted BGRA masks; no inference or decode.
#include "../src/encoder/avfoundation/native.hpp"
#import <Foundation/Foundation.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <cmath>

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 5 || argc > 8) {
      fprintf(stderr, "usage: benchmark_encoder MASKS.bgra WIDTH HEIGHT OUTPUT.mov [FPS [FAST_H264_0_OR_1 [EXPECTED_FPS]]]\n");
      return 2;
    }
    const int fps = argc >= 6 ? atoi(argv[5]) : 30;
    uint32_t fast_h264 = 0;
    if (argc >= 7) {
      char *end = nullptr;
      unsigned long long parsed = strtoull(argv[6], &end, 10);
      assert(end != argv[6] && !*end && parsed <= 1);
      fast_h264 = static_cast<uint32_t>(parsed);
    }
    double expected_fps = 0;
    if (argc == 8) {
      char *end = nullptr;
      expected_fps = strtod(argv[7], &end);
      assert(end != argv[7] && !*end && std::isfinite(expected_fps) && expected_fps >= 0);
    }
    const uint32_t width = atoi(argv[2]), height = atoi(argv[3]);
    assert(fps > 0 && fps <= 1000 && width > 0 && height > 0 && width <= 16384 && height <= 16384);
    int fd = open(argv[1], O_RDONLY);
    assert(fd >= 0);
    struct stat stat_buffer;
    assert(fstat(fd, &stat_buffer) == 0 && stat_buffer.st_size > 0);
    auto *pixels = static_cast<uint8_t *>(mmap(nullptr, stat_buffer.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
    assert(pixels != MAP_FAILED);
    volatile unsigned touch = 0;
    for (size_t i = 0; i < size_t(stat_buffer.st_size); i += 4096) touch += pixels[i];
    const size_t frame_size = size_t(width) * height * 4;
    assert(stat_buffer.st_size % frame_size == 0);
    const size_t frames = stat_buffer.st_size / frame_size;
    SEGEncodeWriterOptions options = {};
    assert(strlen(argv[4]) < sizeof(options.file_path));
    strcpy(options.file_path, argv[4]);
    options.format = SEGE_FORMAT_MOV;
    int32_t error = 0;
    using Clock = std::chrono::steady_clock;
    auto begin = Clock::now();
    void *writer = sege_initialize_asset_writer(&options, &error);
    assert(writer);
    SEGEncodeWriterInputOptions input_options = {writer, SEGE_VIDEO_CODEC_H264, width, height, 1, fps, fast_h264, expected_fps};
    void *input = sege_initialize_asset_writer_input(&input_options, &error);
    assert(input);
    SEGEncodeTime start = {0, fps};
    assert(sege_start_asset_writer(writer, &start) == SEGE_SUCCESS);
    for (size_t i = 0; i < frames; ++i) {
      @autoreleasepool {
        SEGEncodeVideoSample sample = {};
        sample.format = SEGE_VIDEO_FORMAT_BGRA;
        sample.width = width; sample.height = height; sample.planes_count = 1;
        sample.pts = {int64_t(i), fps};
        sample.planes[0] = {pixels + i * frame_size, frame_size, width * 4};
        assert(sege_wait_for_asset_writer_input_ready(input) == SEGE_SUCCESS);
        assert(sege_send_video_sample(writer, input, &sample) == SEGE_SUCCESS);
      }
    }
    auto submitted = Clock::now();
    assert(sege_finalize_asset_writer_input(input) == SEGE_SUCCESS);
    assert(sege_finalize_asset_writer(writer) == SEGE_SUCCESS);
    auto finish = Clock::now();
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    printf("frames=%zu submit_ms=%.3f finalize_ms=%.3f total_ms=%.3f\n", frames,
        ms(begin, submitted), ms(submitted, finish), ms(begin, finish));
    sege_release_asset_writer_input(input); sege_release_asset_writer(writer);
    munmap(pixels, stat_buffer.st_size); close(fd);
  }
}
