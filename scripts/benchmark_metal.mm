// Native Metal regression benchmark. Build instructions are in README.md.
// Hashes cover uncompressed masks, dimensions, and original presentation times.
#include "../src/engine/metal/native.h"
#include "../src/decoder/avfoundation/native.hpp"
#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <vector>

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 4 || argc > 5) {
            std::fprintf(stderr, "usage: benchmark_metal MODEL VIDEO HASHES [RAW_MASKS]\n");
            return 2;
        }
        try {
            using Clock = std::chrono::steady_clock;
            char error[4096] = {};
            auto *engine = segmenter_rvm_metal_create(argv[1], error, sizeof(error));
            if (!engine) throw std::runtime_error(error);
            SEGDecodeOptions options = {};
            if (std::strlen(argv[2]) >= sizeof(options.file_path)) throw std::runtime_error("path too long");
            std::strcpy(options.file_path, argv[2]);
            options.output_video_format = SEGD_VIDEO_FORMAT_BGRA;
            const bool native_buffers = std::getenv("SEGMENTER_BENCH_NATIVE") != nullptr;
            options.native_buffers = native_buffers;
            int32_t code = 0, rotation = 0;
            void *asset = segd_initialize_asset(&options, &code);
            if (!asset) throw std::runtime_error("asset failed");
            segd_get_asset_rotation(asset, &rotation);
            if (rotation != 0) throw std::runtime_error("benchmark requires an unrotated input");
            void *reader = segd_initialize_asset_reader(&options, asset, &code);
            void *track = segd_initialize_video_track_output(&options, asset, reader, &code);
            if (!reader || !track || segd_start_asset_reader(reader)) throw std::runtime_error("decoder failed");
            FILE *hashes = std::fopen(argv[3], "w");
            FILE *raw = argc == 5 ? std::fopen(argv[4], "wb") : nullptr;
            if (!hashes || (argc == 5 && !raw)) throw std::runtime_error("output open failed");
            std::vector<double> timings;
            std::vector<uint8_t> mask;
            std::deque<SEGDecodedSample> pending;
            size_t written = 0;
            const uint8_t *pixels = nullptr;
            size_t pixel_bytes = 0;
            const bool async = std::getenv("SEGMENTER_BENCH_ASYNC") != nullptr;
            const bool hash = std::getenv("SEGMENTER_BENCH_NO_HASH") == nullptr;
            const bool predecode = std::getenv("SEGMENTER_BENCH_PREDECODE") != nullptr;
            size_t loops = 1;
            if (const char *value = std::getenv("SEGMENTER_BENCH_LOOPS")) {
                char *end = nullptr;
                const long parsed = std::strtol(value, &end, 10);
                if (!predecode || hash || raw || parsed < 1 || parsed > 100 || *end)
                    throw std::runtime_error("loops require predecoded timing mode and a count in 1..100");
                loops = static_cast<size_t>(parsed);
            }
            std::vector<SEGDecodedSample> decoded;
            size_t decode_index = 0;
            if (predecode) {
                for (;;) {
                    @autoreleasepool {
                        SEGDecodedSample sample = {};
                        auto status = segd_read_sample(reader, track, &sample);
                        if (status == SEGD_READ_SAMPLE_COMPLETED || status == SEGD_READ_SAMPLE_NO_SAMPLE) break;
                        if (status != SEGD_READ_SAMPLE_SUCCESS) throw std::runtime_error("predecode failed");
                        decoded.push_back(sample);
                    }
                }
            }
            auto wall_start = Clock::now(), steady_start = wall_start;
            auto write_mask = [&](const SEGDecodedSample &sample) {
                unsigned char digest[CC_SHA256_DIGEST_LENGTH] = {};
                if (hash) CC_SHA256(pixels, static_cast<CC_LONG>(pixel_bytes), digest);
                std::fprintf(hashes, "%zu %u %u %lld %d ", ++written, sample.width, sample.height,
                    static_cast<long long>(sample.pts.value), sample.pts.timescale);
                for (auto b : digest) std::fprintf(hashes, "%02x", b);
                std::fputc('\n', hashes);
                if (raw && std::fwrite(pixels, 1, pixel_bytes, raw) != pixel_bytes) throw std::runtime_error("raw write failed");
                if (written == 10) steady_start = Clock::now();
            };
#ifndef BENCH_SYNC
            auto receive = [&] {
                const auto sample = pending.front();
                mask.resize(size_t(sample.width) * sample.height * 4);
                auto start = Clock::now();
                void *lease = nullptr;
                if (native_buffers) {
                    if (segmenter_rvm_metal_receive_buffer(engine, &lease, &pixels, &pixel_bytes, error, sizeof(error))) throw std::runtime_error(error);
                } else {
                    if (segmenter_rvm_metal_receive(engine, mask.data(), mask.size(), error, sizeof(error))) throw std::runtime_error(error);
                    pixels = mask.data(); pixel_bytes = mask.size();
                }
                timings[written] += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                write_mask(sample);
                if (lease) segmenter_rvm_metal_release_buffer(lease);
                pending.pop_front();
            };
#else
            if (async) throw std::runtime_error("baseline benchmark is synchronous");
#endif
            while (true) {
                @autoreleasepool {
                    SEGDecodedSample sample = {};
                    auto status = predecode ? (decode_index < decoded.size() * loops ?
                        (sample = decoded[decode_index++ % decoded.size()], SEGD_READ_SAMPLE_SUCCESS) : SEGD_READ_SAMPLE_COMPLETED) :
                        segd_read_sample(reader, track, &sample);
                    if (status == SEGD_READ_SAMPLE_COMPLETED || status == SEGD_READ_SAMPLE_NO_SAMPLE) break;
                    if (status != SEGD_READ_SAMPLE_SUCCESS) throw std::runtime_error("sample failed");
                    segd_lock_sample(&sample);
                    auto data = segd_get_sample_data(&sample);
                    if (!data.valid) throw std::runtime_error("invalid sample");
                    mask.resize(size_t(sample.width) * sample.height * 4);
                    pixels = mask.data(); pixel_bytes = mask.size();
                    auto start = Clock::now();
                    int result;
#ifndef BENCH_SYNC
                    if (async) {
                        if (native_buffers) {
                            void *buffer = segd_retain_pixel_buffer(&sample);
                            result = segmenter_rvm_metal_submit_pixel_buffer(engine, buffer, 0.25f, error, sizeof(error));
                            segd_release_pixel_buffer(buffer);
                        } else result = segmenter_rvm_metal_submit(engine, static_cast<uint8_t *>(data.planes[0]),
                            data.planes_size[0], sample.width, sample.height, data.bytes_per_row[0], 0.25f,
                            error, sizeof(error));
                        pending.push_back(sample);
                    } else
#endif
                    result = segmenter_rvm_metal_run(engine, static_cast<uint8_t *>(data.planes[0]),
                        data.planes_size[0], sample.width, sample.height, data.bytes_per_row[0], 0.25f,
                        mask.data(), mask.size(), error, sizeof(error));
                    timings.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
                    segd_unlock_sample(&sample);
                    if (result) throw std::runtime_error(error);
#ifndef BENCH_SYNC
                    if (async) {
                        if (pending.size() == segmenter_rvm_metal_queue_capacity(engine)) receive();
                    } else
#endif
                    write_mask(sample);
                    if (!predecode) segd_release_sample(&sample);
                }
            }
#ifndef BENCH_SYNC
            while (!pending.empty()) receive();
#endif
            auto finished = Clock::now();
            for (auto &sample : decoded) segd_release_sample(&sample);
            if (timings.size() < 11) throw std::runtime_error("insufficient frames");
            double total = 0, steady = 0;
            for (size_t i = 0; i < timings.size(); ++i) {
                total += timings[i];
                if (i >= 10) steady += timings[i];
            }
            auto sorted = timings;
            std::sort(sorted.begin(), sorted.end());
            std::printf("frames=%zu first_ms=%.3f total_ms=%.3f steady_mean_ms=%.3f median_ms=%.3f p95_ms=%.3f\n",
                timings.size(), timings[0], total, steady / (timings.size()-10),
                sorted[sorted.size()/2], sorted[sorted.size()*95/100]);
            std::printf("decode_infer_wall_ms=%.3f steady_fps=%.3f hash=%d async=%d predecode=%d loops=%zu\n",
                std::chrono::duration<double, std::milli>(finished-wall_start).count(),
                (written-10) / std::chrono::duration<double>(finished-steady_start).count(), hash, async, predecode, loops);
            std::fclose(hashes);
            if (raw) std::fclose(raw);
            segd_release_track_output(track);
            segd_release_asset_reader(reader);
            segd_release_asset(asset);
            segmenter_rvm_metal_destroy(engine);
            return 0;
        } catch (const std::exception &ex) {
            std::fprintf(stderr, "%s\n", ex.what());
            return 1;
        }
    }
}
