// Compare complete uncompressed BGRA mask streams. Uses only the C++ standard library.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    try {
        if (argc != 5) throw std::runtime_error("usage: compare_masks BASELINE.bgra CANDIDATE.bgra WIDTH HEIGHT");
        const auto width = std::stoull(argv[3]), height = std::stoull(argv[4]);
        if (!width || !height || width > UINT32_MAX || height > UINT32_MAX ||
            width > std::numeric_limits<size_t>::max() / height / 4) throw std::runtime_error("invalid dimensions");
        const size_t count = static_cast<size_t>(width * height);
        std::ifstream baseline(argv[1], std::ios::binary), candidate(argv[2], std::ios::binary);
        if (!baseline || !candidate) throw std::runtime_error("could not open mask streams");
        std::vector<uint8_t> a(count * 4), b(count * 4);
        std::array<uint64_t, 256> histogram = {};
        uint64_t pixels = 0, absolute_error = 0, squared_error = 0;
        uint64_t edge_pixels = 0, edge_error = 0;
        size_t frames = 0, worst_frame = 0;
        double worst_mae = 0;
        for (;;) {
            baseline.read(reinterpret_cast<char *>(a.data()), a.size());
            candidate.read(reinterpret_cast<char *>(b.data()), b.size());
            if (!baseline.gcount() && !candidate.gcount()) break;
            if (baseline.gcount() != static_cast<std::streamsize>(a.size()) ||
                candidate.gcount() != static_cast<std::streamsize>(b.size())) {
                throw std::runtime_error("mask streams differ in length or contain a partial frame");
            }
            uint64_t frame_error = 0;
            for (size_t i = 0; i < a.size(); i += 4) {
                if (a[i] != a[i+1] || a[i] != a[i+2] || a[i+3] != 255 ||
                    b[i] != b[i+1] || b[i] != b[i+2] || b[i+3] != 255) {
                    throw std::runtime_error("expected opaque grayscale BGRA masks");
                }
                const auto error = static_cast<unsigned>(std::abs(int(a[i]) - int(b[i])));
                ++histogram[error];
                frame_error += error;
                squared_error += error * error;
                if ((a[i] > 0 && a[i] < 255) || (b[i] > 0 && b[i] < 255)) {
                    ++edge_pixels;
                    edge_error += error;
                }
            }
            const double mae = double(frame_error) / count;
            if (mae > worst_mae) { worst_mae = mae; worst_frame = frames; }
            ++frames;
            pixels += count;
            absolute_error += frame_error;
        }
        if (!frames) throw std::runtime_error("mask streams are empty");
        size_t maximum = 255;
        while (maximum && !histogram[maximum]) --maximum;
        std::printf("{\n  \"frames\": %zu,\n  \"pixels\": %llu,\n  \"identical_percent\": %.8f,\n"
            "  \"mean_absolute_error_255\": %.8f,\n  \"max_absolute_error_255\": %zu,\n"
            "  \"edge_mean_absolute_error_255\": %.8f,\n  \"worst_frame_index\": %zu,\n"
            "  \"worst_frame_mean_absolute_error_255\": %.8f,\n  \"psnr_db\": ",
            frames, static_cast<unsigned long long>(pixels), 100.0 * histogram[0] / pixels,
            double(absolute_error) / pixels, maximum, edge_pixels ? double(edge_error) / edge_pixels : 0,
            worst_frame, worst_mae);
        if (squared_error) std::printf("%.6f", 10 * std::log10(255.0 * 255 * pixels / squared_error));
        else std::printf("null"); // Infinite PSNR for bit-identical streams.
        std::printf(",\n  \"absolute_error_histogram\": {");
        bool first = true;
        for (size_t i = 0; i <= maximum; ++i) if (histogram[i]) {
            std::printf("%s\"%zu\": %llu", first ? "" : ", ", i, static_cast<unsigned long long>(histogram[i]));
            first = false;
        }
        std::puts("}\n}");
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
