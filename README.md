A small CLI utility to run RVM on a local video: https://github.com/PeterL1n/RobustVideoMatting

## Native macOS Metal backend

```sh
cargo build --release --offline
target/release/segmenter --input input.mov --output mask.mov \
  --model-path rvm_mobilenetv3_fp16.rvmmetal
```

The native Metal path defaults to pipelined decoding, inference, and encoding.
CNN operations run in temporal batches of up to eight frames. Every GRU update
still runs in frame order, and partial batches carry forward the state of their
last real frame. Frame count, timestamps, resolution, downsample ratio, and model
weights are preserved. ONNX's inference implementation is unchanged.

The graph applies the algebraic identity `W[x,h] = Wx*x + Wh*h`: frame-dependent
convolutions run as a batch while state-dependent convolutions remain sequential.
The final guided refinement is fused into one Metal kernel that interpolates the
learned coefficients, applies them to the original image, clamps alpha, and writes
BGRA. FP16 Winograd intermediates are enabled for FP16 models on macOS 26+.
These changes permit small floating-point differences rather than pixel identity.

Decoded CoreVideo buffers are read directly as Metal textures. Immutable output
leases avoid copying completed masks and remain valid across runtime teardown.
Uploads, inference, state updates, and mask conversion share a command buffer per
batch. Packed recurrent outputs reduce graph bindings; a GPU copy selects the
last valid state. CPU staging is allocated only when native textures are unavailable.
Rotated inputs retain the existing rotation path.

Batch size decreases automatically for larger images or higher downsample ratios
to bound working memory. `SEGMENTER_METAL_BATCH=1` selects one frame; explicit
values 1–16 override the automatic choice. Synchronous callers (`--pipeline=false`)
select a one-frame graph automatically unless a batch size was explicitly set.

On macOS 14+, compiled graph packages are cached in a `segmenter-metal-cache`
subdirectory of the user's system temporary directory. Keys include the model
hash, GPU, OS, dimensions, ratio, batch size, and graph options. Cache failures
fall back to compilation. The first encounter with a model/shape includes
compilation and can take seconds; the warmed measurements below exclude that
one-time cost. `SEGMENTER_METAL_NO_CACHE=1` disables this cache.

AVFoundation uses pooled buffers and offline encoding. Native grayscale masks
use IOSurface-backed NV12 buffers. A SIMD integer transform replaces the grayscale
lookup when all 256 entries match macOS's vImage Rec.709 conversion exactly;
the original lookup remains the fallback. Encoder initialization overlaps the
first inference batch, using the engine's guaranteed output dimensions and timing.
The writer preserves the source timebase: `clip-b.mp4` has adjacent timestamps only
1/15360 s apart, which the previous default 1/600 s clock rounded.
No dependencies were installed or added.

Native Metal now defaults to a faster **standard H.264 High, 8-bit 4:2:0**
preset: no B-frames and constant quality 0.83. It uses Apple's built-in
VideoToolbox quality property through AVFoundation; no packages or bundled
codecs are added. This is not the High 4:4:4 lossless mode. `--h264-fast=false`
restores the original compression settings. ONNX and Windows encoding defaults
are unchanged; explicitly enabling this flag with ONNX is rejected.

Use an `.mp4` output for Windows interchange. Microsoft's Media Foundation
decoder documents Baseline/Main/High support up to level 5.1 and 4:2:0 chroma:
https://learn.microsoft.com/en-us/windows/win32/medfound/h-264-video-decoder.
The test exports use High levels 4.2 and 3.0. Actual Windows playback has not
been tested here. If AVFoundation rejects the optional preset properties,
initialization falls back to the original settings and reports that fallback.

The fast preset passes `AVAssetTrack.nominalFrameRate` through
`AVVideoExpectedSourceFrameRateKey` before compression starts. Apple recommends
this hint for automatic H.264 level selection, especially above 30 fps:
https://developer.apple.com/documentation/avfoundation/avvideoexpectedsourceframeratekey.
For 1080p60 this corrects the declared level from 4.0 to 4.2, with identical
encoded picture slices, decoded pixels, and file size on the test clip.
The hint does not retime frames; fractional rates and original VFR timestamps
are preserved. Missing or invalid rates leave the hint unspecified. Legacy
compression (`--h264-fast=false`) does not receive the hint.

General BGRA and odd dimensions retain their conversion path. AVFoundation's
existing odd-size H.264 path rounds dimensions down to even; use even dimensions
when exact encoded dimensions are required.

Set `SEGMENTER_METAL_PROFILE=1` for stage timings; overlapping stage durations
are not additive wall-clock times. For numerical comparisons, the individual
rewrites can be disabled with `SEGMENTER_METAL_FULL_GRU=1`,
`SEGMENTER_METAL_GRAPH_REFINER=1`, and `SEGMENTER_METAL_PRECISE_MATH=1`.

## Measured performance and quality

Apple M5 Max, macOS 26.4, release builds, unchanged FP16 weights and input
sizes, downsample ratio 0.25. Clip A is 1080p60 with 337 frames; clip B is
640×480 VFR (about 30 fps) with 152 frames. Clip labels are anonymized; source
media and local benchmark artifacts are not included. Five alternating trials
after a warmup; medians:

| Input | Original RVM | Current RVM | Inference speedup |
| --- | ---: | ---: | ---: |
| `clip-a.mov`, 1920×1080 | 399.4 fps | 1061.6 fps | 2.66× |
| `clip-b.mp4`, 640×480 | 646.7 fps | 3365.2 fps | 5.20× |

This measures sustained inference, including input normalization and mask
conversion. Each clip is predecoded and repeated five times in one engine
(1685 / 760 frames); the first ten output frames are excluded. Decoding, encoding,
hashing, and initial compilation are outside this throughput measurement.
Repeating the clips avoids overstating asynchronous throughput on short clips.

Whole-video wall times include process startup, decoding, inference, H.264
encoding, and finalization, with shader and graph caches warmed. The final
H.264 preset was compared with the preceding optimized native Metal pipeline
using original H.264 settings on November 4, 2026. Trials alternated after a
warmup: five per variant for `clip-a.mov`, nine for the noisier short `clip-b.mp4`.

| Input | Frames | Previous optimized | Current | Additional throughput | File-size change |
| --- | ---: | ---: | ---: | ---: | ---: |
| `clip-a.mov` | 337 | 0.941 s | 0.812 s | 1.158× | −0.19% |
| `clip-b.mp4` | 152 | 0.299 s | 0.269 s | 1.112× | +0.79% |

The same quality setting works for both test clips without a separate bitrate.
MOV sizes are 2,263,153 and 532,157 bytes, compared with 2,267,512 and 527,977
bytes previously. Both are within the 5% size-variation target. This is a
constant-quality preset; its size relationship to the old preset depends on
video content and the OS encoder. It does not enforce a universal byte limit.

Compression fidelity against the uncompressed masks improved on the measured
clips: luma PSNR 62.27 → 63.10 dB / 57.48 → 58.14 dB, and mean edge error
1.279 → 1.275 / 1.400 → 1.343 in 8-bit video-range units. Maximum errors were
39 → 35 / 56 → 56. Encoded pixels differ; the inference outputs and model
settings are unchanged. Frame count, dimensions, neutral chroma, and every
original timestamp were verified for all 489 frames.
Inspection of enlarged edge composites from the three frames with the largest
compression differences in each clip found no visible quality regression.

The expected-frame-rate hint was measured separately against that fast preset,
using nine alternating warmed whole-export trials per clip:

| Input | No FPS hint | Source FPS hint | Throughput change |
| --- | ---: | ---: | ---: |
| `clip-a.mov` | 0.80449 s | 0.80600 s | −0.19% |
| `clip-b.mp4` | 0.26376 s | 0.26331 s | +0.17% |

This is no measurable speed improvement. Seven alternating isolated encoder
trials similarly measured 696.698 → 696.698 ms for `clip-a.mov` and
150.471 → 151.239 ms for `clip-b.mp4`. The hint is retained for correct H.264 level
selection. All 489 decoded frames, timestamps, and output byte counts remain
unchanged. Applying the hint to legacy default compression increased the
isolated 60 fps file by 32.46%, which is why that mode remains untouched.

The bounded decode/infer/encode pipeline remains. Larger queues, earlier pixel
conversion, model/media startup overlap, zero-copy BGRA wrapping, readiness
callbacks, entropy-mode changes, and scheduling controls did not deliver a
useful additional improvement. Two parallel H.264 sessions improved an isolated
long test by 21%, but a complete segmented pipeline lost that advantage to
buffering, startup, and joining costs. Its implementation was removed. The
bitrate-controlled fast alternative also had worse faint-edge fidelity than
constant quality 0.83 and was removed.

A separate five-trial comparison against the original unoptimized binary
measured 1.582 → 0.844 s (1.87×) for `clip-a.mov` and 0.562 → 0.283 s (1.98×)
for `clip-b.mp4`. These are end-to-end results from a separate measurement window.
**The minimum 3× whole-export target on both clips remains unmet.**

All 489 raw masks and their original timestamps were checked against the original
backend. Maximum alpha difference was 5/255 on either clip. Identical pixels were
99.87189% (`clip-a.mov`) and 99.67963% (`clip-b.mp4`); mean absolute differences in
0–255 units were 0.001284 and 0.003216. PSNR was 77.03 dB and 73.02 dB. Visual
inspection of the worst mean-error frames found no visible matting change.
With `--h264-fast=false`, exports retain the preceding optimized implementation's
file sizes and decoded pixels. H.264 picture data matches after excluding
per-run SEI metadata. The original source timestamp timescale remains intact,
including `clip-b.mp4` frames separated by just 1/15360 s.

## Reproducing the checks

Save the original release binary before making changes, then run:

```sh
python3 scripts/benchmark_macos.py --baseline path/to/original-segmenter \
  --model rvm_mobilenetv3_fp16.rvmmetal --input path/to/clip-a.mov path/to/clip-b.mp4
cargo test --offline
cargo test --offline queued_masks_match_synchronous -- --ignored --nocapture
```

Use `--baseline-pipeline true` when the saved baseline already defaults to
pipelined Metal inference. The default `false` reproduces the original implementation. To compare the two
H.264 modes using the current binary, pass `--baseline-pipeline true` and
`--baseline-arg=--h264-fast=false`; the candidate uses the fast Metal default.

The last test needs GPU access and a local model; override the model path with
`SEGMENTER_TEST_MODEL`. It covers padded rows, odd dimensions, resolution changes,
recurrent-state ordering, draining, and teardown with outstanding GPU work.
Ordinary tests also cover short clips and decoder, inference, drain, and encoder
failures. Strict Clippy still reports the existing generated `FILE` naming and
AVFoundation `Arc`-with-non-`Send` warnings; other Clippy checks pass.

The SIMD conversion regression compares all grayscale byte values, padded rows,
and vector tails with the system's full vImage conversion:

```sh
clang++ -O3 -std=c++17 -Wno-deprecated-declarations scripts/test_macos_encoder.mm \
  -framework Accelerate -framework AVFoundation -framework CoreMedia \
  -framework CoreVideo -framework Foundation -framework VideoToolbox \
  -o target/rvm-bench/test-macos-encoder
target/rvm-bench/test-macos-encoder
```

Native H.264 regressions cover MOV/MP4 decoding, sub-millisecond frame
timestamps, padded input, odd-size fallback, general BGRA, invalid storage, and
cancellation after early preparation. Both H.264 presets are checked for standard
Baseline/Main/High SPS profiles, excluding High 4:4:4 Predictive. Checks also
cover fractional FPS hints and correct level selection for 1080p60:

```sh
clang++ -O3 -std=c++17 -Wno-deprecated-declarations scripts/test_macos_h264.mm \
  -framework Accelerate -framework AVFoundation -framework CoreMedia \
  -framework CoreVideo -framework Foundation -framework VideoToolbox \
  -o target/rvm-bench/test-macos-h264
target/rvm-bench/test-macos-h264
```

To isolate encoder throughput, alternate warmed runs with unique output paths.
After FPS, select the original (`0`) or faster (`1`) H.264 settings. The final
optional argument supplies an expected-FPS hint for the fast preset; omit it
or use `0` to measure without the hint:

```sh
clang++ -O3 -std=c++17 -Wno-deprecated-declarations scripts/benchmark_encoder.mm \
  src/encoder/avfoundation/native.mm -framework Accelerate -framework AVFoundation \
  -framework CoreMedia -framework CoreVideo -framework Foundation -framework VideoToolbox \
  -o target/rvm-bench/bench-encoder
target/rvm-bench/bench-encoder masks.bgra 1920 1080 output.mov 60 1 60
```

To check a mask export against raw masks and their per-frame metadata, using only
system frameworks:

```sh
clang++ -O3 -std=c++17 -Wno-deprecated-declarations scripts/check_macos_mask_export.mm \
  -framework Accelerate -framework AVFoundation -framework CoreMedia \
  -framework CoreVideo -framework Foundation -framework VideoToolbox \
  -o target/rvm-bench/check-mask-export
target/rvm-bench/check-mask-export masks.bgra output.mov masks.sha256
```

For uncompressed per-frame SHA-256 checks, build the native harness using the
installed Xcode tools and system frameworks:

```sh
mkdir -p target/rvm-bench
clang++ -O3 -std=c++17 -Wno-deprecated-declarations \
  scripts/benchmark_metal.mm src/decoder/avfoundation/native.mm \
  src/engine/metal/native.mm \
  -framework Foundation -framework AVFoundation -framework CoreVideo \
  -framework CoreMedia -framework Metal -framework MetalPerformanceShaders \
  -framework MetalPerformanceShadersGraph -o target/rvm-bench/bench-current
SEGMENTER_BENCH_NATIVE=1 SEGMENTER_BENCH_ASYNC=1 target/rvm-bench/bench-current \
  rvm_mobilenetv3_fp16.rvmmetal input.mov target/rvm-bench/current.sha256
```

Compile the same harness with `-DBENCH_SYNC` and the original Metal source to
generate reference hashes and raw masks. `cmp` checks pixel identity; the alpha
rewrites permit small rounding differences, so use `compare_masks` below
for a numerical comparison. The harness
requires an unrotated clip with at least eleven frames. An optional fourth
argument writes raw BGRA masks. For timing runs, use `SEGMENTER_BENCH_NO_HASH=1`
and no raw output; compare `steady_fps`, not the sum of asynchronous API-call
durations. The whole-video Python benchmark uses only the standard library and writes its
logs, output videos, and `results.json` under `target/rvm-bench/final` by default.

For the sustained inference numbers above, build both native harnesses and run:

```sh
python3 scripts/benchmark_metal.py \
  --baseline path/to/bench-original --candidate target/rvm-bench/bench-current \
  --model rvm_mobilenetv3_fp16.rvmmetal \
  --input path/to/clip-a.mov path/to/clip-b.mp4 \
  --output-dir target/rvm-bench/inference --repeats 5 --loops 5
```

This standard-library driver enables `SEGMENTER_BENCH_PREDECODE=1`, disables
hashing, and repeats each predecoded clip five times without resetting recurrence.
It alternates the original synchronous path with the candidate's native queued
path, warms each once, and saves per-trial logs plus `inference.json`.

The shared-buffer GPU regression covers padded and odd dimensions, texture
uploads, queue ordering, partial-batch draining and resumed recurrence,
resolution changes, masks surviving engine destruction,
and release from another thread:

```sh
clang++ -O3 -std=c++17 -Wno-deprecated-declarations scripts/test_metal_buffers.mm \
  src/engine/metal/native.mm -framework Foundation -framework CoreVideo \
  -framework Metal -framework MetalPerformanceShaders \
  -framework MetalPerformanceShadersGraph -o target/rvm-bench/test-metal-buffers
MTL_DEBUG_LAYER=1 target/rvm-bench/test-metal-buffers rvm_mobilenetv3_fp16.rvmmetal
clang++ -O3 -std=c++17 scripts/compare_masks.cpp -o target/rvm-bench/compare-masks
target/rvm-bench/compare-masks baseline.bgra candidate.bgra 1920 1080
```

The comparison reports every mask pixel, including an error histogram, edge
error, PSNR, and the worst frame. The default build has 54 passing CPU tests and
one separately run GPU test. Clippy passes with the two existing exemptions
listed above. An offline all-features check could not complete because optional
ONNX dependencies were not unpacked in the local Cargo cache; no additional
dependencies were fetched or installed.

Anonymized numeric results and validation notes are available in
[docs/macos-rvm-benchmarks.json](docs/macos-rvm-benchmarks.json). The published
report contains aggregate metrics only; local paths, media, masks, profiles,
model weights, raw frame hashes, and machine-specific logs are excluded.
