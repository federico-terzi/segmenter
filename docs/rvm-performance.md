# RVM / DirectML performance investigation

**Update, 2026-11-04: a DirectX/DirectML research prototype exceeds 5x throughput
on both supplied clips on the NVIDIA GPU, at the unchanged CLI default ratio
of 0.25.** It uses the original weights and evaluates every frame. The harder
clip takes about 0.90 ms per frame versus 4.73 ms originally; the easier
clip takes 0.83 ms versus 4.59 ms. Both complete supplied clips pass the numerical quality gate, with no
visible difference in the inspected comparison frames.

This result has specific limits: it batches four encoder frames, measures
completed inference throughput, and is currently an opt-in research test rather
than the default application engine. The passing AMD configuration reaches about
4.7x; ratio-1 inference has not reached 5x. The default engine retains the earlier
2–3x NVIDIA optimization. Neither arbitrary footage, other GPUs nor full video
export speed has been qualified. These are Windows measurements, not a macOS
performance claim.

The evaluation videos are private and are identified here as **Clip A** (harder) and **Clip B** (easier). Only aggregate measurements and source code are published. Video, previews, masks, activations, model weights and local diagnostic reports are excluded. Example input paths are placeholders; generated artifacts remain local under `target/`.

## Fivefold throughput prototype (2026-11-04)

### Matched measurements and quality

All comparisons in this section use the same original FP16 model, GPU, decoded
frames, maximum image dimension of 480, and **ratio 0.25 on both sides**. The CLI
already defaults to 0.25. This is not a comparison of the new quarter-ratio path
against the slower ratio-1 baseline used in much of the earlier investigation.
Lowering the ratio relative to a requested ratio of 1 was rejected for quality
and remains rejected.

| GPU / clip | Original inference | Accepted prototype | Speedup | Encoder batch |
|---|---:|---:|---:|---:|
| RTX 5060 Laptop / Clip A, 480x360, complete clip | 4.733 ms | 0.8962 ms | 5.28x | 4 |
| RTX 5060 Laptop / Clip B, 480x270, complete clip | 4.586 ms | 0.8314 ms | 5.52x | 4 |
| Radeon 780M / Clip A, 480x360, first 150 frames | 7.166 ms | 1.5219 ms | 4.71x | 1, fused network |

The final NVIDIA runs use every decoded frame: Clip A has 152 frames / 5.1 seconds,
and the easier clip has 222 frames / 7.4 seconds. Each run repeats its complete
sequence ten times, discards two complete sequences as warmup and measures the
remaining eight: 1,216 Clip A frames or 1,776 easy frames. Both baseline and candidate
reset recurrence at sequence boundaries; baseline uses the new benchmark option
`SEGMENTER_BENCH_RESET_SEQUENCE=1`. No model state is reset within a clip.
Repeating improves timing stability, not quality coverage: there are **152 and
222 distinct evaluated frames**, not thousands of independent examples.
The original path is selected with `SEGMENTER_DISABLE_SPECIALIZATION=1`.
No ORT profiler is enabled during timing.

Earlier 150-frame timing controls gave 4.570–4.630 ms versus 0.8904–0.9030 ms on
Clip A, and 4.648 ms versus 0.8035 ms on the easier clip. Those original baseline
runs carried state across repeats, whereas the prototype reset at repeats.
The final complete-clip comparison above corrects that history mismatch. The
AMD result still uses 150-frame sequences and is labeled separately;
it is not used to claim fivefold performance.

A longer final Clip A repetition check processed 4,560 frames (4,256 measured)
and gave **4.609 ms original versus 0.8814 ms prototype, or 5.23x**. Quality
metrics were unchanged. There was also a slower 1.0321 ms prototype run shortly
after AMD testing, followed by a 4.599 ms baseline: about 4.46x. Its cause was
not isolated. The slower result is retained here rather than excluded from the
record; these experiments demonstrate fivefold performance in multiple runs,
but do not establish a fivefold minimum across laptop power states or contention.

The prototype timer includes BGRA CPU upload, GPU normalization, encoder,
sequential recurrent decoder, custom refinement, mask quantization, the GPU
completion fence, copying every final BGRA mask to CPU memory, and checking for
nonfinite output. Quality-only extraction of alpha bytes and preview rendering
are outside timing, as in the original benchmark. Decode, encode, session/shader
creation and initial graph capture are excluded from both steady-state results.
Partial final encoder batches are padded with the last frame; their compute is
included, but padding never advances recurrence or produces an output frame.

Four-frame batch p95 completion times were 3.84 ms on Clip A and 3.67 ms on
the easier clip. The 0.90 ms figure is amortized completed throughput, **not
single-frame response latency**. Live capture would also wait for up to three
additional frames: 100 ms at 30 fps or 50 ms at 60 fps before computation.
Offline export can normally supply these frames immediately after decoding.

Errors below are alpha byte levels (0–255), compared with the original inference
on the same GPU. The acceptance gate is mean absolute error <= 0.05, maximum
error <= 8, and fraction of pixels differing by more than 4 <= 0.0001. This gate
was also used to reject faster candidates; it is not a proof of perceptual
equivalence on unseen footage.

| Accepted configuration | Mean error | Max error | Fraction > 4 | Boundary mean error | Temporal-change mean error |
|---|---:|---:|---:|---:|---:|
| NVIDIA Clip A, batch 4, all 152 frames | 0.008194 | 6 | 0.000000381 | 0.108856 | 0.011902 |
| NVIDIA easy, batch 4, all 222 frames | 0.003808 | 5 | 0.000000070 | 0.059612 | 0.006841 |
| AMD Clip A, batch 1 | 0.010094 | 8 | 0.000013272 | 0.133818 | 0.014113 |

Only 10 of 26,265,600 Clip A pixels differ by more than four levels in the accepted
NVIDIA run. The easier clip has 28,771,200 evaluated pixels; two differ by more
than four levels. Boundary and temporal
metrics come from `scripts/rvm_alpha_quality.py`; temporal-change maxima are
6, 7 and 12 levels respectively. These measure changes relative to the original
model, not ground-truth segmentation accuracy.

Inspected the comparison frames with the largest summed alpha error: Clip A frame
17 and easy frame 78, and the final frames of both complete clips. Their original and candidate composites have no visible
difference at the supplied preview scale. This is a still-frame inspection plus
sequence-wide numerical analysis, not a blinded playback study. Previews are in
`target/rvm-temporal-cam-full-previews` and
`target/rvm-temporal-easy-full-previews`; columns show source, original,
candidate and 16x absolute difference. Raw masks are
`target/rvm-temporal-{cam,easy}-quarter-full.alpha` and
`target/rvm-amd-temporal1-cam-quarter.alpha`.

### How much each optimization contributes

A follow-up for contribution attribution used the complete Clip A clip at 480x360,
ratio 0.25, on NVIDIA: 30 repeats, 304 warmup frames and 4,256 measured frames.
Original inference measured **4.682 ms/frame**; the production specialization
bundle measured **1.755 ms/frame**. The final prototype controls ranged from
**0.8975 to 1.0649 ms/frame** during the ablation pass.

The specialization bundle therefore removed about **2.927 ms/frame**. Later
changes collectively removed another **0.690–0.858 ms/frame**. As an approximate
allocation of the total observed time saved, that is **77–81% from specialization
and 19–23% from the later work**. This is a milestone comparison across two
harnesses, not a causal decomposition of every individual optimization. Fixed
shapes, a constant ratio and dead foreground-output removal were applied as a
bundle; their separate contributions have not been measured.

For the final prototype, individual switches were disabled while retaining the
other optimizations. Runs were sequential, in the order below. Every candidate
passed the same alpha-quality gate. Raw outputs are saved locally in
`target/rvm-attribution-ablation.json` and `target/rvm-attribution-baseline.json`.

| Variant | Completed ms/frame | Interpretation |
|---|---:|---|
| All enabled, initial control | 0.8975 | Control |
| Encoder batch 1 instead of 4 | 1.2821 | Clear batching benefit; initial-control comparison gives about 30% less frame time with batching |
| Eight separate feature outputs instead of packed output | 0.9210 | Small observed difference, insufficient to isolate from drift |
| All enabled, middle control | 1.0168 | Unchanged code became slower |
| Copy recurrent states instead of alternating buffers | 1.0297 | Individual benefit unresolved at this variability |
| Disable DirectML graph capture | 1.0415 | Individual benefit unresolved at this variability |
| Portable FP16 rounding instead of NVIDIA-qualified native-half shaders | 1.0768 | Individual benefit unresolved at this variability; both pass quality |
| All enabled, final control | 1.0649 | Control drift is about 19% relative to the initial control |

The batching comparison remains slower than every fully enabled control. Its
apparent benefit is roughly 17–30% less complete-frame time across the control
range; this range is descriptive, not a statistical confidence interval. The
separate encoder-only measurement, 0.5889 to 0.2006 ms/frame, is a 66% reduction
for that stage and must not be advertised as a 66% whole-frame saving.

The smaller differences cannot responsibly be assigned precise percentages
from this pass. In particular, feature packing's earlier 1.1693-to-about-1.01 ms
milestone should not be interpreted as a proven 14% independent contribution in
the final implementation. Earlier bulk-copy, uninitialized-readback and fast
finite-validation work collectively changed about 1.01 to 0.93 ms (about 8%);
those three changes were not isolated individually. The small final changes are
also interdependent with batching and buffer layout. Their percentages cannot
be added or multiplied to reconstruct the total gain.

The custom refiner has a separate, better-isolated ratio-1 result: its stage was
2.37x faster on NVIDIA and 9.60x on AMD, but the longer complete-inference
comparisons improved only **5.2% and 14.4%**, respectively. This is useful but
is not the principal cause of the overall fivefold result. Its independent
contribution at ratio 0.25 has not been cleanly separated from GPU pixel work
and output packing.

No accepted gain came from the tested NHWC rewrites, generic or custom Winograd
convolutions, Vulkan backend replacement or weight-only low-rank approximations.
Larger encoder batches failed quality, and no student was trained. Explicit
FP16 rounding and the AMD alpha-Gather rewrite primarily enabled correct or
compatible execution; they should not be assigned invented speedup percentages.

### What produces the improvement

The successful design combines DirectML's convolutions with custom HLSL around
them. Replacing all convolutions with hand-written shaders was slower in the
earlier experiments below.

1. **Batch only the independent encoder.** The encoder before `Split_172` has
   no recurrent inputs. Evaluate four successive frames together, then run the
   recurrent decoder once per real frame in its original order. Every GRU sees
   the preceding frame's updated state. No frame or recurrent update is skipped.
   The unchanged weights remain FP16. Different batch sizes can select different
   floating-point kernels, so equivalence is checked rather than assumed.
2. **Keep features in one GPU buffer.** The encoder packs its eight outputs into
   one contiguous row per frame. Each decoder input takes one row via one GPU
   copy, followed by graph slices/views. Eight separate feature copies and their
   barriers cost enough to miss the target. The low-resolution RGB feature is
   first in the packed row, allowing the refiner to use that same allocation.
3. **Keep stable allocations and alternate recurrent buffers.** One D3D12 device
   and queue are shared with DirectML. Two fixed bindings alternate state A to B
   and B to A, avoiding recurrent copies. Input and output states do not alias
   within one inference. Two graph-capture IDs preserve the two binding layouts.
   Capture is enabled before the execution provider is registered. The run-option
   key is named `gpu_graph_id` in ORT's common implementation; this uses DirectML,
   with no CUDA provider or CUDA inference.
4. **Fuse the surrounding pixel work.** Upload packed BGRA directly. A shader
   normalizes it into FP16 RGB. Two refinement passes compute the original
   low-resolution coefficients and source-guided full-resolution alpha, then
   quantize directly to the final packed BGRA mask. The quarter-ratio resize
   inside the encoder stays in DirectML. Only the final mask returns to the CPU.
5. **Remove avoidable CPU and submission work.** Use a persistent upload ring,
   bulk copies for contiguous frames, one completion fence per batch, and mask
   buffers initialized by readback rather than zero-filled first. An AVX2 check
   preserves nonfinite-output validation, including tail pixels; a scalar
   fallback covers other CPUs. Refinement and its readback share a command list.

The encoder-only quarter-ratio measurement changed from 0.5889 ms for one frame
to 0.8023 ms for four frames, about 0.2006 ms per frame. The decoder still runs
sequentially. In the complete pipeline, separate feature copies took 1.1693 ms
per frame; packing reduced that to about 1.01 ms. Bulk transfers and the output
guard brought it to about 0.93 ms, and alternating recurrent buffers brought
the final sustained measurements to about 0.90 ms. These intermediate figures
describe experiments, not additive isolated speedup factors.

Implementation is in `dx12_resident_probe.rs`, `dx12_probe.cpp`,
`preprocess_quarter_probe.hlsl`, `refiner_probe.hlsl`,
`refiner_upsample_probe.hlsl` and `fp16_round.hlsli` under
`src/engine/onnx/`. The two `scripts/export_rvm_{quarter,temporal}_probe.py`
exporters derive graphs and refiner weights from the original model. They are
specific to this RVM graph and are not general ONNX converters. The code is
behind `research-dx12` and ignored tests, so ordinary CLI inference does not
select it accidentally. No model was distilled, and no training data was used.

### Rounding and hardware differences were decisive

The first GPU normalization implementation was fast but failed quality, with a
maximum alpha difference of 69. On the first Clip A frame, 254,761 of 518,400
normalized channels differed from CPU preprocessing by one half-precision step.
The legacy HLSL conversion lowered values rather than matching CPU nearest-even
rounding on the tested devices. This small input discrepancy propagated through
the recurrent network; it could not be dismissed as harmless FP16 noise.

`fp16_round.hlsli` therefore implements IEEE binary16 nearest-even conversion
explicitly, including subnormals, overflow, infinities and NaNs. Its fast normal
case rounds float32 mantissa bits directly. The GPU audit checks 256,004 inputs:
all half bit patterns, float32 values at and adjacent to every finite half
midpoint, both signs, and overflow cases. Both conversion helper paths are
compared with Rust's `half` implementation (NaNs by classification).

The portable SM6.0 implementation passed on both GPUs. An optional SM6.2
`float16_t` shortcut passed the audit and exact normalized-input check on NVIDIA,
but failed the normalized-input check on AMD. **AMD must use the portable
implementation**; native-half rounding must be qualified per adapter/driver,
not inferred from shader model support. Portable rounding is the source default.
The newer shared rounding helper also applies when recompiling the old ratio-1
refiner; the historical ratio-1 measurements below used its earlier bytecode.

Other faster configurations were rejected:

| Experiment | Result / reason rejected |
|---|---|
| NVIDIA encoder batch 8 | Original decoder/refiner quality test: max error 32; fraction > 4 = 0.001280 |
| NVIDIA encoder batch 16 | Original decoder/refiner quality test: max error 27; fraction > 4 = 0.000541 |
| AMD complete batch 4 | 1.1990 ms, but max error 27; fraction > 4 = 0.000859 |
| AMD complete batch 2 | 1.3656 ms, but max error 23; fraction > 4 = 0.000712 |
| AMD fused GPU quarter resize | Max error 9, beyond the gate; keep the original DirectML resize |
| Queuing eight frames without encoder batching | About 1.24 ms on NVIDIA; no useful gain over the single-frame network path |

AMD split-graph batch 1 passes at 1.5831 ms against a sustained 7.166 ms baseline.
A final variant keeps the encoder and decoder in one DirectML graph while using
the new upload/readback path and alternating state buffers. It passes with the
same alpha metrics at **1.5219 ms**, about **4.71x**. Its fivefold threshold is
1.4332 ms, so it still needs about 5.8% less inference time. Switching this variant
to a compute queue was slower (1.6786 ms). Reporting the rejected 1.1990 ms batched
result as an AMD success would hide a quality regression. The fused candidate's
mask is `target/rvm-amd-fused-cam-quarter.alpha`.

### Reproduction and remaining integration

Use the existing pinned ONNX/Python environment and Windows SDK DXC. The
successful path adds no new dependency. The ncnn SDK described later was used
only in a rejected alternative. All generated models, bytecode, masks and SDKs
stay under ignored `target/`; keep original model weights untouched.

Generate a static ratio-0.25 model with `resident_network_benchmark`, using
`SEGMENTER_BENCH_RATIO=0.25`, width 480, height 360 for Clip A or 270 for the easy
clip, and `SEGMENTER_BENCH_EXPORT_STATIC`. Unset
`SEGMENTER_DISABLE_SPECIALIZATION` for this export. Then use `graph_benchmark`
with `SEGMENTER_BENCH_GRAPH` set to that file and `SEGMENTER_BENCH_INFER_SHAPES`
set to a destination file to obtain CPU-propagated metadata. The already
generated Clip A files are `target/rvm-quarter-static.onnx` and
`target/rvm-quarter-shapes.onnx`.

```powershell
python scripts/export_rvm_quarter_probe.py target/rvm-quarter-static.onnx target/rvm-quarter-shapes.onnx target/rvm-dx12-quarter
python scripts/export_rvm_temporal_probe.py target/rvm-quarter-static.onnx target/rvm-quarter-shapes.onnx target/rvm-quarter-temporal
$rvmDxc = 'C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/dxc.exe'
# NVIDIA-qualified native-half build; use SM6.0 without the two native-half flags on AMD.
& $rvmDxc -T cs_6_2 -enable-16bit-types -E main -O3 -D NATIVE_HALF_RNE=1 -D PREPROCESS_SCALE=1 -Fo target/rvm-dx12-quarter/preprocess-source.cso src/engine/onnx/preprocess_quarter_probe.hlsl
& $rvmDxc -T cs_6_2 -enable-16bit-types -E main -O3 -D NATIVE_HALF_RNE=1 -D REFINER_SCALE=4 -D OUTPUT_COEFFICIENTS=1 -Fo target/rvm-dx12-quarter/refiner-shared.cso src/engine/onnx/refiner_probe.hlsl
& $rvmDxc -T cs_6_2 -enable-16bit-types -E main -O3 -D NATIVE_HALF_RNE=1 -D PACKED_MASK=1 -D BGRA_SOURCE=1 -Fo target/rvm-dx12-quarter/refiner-bgra.cso src/engine/onnx/refiner_upsample_probe.hlsl

$env:SEGMENTER_SHADER_DIR = 'target/rvm-dx12-quarter'
$env:SEGMENTER_TEMPORAL_DIR = 'target/rvm-quarter-temporal'
$env:SEGMENTER_BENCH_VIDEO = 'C:/path/to/clip-a.mp4'
$env:SEGMENTER_BENCH_REFERENCE = 'target/rvm-baseline-cam-quarter.alpha'
$env:SEGMENTER_BENCH_SAVE = 'target/rvm-temporal-packed-quarter-cam.alpha'
$env:SEGMENTER_DX12_CAPTURE = '1'
$env:SEGMENTER_TEMPORAL_PACK = '1'
$env:SEGMENTER_TEMPORAL_PINGPONG = '1'
$env:SEGMENTER_DX12_BATCH = '4'
$env:SEGMENTER_HYBRID_REPEATS = '10'
$env:SEGMENTER_BENCH_WARMUP = '300'
cargo test --offline --release --features research-dx12 dx12_temporal_video -- --ignored --nocapture --test-threads=1
python scripts/rvm_alpha_quality.py target/rvm-baseline-cam-quarter.alpha target/rvm-temporal-packed-quarter-cam.alpha 480 360
```

These commands reproduce the original 150-frame probe. For the final complete-
clip runs, set `SEGMENTER_BENCH_FRAMES=2000` in both reference and prototype runs
(the decoder stops at end of file), use the `*-quarter-full.alpha` reference
and candidate names, and set warmup to 304 for Clip A or 444 for the easier clip
when repeating ten times. For a single reference pass, retain warmup 30.
Set `SEGMENTER_BENCH_RESET_SEQUENCE=1` for the repeated original baseline.

Generate the 150-frame reference separately with `inference_benchmark`, original
model, matching video/device, ratio 0.25, specialization disabled, repeats 1,
warmup 30 and `SEGMENTER_BENCH_SAVE` set to the reference path. Unset reference
and preview variables for that run. For sustained baseline timing, use repeats
10 and warmup 300 with save/reference/preview unset. Do not feed a 150-frame
reference into the original 1,500-call benchmark. Run GPU benchmarks sequentially.

For the easy clip use the `rvm-easy-quarter-{static,shapes}.onnx` exports,
`target/rvm-dx12-quarter-easy` and `target/rvm-quarter-temporal-easy` directories,
and its matching video and reference. For AMD set `SEGMENTER_BENCH_DEVICE=1`,
batch 1, `target/rvm-dx12-quarter-amd` (portable bytecode) and the AMD reference.
For the final AMD fused variant additionally set `SEGMENTER_TEMPORAL_FUSED=1`;
it uses `resident-source.onnx` from the shader directory and requires batch 1.
Do not select batch 4 or reuse NVIDIA's native-half bytecode on AMD. Leave
`SEGMENTER_TEMPORAL_FUSED` unset for the NVIDIA batched path.

Before enabling this in the application, extract the harness into a runtime
backend with model/shape validation, device-specific qualification and fallback,
batch-aware frame scheduling, orderly final-batch draining and resource cleanup.
Validate complete decode-to-encode exports and longer sequences when such
footage becomes available. Both supplied videos have now been evaluated in full,
but their 5.1- and 7.4-second durations cannot establish perceptual equivalence for
unseen hair, transparency, scene cuts or long recurrent histories. The ratio-1
target and broader GPU coverage remain open. This prototype establishes a
measured route past fivefold throughput for the existing default configuration;
it does not establish a universal fivefold product speedup.

Final ordinary checks: the default release suite passes 59 tests (5 ignored);
enabling both research features passes 60 tests (17 ignored). The ignored GPU
benchmarks and rounding audits are run explicitly as described above.

## Implementation

### Potential transfer to macOS

No macOS speedup was measured in this investigation. Inspection of the existing
Metal backend shows it already uses fixed shapes, a constant downsample ratio,
persistent input/output buffers, alternating recurrent-state buffers and GPU
pixel conversion. Much of the largest Windows gain therefore has no direct
equivalent left to apply there.

The remaining candidates are batching the independent encoder while preserving
sequential decoding; porting the custom guided refiner to Metal shaders;
processing multiple frames before waiting for GPU completion; and writing FP16
input directly instead of producing FP32 input followed by a graph cast. Keeping
decoder and encoder video surfaces in Metal-compatible buffers could also avoid
the current CPU copies, but would require a broader frame-interface change.
These are hypotheses, not measured Mac improvements. Splitting an MPSGraph may
cost more than batching saves, and both precision and performance need evaluation
on Apple hardware. Apple's shared memory still requires synchronization before
the other processor accesses completed results; see the
[shared-storage documentation](https://developer.apple.com/documentation/metal/mtlstoragemode/shared).

### Detailed initialization and execution

RVM consumes the current RGB image plus four recurrent tensors carrying features
from the preceding frame. Fixing their **dimensions** does not fix their
**contents**: both the image and recurrent values still change on every frame.
The earlier code fixed only source width/height; the new path also fixes batch
size to one and resolves all recurrent dimensions.

Initialization performs these steps once per process/video:

1. Read the supplied ONNX bytes and apply the existing source/identity-resize
   compatibility patches.
2. Make a temporary in-memory graph with fixed source shape, scalar-shaped zero
   recurrent inputs, and the requested ratio embedded as a FLOAT initializer.
3. Construct a CPU session to propagate shapes; do not execute a video frame.
   Read the four recurrent output shapes, then drop this temporary session.
4. Specialize the original prepared graph again using those recurrent shapes.
   Initialize full-sized zero tensors and create the actual DirectML session.
5. Run every frame in order, feeding each recurrent output into the next frame.

The ratio keeps its exact requested value (including the pre-existing tolerance
around 1.0). Making it constant allows resize/shape arithmetic to be evaluated
ahead of inference. The disk model and trainable weights are not rewritten.
The unused `fgr` graph output is removed; computations shared with `pha` remain.
One fused DirectML execution graph does **not** mean one GPU kernel. It means
ORT delegates a compiled partition to DirectML, which may execute many kernels.

This is execution specialization, not a smaller trained model. It preserves
the original temporal recurrence and adds no CUDA inference dependency. Graph
compilation is not cached across separate CLI processes, explaining why short
exports gain less than sustained inference.

The previous graph fixed only the source spatial dimensions. Batch and recurrent
input shapes remained symbolic, and non-unit downsample ratios were runtime
inputs. DirectML dispatched many separate operators.

The Windows ONNX engine now specializes batch, source and recurrent shapes, and
embeds the requested downsample ratio as a constant. A temporary CPU session
resolves the recurrent output dimensions using the original broadcast-zero
initial states; it does not run inference. The actual DirectML session starts
with full-sized zero recurrent tensors. This avoids hardcoding a particular
RVM architecture or resetting recurrence midway through a video.

The unused foreground output is removed so ORT can eliminate its dead branches.
The model's alpha/refinement computations and trained weights remain intact.
Profiling confirmed one fused DirectML graph in steady state. This follows
[DirectML's documented preference for known input shapes](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html).

If specialization or specialized session construction fails, the engine logs the
reason and uses the previous dynamic session path, including its existing
identity-resize retry. Set `SEGMENTER_DISABLE_SPECIALIZATION=1` to explicitly
select the previous path for diagnosis or comparison.

No runtime dependencies were added. CUDA is not used for inference. No student
model was trained. The model file is never rewritten.

### Additional compatibility optimization: alpha extraction

Further testing on the AMD Radeon 780M exposed a compiler failure in the fully
specialized graph. Block extraction isolated it to the final refinement stage.
Replacing the last `Split` with `Slice` did not help. Replacing it with `Gather`
did: the latter selects the same alpha channel with an index tensor of shape
`[1]`, retaining the channel dimension. No weights or arithmetic are changed.
This isolates a problematic graph pattern; the underlying driver/compiler bug
has not been established.

The implementation now retries a failed specialized session with this equivalent
rewrite before falling back to the dynamic graph. It only rewrites a standard
ONNX `Split` with explicit positive split lengths, one live output, and a selected
output of length one. It removes dead consumers, avoids initializer-name
collisions, and leaves graphs with nested subgraphs alone because their implicit
captures need a different liveness analysis. Both packed and unpacked protobuf
split lengths are supported. A successful original specialization retains its
original layout, since an unconditional rewrite can alter kernel selection.

On AMD, the retained retry was verified on both clips at ratio 1:

| Clip | Baseline | Specialized with Gather retry | Speedup | Alpha differences |
|---|---:|---:|---:|---|
| Clip A, 480x360 | 16.320 ms | 11.241 ms | 1.45x | None in 25,920,000 pixels |
| Clip B, 480x270 | 12.765 ms | 8.520 ms | 1.50x | None in 19,440,000 pixels |

The failed initial compilation still adds startup overhead (first segment call
about 2.6–2.8 seconds in these retry runs). This
is a measured AMD improvement, not a claim that the NVIDIA speedup transfers
unchanged to other GPUs.

## Measurements (2026-11-03)

Machine: Windows, NVIDIA GeForce RTX 5060 Laptop GPU, driver 572.97, existing
`ort` 2.0.0-rc.10 and cached DirectML 1.15.4. The model SHA-256 is
`6a0d5ce6cc17702613be548559879b4521ed424cfe14ddc48d1acaa44d616f64`.

Both supplied clips were tested at maximum dimension 480 and downsample ratio
1.0. These are synchronized `Engine::segment` times, including preprocessing,
upload, inference, alpha readback, and mask conversion. Decode and encode are
excluded. Profiling was disabled for timing.

| Workload | Baseline mean | Specialized mean | Speedup |
|---|---:|---:|---:|
| Clip A, 480x360, 1,500 calls / 300 warmup | 7.753 ms | 3.845 ms | 2.02x |
| Clip B, 480x270, 150 frames / 10 warmup | 7.053 ms | 3.062 ms | 2.30x |
| Clip A, same 480x360 at ratio 0.25, 150 frames / 10 warmup | 4.815 ms | 1.655 ms | 2.91x |

The long Clip A run repeats the first 150 decoded frames, retaining recurrent
state across repeats; it is a sustained timing workload, not additional video
coverage. Short-run GPU timings varied (roughly 3.8–4.5 ms for optimized Clip A at
ratio 1), so these results are not universal hardware guarantees.

Full CLI export of all 152 Clip A frames, including initialization, decode, encode,
and finalization: **2.52 s baseline / 2.13 s specialized (1.18x)**. Compilation of
the fused graph adds startup cost, so short clips benefit much less and may
regress depending on compilation time. Do not advertise the inference speedup
as an end-to-end export speedup.

## Quality checks

All 150 evaluated frames of each supplied clip were compared before video
encoding. Errors below use 8-bit alpha levels, not normalized alpha.

| Clip | Pixels compared | Mean absolute error | Maximum error | Fraction differing by >4 |
|---|---:|---:|---:|---:|
| Clip A, ratio 1 | 25,920,000 | 0.002248 | 3 | 0 |
| Clip B, ratio 1 | 19,440,000 | 0.001410 | 5 | 0.000000051 |
| Clip A, ratio 0.25 | 25,920,000 | 0 | 0 | 0 |

An odd-size synthetic case (481x270, ratio 0.37, 30 frames) also produced
identical masks. Recurrent inputs/outputs are asserted not to alias in the
real-model benchmark. Inspected source/composite/difference panels showed no
visible change on the supplied clips. This does not substitute for validation
on diverse hair, motion, transparency, lighting, and hardware.

Additional checks with `scripts/rvm_alpha_quality.py` focus on edges and changes
over time. The boundary region contains partially transparent pixels and adjacent
alpha changes, dilated by two pixels. Temporal error compares the candidate's
frame-to-frame alpha change against the reference's change, without penalizing
motion already present in the reference. All numbers are in 8-bit alpha levels.

| NVIDIA, ratio 1 | Boundary pixels | Boundary MAE | Temporal-change MAE | Maximum temporal-change error |
|---|---:|---:|---:|---:|
| Clip A | 1,487,429 | 0.039167 | 0.004118 | 5 |
| Clip B | 938,172 | 0.029219 | 0.002636 | 6 |

AMD's retained retry is bit-identical on both clips, so these difference metrics
are zero there. Final reruns also passed the ratio-0.25 Clip A and odd-size ratio-0.37
synthetic comparisons exactly. The ordinary Rust suite passes 59 tests, with
five explicit opt-in media/research tests ignored by the default run.

## Reproduce

The ignored Rust test uses the project's existing dependencies. No Python or
downloads are needed. Run the baseline and candidate sequentially on the same
GPU, clip, frame count, dimensions, and ratio:

```powershell
$env:SEGMENTER_BENCH_MODEL = 'rvm_mobilenetv3_fp16.onnx'
$env:SEGMENTER_BENCH_VIDEO = 'C:/path/to/video.mp4'
$env:SEGMENTER_BENCH_DIMENSION = '480'
$env:SEGMENTER_BENCH_RATIO = '1'
$env:SEGMENTER_BENCH_FRAMES = '150'
$env:SEGMENTER_DISABLE_SPECIALIZATION = '1'
$env:SEGMENTER_BENCH_SAVE = 'target/reference.alpha'
cargo test --offline --release inference_benchmark -- --ignored --nocapture
Remove-Item Env:SEGMENTER_DISABLE_SPECIALIZATION
Remove-Item Env:SEGMENTER_BENCH_SAVE
$env:SEGMENTER_BENCH_REFERENCE = 'target/reference.alpha'
$env:SEGMENTER_BENCH_PREVIEWS = 'target/comparisons'
cargo test --offline --release inference_benchmark -- --ignored --nocapture
```

The reference is raw, row-major grayscale, one byte per pixel per frame. Keep
its workload settings with it. The test rejects extra/missing reference data
and fails if alpha MAE exceeds 0.05 levels, maximum error exceeds 8 levels, or
more than 0.01% of pixels differ by over 4 levels. These are regression gates,
not a universal definition of perceptual equivalence.

Optional variables:

- `SEGMENTER_BENCH_REPEATS=10`, `SEGMENTER_BENCH_WARMUP=300`: sustained timing.
  Unset reference/save/preview variables for timing-only runs, or generate a
  matching repeated reference first.
- `SEGMENTER_BENCH_PROFILE=target/profile`: ORT trace. **Do not use for timing
  comparisons**; tracing disproportionately slows the unfused baseline.
- `SEGMENTER_BENCH_DEVICE=1`: select another DirectML adapter in the test only.
- Omit the video variable for a synthetic input sequence.

Preview columns are source, baseline composite, candidate composite, and 16x
absolute alpha difference. Outputs from this investigation are under the ignored
`target/rvm-comparison-cam` and `target/rvm-comparison-easy` directories.

The existing `target/release/DirectML.dll` was version 1.13.1, incompatible with
this cached ORT build: both baseline and candidate failed with it. The validated
CLI and the existing cached 1.15.4 DLL are together in `target/rvm-validated`.
Keep them together when running that local build. The older DLL was not replaced.

## Earlier remaining work toward 5x (ratio 1)

Lowering the ratio from 1 to 0.5 changed Clip A masks by up to 255 levels, with
1.9% of pixels changing by more than 4, and was rejected. Buffer reuse did not
materially improve timing. A command-capture experiment was discarded after
cross-session synchronization problems; its faster but incorrect output is not
included. TensorRT deployment was excluded because inference must not use CUDA.

A plausible next research direction is a narrower recurrent student, trained
on CUDA and exported to ordinary ONNX/DirectML. It would need spatial alpha,
edge, multiscale, and temporal distillation losses, multi-frame training, and
held-out diverse videos. These two evaluation clips alone would not establish
general matting quality. Profile candidate channel widths on DirectML before
training, since fewer FLOPs do not guarantee a 5x latency reduction. No measured
student quality or speedup is claimed here.

## Continued investigation

The next experiments prioritize unchanged mathematical operations and controlled
measurements. No candidate is accepted solely on latency; compare unencoded
alpha masks against the original path, using the same frames, dimensions,
ratio, recurrence history, and GPU. Retain the two supplied clips as evaluation
data if a student-training experiment is introduced.

Open questions:

- How much of synchronized latency is actual network execution, versus source
  upload, alpha readback, pixel conversion, and CPU/GPU synchronization?
- Which network blocks dominate after whole-graph fusion? ORT's CPU operator
  trace is insufficient to establish GPU kernel cost.
- Can graph/layout or algebraic rewrites improve DirectML's selected kernels
  while preserving the original alpha function?
- Can transfers and inference share one D3D12 queue without extra waits? The
  discarded cross-session copy experiment does not establish a lower bound.
- Can the frame-independent encoder be evaluated in batches while keeping
  recurrent decoding sequential and identical in frame order?

For GPU-only tests, bind outputs to GPU memory and explicitly synchronize them
before stopping the timer. Otherwise an apparent speedup may measure queued work
instead of completed work. Primary references: [D3D12 timing](https://learn.microsoft.com/en-us/windows/win32/direct3d12/timing)
and [ORT's DML device/queue API](https://github.com/microsoft/onnxruntime/blob/v1.22.0/include/onnxruntime/core/providers/dml/dml_provider_factory.h).

### Completed execution-cost experiments

The synchronized resident-network test measured **3.445 ms** on the RTX 5060,
versus approximately **3.845 ms** for the complete steady-state segment call.
Source data is uploaded once; recurrent states still advance, outputs remain
on DirectML, and `synchronize_outputs()` completes the work before timing ends.
This is a zero-image cost probe, not a video quality test or a GPU timestamp
measurement. It includes host dispatch and waiting. The difference suggests
that eliminating all conversion and transfer overhead alone cannot reach 5x.

For the 7.753 ms Clip A baseline, 5x requires **at most 1.551 ms per frame**. The
current network submission/completion cost itself exceeds that target. Amdahl's
law matters here: optimizing the roughly 0.4 ms outside this measured network
path cannot remove the remaining approximately 2.3 ms.

The research harness extracts fixed-shape blocks, uploads their inputs once,
and synchronizes GPU outputs after each run. Each row below measures 500 runs
after 100 warmups. Inputs are constant synthetic tensors. Block boundaries
change fusion and allocation, so times must not be added as an exact full-model
breakdown. MAC counts count convolution multiply-accumulates, not all operations.

| Block, 480x360, ratio 1 | Convolution MACs/frame | RTX 5060 mean |
|---|---:|---:|
| Entire specialized graph | 5.441 billion | 3.369 ms |
| Encoder, ASPP, source pyramid | 1.315 billion | 1.340 ms |
| Recurrent bottleneck | 0.153 billion | 0.141 ms |
| Decoder at 1/8 resolution | 0.566 billion | 0.253 ms |
| Decoder at 1/4 resolution | 0.649 billion | 0.469 ms |
| Decoder at 1/2 resolution | 1.331 billion | 0.644 ms |
| Full-resolution output head | 1.280 billion | 0.746 ms |
| Guided refinement | 0.147 billion | 0.444 ms |
| Entire recurrent decoder plus output head | 3.979 billion | 1.879 ms |

The high-resolution decoder/head contributes substantially more convolution
work than the encoder. Replacing only the backbone cannot be assumed to achieve
the desired speedup. Refinement's latency is relatively large for its MAC count,
which also demonstrates why FLOPs alone are a poor architecture-selection metric.

### Equivalent graph and submission experiments

These measurements are completed GPU-output graph runs, not `Engine::segment`
times. They use the same 480x360 shape and ratio 1. The control in the variant
sweep was 3.420 ms. Small differences are within laptop clock/run variability;
only a repeatable gain followed by video validation would justify promotion.

| Experiment | Mean | Decision |
|---|---:|---|
| Pad unaligned dense-convolution input channels to multiples of 8, append zero weights | 3.329 ms | Only about 3%; not enabled |
| Pad to multiples of 16 | 3.551 ms | Slower |
| Pad to multiples of 32 | 4.237 ms | Slower |
| Compute only alpha in the final refiner projection and subsequent channelwise work | 3.474 ms | No NVIDIA gain; compiles on AMD |
| Combine four 4-channel learned box convolutions into one 16-group convolution | 3.545 ms | Slower |
| Combine that grouping with alpha-only projection | 3.601 ms | Slower |
| Rewrite 1x1 convolutions as transposed-layout matrix multiplications | 3.784 ms | Slower |
| Capture with fixed GPU input/output addresses and explicit completion | 3.548 ms | No substantial gain |
| Convert the export to FP32 | Compilation failed | No valid latency claim |
| FP32 plus the equivalent alpha Gather | 5.136 ms | Compiles, but slower than FP16 |
| Disable DirectML metacommands, with alpha Gather | 6.098 ms | Slower; left enabled in production |
| Replace same-size refiner resizes with identity | AMD compilation still failed | Did not solve compatibility |
| Replace final alpha Split with Slice | AMD compilation still failed | Did not solve compatibility |
| Replace final alpha Split with Gather | AMD 10.662 ms network-only | Retained as compilation retry |

Padding to 8 passed the Clip A quality gate (MAE 0.002778, maximum 7 levels), but
does not address the missing multi-fold speedup. The alpha-only projection
variant passed on AMD (MAE 0.001019, maximum 1 level), yet Gather is a smaller
change that leaves the projection weights untouched and produced exact masks.

Capture was configured **before provider registration**, as required by the
[ORT provider implementation](https://github.com/microsoft/onnxruntime/blob/v1.22.0/onnxruntime/core/providers/dml/dml_provider_factory.cc).
Earlier very fast capture readings with incorrect masks or incomplete GPU work
are rejected. The retained capture probe has constant inputs and is only a
submission-cost experiment; it does not implement captured video recurrence.

The export contains trained, nonuniform `refiner.box_filter.weight` values.
Despite the variable name and the source's initial 1/9 initialization, replacing
them with an average-pooling box filter would change the learned function.
Likewise, this particular ONNX export continues to refine at ratio 1, even though
the [upstream PyTorch model](https://github.com/PeterL1n/RobustVideoMatting/blob/master/model/model.py)
conditionally skips refinement at that ratio. The loaded ONNX model remains the
quality reference. Omitting refinement is not an equivalent optimization.

### Temporal batching experiments

The graph generator can unroll 1, 2, 4, or 8 full frames into one ONNX graph,
sharing weights and explicitly wiring each recurrent output into the next frame.
There is no frame skipping. One/two-frame synthetic probes measured 3.375 and
6.644 ms respectively: approximately 3.38 versus 3.32 ms per frame. Larger runs
also remained around 3.3 ms/frame. One larger-batch measurement overlapped another
GPU test, so it is not used for a speedup claim. Unrolling alone is not promising.

The frame-independent encoder separately achieved about 0.83 ms/frame at batch
4 and 0.80 ms/frame at batch 8. This could reduce encoder submission overhead,
but would require a split encoder/decoder execution pipeline, extra buffering,
and careful queue synchronization. It does not batch recurrent decoding across
time. That encoder improvement alone would not bring the complete path to
1.55 ms/frame. The sequence probes measure throughput; they do not establish
interactive single-frame latency or validate a batched video pipeline.

### Smaller-model probes and the no-additional-footage constraint

No training has been run. The user chose to continue without additional footage;
the two supplied clips remain held out. The script generated separate, explicitly
untrained architecture probes to test whether a student is fast enough to justify
training. They are not selected by the application and are not quality-equivalent.

Two families were measured on DirectML:

1. Replace decoder 3x3 dense convolutions with per-input-channel spatial SVD
   factors and 1x1 channel mixing, optionally removing encoder residual blocks.
   This preserves recurrent tensor sizes but approximates learned convolutions.
2. Reduce channel widths while preserving compatible skips, both GRU gates,
   recurrent shapes, and concatenation indices. Channel counts round to multiples
   of eight. Initialization simply selects channels; it is not learned pruning.

| Untrained architecture | Stored initializer elements (approx.) | RTX network-only mean |
|---|---:|---:|
| Rank-1 spatial decoder factors, unchanged encoder | 3.27 million | 3.818 ms |
| Rank-2 factors, four encoder residual blocks removed | 2.11 million | 8.666 ms |
| Rank-1 factors, ten residual blocks removed | 1.16 million | 3.103 ms |
| 75% widths, rounded to 8 | 2.20 million | 2.824 ms |
| 50% widths, rounded to 8 | 1.01 million | 2.191 ms |
| 25% widths, rounded to 8 | 0.30 million | 1.846 ms |

The factorized versions illustrate a concrete failure of FLOP-based reasoning:
additional grouped convolutions can make DirectML slower. Even the quarter-width
model's network time already exceeds the full-call 5x target. A more aggressive
quarter-width model with ten residual blocks removed measured **2.132 ms for the
full segment call**, and failed Clip A quality severely: **MAE 109.999805 levels,
maximum 248**, with almost every pixel differing by over 4. It is rejected.
This result is evidence against naïve channel deletion, not evidence that a
properly trained student must fail.

Distillation without manual alpha labels is feasible in principle: a teacher
can label ordinary camera video. Distillation without representative imagery
is a different, substantially harder problem. With no additional footage, a
possible experimental source is synthetic video generated from inverted teacher
images plus coherent motion/appearance changes. There is no evidence yet that
this would preserve hair, disocclusions, transparency, or long-term recurrence.

[DeepInversion](https://arxiv.org/abs/1912.08795) uses teacher batch-normalization
statistics to regularize synthesized inputs. This supplied ONNX graph contains
**zero BatchNormalization nodes**; its convolutions already absorb those layers.
The required original statistics cannot simply be read from this export.
Obtaining the original training checkpoint would help that experiment, but would
not establish that synthesized frames cover real video behavior.

If a viable student architecture is identified, a staged training experiment
would use CUDA only for training and ordinary ONNX/DirectML for deployment:

- First fit feature projections and alpha on short sequences; initialize from
  teacher weights where shapes match. Keep teacher and student histories separate.
- Extend unroll length, with teacher/student burn-in before scored frames, so
  training does not teach a sequence of independent first frames.
- Combine alpha error, boundary-weighted error, spatial gradient and multiscale
  losses, plus matching temporal changes. Penalize disagreement with teacher
  motion rather than simply forcing consecutive student masks to be identical.
- Validate every checkpoint on held-out sequences at the deployment resolution
  and ratio. Examine hair/hand crops and contrasting composites, and measure
  boundary and temporal error as well as whole-image means.
- Accept only a checkpoint that passes both the quality checks and measured
  non-CUDA latency. Neither synthetic training loss nor parameter count qualifies.

The [RVM paper](https://arxiv.org/abs/2108.11515) supports preserving temporal
modeling as an architectural requirement. It does not establish a 5x smaller
student with unchanged quality; that remains an unproven research target here.

### Reproducing the deeper probes

Research scripts: `scripts/research_rvm.py`; ignored Rust probes in
`src/engine/onnx/benchmark.rs`. Generated ONNX graphs stay under `target/`.
The generator is deliberately tied to this export's node layout and asserts
key node names before extracting blocks. Do not use it blindly on other models.

```powershell
# Export a fixed-shape research copy while running the completed-network probe.
$env:SEGMENTER_BENCH_MODEL = 'rvm_mobilenetv3_fp16.onnx'
$env:SEGMENTER_BENCH_EXPORT_STATIC = 'target/rvm-static-480x360.onnx'
cargo test --offline --release resident_network_benchmark -- --ignored --nocapture
Remove-Item Env:SEGMENTER_BENCH_EXPORT_STATIC

# Obtain shape information. This CPU session is never used for timing.
$env:SEGMENTER_BENCH_GRAPH = 'target/rvm-static-480x360.onnx'
$env:SEGMENTER_BENCH_INFER_SHAPES = 'target/rvm-cpu-inferred.onnx'
cargo test --offline --release graph_benchmark -- --ignored --nocapture
Remove-Item Env:SEGMENTER_BENCH_INFER_SHAPES

# With the pinned research-only Python modules available:
python scripts/research_rvm.py target/rvm-static-480x360.onnx target/rvm-cpu-inferred.onnx target/rvm-blocks
$env:SEGMENTER_BENCH_GRAPH = 'target/rvm-blocks'
cargo test --offline --release graph_benchmark -- --ignored --nocapture
```

`SEGMENTER_BENCH_GRAPH` accepts a model or a directory. All directory models run
sequentially, and compilation failures are reported individually before the test
fails. Use `SEGMENTER_BENCH_DEVICE=1` for the AMD adapter on this machine and
`SEGMENTER_BENCH_CAPTURE=1` for the fixed-address capture probe.
`SEGMENTER_BENCH_NO_METACOMMANDS=1` tests the actual DirectML provider option
through ORT's provider-registration API; an invented session configuration key
would not test this setting. Do not run timing
experiments concurrently. Do not interpret `target/rvm-student-probes` files as
trained models. Keep model, device, ratio, dimensions, frame counts, and capture
settings with any reported result.

For additional quality metrics after saving raw masks:

```powershell
python scripts/rvm_alpha_quality.py target/rvm-baseline-cam.alpha target/rvm-final-cam.alpha 480 360
```

Research-only Python tooling uses already selected versions of well-established
projects: [ONNX 1.18.0 (2025-05-12)](https://pypi.org/project/onnx/1.18.0/),
[protobuf 5.29.5 (2025-05-28)](https://pypi.org/project/protobuf/5.29.5/), and
[typing_extensions 4.14.0 (2025-06-02)](https://pypi.org/project/typing-extensions/4.14.0/).
They are isolated under `target/rvm-python`, use the existing Python/NumPy runtime,
and are older than the requested two-month minimum. Production Cargo dependencies
have not changed. No CUDA inference runtime is added.

### Where the evidence points next

The missing speedup is substantial. On Clip A, the retained NVIDIA path still needs
roughly another 2.5x improvement to reach the original 5x request. Current results
do not justify promising that a smaller model or another configuration flag will
deliver it without a quality tradeoff.

The shared-queue timing and native shader experiments proposed here have now
been implemented; see the next section. They support selective fusion, with a
measured but modest whole-inference gain. They do not support expecting a 5x
speedup simply from replacing the runtime or using HLSL.

Do not start a long distillation run merely because a graph has fewer weights:
the measured quarter-width candidate is already too slow for the full-call target,
and the factorized candidates can regress badly. A viable student first needs
measured DirectML latency headroom for uploads, readback, and conversion, followed
by evidence that training without additional footage can recover quality.

## Custom DirectX investigation (2026-11-03)

**A custom shader does win for the refiner. A complete, GPU-buffer-sharing
prototype also runs faster, but is nowhere near another multi-fold speedup.**
This code is opt-in research, behind `research-dx12`; normal CLI inference still
uses the retained DirectML specialization. No model was trained and no new
dependency was installed for these shader experiments.

### What the native implementation does

`dx12_probe.cpp` creates a D3D12 device, one DIRECT queue, a DirectML device,
timestamp queries, and fences. ONNX Runtime uses that same device and queue via
its [DirectML provider API](https://github.com/microsoft/onnxruntime/blob/v1.22.0/include/onnxruntime/core/providers/dml/dml_provider_factory.h).
The shader uses HLSL Shader Model 6.0, compiled by the already installed Windows
SDK 10.0.26100.0 DXC. It runs on both tested adapters without CUDA.

`refiner_probe.hlsl` fuses the equal-size guided-refinement stage into a single
dispatch:

- Load RGB guidance and the four-channel foreground/alpha projection.
- Compute learned 3x3 local means, covariance and variance, retaining the
  actual nonuniform box-filter weights.
- Combine those statistics with all 16 decoder feature channels.
- Run the learned 24-to-16-to-16 multilayer perceptron and the final alpha row.
  RGB statistics still influence alpha through the hidden layers; they are
  not removed. Only final channels unused by alpha are omitted.
- Produce clamped alpha, rounding to FP16 at the original operation boundaries.
  Accumulation uses FP32. Different accumulation order still allows small
  numerical differences, so this is not a bit-identical implementation.

The shared-memory version uses 16x8 thread groups with an 18x10 halo, holding
5,760 bytes of guidance/projection data per group. The main saving comes from
fusing operations and eliminating intermediate tensor traffic. A version using
direct cached reads is only slightly slower on the tested workloads.

This implementation currently requires full-resolution, equal-size refinement
(the tested ratio is 1). It is tied to this model's architecture and weights.
It does not implement arbitrary ratios, resize coordinate modes, arbitrary RVM
exports, CPU fallback, or macOS. Those are deployment work, not assumptions.

### GPU timing, with completed work

The probe places timestamps on the shared queue, resolves them, waits for a
fence, and divides by that queue's timestamp frequency. D3D12 timestamps mark
completion of preceding work; see Microsoft's [timing documentation](https://learn.microsoft.com/en-us/windows/win32/direct3d12/timing).
These measurements are **queue spans**, including any idle gaps between
submissions, not a hardware-counter sum of kernel busy time.

For the complete specialized NVIDIA network, constant resident inputs gave:

| Calls between timestamp pairs | Queue span per call | Wall time per call |
|---|---:|---:|
| 1 | 3.2605 ms | 3.4062 ms |
| 16 | 3.2553 ms | 3.2666 ms |
| 64 | 3.2267 ms | 3.2305 ms |

This is a scheduling probe, not video-quality evaluation. Bulk submission barely
changes its cost, so removing a per-frame CPU wait is insufficient to reach the
1.551 ms target. It does not prove that every internal kernel is efficient.

The isolated refiner uses actual frame-75 activations from Clip A at 480x360.
Inputs/outputs are GPU resident during timing; setup, upload and readback are
excluded. Both implementations use the same device/queue. AMD uses the equivalent
Gather export because the original alpha Split fails compilation on that adapter.

| Adapter / measurement | DirectML | HLSL direct reads | HLSL shared halo | DML / shared |
|---|---:|---:|---:|---:|
| NVIDIA, one call per timestamp pair | 0.3959 ms | 0.1629 ms | 0.1595 ms | 2.48x |
| NVIDIA, 50 calls per pair | 0.3534 ms | 0.1543 ms | 0.1493 ms | 2.37x |
| AMD, one call per pair | 1.6050 ms | 0.3491 ms | 0.3199 ms | 5.02x |
| AMD, 50 calls per pair | 1.6560 ms | 0.1809 ms | 0.1725 ms | 9.60x |

The AMD difference between isolated calls and repeated dispatches is substantial;
report both rather than presenting the throughput result as single-frame latency.
These factors apply to **one stage**, not the entire model or video export.

### Actual DirectML-to-HLSL handoff

The `dx12_hybrid_video` probe removes refinement from the ONNX graph and exports
only the needed decoder features, projection, guidance and recurrent outputs.
It preallocates the three intermediate outputs on DirectML and obtains their
D3D12 resources through the provider API. HLSL reads those buffers directly on
the same queue, with UAV/SRV state transitions restored afterward. Recurrent
features remain on the GPU and advance on every frame. The CPU reads back only
the final alpha; intermediate tensors never make a CPU round trip.

The custom path uses a persistent output/readback buffer and recorded command
lists. A fence after the readback covers both DirectML and HLSL work. No runtime
shader compilation or resource setup is included in steady-state timings.

This comparison runs two implementations inside the same research harness:
the full specialized DirectML graph and the prefix plus HLSL. It includes CPU
preprocessing, source upload, recurrent binding, inference, alpha readback and
byte conversion; decoding and session creation are excluded. The harness has
different allocation details from `Engine::segment`, so compare the two columns
within this table, not directly against earlier production-path timings.

| Adapter / clip | Full specialized DML | DML prefix + HLSL | Time reduction |
|---|---:|---:|---:|
| NVIDIA / Clip A | 4.3184 ms | 4.1303 ms | 4.4% |
| NVIDIA / easy | 3.6816 ms | 3.5969 ms | 2.3% |
| AMD / Clip A | 11.7796 ms | 10.3927 ms | 11.8% |
| AMD / easy | 8.7691 ms | 7.7321 ms | 11.8% |

Each row uses 150 decoded frames, excluding the first 30 from timing. Laptop
clocks and other activity affect short runs; the NVIDIA easy run had noticeable
tail variance. A longer, reversed-order NVIDIA Clip A check used 1,500 calls,
excluding 30: **4.3634 ms full / 4.1374 ms hybrid (5.2% less time)**.
The same longer, reversed-order check on AMD measured **12.2636 ms full /
10.4959 ms hybrid (14.4% less time)**.
Each 150-frame repeat resets recurrent state and then processes the complete
sequence. Repeats increase timing duration, not evaluation-video diversity.

The measured saving is consistent with the refiner's limited share of total
time. Even eliminating a 0.35 ms stage entirely cannot turn a roughly 3.3 ms
network into a 1.55 ms complete segment call. This is the key limit on the
otherwise impressive isolated-stage numbers.

### Quality of the actual hybrid

The comparison below is against the original unspecialized alpha files, not
just the graph with additional outputs. All 150 frames of each clip were
checked, preserving recurrent history. Units are 8-bit alpha levels.

| Adapter / clip | Pixels | MAE | Max | Fraction >4 | Boundary MAE | Temporal-change MAE |
|---|---:|---:|---:|---:|---:|---:|
| NVIDIA / Clip A | 25,920,000 | 0.004345 | 4 | 0 | 0.075718 | 0.006689 |
| NVIDIA / easy | 19,440,000 | 0.003140 | 6 | 0.000000823 | 0.065045 | 0.004926 |
| AMD / Clip A | 25,920,000 | 0.004132 | 6 | 0.000000386 | 0.071998 | 0.006436 |
| AMD / easy | 19,440,000 | 0.003177 | 8 | 0.000002984 | 0.065769 | 0.005040 |

All four pass the existing numerical gate. AMD easy reaches its maximum-error
limit of 8; this is evidence for similarity on the supplied clips, not a
universal promise of imperceptible loss. Temporal-change maximum differences
are 5, 8, 7 and 10 respectively. No additional footage was used for training
or validation. The shader weights were extracted, not fitted to these clips.
Inspected AMD comparison panels include the frames with the largest summed
alpha difference in each clip (Clip A frame 147 and easy frame 28). The reference
and hybrid composites showed no visible difference in those panels; the 16x
error view showed sparse boundary differences. Still-image inspection does not
establish perceptual equivalence for all motion or unseen footage.

### Dense convolution experiment: rejected

A second HLSL candidate tiled the 35-input, 16-output, 3x3 `Conv_290` and ReLU,
using FP16 storage, FP32 accumulation, shared input/weight tiles and 16x8 groups.
DirectML executes that extracted convolution in approximately **0.460 ms**.
The custom version returned `DXGI_ERROR_DEVICE_REMOVED` (`0x887a0005`) at full
480x360 dimensions, including a bounded one-dispatch run. A single 16x8 group
completed, but provides neither valid full-frame throughput nor correctness.
The precise driver/kernel failure cause has not been established.

This candidate has no valid speedup or quality claim and is disabled by default
even in its ignored test. `SEGMENTER_CONV_GROUP_ONLY=1` permits just the tiny
diagnostic; the rejected full-size experiment requires the explicit
`SEGMENTER_RUN_REJECTED_CONV` override. No GPU timeout or driver settings were
changed. It is not part of the working hybrid.

### Why a complete shader rewrite is a larger research project

DirectML already compiles GPU graphs and uses optimized convolution kernels.
The earlier measured regression with metacommands disabled is a reason to retain
those kernels until a replacement wins under completed-work timing. Ordinary
HLSL multiply-add code is not automatically an efficient matrix-engine kernel.

RVM has multi-resolution features, convolutions with halos, global channel
reductions, recurrent gates and upsampling. A single ordinary dispatch cannot
provide all the required global synchronization across independent thread groups.
A native implementation would use multiple kernels, carefully chosen layouts,
buffer lifetimes and barriers. The opportunity is to fuse compatible local work
between those boundaries, not to mechanically translate every ONNX operator.

Modern matrix/vector shader features also need explicit hardware and driver
checks. Microsoft's [Shader Model 6.9 retail announcement](https://devblogs.microsoft.com/directx/shader-model-6-9-retail-and-more/)
lists new driver/hardware requirements and says Cooperative Vector was deprecated
in favor of a future unified approach. It is not a universally available route
to matrix acceleration on customers' existing PCs. The tested working prototype
uses SM 6.0 instead.

The next useful targets are fused recurrent pointwise operations, resize/concat
feeding high-resolution convolutions, and layout-aware dense decoder kernels.
Measure each as an isolated block, then in the complete recurrent pipeline:
materializing an intermediate output can change DirectML fusion and erase a local
win. A larger speedup requires improvements across the dominant encoder/decoder
work, or a successfully trained architecture with substantially less computation.
Neither has been demonstrated here.

### Reproduce the shader experiments

Source files: `dx12_probe.cpp`, `dx12_probe.rs`, `refiner_probe.hlsl`, the rejected
`conv_probe.hlsl`, and `scripts/export_rvm_shader_probe.py`. Native code is built
only with `--features research-dx12`. It uses the existing `cc` dependency,
Windows SDK and DirectML import library. The default build does not link this
helper. Artifacts and raw masks remain under ignored `target/` directories.
The ordinary release test suite passes 59 tests (5 opt-in tests ignored).
With the research feature enabled it passes the same 59 (10 opt-in tests
ignored); the actual video and timing probes were also run explicitly as
described above. The failed dense-convolution experiment remains rejected.

Starting with the fixed-shape and CPU-shape exports described earlier:

```powershell
$env:PYTHONPATH = 'target/rvm-python'
python scripts/export_rvm_shader_probe.py target/rvm-static-480x360.onnx target/rvm-cpu-inferred.onnx target/rvm-dx12
$rvmDxc = 'C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/dxc.exe'
& $rvmDxc -T cs_6_0 -E main -O3 -D USE_SHARED=1 -Fo target/rvm-dx12/refiner-shared.cso src/engine/onnx/refiner_probe.hlsl
& $rvmDxc -T cs_6_0 -E main -O3 -D USE_SHARED=0 -Fo target/rvm-dx12/refiner-direct.cso src/engine/onnx/refiner_probe.hlsl

$env:SEGMENTER_BENCH_VIDEO = 'C:/path/to/clip-a.mp4'
$env:SEGMENTER_BENCH_REFERENCE = 'target/rvm-baseline-cam.alpha'
$env:SEGMENTER_BENCH_SAVE = 'target/rvm-hybrid-cam.alpha'
cargo test --offline --release --features research-dx12 dx12_hybrid_video -- --ignored --nocapture --test-threads=1
```

The hybrid test runs both full DML and prefix/HLSL. For AMD, set
`SEGMENTER_BENCH_DEVICE=1`, `SEGMENTER_HYBRID_GATHER=1`, and use its own baseline
mask file. `SEGMENTER_HYBRID_REPEATS=10` extends timing;
`SEGMENTER_HYBRID_REVERSE=1` reverses execution order. Optional
`SEGMENTER_BENCH_PREVIEWS` writes comparison panels outside timed regions.

For isolated timing, first run `dx12_refiner_video` to save real frame-75
activations, then `dx12_refiner_timing`. On AMD, set `SEGMENTER_SHADER_MODEL` to
`target/rvm-dx12/intermediates-gather.onnx` for the former and
`SEGMENTER_BENCH_GRAPH` to `target/rvm-dx12/refiner-gather.onnx` for the latter.
Use `SEGMENTER_SHADER_DIR` for another export directory; the easy clip needs a
separate 480x270 fixed graph. All shader tests are opt-in and must run sequentially.

## Deeper backend and algorithm experiments (2026-11-03)

**The ratio-1 5x target remains unmet.** These experiments did not replace the retained
specialization. For Clip A at 480x360, ratio 1, the original 7.753 ms segment call
requires **at most 1.5506 ms** to claim 5x. The retained 3.845 ms result needs a
further 2.48x improvement. Removing the entire refiner would not close that gap.
No training occurred, neither clip was used to fit weights, and no additional
footage was acquired. CUDA was not used for inference.

### Channels-last layouts

`scripts/rvm_layout_probe.py` tests ONNX Runtime's `com.microsoft.NhwcConv`
operator, including OHWI weight conversion. Candidates wrap selected convolutions
or propagate NHWC through the complete graph, remapping channel axes, broadcast
constants, resize dimensions and shape operations. The latter keeps NCHW only at
external interfaces and where required by supported pooling operators.
The operator is implemented in the
[ORT DirectML convolution source](https://github.com/microsoft/onnxruntime/blob/v1.22.0/onnxruntime/core/providers/dml/DmlExecutionProvider/src/Operators/DmlOperatorConvolution.cpp).
Physical layout can matter, as described in
[DirectML's tensor-stride documentation](https://learn.microsoft.com/en-us/windows/ai/directml/dml-strides),
but does not by itself guarantee a faster compiled graph.

NVIDIA, fixed 480x360 graph, 100 warmup and 500 completed calls per candidate:

| Layout | Mean graph time |
|---|---:|
| Control | 3.3611 ms |
| All convolutions wrapped | 3.9908 ms |
| Dense convolutions wrapped | 3.9912 ms |
| Decoder convolutions wrapped | 3.7341 ms |
| Encoder convolutions wrapped | 3.5675 ms |
| Continuous NHWC graph | 3.7624 ms |

All candidates were slower, so none advanced to full-video quality acceptance.
Successful compilation alone is not evidence of equivalent masks. These graph
times exclude video preprocessing and alpha conversion; they must not be compared
directly with the 7.753 ms original whole-call baseline.

### Winograd as an ONNX graph

`scripts/rvm_winograd_probe.py` implements F(2x2,3x3): input transforms, batched
matrix multiplication and output transforms. The transform algebra was checked
independently in NumPy. The learned convolution weights are transformed, not
retrained. [Lavin and Gray](https://arxiv.org/abs/1509.09308) describe the arithmetic
savings of this family of algorithms. Algebraic equivalence over real numbers
does not imply identical FP16 results or efficient execution as a generic graph.

| Candidate | Mean graph time |
|---|---:|
| Control | 3.3555 ms |
| Decoder Winograd | 29.7885 ms |
| Output head Winograd | 17.8217 ms |
| Output head with FP32 transforms | 19.3115 ms |
| Higher-resolution decoder Winograd | 27.4061 ms |

The extra transforms and materialized tensors overwhelm the arithmetic saving in
this implementation. These variants were rejected on speed before video quality
validation. This does not establish that a well-tuned native Winograd kernel is
inherently slow; the native experiment below tests that separately.

### Alternative Vulkan runtime with the original weights

An isolated `research-ncnn` feature builds a native Vulkan probe using Tencent
ncnn. `scripts/export_rvm_ncnn_probe.py` converts the supplied specialized ONNX
graph into ncnn's documented parameter format, retaining its learned weights.
FP16 weight values are expanded exactly into FP32 serialization. This does not
substitute the checkpoint used in ncnn's RVM example.

The probe holds all four recurrent states in Vulkan buffers, uploads RGB and
downloads alpha, and waits for GPU completion before stopping the timer. It runs
all 150 frames in order. Repeated sequences reset recurrence at their boundaries.
Unsupported CPU layers cause an error. The measured configurations used Vulkan
on both adapters, with no CUDA and no CPU inference fallback.

The converter required several correctness fixes before measurement: explicit
fanout splits, decimal/exponent serialization for floating-point parameters,
preservation of recurrent buffers consumed later in the graph, and input binding
by layer name after optimization. An early positional-input variant crashed;
its run is not a valid measurement. Diagnostic outputs also isolated an incorrect
full-resolution channel mean to the generic Vulkan reduction path. That mean was
replaced by an equivalent 1x1 convolution with coefficients 1/C, using FP32 for
the reduction; global averages use dedicated global pooling. The inference that
the original reduction exceeded a dispatch-dimension limit comes from inspecting
the [pinned Vulkan reduction source](https://github.com/Tencent/ncnn/blob/20260526/src/layer/vulkan/reduction_vulkan.cpp)
and intermediate tensors, not a Vulkan validation-layer report.

Corrected Clip A results, 150 frames, first 30 excluded from timing, with the strict
alpha gate used elsewhere in this document:

| Device / variant | Whole probe call | Alpha MAE (byte levels) | Maximum difference | Gate |
|---|---:|---:|---:|---|
| NVIDIA, unoptimized | 7.4833 ms | 0.006200 | 7 | Pass |
| NVIDIA, optimized graph | 6.1557 ms | 0.006179 | 7 | Pass |
| NVIDIA, optimized + FP16 arithmetic | 7.7864 ms | 0.011060 | 13 | Fail |
| NVIDIA, optimized + channels padded to 16 | 6.9774 ms | 0.006263 | 8 | Pass |
| AMD, optimized graph | 38.0344 ms | 0.007886 | 7 | Pass |

The unoptimized FP16-arithmetic variant was also slower (10.1885 ms) and failed
the gate (maximum difference 13). FP16 storage with FP32 arithmetic is distinct
from FP16 arithmetic. The latter is not a free speed or quality improvement here.
For optimized NVIDIA, native command recording averaged 1.1514 ms and
submission/completion wait 4.2654 ms. On AMD these were 1.3522 and 35.9924 ms.
Those diagnostic CPU spans include their named work; they are not isolated kernel
timestamps. No conclusion about all Vulkan implementations follows from one
runtime, conversion and device/driver combination.

This backend passed the numerical gate on the hard clip in its conservative
configuration but was slower than the existing DirectML path on both GPUs. It
was not promoted, and no broader perceptual-equivalence claim is made.

Dependency provenance: the only newly installed package in this deeper pass was
the official **ncnn 20260526 Windows VS2022 SDK**, released 2026-05-26, over five
months before testing. ncnn is a widely used Tencent project (approximately
24,000 GitHub stars at inspection, repository created in 2017).
The [official release](https://github.com/Tencent/ncnn/releases/tag/20260526)
asset was `ncnn-20260526-windows-vs2022.zip`; its verified SHA-256 was
`5ed388b134c50ba8eb2e6605f3a16f692f26598d45113bc9794aa4857d691e26`.
It lives under ignored `target/rvm-ncnn-sdk/`; default builds do not require it.
The existing Vulkan driver, Windows SDK, C++ compiler and Rust `cc` dependency
were reused. No CUDA, GPU driver, Vulkan SDK or latest/unpinned package was
installed. Earlier ONNX/NumPy research dependencies were reused.

### Revised native DirectX convolution kernels

Two further kernels address the failed dense-convolution prototype. Both retain
the actual Conv_290 weights, FP16 input/output boundaries, FP32 accumulation,
3x3 padding and ReLU. They first pass a bounded 64x64 real-activation check, then
run full 480x360. GPU timestamps cover completed queue work. These are isolated
layer comparisons, not full-model speedups.

| Native candidate | Full-size DirectML control | Full-size HLSL | Feature MAE | Maximum feature error |
|---|---:|---:|---:|---:|
| Four output channels per thread, shared halo | 0.4883 ms | 1.2790 ms | 0.00031958 | 0.01171875 |
| Fused Winograd F(2x2,3x3), four-channel input tiles | 0.5010 ms | 0.9465 ms | 0.00031959 | 0.01171875 |

`conv_v2_probe.hlsl` bounds accumulator count and shares the complete input halo.
`conv_winograd_probe.hlsl` performs input transforms, transformed-weight products,
output transforms and ReLU in one dispatch, without large intermediate buffers.
Its final design uses 128 threads, a 32x16 output region and four input channels
per shared tile. A first 32-thread variant was rejected by the small-tensor speed
gate; the revised variant passed at 0.0736 ms versus DirectML's 0.1273 ms at
64x64, but lost at full resolution. Small-tensor wins did not predict full-frame
wins. Neither replacement is used by the application.

These results reinforce the measured opportunity for targeted elementwise and
refinement fusion, while showing that ordinary multiply-add shaders did not beat
DirectML's dense convolution kernels on this device. They do not prove that a
specialist matrix-engine kernel or a future compiler could not do better.

### Data-free low-rank approximation

`scripts/rvm_lowrank_probe.py` uses only the supplied weights to factor selected
3x3 convolutions into a lower-rank 3x3 convolution and a 1x1 projection, with
FP16 factors. It does not fit the two evaluation clips. This is an approximation;
retaining most weight energy is not a guarantee about output masks.

| Candidate | Mean completed graph time |
|---|---:|
| Control | 3.3408 ms |
| Decoder at half output rank | 3.3522 ms |
| Decoder at quarter output rank | 3.3670 ms |
| Selected high-resolution layers at half rank | 3.4398 ms |
| Selected high-resolution layers at three-quarter rank | 3.4786 ms |

The quarter-rank decoder was also tested on the complete hard sequence. It took
3.821 ms per segment call and failed decisively: alpha MAE **89.697339/255**,
maximum error 212, and 98.303295% of pixels differing by more than four byte
levels. Reducing theoretical convolution work did not deliver useful runtime
or preserve the matte. No rank approximation was retained.

### Consequence for the 5x plan

The evidence does not support promising 5x from a custom shader rewrite, a
backend switch or weight-only compression. The existing refiner shader is useful
but optimizes too small a fraction of the complete frame. The earlier quarter-
width architecture probe also missed the whole-call target before any quality
recovery, so simply training that candidate is not yet justified by speed.

A viable student must first demonstrate a completed-call time below 1.55 ms
with preprocessing, recurrent state and alpha readback included. It likely needs
fewer sequential operations and substantially less high-resolution decoder work,
with a cheap source-guided edge stage. Only then would distillation be worth its
cost. Synthetic training examples are possible without new camera footage, but
there is no evidence here that they recover hair, motion and transparency to the
required quality. Neither the absence of a labeled dataset nor a teacher model
eliminates the need for representative training inputs and independent video
validation. No such student has been trained or qualified in this work.

### Reproducing this pass

Use the previously prepared Python/ONNX environment and fixed-shape exports.
Generated graphs, SDK files and masks remain under `target/`. Run GPU tests
sequentially, never concurrently.

```powershell
python scripts/rvm_layout_probe.py target/rvm-static-480x360.onnx target/rvm-cpu-inferred.onnx target/rvm-layout
python scripts/rvm_winograd_probe.py target/rvm-static-480x360.onnx target/rvm-cpu-inferred.onnx target/rvm-winograd
python scripts/rvm_lowrank_probe.py target/rvm-static-480x360.onnx target/rvm-cpu-inferred.onnx target/rvm-lowrank
$env:SEGMENTER_BENCH_GRAPH = 'target/rvm-layout' # or either other directory
cargo test --offline --release graph_benchmark -- --ignored --nocapture --test-threads=1

python scripts/export_rvm_ncnn_probe.py target/rvm-static-480x360.onnx target/rvm-cpu-inferred.onnx target/rvm-ncnn-cam
& target/rvm-ncnn-sdk/ncnn-20260526-windows-vs2022/x64/bin/ncnnoptimize.exe target/rvm-ncnn-cam/rvm.param target/rvm-ncnn-cam/rvm.bin target/rvm-ncnn-cam/optimized.param target/rvm-ncnn-cam/optimized.bin 0
$env:SEGMENTER_NCNN_MODEL = 'target/rvm-ncnn-cam'
$env:SEGMENTER_NCNN_FLAGS = '64'
$env:SEGMENTER_BENCH_VIDEO = 'C:/path/to/clip-a.mp4'
$env:SEGMENTER_BENCH_REFERENCE = 'target/rvm-baseline-cam.alpha'
cargo test --offline --release --features research-ncnn ncnn_video_benchmark -- --ignored --nocapture --test-threads=1

python scripts/export_rvm_shader_probe.py target/rvm-static-480x360.onnx target/rvm-cpu-inferred.onnx target/rvm-dx12
& $rvmDxc -T cs_6_0 -E main -O3 -Fo target/rvm-dx12/conv-v2.cso src/engine/onnx/conv_v2_probe.hlsl
& $rvmDxc -T cs_6_0 -E main -O3 -Fo target/rvm-dx12/conv-winograd.cso src/engine/onnx/conv_winograd_probe.hlsl
# Saved frame75 activations from dx12_refiner_video are also required.
$env:SEGMENTER_CONV_V2_FULL = '1'
cargo test --offline --release --features research-dx12 dx12_convolution_v2 -- --ignored --nocapture --test-threads=1
$env:SEGMENTER_CONV_WINOGRAD = '1'
cargo test --offline --release --features research-dx12 dx12_convolution_v2 -- --ignored --nocapture --test-threads=1
```

For the Vulkan AMD comparison, use `SEGMENTER_NCNN_DEVICE=1` and the AMD reference
mask. Flags 66 enable optimized graph plus FP16 arithmetic, which failed quality.
The standard graph and inference benchmark flags described earlier also apply to
the low-rank candidates. The benchmark correctly fails when its quality gate is
exceeded; those failures are rejected research candidates, not passing checks.
