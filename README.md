A small CLI utility to run RVM on a local video: https://github.com/PeterL1n/RobustVideoMatting

macOS release builds created by `scripts/release.py` include Metal on both architectures
and ONNX Runtime on Intel (`x86_64-apple-darwin`). Apple Silicon releases keep the
existing Metal-only default. To build the Intel binary manually:

```sh
cargo build --release --target x86_64-apple-darwin --features engine-onnx
```

Pass an RVM FP32 ONNX model to `--model-path` to use the CPU backend on Intel macOS;
no additional engine flag is needed. Model format is detected from the file contents,
so a downloaded ONNX model named `.bin` also works:

```sh
./target/x86_64-apple-darwin/release/segmenter \
  --input input.mov --output mask.mov \
  --model-path segmenter-rvm-mobilenetv3-fp32-onnx.bin
```

Passing an RVM Metal model continues to select the Metal backend.
