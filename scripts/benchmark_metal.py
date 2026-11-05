#!/usr/bin/env python3
"""Compare native RVM throughput on predecoded clips using only the standard library."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--loops", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1 or not 1 <= args.loops <= 100:
        parser.error("repeats must be positive; loops must be in 1..100")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "downsample_ratio": 0.25,
        "predecoded": True,
        "loops": args.loops,
        "warmup_runs": 1,
        "excluded_initial_frames": 10,
        "clips": [],
    }
    for index, video in enumerate(args.input):
        result = {"input": str(video.resolve()), "baseline": [], "candidate": []}
        for iteration in range(-1, args.repeats):
            order = ["baseline", "candidate"] if iteration % 2 == 0 else ["candidate", "baseline"]
            for variant in order:
                env = os.environ.copy()
                for key in ("SEGMENTER_METAL_PROFILE", "SEGMENTER_BENCH_ASYNC", "SEGMENTER_BENCH_NATIVE"):
                    env.pop(key, None)
                env.update(SEGMENTER_BENCH_PREDECODE="1", SEGMENTER_BENCH_NO_HASH="1",
                           SEGMENTER_BENCH_LOOPS=str(args.loops))
                if variant == "candidate":
                    env.update(SEGMENTER_BENCH_ASYNC="1", SEGMENTER_BENCH_NATIVE="1")
                command = [str(getattr(args, variant).resolve()), str(args.model.resolve()),
                           str(video.resolve()), str((args.output_dir / "timing.sha256").resolve())]
                run = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
                (args.output_dir / f"{index}-{variant}-{iteration}.log").write_text(run.stdout + run.stderr)
                frames = int(re.search(r"frames=(\d+)", run.stdout)[1])
                if "frames" in result and result["frames"] != frames:
                    raise RuntimeError("baseline and candidate frame counts differ")
                result["frames"] = frames
                if iteration >= 0:
                    result[variant].append(float(re.search(r"steady_fps=([0-9.]+)", run.stdout)[1]))
        for variant in ("baseline", "candidate"):
            result[f"{variant}_median_fps"] = statistics.median(result[variant])
        result["speedup"] = result["candidate_median_fps"] / result["baseline_median_fps"]
        report["clips"].append(result)
        print(json.dumps(result), flush=True)
    (args.output_dir / "inference.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
