#!/usr/bin/env python3
"""Compare release binaries with alternating, warmed full-video runs (stdlib only)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, default=Path("target/release/segmenter"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("target/rvm-bench/final"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--baseline-pipeline", choices=("false", "true"), default="false",
                        help="Use true when comparing against an already pipelined baseline")
    parser.add_argument("--candidate-arg", action="append", default=[],
                        help="Extra candidate-only argument; repeat, using --candidate-arg=--flag for flags")
    parser.add_argument("--baseline-arg", action="append", default=[],
                        help="Extra baseline-only argument; repeat for each argument")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("SEGMENTER_METAL_PROFILE", None)
    report = {"model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
              "downsample_ratio": 0.25, "warmups": 1,
              "baseline_pipeline": args.baseline_pipeline, "clips": []}
    report["candidate_args"] = args.candidate_arg
    report["baseline_args"] = args.baseline_arg
    for index, video in enumerate(args.input):
        results = {"input": str(video.resolve()), "baseline": [], "candidate": []}
        for iteration in range(-1, args.repeats):
            # Alternate order to reduce systematic thermal/cache bias.
            order = ["baseline", "candidate"] if iteration % 2 == 0 else ["candidate", "baseline"]
            for variant in order:
                output = args.output_dir / f"{index}-{video.stem}-{variant}-{iteration}.mov"
                command = [str(getattr(args, variant).resolve()), "--input", str(video.resolve()),
                           "--output", str(output.resolve()), "--model-path", str(args.model.resolve()),
                           "--downsample-ratio", "0.25"]
                if variant == "baseline":
                    command.append(f"--pipeline={args.baseline_pipeline}")
                    command.extend(args.baseline_arg)
                else:
                    command.extend(args.candidate_arg)
                start = time.perf_counter()
                run = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
                elapsed = time.perf_counter() - start
                output.with_suffix(".log").write_bytes(run.stdout + run.stderr)
                if run.returncode:
                    raise RuntimeError(f"{variant} failed: {run.stderr.decode(errors='replace')}")
                progress = re.findall(rb"PROGRESS: \S+ (\d+)", run.stderr)
                if not progress:
                    raise RuntimeError(f"{variant} did not report completed frames")
                frames = int(progress[-1])
                if "frames" in results and results["frames"] != frames:
                    raise RuntimeError("baseline and candidate frame counts differ")
                results["frames"] = frames
                if iteration >= 0:
                    results[variant].append(elapsed)
        for variant in ["baseline", "candidate"]:
            results[f"{variant}_median_seconds"] = statistics.median(results[variant])
        results["speedup"] = results["baseline_median_seconds"] / results["candidate_median_seconds"]
        report["clips"].append(results)
        print(json.dumps(results), flush=True)
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
