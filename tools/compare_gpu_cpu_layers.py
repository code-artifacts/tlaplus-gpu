#!/usr/bin/env python3
"""Compare complete canonical state sets exported for each BFS layer."""

import argparse
import collections
import re
import sys
from pathlib import Path


LAYER_RE = re.compile(r"layer-(\d+)\.states$")


def read_layers(directory):
    layers = {}
    for path in sorted(Path(directory).glob("layer-*.states")):
        match = LAYER_RE.match(path.name)
        if match is None:
            continue
        lines = [line.rstrip("\n") for line in path.open("r", encoding="utf-8") if line.strip()]
        layers[int(match.group(1))] = lines
    return layers


def compare(cpu_dir, gpu_dir, examples):
    cpu = read_layers(cpu_dir)
    gpu = read_layers(gpu_dir)
    all_layers = sorted(set(cpu) | set(gpu))
    failed = False
    total_cpu = 0
    total_gpu = 0

    for depth in all_layers:
        cpu_lines = cpu.get(depth, [])
        gpu_lines = gpu.get(depth, [])
        cpu_counts = collections.Counter(cpu_lines)
        gpu_counts = collections.Counter(gpu_lines)
        cpu_set = set(cpu_lines)
        gpu_set = set(gpu_lines)
        only_cpu = sorted(cpu_set - gpu_set)
        only_gpu = sorted(gpu_set - cpu_set)
        duplicate_cpu = sum(count - 1 for count in cpu_counts.values() if count > 1)
        duplicate_gpu = sum(count - 1 for count in gpu_counts.values() if count > 1)
        total_cpu += len(cpu_lines)
        total_gpu += len(gpu_lines)
        layer_ok = not only_cpu and not only_gpu and duplicate_cpu == 0 and duplicate_gpu == 0
        print("layer=%d cpu=%d gpu=%d only_cpu=%d only_gpu=%d duplicate_cpu=%d duplicate_gpu=%d %s" % (
            depth, len(cpu_lines), len(gpu_lines), len(only_cpu), len(only_gpu),
            duplicate_cpu, duplicate_gpu, "PASS" if layer_ok else "FAIL"))
        if not layer_ok:
            failed = True
            for label, values in (("only_cpu", only_cpu), ("only_gpu", only_gpu)):
                for value in values[:examples]:
                    print("  %s[%d]=%s" % (label, depth, value))

    print("total_cpu_states=%d" % total_cpu)
    print("total_gpu_states=%d" % total_gpu)
    print("layers=%d" % len(all_layers))
    print("RESULT=%s" % ("FAIL" if failed or not all_layers else "PASS"))
    return 1 if failed or not all_layers else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cpu_dir")
    parser.add_argument("gpu_dir")
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()
    return compare(args.cpu_dir, args.gpu_dir, max(0, args.examples))


if __name__ == "__main__":
    sys.exit(main())
