"""Compare raw benchmark masks, including boundaries and temporal changes.

Usage: python scripts/rvm_alpha_quality.py reference.alpha candidate.alpha WIDTH HEIGHT
Outputs JSON; all errors use 0..255 alpha units. No ground-truth quality claim.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("width", type=int)
    p.add_argument("height", type=int)
    args = p.parse_args()
    assert args.width > 0 and args.height > 0
    ref = np.fromfile(args.reference, dtype=np.uint8)
    actual = np.fromfile(args.candidate, dtype=np.uint8)
    assert len(ref) == len(actual) and len(ref) % (args.width * args.height) == 0
    assert len(ref) > 0
    ref = ref.reshape(-1, args.height, args.width)
    actual = actual.reshape(ref.shape)
    absolute_sum = boundary_sum = boundary_pixels = temporal_sum = temporal_pixels = 0
    max_error = max_temporal = over4 = 0
    previous_error = None
    for r, a in zip(ref, actual):
        error = a.astype(np.int16) - r.astype(np.int16)
        magnitude = np.abs(error)
        absolute_sum += int(magnitude.sum())
        max_error = max(max_error, int(magnitude.max()))
        over4 += int(np.count_nonzero(magnitude > 4))
        # Include all partial alpha and both sides of any adjacent alpha change,
        # then dilate by 2 pixels. This catches sharp binary boundaries as well.
        edge = (r > 0) & (r < 255)
        vertical = r[1:] != r[:-1]
        horizontal = r[:, 1:] != r[:, :-1]
        edge[1:] |= vertical
        edge[:-1] |= vertical
        edge[:, 1:] |= horizontal
        edge[:, :-1] |= horizontal
        padded = np.pad(edge, 2)
        band = np.zeros(edge.shape, dtype=bool)
        for dy in range(5):
            for dx in range(5):
                band |= padded[dy:dy + args.height, dx:dx + args.width]
        boundary_sum += int(magnitude[band].sum())
        boundary_pixels += int(band.sum())
        if previous_error is not None:
            delta = np.abs(error - previous_error)
            temporal_sum += int(delta.sum())
            temporal_pixels += delta.size
            max_temporal = max(max_temporal, int(delta.max()))
        previous_error = error
    print(json.dumps(dict(
        frames=len(ref), pixels=ref.size,
        mae_u8=absolute_sum/ref.size, max_u8=max_error, fraction_over4=over4/ref.size,
        boundary_pixels=boundary_pixels,
        boundary_mae_u8=boundary_sum/boundary_pixels if boundary_pixels else None,
        temporal_change_mae_u8=temporal_sum/temporal_pixels if temporal_pixels else None,
        temporal_change_max_u8=max_temporal,
    ), indent=2))


if __name__ == "__main__":
    main()
