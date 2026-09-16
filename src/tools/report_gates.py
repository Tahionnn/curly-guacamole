"""Inspect learned fusion gate magnitudes in a checkpoint.

A small final gate is not proof that a branch never trained; inspect training
history and ablations before interpreting whether the branch is useful.
"""

import argparse
from pathlib import Path

import numpy as np
import torch


class GateReport:
    SUFFIXES = ('channel_gate', 'gamma', 'scale')

    def __init__(self, state: dict):
        self.state = state

    def rows(self) -> list[tuple[str, float, float]]:
        rows = []
        for key, value in sorted(self.state.items()):
            if isinstance(value, torch.Tensor) and value.numel() and key.endswith(self.SUFFIXES):
                magnitude = np.abs(value.detach().float().cpu().numpy())
                rows.append((key, float(magnitude.mean()), float(magnitude.max())))
        return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--section', default='ema', choices=('model', 'ema'))
    parser.add_argument('--open-above', type=float, default=.01, help='label mean magnitudes above this value')
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state = saved.get(args.section)
    if not isinstance(state, dict):
        parser.error(f'checkpoint has no {args.section!r} state dict')
    rows = GateReport(state).rows()
    if not rows:
        parser.error('no fusion gates found in this checkpoint')
    width = max(len(key) for key, _, _ in rows)
    for key, mean, peak in rows:
        label = 'above threshold' if mean > args.open_above else 'small magnitude'
        print(f'{key:<{width}}  mean|g|={mean:.5f}  max|g|={peak:.5f}  {label}')
    print(f'\nepoch={saved.get("epoch")} best_aic={saved.get("best_aic")}')


if __name__ == '__main__':
    main()
