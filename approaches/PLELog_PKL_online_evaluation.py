"""Validation-calibrated causal online evaluation for an unchanged PLELog model.

This is an evaluation *adapter*, not a reimplementation of PLELog training.
Supply a scorer from your PLELog repository using --scorer module:function.
The scorer accepts a tuple of event IDs/templates and must return the
probability of the *Anomalous* class, using an already-trained frozen model.

Example:
 python PLELog_PKL_online_evaluation.py \
   --data_root ../../LWADLS/datasets --dataset_name HDFS \
   --event-column EventId --label-column Label \
   --scorer my_plelog_adapter:score_anomaly --window 120

You MUST implement my_plelog_adapter.score_anomaly(events) against your
repository's PLELog instance/encoding APIs. No model or weights are changed.

Security: pickle files execute arbitrary code when loaded. Only open trusted PKLs.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Sequence:
    sequence_id: str
    events: tuple
    anomalous: bool


def parse_label(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)) and not pd.isna(value):
        if float(value) in (0, 1):
            return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ('normal', '0', 'false', 'benign'):
            return False
        if s in ('anomalous', 'anomaly', 'abnormal', '1', 'true'):
            return True
    raise ValueError(f'Unrecognized binary label: {value!r}')


def read_sequences(path: Path, id_column: str, event_column: str,
                   label_column: str) -> list[Sequence]:
    if not path.is_file():
        raise FileNotFoundError(f'Missing PKL split: {path}')
    data = pd.read_pickle(path)  # Trusted local PKL files ONLY.
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f'{path} must contain a pandas DataFrame, got {type(data).__name__}')
    for name in (id_column, event_column, label_column):
        if name not in data.columns:
            raise ValueError(f'{path}: required column {name!r} missing; found {list(data.columns)!r}')
    if data.empty:
        raise ValueError(f'{path}: split is empty')
    if data[id_column].isna().any():
        raise ValueError(f'{path}: null {id_column!r} values found')
    if data[label_column].isna().any():
        raise ValueError(f'{path}: null {label_column!r} values found')
    if data[event_column].map(lambda x: x is None or (isinstance(x, (float, np.floating)) and math.isnan(x))).any():
        raise ValueError(f'{path}: null {event_column!r} values found')

    seqs: dict[str, list] = {}
    labels: dict[str, bool] = {}
    for idx, row in data.iterrows():
        sid = str(row[id_column])
        if not sid.strip():
            raise ValueError(f'{path}: empty ID at row {idx!r}')
        label = parse_label(row[label_column])
        if sid in labels and labels[sid] != label:
            raise ValueError(f'{path}: conflicting labels for {sid}')
        labels[sid] = label
        value = row[event_column]
        if isinstance(value, (list, tuple, np.ndarray)):
            values = list(value)
        else:
            values = [value]
        if not values or any(x is None or (isinstance(x, float) and math.isnan(x)) for x in values):
            raise ValueError(f'{path}: empty/null event for {sid}')
        seqs.setdefault(sid, []).extend(values)
    return [Sequence(sid, tuple(events), labels[sid]) for sid, events in seqs.items()]


def check_disjoint(splits: dict[str, list[Sequence]]) -> None:
    seen: dict[str, str] = {}
    for split, seqs in splits.items():
        for seq in seqs:
            if seq.sequence_id in seen:
                raise ValueError(f'Leakage: Node_block_id {seq.sequence_id!r} is in both '
                                 f'{seen[seq.sequence_id]} and {split}')
            seen[seq.sequence_id] = split


def import_scorer(spec: str) -> Callable[[tuple], float]:
    if ':' not in spec:
        raise ValueError('--scorer must be module:function (e.g. my_plelog_adapter:score_anomaly)')
    module, name = spec.split(':', 1)
    fn = getattr(importlib.import_module(module), name)
    if not callable(fn):
        raise TypeError(f'{spec} is not callable')
    return fn


def score_causally(seqs: list[Sequence], scorer: Callable[[tuple], float],
                   window: int) -> list[np.ndarray]:
    outputs = []
    for seq in seqs:
        scores = []
        for index in range(len(seq.events)):
            # Only the current event and up to window - 1 preceding events.
            history = seq.events[max(0, index + 1 - window):index + 1]
            prob = float(scorer(history))
            if not math.isfinite(prob) or not 0 <= prob <= 1:
                raise ValueError(f'Invalid anomaly probability {prob!r} for {seq.sequence_id}')
            scores.append(prob)
        outputs.append(np.asarray(scores, dtype=float))
    return outputs


def first_alert(scores: np.ndarray, threshold: float, consecutive: int) -> int | None:
    count = 0
    for index, p in enumerate(scores):
        count = count + 1 if p >= threshold else 0
        if count >= consecutive:
            return index + 1  # 1-based number of processed events
    return None


def evaluate(seqs: list[Sequence], score_arrays: list[np.ndarray],
             threshold: float, consecutive: int, fp_weight: float,
             fn_weight: float, delay_weight: float) -> dict[str, Any]:
    tp = tn = fp = fn = 0
    early_ratios = []
    anomalous = sum(s.anomalous for s in seqs)
    edr = {25: 0, 50: 0, 75: 0}
    for seq, scores in zip(seqs, score_arrays):
        index = first_alert(scores, threshold, consecutive)
        if seq.anomalous:
            if index is None:
                fn += 1
            else:
                tp += 1
                ratio = index / len(seq.events)
                early_ratios.append(ratio)
                for pct in edr:
                    if ratio <= pct / 100:
                        edr[pct] += 1
        elif index is None:
            tn += 1
        else:
            fp += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    # AvgRatio is conditional on successful detections; coverage/recall is separate.
    avg_ratio = float(np.mean(early_ratios)) if early_ratios else None
    fp_cost, fn_cost = fp * fp_weight, fn * fn_weight
    delay_cost = delay_weight * sum(early_ratios)
    return {
        'n_sequences': len(seqs), 'n_anomalous': anomalous,
        'threshold': threshold, 'consecutive': consecutive,
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'precision': precision, 'recall': recall, 'F1': f1, 'FPR': fpr,
        'coverage': recall, 'AvgRatio_detected_only': avg_ratio,
        'EDR@25': edr[25] / anomalous if anomalous else 0.0,
        'EDR@50': edr[50] / anomalous if anomalous else 0.0,
        'EDR@75': edr[75] / anomalous if anomalous else 0.0,
        'FP_cost': fp_cost, 'FN_cost': fn_cost, 'delay_cost': delay_cost,
        'total_cost': fp_cost + fn_cost + delay_cost,
    }


def calibrate(seqs: list[Sequence], score_arrays: list[np.ndarray],
              thresholds: list[float], k_candidates: list[int], max_fpr: float | None,
              fp_weight: float, fn_weight: float, delay_weight: float) -> dict:
    candidates = []
    for k in k_candidates:
        for threshold in thresholds:
            result = evaluate(seqs, score_arrays, threshold, k,
                              fp_weight, fn_weight, delay_weight)
            if max_fpr is None or result['FPR'] <= max_fpr:
                candidates.append(result)
    if not candidates:
        raise ValueError('No validation calibration candidate satisfies --max-validation-fpr')
    # F1 is primary, FPR tie-breaks; costs are reported and never tuned on test.
    return max(candidates, key=lambda r: (r['F1'], -r['FPR'], r['recall'],
                                          -r['total_cost'], -r['threshold'], -r['consecutive']))


def build_split_paths(base_dir: Path, dataset_name: str) -> dict[str, Path]:
    """Match the provided dataset-folder and PKL filename convention."""
    if not dataset_name or dataset_name in ('.', '..') or '/' in dataset_name or '\\' in dataset_name:
        raise ValueError('dataset-name must be a single dataset folder name')
    split_folder = f"1_{dataset_name}_Splitted_Datasets"
    folder = base_dir / dataset_name / split_folder
    return {
        'train': folder / 'train_df.pkl',
        'val': folder / 'val_df.pkl',
        'test': folder / 'test_df.pkl',
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_root', '--base-dir', dest='data_root',
                   default=Path('../../LWADLS/datasets'), type=Path,
                   help='Root folder for reading PKL files, e.g., ../../LWADLS/datasets')
    p.add_argument('--dataset_name', '--dataset-name', dest='dataset_name', required=True,
                   help='Dataset folder name, e.g. HDFS, BGL or TH_1G')
    p.add_argument('--event-column', required=True,
                   help='Event list column, or one-event-per-row column')
    p.add_argument('--label-column', required=True)
    p.add_argument('--id-column', default='Node_block_id')
    p.add_argument('--scorer', required=True, help='Trusted module:function, returning P(anomaly)')
    p.add_argument('--window', type=int, default=120)
    p.add_argument('--consecutive-candidates', default='1,2,3,5')
    p.add_argument('--thresholds', default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99')
    p.add_argument('--max-validation-fpr', type=float, default=None)
    p.add_argument('--fp-weight', type=float, default=1.0)
    p.add_argument('--fn-weight', type=float, default=1.0)
    p.add_argument('--delay-weight', type=float, default=1.0)
    p.add_argument('--output', type=Path, default=Path('plelog_online_metrics.json'))
    args = p.parse_args()
    if args.window < 1:
        p.error('--window must be positive')
    for name in ('fp_weight', 'fn_weight', 'delay_weight'):
        if getattr(args, name) < 0:
            p.error(f'--{name.replace("_", "-")} cannot be negative')
    thresholds = [float(x) for x in args.thresholds.split(',')]
    k_candidates = [int(x) for x in args.consecutive_candidates.split(',')]
    if not thresholds or not all(0 <= x <= 1 for x in thresholds):
        p.error('all thresholds must be in [0,1]')
    if not k_candidates or not all(x >= 1 for x in k_candidates):
        p.error('consecutive-candidates must be positive integers')
    if args.max_validation_fpr is not None and not 0 <= args.max_validation_fpr <= 1:
        p.error('--max-validation-fpr must be in [0,1]')

    split_paths = build_split_paths(args.data_root, args.dataset_name)
    print('PKL split paths:', {k: str(v) for k, v in split_paths.items()}, file=sys.stderr)
    splits = {split: read_sequences(path, args.id_column,
                                    args.event_column, args.label_column)
              for split, path in split_paths.items()}
    check_disjoint(splits)
    print('Validated input sizes:', {k: len(v) for k, v in splits.items()}, file=sys.stderr)
    scorer = import_scorer(args.scorer)
    # The original model is trained externally, on full sequences, with no prefix augmentation.
    # No training/weight changes occur here. Validation is used ONLY for alert calibration.
    val_scores = score_causally(splits['val'], scorer, args.window)
    selected = calibrate(splits['val'], val_scores, thresholds, k_candidates,
                         args.max_validation_fpr, args.fp_weight,
                         args.fn_weight, args.delay_weight)
    test_scores = score_causally(splits['test'], scorer, args.window)
    test_results = evaluate(splits['test'], test_scores,
                            selected['threshold'], selected['consecutive'],
                            args.fp_weight, args.fn_weight, args.delay_weight)
    result = {
        'protocol': 'frozen original PLELog weights; rolling causal inference; validation-calibrated alerts',
        'window': args.window, 'split_counts': {k: len(v) for k, v in splits.items()},
        'validation_selection': selected, 'test_metrics': test_results,
        'notes': ['EDR denominator = all anomalous sequences',
                  'AvgRatio conditional on true-positive detected sequences',
                  'Delay cost = delay_weight * sum(detection fraction over true positives)',
                  'Costs are illustrative unless weights match the study protocol',
                  'This script does not train the PLELog model or test original full-sequence baseline'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
