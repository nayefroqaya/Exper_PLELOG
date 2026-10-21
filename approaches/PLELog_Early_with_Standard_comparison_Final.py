import sys
print("Imported _1 ")
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = ""   # ⛔ Disable GPU completely
import colorama
colorama.init()
sys.path.extend([".", ".."])
from CONSTANTS import *
print("Imported _2 ")
import time
from utils.common import get_precision_recall
import shutil
import json
import copy
import threading
import subprocess
from contextlib import contextmanager
try:
    from tqdm.auto import tqdm
except ImportError:
    # Lightweight fallback so the script still runs without tqdm.
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, unit=None, leave=True, **kwargs):
            self.iterable = iterable
            self.total = total
            self.desc = desc or "Progress"
            self.n = 0
            if iterable is not None and total is None:
                try:
                    self.total = len(iterable)
                except Exception:
                    self.total = None
            print(f"[{self.desc}] started")

        def __iter__(self):
            for item in self.iterable:
                yield item
                self.update(1)
            self.close()

        def update(self, n=1):
            self.n += n
            if self.total:
                pct = 100.0 * self.n / max(1, self.total)
                print(f"\r[{self.desc}] {self.n}/{self.total} ({pct:.1f}%)", end="", flush=True)

        def set_postfix(self, *args, **kwargs):
            pass

        def set_postfix_str(self, s="", refresh=True):
            print(f"\r[{self.desc}] {self.n} sec | {s}", end="", flush=True)

        def set_description(self, desc=None, refresh=True):
            if desc is not None:
                self.desc = desc

        def close(self):
            print()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.close()


from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from sklearn.decomposition import FastICA
from representations.templates.statistics import Simple_template_TF_IDF, Template_TF_IDF_without_clean
from representations.sequences.statistics import Sequential_TF
from preprocessing.datacutter.SimpleCutting import cut_by_613
from preprocessing.AutoLabeling import Probabilistic_Labeling
from preprocessing.Preprocess import PKLPreprocessor

from module.Optimizer import Optimizer
from module.Common import data_iter, generate_tinsts_binary_label, batch_variable_inst
from models.gru import AttGRUModel
from utils.Vocab import Vocab
from sklearn.preprocessing import StandardScaler
print("Imported all......")

lstm_hiddens = 100
num_layer = 1  #2
batch_size = 100
epochs = 20 #5
print('beging ------')

# ============================================================
# Minimal early-aware training augmentation
# ============================================================
EARLY_TRAINING_ENABLED = True
TRAIN_PREFIX_RATIOS = (0.25, 0.50, 0.75)


def _event_is_anomaly(value):
    """Convert common event-level label formats to a binary anomaly flag."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) != 0.0

    if isinstance(value, str):
        v = value.strip().lower()
        normal_values = {
            "normal", "0", "false", "benign", "negative", "-", "none", ""
        }
        anomaly_values = {
            "anomaly", "anomalous", "1", "true", "abnormal",
            "positive", "error", "attack"
        }
        if v in anomaly_values:
            return True
        if v in normal_values:
            return False

    return None


def _find_event_level_labels(inst):
    """
    Find event-level labels already stored in an instance.
    Only metadata with exactly one label per event is accepted.
    """
    candidate_attrs = (
        "event_labels",
        "sequence_labels",
        "line_labels",
        "anomaly_flags",
        "event_anomalies",
        "token_labels",
    )

    sequence = list(getattr(inst, "sequence", []))
    T = len(sequence)

    for attr in candidate_attrs:
        if not hasattr(inst, attr):
            continue

        values = getattr(inst, attr)
        if values is None or isinstance(values, (str, bytes)):
            continue

        try:
            values = list(values)
        except Exception:
            continue

        if len(values) != T:
            continue

        converted = [_event_is_anomaly(v) for v in values]
        if all(v is not None for v in converted):
            return converted, attr

    return None, None


def build_early_training_instances(original_train, prefix_ratios=TRAIN_PREFIX_RATIOS):
    """
    Add a small number of prefix samples to TRAIN while preserving PLELog.

    Rules:
      1) Keep every original full sequence.
      2) A Normal sequence can safely provide Normal prefixes.
      3) For an Anomaly sequence, prefixes are used only when true event-level
         labels reveal the first anomalous event:
           before onset -> Normal
           at/after onset -> Anomaly
      4) If only a sequence-level Anomaly label exists, no early anomalous
         prefix is fabricated. The original full anomalous sequence is kept.
    """
    augmented = list(original_train)

    normal_prefixes = 0
    anomaly_prefixes = 0
    anomaly_sequences_with_onset = 0
    anomaly_sequences_without_onset = 0
    onset_label_sources = {}

    for inst in tqdm(
        original_train,
        total=len(original_train),
        desc="Build early-training prefixes",
        unit="sequence",
        dynamic_ncols=True,
    ):
        sequence = list(getattr(inst, "sequence", []))
        T = len(sequence)

        if T <= 1:
            continue

        # Full T is already present in `augmented`; add only shorter prefixes.
        prefix_steps = sorted({
            max(1, min(T - 1, int(round(T * float(ratio)))))
            for ratio in prefix_ratios
            if 0.0 < float(ratio) < 1.0
        })

        if not prefix_steps:
            continue

        if inst.label == "Normal":
            for t in prefix_steps:
                prefix_inst = copy.copy(inst)
                prefix_inst.sequence = sequence[:t]
                prefix_inst.label = "Normal"
                prefix_inst.early_prefix_step = int(t)
                prefix_inst.early_prefix_ratio = float(t / T)
                prefix_inst.early_training_source = "normal_sequence"
                augmented.append(prefix_inst)
                normal_prefixes += 1
            continue

        # Anomalous sequence: use early labels only if event-level onset exists.
        event_flags, source_attr = _find_event_level_labels(inst)

        if event_flags is None:
            anomaly_sequences_without_onset += 1
            continue

        anomaly_positions = [
            idx for idx, is_anomaly in enumerate(event_flags) if is_anomaly
        ]

        if not anomaly_positions:
            anomaly_sequences_without_onset += 1
            continue

        first_anomaly_step = anomaly_positions[0] + 1  # 1-based
        anomaly_sequences_with_onset += 1
        onset_label_sources[source_attr] = onset_label_sources.get(source_attr, 0) + 1

        for t in prefix_steps:
            prefix_inst = copy.copy(inst)
            prefix_inst.sequence = sequence[:t]

            if t < first_anomaly_step:
                prefix_inst.label = "Normal"
                normal_prefixes += 1
            else:
                prefix_inst.label = "Anomaly"
                anomaly_prefixes += 1

            prefix_inst.early_prefix_step = int(t)
            prefix_inst.early_prefix_ratio = float(t / T)
            prefix_inst.early_first_anomaly_step = int(first_anomaly_step)
            prefix_inst.early_training_source = f"event_level:{source_attr}"

            # Keep copied per-event metadata consistent with the shorter sequence.
            try:
                original_values = getattr(inst, source_attr)
                setattr(prefix_inst, source_attr, list(original_values)[:t])
            except Exception:
                pass

            augmented.append(prefix_inst)

    random.shuffle(augmented)

    print("\n" + "=" * 72)
    print("EARLY-AWARE TRAINING AUGMENTATION")
    print("=" * 72)
    print(f"Original full training sequences : {len(original_train)}")
    print(f"Added Normal prefixes            : {normal_prefixes}")
    print(f"Added Anomaly prefixes           : {anomaly_prefixes}")
    print(f"Anomaly seq. with onset metadata : {anomaly_sequences_with_onset}")
    print(f"Anomaly seq. without onset data  : {anomaly_sequences_without_onset}")
    print(f"Final training instances         : {len(augmented)}")
    print(f"Training prefix ratios           : {prefix_ratios}")
    if onset_label_sources:
        print(f"Event-level label sources        : {onset_label_sources}")
    else:
        print(
            "Event-level anomaly onset       : NOT FOUND -> "
            "no fabricated anomalous prefixes"
        )
    print("=" * 72 + "\n")

    return augmented


def prepare_prefix_representations(
    prefix_instances,
    encoder,
    scaler=None,
    transformer=None,
    chunk_size=10000,
):
    """
    Compute representations for newly created prefix samples only.

    IMPORTANT:
      - encoder is the same PLELog Sequential_TF encoder
      - scaler and FastICA were already fitted on ORIGINAL full-sequence TRAIN
      - nothing is refitted on prefixes
      - processing is chunked to avoid HDFS memory spikes
    """
    if not prefix_instances:
        return

    chunk_size = max(1, int(chunk_size))

    bar = tqdm(
        total=len(prefix_instances),
        desc="Prepare early-prefix representations",
        unit="prefix",
        dynamic_ncols=True,
    )

    for start in range(0, len(prefix_instances), chunk_size):
        chunk = prefix_instances[start:start + chunk_size]

        raw = np.asarray(encoder.present(chunk))

        if scaler is not None:
            raw = scaler.transform(raw)
            raw = np.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=-1e6)

        if transformer is not None:
            raw = transformer.transform(raw)

        for inst, vec in zip(chunk, raw):
            inst.repr = vec

        bar.update(len(chunk))

    bar.close()


def compute_static_baseline_metrics(
    y_true,
    y_pred,
    y_score=None,
    seq_lengths=None,
    fp_unit_cost=10.0,
    fn_unit_cost=20.0,
    delay_unit_cost=5.0,
):
    """
    Compute all available test metrics for static full-sequence baselines
    such as PLELog.

    PLELog predicts after observing the complete sequence. Therefore, for
    correctly detected anomalous sequences, the default early-detection values are:
        detection_step  = sequence length T
        detection_ratio = 1.0
        EDR@25/50/75    = 0.0
        delay_cost      = TP * delay_unit_cost
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if seq_lengths is None:
        seq_lengths = np.ones_like(y_true, dtype=float)
    else:
        seq_lengths = np.asarray(seq_lengths).astype(float)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)

    auroc = None
    auprc = None
    if y_score is not None:
        y_score = np.asarray(y_score).astype(float)
        if len(np.unique(y_true)) == 2:
            try:
                auroc = roc_auc_score(y_true, y_score)
            except Exception:
                auroc = None
            try:
                auprc = average_precision_score(y_true, y_score)
            except Exception:
                auprc = None

    total_anomalies = int(tp + fn)
    detected_anomalies = int(tp)
    detection_coverage = detected_anomalies / total_anomalies if total_anomalies > 0 else 0.0

    detected_mask = (y_true == 1) & (y_pred == 1)
    if detected_anomalies > 0:
        detected_lengths = seq_lengths[detected_mask]
        average_detection_step = float(np.mean(detected_lengths))
        average_detection_ratio = 1.0
        median_detection_ratio = 1.0
    else:
        average_detection_step = None
        average_detection_ratio = None
        median_detection_ratio = None

    # Static full-sequence classifier: it never alerts before the end.
    edr_25 = 0.0
    edr_50 = 0.0
    edr_75 = 0.0

    # Fraction of sequences predicted as anomaly.
    alert_rate = float(np.mean(y_pred == 1))

    # PLELog is not an RL method, so average reward is unavailable.
    average_reward = None

    fp_total_cost = float(fp * fp_unit_cost)
    fn_total_cost = float(fn * fn_unit_cost)
    delay_total_cost = float(tp * delay_unit_cost)  # delay ratio = 1.0 for each TP
    total_cost = fp_total_cost + fn_total_cost + delay_total_cost
    average_cost_per_sequence = total_cost / max(1, len(y_true))

    return {
        "num_sequences": int(len(y_true)),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity_tnr": float(specificity),
        "f1_score": float(f1),
        "f2_score": float(f2),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "mcc": float(mcc),
        "auroc": None if auroc is None else float(auroc),
        "auprc": None if auprc is None else float(auprc),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "confusion_matrix": cm.tolist(),
        "total_anomalies": int(total_anomalies),
        "detected_anomalies": int(detected_anomalies),
        "anomaly_detection_coverage": float(detection_coverage),
        "average_detection_step": average_detection_step,
        "average_detection_ratio": average_detection_ratio,
        "median_detection_ratio": median_detection_ratio,
        "edr_25": float(edr_25),
        "edr_50": float(edr_50),
        "edr_75": float(edr_75),
        "average_reward": average_reward,
        "alert_rate": float(alert_rate),
        "false_positive_unit_cost": float(fp_unit_cost),
        "false_negative_unit_cost": float(fn_unit_cost),
        "delay_unit_cost": float(delay_unit_cost),
        "false_positive_total_cost": float(fp_total_cost),
        "false_negative_total_cost": float(fn_total_cost),
        "delay_total_cost": float(delay_total_cost),
        "total_cost": float(total_cost),
        "average_cost_per_sequence": float(average_cost_per_sequence),
    }


def format_static_baseline_metrics(metrics, model_name="PLELog"):
    """
    Return a compact readable text report with the selected metrics only:
      - Precision, Recall, F1 for anomaly class
      - Classification report for class 0 and class 1
      - Confusion matrix
      - Important early-detection metrics
      - FP cost, FN cost, Delay cost
    """
    avg_step_text = "N/A" if metrics["average_detection_step"] is None else f"{metrics['average_detection_step']:.4f}"
    avg_ratio_text = "N/A" if metrics["average_detection_ratio"] is None else f"{metrics['average_detection_ratio']:.4f}"

    class_report_text = metrics.get("classification_report_text", "Classification report was not saved.")

    lines = []
    lines.append("#" * 80)
    lines.append(f"{model_name}: SELECTED TEST METRICS")
    lines.append("#" * 80)
    lines.append(f"Number of sequences   : {metrics['num_sequences']}")
    lines.append("")

    lines.append("[Classification Metrics - Anomaly Class]")
    lines.append("Positive class        : 1 = anomaly")
    lines.append(f"Precision             : {metrics['precision']:.4f}")
    lines.append(f"Recall / TPR          : {metrics['recall']:.4f}")
    lines.append(f"F1-score              : {metrics['f1_score']:.4f}")
    lines.append("")

    lines.append("[Classification Report - Class 0 and Class 1]")
    lines.append("Class 0               : Normal")
    lines.append("Class 1               : Anomaly")
    lines.append("")
    lines.append(class_report_text.rstrip())
    lines.append("")

    lines.append("[Confusion Matrix]")
    lines.append("Labels: 0=normal, 1=anomaly")
    lines.append(str(np.asarray(metrics["confusion_matrix"])))
    lines.append(f"TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}")
    lines.append("")

    lines.append("[Early Detection Metrics]")
    lines.append("PLELog is treated as a static full-sequence classifier.")
    lines.append("Default assumption: detected anomalies are detected at the end of the sequence.")
    lines.append(f"Total anomalies       : {metrics['total_anomalies']}")
    lines.append(f"Detected anomalies    : {metrics['detected_anomalies']}")
    lines.append(f"Detection coverage    : {metrics['anomaly_detection_coverage']:.4f}")
    lines.append(f"Avg detection step    : {avg_step_text}")
    lines.append(f"Avg detection ratio   : {avg_ratio_text}")
    lines.append(f"EDR@25                : {metrics['edr_25']:.4f}")
    lines.append(f"EDR@50                : {metrics['edr_50']:.4f}")
    lines.append(f"EDR@75                : {metrics['edr_75']:.4f}")
    lines.append("")

    lines.append("[Cost-Sensitive Metrics]")
    lines.append(f"False-positive cost   : {metrics['false_positive_total_cost']:.4f}")
    lines.append(f"False-negative cost   : {metrics['false_negative_total_cost']:.4f}")
    lines.append(f"Delay cost            : {metrics['delay_total_cost']:.4f}")
    lines.append("#" * 80)

    return "\n".join(lines)


def compute_sequential_baseline_metrics(
    y_true,
    y_pred,
    y_score,
    seq_lengths,
    detection_steps,
    fp_unit_cost=10.0,
    fn_unit_cost=20.0,
    delay_unit_cost=5.0,
):
    """
    Metrics for prefix-based sequential inference using the SAME trained PLELog.

    Sequence-level prediction:
      - predicted anomaly if any prefix raises an anomaly alert
      - predicted normal if no prefix raises an alert

    Early-detection metrics are computed only for correctly detected anomalies.
    EDR@25/50/75 = fraction of detected anomalies whose first alert occurs
    within the first 25/50/75% of the sequence. This matches the interpretation
    used for the RL early-detection comparison.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_score = np.asarray(y_score).astype(float)
    seq_lengths = np.asarray(seq_lengths).astype(float)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)

    auroc = None
    auprc = None
    if len(np.unique(y_true)) == 2:
        try:
            auroc = roc_auc_score(y_true, y_score)
        except Exception:
            auroc = None
        try:
            auprc = average_precision_score(y_true, y_score)
        except Exception:
            auprc = None

    total_anomalies = int(tp + fn)
    detected_anomalies = int(tp)
    detection_coverage = (
        detected_anomalies / total_anomalies
        if total_anomalies > 0 else 0.0
    )

    detected_steps = []
    detected_ratios = []

    for idx, (truth, pred) in enumerate(zip(y_true, y_pred)):
        if truth == 1 and pred == 1 and detection_steps[idx] is not None:
            step = int(detection_steps[idx])
            length = max(1.0, float(seq_lengths[idx]))
            detected_steps.append(step)
            detected_ratios.append(step / length)

    if detected_ratios:
        average_detection_step = float(np.mean(detected_steps))
        average_detection_ratio = float(np.mean(detected_ratios))
        median_detection_ratio = float(np.median(detected_ratios))

        # Denominator = correctly detected anomalies.
        edr_25 = float(np.mean(np.asarray(detected_ratios) <= 0.25))
        edr_50 = float(np.mean(np.asarray(detected_ratios) <= 0.50))
        edr_75 = float(np.mean(np.asarray(detected_ratios) <= 0.75))
    else:
        average_detection_step = None
        average_detection_ratio = None
        median_detection_ratio = None
        edr_25 = 0.0
        edr_50 = 0.0
        edr_75 = 0.0

    alert_rate = float(np.mean(y_pred == 1))

    fp_total_cost = float(fp * fp_unit_cost)
    fn_total_cost = float(fn * fn_unit_cost)

    # Delay is proportional to how much of the anomalous sequence had to be
    # observed before the first correct alert.
    delay_total_cost = float(
        sum(detected_ratios) * delay_unit_cost
    )

    total_cost = fp_total_cost + fn_total_cost + delay_total_cost
    average_cost_per_sequence = total_cost / max(1, len(y_true))

    return {
        "evaluation_mode": "prefix_sequential",
        "num_sequences": int(len(y_true)),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity_tnr": float(specificity),
        "f1_score": float(f1),
        "f2_score": float(f2),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "mcc": float(mcc),
        "auroc": None if auroc is None else float(auroc),
        "auprc": None if auprc is None else float(auprc),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "confusion_matrix": cm.tolist(),
        "total_anomalies": int(total_anomalies),
        "detected_anomalies": int(detected_anomalies),
        "anomaly_detection_coverage": float(detection_coverage),
        "average_detection_step": average_detection_step,
        "average_detection_ratio": average_detection_ratio,
        "median_detection_ratio": median_detection_ratio,
        "edr_25": float(edr_25),
        "edr_50": float(edr_50),
        "edr_75": float(edr_75),
        "average_reward": None,
        "alert_rate": float(alert_rate),
        "false_positive_unit_cost": float(fp_unit_cost),
        "false_negative_unit_cost": float(fn_unit_cost),
        "delay_unit_cost": float(delay_unit_cost),
        "false_positive_total_cost": float(fp_total_cost),
        "false_negative_total_cost": float(fn_total_cost),
        "delay_total_cost": float(delay_total_cost),
        "total_cost": float(total_cost),
        "average_cost_per_sequence": float(average_cost_per_sequence),
    }


def format_sequential_baseline_metrics(metrics, model_name="PLELog SEQUENTIAL"):
    avg_step_text = (
        "N/A"
        if metrics["average_detection_step"] is None
        else f"{metrics['average_detection_step']:.4f}"
    )
    avg_ratio_text = (
        "N/A"
        if metrics["average_detection_ratio"] is None
        else f"{metrics['average_detection_ratio']:.4f}"
    )
    median_ratio_text = (
        "N/A"
        if metrics["median_detection_ratio"] is None
        else f"{metrics['median_detection_ratio']:.4f}"
    )

    class_report_text = metrics.get(
        "classification_report_text",
        "Classification report was not saved."
    )

    lines = []
    lines.append("#" * 80)
    lines.append(f"{model_name}: PREFIX-BASED TEST METRICS")
    lines.append("#" * 80)
    lines.append(
        "During test, the trained model is applied to prefixes 1:t and "
        "the first anomaly alert is used."
    )
    lines.append(f"Number of sequences   : {metrics['num_sequences']}")
    lines.append("")

    lines.append("[Classification Metrics - Anomaly Class]")
    lines.append("Positive class        : 1 = anomaly")
    lines.append(f"Accuracy              : {metrics['accuracy']:.4f}")
    lines.append(f"Balanced accuracy     : {metrics['balanced_accuracy']:.4f}")
    lines.append(f"Precision             : {metrics['precision']:.4f}")
    lines.append(f"Recall / TPR          : {metrics['recall']:.4f}")
    lines.append(f"Specificity / TNR     : {metrics['specificity_tnr']:.4f}")
    lines.append(f"F1-score              : {metrics['f1_score']:.4f}")
    lines.append(f"F2-score              : {metrics['f2_score']:.4f}")
    lines.append(f"FPR                   : {metrics['fpr']:.4f}")
    lines.append(f"FNR                   : {metrics['fnr']:.4f}")
    lines.append(f"MCC                   : {metrics['mcc']:.4f}")
    lines.append(
        "AUROC                 : "
        + ("N/A" if metrics["auroc"] is None else f"{metrics['auroc']:.4f}")
    )
    lines.append(
        "AUPRC                 : "
        + ("N/A" if metrics["auprc"] is None else f"{metrics['auprc']:.4f}")
    )
    lines.append("")

    lines.append("[Classification Report - Class 0 and Class 1]")
    lines.append(class_report_text.rstrip())
    lines.append("")

    lines.append("[Confusion Matrix]")
    lines.append("Labels: 0=normal, 1=anomaly")
    lines.append(str(np.asarray(metrics["confusion_matrix"])))
    lines.append(
        f"TP={metrics['tp']} TN={metrics['tn']} "
        f"FP={metrics['fp']} FN={metrics['fn']}"
    )
    lines.append("")

    lines.append("[Early Detection Metrics]")
    lines.append(f"Total anomalies       : {metrics['total_anomalies']}")
    lines.append(f"Detected anomalies    : {metrics['detected_anomalies']}")
    lines.append(
        f"Detection coverage    : {metrics['anomaly_detection_coverage']:.4f}"
    )
    lines.append(f"Avg detection step    : {avg_step_text}")
    lines.append(f"Avg detection ratio   : {avg_ratio_text}")
    lines.append(f"Median detection ratio: {median_ratio_text}")
    lines.append(f"EDR@25                : {metrics['edr_25']:.4f}")
    lines.append(f"EDR@50                : {metrics['edr_50']:.4f}")
    lines.append(f"EDR@75                : {metrics['edr_75']:.4f}")
    lines.append(f"Alert rate            : {metrics['alert_rate']:.4f}")
    lines.append("")

    lines.append("[Cost-Sensitive Metrics]")
    lines.append(
        f"False-positive cost   : {metrics['false_positive_total_cost']:.4f}"
    )
    lines.append(
        f"False-negative cost   : {metrics['false_negative_total_cost']:.4f}"
    )
    lines.append(f"Delay cost            : {metrics['delay_total_cost']:.4f}")
    lines.append(f"Total cost            : {metrics['total_cost']:.4f}")
    lines.append(
        f"Avg cost / sequence   : {metrics['average_cost_per_sequence']:.4f}"
    )
    lines.append("#" * 80)
    return "\n".join(lines)


def evaluate_prefix_all_metrics(
    plelog,
    instances,
    encoder,
    scaler=None,
    transformer=None,
    threshold=0.5,
    fp_unit_cost=10.0,
    fn_unit_cost=20.0,
    delay_unit_cost=5.0,
    prefix_batch_size=32,
    json_save_path=None,
    txt_save_path=None,
    model_name="PLELog SEQUENTIAL",
):
    """
    Sequential prefix evaluation. The caller determines whether the model
    was trained with original full sequences or with early-aware prefixes.

    For each test sequence of length T:
        prefix 1
        prefix 1:2
        ...
        prefix 1:t
    are evaluated until the first anomaly prediction. If no prefix alerts,
    the sequence is predicted normal.

    Prefixes are evaluated in small batches for speed. Computing later prefixes
    inside the same inference batch does not affect the first-alert decision;
    the model has no online state update between prefixes.
    """
    y_true = []
    y_pred = []
    y_score = []
    seq_lengths = []
    detection_steps = []

    plelog.model.eval()

    seq_bar = tqdm(
        instances,
        total=len(instances),
        desc="PLELog sequential prefix evaluation",
        unit="sequence",
        dynamic_ncols=True
    )

    with torch.no_grad():
        for seq_idx, inst in enumerate(seq_bar):
            sequence = list(inst.sequence)
            T = len(sequence)

            true_binary = 0 if inst.label == "Normal" else 1
            first_alert_step = None
            max_score_seen = 0.0

            if T > 0:
                for start_t in range(1, T + 1, prefix_batch_size):
                    end_t = min(T, start_t + prefix_batch_size - 1)

                    prefix_instances = []
                    prefix_steps = []

                    for t in range(start_t, end_t + 1):
                        prefix_inst = copy.copy(inst)
                        prefix_inst.sequence = sequence[:t]
                        prefix_instances.append(prefix_inst)
                        prefix_steps.append(t)

                    # Recompute representation from only the currently
                    # observed prefix. No future events enter this representation.
                    prefix_presented = encoder.present(prefix_instances)
                    prefix_reprs = np.asarray(prefix_presented)

                    # Reuse preprocessing fitted on TRAIN only.
                    if scaler is not None:
                        prefix_reprs = scaler.transform(prefix_reprs)
                        prefix_reprs = np.nan_to_num(
                            prefix_reprs,
                            nan=0.0,
                            posinf=1e6,
                            neginf=-1e6
                        )

                    if transformer is not None:
                        prefix_reprs = transformer.transform(prefix_reprs)

                    for pinst, vec in zip(prefix_instances, prefix_reprs):
                        pinst.repr = vec

                    tinst = generate_tinsts_binary_label(
                        prefix_instances,
                        plelog.vocab,
                        False
                    )
                    tinst.to_device(device)

                    pred_tags, tag_logits = plelog.predict(
                        tinst.inputs,
                        threshold
                    )

                    probs = tag_logits.detach().cpu().numpy()
                    anomaly_probs = probs[:, plelog.anomaly_id]
                    pred_tags = np.asarray(pred_tags).astype(int)

                    # Find the first alert in chronological order.
                    for local_idx, t in enumerate(prefix_steps):
                        prob = float(anomaly_probs[local_idx])
                        max_score_seen = max(max_score_seen, prob)

                        if pred_tags[local_idx] == plelog.anomaly_id:
                            first_alert_step = int(t)
                            break

                    seq_bar.set_postfix(
                        current=f"{min(end_t, T)}/{T}",
                        label=inst.label,
                        alert=(
                            "-" if first_alert_step is None
                            else first_alert_step
                        )
                    )

                    if first_alert_step is not None:
                        break

            pred_binary = 1 if first_alert_step is not None else 0

            y_true.append(true_binary)
            y_pred.append(pred_binary)
            y_score.append(float(max_score_seen))
            seq_lengths.append(max(1, T))
            detection_steps.append(first_alert_step)

    metrics = compute_sequential_baseline_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        seq_lengths=seq_lengths,
        detection_steps=detection_steps,
        fp_unit_cost=fp_unit_cost,
        fn_unit_cost=fn_unit_cost,
        delay_unit_cost=delay_unit_cost,
    )

    class_report_text = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Normal (0)", "Anomaly (1)"],
        digits=4,
        zero_division=0
    )
    metrics["classification_report_text"] = class_report_text

    report_text = format_sequential_baseline_metrics(
        metrics,
        model_name=model_name
    )

    print()
    print(report_text)

    if json_save_path is not None:
        os.makedirs(os.path.dirname(json_save_path), exist_ok=True)
        with open(json_save_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved sequential JSON metrics to: {json_save_path}")

    if txt_save_path is not None:
        os.makedirs(os.path.dirname(txt_save_path), exist_ok=True)
        with open(txt_save_path, "w", encoding="utf-8") as f:
            f.write(report_text)
            f.write("\n")
        print(f"Saved sequential text metrics to: {txt_save_path}")

    return metrics


def save_standard_vs_sequential_comparison(
    standard_metrics,
    sequential_metrics,
    json_path,
    txt_path,
):
    """Save one comparison report for the same DEV-selected checkpoint."""
    comparison = {
        "checkpoint_selection": "best checkpoint selected using DEV F1",
        "standard_full_sequence": standard_metrics,
        "sequential_prefix_inference": sequential_metrics,
    }

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    keys = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("specificity_tnr", "Specificity"),
        ("f1_score", "F1"),
        ("f2_score", "F2"),
        ("fpr", "FPR"),
        ("fnr", "FNR"),
        ("mcc", "MCC"),
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("anomaly_detection_coverage", "Detection Coverage"),
        ("average_detection_step", "Avg Detection Step"),
        ("average_detection_ratio", "Avg Detection Ratio"),
        ("median_detection_ratio", "Median Detection Ratio"),
        ("edr_25", "EDR@25"),
        ("edr_50", "EDR@50"),
        ("edr_75", "EDR@75"),
        ("alert_rate", "Alert Rate"),
        ("false_positive_total_cost", "False-positive Cost"),
        ("false_negative_total_cost", "False-negative Cost"),
        ("delay_total_cost", "Delay Cost"),
        ("total_cost", "Total Cost"),
        ("average_cost_per_sequence", "Avg Cost / Sequence"),
    ]

    def fmt(v):
        if v is None:
            return "N/A"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return f"{float(v):.6f}"

    lines = []
    lines.append("=" * 92)
    lines.append("PLELog-Early BEST CHECKPOINT: FULL-SEQUENCE vs SEQUENTIAL PREFIX INFERENCE")
    lines.append("Checkpoint selected using DEV F1; the same PLELog-Early checkpoint is used for both evaluations.")
    lines.append("=" * 92)
    lines.append(
        f"{'Metric':<28} | {'Standard':>20} | {'Sequential':>20}"
    )
    lines.append("-" * 92)

    for key, label in keys:
        lines.append(
            f"{label:<28} | "
            f"{fmt(standard_metrics.get(key)):>20} | "
            f"{fmt(sequential_metrics.get(key)):>20}"
        )

    lines.append("=" * 92)
    report = "\n".join(lines)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n")

    print()
    print(report)
    print(f"Saved comparison JSON to: {json_path}")
    print(f"Saved comparison TXT to: {txt_path}")


class PLELog:
    _logger = logging.getLogger('PLELog')
    _logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))
    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'PLELog.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))
    _logger.addHandler(console_handler)
    _logger.addHandler(file_handler)
    _logger.info(
        'Construct logger for PLELog succeeded, current working directory: %s, logs will be written in %s' %
        (os.getcwd(), LOG_ROOT))

    @property
    def logger(self):
        return PLELog._logger

    def __init__(self, vocab, num_layer, hidden_size, label2id):
        super(PLELog, self).__init__()

        # ---------------- Labels ----------------
        self.label2id = label2id

        # Required labels check
        required_labels = {'Normal', 'Anomaly'}
        if not required_labels.issubset(label2id.keys()):
            raise ValueError(f"label2id must contain {required_labels}, got {label2id}")

        self.anomaly_id = label2id['Anomaly']
        self.id2tag = {v: k for k, v in label2id.items()}

        # ---------------- Model config ----------------
        self.vocab = vocab
        self.num_layer = num_layer
        self.hidden_size = hidden_size

        self.batch_size = 128
        self.test_batch_size = 1024

        # ---------------- Model ----------------
        self.model = AttGRUModel(vocab, self.num_layer, self.hidden_size)
        #self.model = self.model.to(device) # for CPU
        if torch.cuda.is_available():
            self.model = self.model.cuda(device)
        else: # new
            self.model = self.model.to(device)


        # ---------------- Loss ----------------
        # NOTE: model outputs probabilities after softmax
        self.loss = nn.BCELoss()

    def forward(self, inputs, targets):
        tag_logits = self.model(inputs)
        tag_logits = F.softmax(tag_logits, dim=1)
        loss = self.loss(tag_logits, targets)
        return loss

    def predict(self, inputs, threshold=None):
        with torch.no_grad():
            tag_logits = self.model(inputs)
            tag_logits = F.softmax(tag_logits, dim=1)

        anomaly_id = self.anomaly_id

        if threshold is not None:
            probs = tag_logits.detach().cpu().numpy()
            pred_tags = np.zeros(probs.shape[0], dtype=int)

            for i, logits in enumerate(probs):
                if logits[anomaly_id] >= threshold:
                    pred_tags[i] = anomaly_id
                else:
                    pred_tags[i] = 1 - anomaly_id
        else:
            pred_tags = tag_logits.detach().max(1)[1].cpu().numpy()

        return pred_tags, tag_logits

        '''
        with torch.no_grad():
            tag_logits = self.model(inputs)
            tag_logits = F.softmax(tag_logits)
        if threshold is not None:
            probs = tag_logits.detach().cpu().numpy()
            anomaly_id = self.label2id['Anomaly']
            pred_tags = np.zeros(probs.shape[0])
            for i, logits in enumerate(probs):
                if logits[anomaly_id] >= threshold:
                    pred_tags[i] = anomaly_id
                else:
                    pred_tags[i] = 1 - anomaly_id

        else:
            pred_tags = tag_logits.detach().max(1)[1].cpu()
        return pred_tags, tag_logits
        '''

    def evaluate(self, instances, threshold=0.5):
        self.logger.info('Start evaluating by threshold %.3f' % threshold)

        with torch.no_grad():
            self.model.eval()
            TP, TN, FP, FN = 0, 0, 0, 0

            for onebatch in data_iter(instances, self.test_batch_size, False):
                tinst = generate_tinsts_binary_label(onebatch, self.vocab, False)
                tinst.to_device(device)  # only CPU
                #tinst.to_cuda(device)


                pred_tags, tag_logits = self.predict(tinst.inputs, threshold)

                for inst, bmatch in batch_variable_inst(onebatch, pred_tags, tag_logits, self.id2tag):

                    if bmatch:
                        if inst.label == 'Normal':
                            TN += 1
                        else:
                            TP += 1
                    else:
                        if inst.label == 'Normal':
                            FP += 1
                        else:
                            FN += 1

            self.logger.info('TP: %d, TN: %d, FN: %d, FP: %d' % (TP, TN, FN, FP))

            if TP + FP > 0:
                precision = 100 * TP / (TP + FP)
                recall = 100 * TP / (TP + FN)
                f = 2 * precision * recall / (precision + recall)
            else:
                precision = recall = f = 0

        return precision, recall, f


    def evaluate_all_metrics(
        self,
        instances,
        threshold=0.5,
        fp_unit_cost=10.0,
        fn_unit_cost=20.0,
        delay_unit_cost=5.0,
        json_save_path=None,
        txt_save_path=None,
        model_name="PLELog",
    ):
        """
        Evaluate PLELog with all available test metrics and save them.

        Since PLELog is a static full-sequence classifier, early-detection values
        use the default full-sequence assumption:
            detection ratio = 1.0 for detected anomalies
            EDR@25/50/75 = 0.0
            delay cost = TP * delay_unit_cost
        """
        self.logger.info('Start full metric evaluation by threshold %.3f' % threshold)

        y_true = []
        y_pred = []
        y_score = []
        seq_lengths = []

        with torch.no_grad():
            self.model.eval()

            for onebatch in data_iter(instances, self.test_batch_size, False):
                tinst = generate_tinsts_binary_label(onebatch, self.vocab, False)
                tinst.to_device(device)

                pred_tags, tag_logits = self.predict(tinst.inputs, threshold)
                probs = tag_logits.detach().cpu().numpy()
                anomaly_probs = probs[:, self.anomaly_id]

                for idx, inst in enumerate(onebatch):
                    true_binary = 0 if inst.label == "Normal" else 1
                    pred_class = int(pred_tags[idx])
                    pred_binary = 1 if pred_class == self.anomaly_id else 0

                    y_true.append(true_binary)
                    y_pred.append(pred_binary)
                    y_score.append(float(anomaly_probs[idx]))

                    if hasattr(inst, "sequence"):
                        seq_lengths.append(len(inst.sequence))
                    else:
                        seq_lengths.append(1)

        metrics = compute_static_baseline_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
            seq_lengths=seq_lengths,
            fp_unit_cost=fp_unit_cost,
            fn_unit_cost=fn_unit_cost,
            delay_unit_cost=delay_unit_cost,
        )

        class_report_text = classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Normal (0)", "Anomaly (1)"],
            digits=4,
            zero_division=0
        )

        metrics["classification_report_text"] = class_report_text

        print()
        print("=" * 70)
        print(f"{model_name}: CLASSIFICATION REPORT")
        print("=" * 70)
        print(class_report_text)

        report_text = format_static_baseline_metrics(metrics, model_name=model_name)
        print(report_text)

        if json_save_path is not None:
            os.makedirs(os.path.dirname(json_save_path), exist_ok=True)
            with open(json_save_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            print(f"Saved JSON metrics to: {json_save_path}")

        if txt_save_path is not None:
            os.makedirs(os.path.dirname(txt_save_path), exist_ok=True)
            with open(txt_save_path, "w", encoding="utf-8") as f:
                f.write(report_text)
                f.write("\n")
            print(f"Saved text metrics to: {txt_save_path}")

        return metrics

    def clear_folder(folder_path):
        if not os.path.exists(folder_path):
            print(f"Folder does not exist: {folder_path}")
            return

        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)

            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)

            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

        print(f"Cleared all contents of: {folder_path}")


@contextmanager
def blocking_stage(description, expected_seconds=120):
    """
    Show a percentage bar for a blocking operation whose internal row count
    is not exposed by this script (for example PKLPreprocessor.load_pkl).

    IMPORTANT:
    - 0..95% is an ESTIMATED activity percentage based on elapsed time.
    - 100% is shown only when the blocking operation actually returns.
    - Exact row-level percentages are used elsewhere whenever the iterable
      length is known.
    """
    stop_event = threading.Event()
    start = time.time()

    bar = tqdm(
        total=100,
        initial=0,
        desc=description,
        unit="%",
        dynamic_ncols=True,
        leave=True
    )

    def _pulse():
        last_pct = 0
        while not stop_event.wait(1.0):
            elapsed = time.time() - start

            # Smooth estimated progress that approaches, but never exceeds, 95%
            # until the operation really completes.
            estimated_pct = int(
                min(95, 95 * (1.0 - pow(2.718281828, -elapsed / max(1.0, expected_seconds))))
            )

            if estimated_pct > last_pct:
                bar.update(estimated_pct - last_pct)
                last_pct = estimated_pct

            bar.set_postfix_str(
                f"estimated | elapsed={elapsed:.0f}s",
                refresh=True
            )

    pulse_thread = threading.Thread(target=_pulse, daemon=True)
    pulse_thread.start()

    try:
        yield
    finally:
        stop_event.set()
        pulse_thread.join(timeout=2.0)

        # Only mark 100% after the operation has genuinely completed.
        remaining = 100 - bar.n
        if remaining > 0:
            bar.update(remaining)

        elapsed = time.time() - start
        bar.set_postfix_str(
            f"completed | elapsed={elapsed:.0f}s",
            refresh=True
        )
        bar.close()
        print(f"{description} finished in {elapsed:.2f} s")


if __name__ == '__main__':
    print('start main function ......')

    RESET = colorama.Fore.RESET

    # ---------------- Device setup (CPU ONLY) ----------------
    #device = torch.device("cpu")
    #torch.backends.cudnn.enabled = False
    #torch.backends.cuda.enabled = False
    #print(f"Using device: CPU only{RESET}")
    # Automatically select GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Enable cuDNN for GPU acceleration
    torch.backends.cudnn.enabled = True
    print(f"Using device-------------------xxxxxxxxxxxxxx****-------------------: {device}")


    # ---------------- Arguments ----------------
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', default='HDFS', type=str)  # BGL, HDFS, TH_1G, SP_150MB
    argparser.add_argument('--mode', default='train', type=str)
    argparser.add_argument('--parser', default='IBM', type=str)
    argparser.add_argument('--min_cluster_size', type=int, default=100)
    argparser.add_argument('--min_samples', type=int, default=100)
    argparser.add_argument('--reduce_dimension', type=int, default=100)
    argparser.add_argument('--threshold', type=float, default=0.5)
    argparser.add_argument('--case', default='in_domain', type=str, choices=['in_domain', 'cross_dataset'])
    argparser.add_argument(
        '--variant',
        default='early',
        type=str,
        choices=['early'],
        help=(
            "standalone PLELog-Early run"
        )
    )
    argparser.add_argument(
        '--prefix_batch_size',
        default=128,
        type=int,
        help=(
            "Number of prefixes from ONE sequence evaluated together. "
            "128 is fast and exact for 120-event windows; lower to 64/32 if memory is tight."
        )
    )
    argparser.add_argument(
        '--data_root',
        default='../../LWADLS/datasets',
        type=str,
        help='Root folder for reading PKL files, e.g., ../../LWADLS/datasets'
    )
    args, _ = argparser.parse_known_args()

    dataset = args.dataset
    parser = args.parser
    mode = args.mode
    min_cluster_size = args.min_cluster_size
    min_samples = args.min_samples
    reduce_dimension = args.reduce_dimension
    threshold = args.threshold
    case = args.case
    data_root = args.data_root
    prefix_batch_size = max(1, int(args.prefix_batch_size))
    variant = 'early'

    # ---------------- Paths ----------------
    def find_project_root():
        """
        Find the PLELog project root automatically.
        This works from your actual structure:
            ~/PLELog/LICENSE
            ~/PLELog/datasets
            ~/PLELog/models
            ~/PLELog/preprocessing
        """
        candidates = []

        cwd = os.path.abspath(os.getcwd())
        candidates.append(cwd)
        parent = cwd
        for _ in range(8):
            parent = os.path.dirname(parent)
            candidates.append(parent)

        script_dir = os.path.abspath(os.path.dirname(__file__))
        candidates.append(script_dir)
        parent = script_dir
        for _ in range(8):
            parent = os.path.dirname(parent)
            candidates.append(parent)

        for cand in candidates:
            if (
                os.path.isdir(os.path.join(cand, "datasets"))
                and os.path.isdir(os.path.join(cand, "models"))
                and os.path.isdir(os.path.join(cand, "preprocessing"))
            ):
                return cand

        return cwd

    PROJECT_ROOT = find_project_root()

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    DATASETS_ROOT = os.path.join(PROJECT_ROOT, "datasets")

    # Save every dataset output inside its own dataset folder:
    # datasets/BGL/PLELog_results/BGL_IBM/
    # datasets/HDFS/PLELog_results/HDFS_IBM/
    # datasets/SP_150MB/PLELog_results/SP_150MB_IBM/
    # datasets/TH_1G/PLELog_results/TH_1G_IBM/
    results_root = os.path.join(DATASETS_ROOT, dataset, "PLELog_results")

    if variant == "standard":
        save_dir = os.path.join(results_root, "PLELog_Standard")
    elif variant == "early":
        save_dir = os.path.join(results_root, "PLELog_Early")
    else:
        save_dir = os.path.join(results_root, "comparison")

    exp_dir = os.path.join(save_dir, f"{dataset}_{parser}")

    output_model_dir = os.path.join(exp_dir, "model")
    prob_label_res_dir = os.path.join(exp_dir, "prob_label_res")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(output_model_dir, exist_ok=True)
    os.makedirs(prob_label_res_dir, exist_ok=True)

    print("\n==============================")
    print("PLELog resolved paths")
    print("==============================")
    print("Project root:", PROJECT_ROOT)
    print("Datasets root:", DATASETS_ROOT)
    print("Dataset:", dataset)
    print("Save directory:", os.path.abspath(save_dir))
    print("Experiment directory:", os.path.abspath(exp_dir))
    print("Model directory:", os.path.abspath(output_model_dir))
    print("Metrics JSON will be saved in:", os.path.abspath(exp_dir))
    print("Metrics TXT will be saved in:", os.path.abspath(os.path.join(DATASETS_ROOT, dataset)))
    print("==============================\n")

    # ============================================================
    # BOTH mode = two independent experiments
    # ============================================================
    if False:  # standalone file: no subprocess orchestration
        print("\n" + "=" * 92)
        print("PLELog comparison: STANDARD vs EARLY")
        print("Each method is trained independently in its own output directory.")
        print("=" * 92)

        common_cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--dataset", str(dataset),
            "--mode", str(mode),
            "--parser", str(parser),
            "--min_cluster_size", str(min_cluster_size),
            "--min_samples", str(min_samples),
            "--reduce_dimension", str(reduce_dimension),
            "--threshold", str(threshold),
            "--case", str(case),
            "--data_root", str(data_root),
        ]

        print("\n[1/2] Running PLELog-Standard ...")
        subprocess.run(common_cmd + ["--variant", "standard"], check=True)

        print("\n[2/2] Running PLELog-Early ...")
        subprocess.run(common_cmd + ["--variant", "early"], check=True)

        standard_json = os.path.join(
            results_root,
            "PLELog_Standard",
            f"{dataset}_{parser}",
            "PLELog_Standard_metrics.json",
        )
        early_json = os.path.join(
            results_root,
            "PLELog_Early",
            f"{dataset}_{parser}",
            "PLELog_Early_metrics.json",
        )

        with open(standard_json, "r", encoding="utf-8") as f:
            standard_metrics = json.load(f)
        with open(early_json, "r", encoding="utf-8") as f:
            early_metrics = json.load(f)

        comparison_dir = os.path.join(results_root, "comparison")
        os.makedirs(comparison_dir, exist_ok=True)

        comparison_json = os.path.join(
            comparison_dir,
            f"{dataset}_{parser}_PLELog_Standard_vs_Early.json"
        )
        comparison_txt = os.path.join(
            comparison_dir,
            f"{dataset}_{parser}_PLELog_Standard_vs_Early.txt"
        )

        comparison = {
            "PLELog-Standard": standard_metrics,
            "PLELog-Early": early_metrics,
        }
        with open(comparison_json, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)

        columns = [
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1_score", "F1"),
            ("mcc", "MCC"),
            ("auroc", "AUROC"),
            ("auprc", "AUPRC"),
            ("anomaly_detection_coverage", "Detection Coverage"),
            ("average_detection_ratio", "Avg Det. Ratio"),
            ("edr_25", "EDR@25"),
            ("edr_50", "EDR@50"),
            ("edr_75", "EDR@75"),
            ("false_positive_total_cost", "False-positive Cost"),
            ("false_negative_total_cost", "False-negative Cost"),
            ("delay_total_cost", "Delay Cost"),
            ("total_cost", "Total Cost"),
        ]

        def _fmt_metric(v):
            if v is None:
                return "N/A"
            if isinstance(v, int):
                return str(v)
            return f"{float(v):.4f}"

        rows = [
            ("PLELog-Standard", standard_metrics),
            ("PLELog-Early", early_metrics),
        ]

        lines = []
        lines.append("=" * 72)
        lines.append("FINAL RESULTS: PLELog-Standard vs PLELog-Early")
        lines.append("=" * 72)
        lines.append(f"{'Metric':<28} | {'PLELog-Standard':>18} | {'PLELog-Early':>18}")
        lines.append("-" * 72)
        for key, label in columns:
            lines.append(
                f"{label:<28} | "
                f"{_fmt_metric(standard_metrics.get(key)):>18} | "
                f"{_fmt_metric(early_metrics.get(key)):>18}"
            )
        lines.append("=" * 72)
        table_text = "\n".join(lines)

        with open(comparison_txt, "w", encoding="utf-8") as f:
            f.write(table_text + "\n")

        print("\n" + table_text)
        print(f"\nSaved comparison JSON: {comparison_json}")
        print(f"Saved comparison TXT : {comparison_txt}")
        print("\nAll Finished ✅")
        sys.exit(0)

    # For child runs, choose the training protocol explicitly.
    EARLY_TRAINING_ENABLED = (variant == "early")

    # ---------------- Load PKL ----------------
    # first paper :
    #train_pkl = f'../datasets/{dataset}/1_{dataset}_Splitted_Datasets/train_df.pkl'
    #dev_pkl   = f'../datasets/{dataset}/1_{dataset}_Splitted_Datasets/val_df.pkl'
    #test_pkl  = f'../datasets/{dataset}/1_{dataset}_Splitted_Datasets/test_df.pkl'



    # second paper
    #train_pkl = '../../NovaAD_Plus/datasets/SP_150MB_ratio/3_SP_150MB_ratio_Splitted_Datasets/3_SP_150MB_ratio_train_df.pkl'
    #test_pkl = '../../NovaAD_Plus/datasets/SP_150MB_ratio/3_SP_150MB_ratio_Splitted_Datasets/3_SP_150MB_ratio_test_df.pkl'
    #dev_pkl = '../../NovaAD_Plus/datasets/SP_150MB_ratio/3_SP_150MB_ratio_Splitted_Datasets/3_SP_150MB_ratio_val_df.pkl'


    #============================================++++++++++++++++++++++++++++++++++++++++++

    # Change this import path if your PKLPreprocessor file has a different name

    # ============================================================
    # 1. Dataset paths
    # ============================================================
    def resolve_data_root(raw_data_root):
        """
        Resolve the input PKL data root.

        Default reading style requested by the user:
            ../../LWADLS/datasets

        Example generated path:
            ../../LWADLS/datasets/SP_150MB_ratio/3_SP_150MB_ratio_Splitted_Datasets/3_SP_150MB_ratio_train_df.pkl

        The function keeps the requested relative style, but also checks common
        absolute locations so the script can be run from the PLELog root.
        """
        candidates = []

        # 1) exactly as provided, relative to current working directory
        candidates.append(os.path.abspath(raw_data_root))

        # 2) relative to this script location
        script_dir = os.path.abspath(os.path.dirname(__file__))
        candidates.append(os.path.abspath(os.path.join(script_dir, raw_data_root)))

        # 3) common case: LWADLS is next to PLELog
        candidates.append(os.path.abspath(os.path.join(PROJECT_ROOT, "..", "LWADLS", "datasets")))

        # 4) common case if PLELog is inside another folder and ../../LWADLS is correct
        candidates.append(os.path.abspath(os.path.join(PROJECT_ROOT, "..", "..", "LWADLS", "datasets")))

        # 5) local fallback, useful if files are copied into PLELog/datasets
        candidates.append(DATASETS_ROOT)

        seen = set()
        unique_candidates = []
        for cand in candidates:
            if cand not in seen:
                unique_candidates.append(cand)
                seen.add(cand)

        for cand in unique_candidates:
            if os.path.isdir(cand):
                return cand, unique_candidates

        # Return first candidate even if missing, so error message is clear.
        return unique_candidates[0], unique_candidates


    READ_DATASETS_ROOT, data_root_candidates = resolve_data_root(data_root)

    print("\n==============================")
    print("Dataset input root candidates")
    print("==============================")
    print("Requested data_root:", data_root)
    for cand in data_root_candidates:
        print(f"candidate: {cand} | exists={os.path.isdir(cand)}")
    print("Selected read root:", READ_DATASETS_ROOT)
    print("==============================\n")


    def make_dataset_paths(dataset_name, base_dir=None):
        """
        Build dataset paths using the LWADLS folder structure.

        Required format:
            ../../LWADLS/datasets/{DATASET}/3_{DATASET}_Splitted_Datasets/3_{DATASET}_train_df.pkl
            ../../LWADLS/datasets/{DATASET}/3_{DATASET}_Splitted_Datasets/3_{DATASET}_val_df.pkl
            ../../LWADLS/datasets/{DATASET}/3_{DATASET}_Splitted_Datasets/3_{DATASET}_test_df.pkl

        Example:
            ../../LWADLS/datasets/SP_150MB_ratio/3_SP_150MB_ratio_Splitted_Datasets/3_SP_150MB_ratio_train_df.pkl
        """
        if base_dir is None:
            base_dir = READ_DATASETS_ROOT

        split_folder = f"1_{dataset_name}_Splitted_Datasets"

        return {
            "train_pkl": os.path.join(
                base_dir,
                dataset_name,
                split_folder,
                f"train_df.pkl"
            ),
            "dev_pkl": os.path.join(
                base_dir,
                dataset_name,
                split_folder,
                f"val_df.pkl"
            ),
            "test_pkl": os.path.join(
                base_dir,
                dataset_name,
                split_folder,
                f"test_df.pkl"
            ),
        }


    # Datasets supported by the experiment. Output is saved in local PLELog/datasets/{dataset},
    # but input PKL files are read from READ_DATASETS_ROOT using the 3_* LWADLS structure.
    SUPPORTED_DATASETS = [
        "BGL",
        "HDFS",
        "TH_1G",
        "SP_150MB",
        "SP_150MB_ratio",
    ]

    DATASETS = {
        name: make_dataset_paths(name)
        for name in SUPPORTED_DATASETS
    }

    if dataset not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Supported datasets: {SUPPORTED_DATASETS}"
        )

    print("\n==============================")
    print("Dataset PKL paths")
    print("==============================")
    missing_paths = []
    for split_name, split_path in DATASETS[dataset].items():
        exists = os.path.exists(split_path)
        print(f"{split_name}: {os.path.abspath(split_path)} | exists={exists}")
        if not exists:
            missing_paths.append(split_path)
    print("==============================\n")

    if missing_paths:
        raise FileNotFoundError(
            "Missing required PKL files for dataset " + dataset + ":\n"
            + "\n".join(os.path.abspath(p) for p in missing_paths)
            + "\n\nUse --data_root to point to your LWADLS datasets folder. Example:\n"
            + "python PLELog_full_updated.py --dataset SP_150MB_ratio --data_root ../../LWADLS/datasets"
        )

    # ============================================================
    # 2. Settings to change
    # ============================================================

    CASE = case

    # For in-domain, this uses the same dataset variable from args.dataset
    TARGET_DATASET = dataset

    # For cross-dataset only
    #SOURCE_DATASETS = ["TH_1G"]
    #TARGET_TRAIN_FRACTION = 0.20
    RANDOM_SEED = 42

    # ============================================================
    # 3. Create processor
    # ============================================================

    random.seed(RANDOM_SEED)

    processor = PKLPreprocessor()


    # ============================================================
    # 4. Run selected case
    # ============================================================

    if CASE == "in_domain":

        paths = DATASETS[TARGET_DATASET]

        print("\n==============================")
        print("In-domain experiment")
        print("==============================")
        print("Dataset:", TARGET_DATASET)
        print("[1/6] Loading and preprocessing train/dev/test ...")

        # Minimal change:
        # show progress/status while PKLPreprocessor.load_pkl() is running.
        # The percentage here is estimated because load_pkl() does not expose
        # its internal row counter. Later sequence/batch progress bars are exact.
        with blocking_stage(
            f"In-domain {TARGET_DATASET}: load/preprocess [estimated %]",
            expected_seconds=120
        ):
            train, dev, test = processor.load_pkl(
                TARGET_DATASET,
                paths["train_pkl"],
                paths["dev_pkl"],
                paths["test_pkl"]
            )

        print(f"Loaded {len(train)} train / {len(dev)} dev / {len(test)} test")

        # ------------------------------------------------------------
        # RESOURCE-SAFE EARLY TRAINING:
        # Keep the ORIGINAL full-sequence TRAIN set through:
        #   representations -> scaler/ICA -> probabilistic labeling (HDBSCAN)
        # Prefix samples are added only AFTER probabilistic labeling.
        # ------------------------------------------------------------
        original_train_size_before_early = len(train)
        print(
            "[EARLY-TRAIN] Prefix augmentation is delayed until AFTER "
            "PLELog probabilistic labeling."
        )
        print(
            f"[EARLY-TRAIN] HDBSCAN will see only the original "
            f"{original_train_size_before_early} full training sequences."
        )

        print("[2/6] Building event IDs and sequence representations ...")
        print("      Progress bars below will show exact sequence counts where available.")


    elif CASE == "cross_dataset":

        all_source_train = []
        all_source_dev = []

        print("\n==============================")
        print("Cross-dataset experiment")
        print("==============================")
        print("Sources:", SOURCE_DATASETS)
        print("Target:", TARGET_DATASET)
        print("Target train fraction (Normal + Anomaly):", TARGET_TRAIN_FRACTION)

        # ------------------------------------------------------------
        # Load source datasets
        # ------------------------------------------------------------

        for source_dataset in tqdm(
            SOURCE_DATASETS,
            desc="Loading source datasets",
            unit="dataset",
            dynamic_ncols=True
        ):
            source_paths = DATASETS[source_dataset]

            # IMPORTANT:
            # Source TEST is NOT needed in cross-domain training/evaluation.
            #
            # PKLPreprocessor.load_pkl() currently requires three file paths.
            # To avoid opening/preprocessing the real source test.pkl, pass
            # source DEV as the third (dummy) argument and discard that result.
            #
            # This keeps compatibility with the existing PKLPreprocessor while
            # completely avoiding the source test file.
            with blocking_stage(
                f"Load source {source_dataset} train/dev [estimated %]",
                expected_seconds=120
            ):
                source_train, source_dev, _unused_dev_copy = processor.load_pkl(
                    source_dataset,
                    source_paths["train_pkl"],
                    source_paths["dev_pkl"],
                    source_paths["dev_pkl"]   # dummy third split; source test NOT loaded
                )

            # Release the duplicate dev result immediately.
            del _unused_dev_copy

            all_source_train.extend(source_train)
            all_source_dev.extend(source_dev)

            print("\nLoaded source dataset:", source_dataset)
            print("Source train:", len(source_train))
            print("Source dev:", len(source_dev))
            print("Source test: NOT LOADED")

        # ------------------------------------------------------------
        # Load target dataset
        # ------------------------------------------------------------

        target_paths = DATASETS[TARGET_DATASET]

        # Target TEST is required for the final prediction/evaluation.
        # It is never added to the training set.
        with blocking_stage(
            f"Load target {TARGET_DATASET} train/dev/test [estimated %]",
            expected_seconds=120
        ):
            target_train, target_dev, target_test = processor.load_pkl(
                TARGET_DATASET,
                target_paths["train_pkl"],
                target_paths["dev_pkl"],
                target_paths["test_pkl"]
            )

        # ------------------------------------------------------------
        # Take 20% of target train INCLUDING Normal and Anomaly
        # Stratified by class so the target sample preserves the
        # original target-training class distribution.
        #
        # Important:
        #   20% of Normal + 20% of Anomaly = about 20% of the
        #   COMPLETE target training set, NOT 40%.
        # ------------------------------------------------------------

        target_normal_train = []
        target_anomaly_train = []

        for inst in tqdm(
            target_train,
            total=len(target_train),
            desc="Scanning target labels",
            unit="sequence",
            dynamic_ncols=True
        ):
            if inst.label == "Normal":
                target_normal_train.append(inst)
            elif inst.label == "Anomaly":
                target_anomaly_train.append(inst)

        number_normal_to_take = int(
            len(target_normal_train) * TARGET_TRAIN_FRACTION
        )
        number_anomaly_to_take = int(
            len(target_anomaly_train) * TARGET_TRAIN_FRACTION
        )

        target_normal_sample = random.sample(
            target_normal_train,
            number_normal_to_take
        ) if number_normal_to_take > 0 else []

        target_anomaly_sample = random.sample(
            target_anomaly_train,
            number_anomaly_to_take
        ) if number_anomaly_to_take > 0 else []

        target_train_sample = target_normal_sample + target_anomaly_sample
        random.shuffle(target_train_sample)

        # ------------------------------------------------------------
        # Final data
        # ------------------------------------------------------------

        train = all_source_train + target_train_sample

        dev = all_source_dev

        # Important:
        # target test is untouched
        test = target_test

        print("\nCross-domain protocol:")
        print("  Source train     : USED")
        print("  Source dev       : USED")
        print("  Source test      : NOT LOADED")
        print("  Target train     : 20% stratified Normal + Anomaly")
        print("  Target test      : FINAL prediction/evaluation only")

        print("\n==============================")
        print("Final cross-dataset data")
        print("==============================")
        print("Source train:", len(all_source_train))
        print("Source dev:", len(all_source_dev))
        print("Target train:", len(target_train))
        print("Target normal train:", len(target_normal_train))
        print("Target anomaly train:", len(target_anomaly_train))
        print("Target normal used:", len(target_normal_sample))
        print("Target anomaly used:", len(target_anomaly_sample))
        print("Target total used:", len(target_train_sample))
        print(
            "Actual target fraction used:",
            f"{len(target_train_sample) / max(1, len(target_train)):.4f}"
        )
        print("Final train:", len(train))
        print("Final dev:", len(dev))
        print("Target test untouched:", len(test))


    else:
        raise ValueError("CASE must be either 'in_domain' or 'cross_dataset'")

    # ============================================================
    # 5. Final print
    # ============================================================

    print("\n==============================")
    print("Ready for training")
    print("==============================")
    print(f"Loaded {len(train)} train / {len(dev)} dev / {len(test)} test")
    if CASE == "in_domain" and EARLY_TRAINING_ENABLED:
        print(
            "Training protocol: original PLELog preprocessing/HDBSCAN first; "
            "safe prefix-aware supervised augmentation after labeling"
        )
    #======================================================++++++++++++++++++++++++++++++++++++++++


    # use in firt and second paper
    #processor = PKLPreprocessor()
    #train, dev, test = processor.load_pkl(dataset, train_pkl, dev_pkl, test_pkl)


    print(f"Loaded {len(train)} train / {len(dev)} dev / {len(test)} test")

    print("\n[2/6] Embeddings / event-ID preparation")
    # ---------------- Embeddings ----------------
    all_event_ids = set()
    all_instances_for_vocab = train + dev + test
    for inst in tqdm(
        all_instances_for_vocab,
        total=len(all_instances_for_vocab),
        desc="Collecting event IDs",
        unit="sequence",
        dynamic_ncols=True
    ):
        all_event_ids.update(inst.sequence)

    embedding_dim = 50
    processor.embedding = {eid: np.random.rand(embedding_dim) for eid in all_event_ids}

    # ---------------- Sequence representation ----------------
    encoder = Sequential_TF(processor.embedding)

    with blocking_stage("Compute train representations"):
        train_presented = encoder.present(train)
    for inst, vec in tqdm(
        zip(train, train_presented),
        total=len(train),
        desc="Assign train representations",
        unit="sequence",
        dynamic_ncols=True
    ):
        inst.repr = vec

    with blocking_stage("Compute dev representations"):
        dev_presented = encoder.present(dev)
    for inst, vec in tqdm(
        zip(dev, dev_presented),
        total=len(dev),
        desc="Assign dev representations",
        unit="sequence",
        dynamic_ncols=True
    ):
        inst.repr = vec

    with blocking_stage("Compute test representations"):
        test_presented = encoder.present(test)
    for inst, vec in tqdm(
        zip(test, test_presented),
        total=len(test),
        desc="Assign test representations",
        unit="sequence",
        dynamic_ncols=True
    ):
        inst.repr = vec
    print("\n[3/6] Representation scaling / FastICA")
    # ---------------- Dimension Reduction (FastICA) ----------------
    train_reprs = np.array([inst.repr for inst in train])
    dev_reprs = np.array([inst.repr for inst in dev])  # <-- Added
    test_reprs = np.array([inst.repr for inst in test])

    scaler = None
    transformer = None
    if reduce_dimension != -1:
        start_time = time.time()
        print(f"Start FastICA, target dimension: {reduce_dimension}")

        # Add small noise to avoid singular matrix issues
        train_reprs += np.random.normal(0, 1e-5, train_reprs.shape)

        # Standardize
        scaler = StandardScaler()
        with blocking_stage("Standardize train representations"):
            train_reprs = scaler.fit_transform(train_reprs)
        train_reprs = np.nan_to_num(train_reprs, nan=0.0, posinf=1e6, neginf=-1e6)

        # Fit ICA on train
        transformer = FastICA(n_components=reduce_dimension, random_state=0)
        with blocking_stage("Fit FastICA on train"):
            train_reprs = transformer.fit_transform(train_reprs)

        # Assign back to train
        for idx, inst in tqdm(
            enumerate(train),
            total=len(train),
            desc="Store ICA train vectors",
            unit="sequence",
            dynamic_ncols=True
        ):
            inst.repr = train_reprs[idx]

        # --- CHANGE / ADD: Transform dev using SAME scaler + ICA ---
        with blocking_stage("Transform dev with scaler + ICA"):
            dev_reprs = scaler.transform(dev_reprs)
            dev_reprs = transformer.transform(dev_reprs)
        for idx, inst in tqdm(
            enumerate(dev),
            total=len(dev),
            desc="Store ICA dev vectors",
            unit="sequence",
            dynamic_ncols=True
        ):
            inst.repr = dev_reprs[idx]

        # Transform test set
        with blocking_stage("Transform test with scaler + ICA"):
            test_reprs = scaler.transform(test_reprs)
            test_reprs = transformer.transform(test_reprs)
        for idx, inst in tqdm(
            enumerate(test),
            total=len(test),
            desc="Store ICA test vectors",
            unit="sequence",
            dynamic_ncols=True
        ):
            inst.repr = test_reprs[idx]

        print(f"Finished FastICA in {time.time() - start_time:.2f} seconds")

    print("\n[4/6] Probabilistic labeling")
    # ---------------- Probabilistic Labeling ----------------
    train_normal = [i for i, inst in enumerate(train) if inst.label == 'Normal']
    normal_ids = train_normal[:len(train_normal) // 2]

    # Paths to probabilistic labeling results inside the dataset folder.
    prob_label_res_file = os.path.join(
        prob_label_res_dir,
        f"mcs-{min_cluster_size}_ms-{min_samples}"
    )
    rand_state_file = os.path.join(
        prob_label_res_dir,
        "random_state"
    )

    # Remove old probabilistic labeling results safely
    if os.path.exists(prob_label_res_file):
        if os.path.isdir(prob_label_res_file):
            shutil.rmtree(prob_label_res_file)
            print(f"Removed old probabilistic labels folder: {prob_label_res_file}")
        else:
            os.remove(prob_label_res_file)
            print(f"Removed old probabilistic labels file: {prob_label_res_file}")

    # Remove random state file if it exists
    if os.path.exists(rand_state_file):
        if os.path.isdir(rand_state_file):
            shutil.rmtree(rand_state_file)
            print(f"Removed old random state folder: {rand_state_file}")
        else:
            os.remove(rand_state_file)
            print(f"Removed old random state file: {rand_state_file}")

    label_generator = Probabilistic_Labeling(min_samples=min_samples, min_clust_size=min_cluster_size,
        res_file=prob_label_res_file, rand_state_file=rand_state_file)
    with blocking_stage("PLELog probabilistic labeling"):
        labeled_train = label_generator.auto_label(train, normal_ids)

    # ------------------------------------------------------------
    # NEW RESOURCE-SAFE EARLY TRAINING STAGE
    # ------------------------------------------------------------
    # HDBSCAN has finished on ORIGINAL full sequences.
    # Now add safe prefix samples for supervised PLELog training.
    if variant == "early" and EARLY_TRAINING_ENABLED:
        print("\n[4.5/6] Build PLELog-Early training prefixes AFTER HDBSCAN")

        early_labeled_train = build_early_training_instances(
            original_train=labeled_train,
            prefix_ratios=TRAIN_PREFIX_RATIOS,
        )

        # Prefix samples are marked with early_prefix_step.
        prefix_only = [
            inst for inst in early_labeled_train
            if hasattr(inst, "early_prefix_step")
        ]

        print(
            f"[EARLY-TRAIN] Original probabilistically labeled sequences: "
            f"{len(labeled_train)}"
        )
        print(
            f"[EARLY-TRAIN] New prefix training samples: {len(prefix_only)}"
        )

        # Recompute representations ONLY for prefixes. Reuse the scaler/ICA
        # fitted on the original full-sequence training set.
        prepare_prefix_representations(
            prefix_instances=prefix_only,
            encoder=encoder,
            scaler=scaler,
            transformer=transformer,
            chunk_size=10000,
        )

        labeled_train = early_labeled_train

        print(
            f"[EARLY-TRAIN] Final PLELog-Early supervised training instances: "
            f"{len(labeled_train)}"
        )

    # ---------------- Model ----------------
    vocab = Vocab()
    vocab.load_from_dict(processor.embedding)

    label2id = {'Normal': 0, 'Anomaly': 1}
    plelog = PLELog(vocab, num_layer, lstm_hiddens, label2id)
    plelog.anomaly_id = label2id['Anomaly']

    best_model_file = os.path.join(output_model_dir, 'best.pt')
    last_model_file = os.path.join(output_model_dir, 'last.pt')

    print(f"\n[5/6] Model training: {'PLELog-Early' if variant == 'early' else 'PLELog-Standard'}")
    # ========================= TRAIN =========================
    Estimated_training_time = 0.0
    if mode == 'train':
        optimizer = Optimizer(filter(lambda p: p.requires_grad, plelog.model.parameters()))
        bestF = 0.0
        start_train = time.time()

        num_train_batches = max(1, (len(labeled_train) + batch_size - 1) // batch_size)

        epoch_bar = tqdm(
            range(epochs),
            total=epochs,
            desc="Training epochs",
            unit="epoch",
            dynamic_ncols=True
        )

        for epoch in epoch_bar:
            plelog.model.train()
            running_loss = 0.0
            seen_batches = 0

            batch_bar = tqdm(
                data_iter(labeled_train, batch_size, True),
                total=num_train_batches,
                desc=f"Epoch {epoch + 1}/{epochs}",
                unit="batch",
                leave=False,
                dynamic_ncols=True
            )

            for onebatch in batch_bar:
                tinst = generate_tinsts_binary_label(onebatch, vocab)
                tinst.to_device(device)

                loss = plelog.forward(tinst.inputs, tinst.targets)
                loss.backward()

                nn.utils.clip_grad_norm_(plelog.model.parameters(), max_norm=1)
                optimizer.step()
                plelog.model.zero_grad()

                seen_batches += 1
                running_loss += float(loss.item())
                batch_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    avg=f"{running_loss / seen_batches:.4f}"
                )

            # ---- DEV evaluation ----
            if dev:
                with blocking_stage(f"Evaluate dev after epoch {epoch + 1}"):
                    p_dev, r_dev, f_dev = plelog.evaluate(dev, threshold)

                epoch_bar.set_postfix(dev_f1=f"{f_dev:.4f}")
                print(f"[DEV] Epoch {epoch + 1} | F1={f_dev:.4f}")

                if f_dev > bestF:
                    bestF = f_dev
                    torch.save(plelog.model.state_dict(), best_model_file)

        torch.save(plelog.model.state_dict(), last_model_file)
        train_time = (time.time() - start_train) / 60
        Estimated_training_time = train_time
        print(f"\nTotal training time: {Estimated_training_time:.2f} minutes")

    print("\n[6/6] Final test evaluation")
    print(f"Sequential prefix batch size (same sequence only): {prefix_batch_size}")
    # ========================= TEST =========================
    if not os.path.exists(best_model_file):
        raise RuntimeError(
            "DEV-selected best.pt was not found. Train first or check: "
            + os.path.abspath(output_model_dir)
        )

    plelog.model.load_state_dict(
        torch.load(best_model_file, map_location=device)
    )

    if variant == "standard":
        # -----------------------------------------------------
        # PLELog-STANDARD
        # Original full-sequence training + full-sequence test.
        # -----------------------------------------------------
        method_name = "PLELog-Standard"
        start = time.time()
        final_metrics = plelog.evaluate_all_metrics(
            instances=test,
            threshold=threshold,
            fp_unit_cost=10.0,
            fn_unit_cost=20.0,
            delay_unit_cost=5.0,
            json_save_path=os.path.join(
                exp_dir,
                "PLELog_Standard_metrics.json"
            ),
            txt_save_path=os.path.join(
                exp_dir,
                "PLELog_Standard_metrics.txt"
            ),
            model_name=method_name,
        )
        evaluation_time = (time.time() - start) / 60.0

    elif variant == "early":
        # -----------------------------------------------------
        # PLELog-EARLY
        # Prefix-aware training + sequential prefix test.
        # -----------------------------------------------------
        method_name = "PLELog-Early"
        start = time.time()
        final_metrics = evaluate_prefix_all_metrics(
            plelog=plelog,
            instances=test,
            encoder=encoder,
            scaler=scaler,
            transformer=transformer,
            threshold=threshold,
            fp_unit_cost=10.0,
            fn_unit_cost=20.0,
            delay_unit_cost=5.0,
            prefix_batch_size=prefix_batch_size,
            json_save_path=os.path.join(
                exp_dir,
                "PLELog_Early_metrics.json"
            ),
            txt_save_path=os.path.join(
                exp_dir,
                "PLELog_Early_metrics.txt"
            ),
            model_name=method_name,
        )
        evaluation_time = (time.time() - start) / 60.0

    else:
        raise ValueError(f"Unexpected child variant: {variant}")

    print("\n" + "=" * 80)
    print(f"FINAL RESULT: {method_name}")
    print("=" * 80)
    print(f"Precision          : {final_metrics['precision']:.4f}")
    print(f"Recall             : {final_metrics['recall']:.4f}")
    print(f"F1                 : {final_metrics['f1_score']:.4f}")
    print(
        f"Detection Coverage : "
        f"{final_metrics['anomaly_detection_coverage']:.4f}"
    )
    print(
        "Avg Det. Ratio     : "
        + (
            "N/A"
            if final_metrics['average_detection_ratio'] is None
            else f"{final_metrics['average_detection_ratio']:.4f}"
        )
    )
    print(f"EDR@25             : {final_metrics['edr_25']:.4f}")
    print(f"EDR@50             : {final_metrics['edr_50']:.4f}")
    print(f"EDR@75             : {final_metrics['edr_75']:.4f}")
    print(
        f"False-positive Cost: "
        f"{final_metrics['false_positive_total_cost']:.4f}"
    )
    print(
        f"False-negative Cost: "
        f"{final_metrics['false_negative_total_cost']:.4f}"
    )
    print(f"Delay Cost         : {final_metrics['delay_total_cost']:.4f}")
    print(f"Total Cost         : {final_metrics['total_cost']:.4f}")
    print(f"Evaluation Time    : {evaluation_time:.4f} min")
    print(f"Training Time      : {Estimated_training_time:.4f} min")
    print("=" * 80)

    # ============================================================
    # FINAL PAPER TABLE: PLELog-Standard vs PLELog-Early
    # ============================================================
    standard_metrics_path = os.path.join(
        results_root,
        "PLELog_Standard",
        f"{dataset}_{parser}",
        "PLELog_Standard_metrics.json",
    )

    comparison_dir = os.path.join(results_root, "comparison", f"{dataset}_{parser}")
    os.makedirs(comparison_dir, exist_ok=True)

    comparison_json_path = os.path.join(
        comparison_dir,
        "PLELog_Standard_vs_Early.json"
    )
    comparison_txt_path = os.path.join(
        comparison_dir,
        "PLELog_Standard_vs_Early.txt"
    )

    if os.path.exists(standard_metrics_path):
        with open(standard_metrics_path, "r", encoding="utf-8") as f:
            standard_metrics = json.load(f)

        early_metrics = final_metrics

        comparison_rows = [
            ("Accuracy", "accuracy"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("F1", "f1_score"),
            ("MCC", "mcc"),
            ("AUROC", "auroc"),
            ("AUPRC", "auprc"),
            ("Detection Coverage", "anomaly_detection_coverage"),
            ("Avg Det. Ratio", "average_detection_ratio"),
            ("EDR@25", "edr_25"),
            ("EDR@50", "edr_50"),
            ("EDR@75", "edr_75"),
            ("False-positive Cost", "false_positive_total_cost"),
            ("False-negative Cost", "false_negative_total_cost"),
            ("Delay Cost", "delay_total_cost"),
            ("Total Cost", "total_cost"),
        ]

        def _fmt_metric(value):
            if value is None:
                return "N/A"
            try:
                return f"{float(value):.4f}"
            except Exception:
                return str(value)

        table_lines = []
        table_lines.append("")
        table_lines.append("=" * 88)
        table_lines.append("FINAL RESULTS: PLELog-Standard vs PLELog-Early")
        table_lines.append("=" * 88)
        table_lines.append(
            f"{'Metric':<28} | "
            f"{'PLELog-Standard':>22} | "
            f"{'PLELog-Early':>22}"
        )
        table_lines.append("-" * 88)

        for label, key in comparison_rows:
            table_lines.append(
                f"{label:<28} | "
                f"{_fmt_metric(standard_metrics.get(key)):>22} | "
                f"{_fmt_metric(early_metrics.get(key)):>22}"
            )

        table_lines.append("=" * 88)

        comparison_report = "\n".join(table_lines)
        print(comparison_report)

        comparison_payload = {
            "dataset": dataset,
            "parser": parser,
            "standard_metrics_path": standard_metrics_path,
            "PLELog-Standard": standard_metrics,
            "PLELog-Early": early_metrics,
        }

        with open(comparison_json_path, "w", encoding="utf-8") as f:
            json.dump(comparison_payload, f, indent=2)

        with open(comparison_txt_path, "w", encoding="utf-8") as f:
            f.write(comparison_report)
            f.write("\n")

        print(f"Saved comparison JSON to: {comparison_json_path}")
        print(f"Saved comparison TXT  to: {comparison_txt_path}")

    else:
        print("")
        print("=" * 88)
        print("PLELog-Standard vs PLELog-Early comparison NOT printed")
        print("=" * 88)
        print("PLELog-Early finished successfully, but the Standard metrics file was not found:")
        print(standard_metrics_path)
        print("")
        print("Run PLELog-Standard once for the same dataset/parser.")
        print("Then rerun PLELog-Early in test mode to print the combined table.")
        print("=" * 88)

    print("All Finished ✅")
