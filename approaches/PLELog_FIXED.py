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

# Approved protocol: original PLELog training only. No prefix augmentation.
EARLY_TRAINING_ENABLED = False


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
    Compute sequence-level metrics from first-alert records generated by an
    online detector. Inputs must already be produced causally; this function
    does not execute model inference. EDR is initially conditional on successful
    detections; the caller normalizes it by all anomalies for main reporting.
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
        "evaluation_mode": "online_first_alert_sequence_aggregation",
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
    lines.append(f"{model_name}: BOUNDED-WINDOW ONLINE TEST METRICS")
    lines.append("#" * 80)
    lines.append(
        "At each event, the original model scores a bounded recent-event window. "
        "Emitted event decisions cannot be revised."
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


# ================================================================
# Approved online detection protocol: ORIGINAL training, bounded
# history, one immutable *window-score decision* per incoming event.
# PLELog remains a sequence/window classifier, not a native event classifier.
# ================================================================
def evaluate_online_bounded_window(
    plelog, instances, encoder, scaler=None, transformer=None,
    threshold=0.5, window_size=120, min_events=1,
    sequence_csv=None, event_csv=None, json_path=None, txt_path=None,
):
    import csv
    from collections import deque

    if int(window_size) < 1 or not (1 <= int(min_events) <= int(window_size)):
        raise ValueError("Invalid online window / minimum-event configuration")
    if not sequence_csv or not event_csv or not json_path or not txt_path:
        raise ValueError("Online evaluation requires all four output file paths")
    os.makedirs(os.path.dirname(os.path.abspath(sequence_csv)), exist_ok=True)
    y_true, y_pred, y_score, lengths, first_steps = [], [], [], [], []
    emitted = 0
    plelog.model.eval()

    with open(event_csv, "w", newline="", encoding="utf-8") as ef, \
         open(sequence_csv, "w", newline="", encoding="utf-8") as sf:
        ew = csv.writer(ef)
        sw = csv.writer(sf)
        ew.writerow(["sequence_id", "event_step", "memory_window_length",
                     "anomaly_score", "emitted_decision", "status",
                     "decision_kind"])
        sw.writerow(["sequence_id", "true_sequence_label", "predicted_sequence_label",
                     "maximum_emitted_anomaly_score", "sequence_length", "first_alert_step",
                     "window_size", "min_events", "protocol"])

        with torch.no_grad():
            for seq_id, inst in enumerate(tqdm(instances, total=len(instances),
                                               desc="PLELog ONLINE event streaming",
                                               unit="sequence", dynamic_ncols=True)):
                original_sequence = list(inst.sequence)
                T = len(original_sequence)
                truth = 0 if inst.label == "Normal" else 1
                memory = deque(maxlen=int(window_size))
                first_alert = None
                max_anomaly_score = None

                for t, new_event in enumerate(original_sequence, 1):
                    # Only the newly arriving event enters memory. Oldest data
                    # is evicted automatically at window_size; no growing prefix.
                    memory.append(new_event)
                    if t < min_events:
                        ew.writerow([seq_id, t, len(memory), "", "", "warmup", "no_prediction"])
                        continue

                    # This classifier requires a sequence-shaped feature vector;
                    # calculate it from the *bounded* currently retained window.
                    # It cannot re-evaluate data already evicted from memory.
                    observation = copy.copy(inst)
                    observation.sequence = list(memory)
                    observation.label = "Normal"  # Placeholder. Test truth not used to score.
                    # Never reuse a full-sequence representation from preprocessing.
                    observation.repr = None
                    for attr in ("event_labels", "sequence_labels", "line_labels",
                                 "anomaly_flags", "event_anomalies", "token_labels"):
                        if hasattr(observation, attr):
                            setattr(observation, attr, None)
                    try:
                        features = np.asarray(encoder.present([observation]))
                        if scaler is not None:
                            features = scaler.transform(features)
                            features = np.nan_to_num(features, nan=0., posinf=1e6, neginf=-1e6)
                        if transformer is not None:
                            features = transformer.transform(features)
                        observation.repr = features[0]
                        tinst = generate_tinsts_binary_label(
                            [observation], plelog.vocab, False)
                        tinst.to_device(device)
                        prediction, probabilities = plelog.predict(tinst.inputs, threshold)
                        probability = float(probabilities.detach().cpu().numpy()[0, plelog.anomaly_id])
                        decision = int(int(np.asarray(prediction).reshape(-1)[0]) == plelog.anomaly_id)
                    except Exception as exc:
                        # Fail rather than silently substituting a normal result.
                        raise RuntimeError(
                            f"PLELog online scoring failed at sequence={seq_id}, event={t}, "
                            f"memory_len={len(memory)}. Cannot produce a fair online score."
                        ) from exc

                    ew.writerow([seq_id, t, len(memory), probability, decision, "scored", "window_classification"])
                    emitted += 1
                    max_anomaly_score = (probability if max_anomaly_score is None
                                         else max(max_anomaly_score, probability))
                    if decision == 1 and first_alert is None:
                        first_alert = t
                    # Do not stop after an alert. Each event gets a final decision.
                    # Do not revise old decisions or re-evaluate the full history.

                # A sequence-level anomaly is reported if at least one online
                # alert was emitted. This is an evaluation mapping, NOT a
                # retrospectively revised or merged event-level decision.
                sequence_decision = int(first_alert is not None)
                y_true.append(truth)
                y_pred.append(sequence_decision)
                y_score.append(0.0 if max_anomaly_score is None else max_anomaly_score)
                lengths.append(max(1, T))
                first_steps.append(first_alert)
                sw.writerow([seq_id, truth, sequence_decision,
                             max_anomaly_score if max_anomaly_score is not None else "",
                             T, first_alert if first_alert is not None else "",
                             window_size, min_events, "bounded_online"])

    # Sequence-level classification and time of first emitted alarm.
    metrics = compute_sequential_baseline_metrics(
        y_true=y_true, y_pred=y_pred, y_score=y_score, seq_lengths=lengths,
        detection_steps=first_steps, fp_unit_cost=10.0,
        fn_unit_cost=20.0, delay_unit_cost=5.0,
    )
    # Existing helper gives conditional EDR. Convert to timely coverage:
    # numerator = correct anomalous sequences detected by q; denominator =
    # ALL anomalous sequences. Preserve conditional values for audit only.
    metrics["edr_conditional_25"] = metrics["edr_25"]
    metrics["edr_conditional_50"] = metrics["edr_50"]
    metrics["edr_conditional_75"] = metrics["edr_75"]
    for q, key in ((.25, "edr_25"), (.50, "edr_50"), (.75, "edr_75")):
        total_anomaly_count = sum(int(yy == 1) for yy in y_true)
        count = sum(int(yy == 1 and pp == 1 and step is not None and
                        step / float(length) <= q)
                    for yy, pp, step, length in zip(y_true, y_pred, first_steps, lengths))
        metrics[key] = count / total_anomaly_count if total_anomaly_count else 0.0

    metrics["evaluation_mode"] = "online_bounded_window"
    metrics["online_window_size"] = int(window_size)
    metrics["online_min_events"] = int(min_events)
    metrics["event_decisions_emitted"] = int(emitted)
    metrics["edr_denominator"] = "all anomalous test sequences"
    metrics["sequence_aggregation"] = "any emitted anomaly alert; event decisions never revised"
    metrics["online_learning"] = False
    metrics["training_augmentation"] = False
    metrics["classification_unit"] = "sequence (event predictions are separately saved)"
    metrics["classification_report_text"] = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=["Normal (0)", "Anomaly (1)"],
        digits=4, zero_division=0,
    )

    report = format_sequential_baseline_metrics(metrics, model_name="PLELog ONLINE")
    report += ("\nEDR definition: fraction of ALL anomalous test sequences "
               "alerted before each sequence fraction.\n"
               "Online model input: bounded most-recent-event window, no prefix augmentation.\n"
               "The event decisions CSV contains immutable online outputs.\n"
               "Event-level evaluation requires EVENT-level ground truth.\n")
    print("\n" + report)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report)
    print("Saved ONLINE metrics:", os.path.abspath(json_path))
    print("Saved ONLINE sequence predictions:", os.path.abspath(sequence_csv))
    print("Saved ONLINE event decisions:", os.path.abspath(event_csv))
    return metrics



# ============================================================================
# SEMANTIC EVENT REPRESENTATION (paper-aligned PLELog)
# Event IDs are lookup keys; vectors come from pre-trained GloVe + TF-IDF of
# the parsed/processed template text. NO random EventId embeddings.
# The parser and prepared Node_block_id sequences remain unchanged.
# ============================================================================
def build_semantic_event_embeddings(train_pkl, training_event_ids, glove_path,
                                    template_column="processed_EventTemplate"):
    import gc
    import math
    import re
    from collections import Counter
    import pandas as pd

    if not os.path.isfile(glove_path):
        raise FileNotFoundError(
            f"PLELog requires a pre-trained 300-dimensional GloVe file: {glove_path}. "
            "Download glove.6B.300d.txt and supply --glove_file PATH. "
            "Do NOT substitute random EventId vectors."
        )
    frame = pd.read_pickle(train_pkl)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected prepared event-row pandas DataFrame in train_df.pkl")
    if 'EventId' not in frame.columns:
        raise ValueError("Prepared train_df.pkl must contain EventId")
    if template_column not in frame.columns:
        raise ValueError(f"Missing {template_column!r}; cannot construct PLELog semantics")
    if frame['EventId'].isna().any():
        raise ValueError('Training PKL has null EventId values; cannot build semantic lookup')

    # Keep the prepared PKL intact. A missing processed template in one row
    # is not an excuse to create new windows, remove rows, or train on test text.
    # Since PLELog embeds event TYPES, choose one canonical text per EventId:
    #   (1) a non-empty processed template from TRAIN, if one exists for the ID;
    #   (2) otherwise a non-empty original EventTemplate from TRAIN.
    # Missing text in other occurrences of the same EventId is recovered via
    # that training-only mapping. Conflicting texts for an ID are rejected.
    def nonempty_text(value):
        if not isinstance(value, str):
            return None
        stripped = value.strip()
        if not stripped or stripped.lower() in ('nan', 'none', 'null', '<na>'):
            return None
        return stripped

    known = set(training_event_ids)
    missing_processed_rows = int(frame[template_column].isna().sum())
    preferred_pairs = frame[['EventId', template_column]].drop_duplicates()
    preferred_pairs[template_column] = preferred_pairs[template_column].map(nonempty_text)
    preferred_pairs = preferred_pairs.dropna(subset=[template_column])
    preferred_pairs = preferred_pairs.drop_duplicates()
    conflicts = preferred_pairs.groupby('EventId', sort=False)[template_column].nunique()
    bad_ids = conflicts[conflicts > 1]
    if not bad_ids.empty:
        raise ValueError(
            f"An EventId maps to multiple non-empty {template_column} texts: "
            f"{bad_ids.head(10).to_dict()}. Resolve parser consistency before training."
        )
    preferred_map = dict(zip(preferred_pairs['EventId'], preferred_pairs[template_column]))
    del preferred_pairs

    fallback_ids = known - set(preferred_map)
    fallback_map = {}
    if fallback_ids:
        if template_column == 'EventTemplate' or 'EventTemplate' not in frame.columns:
            raise ValueError(
                f"No usable {template_column} text for {len(fallback_ids)} training EventIds. "
                f"Examples: {[str(x) for x in list(fallback_ids)[:10]]}. "
                "An EventTemplate fallback is unavailable."
            )
        # Read only rows whose EventId lacks any processed text, so repeated
        # log-event rows cannot inflate template frequency or memory usage.
        raw_pairs = frame.loc[
            frame['EventId'].isin(fallback_ids), ['EventId', 'EventTemplate']
        ].drop_duplicates()
        raw_pairs['EventTemplate'] = raw_pairs['EventTemplate'].map(nonempty_text)
        raw_pairs = raw_pairs.dropna(subset=['EventTemplate']).drop_duplicates()
        raw_conflicts = raw_pairs.groupby('EventId', sort=False)['EventTemplate'].nunique()
        bad_raw = raw_conflicts[raw_conflicts > 1]
        if not bad_raw.empty:
            raise ValueError(
                f"An EventId has multiple non-empty original EventTemplate texts: "
                f"{bad_raw.head(10).to_dict()}. Refusing ambiguous embedding mapping."
            )
        fallback_map = dict(zip(raw_pairs['EventId'], raw_pairs['EventTemplate']))
        del raw_pairs

    mapping = {eid: preferred_map[eid] for eid in known if eid in preferred_map}
    mapping.update({eid: fallback_map[eid] for eid in fallback_ids if eid in fallback_map})
    missing = known - set(mapping)
    if missing:
        raise ValueError(
            f"No usable semantic text (processed or EventTemplate) for "
            f"{len(missing)} training EventIds; examples: {[str(x) for x in list(missing)[:10]]}. "
            "No random or zero-text embeddings will be substituted."
        )
    semantic_source_audit = {
        'preferred_column': template_column,
        'fallback_column': 'EventTemplate' if template_column != 'EventTemplate' else None,
        'training_rows_with_null_preferred_text': missing_processed_rows,
        'event_ids_using_preferred_text': len(known & set(preferred_map)),
        'event_ids_using_original_template_fallback': len(known & set(fallback_map)),
        'fallback_event_id_examples': [str(x) for x in sorted(known & set(fallback_map), key=str)[:10]],
        'training_only_template_mapping': True,
        'notes': 'No sequence regrouping or changes to train/validation/test PKLs; one semantic vector per EventId',
    }
    print('[SEMANTIC TEXT AUDIT] ' + json.dumps(semantic_source_audit, sort_keys=True), flush=True)
    del frame, preferred_map, fallback_map
    gc.collect()

    def tokenize(template):
        # processed_EventTemplate is ALREADY normalized by the user's parser.
        # Do not re-parse logs or create windows. Lowercase alphabetic tokens.
        return re.findall(r'[a-z]+', template.lower())

    docs = {eid: tokenize(txt) for eid, txt in mapping.items()}
    bad = [str(eid) for eid, toks in docs.items() if not toks]
    if bad:
        raise ValueError(f"Empty processed template tokens for EventIds {bad[:10]}")
    df = Counter(tok for tokens in docs.values() for tok in set(tokens))
    required = set(df)
    vectors = {}
    with open(glove_path, 'r', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            parts = line.rstrip('\n').split(' ')
            if len(parts) != 301:
                continue
            word = parts[0]
            if word in required and word not in vectors:
                try:
                    vectors[word] = np.asarray(parts[1:], dtype=np.float32)
                except ValueError:
                    continue
            if len(vectors) == len(required):
                break
    if not vectors:
        raise ValueError('No vocabulary overlap between processed templates and GloVe; check file/tokenization')
    dim = 300
    n_docs = len(docs)
    lookup = {}
    all_zero = 0
    for eid, tokens in docs.items():
        counts = Counter(tokens)
        n_words = len(tokens)
        summed = np.zeros(dim, dtype=np.float32)
        for token, count in counts.items():
            v = vectors.get(token)
            if v is None:
                continue
            tf = count / n_words
            idf = math.log(n_docs / df[token])
            summed += (tf * idf) * v
        # Reproduce paper's extra 1/N normalization.
        summed /= n_words
        if not np.any(summed):
            all_zero += 1
        lookup[eid] = summed
    print(f'[SEMANTIC] {len(lookup)} train EventIds; '
          f'{len(vectors)}/{len(required)} template words found in GloVe; '
          f'{all_zero} zero vectors; dim={dim}', flush=True)
    if all_zero == len(lookup):
        raise ValueError('Every semantic embedding is zero. Refusing to train an invalid baseline.')
    return lookup, {**semantic_source_audit, 'source_column': template_column,
                    'pretrained_embeddings': os.path.abspath(glove_path),
                    'dimension': dim, 'number_training_templates': n_docs,
                    'vocabulary_coverage': len(vectors) / max(1, len(required)),
                    'zero_vectors': all_zero,
                    'method': 'training-template TF-IDF weighted pretrained GloVe (paper-aligned)',
                    'idf_fitted_on': 'training templates only'}


# ============================================================================
# CAUSAL BATCHED ONLINE INFERENCE. NO training or architecture changes.
# Validation selects an alert confirmation count; test is scored exactly once.
# ============================================================================
def batched_online_scores(instances, plelog, encoder, scaler, transformer,
                          window_size, min_events, batch_size=512):
    import copy
    from collections import deque
    if not (1 <= min_events <= window_size):
        raise ValueError('online_min_events must be within [1, online_window]')
    if batch_size < 1:
        raise ValueError('online_batch_size must be positive')
    plelog.model.eval()
    group_size = 64  # Bound memory independently of total test-set size.
    for start in tqdm(range(0, len(instances), group_size), desc='Causal batched prefixes', unit='group'):
        group = instances[start:start + group_size]
        scores = [[] for _ in group]
        buffer_obs, buffer_local = [], []
        def flush():
            if not buffer_obs:
                return
            with torch.no_grad():
                features = np.asarray(encoder.present(buffer_obs))
                if scaler is not None:
                    features = np.nan_to_num(scaler.transform(features), nan=0., posinf=1e6, neginf=-1e6)
                if transformer is not None:
                    features = transformer.transform(features)
                for obs, vec in zip(buffer_obs, features):
                    obs.repr = vec
                batch = generate_tinsts_binary_label(buffer_obs, plelog.vocab, False)
                batch.to_device(device)
                _pred, probs = plelog.predict(batch.inputs, None)
                anomaly_probs = probs.detach().cpu().numpy()[:, plelog.anomaly_id]
            if len(anomaly_probs) != len(buffer_local):
                raise AssertionError('Online batch output size mismatch')
            for (local_id, event_step), prob in zip(buffer_local, anomaly_probs):
                scores[local_id].append((event_step, float(prob)))
            buffer_obs.clear()
            buffer_local.clear()

        for local_id, inst in enumerate(group):
            memory = deque(maxlen=window_size)
            for t, event_id in enumerate(inst.sequence, start=1):
                memory.append(event_id)
                if t < min_events:
                    continue
                obs = copy.copy(inst)
                obs.sequence = list(memory)  # 100% causal; never includes future events.
                obs.label = 'Normal'       # Ground truth NEVER passed to scoring.
                obs.repr = None           # Do not reuse full-sequence representation.
                for attr in ('event_labels','sequence_labels','line_labels',
                             'anomaly_flags','event_anomalies','token_labels'):
                    if hasattr(obs, attr):
                        setattr(obs, attr, None)
                buffer_obs.append(obs)
                buffer_local.append((local_id, t))
                if len(buffer_obs) >= batch_size:
                    flush()
        flush()
        for local_id, inst in enumerate(group):
            yield (start + local_id, inst, scores[local_id])


def first_confirmed_alert(scores, threshold, consecutive):
    streak = 0
    for step, prob in scores:
        streak = streak + 1 if prob >= threshold else 0
        if streak >= consecutive:
            return int(step)
    return None


def calibrate_online_confirmation(dev, plelog, encoder, scaler, transformer,
                                   threshold, window_size, min_events,
                                   candidates, batch_size, save_path):
    candidates = sorted(set(int(c) for c in candidates))
    if not candidates or candidates[0] < 1:
        raise ValueError('Online confirmation candidates must be positive integers')
    counts = {k: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'ratios': []} for k in candidates}
    for seq_id, inst, scores in batched_online_scores(
            dev, plelog, encoder, scaler, transformer,
            window_size, min_events, batch_size):
        truth = int(inst.label != 'Normal')
        for k in candidates:
            alarm = first_confirmed_alert(scores, threshold, k)
            pred = int(alarm is not None)
            key = 'tp' if pred and truth else 'fp' if pred else 'fn' if truth else 'tn'
            counts[k][key] += 1
            if pred and truth:
                counts[k]['ratios'].append(alarm / len(inst.sequence))
    reports = []
    for k in candidates:
        c = counts[k]
        tp, tn, fp, fn = [c[x] for x in ('tp', 'tn', 'fp', 'fn')]
        p = tp / (tp + fp) if tp + fp else 0.
        r = tp / (tp + fn) if tp + fn else 0.
        f1 = 2 * p * r / (p + r) if p + r else 0.
        fpr = fp / (fp + tn) if fp + tn else 0.
        avg_ratio = float(np.mean(c['ratios'])) if c['ratios'] else 1.
        reports.append({'confirmations': k, 'f1': f1, 'precision': p, 'recall': r,
                        'fpr': fpr, 'avg_ratio': avg_ratio,
                        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn})
    # Prespecified deterministic rule. Never choose policy on test results.
    selected = max(reports, key=lambda z: (z['f1'], -z['fpr'], -z['avg_ratio'], -z['confirmations']))
    artifact = {'selection_dataset': 'validation', 'criterion':
                'highest validation F1; tie lower FPR; tie earlier AvgRatio; tie fewer confirmations',
                'threshold': threshold, 'window_size': window_size,
                'min_events': min_events, 'candidates': reports,
                'selected_confirmations': selected['confirmations']}
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(artifact, f, indent=2)
    print('[ONLINE RULE] selected k=', selected['confirmations'],
          'from validation only; full candidate results:', save_path, flush=True)
    return int(selected['confirmations'])


def evaluate_online_batched_confirmed(plelog, instances, encoder, scaler, transformer,
                                      threshold, window_size, min_events, confirmations,
                                      batch_size, sequence_csv, event_csv, json_path, txt_path):
    import csv
    y_true, y_pred, y_score, lengths, first_steps = [], [], [], [], []
    emitted = 0
    os.makedirs(os.path.dirname(os.path.abspath(sequence_csv)), exist_ok=True)
    with open(event_csv, 'w', newline='', encoding='utf-8') as ef, \
         open(sequence_csv, 'w', newline='', encoding='utf-8') as sf:
        ew, sw = csv.writer(ef), csv.writer(sf)
        ew.writerow(['sequence_id','event_step','anomaly_score','raw_threshold_positive',
                     'confirmed_alert_on_this_step','memory_window_length'])
        sw.writerow(['sequence_id','true_sequence_label','predicted_sequence_label',
                     'maximum_emitted_anomaly_score','sequence_length','first_alert_step',
                     'window_size','min_events','confirmations','protocol'])
        for seq_id, inst, scores in batched_online_scores(
                instances, plelog, encoder, scaler, transformer,
                window_size, min_events, batch_size):
            truth = int(inst.label != 'Normal')
            length = len(inst.sequence)
            alarm = first_confirmed_alert(scores, threshold, confirmations)
            pred = int(alarm is not None)
            max_score = max((p for _, p in scores), default=0.)
            for t, p in scores:
                ew.writerow([seq_id,t,p,int(p>=threshold),int(alarm==t),min(t,window_size)])
                emitted += 1
            sw.writerow([seq_id,truth,pred,max_score,length,alarm if alarm is not None else '',
                         window_size,min_events,confirmations,'causal_confirmed_window'])
            y_true.append(truth)
            y_pred.append(pred)
            y_score.append(max_score)
            lengths.append(max(1,length))
            first_steps.append(alarm)
    metrics = compute_sequential_baseline_metrics(y_true,y_pred,y_score,lengths,first_steps,
                        fp_unit_cost=10.,fn_unit_cost=20.,delay_unit_cost=5.)
    # Main EDR denominator is ALL anomalous test sequences, not just TP.
    for q, key in ((.25,'edr_25'),(.50,'edr_50'),(.75,'edr_75')):
        metrics[f'{key}_conditional'] = metrics[key]
        denom = sum(int(z==1) for z in y_true)
        num = sum(int(truth==1 and pred==1 and step is not None and step/length <= q)
                  for truth,pred,step,length in zip(y_true,y_pred,first_steps,lengths))
        metrics[key] = num/denom if denom else 0.
    metrics.update({'evaluation_mode':'causal_batched_online',
                   'online_window_size':int(window_size),
                   'online_min_events':int(min_events),
                   'confirmation_count_selected_on_validation':int(confirmations),
                   'decision_threshold':float(threshold),
                   'event_decisions_emitted':int(emitted),
                   'edr_denominator':'all anomalous test sequences',
                   'training_augmentation':False,'online_learning':False,
                   'classification_unit':'sequence',
                   'classification_report_text':classification_report(
                        y_true,y_pred,labels=[0,1],target_names=['Normal (0)','Anomaly (1)'],
                        digits=4,zero_division=0)})
    report = format_sequential_baseline_metrics(metrics, model_name='PLELog ONLINE')
    report += ('\nCausal batched inference uses only observed events; frozen model.\n'
               'Confirmation count was selected using validation ONLY.\n'
               'Event decisions are window decisions, not event anomaly ground truth.\n')
    print('\n'+report)
    with open(json_path,'w',encoding='utf-8') as f: json.dump(metrics,f,indent=2)
    with open(txt_path,'w',encoding='utf-8') as f: f.write(report)
    return metrics

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
        csv_save_path=None,
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

        if csv_save_path is not None:
            import csv
            os.makedirs(os.path.dirname(os.path.abspath(csv_save_path)), exist_ok=True)
            with open(csv_save_path, "w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["sequence_id", "true_sequence_label", "predicted_sequence_label",
                                 "anomaly_score", "sequence_length", "first_alert_step",
                                 "decision_protocol"])
                for i, (truth, pred, score, length) in enumerate(
                    zip(y_true, y_pred, y_score, seq_lengths)
                ):
                    writer.writerow([i, truth, pred, score, length,
                                     int(length) if pred else "", "full_sequence"])
            print("Saved Standard sequence-level predictions:", csv_save_path)

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



def audit_prepared_node_block_sequences(split_paths, loaded_splits, report_path):
    """Fail closed if PKLPreprocessor changed user-prepared Node_block_id sessions.

    This is an audit of the UNCHANGED original PKLPreprocessor output, not a
    replacement for it.  Each raw Node_block_id group defines exactly one
    sequence; intra-group event order and the aggregate binary label must match.
    We match a multiset of complete ordered sequences to permit harmless
    inter-sequence reordering and repeated identical sequences.

    Raw inputs can be event-level rows or already grouped list-valued rows.
    The preprocessor must preserve the original EventId, EventTemplate or
    processed_EventTemplate tokens; transformations of all these identifiers
    require inspecting preprocessing/Preprocess.py before adapting safely.
    """
    import hashlib
    import json
    from collections import Counter
    from pathlib import Path
    import pandas as pd
    import numpy as np

    def bin_label(value):
        if isinstance(value, (list, tuple, np.ndarray)):
            return int(any(bin_label(v) for v in value))
        if pd.isna(value):
            raise ValueError('Missing event-level label in prepared PKL')
        text = str(value).strip().lower()
        if text in ('normal', '0', '0.0', '-', 'false', 'n', 'benign'):
            return 0
        if text in ('anomaly', 'abnormal', '1', '1.0', 'true', 'a'):
            return 1
        raise ValueError(f'Unsupported Label value in prepared PKL: {value!r}')

    def fingerprint(tokens, label):
        digest = hashlib.blake2b(digest_size=20)
        digest.update(b'1' if label else b'0')
        for token in tokens:
            # No ambiguous delimiters: each token is length-prefixed.
            encoded = str(token).encode('utf-8', errors='backslashreplace')
            digest.update(len(encoded).to_bytes(8, 'big'))
            digest.update(encoded)
        return digest.hexdigest()

    report = {'protocol': 'existing Node_block_id groups; no new windows or blocks',
              'comparison': 'multiset of ordered sequence tokens and labels',
              'splits': {}, 'status': 'PASS'}
    failures = []
    entries = [('train', 'train_pkl'), ('validation', 'dev_pkl'), ('test', 'test_pkl')]
    for split, path_key in entries:
        raw = pd.read_pickle(split_paths[path_key])
        if isinstance(raw, (list, tuple)) and all(isinstance(x, dict) for x in raw):
            raw = pd.DataFrame(raw)
        if not isinstance(raw, pd.DataFrame):
            raise TypeError(f'{split}: expected DataFrame or dict records in prepared PKL')
        if 'Label' not in raw.columns:
            raise ValueError(f'{split}: expected event-level Label column')
        candidate_id_columns = ('Node_block_id', 'BlockId', 'node_block_id', 'block_id')
        id_col = next((c for c in candidate_id_columns if c in raw.columns), None)
        if id_col is None:
            raise ValueError(f'{split}: cannot confirm prepared sequences; missing Node_block_id/BlockId')
        if raw[id_col].isna().any() or raw[id_col].astype(str).str.strip().eq('').any():
            raise ValueError(f'{split}: missing/blank {id_col}; cannot verify sequence boundaries')
        token_columns = [c for c in ('EventId', 'EventTemplate', 'processed_EventTemplate') if c in raw.columns]
        if not token_columns:
            raise ValueError(f'{split}: expected EventId or EventTemplate for alignment comparison')

        expected = {c: Counter() for c in token_columns}
        raw_lengths = Counter()
        raw_label_counts = Counter()
        grouped = raw.groupby(id_col, sort=False, dropna=False)
        for group_id, group in grouped:
            # A prepared sequence can be represented by multiple event rows,
            # or by a single row whose event fields contain full lists.
            sequence_label = bin_label(group['Label'].tolist())
            candidate_sequences = {}
            for col in token_columns:
                values = group[col].tolist()
                if len(values) == 1 and isinstance(values[0], (list, tuple, np.ndarray)):
                    values = list(values[0])
                elif any(isinstance(v, (list, tuple, np.ndarray)) for v in values):
                    raise ValueError(f'{split}: mixed list/scalar event values in Node_block_id={group_id!r}')
                candidate_sequences[col] = values
            sizes = {len(v) for v in candidate_sequences.values()}
            if len(sizes) != 1:
                raise ValueError(f'{split}: inconsistent sequence lengths across event columns for {group_id!r}')
            length = sizes.pop()
            raw_lengths[length] += 1
            raw_label_counts[sequence_label] += 1
            for col, values in candidate_sequences.items():
                expected[col][fingerprint(values, sequence_label)] += 1

        processed = loaded_splits[split]
        actual = Counter()
        processed_lengths = Counter()
        processed_labels = Counter()
        for i, inst in enumerate(processed):
            sequence = getattr(inst, 'sequence', None)
            if sequence is None:
                raise ValueError(f'{split}: processed instance {i} missing .sequence')
            labels = bin_label(getattr(inst, 'label', None))
            tokens = list(sequence)
            actual[fingerprint(tokens, labels)] += 1
            processed_lengths[len(tokens)] += 1
            processed_labels[labels] += 1

        # The first matching field explains the exact original representation.
        matches = {col: sum((raw_counter & actual).values())
                   for col, raw_counter in expected.items()}
        exact = next((col for col, raw_counter in expected.items()
                      if raw_counter == actual), None)
        status = 'PASS' if exact else 'FAIL'
        report['splits'][split] = {
            'status': status, 'identifier_column': id_col,
            'input_event_rows': int(len(raw)),
            'prepared_sequence_count': int(sum(raw_lengths.values())),
            'plelog_sequence_count': int(len(processed)),
            'prepared_normal_anomalous': {str(k): v for k,v in raw_label_counts.items()},
            'plelog_normal_anomalous': {str(k): v for k,v in processed_labels.items()},
            'prepared_lengths': {str(k): v for k,v in raw_lengths.items()},
            'plelog_lengths': {str(k): v for k,v in processed_lengths.items()},
            'exact_event_column': exact,
            'matching_sequences_by_field': matches,
            'note': 'Groups match without requiring identical between-group ordering.'
        }
        print(f'[PREPARED ALIGNMENT] {split}: {status}; '
              f'prepared={sum(raw_lengths.values())} processed={len(processed)} '
              f'content_match={exact!r}; partial_matches={matches}', flush=True)
        if not exact:
            failures.append(split)
        del raw, expected, actual

    if failures:
        report['status'] = 'FAIL'
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('[PREPARED ALIGNMENT] saved:', path.resolve(), flush=True)
    if failures:
        raise RuntimeError(
            'PLELog PKLPreprocessor output does NOT match existing Node_block_id '
            f'sequences in {failures}. No new sliding windows or re-grouping is '
            'permitted. STOP before training. Inspect the report and supply '
            'preprocessing/Preprocess.py so the loader can be adapted without '
            'changing PLELog architecture or training.')
    return report


def check_raw_blockid_split_overlap(split_paths, dataset_name, report_path):
    """Fail-fast split-leakage audit on ORIGINAL input PKL BlockIds.

    Checks real BlockId / Node_block_id values in raw PKLs, before PLELog
    preprocessing can discard grouping identifiers. Multiple events within
    the same split are expected to share their BlockId and are NOT an error.
    """
    import gc
    from pathlib import Path
    import json
    import re
    import pandas as pd

    def canonical_block_id(value):
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (ValueError, TypeError):
            pass
        if isinstance(value, (list, tuple, dict, set)):
            raise ValueError("BlockId values must be scalar values")
        result = str(value).strip()
        if result.lower() in {"", "nan", "none", "null", "<na>"}:
            return None
        # Align numeric 123.0 (float) and 123 (integer) without changing
        # alphanumeric IDs such as blk_-123 or zero-padded strings.
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            if isinstance(value, float) and value.is_integer():
                result = str(int(value))
        return result

    def get_column(df):
        candidates = {"blockid", "nodeblockid", "hdfsblockid"}
        matches = [col for col in df.columns
                   if re.sub(r"[^a-z0-9]", "", str(col).lower()) in candidates]
        if not matches:
            return None
        # Prefer the existing prepared Node_block_id sequence key when present.
        matches.sort(key=lambda c: (re.sub(r"[^a-z0-9]", "", str(c).lower()) != "nodeblockid", str(c)))
        return matches[0]

    # Read one split at a time, holding only its set of distinct block IDs.
    seen = {}
    summary = {}
    for split, path_key in (("train", "train_pkl"), ("validation", "dev_pkl"), ("test", "test_pkl")):
        path = split_paths[path_key]
        data = pd.read_pickle(path)
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, (list, tuple)) and all(isinstance(r, dict) for r in data):
            df = pd.DataFrame(data)
        else:
            raise TypeError(
                f"Cannot audit {split} BlockIds in {path}: expected a pandas "
                "DataFrame or a list of dictionaries. Refusing to assume no overlap."
            )
        col = get_column(df)
        if col is None:
            summary[split] = {"path": path, "block_id_column": None, "rows": int(len(df)),
                              "distinct_block_ids": 0, "missing_block_ids": None}
            seen[split] = None
        else:
            ids = set()
            missing_count = 0
            # unique() avoids recomputing repeated event-level BlockIds.
            raw = df[col]
            missing_count = int(raw.isna().sum())
            for value in raw.dropna().unique():
                ident = canonical_block_id(value)
                if ident is None:
                    missing_count += int((raw == value).sum())
                else:
                    ids.add(ident)
            summary[split] = {"path": path, "block_id_column": str(col), "rows": int(len(df)),
                              "distinct_block_ids": len(ids), "missing_block_ids": missing_count}
            seen[split] = ids
        del data, df
        gc.collect()

    pair_reports = {}
    overlaps = []
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if seen[a] is not None and seen[b] is not None:
            shared = seen[a].intersection(seen[b])
            pair_reports[f"{a}_vs_{b}"] = {
                "shared_block_ids": len(shared),
                "examples": sorted(shared)[:10],
            }
            if shared:
                overlaps.append((a, b, len(shared)))
        else:
            pair_reports[f"{a}_vs_{b}"] = {"shared_block_ids": None, "examples": []}

    absent = [split for split, ids in seen.items() if ids is None]
    incomplete = [split for split, rec in summary.items()
                  if rec["missing_block_ids"] is not None and rec["missing_block_ids"] > 0]
    # On HDFS every row needs a usable BlockId. For non-block datasets, all
    # 3 splits may lack BlockId; then log that the check is not applicable.
    require_blocks = True  # Node_block_id is mandatory for ALL prepared datasets
    if (absent and (require_blocks or len(absent) != len(seen))) or incomplete:
        status = "ERROR_INCOMPLETE_BLOCKID_DATA"
    elif overlaps:
        status = "ERROR_BLOCKID_OVERLAP"
    elif len(absent) == len(seen):
        status = "SKIPPED_NO_BLOCKID_COLUMNS"
    else:
        status = "PASS_NO_BLOCKID_OVERLAP"

    report = {
        "dataset": str(dataset_name),
        "status": status,
        "split_information": summary,
        "pairwise_block_id_overlap": pair_reports,
        "require_block_ids": require_blocks,
        "note": "Repeated BlockIds inside one split are expected; shared BlockIds across splits are forbidden.",
    }
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 76)
    print(f"RAW BLOCKID SPLIT AUDIT | dataset={dataset_name} | status={status}")
    for split, rec in summary.items():
        print(f"  {split:<10} BlockId column={str(rec['block_id_column']):<15} "
              f"distinct IDs={rec['distinct_block_ids']:,} | missing={rec['missing_block_ids']}")
    for pair, item in pair_reports.items():
        print(f"  {pair}: shared BlockIds={item['shared_block_ids']} | examples={item['examples']}")
    print(f"  Audit saved: {report_file.resolve()}")
    print("=" * 76)

    if status.startswith("ERROR"):
        if overlaps:
            message = "; ".join(f"{a} vs {b}: {n} shared BlockIds" for a, b, n in overlaps)
            print("\n*** ALERT: BLOCKID LEAKAGE DETECTED — STOPPING BEFORE TRAINING ***")
            raise RuntimeError("BLOCKID OVERLAP: " + message + ". Check " + str(report_file))
        print("\n*** ALERT: BLOCKID AUDIT INCOMPLETE — STOPPING BEFORE TRAINING ***")
        raise RuntimeError(
            "Cannot guarantee BlockId split independence: "
            f"missing BlockId column in {absent}, null/invalid BlockIds in {incomplete}. "
            f"Check {report_file}"
        )
    if status == "SKIPPED_NO_BLOCKID_COLUMNS":
        print("  NOTE: No BlockId columns exist in this dataset. BlockId overlap check is not applicable.")
    return report


def check_plelog_split_overlap(train, dev, test, report_path=None, fail_on_overlap=False, max_examples=5):
    """Audit post-preprocessing sample overlap across train/dev/test.

    Fingerprints use event sequences WITHOUT labels. Separate checks use any
    original block/session identifiers retained on the instance. A content
    duplicate is a warning, not definitive proof of split leakage.
    """
    import hashlib
    import json
    import os
    import pickle
    from collections import Counter

    split_data = {"train": train, "val": dev, "test": test}
    id_names = ("BlockId", "block_id", "Node_block_id", "node_block_id",
                "SessionId", "session_id", "blockId")
    indexed = {}
    report = {"split_sizes": {}, "within_split": {}, "between_splits": {}}

    for name, instances in split_data.items():
        by_content, by_id = {}, {}
        duplicate_content, conflicting_content_labels = 0, 0
        for i, inst in enumerate(instances):
            seq = getattr(inst, "sequence", None)
            if seq is None:
                raise ValueError(f"{name}[{i}] has no .sequence field")
            digest = hashlib.sha256()
            for event in seq:
                try:
                    raw = pickle.dumps(event, protocol=4)
                except (TypeError, ValueError, pickle.PicklingError):
                    raw = repr(event).encode("utf-8", errors="backslashreplace")
                digest.update(len(raw).to_bytes(8, "big"))
                digest.update(raw)
            key = digest.hexdigest()
            label = str(getattr(inst, "label", "unknown"))
            if key in by_content:
                duplicate_content += 1
                if by_content[key][1] != label:
                    conflicting_content_labels += 1
            else:
                by_content[key] = (i, label)

            for attr in id_names:
                value = getattr(inst, attr, None)
                if value is not None and str(value).strip():
                    # Include attribute name so unlike ID fields aren't mixed.
                    by_id.setdefault((attr, str(value)), i)
        indexed[name] = (by_content, by_id)
        report["split_sizes"][name] = len(instances)
        report["within_split"][name] = {
            "extra_identical_sequences": duplicate_content,
            "conflicting_labels_among_duplicates": conflicting_content_labels,
            "original_ids_available": len(by_id),
        }

    print("\n" + "=" * 70)
    print("PLELOG TRAIN / VALIDATION / TEST OVERLAP AUDIT")
    print("=" * 70)
    print("Note: exact repeated log patterns can occur naturally; inspect samples.")
    found_overlap = False
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        a_content, a_ids = indexed[a]
        b_content, b_ids = indexed[b]
        shared = a_content.keys() & b_content.keys()
        shared_ids = a_ids.keys() & b_ids.keys()
        conflicts = sum(a_content[k][1] != b_content[k][1] for k in shared)
        examples = [
            {"left_index": a_content[k][0], "right_index": b_content[k][0],
             "left_label": a_content[k][1], "right_label": b_content[k][1]}
            for k in list(shared)[:max_examples]
        ]
        report["between_splits"][f"{a}_vs_{b}"] = {
            "identical_sequence_fingerprints": len(shared),
            "different_labels_for_same_sequence": conflicts,
            "shared_original_ids": len(shared_ids),
            "sequence_examples": examples,
            "id_examples": [str(v) for v in list(shared_ids)[:max_examples]],
        }
        found_overlap |= bool(shared or shared_ids)
        print(f"{a:5} vs {b:5}: shared sequences={len(shared):6d} | "
              f"shared original IDs={len(shared_ids):6d} | conflicting labels={conflicts:6d}")
        if examples:
            print(f"   examples (indices, labels): {examples}")

    for name, item in report["within_split"].items():
        print(f"Within {name:5}: extra duplicate sequences={item['extra_identical_sequences']} | "
              f"conflicting labels={item['conflicting_labels_among_duplicates']}")
    print("=" * 70 + "\n")

    if report_path:
        parent = os.path.dirname(os.path.abspath(report_path))
        os.makedirs(parent, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("Saved overlap audit:", os.path.abspath(report_path))

    if fail_on_overlap and found_overlap:
        raise RuntimeError("Split overlap detected. Review the report before training.")
    return report


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
    argparser.add_argument('--dataset', default='SP_150MB_ratio', type=str)  # BGL, HDFS, TH_1G, SP_150MB
    argparser.add_argument('--mode', default='train', type=str, choices=['train'],
                           help='Train original PLELog once, then evaluate the requested inference modes. '
                                'Inference-only checkpoint loading is not supported until preprocessing artifacts are restored.')
    argparser.add_argument('--parser', default='IBM', type=str)
    argparser.add_argument('--min_cluster_size', type=int, default=100)
    argparser.add_argument('--min_samples', type=int, default=100)
    argparser.add_argument('--reduce_dimension', type=int, default=100)
    argparser.add_argument('--threshold', type=float, default=0.5,
                           help='Original fixed anomaly threshold (default 0.5); never fitted on test')
    argparser.add_argument('--case', default='in_domain', type=str, choices=['in_domain', 'cross_dataset'])
    argparser.add_argument(
        '--variant',
        default='both',
        type=str,
        choices=['standard', 'online', 'both'],
        help=(
            "standard = original PLELog training + full-sequence test; "
            "online = same original training + bounded-window streaming inference; "
            "both = train once and evaluate Standard followed by Online"
        )
    )
    argparser.add_argument('--audit_only', action='store_true', help='Check prepared PKL alignment after original preprocessing, then exit BEFORE training')
    argparser.add_argument('--glove_file', default=None, type=str,
                           help='Path to pretrained glove.6B.300d.txt (REQUIRED for paper-aligned PLELog)')
    argparser.add_argument('--semantic_column', default='processed_EventTemplate',
                           choices=['processed_EventTemplate', 'EventTemplate'],
                           help='Prepared PKL column to embed; do not use random EventId vectors')
    argparser.add_argument('--online_batch_size', default=512, type=int,
                           help='Batch size for causal online scoring; does not alter scores or training')
    argparser.add_argument('--online_confirm_candidates', type=int, nargs='+', default=[1,2,3],
                           help='Consecutive-positive confirmation candidates selected on validation only')
    argparser.add_argument('--online_window', default=120, type=int,
                           help='Maximum recent log events retained in memory for online inference')
    argparser.add_argument('--online_min_events', default=1, type=int,
                           help='Explicit warm-up before first online decision (default 1)')
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
    online_window = max(1, int(args.online_window))
    online_min_events = max(1, int(args.online_min_events))
    if online_min_events > online_window:
        raise ValueError("--online_min_events must not exceed --online_window")
    variant = args.variant

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

    # All modes use the same unmodified full-sequence PLELog training protocol.
    # The both mode trains once and runs both evaluations from ONE checkpoint.
    save_dir = os.path.join(results_root, "PLELog_SEMANTIC_ORIGINALTRAIN_CAUSAL_ONLINE")

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

    # No independent Early retraining; original PLELog training is shared.
    EARLY_TRAINING_ENABLED = False

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
        "TH_2G",
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

    # CRITICAL: inspect RAW BlockIds BEFORE PKLPreprocessor transforms or drops IDs.
    # A shared BlockId in two splits is a hard error and stops training.
    check_raw_blockid_split_overlap(
        DATASETS[dataset],
        dataset_name=dataset,
        report_path=os.path.join(exp_dir, "blockid_overlap_audit.json"),
    )

    with open(os.path.join(exp_dir, "PLELog_run_configuration.json"), "w", encoding="utf-8") as f:
        json.dump({
            "dataset": dataset,
            "parser": parser,
            "mode": mode,
            "variant": variant,
            "random_seed": 42,
            "online_window": online_window,
            "online_min_events": online_min_events,
            "threshold": threshold,
            "training_prefix_augmentation": False,
            "semantic_embedding_source": "GloVe+training-template TF-IDF",
            "dataset_pkl_paths": DATASETS[dataset],
            "metric_unit": "sequence-level summary; per-event window predictions saved separately",
            "online_decision": "frozen classifier on bounded causal windows; confirmation rule fit on validation only",
            "cost_weights": {"false_positive": 10.0, "false_negative": 20.0, "delay": 5.0},
        }, f, indent=2)

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
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    if CASE != "in_domain":
        raise NotImplementedError(
            "This verified Standard/Online revision currently supports in_domain only. "
            "The uploaded cross_dataset branch references undefined source settings."
        )

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

        # Verify that the ORIGINAL preprocessor respected the existing
        # Node_block_id sequence boundaries, ordered events, and labels.
        # This audit does NOT create, cut, regroup or relabel sequences.
        audit_prepared_node_block_sequences(
            paths,
            {'train': train, 'validation': dev, 'test': test},
            report_path=os.path.join(exp_dir, 'prepared_sequence_alignment_audit.json'),
        )
        if args.audit_only:
            print('Audit-only completed: no model training or evaluation performed.')
            raise SystemExit(0)

        # Supplemental processed-sequence overlap report (repeated log patterns
        # may occur naturally). RAW BlockId overlap has already hard-stopped.
        check_plelog_split_overlap(
            train, dev, test,
            report_path=os.path.join(exp_dir, "split_overlap_audit.json"),
            fail_on_overlap=os.environ.get("PLELOG_FAIL_ON_OVERLAP", "0") == "1",
        )

        # Original training, identical for Standard and Online inference.
        print("[TRAIN] Original PLELog training; no prefix augmentation.")

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
    #======================================================++++++++++++++++++++++++++++++++++++++++


    # use in firt and second paper
    #processor = PKLPreprocessor()
    #train, dev, test = processor.load_pkl(dataset, train_pkl, dev_pkl, test_pkl)


    print(f"Loaded {len(train)} train / {len(dev)} dev / {len(test)} test")

    print("\n[2/6] Paper-aligned semantic event vectors (GloVe + template TF-IDF)")
    all_event_ids = set()
    for inst in tqdm(train, total=len(train), desc="Collect training EventIds", unit="sequence"):
        all_event_ids.update(inst.sequence)
    glove_file = (args.glove_file or os.path.join(PROJECT_ROOT, 'datasets', 'glove.6B.300d.txt'))
    if not os.path.isabs(glove_file):
        glove_file = os.path.abspath(glove_file)
    semantic_vectors, semantic_audit = build_semantic_event_embeddings(
        DATASETS[TARGET_DATASET]['train_pkl'], all_event_ids, glove_file,
        template_column=args.semantic_column,
    )
    embedding_dim = 300
    unknown_vector = np.zeros(embedding_dim, dtype=np.float32)
    class FixedOOVEmbeddings(dict):
        def __missing__(self, key):
            return unknown_vector
        def get(self, key, default=None):
            return dict.get(self, key, unknown_vector)
    processor.embedding = FixedOOVEmbeddings(semantic_vectors)
    with open(os.path.join(exp_dir, 'PLELog_semantic_embedding_audit.json'), 'w', encoding='utf-8') as fh:
        json.dump(semantic_audit, fh, indent=2)
    print('EventId is a lookup key; semantic content comes from template text.', flush=True)

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

    if variant in ("standard", "both"):
        # Full sequence is needed for the Standard reference only.
        with blocking_stage("Compute Standard test representations"):
            test_presented = encoder.present(test)
        for inst, vec in tqdm(
            zip(test, test_presented),
            total=len(test),
            desc="Assign Standard test representations",
            unit="sequence",
            dynamic_ncols=True
        ):
            inst.repr = vec
    else:
        # Online-only mode must not materialize complete test representations.
        test_presented = None
    print("\n[3/6] Representation scaling / FastICA")
    # ---------------- Dimension Reduction (FastICA) ----------------
    train_reprs = np.array([inst.repr for inst in train])
    dev_reprs = np.array([inst.repr for inst in dev])  # <-- Added
    test_reprs = (np.array([inst.repr for inst in test])
                  if variant in ("standard", "both") else None)

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

        # Transform full test sequences for Standard reference only.
        # Online mode constructs each current-window representation on arrival.
        if test_reprs is not None:
            with blocking_stage("Transform Standard test with scaler + ICA"):
                test_reprs = scaler.transform(test_reprs)
                test_reprs = transformer.transform(test_reprs)
            for idx, inst in tqdm(
                enumerate(test),
                total=len(test),
                desc="Store ICA Standard test vectors",
                unit="sequence",
                dynamic_ncols=True
            ):
                inst.repr = test_reprs[idx]

        print(f"Finished FastICA in {time.time() - start_time:.2f} seconds")

    # Persist TRAIN-FITTED inputs so the comparison can be audited and reproduced.
    # These are NOT refitted using validation or test observations.
    import pickle
    preprocessing_path = os.path.join(exp_dir, "PLELog_train_fitted_preprocessing.pkl")
    with open(preprocessing_path, "wb") as preprocessing_file:
        pickle.dump({
            "event_embeddings_from_train": dict(processor.embedding),
            "semantic_embedding_audit": semantic_audit,
            "oov_embedding": np.asarray(unknown_vector).copy(),
            "scaler_fitted_on_train": scaler,
            "fastica_fitted_on_train": transformer,
            "reduce_dimension": int(reduce_dimension),
            "random_seed": int(RANDOM_SEED),
            "online_window_max": int(online_window),
            "online_min_events": int(online_min_events),
            "decision_threshold": float(threshold),
            "no_prefix_training": True,
        }, preprocessing_file, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved TRAIN-fitted preprocessing:", os.path.abspath(preprocessing_path))

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

    # Approved protocol: no prefix augmentation after probabilistic labeling.
    # Full-sequence PLELog training data and objective are unchanged.

    # ---------------- Model ----------------
    vocab = Vocab()
    vocab.load_from_dict(processor.embedding)

    label2id = {'Normal': 0, 'Anomaly': 1}
    plelog = PLELog(vocab, num_layer, lstm_hiddens, label2id)
    plelog.anomaly_id = label2id['Anomaly']

    best_model_file = os.path.join(output_model_dir, 'best.pt')
    last_model_file = os.path.join(output_model_dir, 'last.pt')

    print("\n[5/6] Original PLELog training (shared by both evaluation modes)")
    # ========================= TRAIN =========================
    Estimated_training_time = 0.0
    if mode == 'train':
        optimizer = Optimizer(filter(lambda p: p.requires_grad, plelog.model.parameters()))
        bestF = float("-inf")  # Save a checkpoint even if all dev F1 values are zero
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

    print("\n[6/6] Final test evaluations (one DEV-selected checkpoint)")
    if not os.path.exists(best_model_file):
        raise RuntimeError("DEV-selected best.pt missing: " + os.path.abspath(best_model_file))
    plelog.model.load_state_dict(torch.load(best_model_file, map_location=device))
    plelog.model.eval()

    comparison = {}
    report_rows = []
    if variant in ("standard", "both"):
        print("\n[STANDARD] Full-sequence inference")
        start = time.time()
        m = plelog.evaluate_all_metrics(
            instances=test, threshold=threshold,
            fp_unit_cost=10.0, fn_unit_cost=20.0, delay_unit_cost=5.0,
            json_save_path=os.path.join(exp_dir, "PLELog_Standard_metrics.json"),
            txt_save_path=os.path.join(exp_dir, "PLELog_Standard_metrics.txt"),
            csv_save_path=os.path.join(exp_dir, "PLELog_Standard_predictions.csv"),
            model_name="PLELog-Standard",
        )
        m["protocol"] = "Full-sequence original PLELog"
        m["evaluation_time_min"] = (time.time() - start) / 60.0
        comparison["PLELog-Standard"] = m
        report_rows.append(("Standard", m))

    if variant in ("online", "both"):
        print("\n[ONLINE] Bounded-window sequential event decisions")
        start = time.time()
        confirmed_k = calibrate_online_confirmation(
            dev=dev, plelog=plelog, encoder=encoder,
            scaler=scaler, transformer=transformer,
            threshold=threshold, window_size=online_window,
            min_events=online_min_events,
            candidates=args.online_confirm_candidates,
            batch_size=args.online_batch_size,
            save_path=os.path.join(exp_dir, 'PLELog_Online_validation_alert_rule.json'),
        )
        m = evaluate_online_batched_confirmed(
            plelog=plelog, instances=test, encoder=encoder,
            scaler=scaler, transformer=transformer,
            threshold=threshold, window_size=online_window,
            min_events=online_min_events, confirmations=confirmed_k,
            batch_size=args.online_batch_size,
            sequence_csv=os.path.join(exp_dir, "PLELog_Online_predictions.csv"),
            event_csv=os.path.join(exp_dir, "PLELog_Online_event_decisions.csv"),
            json_path=os.path.join(exp_dir, "PLELog_Online_metrics.json"),
            txt_path=os.path.join(exp_dir, "PLELog_Online_metrics.txt"),
        )
        m["evaluation_time_min"] = (time.time() - start) / 60.0
        comparison["PLELog-Online"] = m
        report_rows.append(("Online", m))

    def fmt(value):
        return "N/A" if value is None else f"{float(value):.6f}"
    keys = [
        ("precision", "Precision"), ("recall", "Recall"),
        ("f1_score", "F1"), ("fpr", "FPR"),
        ("average_detection_ratio", "AvgRatio"),
        ("edr_25", "EDR@25"), ("edr_50", "EDR@50"),
        ("edr_75", "EDR@75"),
        ("false_positive_total_cost", "FP Cost"),
        ("false_negative_total_cost", "FN Cost"),
        ("delay_total_cost", "Delay Cost"),
        ("total_cost", "Total Cost"),
    ]
    lines = ["=" * 76, f"PLELog {dataset} - ORIGINAL TRAINING, STANDARD vs ONLINE",
             "=" * 76, f"{'Metric':<23} {'Standard':>23} {'Online':>23}", "-" * 76]
    for key, name in keys:
        lines.append(f"{name:<23} " +
                     f"{fmt(comparison['PLELog-Standard'].get(key)) if 'PLELog-Standard' in comparison else 'N/A':>23} " +
                     f"{fmt(comparison['PLELog-Online'].get(key)) if 'PLELog-Online' in comparison else 'N/A':>23}")
    lines += ["=" * 76, "Training time (minutes): " + fmt(Estimated_training_time),
              "Model: " + os.path.abspath(best_model_file),
              "Results: " + os.path.abspath(exp_dir)]
    table = "\n".join(lines)
    print("\n" + table)
    with open(os.path.join(exp_dir, "PLELog_Standard_vs_Online.txt"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    with open(os.path.join(exp_dir, "PLELog_Standard_vs_Online.json"), "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print("\nFinished. Note: online sequence scores aggregate irreversible window-alert decisions;")
    print("Online uses immutable causal window scores and validation-calibrated confirmations.")
    print("Window outputs are NOT proven event-specific anomaly labels; event-level evaluation")
    print("needs event-level ground truth and a declared attribution rule.")
