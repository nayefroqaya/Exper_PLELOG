# PLELog Standard + Online: professor-approved experimental protocol

This package adapts the uploaded `PLELog_Standard_then_Early_Reviewer(4).py`. It is a **source-level adaptation**, not a completed experimental run. Full training and test inference have **not** been executed here because the PLELog project dependencies, dataset PKLs, and original `AttGRUModel`, `Sequential_TF`, `Vocab`, and `PKLPreprocessor` implementations were not uploaded.

## Exactly two evaluation cases

| Case | Training | Inference | Output |
| --- | --- | --- | --- |
| **PLELog-Standard** | Original PLELog pseudo-labeling + full-sequence training | Predict once after the complete test sequence | Sequence-level classification |
| **PLELog-Online** | **The same original training and checkpoint** | For each arriving event, score only the currently retained bounded window (default max **120 events**), emit an irrevocable window-alert decision, and discard older observations | Event-by-event decisions + sequence-level first-alert summary |

There is **no prefix augmentation at 25%, 50%, or 75%**, no ever-growing-prefix re-evaluation, no online learning, and no use of test labels in the online model inputs.

**Scientific qualification:** PLELog is originally a *sequence-level classifier*. Applied to a bounded window, its output is an **anomaly decision about the current window**, emitted at the latest event. It is **not** an independently validated event-label classifier. This is a causal replay of online inference over stored sequences, **not a proven low-latency production streaming implementation**. The external PLELog dependencies must be checked to confirm that no information outside the supplied window is accessed internally.

## Install in your existing PLELog repository

Copy `PLELog_Standard_Online_REVISED.py` to the PLELog project directory where you normally run your original `.py` file. Continue using the same Python environment.

Run all of the experiments with one full training run and both evaluations (recommended):

```bash
python PLELog_Standard_Online_REVISED.py \
  --dataset SP_150MB_ratio --case in_domain --mode train --variant both \
  --online_window 120 --online_min_events 1 \
  --data_root ../../LWADLS/datasets
```

To run only the original full-sequence evaluation with its original training:

```bash
python PLELog_Standard_Online_REVISED.py --dataset BGL --case in_domain --mode train --variant standard
```

To train using the same original protocol but run only bounded-window online inference:

```bash
python PLELog_Standard_Online_REVISED.py --dataset BGL --case in_domain --mode train --variant online --online_window 120
```

**For a directly controlled comparison, use `--variant both`.** Both cases then share the exact same trained checkpoint, preprocessing, threshold, train/dev/test split and pseudo-labeling. Running `--variant standard` and `--variant online` separately retrains the model, which can introduce run-to-run differences.

**Runtime caution:** Default online evaluation makes one model prediction per test event; this can be computationally expensive on large datasets. Start with `SP_150MB_ratio` and review the outputs before running BGL, HDFS, and Thunderbird. Do not increase an evaluation stride to skip events without explicitly changing the online evaluation protocol.

## Output files

Output root (automatically created):

`datasets/<dataset>/PLELog_results/PLELog_OriginalTrain_OnlineComparison/<dataset>_<parser>/`

| File | Contents |
| --- | --- |
| `model/best.pt` and `model/last.pt` | Trained original PLELog checkpoints |
| `PLELog_Standard_metrics.json` / `.txt` | Original full-sequence metrics |
| `PLELog_Standard_predictions.csv` | One complete-sequence prediction per test sequence |
| `PLELog_Online_metrics.json` / `.txt` | Bounded-window online metrics |
| `PLELog_Online_predictions.csv` | Per-sequence classification and first-alert position derived from irrevocable window alarms |
| `PLELog_Online_event_decisions.csv` | One emitted window alarm/normal decision **per event** (or explicit warm-up record) |
| `PLELog_Standard_vs_Online.json` / `.txt` | Final side-by-side results |

The final terminal report includes **Precision, Recall, F1, FPR, AvgRatio, EDR@25/50/75, FP/FN/delay/total costs**. Each test sequence's true label is used **only for evaluation**. The per-event model input receives a neutral placeholder label.

## Exact metrics and important limitations

* **Sequence-level F1/P/R/FPR**: Standard evaluates the complete sequence; Online maps each sequence to positive if **any** irreversible anomalous window alert was emitted in that sequence. *This mapping does not revise or retract emitted decisions.* The event-level stream is retained so the mapping can be audited.
* **AvgRatio**: First anomalous alert step divided by that sequence's length, averaged only over correctly detected anomalous sequences.
* **EDR@25/50/75**: Fraction of **all anomalous test sequences** first alerted at or before that proportion of their sequence. These rates need to be reconciled with the RLADLS evaluation's definition before final paper comparisons.
* **Costs**: C_FP=10·FP, C_FN=20·FN, C_D=5·sum(first_alert_step/sequence_length for correctly alerted anomalous sequences). Total is their sum. These are *sequence-level* costs, **not** a penalty for each emitted event-level false alarm.
* **Event-level F1 is NOT computed** from sequence-level labels; true event-onset annotations would be needed for event-level correctness.
* **Threshold**: Fixed original 0.5 default (or `--threshold`). No validation-selected threshold search has been introduced.
* **History**: Online uses a FIFO window of max 120 events, which resets at the original sequence boundary. The first window decision is emitted at the first event by default. If PLELog's original preprocessing cannot score windows that short, an explicit `--online_min_events` warm-up must be justified and documented; the script raises a descriptive error rather than silently treating unsuccessful inferences as normal.

## Changes compared with the uploaded source

1. Removed prefix augmentation from the active training path; the original PLELog pseudo-labeling, optimizer, architecture, learning objective and DEV checkpoint selection remain in place.
2. Removed growing-prefix inference from the active evaluation path. A fixed-memory FIFO window is scored as each event arrives, one emission per event.
3. Preserved **all** event decisions in CSV even after an anomaly alert; previous event decisions cannot be revised.
4. Saved per-sequence predictions in both modes and printed/saved the final metric comparison.
5. Built the embedding vocabulary from **training event IDs only**; unseen events map to one fixed zero vector instead of constructing the training vocabulary using validation and test event IDs as the uploaded source did. This **changes the original preprocessing**, although not the model architecture or learning objective; it is necessary to avoid transductive access to future test data. **Compatibility with `Sequential_TF` and `Vocab` must be verified on the actual PLELog project.**
6. Seeded Python/NumPy/PyTorch and made training embedding initialization independent of set iteration order; runs can still differ due to platform/GPU or external dependencies.
7. The original uploaded `cross_dataset` branch has undefined source settings. This revision deliberately rejects `--case cross_dataset` instead of producing an unreliable experiment.

## What was validated here

- Python syntax compilation: **PASS**.
- Synthetic isolated evaluation test: **PASS** for sequence TP/TN/FP/FN, F1, FPR, EDR, AvgRatio, delay costs, fixed memory bound, irreversible emissions, and predictions invariant to changed ground-truth labels.
- **NOT validated:** training on real PKL data; compatibility with external `Vocab`, `Sequential_TF`, and `AttGRUModel` interfaces; memory/runtime performance on the full test sets; equivalence to PLELog's released implementation; actual reported scientific results.

If an error occurs on your server, please send its traceback **and** the project implementation of `representations/sequences/statistics.py` and `utils/Vocab.py` so we can fix the exact compatibility issue without changing the experimental protocol.
