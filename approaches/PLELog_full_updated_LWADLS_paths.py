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
    argparser.add_argument('--mode', default='train', type=str)
    argparser.add_argument('--parser', default='IBM', type=str)
    argparser.add_argument('--min_cluster_size', type=int, default=100)
    argparser.add_argument('--min_samples', type=int, default=100)
    argparser.add_argument('--reduce_dimension', type=int, default=100)
    argparser.add_argument('--threshold', type=float, default=0.5)
    argparser.add_argument('--case', default='in_domain', type=str, choices=['in_domain', 'cross_dataset'])
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
    save_dir = os.path.join(DATASETS_ROOT, dataset, "PLELog_results")
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
    SOURCE_DATASETS = ["HDFS", "TH_1G", "BGL"]
    TARGET_NORMAL_FRACTION = 0.20
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

        train, dev, test = processor.load_pkl(TARGET_DATASET, paths["train_pkl"], paths["dev_pkl"], paths["test_pkl"])

        print("\n==============================")
        print("In-domain experiment")
        print("==============================")
        print("Dataset:", TARGET_DATASET)
        print(f"Loaded {len(train)} train / {len(dev)} dev / {len(test)} test")


    elif CASE == "cross_dataset":

        all_source_train = []
        all_source_dev = []

        print("\n==============================")
        print("Cross-dataset experiment")
        print("==============================")
        print("Sources:", SOURCE_DATASETS)
        print("Target:", TARGET_DATASET)
        print("Target normal fraction:", TARGET_NORMAL_FRACTION)

        # ------------------------------------------------------------
        # Load source datasets
        # ------------------------------------------------------------

        for source_dataset in SOURCE_DATASETS:
            source_paths = DATASETS[source_dataset]

            source_train, source_dev, source_test = processor.load_pkl(source_dataset, source_paths["train_pkl"],
                source_paths["dev_pkl"], source_paths["test_pkl"])

            all_source_train.extend(source_train)
            all_source_dev.extend(source_dev)

            print("\nLoaded source dataset:", source_dataset)
            print("Source train:", len(source_train))
            print("Source dev:", len(source_dev))
            print("Source test not used:", len(source_test))

        # ------------------------------------------------------------
        # Load target dataset
        # ------------------------------------------------------------

        target_paths = DATASETS[TARGET_DATASET]

        target_train, target_dev, target_test = processor.load_pkl(TARGET_DATASET, target_paths["train_pkl"],
            target_paths["dev_pkl"], target_paths["test_pkl"])

        # ------------------------------------------------------------
        # Take only Normal from target train
        # ------------------------------------------------------------

        target_normal_train = []

        for inst in target_train:
            if inst.label == "Normal":
                target_normal_train.append(inst)

        number_to_take = int(len(target_normal_train) * TARGET_NORMAL_FRACTION)

        target_normal_sample = random.sample(target_normal_train, number_to_take)

        # ------------------------------------------------------------
        # Final data
        # ------------------------------------------------------------

        train = all_source_train + target_normal_sample

        dev = all_source_dev

        # Important:
        # target test is untouched
        test = target_test

        print("\n==============================")
        print("Final cross-dataset data")
        print("==============================")
        print("Source train:", len(all_source_train))
        print("Source dev:", len(all_source_dev))
        print("Target train:", len(target_train))
        print("Target normal train:", len(target_normal_train))
        print("Target normal used:", len(target_normal_sample))
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

    # ---------------- Embeddings ----------------
    all_event_ids = set()
    for inst in train + dev + test:
        all_event_ids.update(inst.sequence)

    embedding_dim = 50
    processor.embedding = {eid: np.random.rand(embedding_dim) for eid in all_event_ids}

    # ---------------- Sequence representation ----------------
    encoder = Sequential_TF(processor.embedding)
    for inst, vec in zip(train, encoder.present(train)):
        inst.repr = vec

    for inst, vec in zip(dev, encoder.present(dev)):
        inst.repr = vec

    for inst, vec in zip(test, encoder.present(test)):
        inst.repr = vec
    # ---------------- Dimension Reduction (FastICA) ----------------
    train_reprs = np.array([inst.repr for inst in train])
    dev_reprs = np.array([inst.repr for inst in dev])  # <-- Added
    test_reprs = np.array([inst.repr for inst in test])

    transformer = None
    if reduce_dimension != -1:
        start_time = time.time()
        print(f"Start FastICA, target dimension: {reduce_dimension}")

        # Add small noise to avoid singular matrix issues
        train_reprs += np.random.normal(0, 1e-5, train_reprs.shape)

        # Standardize
        scaler = StandardScaler()
        train_reprs = scaler.fit_transform(train_reprs)
        train_reprs = np.nan_to_num(train_reprs, nan=0.0, posinf=1e6, neginf=-1e6)

        # Fit ICA on train
        transformer = FastICA(n_components=reduce_dimension, random_state=0)
        train_reprs = transformer.fit_transform(train_reprs)

        # Assign back to train
        for idx, inst in enumerate(train):
            inst.repr = train_reprs[idx]

        # --- CHANGE / ADD: Transform dev using SAME scaler + ICA ---
        dev_reprs = scaler.transform(dev_reprs)
        dev_reprs = transformer.transform(dev_reprs)
        for idx, inst in enumerate(dev):
            inst.repr = dev_reprs[idx]  # <-- Added

        # Transform test set
        test_reprs = scaler.transform(test_reprs)
        test_reprs = transformer.transform(test_reprs)
        for idx, inst in enumerate(test):
            inst.repr = test_reprs[idx]

        print(f"Finished FastICA in {time.time() - start_time:.2f} seconds")

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
    labeled_train = label_generator.auto_label(train, normal_ids)

    # ---------------- Model ----------------
    vocab = Vocab()
    vocab.load_from_dict(processor.embedding)

    label2id = {'Normal': 0, 'Anomaly': 1}
    plelog = PLELog(vocab, num_layer, lstm_hiddens, label2id)
    plelog.anomaly_id = label2id['Anomaly']

    best_model_file = os.path.join(output_model_dir, 'best.pt')
    last_model_file = os.path.join(output_model_dir, 'last.pt')

    # ========================= TRAIN =========================
    Estimated_training_time = 0.0
    if mode == 'train':
        optimizer = Optimizer(filter(lambda p: p.requires_grad, plelog.model.parameters()))
        bestF = 0.0
        start_train = time.time()

        for epoch in range(epochs):
            plelog.model.train()
            for onebatch in data_iter(labeled_train, batch_size, True):
                tinst = generate_tinsts_binary_label(onebatch, vocab)
                tinst.to_device(device)  # only CPU
                #tinst.to_cuda(device)


                loss = plelog.forward(tinst.inputs, tinst.targets)
                loss.backward()

                nn.utils.clip_grad_norm_(plelog.model.parameters(), max_norm=1)
                optimizer.step()
                plelog.model.zero_grad()

            # ---- DEV evaluation ----
            if dev:
                p_dev, r_dev, f_dev = plelog.evaluate(dev, threshold)
                print(f"[DEV] Epoch {epoch + 1} | F1={f_dev:.4f}")

                if f_dev > bestF:
                    bestF = f_dev
                    torch.save(plelog.model.state_dict(), best_model_file)

        torch.save(plelog.model.state_dict(), last_model_file)
        train_time = (time.time() - start_train) / 60
        Estimated_training_time = train_time
        print(f"\nTotal training time: {Estimated_training_time:.2f} minutes")

    # ========================= TEST =========================
    results = {}

    if os.path.exists(last_model_file):
        plelog.model.load_state_dict(
            torch.load(
                last_model_file,
                map_location=device
            )
        )

        start = time.time()

        last_metrics = plelog.evaluate_all_metrics(
            instances=test,
            threshold=threshold,
            fp_unit_cost=10.0,
            fn_unit_cost=20.0,
            delay_unit_cost=5.0,
            json_save_path=os.path.join(
                exp_dir,
                "last_model_test_metrics.json"
            ),
            txt_save_path=os.path.join(
                DATASETS_ROOT,
                dataset,
                "last_model_test_metrics.txt"
            ),
            model_name="PLELog LAST MODEL"
        )

        runtime = (time.time() - start) / 60

        results['LAST'] = (
            last_metrics['precision'] * 100,
            last_metrics['recall'] * 100,
            last_metrics['f1_score'] * 100,
            runtime
        )

    if os.path.exists(best_model_file):
        plelog.model.load_state_dict(
            torch.load(
                best_model_file,
                map_location=device
            )
        )

        start = time.time()

        best_metrics = plelog.evaluate_all_metrics(
            instances=test,
            threshold=threshold,
            fp_unit_cost=10.0,
            fn_unit_cost=20.0,
            delay_unit_cost=5.0,
            json_save_path=os.path.join(
                exp_dir,
                "best_model_test_metrics.json"
            ),
            txt_save_path=os.path.join(
                DATASETS_ROOT,
                dataset,
                "best_model_test_metrics.txt"
            ),
            model_name="PLELog BEST MODEL"
        )

        runtime = (time.time() - start) / 60

        results['BEST'] = (
            best_metrics['precision'] * 100,
            best_metrics['recall'] * 100,
            best_metrics['f1_score'] * 100,
            runtime
        )

    if not results:
        raise RuntimeError(
            "No model checkpoint was found. Train first or check the model directory: "
            + os.path.abspath(output_model_dir)
        )

    # ========================= COMPARE =========================
    print("\n=========== FINAL TEST RESULTS ===========")
    for k, (p, r, f, t) in results.items():
        print(f"{k} MODEL | Precision={p:.4f} Recall={r:.4f} F1={f:.4f} Time={t:.2f} min")

    winner = max(results.items(), key=lambda x: x[1][2])[0]
    print(f"\n🏆 Best model on TEST set: {winner}")
    print("=========================================")
    print(f"\nTotal training time: {Estimated_training_time:.2f} minutes")

    print("All Finished ✅")

