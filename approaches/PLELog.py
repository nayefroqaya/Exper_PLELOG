import sys
print("Imported _1 ")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # ⛔ Disable GPU completely
import colorama
colorama.init()
sys.path.extend([".", ".."])
from CONSTANTS import *
print("Imported _2 ")
import time
from utils.common import get_precision_recall
import shutil

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
    argparser.add_argument('--dataset', default='SP_150MB', type=str)
    argparser.add_argument('--mode', default='train', type=str)
    argparser.add_argument('--parser', default='IBM', type=str)
    argparser.add_argument('--min_cluster_size', type=int, default=100)
    argparser.add_argument('--min_samples', type=int, default=100)
    argparser.add_argument('--reduce_dimension', type=int, default=100)
    argparser.add_argument('--threshold', type=float, default=0.5)
    args, _ = argparser.parse_known_args()

    dataset = args.dataset
    parser = args.parser
    mode = args.mode
    min_cluster_size = args.min_cluster_size
    min_samples = args.min_samples
    reduce_dimension = args.reduce_dimension
    threshold = args.threshold

    # ---------------- Paths ----------------
    PROJECT_ROOT = '.'  # adjust as needed
    #save_dir = os.path.join(PROJECT_ROOT, 'outputs')
    #output_model_dir = os.path.join(save_dir, f'models/PLELog/{dataset}_{parser}/model')
    #prob_label_res_file = os.path.join(save_dir, f'results/PLELog/{dataset}_{parser}/prob_label_res/mcs-{min_cluster_size}_ms-{min_samples}')
    #rand_state = os.path.join(save_dir, f'results/PLELog/{dataset}_{parser}/prob_label_res/random_state')
    #os.makedirs(output_model_dir, exist_ok=True)
    # ---------------- Paths ----------------
    # Base outputs directory
    save_dir = os.path.join(PROJECT_ROOT, 'outputs')

    # Base experiment directory
    exp_dir = os.path.join(save_dir, 'results', 'PLELog', f'{dataset}_{parser}')

    # Sub-directories
    output_model_dir = os.path.join(exp_dir, 'model')
    prob_label_res_dir = os.path.join(exp_dir, 'prob_label_res')

    # CREATE REQUIRED DIRECTORIES
    os.makedirs(output_model_dir, exist_ok=True)
    os.makedirs(prob_label_res_dir, exist_ok=True)

    # ---------------- Create directories ----------------
    os.makedirs(output_model_dir, exist_ok=True)
    os.makedirs(os.path.dirname(prob_label_res_dir), exist_ok=True)
    # ---------------- Load PKL ----------------
    # first paper :
    #train_pkl = f'../datasets/{dataset}/1_{dataset}_Splitted_Datasets/train_df.pkl'
    #dev_pkl   = f'../datasets/{dataset}/1_{dataset}_Splitted_Datasets/val_df.pkl'
    #test_pkl  = f'../datasets/{dataset}/1_{dataset}_Splitted_Datasets/test_df.pkl'



    # second
    train_pkl = '../../NovaAD_Plus/datasets/SP_150MB/1_SP_150MB_Splitted_Datasets/train_df.pkl'
    test_pkl = '../../NovaAD_Plus/datasets/SP_150MB/1_SP_150MB_Splitted_Datasets/test_df.pkl'
    dev_pkl = '../../NovaAD_Plus/datasets/SP_150MB/1_SP_150MB_Splitted_Datasets/val_df.pkl'

    #PLELog.clear_folder(save_dir)

    processor = PKLPreprocessor()
    train, dev, test = processor.load_pkl(dataset, train_pkl, dev_pkl, test_pkl)

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

    # Paths to old probabilistic labeling results
    prob_label_res_file = os.path.join(save_dir,
                                       f'results/PLELog/{dataset}_{parser}/prob_label_res/mcs-{min_cluster_size}_ms-{min_samples}')
    rand_state_file = os.path.join(save_dir, f'results/PLELog/{dataset}_{parser}/prob_label_res/random_state')

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
        plelog.model.load_state_dict(torch.load(last_model_file))
        start = time.time()
        p, r, f = plelog.evaluate(test, threshold)
        runtime = (time.time() - start) / 60
        results['LAST'] = (p, r, f, runtime)

    if os.path.exists(best_model_file):
        plelog.model.load_state_dict(torch.load(best_model_file))
        start = time.time()
        p, r, f = plelog.evaluate(test, threshold)
        runtime = (time.time() - start) / 60
        results['BEST'] = (p, r, f, runtime)

    # ========================= COMPARE =========================
    print("\n=========== FINAL TEST RESULTS ===========")
    for k, (p, r, f, t) in results.items():
        print(f"{k} MODEL | Precision={p:.4f} Recall={r:.4f} F1={f:.4f} Time={t:.2f} min")

    winner = max(results.items(), key=lambda x: x[1][2])[0]
    print(f"\n🏆 Best model on TEST set: {winner}")
    print("=========================================")
    print(f"\nTotal training time: {Estimated_training_time:.2f} minutes")

    print("All Finished ✅")

