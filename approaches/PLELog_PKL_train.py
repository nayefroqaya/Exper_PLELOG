"""Train the ORIGINAL PLELog AttGRU on trusted PKL splits, saving reproducible artifacts.

Original pipeline: GloVe template TF-IDF -> Sequential_TF -> FastICA ->
original HDBSCAN probabilistic labels -> original AttGRU + BCELoss + Optimizer.
No prefix augmentation. No validation/test data used to construct vocabulary/embeddings.
Requires real event-template text in training PKL (default EventTemplate) or an
explicit EventId/Template CSV; EventId alone is NOT sufficient for TF-IDF.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import FastICA


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset_name', default='SP_150MB_ratio')
    p.add_argument('--data_root', default='../../LWADLS/datasets')
    p.add_argument('--repo-root', default='..')
    p.add_argument('--output-dir', default=None)
    p.add_argument('--event-column', default='EventId')
    p.add_argument('--id-column', default='Node_block_id')
    p.add_argument('--label-column', default='Original_Label')
    p.add_argument('--template-column', default='EventTemplate')
    p.add_argument('--templates-csv', default=None, help='CSV with EventId,EventTemplate if PKLs lack template text')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=100)
    p.add_argument('--hidden-size', type=int, default=100)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--reduce-dimension', type=int, default=50, help='-1 disables FastICA (original option)')
    p.add_argument('--min-cluster-size', type=int, default=100)
    p.add_argument('--min-samples', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--eval-every', type=int, default=500, help='Original dev-check frequency in batches')
    p.add_argument('--run-online', action='store_true', help='Evaluate after training (requires my_plelog_adapter.py)')
    p.add_argument('--window', type=int, default=120)
    args = p.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA requested but unavailable')
    if args.epochs < 1 or args.batch_size < 1 or args.window < 1:
        p.error('epochs, batch-size and window must be positive')
    root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(root))
    # The original PLELog modules and training algorithms are imported, not replaced.
    from entities.instances import Instance
    from representations.templates.statistics import Simple_template_TF_IDF
    from representations.sequences.statistics import Sequential_TF
    from preprocessing.AutoLabeling import Probabilistic_Labeling
    from module.Optimizer import Optimizer
    from module.Common import data_iter, generate_tinsts_binary_label
    from models.gru import AttGRUModel
    from utils.Vocab import Vocab
    from PLELog_PKL_online_evaluation import (build_split_paths, read_sequences,
                                               check_disjoint, parse_label)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    paths = build_split_paths(Path(args.data_root), args.dataset_name)
    splits = {key: read_sequences(path, args.id_column, args.event_column, args.label_column)
              for key, path in paths.items()}
    check_disjoint(splits)
    print('Validated sequence counts:', {k: len(v) for k, v in splits.items()}, flush=True)
    out = Path(args.output_dir or ('plelog_runs/' + args.dataset_name)).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config['split_paths'] = {k: str(v.resolve()) for k, v in paths.items()}
    (out / 'training_config.json').write_text(json.dumps(config, indent=2), encoding='utf8')

    # Training-derived templates only. Do not use templates taken from validation/test.
    train_df = pd.read_pickle(paths['train'])  # trusted local file ONLY
    if args.templates_csv:
        table = pd.read_csv(args.templates_csv)
    else:
        table = train_df
    for name in (args.event_column, args.template_column):
        if name not in table.columns:
            raise ValueError(f'Missing {name!r} in training data/template CSV. '
                             'Original PLELog needs parsed template TEXT. '
                             'Provide --templates-csv with EventId,EventTemplate; do not substitute EventId as text.')
    template_pairs = table[[args.event_column, args.template_column]].copy()
    if template_pairs.isna().any().any():
        raise ValueError('Null template IDs or template text found')
    template_pairs[args.event_column] = template_pairs[args.event_column].astype(str)
    template_pairs[args.template_column] = template_pairs[args.template_column].astype(str)
    if (template_pairs[args.template_column].str.strip() == '').any():
        raise ValueError('Empty template text found')
    conflicting = template_pairs.groupby(args.event_column)[args.template_column].nunique()
    if (conflicting > 1).any():
        raise ValueError(f'One EventId maps to different templates: {conflicting[conflicting > 1].head().to_dict()}')
    id2templates = dict(template_pairs.drop_duplicates(args.event_column).itertuples(index=False, name=None))
    train_events = {str(e) for s in splits['train'] for e in s.events}
    missing = train_events.difference(id2templates)
    if missing:
        raise ValueError(f'Missing templates for training EventIds: {sorted(missing)[:12]}')
    id2templates = {k: id2templates[k] for k in sorted(train_events)}
    glove_path = root / 'datasets' / 'glove.6B.300d.txt'
    if not glove_path.is_file():
        raise FileNotFoundError(f'Original PLELog template encoder requires GloVe: {glove_path}')
    print('Building original GloVe TF-IDF template embeddings...', flush=True)
    embedding = Simple_template_TF_IDF().present(id2templates)
    if not embedding or any(not np.isfinite(x).all() for x in embedding.values()):
        raise ValueError('Invalid training template embeddings')
    vocab = Vocab()
    vocab.load_from_dict(embedding)
    word2id = {str(k): int(v) for k, v in vocab._word2id.items()}
    (out / 'plelog_word2id.json').write_text(json.dumps(word2id, indent=2), encoding='utf8')
    (out / 'label2id.json').write_text(json.dumps({'Normal': 0, 'Anomaly': 1}, indent=2), encoding='utf8')
    (out / 'train_id2templates.json').write_text(json.dumps(id2templates, indent=2), encoding='utf8')

    def to_instances(sequences):
        instances = []
        for seq in sequences:
            inst = Instance(block_id=seq.sequence_id, log_sequence=[str(e) for e in seq.events],
                            label='Anomaly' if seq.anomalous else 'Normal')
            instances.append(inst)
        return instances

    train = to_instances(splits['train'])
    dev = to_instances(splits['val'])
    train_reprs = Sequential_TF(embedding).present(train)
    for i, inst in enumerate(train):
        inst.repr = train_reprs[i]
    if args.reduce_dimension != -1:
        components = args.reduce_dimension
        if components > min(train_reprs.shape):
            raise ValueError(f'FastICA dimension {components} exceeds representation shape {train_reprs.shape}')
        print('Original FastICA dimension:', components, flush=True)
        reduced = FastICA(n_components=components, random_state=args.seed).fit_transform(train_reprs)
        for inst, r in zip(train, reduced):
            inst.repr = r

    # As in the original source: first half of normal TRAINING sequences seed labeling.
    normal_ids = [i for i, inst in enumerate(train) if inst.label == 'Normal']
    normal_ids = normal_ids[:int(0.5 * len(normal_ids))]
    if not normal_ids:
        raise ValueError('No normal training sequences available for probabilistic labeling')
    print(f'Original probabilistic labeling on {len(train)} training sequences; '
          f'{len(normal_ids)} known-normal seeds', flush=True)
    labeler = Probabilistic_Labeling(
        min_samples=args.min_samples, min_clust_size=args.min_cluster_size,
        res_file=str(out / 'probabilistic_labels.txt'),
        rand_state_file=str(out / 'probabilistic_random_state.pkl'))
    labeled_train = labeler.auto_label(train, normal_ids)
    # Common.py has a fallback from predicted->ground-truth; prohibit that path for training.
    for inst in labeled_train:
        if getattr(inst, 'predicted', None) not in ('Normal', 'Anomaly'):
            raise ValueError(f'Probabilistic labeling invalid for {inst.id}: {getattr(inst, "predicted", None)!r}')
        if getattr(inst, 'confidence', None) is None:
            raise ValueError(f'Probabilistic labeling missing confidence for {inst.id}')
    for inst in dev:
        inst.predicted = inst.label  # Only used by existing tensor helper for evaluation target.
        inst.confidence = 0.0

    # Keep the original GRU + BCE on soft pseudo-labels and original Optimizer.
    model = AttGRUModel(vocab, args.layers, args.hidden_size)
    model.to(torch.device(args.device))
    original_optimizer = Optimizer(filter(lambda param: param.requires_grad, model.parameters()))
    best_f1 = -1.0
    best_path, last_path = out / 'model_best.pt', out / 'model_last.pt'
    device = torch.device(args.device)

    def batch_tensors(batch, eval_mode=False):
        t = generate_tinsts_binary_label(batch, vocab, eval_mode)
        # Use the original project's TensorInstances layout, preserve targets.
        # Original CPUEmbedding keeps its lookup weights on CPU.
        words = torch.as_tensor(t.src_words, dtype=torch.long, device="cpu")
        masks = torch.as_tensor(t.src_masks, dtype=torch.float32, device=device)
        lengths = torch.as_tensor(t.word_len, dtype=torch.long, device=device)
        targets = torch.as_tensor(t.tags, dtype=torch.float32, device=device)
        return (words, masks, lengths), targets

    def eval_dev():
        model.eval()
        tp = tn = fp = fn = 0
        with torch.inference_mode():
            for i in range(0, len(dev), args.batch_size):
                batch = dev[i:i + args.batch_size]
                inputs, _ = batch_tensors(batch, True)
                probs = torch.softmax(model(inputs), dim=1)[:, 1].detach().cpu().numpy()
                for inst, prob in zip(batch, probs):
                    pred = prob >= 0.5
                    actual = inst.label == 'Anomaly'
                    tp += bool(pred and actual)
                    tn += bool(not pred and not actual)
                    fp += bool(pred and not actual)
                    fn += bool(not pred and actual)
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn, 'F1': f1,
                'precision': prec, 'recall': rec}

    history = []
    for epoch in range(args.epochs):
        model.train()
        np.random.shuffle(labeled_train)
        for batch_ix, start in enumerate(range(0, len(labeled_train), args.batch_size), 1):
            batch = labeled_train[start:start+args.batch_size]
            inputs, targets = batch_tensors(batch)
            # Original forward loss: softmax then binary cross entropy.
            loss = torch.nn.functional.binary_cross_entropy(torch.softmax(model(inputs), dim=1), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda x: x.requires_grad, model.parameters()), 1)
            original_optimizer.step()
            model.zero_grad()
            if batch_ix % 100 == 0:
                print(f'epoch={epoch+1} batch={batch_ix} loss={loss.item():.5f}', flush=True)
            if dev and (batch_ix % args.eval_every == 0):
                metrics = eval_dev()
                print('Validation:', metrics, flush=True)
                if metrics['F1'] > best_f1:
                    best_f1 = metrics['F1']
                    torch.save(model.state_dict(), best_path)
                model.train()
        metrics = eval_dev()
        print(f'Epoch {epoch+1}/{args.epochs} validation:', metrics, flush=True)
        history.append({'epoch': epoch+1, **metrics})
        if metrics['F1'] > best_f1:
            best_f1 = metrics['F1']
            torch.save(model.state_dict(), best_path)
        torch.save(model.state_dict(), last_path)
        model.train()
    (out / 'training_history.json').write_text(json.dumps(history, indent=2), encoding='utf8')
    print('Saved checkpoint:', best_path)
    print('Saved word2id:', out / 'plelog_word2id.json')
    print('Next run:')
    cmd = [sys.executable, 'PLELog_PKL_online_evaluation.py',
           '--dataset_name', args.dataset_name, '--data_root', args.data_root,
           '--repo-root', str(root), '--model-checkpoint', str(best_path),
           '--word2id-json', str(out / 'plelog_word2id.json'),
           '--label-column', args.label_column, '--event-column', args.event_column,
           '--id-column', args.id_column, '--device', args.device,
           '--window', str(args.window), '--output', str(out / 'online_metrics.json')]
    print(' '.join(cmd), flush=True)
    if args.run_online:
        import subprocess
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
