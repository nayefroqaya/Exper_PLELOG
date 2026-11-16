"""Frozen PLELog inference bridge, preserving the repository's AttGRUModel.

Requires a checkpoint AND exact training-time word-to-ID mapping.
Create the mapping while training, e.g.:
    import json
    json.dump(vocab._word2id, open('plelog_word2id.json', 'w'))

This adapter intentionally refuses to reconstruct the vocabulary from test
samples: that would change embedding-to-event alignment and invalidate results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

_model = None
_word2id = None
_device = None
_anomaly_id = 1


def configure(*, checkpoint: str, word2id_json: str, repo_root: str = '.',
              anomaly_id: int = 1, device: str = 'cpu') -> None:
    global _model, _word2id, _device, _anomaly_id
    import numpy as np
    import torch

    root = str(Path(repo_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from models.gru import AttGRUModel  # original architecture, not reimplemented

    _device = torch.device(device)
    mapping = json.loads(Path(word2id_json).read_text(encoding='utf8'))
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError('word2id JSON must be a nonempty token-to-index dictionary')
    _word2id = {str(k): int(v) for k, v in mapping.items()}
    if any(v < 0 for v in _word2id.values()) or len(set(_word2id.values())) != len(_word2id):
        raise ValueError('word2id indices must be distinct nonnegative integers')
    if sorted(_word2id.values()) != list(range(len(_word2id))):
        raise ValueError('word2id indices must be consecutive from zero')
    if _word2id.get('<oov>') != 3 or _word2id.get('<pad>') != 0:
        raise ValueError('Expected original PLELog special IDs <pad>=0 and <oov>=3')

    # Load only trusted local checkpoints. A standard PLELog checkpoint holds a state_dict.
    try:
        weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    except TypeError as exc:
        raise RuntimeError('Use a recent version of PyTorch supporting safe weights_only loading') from exc
    if not isinstance(weights, dict) or 'word_embed.weight' not in weights:
        raise ValueError('Checkpoint must be the unmodified PLELog state_dict containing word_embed.weight')
    embed = weights['word_embed.weight'].detach().cpu()
    vocab_size, dims = embed.shape
    if vocab_size != len(_word2id):
        raise ValueError(f'Checkpoint vocab size {vocab_size} differs from saved mapping {len(_word2id)}')
    gru_weight = weights['rnn.weight_ih_l0']
    if gru_weight.shape[1] != dims or gru_weight.shape[0] % 3:
        raise ValueError('Unexpected GRU weight dimensions in checkpoint')
    hidden = int(gru_weight.shape[0] // 3)
    layers = sum(f'rnn.weight_ih_l{i}' in weights for i in range(100))
    if layers < 1:
        raise ValueError('No GRU layers in checkpoint')
    # Model constructor requires only these vocab fields, not label-related state.
    vocab = SimpleNamespace(vocab_size=vocab_size, word_dim=dims,
                            embeddings=embed.numpy().astype(np.float32, copy=True))
    model = AttGRUModel(vocab, layers, hidden)
    model.load_state_dict(weights, strict=True)
    model.to(_device)
    model.eval()
    if anomaly_id not in (0, 1):
        raise ValueError('anomaly_id must be 0 or 1; verify your original training label2id')
    _anomaly_id = anomaly_id
    _model = model


def score_anomaly(events: tuple) -> float:
    """Score ONLY the supplied causal history, without modifying any model state."""
    import torch
    if _model is None or _word2id is None:
        raise RuntimeError('Call configure(checkpoint=..., word2id_json=..., repo_root=...) first')
    if not events:
        raise ValueError('Cannot score an empty event history')
    ids = [_word2id.get(str(e), 3) for e in events]
    words = torch.tensor([ids], dtype=torch.long, device=_device)
    masks = torch.ones((1, len(ids)), dtype=torch.float32, device=_device)
    word_len = torch.tensor([len(ids)], dtype=torch.long, device=_device)
    with torch.inference_mode():
        logits = _model((words, masks, word_len))
        probs = torch.softmax(logits, dim=1)
    return float(probs[0, _anomaly_id].item())
