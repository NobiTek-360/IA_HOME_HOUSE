import json
import os
import sys
import math
import random
import time
import hashlib
import io
import re
import argparse
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, desc="", total=None, **kwargs):
        total = total or len(iterable)
        for i, item in enumerate(iterable):
            if i % max(1, total // 20) == 0 or i == total - 1:
                pct = (i + 1) / total * 100
                print(f"\r{desc} {pct:.0f}% ({i+1}/{total})", end="")
            yield item
        print()

# ----------------------------------------------------------------------
# Configurações
# ----------------------------------------------------------------------
@dataclass
class NLUConfig:
    # v12: encoder contextual mais forte, sem alterar o contrato do dataset.
    embed_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 5
    num_heads: int = 4
    ff_dim: int = 384
    dropout: float = 0.18
    architecture_version: str = "v12.1-context-specialized"
    max_seq_len: int = 40
    max_bpe_merges: int = 1500
    max_token_length: int = 6
    batch_size: int = 128
    epochs: int = 120
    lr: float = 0.001
    min_lr: float = 5e-5
    weight_decay: float = 5e-3
    entity_loss_weight: float = 1.0
    operation_loss_weight: float = 1.0
    action_mode_loss_weight: float = 1.0      # novo
    target_scope_loss_weight: float = 1.0      # novo
    value_type_loss_weight: float = 1.0        # novo
    patience: int = 3
    confidence_threshold: float = 0.60
    confidence_margin_threshold: float = 0.08
    extra_chars: str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789áàâãéèêíïóôõúçÁÀÂÃÉÈÊÍÏÓÔÕÚÇ"
    
    use_char_cnn: bool = True
    max_char_len: int = 8
    char_embed_dim: int = 32
    char_num_filters: int = 64
    char_kernel_sizes: tuple = (3, 4, 5)
    use_crf: bool = True

    contrastive_weight: float = 0.05
    temperature: float = 0.10
    temperature_min: float = 0.07
    temperature_decay_epochs: int = 30
    hard_negative_weight: float = 1.50
    hard_negative_power: float = 2.0
    label_smoothing: float = 0.05
    proj_dim: int = 128

    unknown_confidence_threshold: float = 0.60
    unknown_margin_threshold: float = 0.08
    unknown_distance_threshold: float = 0.62
    unknown_require_two_signals: bool = True
    unknown_detection_enabled: bool = True

    context_max_turns: int = 5
    context_pronouns: tuple = ("ela", "ele", "isso", "isto", "essa", "esse", "aquela", "aquele", "aquilo", "lá", "ali")

    # Aprendizado de estrutura: o MLM agora mascara palavras inteiras,
    # preservando melhor as relações entre as palavras da frase.
    mlm_weight: float = 0.30
    mlm_probability: float = 0.15

    # Foco adaptativo em frases difíceis. Não altera o balanceamento por intenção:
    # apenas aumenta o peso das amostras nas quais o modelo ainda erra/confunde.
    intent_hard_example_weight: float = 1.0
    intent_focal_gamma: float = 2.0

    # Mantido por compatibilidade com checkpoints antigos. O mixup de logits
    # fica desativado porque misturar logits de intenções diferentes pode
    # destruir a estrutura semântica de frases difíceis.
    mixup_alpha: float = 0.0

# ----------------------------------------------------------------------
# Mapeamentos globais
# ----------------------------------------------------------------------
INTENTS = []
intent2idx = {}
idx2intent = {}

ENTITY_TYPES = []
entity_type2idx = {}
idx2entity_type = {}

BIO_TAGS = []
bio2idx = {}
idx2bio = {}
NUM_BIO_TAGS = 0

OPERATIONS = []
operation2idx = {}
idx2operation = {}

ACTION_MODES = []
action_mode2idx = {}
idx2action_mode = {}

TARGET_SCOPES = []
target_scope2idx = {}
idx2target_scope = {}

VALUE_TYPES = []
value_type2idx = {}
idx2value_type = {}

WORD_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def get_word_spans(text):
    return [(m.start(), m.end(), m.group()) for m in WORD_PATTERN.finditer(text)]

def merge_adjacent_entities(entities, token_to_word_idx):
    merged = []
    for ent in entities:
        word_indices = sorted(set(token_to_word_idx[t] for t in ent["tokens"] if token_to_word_idx[t] != -1))
        if not word_indices:
            continue
        ent = dict(ent)
        ent["word_indices"] = word_indices
        ent_type = ent.get("type_id", ent.get("type"))
        if merged:
            prev = merged[-1]
            prev_type = prev.get("type_id", prev.get("type"))
            if prev_type == ent_type and word_indices[0] == prev["word_indices"][-1] + 1:
                prev["tokens"] = prev["tokens"] + ent["tokens"]
                prev["word_indices"] = prev["word_indices"] + word_indices
                continue
        merged.append(ent)
    return merged

FUNCTION_WORDS = {
    "a", "o", "as", "os", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "por", "para", "com", "sem",
    "um", "uma", "uns", "umas",
    "este", "esta", "estes", "estas",
    "esse", "essa", "esses", "essas",
    "aquele", "aquela", "aquilo",
    "que", "e", "ou", "se", "ao", "à", "às", "aos",
}

SEED = 42

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def stratified_split(dataset: "NLUDataset", val_ratio: float, seed: int):
    by_intent = defaultdict(list)
    for idx in range(len(dataset)):
        by_intent[dataset.examples[idx]["intent_label"]].append(idx)
    rng = random.Random(seed)
    train_indices, val_indices = [], []
    for indices in by_intent.values():
        indices = list(indices)
        rng.shuffle(indices)
        n_val = max(1, int(round(len(indices) * val_ratio))) if len(indices) > 2 else 1
        if len(indices) <= 2:
            n_val = 1
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    if not val_indices:
        val_indices = train_indices[:1]
        train_indices = train_indices[1:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices), train_indices, val_indices

# ----------------------------------------------------------------------
# Tokenizador Híbrido
# ----------------------------------------------------------------------
class Vocab:
    def __init__(self):
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.sos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.sep_token = "<SEP>"
        self.mask_token = "<MASK>"
        self.WordToIdx = {self.pad_token: 0, self.unk_token: 1, self.sos_token: 2,
                          self.eos_token: 3, self.sep_token: 4, self.mask_token: 5}
        self.IdxToWord = {v: k for k, v in self.WordToIdx.items()}
        self.pad_idx = 0
        self.unk_idx = 1

    def add_word(self, w):
        if w not in self.WordToIdx:
            idx = len(self.WordToIdx)
            self.WordToIdx[w] = idx
            self.IdxToWord[idx] = w
        return self.WordToIdx[w]

    def size(self):
        return len(self.WordToIdx)

    def encode(self, text, tokenizer, max_len, add_special_tokens=False):
        words = text.split()
        ids = []
        for w in words:
            subwords = tokenizer.tokenize_word(w)
            for sub in subwords:
                ids.append(self.WordToIdx.get(sub, self.unk_idx))
                if len(ids) >= max_len:
                    break
            if len(ids) >= max_len:
                break
        if add_special_tokens:
            ids = [self.WordToIdx[self.sos_token]] + ids[:max_len-2] + [self.WordToIdx[self.eos_token]]
        else:
            ids = ids[:max_len]
        real_len = len(ids)
        ids = ids + [self.pad_idx] * (max_len - len(ids))
        return ids[:max_len], real_len

class SubwordTokenizer:
    def __init__(self, merges=None):
        self.Merges = merges or []
        self.MergeRank = {m: i for i, m in enumerate(self.Merges)}

    def tokenize_word(self, word):
        if not word:
            return []
        toks = list(word)
        toks[-1] += "</w>"
        while len(toks) >= 2:
            best_rank = float('inf')
            best_idx = -1
            for i in range(len(toks) - 1):
                pair = toks[i] + " " + toks[i+1]
                if pair in self.MergeRank and self.MergeRank[pair] < best_rank:
                    best_rank = self.MergeRank[pair]
                    best_idx = i
            if best_idx == -1:
                break
            merged = toks[best_idx] + toks[best_idx+1]
            toks = toks[:best_idx] + [merged] + toks[best_idx+2:]
        return toks

def train_bpe_from_texts(texts, max_merges, max_token_length, vocab, extra_chars):
    word_freqs = defaultdict(int)
    for txt in texts:
        for _, _, w in get_word_spans(txt):
            word_freqs[w] += 1
    splits = []
    for w, freq in word_freqs.items():
        if not w:
            continue
        toks = list(w)
        toks[-1] += "</w>"
        splits.append({'tokens': toks, 'freq': freq})
    merges = []
    for _ in range(max_merges):
        pair_freqs = defaultdict(int)
        for sp in splits:
            toks = sp['tokens']
            for i in range(len(toks) - 1):
                pair = toks[i] + " " + toks[i+1]
                pair_freqs[pair] += sp['freq']
        if not pair_freqs:
            break
        sorted_pairs = sorted(pair_freqs.items(), key=lambda x: x[1], reverse=True)
        best_pair = None
        for pair, freq in sorted_pairs:
            p1, p2 = pair.split(" ")
            merged = p1 + p2
            clean_len = len(merged.replace("</w>", ""))
            if clean_len <= max_token_length:
                best_pair = pair
                break
        if best_pair is None or pair_freqs[best_pair] <= 1:
            break
        merges.append(best_pair)
        p1, p2 = best_pair.split(" ")
        merged = p1 + p2
        for sp in splits:
            toks = sp['tokens']
            new_toks = []
            i = 0
            while i < len(toks):
                if i < len(toks)-1 and toks[i] == p1 and toks[i+1] == p2:
                    new_toks.append(merged)
                    i += 2
                else:
                    new_toks.append(toks[i])
                    i += 1
            sp['tokens'] = new_toks
    tokenizer = SubwordTokenizer(merges)
    for sp in splits:
        for tok in sp['tokens']:
            vocab.add_word(tok)
    for ch in extra_chars:
        vocab.add_word(ch)
    print(f"BPE treinado: {len(merges)} merges, vocabulário final de {vocab.size()} tokens.")
    return tokenizer

# ----------------------------------------------------------------------
# CharCNN e CRF
# ----------------------------------------------------------------------
class CharCNNEmbedding(nn.Module):
    def __init__(self, char_vocab_size, char_embed_dim, num_filters, kernel_sizes, dropout=0.1):
        super().__init__()
        self.char_embed = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(char_embed_dim, num_filters, kernel_size, padding=kernel_size//2)
            for kernel_size in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_dim = num_filters * len(kernel_sizes)

    def forward(self, char_ids):
        batch_size, seq_len, max_char_len = char_ids.size()
        char_emb = self.char_embed(char_ids)
        char_emb = char_emb.view(batch_size * seq_len, max_char_len, -1).transpose(1, 2)
        conv_outs = []
        for conv in self.convs:
            conv_out = conv(char_emb)
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outs.append(pooled)
        concat = torch.cat(conv_outs, dim=1)
        concat = concat.view(batch_size, seq_len, -1)
        return self.dropout(concat)

class CRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags) * 0.01)

    def forward(self, emissions, tags, mask):
        batch_size, seq_len, _ = emissions.size()
        score = torch.zeros(batch_size, device=emissions.device)
        for t in range(seq_len):
            mask_t = mask[:, t].bool()
            score += emissions[torch.arange(batch_size), t, tags[:, t]] * mask_t.float()
            if t > 0:
                prev_tags = tags[:, t-1]
                curr_tags = tags[:, t]
                trans = self.transitions[prev_tags, curr_tags]
                score += trans * mask_t.float()
        log_norm = self._compute_log_norm(emissions, mask)
        n_valid = mask.sum().float() + 1e-9
        return - (score - log_norm).sum() / n_valid

    def _compute_log_norm(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.size()
        alpha = emissions[:, 0, :]
        for t in range(1, seq_len):
            mask_t = mask[:, t].bool()
            alpha_exp = alpha.unsqueeze(2) + self.transitions
            alpha_t = torch.logsumexp(alpha_exp, dim=1) + emissions[:, t, :]
            alpha = torch.where(mask_t.unsqueeze(1), alpha_t, alpha)
        log_norm = torch.logsumexp(alpha, dim=1)
        return log_norm

    def decode(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.size()
        scores = []
        for b in range(batch_size):
            seq_len_b = mask[b].sum().int().item()
            if seq_len_b == 0:
                scores.append([])
                continue
            seq = emissions[b, :seq_len_b, :]
            dp = torch.full((seq_len_b, num_tags), -10000.0, device=emissions.device)
            dp[0] = seq[0]
            backpointers = torch.zeros((seq_len_b, num_tags), dtype=torch.long, device=emissions.device)
            for t in range(1, seq_len_b):
                scores_t = dp[t-1].unsqueeze(1) + self.transitions
                max_scores, best_tags = torch.max(scores_t, dim=0)
                dp[t] = seq[t] + max_scores
                backpointers[t] = best_tags
            best_tags = []
            best_idx = torch.argmax(dp[-1])
            for t in range(seq_len_b-1, -1, -1):
                best_tags.insert(0, best_idx.item())
                if t > 0:
                    best_idx = backpointers[t, best_idx]
            scores.append(best_tags)
        return scores

# ----------------------------------------------------------------------
# Transformer
# ----------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Posição aprendida: adequada ao contexto curto e fixo do NLU."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.max_len = max_len

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequência {seq_len} excede max_len={self.max_len}.")
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return x + self.position_embedding(positions)


class MultiHeadAttention(nn.Module):
    """Self-attention bidirecional com viés de posição relativa."""
    def __init__(self, embed_dim, num_heads, dropout=0.1, max_len=40):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        # Cada cabeça aprende preferências de ordem/distância entre tokens.
        self.relative_bias = nn.Embedding(2 * max_len - 1, num_heads)
        self.max_len = max_len

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.size()
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Relative position: j-i, limitado ao comprimento treinado.
        pos = torch.arange(seq_len, device=query.device)
        rel = pos.unsqueeze(0) - pos.unsqueeze(1)
        rel = rel.clamp(-(self.max_len - 1), self.max_len - 1) + (self.max_len - 1)
        rel_bias = self.relative_bias(rel).permute(2, 0, 1).unsqueeze(0)
        attn_scores = attn_scores + rel_bias

        if mask is not None:
            key_mask = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(key_mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(context)


class GatedFeedForward(nn.Module):
    """GEGLU: permite combinações contextuais mais seletivas que um FFN simples."""
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.value = nn.Linear(embed_dim, ff_dim)
        self.gate = nn.Linear(embed_dim, ff_dim)
        self.out = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.value(x) * F.gelu(self.gate(x))
        return self.out(self.dropout(x))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, max_len=40):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout, max_len=max_len)
        self.feed_forward = GatedFeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, h, mask))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, dropout, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout, max_len=max_len)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids, mask):
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class ContextualAttentionPooling(nn.Module):
    """
    Pooling aprendido para classificação.

    O modelo antigo fazia média simples de todos os tokens. Isso pode diluir
    verbos/estruturas decisivas ("fecha", "abre", "interrompe", "ligada").
    Aqui quatro consultas aprendidas extraem aspectos diferentes do contexto;
    a saída continua com exatamente `embed_dim`, preservando as cabeças NLU.
    """
    def __init__(self, embed_dim, num_queries=4, num_heads=4, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(num_queries * embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, mask):
        bsz, seq_len, dim = x.size()
        q = self.query.unsqueeze(0).expand(bsz, -1, -1)
        Q = self.q_proj(q).view(bsz, self.num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(mask[:, None, None, :] == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        ctx = torch.matmul(weights, V)
        ctx = ctx.transpose(1, 2).contiguous().view(bsz, self.num_queries, dim)
        pooled = self.out_proj(ctx.reshape(bsz, self.num_queries * dim))
        return self.norm(pooled)


# ----------------------------------------------------------------------
# Cabeça especializada de intenção — v12.1
# ----------------------------------------------------------------------
class IntentContextHead(nn.Module):
    """
    Extrai uma representação específica para INTENT a partir da sequência
    contextualizada.

    São usadas quatro consultas aprendidas, cada uma podendo especializar-se
    em um aspecto diferente da construção: ação/verbo, objeto, estado e
    contexto/modificador. Não há regras linguísticas codificadas; as
    relações são aprendidas por backpropagation a partir do rótulo de intenção.

    A saída mantém exatamente `embed_dim`, portanto o contrato das cabeças
    existentes permanece inalterado.
    """
    def __init__(self, embed_dim, num_heads=4, num_queries=4, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.head_dim = embed_dim // num_heads

        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(num_queries * embed_dim, embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, mask):
        bsz, seq_len, dim = x.size()
        q = self.queries.unsqueeze(0).expand(bsz, -1, -1)

        Q = self.q_proj(q).view(bsz, self.num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(mask[:, None, None, :] == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        ctx = torch.matmul(weights, V)
        ctx = ctx.transpose(1, 2).contiguous().view(bsz, self.num_queries, dim)
        ctx = self.out_proj(ctx.reshape(bsz, self.num_queries * dim))

        # Gate residual: combina a representação especializada com um resumo
        # global da frase. Assim a cabeça pode enfatizar relações decisivas
        # sem perder o contexto geral.
        valid = mask.unsqueeze(-1).float()
        base = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        gate = self.gate(ctx)
        return self.norm(ctx * gate + base * (1.0 - gate))


# ----------------------------------------------------------------------
# Modelo NLU com todas as cabeças
# ----------------------------------------------------------------------
class NLUModel(nn.Module):
    def __init__(self, config: NLUConfig, vocab_size: int,
                 num_intents: int, num_bio_tags: int,
                 num_operations: int, num_action_modes: int,
                 num_target_scopes: int, num_value_types: int,
                 char_vocab_size: int = 128):
        super().__init__()
        self.config = config
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size, embed_dim=config.embed_dim,
            num_layers=config.num_layers, num_heads=config.num_heads,
            ff_dim=config.ff_dim, dropout=config.dropout,
            max_len=config.max_seq_len
        )

        self.use_char_cnn = config.use_char_cnn
        if self.use_char_cnn:
            self.char_cnn = CharCNNEmbedding(
                char_vocab_size=char_vocab_size,
                char_embed_dim=config.char_embed_dim,
                num_filters=config.char_num_filters,
                kernel_sizes=config.char_kernel_sizes,
                dropout=config.dropout
            )
            combined_dim = config.embed_dim + self.char_cnn.output_dim
            self.proj = nn.Linear(combined_dim, config.embed_dim)
            self.fusion_norm = nn.LayerNorm(config.embed_dim)
        else:
            self.char_cnn = None
            self.proj = None
            self.fusion_norm = nn.LayerNorm(config.embed_dim)

        # Uma camada contextual extra depois da fusão BPE + caracteres.
        self.fusion_context = TransformerEncoderLayer(
            config.embed_dim, config.num_heads, config.ff_dim,
            config.dropout, max_len=config.max_seq_len
        )
        self.context_pool = ContextualAttentionPooling(
            config.embed_dim, num_queries=4, num_heads=config.num_heads, dropout=config.dropout
        )

        # v12.1: cabeça especializada de intenção.
        # Não altera o contrato do dataset nem as dimensões das demais cabeças.
        # Em vez de depender apenas do pooling global, a intenção recebe
        # consultas aprendidas que procuram relações decisivas entre tokens
        # (ação/verbo, objeto, estado e contexto).
        self.intent_context = IntentContextHead(
            config.embed_dim, num_heads=config.num_heads,
            num_queries=4, dropout=config.dropout
        )
        self.intent_fusion = nn.Sequential(
            nn.LayerNorm(config.embed_dim * 2),
            nn.Linear(config.embed_dim * 2, config.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )

        # Cabeças existentes: mantidas.
        self.intent_head = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim, num_intents)
        )
        self.entity_head = nn.Linear(config.embed_dim, num_bio_tags)
        self.operation_head = nn.Sequential(nn.LayerNorm(config.embed_dim), nn.Linear(config.embed_dim, num_operations))
        self.action_mode_head = nn.Sequential(nn.LayerNorm(config.embed_dim), nn.Linear(config.embed_dim, num_action_modes))
        self.target_scope_head = nn.Sequential(nn.LayerNorm(config.embed_dim), nn.Linear(config.embed_dim, num_target_scopes))
        self.value_type_head = nn.Sequential(nn.LayerNorm(config.embed_dim), nn.Linear(config.embed_dim, num_value_types))
        self.mlm_head = nn.Linear(config.embed_dim, vocab_size)

        self.use_crf = config.use_crf
        if self.use_crf:
            self.crf = CRF(num_bio_tags)

        self.projection_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim, config.proj_dim)
        )

    def forward(self, input_ids, mask, char_ids=None):
        encoder_out = self.encoder(input_ids, mask)
        if self.use_char_cnn and char_ids is not None:
            char_feats = self.char_cnn(char_ids)
            combined = torch.cat([encoder_out, char_feats], dim=-1)
            encoder_out = self.proj(combined)
            encoder_out = self.fusion_norm(encoder_out)
            encoder_out = self.fusion_context(encoder_out, mask)

        # Pooling contextualizado, em vez da média simples.
        pooled = self.context_pool(encoder_out, mask)

        # A cabeça de intenção enxerga a sequência inteira novamente e aprende
        # a dar peso diferente a verbos/estados/objetos conforme a construção.
        intent_context = self.intent_context(encoder_out, mask)
        intent_repr = self.intent_fusion(torch.cat([pooled, intent_context], dim=-1))

        intent_logits = self.intent_head(intent_repr)
        entity_logits = self.entity_head(encoder_out)
        operation_logits = self.operation_head(pooled)
        action_mode_logits = self.action_mode_head(pooled)
        target_scope_logits = self.target_scope_head(pooled)
        value_type_logits = self.value_type_head(pooled)
        mlm_logits = self.mlm_head(encoder_out)
        projected = F.normalize(self.projection_head(pooled), dim=-1)

        return (intent_logits, entity_logits, operation_logits,
                action_mode_logits, target_scope_logits, value_type_logits,
                projected, mlm_logits)

# ----------------------------------------------------------------------
# Dataset (agora com todos os labels)
# ----------------------------------------------------------------------
class NLUDataset(Dataset):
    def __init__(self, data: List[Dict], vocab: Vocab, tokenizer: SubwordTokenizer,
                 config: NLUConfig, char_vocab: Dict, augment: bool = False):
        self.examples = []
        self.char_vocab = char_vocab
        self.augment = augment
        self.mlm_probability = config.mlm_probability
        max_len = config.max_seq_len
        max_char_len = config.max_char_len

        self.synonyms = {
            "luz": ["lâmpada", "iluminação", "claridade"],
            "ligar": ["acender", "ativar", "acionar"],
            "desligar": ["apagar", "desativar", "desconectar"],
            "sala": ["estar", "sala de estar", "living"],
            "quarto": ["dormitório", "quarto de dormir", "suite"],
            "cozinha": ["copa", "kitchen"],
            "garagem": ["abrigo", "estacionamento"],
            "ventilador": ["ventilador de teto", "exaustor"],
            "ar condicionado": ["climatizador", "ar-condicionado"],
            "luminária": ["abajur", "lâmpada de mesa"],
            "tv": ["televisão", "televisor"],
            "volume": ["altura", "nível de som"],
            "brilho": ["intensidade", "luminosidade"],
        }

        for item in data:
            text = item["text"]
            intent_id = item["intent"]
            operation_id = item.get("operation", 0)
            action_mode_id = item.get("action_mode", 0)
            target_scope_id = item.get("target_scope", 0)
            value_type_id = item.get("value_type", 5)  # UNKNOWN
            entities_compact = item["entities"]

            if self.augment:
                augmented = self._generate_augmentations(text, entities_compact)
                examples_to_process = [(text, entities_compact)] + augmented
            else:
                examples_to_process = [(text, entities_compact)]

            for ex_text, ex_entities in examples_to_process:
                self._add_example(
                    ex_text, intent_id, operation_id, action_mode_id,
                    target_scope_id, value_type_id, ex_entities,
                    vocab, tokenizer, config, char_vocab, max_len, max_char_len
                )

    def _generate_augmentations(self, text, entities_compact):
        variations = []
        spans = get_word_spans(text)
        lower_text = text.lower()

        rules = sorted(self.synonyms.items(), key=lambda x: len(x[0]), reverse=True)

        candidates = []
        for source, replacements in rules:
            source_l = source.lower()
            start = 0
            while True:
                pos = lower_text.find(source_l, start)
                if pos < 0:
                    break
                end = pos + len(source)
                left_ok = pos == 0 or not lower_text[pos - 1].isalnum()
                right_ok = end == len(text) or not lower_text[end].isalnum()
                if left_ok and right_ok:
                    for replacement in replacements:
                        candidates.append((pos, end, replacement))
                start = pos + 1

        seen = {text}
        for start, end, replacement in candidates:
            new_text = text[:start] + replacement + text[end:]
            if new_text in seen:
                continue
            seen.add(new_text)

            delta = len(replacement) - (end - start)
            new_entities = []
            valid = True

            for ent_start, ent_end, ent_type in entities_compact:
                if ent_end <= start:
                    ns, ne = ent_start, ent_end
                elif ent_start >= end:
                    ns, ne = ent_start + delta, ent_end + delta
                elif ent_start >= start and ent_end <= end:
                    ns = start
                    ne = start + len(replacement)
                else:
                    valid = False
                    break

                new_entities.append((ns, ne, ent_type))

            if valid:
                variations.append((new_text, new_entities))
            if len(variations) >= 3:
                break

        return variations

    def _add_example(self, text, intent_id, operation_id, action_mode_id,
                     target_scope_id, value_type_id, entities_compact,
                     vocab, tokenizer, config, char_vocab, max_len, max_char_len):
        word_spans = get_word_spans(text)
        all_subtokens = []
        token_to_word_idx = []
        for wi, (start, end, word) in enumerate(word_spans):
            subs = tokenizer.tokenize_word(word)
            for sub in subs:
                all_subtokens.append(sub)
                token_to_word_idx.append(wi)

        token_ids = []
        char_ids_list = []
        for sub in all_subtokens:
            if sub in vocab.WordToIdx:
                token_ids.append(vocab.WordToIdx[sub])
            else:
                token_ids.append(vocab.unk_idx)
            chars = [char_vocab.get(c, char_vocab['<UNK>']) for c in sub]
            if len(chars) > max_char_len:
                chars = chars[:max_char_len]
            else:
                chars = chars + [0] * (max_char_len - len(chars))
            char_ids_list.append(chars)

        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
            token_to_word_idx = token_to_word_idx[:max_len]
            char_ids_list = char_ids_list[:max_len]
        real_len = len(token_ids)

        # MLM com MASCARAMENTO POR PALAVRA INTEIRA.
        # Em vez de esconder sub-tokens aleatórios, escolhemos palavras completas.
        # Isso força o Transformer a usar contexto sintático/semântico mais amplo
        # para reconstruir a palavra ausente.
        masked_positions = []
        masked_labels = []
        if self.augment and self.mlm_probability > 0 and real_len > 2:
            word_to_token_positions = defaultdict(list)
            for pos, wi in enumerate(token_to_word_idx):
                if wi >= 0 and pos < real_len:
                    word_to_token_positions[wi].append(pos)

            candidate_words = [
                wi for wi, positions in word_to_token_positions.items()
                if positions
            ]
            target_tokens = max(1, int(real_len * self.mlm_probability))
            random.shuffle(candidate_words)
            chosen_positions = []
            for wi in candidate_words:
                positions = word_to_token_positions[wi]
                if len(chosen_positions) >= target_tokens:
                    break
                # Não ultrapassa muito o orçamento, mas mantém a palavra inteira.
                if chosen_positions and len(chosen_positions) + len(positions) > max(1, int(real_len * self.mlm_probability * 1.5)):
                    continue
                chosen_positions.extend(positions)

            # Garante pelo menos uma palavra, quando possível.
            if not chosen_positions and candidate_words:
                chosen_positions = word_to_token_positions[candidate_words[0]][:]

            chosen_positions = sorted(set(p for p in chosen_positions if p < real_len))
            for pos in chosen_positions:
                original_token = token_ids[pos]
                masked_labels.append(original_token)
                token_ids[pos] = vocab.WordToIdx.get(vocab.mask_token, vocab.unk_idx)
                masked_positions.append(pos)

            masked_positions = masked_positions[:max_len]
            masked_labels = masked_labels[:max_len]
            masked_positions = masked_positions + [-1] * (max_len - len(masked_positions))
            masked_labels = masked_labels + [-1] * (max_len - len(masked_labels))
        else:
            masked_positions = [-1] * max_len
            masked_labels = [-1] * max_len

        token_ids = token_ids + [vocab.pad_idx] * (max_len - len(token_ids))
        token_to_word_idx = token_to_word_idx + [-1] * (max_len - len(token_to_word_idx))
        char_ids_pad = char_ids_list + [[0]*max_char_len] * (max_len - len(char_ids_list))

        bio_indices = [0] * max_len
        for start, end, type_id in entities_compact:
            b_idx = 1 + (type_id * 2)
            i_idx = 2 + (type_id * 2)

            all_token_indices = []
            for i, (s, e, word) in enumerate(word_spans):
                if max(s, start) < min(e, end):
                    token_indices = [t_idx for t_idx, w_idx in enumerate(token_to_word_idx)
                                     if w_idx == i and t_idx < max_len]
                    all_token_indices.extend(token_indices)

            all_token_indices = sorted(set(all_token_indices))

            if all_token_indices:
                bio_indices[all_token_indices[0]] = b_idx
                for t_idx in all_token_indices[1:]:
                    bio_indices[t_idx] = i_idx

        self.examples.append({
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "char_ids": torch.tensor(char_ids_pad, dtype=torch.long),
            "attention_mask": torch.tensor([1]*real_len + [0]*(max_len-real_len), dtype=torch.long),
            "intent_label": intent_id,
            "operation_label": operation_id,
            "action_mode_label": action_mode_id,
            "target_scope_label": target_scope_id,
            "value_type_label": value_type_id,
            "bio_labels": torch.tensor(bio_indices, dtype=torch.long),
            "masked_positions": torch.tensor(masked_positions, dtype=torch.long),
            "masked_labels": torch.tensor(masked_labels, dtype=torch.long),
            "text": text,
            "token_to_word_idx": token_to_word_idx,
            "word_spans": word_spans
        })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch):
    return {
        "input_ids": torch.stack([ex["input_ids"] for ex in batch]),
        "char_ids": torch.stack([ex["char_ids"] for ex in batch]),
        "attention_mask": torch.stack([ex["attention_mask"] for ex in batch]),
        "intent_labels": torch.tensor([ex["intent_label"] for ex in batch], dtype=torch.long),
        "operation_labels": torch.tensor([ex["operation_label"] for ex in batch], dtype=torch.long),
        "action_mode_labels": torch.tensor([ex["action_mode_label"] for ex in batch], dtype=torch.long),
        "target_scope_labels": torch.tensor([ex["target_scope_label"] for ex in batch], dtype=torch.long),
        "value_type_labels": torch.tensor([ex["value_type_label"] for ex in batch], dtype=torch.long),
        "bio_labels": torch.stack([ex["bio_labels"] for ex in batch]),
        "masked_positions": torch.stack([ex["masked_positions"] for ex in batch]),
        "masked_labels": torch.stack([ex["masked_labels"] for ex in batch]),
        "texts": [ex["text"] for ex in batch],
        "token_to_word_idx": [ex["token_to_word_idx"] for ex in batch],
        "word_spans": [ex["word_spans"] for ex in batch]
    }

# ----------------------------------------------------------------------
# Função de perda contrastiva (igual)
# ----------------------------------------------------------------------
def get_dynamic_temperature(config, epoch: int = 1):
    warm = max(1, config.temperature_decay_epochs)
    progress = min(1.0, max(0.0, (epoch - 1) / warm))
    return config.temperature + (config.temperature_min - config.temperature) * progress

def contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor,
                     temperature: float = 0.07,
                     hard_negative_weight: float = 1.5,
                     hard_negative_power: float = 2.0) -> torch.Tensor:
    if embeddings.size(0) < 2:
        return embeddings.sum() * 0.0

    z = F.normalize(embeddings, dim=-1)
    logits = torch.matmul(z, z.T) / max(float(temperature), 1e-4)
    bsz = z.size(0)
    eye = torch.eye(bsz, dtype=torch.bool, device=z.device)
    positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye
    negative_mask = (~positive_mask) & ~eye
    valid = positive_mask.any(dim=1)
    if not valid.any():
        return z.sum() * 0.0

    sim = torch.matmul(z, z.T).detach()
    hard_score = torch.relu(sim)
    neg_weight = 1.0 + hard_negative_weight * hard_score.pow(hard_negative_power)
    weighted_logits = logits + torch.where(
        negative_mask,
        torch.log(neg_weight.clamp_min(1.0)),
        torch.zeros_like(logits)
    )
    weighted_logits = weighted_logits.masked_fill(eye, -1e9)

    log_prob = weighted_logits - torch.logsumexp(weighted_logits, dim=1, keepdim=True)
    pos_count = positive_mask.sum(dim=1).clamp_min(1)
    mean_pos = (log_prob * positive_mask.float()).sum(dim=1) / pos_count
    return -mean_pos[valid].mean()

def evaluate_contrastive_metrics(model, dataloader, device, temperature=0.07, hard_negative_weight=1.5, hard_negative_power=2.0):
    model.eval()
    all_embeds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Coletando embeddings contrastivos", leave=False):
            input_ids = batch["input_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            intent_labels = batch["intent_labels"].to(device)
            (intent_logits, entity_logits, operation_logits,
             action_mode_logits, target_scope_logits, value_type_logits,
             projected, mlm_logits) = model(input_ids, mask, char_ids)
            all_embeds.append(projected.cpu())
            all_labels.append(intent_labels.cpu())
    all_embeds = torch.cat(all_embeds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if all_embeds.size(0) < 2:
        return 0.0, 0.0, 0.0, 0.0

    sim = torch.matmul(all_embeds, all_embeds.T)
    mask_self = torch.eye(all_embeds.size(0), dtype=torch.bool)
    labels_eq = all_labels.unsqueeze(1) == all_labels.unsqueeze(0)
    pos_mask = labels_eq & ~mask_self
    neg_mask = ~labels_eq & ~mask_self

    intra_sim = sim[pos_mask].mean().item() if pos_mask.any() else 0.0
    inter_sim = sim[neg_mask].mean().item() if neg_mask.any() else 0.0
    separation = intra_sim - inter_sim

    loss_sum = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculando perda contrastiva", leave=False):
            input_ids = batch["input_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            intent_labels = batch["intent_labels"].to(device)
            (intent_logits, entity_logits, operation_logits,
             action_mode_logits, target_scope_logits, value_type_logits,
             projected, mlm_logits) = model(input_ids, mask, char_ids)
            loss = contrastive_loss(projected, intent_labels, temperature, hard_negative_weight, hard_negative_power)
            loss_sum += loss.item()
            n_batches += 1
    avg_loss = loss_sum / n_batches if n_batches > 0 else 0.0

    return intra_sim, inter_sim, separation, avg_loss

# ----------------------------------------------------------------------
# Treinamento e Avaliação (atualizados)
# ----------------------------------------------------------------------
def train_epoch(model, dataloader, optimizer, config, device, epoch=1):
    model.train()
    total_intent_loss = 0
    total_entity_loss = 0
    total_operation_loss = 0
    total_action_mode_loss = 0
    total_target_scope_loss = 0
    total_value_type_loss = 0
    total_contrastive_loss = 0
    total_mlm_loss = 0
    total_loss = 0
    n_batches = 0

    for batch in tqdm(dataloader, desc=f"Treino época {epoch}", leave=False):
        input_ids = batch["input_ids"].to(device)
        char_ids = batch["char_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        intent_labels = batch["intent_labels"].to(device)
        operation_labels = batch["operation_labels"].to(device)
        action_mode_labels = batch["action_mode_labels"].to(device)
        target_scope_labels = batch["target_scope_labels"].to(device)
        value_type_labels = batch["value_type_labels"].to(device)
        bio_labels = batch["bio_labels"].to(device)
        masked_positions = batch["masked_positions"].to(device)
        masked_labels = batch["masked_labels"].to(device)

        optimizer.zero_grad()
        (intent_logits, entity_logits, operation_logits,
         action_mode_logits, target_scope_logits, value_type_logits,
         projected_emb, mlm_logits) = model(input_ids, mask, char_ids)

        # Intent loss com HARD-EXAMPLE MINING adaptativo.
        # Cada intenção continua tendo o mesmo número de exemplos; o peso extra
        # depende somente da dificuldade atual da frase. Assim, frases fáceis
        # não dominam o treino e as construções ambíguas recebem mais gradiente.
        per_example_intent_loss = F.cross_entropy(
            intent_logits, intent_labels,
            label_smoothing=config.label_smoothing,
            reduction='none'
        )
        with torch.no_grad():
            probs = F.softmax(intent_logits, dim=1)
            p_correct = probs.gather(1, intent_labels.unsqueeze(1)).squeeze(1).clamp(0.0, 1.0)
            difficulty = (1.0 - p_correct).pow(config.intent_focal_gamma)
            weights = 1.0 + config.intent_hard_example_weight * difficulty
            weights = weights / weights.mean().clamp_min(1e-6)
        intent_loss = (per_example_intent_loss * weights).mean()

        # Entity loss
        if config.use_crf and hasattr(model, 'crf'):
            entity_loss = model.crf(entity_logits, bio_labels, mask)
        else:
            entity_loss = F.cross_entropy(
                entity_logits.view(-1, entity_logits.size(-1)),
                bio_labels.view(-1),
                ignore_index=-1
            )

        operation_loss = F.cross_entropy(operation_logits, operation_labels)
        action_mode_loss = F.cross_entropy(action_mode_logits, action_mode_labels)
        target_scope_loss = F.cross_entropy(target_scope_logits, target_scope_labels)
        value_type_loss = F.cross_entropy(value_type_logits, value_type_labels)

        # MLM loss
        mlm_loss = 0.0
        mask_pos_flat = (masked_labels.view(-1) != -1)
        if mask_pos_flat.sum() > 0:
            mlm_logits_flat = mlm_logits.view(-1, mlm_logits.size(-1))
            mlm_loss = F.cross_entropy(
                mlm_logits_flat[mask_pos_flat],
                masked_labels.view(-1)[mask_pos_flat],
                ignore_index=-1
            )

        temp = get_dynamic_temperature(config, epoch)
        c_loss = contrastive_loss(projected_emb, intent_labels, temperature=temp,
                                  hard_negative_weight=config.hard_negative_weight,
                                  hard_negative_power=config.hard_negative_power)

        loss = (intent_loss +
                config.entity_loss_weight * entity_loss +
                config.operation_loss_weight * operation_loss +
                config.action_mode_loss_weight * action_mode_loss +
                config.target_scope_loss_weight * target_scope_loss +
                config.value_type_loss_weight * value_type_loss +
                config.contrastive_weight * c_loss +
                config.mlm_weight * mlm_loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_intent_loss += intent_loss.item()
        total_entity_loss += entity_loss.item()
        total_operation_loss += operation_loss.item()
        total_action_mode_loss += action_mode_loss.item()
        total_target_scope_loss += target_scope_loss.item()
        total_value_type_loss += value_type_loss.item()
        total_contrastive_loss += c_loss.item()
        total_mlm_loss += mlm_loss.item() if isinstance(mlm_loss, torch.Tensor) else 0.0
        total_loss += loss.item()
        n_batches += 1

    return (total_loss / n_batches,
            total_intent_loss / n_batches,
            total_entity_loss / n_batches,
            total_operation_loss / n_batches,
            total_action_mode_loss / n_batches,
            total_target_scope_loss / n_batches,
            total_value_type_loss / n_batches,
            total_contrastive_loss / n_batches,
            total_mlm_loss / n_batches)

def bio_ids_to_spans(tag_ids, idx2bio):
    spans = []
    start = None
    cur_type = None
    for i, t in enumerate(tag_ids):
        tag = idx2bio.get(int(t), "O")
        if tag.startswith("B-"):
            if start is not None:
                spans.append((cur_type, start, i - 1))
            start = i
            cur_type = tag[2:]
        elif tag.startswith("I-") and cur_type == tag[2:] and start is not None:
            continue
        else:
            if start is not None:
                spans.append((cur_type, start, i - 1))
            start = None
            cur_type = None
    if start is not None:
        spans.append((cur_type, start, len(tag_ids) - 1))
    return spans

def evaluate(model, dataloader, config, device, idx2bio):
    model.eval()
    total_intent_loss = 0
    total_entity_loss = 0
    total_operation_loss = 0
    total_action_mode_loss = 0
    total_target_scope_loss = 0
    total_value_type_loss = 0
    total_loss = 0
    n_batches = 0

    all_intent_preds = []
    all_intent_labels = []
    all_operation_preds = []
    all_operation_labels = []
    all_action_mode_preds = []
    all_action_mode_labels = []
    all_target_scope_preds = []
    all_target_scope_labels = []
    all_value_type_preds = []
    all_value_type_labels = []
    all_bio_preds = []
    all_bio_labels = []
    all_masks = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Avaliando", leave=False):
            input_ids = batch["input_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            intent_labels = batch["intent_labels"].to(device)
            operation_labels = batch["operation_labels"].to(device)
            action_mode_labels = batch["action_mode_labels"].to(device)
            target_scope_labels = batch["target_scope_labels"].to(device)
            value_type_labels = batch["value_type_labels"].to(device)
            bio_labels = batch["bio_labels"].to(device)

            (intent_logits, entity_logits, operation_logits,
             action_mode_logits, target_scope_logits, value_type_logits,
             _, _) = model(input_ids, mask, char_ids)

            intent_loss = F.cross_entropy(intent_logits, intent_labels)
            if config.use_crf and hasattr(model, 'crf'):
                entity_loss = model.crf(entity_logits, bio_labels, mask)
            else:
                entity_loss = F.cross_entropy(
                    entity_logits.view(-1, entity_logits.size(-1)),
                    bio_labels.view(-1),
                    ignore_index=-1
                )
            operation_loss = F.cross_entropy(operation_logits, operation_labels)
            action_mode_loss = F.cross_entropy(action_mode_logits, action_mode_labels)
            target_scope_loss = F.cross_entropy(target_scope_logits, target_scope_labels)
            value_type_loss = F.cross_entropy(value_type_logits, value_type_labels)

            loss = (intent_loss +
                    config.entity_loss_weight * entity_loss +
                    config.operation_loss_weight * operation_loss +
                    config.action_mode_loss_weight * action_mode_loss +
                    config.target_scope_loss_weight * target_scope_loss +
                    config.value_type_loss_weight * value_type_loss)

            total_intent_loss += intent_loss.item()
            total_entity_loss += entity_loss.item()
            total_operation_loss += operation_loss.item()
            total_action_mode_loss += action_mode_loss.item()
            total_target_scope_loss += target_scope_loss.item()
            total_value_type_loss += value_type_loss.item()
            total_loss += loss.item()
            n_batches += 1

            intent_preds = intent_logits.argmax(dim=1)
            all_intent_preds.append(intent_preds.cpu())
            all_intent_labels.append(intent_labels.cpu())

            operation_preds = operation_logits.argmax(dim=1)
            all_operation_preds.append(operation_preds.cpu())
            all_operation_labels.append(operation_labels.cpu())

            action_mode_preds = action_mode_logits.argmax(dim=1)
            all_action_mode_preds.append(action_mode_preds.cpu())
            all_action_mode_labels.append(action_mode_labels.cpu())

            target_scope_preds = target_scope_logits.argmax(dim=1)
            all_target_scope_preds.append(target_scope_preds.cpu())
            all_target_scope_labels.append(target_scope_labels.cpu())

            value_type_preds = value_type_logits.argmax(dim=1)
            all_value_type_preds.append(value_type_preds.cpu())
            all_value_type_labels.append(value_type_labels.cpu())

            if config.use_crf and hasattr(model, 'crf'):
                for b in range(entity_logits.size(0)):
                    valid_len = mask[b].sum().int().item()
                    if valid_len == 0:
                        continue
                    emissions = entity_logits[b:b+1, :valid_len, :]
                    mask_sub = mask[b:b+1, :valid_len]
                    tags_seq = model.crf.decode(emissions, mask_sub)[0]
                    all_bio_preds.append(torch.tensor(tags_seq, dtype=torch.long))
                    all_bio_labels.append(bio_labels[b, :valid_len].cpu())
                    all_masks.append(torch.ones(valid_len, dtype=torch.bool))
            else:
                entity_preds = entity_logits.argmax(dim=-1)
                for i in range(entity_preds.size(0)):
                    valid_len = mask[i].sum().item()
                    all_bio_preds.append(entity_preds[i, :valid_len].cpu())
                    all_bio_labels.append(bio_labels[i, :valid_len].cpu())
                    all_masks.append(torch.ones(valid_len, dtype=torch.bool))

    intent_preds = torch.cat(all_intent_preds)
    intent_labels = torch.cat(all_intent_labels)
    intent_acc = (intent_preds == intent_labels).float().mean().item()

    operation_preds = torch.cat(all_operation_preds)
    operation_labels = torch.cat(all_operation_labels)
    operation_acc = (operation_preds == operation_labels).float().mean().item()

    action_mode_preds = torch.cat(all_action_mode_preds)
    action_mode_labels = torch.cat(all_action_mode_labels)
    action_mode_acc = (action_mode_preds == action_mode_labels).float().mean().item()

    target_scope_preds = torch.cat(all_target_scope_preds)
    target_scope_labels = torch.cat(all_target_scope_labels)
    target_scope_acc = (target_scope_preds == target_scope_labels).float().mean().item()

    value_type_preds = torch.cat(all_value_type_preds)
    value_type_labels = torch.cat(all_value_type_labels)
    value_type_acc = (value_type_preds == value_type_labels).float().mean().item()

    if all_bio_preds:
        bio_preds = torch.cat(all_bio_preds)
        bio_labels = torch.cat(all_bio_labels)
        entity_class_ids = [i for tag, i in bio2idx.items() if tag != "O"]
        tp = fp = fn = torch.tensor(0, dtype=torch.long)
        for c in entity_class_ids:
            pred_c = (bio_preds == c)
            true_c = (bio_labels == c)
            tp = tp + (pred_c & true_c).sum()
            fp = fp + (pred_c & ~true_c).sum()
            fn = fn + (~pred_c & true_c).sum()
        total_tp = tp.float()
        total_fp = fp.float()
        total_fn = fn.float()
        precision = total_tp / (total_tp + total_fp + 1e-9)
        recall = total_tp / (total_tp + total_fn + 1e-9)
        bio_f1_micro = 2 * precision * recall / (precision + recall + 1e-9)
        bio_f1 = bio_f1_micro.item()
    else:
        bio_f1 = 0.0

    span_tp = span_fp = span_fn = 0
    for pred_seq, true_seq in zip(all_bio_preds, all_bio_labels):
        pred_spans = set(bio_ids_to_spans(pred_seq.tolist(), idx2bio))
        true_spans = set(bio_ids_to_spans(true_seq.tolist(), idx2bio))
        span_tp += len(pred_spans & true_spans)
        span_fp += len(pred_spans - true_spans)
        span_fn += len(true_spans - pred_spans)
    span_precision = span_tp / (span_tp + span_fp + 1e-9)
    span_recall = span_tp / (span_tp + span_fn + 1e-9)
    entity_span_f1 = 2 * span_precision * span_recall / (span_precision + span_recall + 1e-9)

    return (total_loss / n_batches,
            total_intent_loss / n_batches,
            total_entity_loss / n_batches,
            total_operation_loss / n_batches,
            total_action_mode_loss / n_batches,
            total_target_scope_loss / n_batches,
            total_value_type_loss / n_batches,
            intent_acc,
            operation_acc,
            action_mode_acc,
            target_scope_acc,
            value_type_acc,
            bio_f1,
            entity_span_f1)

def entity_type_metrics(all_bio_preds, all_bio_labels, idx2bio):
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for pred_seq, true_seq in zip(all_bio_preds, all_bio_labels):
        pred = pred_seq.tolist()
        true = true_seq.tolist()
        for c in range(len(idx2bio)):
            tag = idx2bio.get(c, "O")
            if tag == "O":
                continue
            ent_type = tag[2:] if tag.startswith(("B-", "I-")) else tag
            stats[ent_type]["tp"] += sum(1 for p, t in zip(pred, true) if p == c and t == c)
            stats[ent_type]["fp"] += sum(1 for p, t in zip(pred, true) if p == c and t != c)
            stats[ent_type]["fn"] += sum(1 for p, t in zip(pred, true) if p != c and t == c)

    result = {}
    for ent_type, s in sorted(stats.items()):
        precision = s["tp"] / (s["tp"] + s["fp"] + 1e-9)
        recall = s["tp"] / (s["tp"] + s["fn"] + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        result[ent_type] = {"precision": precision, "recall": recall, "f1": f1}
    return result

# ----------------------------------------------------------------------
# Predição (atualizada)
# ----------------------------------------------------------------------
def predict(model, tokenizer, vocab, text, config, device, char_vocab,
            idx2intent, idx2bio, idx2operation, idx2action_mode,
            idx2target_scope, idx2value_type, intent_prototypes=None):
    model.eval()
    word_spans = get_word_spans(text)
    words = [w for (_, _, w) in word_spans]
    all_subtokens = []
    token_to_word_idx = []
    for wi, w in enumerate(words):
        subs = tokenizer.tokenize_word(w)
        for sub in subs:
            all_subtokens.append(sub)
            token_to_word_idx.append(wi)

    ids = []
    char_ids_list = []
    for sub in all_subtokens:
        if sub in vocab.WordToIdx:
            ids.append(vocab.WordToIdx[sub])
        else:
            ids.append(vocab.unk_idx)
        chars = [char_vocab.get(c, char_vocab['<UNK>']) for c in sub]
        if len(chars) > config.max_char_len:
            chars = chars[:config.max_char_len]
        else:
            chars = chars + [0] * (config.max_char_len - len(chars))
        char_ids_list.append(chars)

    if len(ids) > config.max_seq_len:
        ids = ids[:config.max_seq_len]
        token_to_word_idx = token_to_word_idx[:config.max_seq_len]
        char_ids_list = char_ids_list[:config.max_seq_len]

    real_len = len(ids)
    ids = ids + [vocab.pad_idx] * (config.max_seq_len - len(ids))
    char_ids_pad = char_ids_list + [[0]*config.max_char_len] * (config.max_seq_len - len(char_ids_list))
    token_to_word_idx = token_to_word_idx + [-1] * (config.max_seq_len - len(token_to_word_idx))

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    char_ids_t = torch.tensor([char_ids_pad], dtype=torch.long, device=device)
    mask = torch.tensor([[1]*real_len + [0]*(config.max_seq_len-real_len)], dtype=torch.long, device=device)

    with torch.no_grad():
        (intent_logits, entity_logits, operation_logits,
         action_mode_logits, target_scope_logits, value_type_logits,
         projected, _) = model(input_ids, mask, char_ids_t)

        intent_probs = F.softmax(intent_logits, dim=-1)[0]
        top2 = torch.topk(intent_probs, k=min(2, intent_probs.numel()))
        intent_idx = top2.indices[0].item()
        intent_confidence = top2.values[0].item()
        intent_margin = (
            (top2.values[0] - top2.values[1]).item()
            if top2.values.numel() > 1 else intent_confidence
        )

        operation_idx = operation_logits[0].argmax().item()
        operation_confidence = F.softmax(operation_logits[0], dim=0).max().item()

        action_mode_idx = action_mode_logits[0].argmax().item()
        action_mode_confidence = F.softmax(action_mode_logits[0], dim=0).max().item()

        target_scope_idx = target_scope_logits[0].argmax().item()
        target_scope_confidence = F.softmax(target_scope_logits[0], dim=0).max().item()

        value_type_idx = value_type_logits[0].argmax().item()
        value_type_confidence = F.softmax(value_type_logits[0], dim=0).max().item()

        prototype_distance = 0.0
        if intent_prototypes is not None and intent_prototypes.numel() > 0:
            proto = F.normalize(intent_prototypes.to(device), dim=-1)
            sample = F.normalize(projected[0], dim=-1)
            prototype_distance = float(1.0 - torch.matmul(proto, sample).max().item())

        if config.use_crf and hasattr(model, 'crf'):
            emissions = entity_logits[:, :real_len, :]
            mask_sub = mask[:, :real_len]
            tags_seq = model.crf.decode(emissions, mask_sub)[0]
        else:
            tags_seq = entity_logits.argmax(dim=-1)[0, :real_len].tolist()

        bio_tags = [idx2bio.get(t, "O") for t in tags_seq]

    entities = []
    current_entity = None
    for idx, tag in enumerate(bio_tags):
        if tag == "O":
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None
            continue

        if tag.startswith("B-"):
            if current_entity is not None:
                entities.append(current_entity)
            entity_type = tag[2:]
            wi = token_to_word_idx[idx]
            current_entity = {"type": entity_type, "tokens": [idx], "word_idx": wi}

        elif tag.startswith("I-") and current_entity is not None and tag[2:] == current_entity["type"]:
            current_entity["tokens"].append(idx)
        else:
            if current_entity is not None:
                entities.append(current_entity)
            current_entity = None

    if current_entity is not None:
        entities.append(current_entity)

    entities = merge_adjacent_entities(entities, token_to_word_idx)

    final_entities = []
    for ent in entities:
        word_indices = ent["word_indices"]
        entity_words = [words[w] for w in word_indices if w < len(words)]
        if not entity_words:
            continue
        if not any(c.isalnum() for w in entity_words for c in w):
            continue
        entity_text = " ".join(entity_words)
        final_entities.append({"type": ent["type"], "value": entity_text})

    if config.unknown_detection_enabled:
        low_conf = intent_confidence < config.unknown_confidence_threshold
        low_margin = intent_margin < config.unknown_margin_threshold
        far_embedding = (intent_prototypes is not None and prototype_distance > config.unknown_distance_threshold)
        if config.unknown_require_two_signals:
            below_threshold = (low_conf and low_margin) or (low_conf and far_embedding) or (low_margin and far_embedding)
        else:
            below_threshold = low_conf or low_margin or far_embedding
    else:
        below_threshold = False

    return {
        "intent": "UNKNOWN" if below_threshold else idx2intent.get(intent_idx, "UNKNOWN"),
        "raw_intent": idx2intent.get(intent_idx, "UNKNOWN"),
        "operation": idx2operation.get(operation_idx, "NONE"),
        "action_mode": idx2action_mode.get(action_mode_idx, "UNKNOWN"),
        "target_scope": idx2target_scope.get(target_scope_idx, "UNKNOWN"),
        "value_type": idx2value_type.get(value_type_idx, "UNKNOWN"),
        "operation_confidence": operation_confidence,
        "action_mode_confidence": action_mode_confidence,
        "target_scope_confidence": target_scope_confidence,
        "value_type_confidence": value_type_confidence,
        "confidence": intent_confidence,
        "margin": intent_margin,
        "prototype_distance": prototype_distance,
        "below_threshold": below_threshold,
        "entities": final_entities
    }

# ----------------------------------------------------------------------
# Contexto (mantido)
# ----------------------------------------------------------------------
class ContextManager:
    def __init__(self, max_turns=5):
        self.max_turns = max_turns
        self.turns = []

    def clear(self):
        self.turns.clear()

    def update(self, result):
        entities = list(result.get("entities", []))
        if entities or result.get("raw_intent") not in (None, "UNKNOWN"):
            self.turns.append({
                "intent": result.get("raw_intent", result.get("intent")),
                "entities": entities
            })
            self.turns = self.turns[-self.max_turns:]

    def _last_entity(self, entity_type):
        for turn in reversed(self.turns):
            for ent in reversed(turn["entities"]):
                if ent.get("type") == entity_type:
                    return ent
        return None

    def resolve(self, text, result):
        lower = text.lower()
        has_reference = any(re.search(r"\b" + re.escape(p) + r"\b", lower) for p in NLU_PRONOUNS)
        if not has_reference:
            return result

        existing = {(e.get("type"), e.get("value", "").lower()) for e in result.get("entities", [])}
        resolved = list(result.get("entities", []))
        for etype in ("DEVICE", "LOCATION"):
            ent = self._last_entity(etype)
            if ent and (etype, ent.get("value", "").lower()) not in existing:
                resolved.append({"type": etype, "value": ent["value"], "resolved_from_context": True})

        result = dict(result)
        result["entities"] = resolved
        result["context_resolved"] = True
        return result

NLU_PRONOUNS = ("ela", "ele", "isso", "isto", "essa", "esse", "aquela", "aquele", "aquilo", "lá", "ali")

def predict_contextual(model, tokenizer, vocab, text, config, device, char_vocab,
                       idx2intent, idx2bio, idx2operation, idx2action_mode,
                       idx2target_scope, idx2value_type, context, intent_prototypes=None):
    result = predict(model, tokenizer, vocab, text, config, device, char_vocab,
                     idx2intent, idx2bio, idx2operation, idx2action_mode,
                     idx2target_scope, idx2value_type, intent_prototypes)
    result = context.resolve(text, result)
    context.update(result)
    return result

# ----------------------------------------------------------------------
# Salvamento e checkpoint (atualizados)
# ----------------------------------------------------------------------
def _state_checksum(model_state):
    buffer = io.BytesIO()
    torch.save(model_state, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()

def save_package(path, model, vocab, tokenizer, config,
                 intent_labels, bio_labels, operation_labels,
                 action_mode_labels, target_scope_labels, value_type_labels,
                 char_vocab, intent_prototypes=None, scheduler_state=None):
    package = {
        "version": "7.0",  # pipeline compatível com dataset 63.x; encoder contextual v12
        "config": config,
        "architecture_version": getattr(config, "architecture_version", "v12.1-context-specialized"),
        "vocab": vocab,
        "tokenizer_merges": tokenizer.Merges,
        "intent_labels": intent_labels,
        "bio_labels": bio_labels,
        "operation_labels": operation_labels,
        "action_mode_labels": action_mode_labels,
        "target_scope_labels": target_scope_labels,
        "value_type_labels": value_type_labels,
        "char_vocab": char_vocab,
        "model_state": model.state_dict(),
        "intent_prototypes": intent_prototypes.cpu() if isinstance(intent_prototypes, torch.Tensor) else None,
        "scheduler_state": scheduler_state,
    }
    package["checksum"] = _state_checksum(package["model_state"])
    tmp_path = path + ".tmp"
    torch.save(package, tmp_path)
    os.replace(tmp_path, path)
    print(f"✅ Pacote salvo em {path} (SHA256 do model_state: {package['checksum'][:16]}...)")

def save_checkpoint(path, epoch, model, optimizer, scheduler, best_val_loss,
                    patience_counter, vocab, tokenizer, config,
                    intent_labels, bio_labels, operation_labels,
                    action_mode_labels, target_scope_labels, value_type_labels,
                    char_vocab, train_indices=None, val_indices=None):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
        "vocab_word_to_idx": vocab.WordToIdx,
        "tokenizer_merges": tokenizer.Merges,
        "config": config,
        "architecture_version": getattr(config, "architecture_version", "v12.1-context-specialized"),
        "intent_labels": list(intent_labels),
        "bio_labels": list(bio_labels),
        "operation_labels": list(operation_labels),
        "action_mode_labels": list(action_mode_labels),
        "target_scope_labels": list(target_scope_labels),
        "value_type_labels": list(value_type_labels),
        "char_vocab": char_vocab,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "pipeline_version": "7.0-dataset63-context-specialized",
    }
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)

def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=False)

# ----------------------------------------------------------------------
# Fine-tuning com expansão (mantido)
# ----------------------------------------------------------------------
def expand_vocabulary_and_embeddings(model, vocab, new_vocab_size, embed_dim):
    old_embedding = model.encoder.embedding
    old_vocab_size = old_embedding.num_embeddings
    if new_vocab_size <= old_vocab_size:
        return model

    print(f"🔧 Expandindo embedding de {old_vocab_size} para {new_vocab_size} tokens.")
    new_embedding = nn.Embedding(new_vocab_size, embed_dim, padding_idx=0)
    with torch.no_grad():
        new_embedding.weight[:old_vocab_size] = old_embedding.weight
        nn.init.normal_(new_embedding.weight[old_vocab_size:], mean=0.0, std=0.02)
    model.encoder.embedding = new_embedding
    return model

# ----------------------------------------------------------------------
# Função run_training (atualizada)
# ----------------------------------------------------------------------
def run_training(config, data, intent_map, entity_type_map, operation_map,
                 action_mode_map, target_scope_map, value_type_map,
                 char_vocab, device, resume=False, fresh=False, fine_tune=False,
                 checkpoint_path="checkpoint_nlu_v11.pt", best_model_path="house_nlu_v11.bin"):

    global INTENTS, intent2idx, idx2intent, ENTITY_TYPES, entity_type2idx, idx2entity_type
    global BIO_TAGS, bio2idx, idx2bio, NUM_BIO_TAGS
    global OPERATIONS, operation2idx, idx2operation
    global ACTION_MODES, action_mode2idx, idx2action_mode
    global TARGET_SCOPES, target_scope2idx, idx2target_scope
    global VALUE_TYPES, value_type2idx, idx2value_type

    # Inicializa mapeamentos
    INTENTS = sorted(intent_map.keys(), key=lambda k: intent_map[k])
    intent2idx = intent_map
    idx2intent = {v: k for k, v in intent_map.items()}

    ENTITY_TYPES = sorted(entity_type_map.keys(), key=lambda k: entity_type_map[k])
    entity_type2idx = entity_type_map
    idx2entity_type = {v: k for k, v in entity_type_map.items()}

    OPERATIONS = sorted(operation_map.keys(), key=lambda k: operation_map[k])
    operation2idx = operation_map
    idx2operation = {v: k for k, v in operation_map.items()}

    ACTION_MODES = sorted(action_mode_map.keys(), key=lambda k: action_mode_map[k])
    action_mode2idx = action_mode_map
    idx2action_mode = {v: k for k, v in action_mode_map.items()}

    TARGET_SCOPES = sorted(target_scope_map.keys(), key=lambda k: target_scope_map[k])
    target_scope2idx = target_scope_map
    idx2target_scope = {v: k for k, v in target_scope_map.items()}

    VALUE_TYPES = sorted(value_type_map.keys(), key=lambda k: value_type_map[k])
    value_type2idx = value_type_map
    idx2value_type = {v: k for k, v in value_type_map.items()}

    BIO_TAGS = ["O"]
    for ent_type in ENTITY_TYPES:
        BIO_TAGS.append(f"B-{ent_type}")
        BIO_TAGS.append(f"I-{ent_type}")
    bio2idx = {tag: i for i, tag in enumerate(BIO_TAGS)}
    idx2bio = {i: tag for i, tag in enumerate(BIO_TAGS)}
    NUM_BIO_TAGS = len(BIO_TAGS)

    vocab = None
    tokenizer = None
    model = None
    optimizer = None
    scheduler = None
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    train_indices = None
    val_indices = None

    if resume or fine_tune:
        if not os.path.isfile(checkpoint_path):
            print(f"❌ Checkpoint '{checkpoint_path}' não encontrado para resume/fine-tune.")
            sys.exit(1)
        print(f"📂 Carregando checkpoint '{checkpoint_path}'...")
        checkpoint = load_checkpoint(checkpoint_path, device)
        saved_arch = checkpoint.get("architecture_version")
        saved_config = checkpoint.get("config", config)
        if saved_arch not in (None, "v12.1-context-specialized"):
            print(f"❌ Checkpoint incompatível: arquitetura '{saved_arch}' não é v12.1-context-specialized.")
            print("   Para a nova arquitetura contextual, inicie com a opção 1 (fresh).")
            sys.exit(1)
        if not fine_tune:
            config = saved_config

        intent_labels = checkpoint["intent_labels"]
        bio_labels = checkpoint["bio_labels"]
        operation_labels = checkpoint.get("operation_labels", ["NONE"])
        # Carrega novos labels, com fallback para valores padrão
        action_mode_labels = checkpoint.get("action_mode_labels", ["IMMEDIATE"])
        target_scope_labels = checkpoint.get("target_scope_labels", ["SINGLE"])
        value_type_labels = checkpoint.get("value_type_labels", ["UNKNOWN"])

        # Alinha os IDs do dataset à ordem semântica usada pelo checkpoint.
        data = remap_data_to_label_maps(
            data,
            (intent_map, entity_type_map, operation_map, action_mode_map, target_scope_map, value_type_map),
            (
                {name: i for i, name in enumerate(intent_labels)},
                # IMPORTANTE: o índice do tipo de entidade é CONTÍGUO (0..N-1).
                # enumerate(bio_labels) aqui seria errado, pois incluiria os I-*
                # e o O na numeração. Ex.: B-DEVICE teria ID 1, B-LOCATION 3...
                # e _add_example() transforma esse ID em 1 + type_id*2,
                # produzindo tags inválidas (ex.: 31) para um CRF de 17 tags.
                {tag[2:]: entity_idx for entity_idx, tag in enumerate(
                    [t for t in bio_labels if t.startswith("B-")]
                )},
                {name: i for i, name in enumerate(operation_labels)},
                {name: i for i, name in enumerate(action_mode_labels)},
                {name: i for i, name in enumerate(target_scope_labels)},
                {name: i for i, name in enumerate(value_type_labels)},
            )
        )

        INTENTS = list(intent_labels)
        intent2idx = {name: i for i, name in enumerate(INTENTS)}
        idx2intent = {i: name for i, name in enumerate(INTENTS)}

        ENTITY_TYPES = []
        for tag in bio_labels:
            if tag.startswith("B-"):
                ent = tag[2:]
                if ent not in ENTITY_TYPES:
                    ENTITY_TYPES.append(ent)
        ENTITY_TYPES = sorted(ENTITY_TYPES)
        entity_type2idx = {name: i for i, name in enumerate(ENTITY_TYPES)}
        idx2entity_type = {i: name for i, name in enumerate(ENTITY_TYPES)}
        BIO_TAGS = bio_labels
        bio2idx = {tag: i for i, tag in enumerate(BIO_TAGS)}
        idx2bio = {i: tag for i, tag in enumerate(BIO_TAGS)}
        NUM_BIO_TAGS = len(BIO_TAGS)

        OPERATIONS = list(operation_labels)
        operation2idx = {name: i for i, name in enumerate(OPERATIONS)}
        idx2operation = {i: name for i, name in enumerate(OPERATIONS)}

        ACTION_MODES = list(action_mode_labels)
        action_mode2idx = {name: i for i, name in enumerate(ACTION_MODES)}
        idx2action_mode = {i: name for i, name in enumerate(ACTION_MODES)}

        TARGET_SCOPES = list(target_scope_labels)
        target_scope2idx = {name: i for i, name in enumerate(TARGET_SCOPES)}
        idx2target_scope = {i: name for i, name in enumerate(TARGET_SCOPES)}

        VALUE_TYPES = list(value_type_labels)
        value_type2idx = {name: i for i, name in enumerate(VALUE_TYPES)}
        idx2value_type = {i: name for i, name in enumerate(VALUE_TYPES)}

        vocab = Vocab()
        vocab.WordToIdx = checkpoint["vocab_word_to_idx"]
        vocab.IdxToWord = {v: k for k, v in vocab.WordToIdx.items()}
        tokenizer = SubwordTokenizer(checkpoint["tokenizer_merges"])
        char_vocab = checkpoint.get("char_vocab", char_vocab)

        model = NLUModel(
            config=config,
            vocab_size=vocab.size(),
            num_intents=len(intent2idx),
            num_bio_tags=NUM_BIO_TAGS,
            num_operations=len(operation2idx),
            num_action_modes=len(action_mode2idx),
            num_target_scopes=len(target_scope2idx),
            num_value_types=len(value_type2idx),
            char_vocab_size=len(char_vocab)
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])

        if fine_tune:
            print("🔄 Modo FINE-TUNE: o vocabulário será expandido se houver novos tokens.")
            all_texts = [item["text"] for item in data]
            new_vocab = Vocab()
            new_tokenizer = train_bpe_from_texts(
                all_texts,
                max_merges=config.max_bpe_merges,
                max_token_length=config.max_token_length,
                vocab=new_vocab,
                extra_chars=config.extra_chars
            )
            # IMPORTANTE: não substituímos o vocabulário antigo.
            # Os IDs existentes precisam permanecer idênticos ao checkpoint.
            old_vocab_size = vocab.size()
            for token, old_id in list(vocab.WordToIdx.items()):
                if token not in vocab.IdxToWord.values():
                    vocab.IdxToWord[old_id] = token
            added_tokens = []
            for token in new_vocab.WordToIdx:
                if token not in vocab.WordToIdx:
                    vocab.add_word(token)
                    added_tokens.append(token)
            if added_tokens:
                print(f"🔧 Fine-tuning: adicionando {len(added_tokens)} tokens novos ao vocabulário.")
                model = expand_vocabulary_and_embeddings(model, vocab, vocab.size(), config.embed_dim)
                # O MLM head também depende do tamanho do vocabulário.
                old_mlm = model.mlm_head
                new_mlm = nn.Linear(config.embed_dim, vocab.size()).to(device)
                with torch.no_grad():
                    keep = min(old_mlm.out_features, new_mlm.out_features)
                    new_mlm.weight[:keep] = old_mlm.weight[:keep]
                    new_mlm.bias[:keep] = old_mlm.bias[:keep]
                    if new_mlm.out_features > keep:
                        nn.init.normal_(new_mlm.weight[keep:], mean=0.0, std=0.02)
                        nn.init.zeros_(new_mlm.bias[keep:])
                model.mlm_head = new_mlm
                # Mantém merges antigos e adiciona apenas merges realmente novos.
                tokenizer = SubwordTokenizer(tokenizer.Merges + [m for m in new_tokenizer.Merges if m not in tokenizer.Merges])
            else:
                print("Nenhum novo token detectado. Mantendo vocabulário original.")
        else:
            pass

    # Se não resume nem fine-tune, ou se fine-tune, precisamos criar dataset e split
    if not resume or fine_tune:
        temp_dataset = NLUDataset(data, vocab if vocab else Vocab(), tokenizer if tokenizer else SubwordTokenizer([]),
                                  config, char_vocab, augment=False)
        _, _, raw_train_indices, raw_val_indices = stratified_split(temp_dataset, val_ratio=0.25, seed=SEED)
        train_data = [data[i] for i in raw_train_indices]
        val_data = [data[i] for i in raw_val_indices]
        train_indices = raw_train_indices
        val_indices = raw_val_indices

        if model is None:
            print("🆕 Criando modelo do zero.")
            all_texts = [item["text"] for item in train_data]
            vocab = Vocab()
            tokenizer = train_bpe_from_texts(
                all_texts,
                max_merges=config.max_bpe_merges,
                max_token_length=config.max_token_length,
                vocab=vocab,
                extra_chars=config.extra_chars
            )
            model = NLUModel(
                config=config,
                vocab_size=vocab.size(),
                num_intents=len(intent2idx),
                num_bio_tags=NUM_BIO_TAGS,
                num_operations=len(operation2idx),
                num_action_modes=len(action_mode2idx),
                num_target_scopes=len(target_scope2idx),
                num_value_types=len(value_type2idx),
                char_vocab_size=len(char_vocab)
            ).to(device)

        train_dataset = NLUDataset(train_data, vocab, tokenizer, config, char_vocab, augment=True)
        val_dataset = NLUDataset(val_data, vocab, tokenizer, config, char_vocab, augment=False)
        print(f"📐 Split estratificado: {len(train_dataset)} treino (com aumento) / {len(val_dataset)} validação")

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.min_lr)

        if fine_tune:
            print("🔧 Fine-tuning: resetando LR para 1e-5.")
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5
            scheduler.base_lrs = [1e-5]

    else:
        raw_train_indices = checkpoint.get("train_indices")
        raw_val_indices = checkpoint.get("val_indices")
        if raw_train_indices is None or raw_val_indices is None:
            temp_dataset = NLUDataset(data, vocab, tokenizer, config, char_vocab, augment=False)
            _, _, raw_train_indices, raw_val_indices = stratified_split(temp_dataset, val_ratio=0.25, seed=SEED)
        train_data = [data[i] for i in raw_train_indices]
        val_data = [data[i] for i in raw_val_indices]
        train_dataset = NLUDataset(train_data, vocab, tokenizer, config, char_vocab, augment=True)
        val_dataset = NLUDataset(val_data, vocab, tokenizer, config, char_vocab, augment=False)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
        train_indices = raw_train_indices
        val_indices = raw_val_indices

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.min_lr)
        if "scheduler_state" in checkpoint and checkpoint["scheduler_state"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_val_loss = checkpoint["best_val_loss"]
        patience_counter = checkpoint["patience_counter"]
        start_epoch = checkpoint["epoch"] + 1

    # Loop de treinamento
    prev_train_loss = None
    overfit_counter = 0

    print("🚀 Iniciando treinamento..." if not resume and not fine_tune else "🚀 Retomando treinamento..." if resume else "🚀 Fine-tuning...")
    for epoch in range(start_epoch, config.epochs):
        (train_loss, train_intent_loss, train_entity_loss, train_operation_loss,
         train_action_mode_loss, train_target_scope_loss, train_value_type_loss,
         train_contrastive_loss, train_mlm_loss) = train_epoch(
            model, train_loader, optimizer, config, device, epoch + 1)

        (val_loss, val_intent_loss, val_entity_loss, val_operation_loss,
         val_action_mode_loss, val_target_scope_loss, val_value_type_loss,
         val_intent_acc, val_operation_acc, val_action_mode_acc,
         val_target_scope_acc, val_value_type_acc, val_bio_f1, val_span_f1) = evaluate(
            model, val_loader, config, device, idx2bio)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        intra_sim, inter_sim, separation, avg_contrastive_loss = evaluate_contrastive_metrics(
            model, val_loader, device, temperature=get_dynamic_temperature(config, epoch + 1),
            hard_negative_weight=config.hard_negative_weight, hard_negative_power=config.hard_negative_power
        )

        print(f"\nÉpoca {epoch+1}/{config.epochs}")
        print(f"  Train Loss: {train_loss:.4f} (Intent: {train_intent_loss:.4f}, Entity: {train_entity_loss:.4f}, Op: {train_operation_loss:.4f}, "
              f"AM: {train_action_mode_loss:.4f}, TS: {train_target_scope_loss:.4f}, VT: {train_value_type_loss:.4f}, "
              f"Contr: {train_contrastive_loss:.4f}, MLM: {train_mlm_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f} (Intent: {val_intent_loss:.4f}, Entity: {val_entity_loss:.4f}, Op: {val_operation_loss:.4f}, "
              f"AM: {val_action_mode_loss:.4f}, TS: {val_target_scope_loss:.4f}, VT: {val_value_type_loss:.4f})")
        print(f"  Val Acc: Intent {val_intent_acc:.4f}, Op {val_operation_acc:.4f}, AM {val_action_mode_acc:.4f}, TS {val_target_scope_acc:.4f}, VT {val_value_type_acc:.4f}")
        print(f"  Val Entity F1 (span): {val_span_f1:.4f}")
        print(f"  🔍 Contraste: Intra={intra_sim:.4f}, Inter={inter_sim:.4f}, Sep={separation:.4f}, Perda={avg_contrastive_loss:.4f}")
        print(f"  📊 LR: {current_lr:.6f} | Temp: {get_dynamic_temperature(config, epoch+1):.4f} | Melhor val_loss: {best_val_loss:.4f}")

        if prev_train_loss is not None:
            train_decreased = train_loss < prev_train_loss
            val_increased = val_loss > best_val_loss if best_val_loss != float('inf') else False
            if train_decreased and val_increased:
                overfit_counter += 1
                print(f"  ⚠️  Indício de overfitting ({overfit_counter}/{config.patience})")
            else:
                overfit_counter = 0
        prev_train_loss = train_loss

        if overfit_counter >= config.patience:
            print(f"🛑 Overfitting detectado por {config.patience} épocas consecutivas. Encerrando.")
            break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            scheduler_state = scheduler.state_dict() if scheduler else None
            save_package(best_model_path, model, vocab, tokenizer, config,
                         INTENTS, BIO_TAGS, OPERATIONS, ACTION_MODES,
                         TARGET_SCOPES, VALUE_TYPES, char_vocab,
                         intent_prototypes=None, scheduler_state=scheduler_state)
            print(f"  ⭐ Novo melhor modelo salvo em '{best_model_path}' (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  ⏳ Early stopping: {patience_counter}/{config.patience}")

        save_checkpoint(
            checkpoint_path, epoch, model, optimizer, scheduler,
            best_val_loss, patience_counter, vocab, tokenizer, config,
            INTENTS, BIO_TAGS, OPERATIONS, ACTION_MODES,
            TARGET_SCOPES, VALUE_TYPES, char_vocab,
            train_indices, val_indices
        )
        print(f"  💾 Checkpoint atualizado em '{checkpoint_path}' (época {epoch+1})")

        if patience_counter >= config.patience:
            print(f"🛑 Early stopping na época {epoch+1} (sem melhora por {config.patience} épocas)")
            break

    # Carregar melhor modelo
    if os.path.isfile(best_model_path):
        best_package = torch.load(best_model_path, map_location=device, weights_only=False)
        expected_checksum = best_package.get("checksum")
        actual_checksum = _state_checksum(best_package["model_state"])
        if expected_checksum and expected_checksum != actual_checksum:
            raise RuntimeError("Checksum do modelo não confere. O arquivo pode estar corrompido.")
        model.load_state_dict(best_package["model_state"])
        model.to(device)
        print(f"\n✅ Melhor modelo recarregado de '{best_model_path}' (val_loss: {best_val_loss:.4f})")

    # Protótipos
    intent_prototypes = build_intent_prototypes(model, train_loader, device, len(INTENTS))
    save_package(best_model_path, model, vocab, tokenizer, config,
                 INTENTS, BIO_TAGS, OPERATIONS, ACTION_MODES,
                 TARGET_SCOPES, VALUE_TYPES, char_vocab, intent_prototypes)
    print("🧭 Protótipos de intenção calculados para detecção UNKNOWN.")

    intent_acc, ent_f1 = evaluate_generalization(model, tokenizer, vocab, config, device, char_vocab,
                                                 idx2intent, idx2bio, idx2operation,
                                                 idx2action_mode, idx2target_scope, idx2value_type,
                                                 intent_prototypes)
    if intent_acc < 0.85 or ent_f1 < 0.85:
        print("\n⚠️  Generalização abaixo de 85% — considere mais diversidade sintática no dataset.")

    print(f"\n✅ Treinamento concluído. Melhor modelo salvo em '{best_model_path}'.")

# ----------------------------------------------------------------------
# OOD Test Set (mantido)
# ----------------------------------------------------------------------
OOD_TEST_SET = [
    ("faz a lâmpada do quarto funcionar", "TURN_ON", [("lâmpada", "DEVICE"), ("quarto", "LOCATION")]),
    ("quero a luz do quarto ligada", "TURN_ON", [("luz", "DEVICE"), ("quarto", "LOCATION")]),
    ("deixa a luz do corredor acesa", "TURN_ON", [("luz", "DEVICE"), ("corredor", "LOCATION")]),
    ("não quero mais a lâmpada do quarto ligada", "TURN_OFF", [("lâmpada", "DEVICE"), ("quarto", "LOCATION")]),
    ("tira a luz da sala", "TURN_OFF", [("luz", "DEVICE"), ("sala", "LOCATION")]),
    ("esfria o ar condicionado da sala", "SET_TEMPERATURE", [("ar condicionado", "DEVICE"), ("sala", "LOCATION")]),
    ("coloca o ventilador da cozinha mais rápido", "SET_SPEED", [("ventilador", "DEVICE"), ("cozinha", "LOCATION")]),
    ("pode destravar a janela da sala?", "OPEN", [("janela", "DEVICE"), ("sala", "LOCATION")]),
    ("fecha bem a porta da garagem", "CLOSE", [("porta", "DEVICE"), ("garagem", "LOCATION")]),
    ("como está a luz da cozinha", "GET_STATUS", [("luz", "DEVICE"), ("cozinha", "LOCATION")]),
    ("qual a temperatura atual do ar condicionado do quarto", "GET_STATUS", [("ar condicionado", "DEVICE"), ("quarto", "LOCATION")]),
    ("aumenta bastante o brilho da luminária da sala", "SET_BRIGHTNESS", [("luminária", "DEVICE"), ("sala", "LOCATION")]),
    ("deixa a luz do quarto azul", "SET_COLOR", [("luz", "DEVICE"), ("quarto", "LOCATION"), ("azul", "COLOR")]),
    ("para o aspirador da garagem imediatamente", "STOP", [("aspirador", "DEVICE"), ("garagem", "LOCATION")]),
    ("começa a lavadora do banheiro", "START", [("lavadora", "DEVICE"), ("banheiro", "LOCATION")]),
    ("abaixa o volume da tv da sala", "SET_VOLUME", [("tv", "DEVICE"), ("sala", "LOCATION")]),
    ("sobe a voltagem da fonte de bancada", "SET_VOLTAGE", [("fonte de bancada", "DEVICE")]),
    ("pode acender a luz da sala?", "TURN_ON", [("luz", "DEVICE"), ("sala", "LOCATION")]),
    ("apaga a iluminação do quarto", "TURN_OFF", [("iluminação", "DEVICE"), ("quarto", "LOCATION")]),
    ("interrompe o ventilador da cozinha", "STOP", [("ventilador", "DEVICE"), ("cozinha", "LOCATION")]),
    ("inicia a máquina da lavanderia", "START", [("máquina", "DEVICE"), ("lavanderia", "LOCATION")]),
    ("me diga se a porta da garagem está aberta", "GET_STATUS", [("porta", "DEVICE"), ("garagem", "LOCATION")]),
    ("deixa a lâmpada da sala em 40 por cento", "SET_BRIGHTNESS", [("lâmpada", "DEVICE"), ("sala", "LOCATION"), ("40", "VALUE")]),
    ("coloca a luz da cozinha em vermelho", "SET_COLOR", [("luz", "DEVICE"), ("cozinha", "LOCATION"), ("vermelho", "COLOR")]),
    ("reduz a velocidade do exaustor do banheiro", "SET_SPEED", [("exaustor", "DEVICE"), ("banheiro", "LOCATION")]),
    ("fecha a janela do quarto agora", "CLOSE", [("janela", "DEVICE"), ("quarto", "LOCATION")]),
    ("abre a porta da sala", "OPEN", [("porta", "DEVICE"), ("sala", "LOCATION")]),
]

def build_intent_prototypes(model, dataloader, device, num_intents):
    model.eval()
    sums = [None] * num_intents
    counts = [0] * num_intents
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculando protótipos", leave=False):
            input_ids = batch["input_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["intent_labels"].to(device)
            (intent_logits, entity_logits, operation_logits,
             action_mode_logits, target_scope_logits, value_type_logits,
             projected, _) = model(input_ids, mask, char_ids)
            projected = F.normalize(projected, dim=-1).cpu()
            for i, label in enumerate(labels.cpu().tolist()):
                if sums[label] is None:
                    sums[label] = projected[i].clone()
                else:
                    sums[label] += projected[i]
                counts[label] += 1
    prototypes = []
    for i in range(num_intents):
        if counts[i] == 0:
            prototypes.append(torch.zeros(model.projection_head[-1].out_features))
        else:
            prototypes.append(F.normalize(sums[i] / counts[i], dim=0))
    return torch.stack(prototypes)

def evaluate_generalization(model, tokenizer, vocab, config, device, char_vocab,
                            idx2intent, idx2bio, idx2operation,
                            idx2action_mode, idx2target_scope, idx2value_type,
                            intent_prototypes=None):
    print("\n🧪 Teste de generalização (frases fora da distribuição do gerador):")
    intent_correct = 0
    ent_tp = ent_fp = ent_fn = 0

    for text, expected_intent, expected_entities in tqdm(OOD_TEST_SET, desc="Testando generalização", leave=False):
        result = predict(model, tokenizer, vocab, text, config, device, char_vocab,
                         idx2intent, idx2bio, idx2operation,
                         idx2action_mode, idx2target_scope, idx2value_type,
                         intent_prototypes)
        pred_intent = result["intent"]
        ok_intent = pred_intent == expected_intent
        intent_correct += int(ok_intent)

        expected_set = {(etype, val.lower()) for val, etype in expected_entities}
        predicted_set = {(e["type"], e["value"].lower()) for e in result["entities"]}
        ent_tp += len(expected_set & predicted_set)
        ent_fp += len(predicted_set - expected_set)
        ent_fn += len(expected_set - predicted_set)

        flag = "✅" if ok_intent else "❌"
        print(f"  {flag} '{text}'")
        print(f"      esperado: {expected_intent} {sorted(expected_set)}")
        print(f"      previsto: {pred_intent} (conf {result['confidence']:.2f}) {sorted(predicted_set)}")

    n = len(OOD_TEST_SET)
    intent_acc = intent_correct / n
    precision = ent_tp / (ent_tp + ent_fp + 1e-9)
    recall = ent_tp / (ent_tp + ent_fn + 1e-9)
    ent_f1 = 2 * precision * recall / (precision + recall + 1e-9)

    print(f"\n  Generalização — Intent Acc: {intent_acc:.4f} ({intent_correct}/{n})")
    print(f"  Generalização — Entity F1: {ent_f1:.4f} (precision {precision:.4f}, recall {recall:.4f})")
    return intent_acc, ent_f1

# ----------------------------------------------------------------------
# Carregamento do dataset compacto (agora com todos os maps)
# ----------------------------------------------------------------------
def _validate_index_map(name: str, mapping: Dict[str, int]):
    """Valida um mapa semântico: IDs inteiros, únicos e contíguos."""
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"Mapa '{name}' ausente ou inválido.")
    try:
        ids = [int(v) for v in mapping.values()]
    except Exception as exc:
        raise ValueError(f"Mapa '{name}' contém IDs não inteiros.") from exc
    if len(ids) != len(set(ids)):
        raise ValueError(f"Mapa '{name}' contém IDs duplicados: {mapping}")
    expected = list(range(len(ids)))
    if sorted(ids) != expected:
        raise ValueError(
            f"Mapa '{name}' deve possuir IDs contíguos 0..N-1. Recebido: {sorted(ids)}"
        )
    return {str(k): int(v) for k, v in mapping.items()}


def _read_entity_type(ent: Any, entity_type_map: Dict[str, int], idx2entity: Dict[int, str]):
    """Aceita os dois formatos de entidade do dataset:

      novo: {start, end, type: 2, value: "luz"}
      legado: {start, end, type: "DEVICE"}
      legado compacto: [start, end, "DEVICE"] / [start, end, 2]
    """
    if isinstance(ent, dict):
        if "start" not in ent or "end" not in ent or "type" not in ent:
            raise ValueError(f"Entidade inválida: {ent}")
        start, end, typ = int(ent["start"]), int(ent["end"]), ent["type"]
    elif isinstance(ent, (list, tuple)) and len(ent) >= 3:
        start, end, typ = int(ent[0]), int(ent[1]), ent[2]
    else:
        raise ValueError(f"Formato de entidade inválido: {ent!r}")

    if start < 0 or end < start:
        raise ValueError(f"Span inválido: ({start}, {end})")

    if isinstance(typ, bool):
        raise ValueError(f"Tipo de entidade booleano inválido: {typ}")

    if isinstance(typ, (int, np.integer)) or (isinstance(typ, str) and typ.strip().isdigit()):
        type_id = int(typ)
        if type_id not in idx2entity:
            raise ValueError(
                f"ID de entidade desconhecido: {type_id}. "
                f"IDs válidos: {sorted(idx2entity)}"
            )
        return start, end, type_id

    type_str = str(typ)
    if type_str not in entity_type_map:
        raise ValueError(
            f"Tipo de entidade desconhecido: {type_str}. "
            f"Tipos disponíveis: {list(entity_type_map.keys())}"
        )
    return start, end, int(entity_type_map[type_str])


def _validate_compact_item(item: Dict[str, Any], idx_maps: Dict[str, Dict[str, int]], idx2entities: Dict[int, str]):
    """Validação rigorosa de uma amostra antes de entrar no TensorDataset."""
    if not isinstance(item, dict):
        raise ValueError(f"Amostra não é objeto JSON: {item!r}")
    for required in ("text", "intent", "entities"):
        if required not in item:
            raise ValueError(f"Item do dataset compacto sem campo obrigatório '{required}'.")
    if not isinstance(item["text"], str) or not item["text"].strip():
        raise ValueError("Campo 'text' vazio ou inválido.")

    for field, default in (
        ("intent", None), ("operation", 0), ("action_mode", 0),
        ("target_scope", 0), ("value_type", 5)
    ):
        value = item[field] if field in item else default
        if value is None:
            raise ValueError("Intent ausente.")
        try:
            value = int(value)
        except Exception as exc:
            raise ValueError(f"Campo '{field}' não é inteiro: {value!r}") from exc
        valid_ids = set(idx_maps[field].values())
        if value not in valid_ids:
            raise ValueError(f"ID inválido em '{field}': {value}. Válidos: {sorted(valid_ids)}")

    text_len = len(item["text"])
    for ent in item.get("entities", []):
        start, end, type_id = _read_entity_type(ent, idx_maps["entity_type"], idx2entities)
        if end > text_len:
            raise ValueError(
                f"Span da entidade ({start}, {end}) ultrapassa o texto de {text_len} caracteres: {item['text']!r}"
            )


def load_compact_dataset(json_path: str):
    """Carrega dataset 63.x e também formatos compactos legados.

    A V63.2 grava `entities[*].type` como ID numérico. A versão 11.8
    esperava exclusivamente o nome textual (DEVICE, LOCATION, ...),
    causando o erro 'Tipo de entidade desconhecido: 2'.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict) or not raw.get("compact", False):
        raise ValueError(
            "Arquivo não está no formato compacto esperado (chave 'compact': true)."
        )

    required_maps = {
        "intent_map": raw.get("intent_map"),
        "entity_type_map": raw.get("entity_type_map"),
        "operation_map": raw.get("operation_map"),
        "action_mode_map": raw.get("action_mode_map"),
        "target_scope_map": raw.get("target_scope_map"),
        "value_type_map": raw.get("value_type_map"),
    }
    defaults = {
        "operation_map": {"NONE": 0},
        "action_mode_map": {"IMMEDIATE": 0},
        "target_scope_map": {"SINGLE": 0},
        "value_type_map": {"UNKNOWN": 0},
    }
    for name, value in list(required_maps.items()):
        if value is None:
            value = defaults.get(name)
        required_maps[name] = _validate_index_map(name, value)

    intent_map = required_maps["intent_map"]
    entity_type_map = required_maps["entity_type_map"]
    operation_map = required_maps["operation_map"]
    action_mode_map = required_maps["action_mode_map"]
    target_scope_map = required_maps["target_scope_map"]
    value_type_map = required_maps["value_type_map"]
    data_compact = raw.get("data")

    if not isinstance(data_compact, list) or not data_compact:
        raise ValueError("Dataset compacto não contém uma lista 'data' válida e não vazia.")

    idx2entity = {v: k for k, v in entity_type_map.items()}
    idx_maps = {
        "intent": intent_map,
        "entity_type": entity_type_map,
        "operation": operation_map,
        "action_mode": action_mode_map,
        "target_scope": target_scope_map,
        "value_type": value_type_map,
    }

    global ENTITY_TYPES, entity_type2idx, idx2entity_type
    global OPERATIONS, operation2idx, idx2operation
    global ACTION_MODES, action_mode2idx, idx2action_mode
    global TARGET_SCOPES, target_scope2idx, idx2target_scope
    global VALUE_TYPES, value_type2idx, idx2value_type

    ENTITY_TYPES = sorted(entity_type_map.keys(), key=lambda k: entity_type_map[k])
    entity_type2idx = entity_type_map
    idx2entity_type = {v: k for k, v in entity_type_map.items()}
    OPERATIONS = sorted(operation_map.keys(), key=lambda k: operation_map[k])
    operation2idx = operation_map
    idx2operation = {v: k for k, v in operation_map.items()}
    ACTION_MODES = sorted(action_mode_map.keys(), key=lambda k: action_mode_map[k])
    action_mode2idx = action_mode_map
    idx2action_mode = {v: k for k, v in action_mode_map.items()}
    TARGET_SCOPES = sorted(target_scope_map.keys(), key=lambda k: target_scope_map[k])
    target_scope2idx = target_scope_map
    idx2target_scope = {v: k for k, v in target_scope_map.items()}
    VALUE_TYPES = sorted(value_type_map.keys(), key=lambda k: value_type_map[k])
    value_type2idx = value_type_map
    idx2value_type = {v: k for k, v in value_type_map.items()}

    processed_data = []
    for n, item in enumerate(data_compact):
        # Normaliza os campos principais para IDs.
        item2 = dict(item)
        for field, default in (
            ("operation", 0), ("action_mode", 0),
            ("target_scope", 0), ("value_type", None)
        ):
            if field not in item2:
                if field == "value_type":
                    # UNKNOWN pode ter qualquer ID; procurar pelo nome é mais seguro.
                    item2[field] = value_type_map.get("UNKNOWN", 0)
                else:
                    item2[field] = default

        # Aceita datasets onde cabeças ainda estão em nomes textuais.
        for field, mapping in (
            ("intent", intent_map), ("operation", operation_map),
            ("action_mode", action_mode_map), ("target_scope", target_scope_map),
            ("value_type", value_type_map)
        ):
            value = item2[field]
            if isinstance(value, str) and value in mapping:
                item2[field] = mapping[value]
            else:
                item2[field] = int(value)

        normalized_entities = []
        for ent in item2["entities"]:
            normalized_entities.append(list(_read_entity_type(ent, entity_type_map, idx2entity)))
        item2["entities"] = normalized_entities

        try:
            _validate_compact_item(item2, idx_maps, idx2entity)
        except Exception as exc:
            raise ValueError(f"Erro no exemplo {n}: {exc}") from exc

        processed_data.append({
            "text": item2["text"],
            "intent": int(item2["intent"]),
            "operation": int(item2["operation"]),
            "action_mode": int(item2["action_mode"]),
            "target_scope": int(item2["target_scope"]),
            "value_type": int(item2["value_type"]),
            "entities": normalized_entities,
        })

    print(f"📦 Dataset {raw.get('version', 'sem versão')} validado: {len(processed_data)} exemplos.")
    print(f"   Intenções: {len(intent_map)} | Entidades: {len(entity_type_map)} | "
          f"Operações: {len(operation_map)} | Modos: {len(action_mode_map)} | "
          f"Escopos: {len(target_scope_map)} | Tipos de valor: {len(value_type_map)}")
    return processed_data, intent_map, entity_type_map, operation_map, action_mode_map, target_scope_map, value_type_map


def remap_data_to_label_maps(data, current_maps, target_maps):
    """Converte IDs do dataset para a ordem dos labels do checkpoint/modelo.

    Isso evita um bug silencioso: mapas semanticamente iguais, mas com IDs
    em ordem diferente, faziam o modelo receber o rótulo errado no resume.
    """
    current_intent, current_entity, current_op, current_am, current_ts, current_vt = current_maps
    target_intent, target_entity, target_op, target_am, target_ts, target_vt = target_maps

    def remap_id(value, src, dst, field):
        src_idx2name = {int(v): k for k, v in src.items()}
        name = src_idx2name.get(int(value))
        if name is None or name not in dst:
            raise ValueError(f"Checkpoint incompatível em {field}: ID {value} / nome {name!r}.")
        return int(dst[name])

    out = []
    for item in data:
        x = dict(item)
        x["intent"] = remap_id(x["intent"], current_intent, target_intent, "intent")
        x["operation"] = remap_id(x["operation"], current_op, target_op, "operation")
        x["action_mode"] = remap_id(x["action_mode"], current_am, target_am, "action_mode")
        x["target_scope"] = remap_id(x["target_scope"], current_ts, target_ts, "target_scope")
        x["value_type"] = remap_id(x["value_type"], current_vt, target_vt, "value_type")
        x["entities"] = [
            [start, end, remap_id(type_id, current_entity, target_entity, "entity_type")]
            for start, end, type_id in x["entities"]
        ]
        out.append(x)
    return out

# ----------------------------------------------------------------------
# Menu principal (atualizado)
# ----------------------------------------------------------------------
def main():
    set_seed(SEED)
    config = NLUConfig()
    
    json_file = "dataset_compact.json"
    checkpoint_path = "checkpoint_nlu_v12_1_contexto.pt"
    best_model_path = "house_nlu_v12_1_contexto.bin"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Dispositivo: {device}")
    print("=" * 60)
    print("  🏠 SISTEMA NLU - MENU INTERATIVO (v12.1 / Dataset 63.x — CONTEXTO ESPECIALIZADO)")
    print("=" * 60)
    print("Escolha uma opção:\n")
    print("  1 - Treinar do ZERO (fresh)")
    print("  2 - Retomar treino de CHECKPOINT (resume)")
    print("  3 - Fine-tuning com NOVOS DADOS (expande vocabulário se necessário)")
    print("  4 - Avaliar modelo existente (sem treinar)")
    print("  5 - Sair")
    print("=" * 60)
    
    choice = input("Digite o número da opção: ").strip()
    
    if choice == "5":
        print("Saindo...")
        sys.exit(0)
    
    if choice not in ("1", "2", "3", "4"):
        print("❌ Opção inválida. Saindo.")
        sys.exit(1)
    
    if not os.path.isfile(json_file):
        print(f"❌ Arquivo '{json_file}' não encontrado. Certifique-se de que o dataset está presente.")
        sys.exit(1)
    
    data, intent_map, entity_type_map, operation_map, action_mode_map, target_scope_map, value_type_map = load_compact_dataset(json_file)
    print(f"📊 Carregados {len(data)} exemplos do dataset.")
    
    char_vocab = {'<PAD>': 0, '<UNK>': 1}
    for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789áàâãéèêíïóôõúçÁÀÂÃÉÈÊÍÏÓÔÕÚÇ,.-!? ":
        if c not in char_vocab:
            char_vocab[c] = len(char_vocab)
    
    if choice == "1":
        print("🆕 Iniciando treinamento do zero...")
        run_training(config, data, intent_map, entity_type_map, operation_map,
                     action_mode_map, target_scope_map, value_type_map,
                     char_vocab, device, fresh=True, resume=False, fine_tune=False,
                     checkpoint_path=checkpoint_path, best_model_path=best_model_path)
    
    elif choice == "2":
        if not os.path.isfile(checkpoint_path):
            print(f"❌ Checkpoint '{checkpoint_path}' não encontrado. Não é possível retomar.")
            sys.exit(1)
        print("🔄 Retomando treino do checkpoint...")
        run_training(config, data, intent_map, entity_type_map, operation_map,
                     action_mode_map, target_scope_map, value_type_map,
                     char_vocab, device, fresh=False, resume=True, fine_tune=False,
                     checkpoint_path=checkpoint_path, best_model_path=best_model_path)
    
    elif choice == "3":
        if not os.path.isfile(checkpoint_path):
            print(f"❌ Checkpoint '{checkpoint_path}' não encontrado. Fine-tuning requer um modelo pré-treinado.")
            sys.exit(1)
        print("🔧 Iniciando fine-tuning com novos dados...")
        run_training(config, data, intent_map, entity_type_map, operation_map,
                     action_mode_map, target_scope_map, value_type_map,
                     char_vocab, device, fresh=False, resume=False, fine_tune=True,
                     checkpoint_path=checkpoint_path, best_model_path=best_model_path)
    
    elif choice == "4":
        if not os.path.isfile(best_model_path):
            print(f"❌ Modelo '{best_model_path}' não encontrado. Execute treinamento primeiro.")
            sys.exit(1)
        print("📊 Carregando modelo para avaliação...")
        package = torch.load(best_model_path, map_location=device, weights_only=False)
        intent_labels = package["intent_labels"]
        bio_labels = package["bio_labels"]
        operation_labels = package.get("operation_labels", ["NONE"])
        action_mode_labels = package.get("action_mode_labels", ["IMMEDIATE"])
        target_scope_labels = package.get("target_scope_labels", ["SINGLE"])
        value_type_labels = package.get("value_type_labels", ["UNKNOWN"])

        INTENTS = list(intent_labels)
        intent2idx = {name: i for i, name in enumerate(INTENTS)}
        idx2intent = {i: name for i, name in enumerate(INTENTS)}
        ENTITY_TYPES = []
        for tag in bio_labels:
            if tag.startswith("B-"):
                ent = tag[2:]
                if ent not in ENTITY_TYPES:
                    ENTITY_TYPES.append(ent)
        ENTITY_TYPES = sorted(ENTITY_TYPES)
        entity_type2idx = {name: i for i, name in enumerate(ENTITY_TYPES)}
        idx2entity_type = {i: name for i, name in enumerate(ENTITY_TYPES)}
        BIO_TAGS = bio_labels
        bio2idx = {tag: i for i, tag in enumerate(BIO_TAGS)}
        idx2bio = {i: tag for i, tag in enumerate(BIO_TAGS)}
        NUM_BIO_TAGS = len(BIO_TAGS)
        OPERATIONS = list(operation_labels)
        operation2idx = {name: i for i, name in enumerate(OPERATIONS)}
        idx2operation = {i: name for i, name in enumerate(OPERATIONS)}
        ACTION_MODES = list(action_mode_labels)
        action_mode2idx = {name: i for i, name in enumerate(ACTION_MODES)}
        idx2action_mode = {i: name for i, name in enumerate(ACTION_MODES)}
        TARGET_SCOPES = list(target_scope_labels)
        target_scope2idx = {name: i for i, name in enumerate(TARGET_SCOPES)}
        idx2target_scope = {i: name for i, name in enumerate(TARGET_SCOPES)}
        VALUE_TYPES = list(value_type_labels)
        value_type2idx = {name: i for i, name in enumerate(VALUE_TYPES)}
        idx2value_type = {i: name for i, name in enumerate(VALUE_TYPES)}
        
        vocab = Vocab()
        vocab.WordToIdx = package["vocab"].WordToIdx if hasattr(package["vocab"], "WordToIdx") else package["vocab_word_to_idx"]
        vocab.IdxToWord = {v: k for k, v in vocab.WordToIdx.items()}
        tokenizer = SubwordTokenizer(package["tokenizer_merges"])
        char_vocab = package.get("char_vocab", char_vocab)
        config = package.get("config", config)
        
        model = NLUModel(
            config=config,
            vocab_size=vocab.size(),
            num_intents=len(intent2idx),
            num_bio_tags=NUM_BIO_TAGS,
            num_operations=len(operation2idx),
            num_action_modes=len(action_mode2idx),
            num_target_scopes=len(target_scope2idx),
            num_value_types=len(value_type2idx),
            char_vocab_size=len(char_vocab)
        ).to(device)
        model.load_state_dict(package["model_state"])
        model.eval()
        print("✅ Modelo carregado com sucesso.")
        
        print("🔄 Preparando dataset para avaliação...")
        temp_dataset = NLUDataset(data, vocab, tokenizer, config, char_vocab, augment=False)
        eval_loader = DataLoader(temp_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
        
        print("📈 Avaliando no dataset completo (pode demorar alguns minutos)...")
        (val_loss, val_intent_loss, val_entity_loss, val_operation_loss,
         val_action_mode_loss, val_target_scope_loss, val_value_type_loss,
         val_intent_acc, val_operation_acc, val_action_mode_acc,
         val_target_scope_acc, val_value_type_acc, val_bio_f1, val_span_f1) = evaluate(
            model, eval_loader, config, device, idx2bio)
        print(f"\n📈 Desempenho no dataset completo:")
        print(f"  Loss: {val_loss:.4f}")
        print(f"  Intent Acc: {val_intent_acc:.4f}")
        print(f"  Operation Acc: {val_operation_acc:.4f}")
        print(f"  Action Mode Acc: {val_action_mode_acc:.4f}")
        print(f"  Target Scope Acc: {val_target_scope_acc:.4f}")
        print(f"  Value Type Acc: {val_value_type_acc:.4f}")
        print(f"  Entity F1 (span): {val_span_f1:.4f}")
        
        intent_prototypes = package.get("intent_prototypes")
        if intent_prototypes is not None:
            print("🔄 Usando protótipos salvos.")
            intent_prototypes = intent_prototypes.to(device)
        else:
            print("🔄 Calculando protótipos das intenções (pode demorar)...")
            intent_prototypes = build_intent_prototypes(model, eval_loader, device, len(INTENTS))
            print("✅ Protótipos calculados.")
        
        print("🧪 Executando teste de generalização (27 frases)...")
        intent_acc, ent_f1 = evaluate_generalization(model, tokenizer, vocab, config, device, char_vocab,
                                                     idx2intent, idx2bio, idx2operation,
                                                     idx2action_mode, idx2target_scope, idx2value_type,
                                                     intent_prototypes)
        print("\n✅ Avaliação concluída.")

if __name__ == "__main__":
    main()