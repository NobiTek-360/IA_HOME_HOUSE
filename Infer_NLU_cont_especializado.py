import sys
import math
import re
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# CONFIGURAÇÃO (igual ao treino, com os parâmetros contrastivos)
# ------------------------------------------------------------
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
# ------------------------------------------------------------
# UTILITÁRIOS DE TEXTO
# ------------------------------------------------------------
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
        if merged:
            prev = merged[-1]
            if prev["type"] == ent["type"] and word_indices[0] == prev["word_indices"][-1] + 1:
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

# ------------------------------------------------------------
# TOKENIZADOR BPE
# ------------------------------------------------------------
class SubwordTokenizer:
    def __init__(self, merges=None):
        self.Merges = []
        for m in (merges or []):
            if isinstance(m, tuple):
                self.Merges.append(f"{m[0]} {m[1]}")
            else:
                self.Merges.append(str(m))
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

# ------------------------------------------------------------
# VOCABULÁRIO
# ------------------------------------------------------------
class Vocab:
    def __init__(self):
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.WordToIdx = {}
        self.IdxToWord = {}
        self.pad_idx = 0
        self.unk_idx = 1

    def size(self):
        return len(self.WordToIdx)

# ------------------------------------------------------------
# ------------------------------------------------------------
# MÓDULOS NEURAIS — arquitetura V12.1 contexto especializado
# Idênticos à arquitetura usada no treinamento V12.1.
# ------------------------------------------------------------
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

class NLUEngine:
    """Inferência compatível com house_nlu_v12_1_contexto.bin — arquitetura V12.1-context-specialized / Dataset 63.x."""
    def __init__(self, model_path, device=None, confidence_threshold=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        package = torch.load(model_path, map_location=self.device, weights_only=False)

        self.package_version = str(package.get("version", package.get("pipeline_version", "unknown")))
        self.config = package["config"]
        saved_arch = package.get("architecture_version", getattr(self.config, "architecture_version", "unknown"))
        if saved_arch != "v12.1-context-specialized":
            raise RuntimeError(
                f"Modelo incompatível: arquitetura '{saved_arch}'. "
                "Esta inferência exige house_nlu_v12_1_contexto.bin treinado com a V12.1."
            )
        if confidence_threshold is not None:
            self.config.confidence_threshold = confidence_threshold

        self.vocab = Vocab()
        vocab_obj = package["vocab"]
        self.vocab.WordToIdx = dict(getattr(vocab_obj, "WordToIdx", package.get("vocab_word_to_idx", {})))
        self.vocab.IdxToWord = dict(getattr(vocab_obj, "IdxToWord",
                                             {v:k for k,v in self.vocab.WordToIdx.items()}))
        self.vocab.pad_idx = int(getattr(vocab_obj, "pad_idx", 0))
        self.vocab.unk_idx = int(getattr(vocab_obj, "unk_idx", 1))

        self.tokenizer = SubwordTokenizer(package.get("tokenizer_merges", []))
        self.char_vocab = package.get("char_vocab", {"<PAD>":0, "<UNK>":1})

        self.intent_labels = list(package["intent_labels"])
        self.bio_labels = list(package["bio_labels"])
        self.operation_labels = list(package.get(
            "operation_labels", ["NONE", "INCREASE", "DECREASE", "SET"]))
        self.action_mode_labels = list(package.get(
            "action_mode_labels", ["IMMEDIATE", "SCHEDULED", "RECURRING"]))
        self.target_scope_labels = list(package.get(
            "target_scope_labels", ["SINGLE", "GROUP"]))
        self.value_type_labels = list(package.get(
            "value_type_labels",
            ["NUMBER", "PERCENTAGE", "TEMPERATURE", "VOLTAGE", "COLOR", "UNKNOWN"]))

        self.idx2intent = {i:x for i,x in enumerate(self.intent_labels)}
        self.idx2bio = {i:x for i,x in enumerate(self.bio_labels)}
        self.idx2operation = {i:x for i,x in enumerate(self.operation_labels)}
        self.idx2action_mode = {i:x for i,x in enumerate(self.action_mode_labels)}
        self.idx2target_scope = {i:x for i,x in enumerate(self.target_scope_labels)}
        self.idx2value_type = {i:x for i,x in enumerate(self.value_type_labels)}

        self.model = NLUModel(
            config=self.config,
            vocab_size=self.vocab.size(),
            num_intents=len(self.intent_labels),
            num_bio_tags=len(self.bio_labels),
            num_operations=len(self.operation_labels),
            num_action_modes=len(self.action_mode_labels),
            num_target_scopes=len(self.target_scope_labels),
            num_value_types=len(self.value_type_labels),
            char_vocab_size=len(self.char_vocab)).to(self.device)

        state = package["model_state"]
        try:
            self.model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint incompatível com a arquitetura V12.1-context-specialized / Dataset 63.x. "
                "Verifique se o modelo foi treinado com "
                "operation/action_mode/target_scope/value_type heads."
            ) from exc
        self.model.eval()
        self.intent_prototypes = package.get("intent_prototypes")
        if isinstance(self.intent_prototypes, torch.Tensor):
            self.intent_prototypes = self.intent_prototypes.to(self.device)

    def _unknown_decision(self, confidence, margin, prototype_distance):
        enabled = getattr(self.config, "unknown_detection_enabled", False)
        if not enabled:
            return confidence < getattr(self.config, "confidence_threshold", 0.5)

        low_conf = confidence < getattr(
            self.config, "unknown_confidence_threshold",
            getattr(self.config, "confidence_threshold", 0.5))
        low_margin = margin < getattr(self.config, "unknown_margin_threshold", 0.10)
        far = self.intent_prototypes is not None and prototype_distance > getattr(
            self.config, "unknown_distance_threshold", 0.50)

        if getattr(self.config, "unknown_require_two_signals", True):
            return (low_conf and low_margin) or (low_conf and far) or (low_margin and far)
        return low_conf or low_margin or far

    def predict(self, text, debug=False):
        if not isinstance(text, str) or not text.strip():
            return self._empty_result()

        config = self.config
        word_spans = get_word_spans(text)
        words = [w for _,_,w in word_spans]

        all_subtokens, token_to_word_idx = [], []
        for wi, word in enumerate(words):
            subs = self.tokenizer.tokenize_word(word)
            all_subtokens.extend(subs)
            token_to_word_idx.extend([wi] * len(subs))

        if not all_subtokens:
            return self._empty_result()

        ids, char_ids_list = [], []
        unk_char = self.char_vocab.get("<UNK>", 1)
        for sub in all_subtokens:
            ids.append(self.vocab.WordToIdx.get(sub, self.vocab.unk_idx))
            chars = [self.char_vocab.get(c, unk_char) for c in sub]
            chars = chars[:config.max_char_len]
            chars += [0] * (config.max_char_len - len(chars))
            char_ids_list.append(chars)

        truncated = len(ids) > config.max_seq_len
        ids = ids[:config.max_seq_len]
        token_to_word_idx = token_to_word_idx[:config.max_seq_len]
        char_ids_list = char_ids_list[:config.max_seq_len]
        real_len = len(ids)

        ids += [self.vocab.pad_idx] * (config.max_seq_len - real_len)
        char_ids_list += [[0]*config.max_char_len] * (config.max_seq_len-real_len)
        token_to_word_idx += [-1] * (config.max_seq_len-real_len)

        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        char_ids = torch.tensor([char_ids_list], dtype=torch.long, device=self.device)
        mask = torch.tensor([[1]*real_len + [0]*(config.max_seq_len-real_len)],
                            dtype=torch.long, device=self.device)

        with torch.no_grad():
            (intent_logits, entity_logits, operation_logits,
             action_mode_logits, target_scope_logits, value_type_logits,
             projected, _) = self.model(input_ids, mask, char_ids)

            def head(logits, mapping):
                probs = F.softmax(logits[0], dim=-1)
                idx = int(probs.argmax())
                return idx, float(probs[idx])

            intent_probs = F.softmax(intent_logits[0], dim=-1)
            topk = torch.topk(intent_probs, min(2, len(intent_probs)))
            intent_idx = int(topk.indices[0])
            intent_conf = float(topk.values[0])
            margin = float(topk.values[0] - topk.values[1]) if len(topk.values) > 1 else intent_conf

            op_idx, op_conf = head(operation_logits, self.idx2operation)
            mode_idx, mode_conf = head(action_mode_logits, self.idx2action_mode)
            scope_idx, scope_conf = head(target_scope_logits, self.idx2target_scope)
            value_idx, value_conf = head(value_type_logits, self.idx2value_type)

            proto_distance = 0.0
            if isinstance(self.intent_prototypes, torch.Tensor) and self.intent_prototypes.numel():
                proto = F.normalize(self.intent_prototypes, dim=-1)
                sample = F.normalize(projected[0], dim=-1)
                proto_distance = float(1.0 - torch.matmul(proto, sample).max())

            if getattr(config, "use_crf", True) and hasattr(self.model, "crf"):
                tags_seq = self.model.crf.decode(
                    entity_logits[:, :real_len, :], mask[:, :real_len])[0]
            else:
                tags_seq = entity_logits.argmax(dim=-1)[0, :real_len].tolist()

        bio_tags = [self.idx2bio.get(t, "O") for t in tags_seq]
        entities = self._decode_entities(text, words, bio_tags, token_to_word_idx)

        unknown = self._unknown_decision(intent_conf, margin, proto_distance)
        result = {
            "intent": "UNKNOWN" if unknown else self.idx2intent.get(intent_idx, "UNKNOWN"),
            "raw_intent": self.idx2intent.get(intent_idx, "UNKNOWN"),
            "confidence": intent_conf,
            "intent_margin": margin,
            "prototype_distance": proto_distance,
            "below_threshold": unknown,
            "operation": self.idx2operation.get(op_idx, "NONE"),
            "operation_confidence": op_conf,
            "action_mode": self.idx2action_mode.get(mode_idx, "UNKNOWN"),
            "action_mode_confidence": mode_conf,
            "target_scope": self.idx2target_scope.get(scope_idx, "UNKNOWN"),
            "target_scope_confidence": scope_conf,
            "value_type": self.idx2value_type.get(value_idx, "UNKNOWN"),
            "value_type_confidence": value_conf,
            "entities": entities,
            "truncated": truncated,
            "model_version": self.package_version,
        }

        # Correção semântica mínima para construções explícitas de ajuste.
        # O modelo pode confundir frases sem valor explícito, como
        # "ajustar temperatura do quarto", com GET_STATUS. Nesses casos
        # o objetivo da frase é inequivocamente uma intenção SET_*; porém
        # a ausência de valor continua sendo tratada pela validação abaixo.
        result = self._repair_explicit_adjustment_intent(result, text)

        # Segurança semântica: se não existe valor numérico/colorido na frase,
        # não inventa VALUE/MEASURE. A cabeça continua sendo reportada.
        result["validation"] = self._validate_semantics(result, text)
        if debug:
            print("[DEBUG] Palavras:", words)
            print("[DEBUG] Subpalavras:", all_subtokens[:config.max_seq_len])
            print("[DEBUG] BIO:", bio_tags)
            print("[DEBUG] Resultado:", result)
        return result

    def _postprocess_entities(self, text, entities):
        """Corrige casos óbvios que o BIO pode perder/absorver.
        Não substitui o modelo: apenas recupera entidades temporais/medidas
        explícitas e impede LOCATION de engolir DATE/TIME.
        """
        out = []
        # 1) Quebra LOCATION que absorveu um marcador temporal explícito.
        temporal_re = re.compile(
            r"\b(?:amanhã|amanha|hoje|ontem|depois de amanhã|depois de amanha|"
            r"segunda(?:-feira)?|terça(?:-feira)?|terca(?:-feira)?|quarta(?:-feira)?|"
            r"quinta(?:-feira)?|sexta(?:-feira)?|sábado|sabado|domingo|"
            r"logo mais|em breve|daqui a)\b|\b\d{1,2}:\d{2}\b|\b\d{1,2}h(?:\d{2})?\b",
            re.I,
        )
        for e in entities:
            if e.get("type") != "LOCATION":
                out.append(e)
                continue
            start, end = e["start"], e["end"]
            value = text[start:end]
            matches = list(temporal_re.finditer(value))
            if not matches:
                out.append(e)
                continue
            cut = matches[0].start()
            loc_value = value[:cut].rstrip(" ,")
            if loc_value.strip():
                newe = dict(e)
                newe["value"] = loc_value
                newe["end"] = start + len(loc_value)
                out.append(newe)

        # 2) Recupera DATE/TIME explícitos que o BIO perdeu.
        occupied = {(e["start"], e["end"], e["type"]) for e in out}
        patterns = [
            ("DATE", re.compile(r"\b(?:amanhã|amanha|hoje|ontem|depois de amanhã|depois de amanha|segunda(?:-feira)?|terça(?:-feira)?|terca(?:-feira)?|quarta(?:-feira)?|quinta(?:-feira)?|sexta(?:-feira)?|sábado|sabado|domingo)\b", re.I)),
            ("TIME", re.compile(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}h(?:\d{2})?\b", re.I)),
        ]
        for typ, pat in patterns:
            for m in pat.finditer(text):
                if any(a <= m.start() and m.end() <= b for a,b,_ in occupied):
                    continue
                out.append({"type": typ, "value": m.group(), "start": m.start(), "end": m.end(), "source": "rule"})
                occupied.add((m.start(), m.end(), typ))

        # 3) Recupera MEASURE explícito: 70%, 70 %, 30 graus, 127 V etc.
        measure_re = re.compile(
            r"\b\d+(?:[.,]\d+)?\s*(?:%|por\s*cento|°\s*C|°C|graus?(?:\s*C)?|V|volts?)\b",
            re.I,
        )
        for m in measure_re.finditer(text):
            if any(a <= m.start() and m.end() <= b for a,b,_ in occupied):
                continue
            out.append({"type": "MEASURE", "value": m.group(), "start": m.start(), "end": m.end(), "source": "rule"})
            occupied.add((m.start(), m.end(), "MEASURE"))

        # Ordenação e remoção de duplicatas exatas.
        unique = {}
        for e in out:
            key = (e.get("type"), e.get("start"), e.get("end"))
            unique[key] = e
        return sorted(unique.values(), key=lambda x: (x["start"], x["end"]))

    def _decode_entities(self, text, words, bio_tags, token_to_word_idx):
        entities, current = [], None
        for i, tag in enumerate(bio_tags):
            if tag == "O":
                if current: entities.append(current)
                current = None
            elif tag.startswith("B-"):
                if current: entities.append(current)
                current = {"type":tag[2:], "tokens":[i]}
            elif tag.startswith("I-") and current and tag[2:] == current["type"]:
                current["tokens"].append(i)
            else:
                if current: entities.append(current)
                current = None
        if current: entities.append(current)

        merged = merge_adjacent_entities(entities, token_to_word_idx)
        out = []
        for ent in merged:
            wis = ent["word_indices"]
            wis = [w for w in wis if 0 <= w < len(words)]
            if not wis: continue
            value = " ".join(words[w] for w in wis)
            if not any(c.isalnum() for c in value): continue
            # spans exatos na frase original
            start = min(word_spans := [get_word_spans(text)[w][0] for w in wis])
            end = max(get_word_spans(text)[w][1] for w in wis)
            out.append({"type":ent["type"], "value":value, "start":start, "end":end})
        return self._postprocess_entities(text, out)

    @staticmethod
    def _repair_explicit_adjustment_intent(result, text):
        """Corrige apenas confusões claras entre GET_STATUS e SET_* em
        construções explícitas de ajuste, sem inventar operação ou valor.

        Exemplos: "ajustar temperatura do quarto" -> SET_TEMPERATURE
        com Operation=NONE, que depois é corretamente rejeitado por estar
        incompleto. Perguntas como "qual a temperatura do quarto?" não
        possuem verbo de ajuste e permanecem GET_STATUS.
        """
        intent = result.get("intent")
        raw_intent = result.get("raw_intent")
        if intent not in {"GET_STATUS", "UNKNOWN"} and raw_intent not in {"GET_STATUS", "UNKNOWN"}:
            return result

        low = re.sub(r"\s+", " ", text.lower().strip())

        # Verbos que explicitamente pedem alteração/configuração.
        adjustment_re = re.compile(
            r"\b(?:ajust(?:ar|e|a|ando)|regular(?:|e|a|ando)|"
            r"configur(?:ar|e|a|ando)|defin(?:ir|a|e|indo)|"
            r"alter(?:ar|e|a|ando))\b", re.I
        )
        if not adjustment_re.search(low):
            return result

        target_map = (
            (r"\btemperatura\b", "SET_TEMPERATURE"),
            (r"\bvelocidade\b|\bmais rápido\b|\bmais devagar\b", "SET_SPEED"),
            (r"\bbrilho\b|\bluminosidade\b|\bluz\b", "SET_BRIGHTNESS"),
            (r"\bvolume\b|\bsom\b", "SET_VOLUME"),
            (r"\btensão\b|\bvoltagem\b", "SET_VOLTAGE"),
            (r"\bcor\b|\bcolorir\b", "SET_COLOR"),
        )

        chosen = None
        for pattern, mapped_intent in target_map:
            if re.search(pattern, low, re.I):
                chosen = mapped_intent
                break

        if chosen is None:
            return result

        result = dict(result)
        result["intent"] = chosen
        result["raw_intent"] = chosen
        # A confiança corrigida não deve fingir que veio diretamente da head.
        # Mantemos a confiança original e apenas alteramos a intenção.
        return result

    def _validate_semantics(self, result, text):
        errors = []
        op = result["operation"]
        vt = result["value_type"]
        low = text.lower()

        adjustment_intents = {"SET_BRIGHTNESS","SET_COLOR","SET_SPEED",
                              "SET_TEMPERATURE","SET_VOLTAGE","SET_VOLUME"}

        # Ajustes relativos podem ser perfeitamente válidos sem uma medida
        # explícita. Exemplos: "aumenta a velocidade", "deixe a luz mais forte",
        # "diminua o volume". Nesses casos UNKNOWN é intencional: a operação
        # INCREASE/DECREASE já informa a direção, enquanto não existe um valor
        # absoluto para classificar como NUMBER/PERCENTAGE/TEMPERATURE/etc.
        #
        # Já SET exige um valor explícito. Ex.: "ajustar a temperatura do quarto"
        # continua sendo sinalizado, pois falta o alvo numérico.
        if op == "SET" and vt == "UNKNOWN" and result["intent"] in adjustment_intents:
            errors.append("operação SET sem tipo de valor identificado")

        if result["intent"] in adjustment_intents and op == "NONE":
            errors.append("intenção de ajuste sem operação SET/INCREASE/DECREASE")

        temporal_types = {"DATE", "TIME", "RELATIVE_TIME"}
        has_temporal = any(e.get("type") in temporal_types for e in result.get("entities", []))
        if result["action_mode"] == "SCHEDULED" and not has_temporal:
            errors.append("modo SCHEDULED sem entidade temporal explícita")

        if result["value_type"] == "PERCENTAGE":
            has_measure = any(e.get("type") == "MEASURE" and "%" in e.get("value", "") for e in result.get("entities", []))
            if not has_measure and result["intent"] in adjustment_intents:
                errors.append("VALUE_TYPE=PERCENTAGE sem medida percentual explícita")

        return {"ok": not errors, "errors": errors}

    @staticmethod
    def _empty_result():
        return {
            "intent":"UNKNOWN", "raw_intent":"UNKNOWN", "confidence":0.0,
            "intent_margin":0.0, "prototype_distance":0.0,
            "below_threshold":True, "operation":"NONE",
            "operation_confidence":0.0, "action_mode":"UNKNOWN",
            "action_mode_confidence":0.0, "target_scope":"UNKNOWN",
            "target_scope_confidence":0.0, "value_type":"UNKNOWN",
            "value_type_confidence":0.0, "entities":[], "truncated":False,
            "validation":{"ok":False,"errors":["entrada vazia"]}
        }


class ContextManager:
    def __init__(self, max_turns=5):
        self.max_turns = max_turns
        self.turns = []

    def clear(self):
        self.turns.clear()

    def update(self, result):
        if result.get("entities") or result.get("raw_intent") not in (None, "UNKNOWN"):
            self.turns.append({
                "intent": result.get("raw_intent"),
                "entities": list(result.get("entities", []))})
            self.turns = self.turns[-self.max_turns:]

    def _last_entity(self, entity_type):
        for turn in reversed(self.turns):
            for ent in reversed(turn["entities"]):
                if ent.get("type") == entity_type:
                    return ent
        return None

    def resolve(self, text, result):
        lower = text.lower()
        pronouns = ("ela","ele","isso","isto","essa","esse","aquela","aquele",
                    "aquilo","lá","ali")
        if not any(re.search(r"\b"+re.escape(p)+r"\b", lower) for p in pronouns):
            return result

        resolved = list(result.get("entities", []))
        existing = {(e["type"], e["value"].lower()) for e in resolved}
        for typ in ("DEVICE","LOCATION"):
            ent = self._last_entity(typ)
            if ent and (typ, ent["value"].lower()) not in existing:
                e = dict(ent)
                e["resolved_from_context"] = True
                resolved.append(e)
        out = dict(result)
        out["entities"] = resolved
        out["context_resolved"] = True
        return out

    def predict(self, engine, text, debug=False):
        result = engine.predict(text, debug=debug)
        result = self.resolve(text, result)
        self.update(result)
        return result


def print_result(result):
    """
    Saída compacta e estável para uso humano e futura integração.
    Não imprime raw_intent, spans, source ou estruturas Python.
    """
    print(f"Intent: {result['intent']} (conf: {result['confidence']:.3f})")
    print(f"Operation: {result['operation']} (conf: {result['operation_confidence']:.3f})")
    print(f"Action Mode: {result['action_mode']} (conf: {result['action_mode_confidence']:.3f})")
    print(f"Target Scope: {result['target_scope']} (conf: {result['target_scope_confidence']:.3f})")
    print(f"Value Type: {result['value_type']} (conf: {result['value_type_confidence']:.3f})")

    for ent in result.get("entities", []):
        typ = ent.get("type", "UNKNOWN")
        value = ent.get("value", "")
        if value:
            print(f"{typ}: {value}")

    validation = result.get("validation", {})
    errors = validation.get("errors", [])

    if errors:
        print("Status: ERROR")
        for error in errors:
            print(f"Error: {error}")
    else:
        print("Status: OK")

# ------------------------------------------------------------
# INTERFACE DE LINHA DE COMANDO
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Inferência NLU V12.1-context-specialized — INTENT + ENTITY + OPERATION + ACTION_MODE + TARGET_SCOPE + VALUE_TYPE")
    parser.add_argument("--model", default="house_nlu_v12_1_contexto.bin")
    parser.add_argument("--text", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--context", action="store_true",
                        help="mantém contexto entre turnos e resolve referências simples")
    parser.add_argument("--clear-context", action="store_true")
    args = parser.parse_args()

    engine = NLUEngine(args.model, confidence_threshold=args.threshold)
    context = ContextManager() if args.context else None

    print("NLU V12.1-context-specialized pronto.")
    print("Digite uma frase (Ctrl+C para sair).")

    if args.clear_context:
        context = ContextManager()

    if args.text:
        result = context.predict(engine, args.text, args.debug) if context else engine.predict(args.text, args.debug)
        print_result(result)
        return

    if context:
        print("Contexto: ATIVO")
    try:
        while True:
            text = input("> ").strip()
            if not text:
                continue
            result = context.predict(engine, text, args.debug) if context else engine.predict(text, args.debug)
            print_result(result)
    except (KeyboardInterrupt, EOFError):
        print("\nSaindo.")
        sys.exit(0)

if __name__ == "__main__":
    main()
