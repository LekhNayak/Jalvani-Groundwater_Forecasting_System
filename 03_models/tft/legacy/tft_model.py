"""
Temporal Fusion Transformer (TFT) — from scratch implementation.
Follows Lim et al. (2021) "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting" exactly.

Components:
    GRN     — Gated Residual Network (core building block)
    VSN     — Variable Selection Network
    InterpretableMultiHeadAttention
    TemporalFusionTransformer
    QuantileLoss (pinball loss)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── GATED RESIDUAL NETWORK ───────────────────────────────────────────────────

class GRN(nn.Module):
    """
    GRN(a, c) = LayerNorm( GLU( W3·ELU(W1·a + W2·c) ) + skip(a) )

    Args:
        input_size:   size of primary input a
        hidden_size:  size of internal hidden layer
        output_size:  size of output (defaults to input_size)
        context_size: size of optional context c (None = no context)
        dropout:      dropout probability
    """

    def __init__(self, input_size, hidden_size, output_size=None,
                 context_size=None, dropout=0.05):
        super().__init__()
        output_size = output_size or input_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size * 2)  # ×2 for GLU
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_size)

        self.context_fc = (
            nn.Linear(context_size, hidden_size, bias=False)
            if context_size else None
        )

        self.skip = (
            nn.Linear(input_size, output_size, bias=False)
            if input_size != output_size else nn.Identity()
        )

    def forward(self, x, context=None):
        residual = self.skip(x)

        h = self.fc1(x)
        if context is not None and self.context_fc is not None:
            h = h + self.context_fc(context)
        h = F.elu(h)
        h = self.dropout(self.fc2(h))

        # GLU: split into two halves, gate with sigmoid
        h1, h2 = h.chunk(2, dim=-1)
        h = torch.sigmoid(h2) * h1

        return self.norm(h + residual)


# ─── VARIABLE SELECTION NETWORK ───────────────────────────────────────────────

class VSN(nn.Module):
    """
    Variable Selection Network.

    Projects each input feature to d_model, then learns soft attention weights
    over features conditioned on static context. Returns the weighted sum of
    per-variable GRN outputs plus the raw weights for interpretability.

    Args:
        num_inputs:   number of input features (each already projected to d_model)
        d_model:      model dimension
        context_size: size of static context vector (None = no conditioning)
        dropout:      dropout probability
    """

    def __init__(self, num_inputs, d_model, context_size=None, dropout=0.05):
        super().__init__()
        self.num_inputs = num_inputs
        self.d_model = d_model

        # Per-variable GRNs
        self.var_grns = nn.ModuleList([
            GRN(d_model, d_model, dropout=dropout)
            for _ in range(num_inputs)
        ])

        # Selection GRN: flattened input → num_inputs weights
        self.selection_grn = GRN(
            input_size=num_inputs * d_model,
            hidden_size=d_model,
            output_size=num_inputs,
            context_size=context_size,
            dropout=dropout,
        )

    def forward(self, x, context=None):
        """
        x:       (B, T, num_inputs * d_model) or (B, num_inputs * d_model)
        context: (B, context_size) or None
        Returns:
            out:     (B, T, d_model) or (B, d_model)
            weights: (B, T, num_inputs) or (B, num_inputs)  [for interpretability]
        """
        has_time = x.dim() == 3
        if has_time:
            B, T, _ = x.shape
            xi = x.view(B, T, self.num_inputs, self.d_model)
            # broadcast context across time for the selection GRN
            ctx = context.unsqueeze(1).expand(-1, T, -1) if context is not None else None
        else:
            B = x.shape[0]
            xi = x.view(B, self.num_inputs, self.d_model)
            ctx = context

        flat = xi.flatten(start_dim=-2)  # (B, [T,] num_inputs*d_model)
        weights = torch.softmax(self.selection_grn(flat, ctx), dim=-1)  # (B,[T,] num_inputs)

        if has_time:
            processed = torch.stack([
                self.var_grns[i](xi[:, :, i, :]) for i in range(self.num_inputs)
            ], dim=2)  # (B, T, num_inputs, d_model)
            out = (weights.unsqueeze(-1) * processed).sum(dim=2)  # (B, T, d_model)
        else:
            processed = torch.stack([
                self.var_grns[i](xi[:, i, :]) for i in range(self.num_inputs)
            ], dim=1)  # (B, num_inputs, d_model)
            out = (weights.unsqueeze(-1) * processed).sum(dim=1)  # (B, d_model)

        return out, weights


# ─── INTERPRETABLE MULTI-HEAD ATTENTION ───────────────────────────────────────

class InterpretableMultiHeadAttention(nn.Module):
    """
    TFT's interpretable attention: H query/key heads but only 1 shared value head.
    This keeps attention weights interpretable — each head attends to the same V.

    Args:
        d_model:  model dimension
        n_heads:  number of attention heads
        dropout:  dropout on attention weights
    """

    def __init__(self, d_model, n_heads, dropout=0.05):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, self.d_head)  # single shared value head
        self.W_o = nn.Linear(self.d_head, d_model)

        self.attn_drop = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

    def forward(self, q, k, v, mask=None):
        """
        q, k, v: (B, T, d_model)
        mask:    (T, T) causal mask (True = keep, False = mask out)
        Returns:
            out:        (B, T, d_model)
            attn_weights: (B, n_heads, T_q, T_k)  [for interpretability]
        """
        B, T_q, _ = q.shape
        T_k = k.shape[1]

        Q = self.W_q(q).view(B, T_q, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(k).view(B, T_k, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(v)  # (B, T_k, d_head) — shared across heads

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, T_q, T_k)

        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        # Average attention weights across heads (interpretable)
        attn_avg = attn.mean(dim=1)  # (B, T_q, T_k)

        # Apply to shared V: expand V for each head then contract
        V_exp = V.unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # (B, H, T_k, d_head)
        context = torch.matmul(attn, V_exp).mean(dim=1)  # (B, T_q, d_head)

        out = self.W_o(context)
        return out, attn_avg


# ─── TEMPORAL FUSION TRANSFORMER ──────────────────────────────────────────────

class TemporalFusionTransformer(nn.Module):
    """
    Full TFT for multi-horizon groundwater level forecasting with quantile output.

    Input categories:
        static_cat:   (B, n_static_cat)        integer district IDs
        static_cont:  (B, n_static_cont)        soil texture etc.
        past_cont:    (B, encoder_len, n_past)  observed history
        future_cont:  (B, decoder_len, n_future) known future (calendar)

    Output:
        (B, decoder_len, n_quantiles)   GWL predictions at each future step

    Args:
        n_static_cat:   number of static categorical features
        cat_vocab_sizes: list of vocabulary sizes for each categorical feature
        cat_embed_dim:  embedding dimension per categorical feature
        n_static_cont:  number of static continuous features
        n_past:         number of observed-past continuous features
        n_future:       number of known-future continuous features
        encoder_len:    past sequence length
        decoder_len:    future sequence length
        d_model:        model hidden dimension
        n_heads:        attention heads
        lstm_layers:    LSTM depth
        dropout:        dropout rate
        quantiles:      list of quantile levels e.g. [0.1, 0.5, 0.9]
    """

    def __init__(
        self,
        n_static_cat,
        cat_vocab_sizes,
        cat_embed_dim,
        n_static_cont,
        n_past,
        n_future,
        encoder_len,
        decoder_len,
        d_model=128,
        n_heads=4,
        lstm_layers=2,
        dropout=0.1,
        quantiles=None,
    ):
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)
        self.encoder_len = encoder_len
        self.decoder_len = decoder_len
        self.d_model = d_model

        # ── categorical embeddings ────────────────────────────────────────────
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size + 1, cat_embed_dim)
            for vocab_size in cat_vocab_sizes
        ])
        static_cat_dim = n_static_cat * cat_embed_dim

        # ── input projections (all features → d_model) ───────────────────────
        static_total_dim = static_cat_dim + n_static_cont
        self.static_proj = nn.Linear(static_total_dim, d_model)
        self.past_proj   = nn.Linear(n_past,   d_model)
        self.future_proj = nn.Linear(n_future, d_model)

        # ── feature counts for VSN ────────────────────────────────────────────
        n_static_features = n_static_cat + n_static_cont  # 1 cat treated as 1 after proj
        # We concat cat embed + cont → 1 projected vector before VSN, so treat as 1 feature
        # Actually: each raw feature gets its own d_model projection → VSN operates on those
        n_static_vsn = 1 + n_static_cont  # 1 for all categoricals combined + n_static_cont
        n_past_vsn   = n_past
        n_future_vsn = n_future

        self.n_static_vsn = n_static_vsn
        self.n_past_vsn   = n_past_vsn
        self.n_future_vsn = n_future_vsn

        # individual feature projections for VSN
        self.static_cat_proj  = nn.Linear(static_cat_dim, d_model)
        self.static_cont_projs = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_static_cont)])
        self.past_feat_projs   = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_past)])
        self.future_feat_projs = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_future)])

        # ── static covariate encoder (4 context vectors) ─────────────────────
        static_vsn_input_size = n_static_vsn * d_model
        self.static_vsn = VSN(n_static_vsn, d_model, dropout=dropout)

        # 4 context vectors from static embedding
        self.cs_grn = GRN(d_model, d_model, dropout=dropout)  # enrichment context
        self.ce_grn = GRN(d_model, d_model, dropout=dropout)  # encoder context (for VSN)
        self.ch_grn = GRN(d_model, d_model, dropout=dropout)  # LSTM initial h
        self.cc_grn = GRN(d_model, d_model, dropout=dropout)  # LSTM initial c

        # ── past / future VSN (conditioned on ce) ────────────────────────────
        self.past_vsn   = VSN(n_past_vsn,   d_model, context_size=d_model, dropout=dropout)
        self.future_vsn = VSN(n_future_vsn, d_model, context_size=d_model, dropout=dropout)

        # ── LSTM encoder + decoder ────────────────────────────────────────────
        self.lstm_encoder = nn.LSTM(
            d_model, d_model, num_layers=lstm_layers,
            batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.lstm_decoder = nn.LSTM(
            d_model, d_model, num_layers=lstm_layers,
            batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0
        )

        # project initial LSTM state (d_model → lstm_layers * d_model)
        self.lstm_layers = lstm_layers
        self.h0_proj = nn.Linear(d_model, lstm_layers * d_model)
        self.c0_proj = nn.Linear(d_model, lstm_layers * d_model)

        # skip connection after LSTM (gate lstm output vs input)
        self.post_lstm_gate = nn.Sequential(nn.Linear(d_model, d_model * 2))
        self.post_lstm_norm = nn.LayerNorm(d_model)

        # ── static enrichment ─────────────────────────────────────────────────
        self.static_enrichment_grn = GRN(d_model, d_model, context_size=d_model, dropout=dropout)

        # ── temporal self-attention ───────────────────────────────────────────
        self.attention = InterpretableMultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.attn_gate = nn.Sequential(nn.Linear(d_model, d_model * 2))
        self.attn_norm = nn.LayerNorm(d_model)

        # ── position-wise feed-forward (GRN) ─────────────────────────────────
        self.ff_grn = GRN(d_model, d_model, dropout=dropout)
        self.ff_gate = nn.Sequential(nn.Linear(d_model, d_model * 2))
        self.ff_norm = nn.LayerNorm(d_model)

        # ── output ────────────────────────────────────────────────────────────
        self.output_proj = nn.Linear(d_model, self.n_quantiles)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def _glu_gate(self, x, gate_layer, norm, residual):
        """Apply a GLU gate + add & norm skip."""
        g = gate_layer(x)
        g1, g2 = g.chunk(2, dim=-1)
        gated = torch.sigmoid(g2) * g1
        return norm(gated + residual)

    def _build_static_features(self, static_cat, static_cont):
        """
        Project each static feature individually to d_model for VSN input.
        Returns tensor (B, n_static_vsn * d_model)
        """
        B = static_cont.shape[0]

        # all categoricals → one embedding concat → project to d_model
        cat_embeds = [emb(static_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        cat_vec = self.static_cat_proj(torch.cat(cat_embeds, dim=-1))  # (B, d_model)

        # each static continuous feature → d_model
        cont_vecs = [proj(static_cont[:, i:i+1]) for i, proj in enumerate(self.static_cont_projs)]

        return torch.stack([cat_vec] + cont_vecs, dim=1)  # (B, n_static_vsn, d_model)

    def _build_temporal_features(self, x, projs):
        """
        Project each feature in x to d_model individually.
        x:     (B, T, n_features)
        projs: ModuleList of n_features Linear(1→d_model)
        Returns: (B, T, n_features, d_model)
        """
        return torch.stack([
            projs[i](x[:, :, i:i+1]) for i in range(x.shape[-1])
        ], dim=2)  # (B, T, n_features, d_model)

    def forward(self, static_cat, static_cont, past_cont, future_cont):
        """
        static_cat:   (B, n_static_cat)
        static_cont:  (B, n_static_cont)
        past_cont:    (B, encoder_len, n_past)
        future_cont:  (B, decoder_len, n_future)

        Returns:
            preds:        (B, decoder_len, n_quantiles)
            vsn_weights:  dict with 'static', 'past', 'future' importance weights
            attn_weights: (B, decoder_len, encoder_len + decoder_len)
        """
        B = static_cat.shape[0]

        # ── 1. Static encoder ─────────────────────────────────────────────────
        static_feats = self._build_static_features(static_cat, static_cont)
        static_flat  = static_feats.flatten(start_dim=1)  # (B, n_static_vsn * d_model)
        static_emb, static_vsn_w = self.static_vsn(static_flat)  # (B, d_model)

        cs = self.cs_grn(static_emb)  # enrichment context
        ce = self.ce_grn(static_emb)  # encoder VSN context
        ch = self.ch_grn(static_emb)  # LSTM initial hidden
        cc = self.cc_grn(static_emb)  # LSTM initial cell

        # ── 2. Past input VSN ─────────────────────────────────────────────────
        past_feats = self._build_temporal_features(past_cont, self.past_feat_projs)
        past_flat  = past_feats.flatten(start_dim=2)   # (B, T_enc, n_past * d_model)
        past_vsn_out, past_vsn_w = self.past_vsn(past_flat, ce)    # (B, T_enc, d_model)

        # ── 3. Future input VSN ───────────────────────────────────────────────
        fut_feats = self._build_temporal_features(future_cont, self.future_feat_projs)
        fut_flat  = fut_feats.flatten(start_dim=2)     # (B, T_dec, n_future * d_model)
        fut_vsn_out, fut_vsn_w = self.future_vsn(fut_flat, ce)     # (B, T_dec, d_model)

        # ── 4. LSTM encoder ───────────────────────────────────────────────────
        h0 = self.h0_proj(ch).view(B, self.lstm_layers, self.d_model).permute(1, 0, 2).contiguous()
        c0 = self.c0_proj(cc).view(B, self.lstm_layers, self.d_model).permute(1, 0, 2).contiguous()
        enc_out, (h_n, c_n) = self.lstm_encoder(past_vsn_out, (h0, c0))

        # ── 5. LSTM decoder ───────────────────────────────────────────────────
        dec_out, _ = self.lstm_decoder(fut_vsn_out, (h_n, c_n))

        # ── 6. Gate + skip after LSTM ────────────────────────────────────────
        combined_vsn = torch.cat([past_vsn_out, fut_vsn_out], dim=1)  # (B, T_enc+T_dec, d_model)
        combined_lstm = torch.cat([enc_out, dec_out], dim=1)           # (B, T_enc+T_dec, d_model)
        combined = self._glu_gate(combined_lstm, self.post_lstm_gate, self.post_lstm_norm, combined_vsn)

        # ── 7. Static enrichment ─────────────────────────────────────────────
        cs_exp = cs.unsqueeze(1).expand(-1, combined.shape[1], -1)
        enriched = self.static_enrichment_grn(combined, cs_exp)

        # ── 8. Temporal self-attention (decoder positions query encoder+decoder) ─
        T_total = self.encoder_len + self.decoder_len
        # causal-style: decoder positions can attend to all encoder + prior decoder
        attn_mask = torch.ones(T_total, T_total, dtype=torch.bool, device=enriched.device)
        attn_mask[self.encoder_len:, self.encoder_len:] = torch.tril(
            torch.ones(self.decoder_len, self.decoder_len, dtype=torch.bool, device=enriched.device)
        )

        attn_out, attn_w = self.attention(enriched, enriched, enriched, attn_mask)

        # gate + skip
        attn_out = self._glu_gate(attn_out, self.attn_gate, self.attn_norm, enriched)

        # ── 9. Position-wise FFN ─────────────────────────────────────────────
        ff_out   = self.ff_grn(attn_out)
        ff_out   = self._glu_gate(ff_out, self.ff_gate, self.ff_norm, attn_out)

        # ── 10. Output — only decoder positions ──────────────────────────────
        decoder_out = ff_out[:, self.encoder_len:, :]  # (B, T_dec, d_model)
        preds = self.output_proj(decoder_out)            # (B, T_dec, n_quantiles)

        vsn_weights = {
            "static":  static_vsn_w,
            "past":    past_vsn_w,
            "future":  fut_vsn_w,
        }
        attn_weights_dec = attn_w[:, self.encoder_len:, :]  # (B, T_dec, T_total)

        return preds, vsn_weights, attn_weights_dec


# ─── QUANTILE (PINBALL) LOSS ─────────────────────────────────────────────────

class QuantileLoss(nn.Module):
    """
    Pinball loss averaged over all quantiles and timesteps.

    For quantile q:  loss = q * max(y-ŷ, 0) + (1-q) * max(ŷ-y, 0)
    """

    def __init__(self, quantiles):
        super().__init__()
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))

    def forward(self, preds, targets):
        """
        preds:   (B, T, n_quantiles)
        targets: (B, T)
        """
        targets = targets.unsqueeze(-1)          # (B, T, 1)
        errors  = targets - preds                # (B, T, n_quantiles)
        q       = self.quantiles.view(1, 1, -1)  # broadcast
        loss    = torch.max(q * errors, (q - 1) * errors)
        return loss.mean()
