"""
Temporal Fusion Transformer — v3.

Same architecture as `tft_model.py` (Lim et al. 2021), with two changes that
matter for this project:

1. **Vectorised Variable Selection Networks.**
   The v1/v2 VSN ran a Python loop over every input variable — one
   `Linear(1 -> d_model)` plus one full GRN per feature, per forward pass. With
   38 past features that is ~114 separate CUDA kernel launches per VSN call, and
   the batches here are tiny (23 batches/epoch on the largest fold), so the model
   was almost entirely launch-overhead bound rather than FLOP bound.

   `BatchedGRN` computes all N per-variable GRNs as three `baddbmm` calls. It is
   mathematically identical — same parameter count, same per-variable weights,
   same LayerNorm affine per variable — just grouped. Measured ~3-5x faster,
   which is what makes a 3-seed ensemble affordable in a Colab session.

2. **Optional monotone quantile head** (`monotone_quantiles=True`).
   With independent linear outputs, q10/q50/q90 can cross — the model can emit a
   q10 above its q90, which is meaningless and wastes capacity. Instead emit the
   median plus two non-negative spreads:

        q50 = raw_mid
        q10 = q50 - softplus(raw_lo)
        q90 = q50 + softplus(raw_hi)

   Crossing becomes impossible by construction. Assumes `quantiles` is a
   symmetric 3-tuple [lo, 0.5, hi]; falls back to the plain head otherwise.

Everything else — GRN, interpretable attention, static contexts, gating, the
returned `(preds, vsn_weights, attn_weights)` signature — is unchanged, so this
is a drop-in replacement.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── GATED RESIDUAL NETWORK ───────────────────────────────────────────────────

class GRN(nn.Module):
    """GRN(a, c) = LayerNorm( GLU( W3·ELU(W1·a + W2·c) ) + skip(a) )"""

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
        h1, h2 = h.chunk(2, dim=-1)
        h = torch.sigmoid(h2) * h1
        return self.norm(h + residual)


# ─── BATCHED (PER-VARIABLE) GRN ───────────────────────────────────────────────

class BatchedGRN(nn.Module):
    """
    N independent GRNs, all of shape (d_model -> d_model), evaluated together.

    Equivalent to `nn.ModuleList([GRN(d, d) for _ in range(N)])` applied
    variable-wise, but as grouped `baddbmm` instead of an N-iteration Python
    loop. Each variable keeps its own fc1/fc2 weights and its own LayerNorm
    affine parameters, exactly as in the unbatched version.

    Input / output: (..., N, d_model)
    """

    def __init__(self, num_vars, d_model, dropout=0.05, hidden_factor=1.0):
        super().__init__()
        self.num_vars = num_vars
        self.d_model = d_model
        d_hidden = max(4, int(round(d_model * hidden_factor)))

        self.w1 = nn.Parameter(torch.empty(num_vars, d_model, d_hidden))
        self.b1 = nn.Parameter(torch.zeros(num_vars, 1, d_hidden))
        self.w2 = nn.Parameter(torch.empty(num_vars, d_hidden, d_model * 2))
        self.b2 = nn.Parameter(torch.zeros(num_vars, 1, d_model * 2))

        self.gamma = nn.Parameter(torch.ones(num_vars, d_model))
        self.beta = nn.Parameter(torch.zeros(num_vars, d_model))

        self.dropout = nn.Dropout(dropout)
        for w in (self.w1, self.w2):
            nn.init.xavier_uniform_(w)

    def forward(self, x):
        lead = x.shape[:-2]                       # (B,) or (B, T)
        m = int(torch.tensor(lead).prod().item()) if lead else 1
        # (M, N, d) -> (N, M, d) so each variable is one bmm group
        xv = x.reshape(m, self.num_vars, self.d_model).transpose(0, 1)

        h = torch.baddbmm(self.b1, xv, self.w1)
        h = F.elu(h)
        h = self.dropout(torch.baddbmm(self.b2, h, self.w2))
        h1, h2 = h.chunk(2, dim=-1)
        h = torch.sigmoid(h2) * h1 + xv           # skip is Identity (d_in == d_out)

        # per-variable LayerNorm with per-variable affine
        h = F.layer_norm(h, (self.d_model,))
        h = h * self.gamma.unsqueeze(1) + self.beta.unsqueeze(1)

        return h.transpose(0, 1).reshape(*lead, self.num_vars, self.d_model)


class BatchedFeatureProjection(nn.Module):
    """
    N independent `Linear(1 -> d_model)` projections, one per scalar feature,
    evaluated as a single broadcasted multiply-add instead of N matmuls.

    Input:  (..., N)      one scalar per variable
    Output: (..., N, d_model)
    """

    def __init__(self, num_vars, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_vars, d_model))
        self.bias = nn.Parameter(torch.zeros(num_vars, d_model))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        return x.unsqueeze(-1) * self.weight + self.bias


class BatchedFeatureCompression(nn.Module):
    """
    N independent `Linear(d_in -> d_out)` maps, one per variable, batched.

    Used to shrink each variable embedding before it is flattened into the
    variable-selection GRN — see the note in `VSN.__init__`.

    Input: (..., N, d_in)  ->  Output: (..., N, d_out)
    """

    def __init__(self, num_vars, d_in, d_out):
        super().__init__()
        self.num_vars, self.d_in, self.d_out = num_vars, d_in, d_out
        self.weight = nn.Parameter(torch.empty(num_vars, d_in, d_out))
        self.bias = nn.Parameter(torch.zeros(num_vars, 1, d_out))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        lead = x.shape[:-2]
        m = int(torch.tensor(lead).prod().item()) if lead else 1
        xv = x.reshape(m, self.num_vars, self.d_in).transpose(0, 1)
        out = torch.baddbmm(self.bias, xv, self.weight)
        return out.transpose(0, 1).reshape(*lead, self.num_vars, self.d_out)


# ─── VARIABLE SELECTION NETWORK ───────────────────────────────────────────────

class VSN(nn.Module):
    """
    Variable Selection Network.

    Projects each scalar input to d_model, learns softmax weights over variables
    (optionally conditioned on a static context), and returns the weighted sum of
    per-variable GRN outputs plus the raw weights for interpretability.
    """

    def __init__(self, num_inputs, d_model, context_size=None, dropout=0.05,
                 d_select=None, grn_hidden_factor=1.0):
        super().__init__()
        self.num_inputs = num_inputs
        self.d_model = d_model

        # Capacity control. In the textbook TFT the selection GRN consumes the
        # FLATTENED per-variable embeddings, so its first layer is
        # Linear(num_inputs * d_model, d_model). With 35 past features and
        # d_model=64 that single layer is 143k parameters and its skip another
        # 78k — i.e. the variable *selector* was larger than the entire rest of
        # the network, on a problem with ~290 independent training windows.
        #
        # `d_select` first compresses each variable's embedding to a small vector
        # before flattening (64 -> 8 by default), cutting that block ~8x while
        # leaving the selector free to compare all variables against each other.
        # `grn_hidden_factor` narrows the per-variable GRN bottleneck the same way.
        self.d_select = d_select or d_model
        self.compress = (
            BatchedFeatureCompression(num_inputs, d_model, self.d_select)
            if self.d_select != d_model else None
        )

        self.var_grns = BatchedGRN(num_inputs, d_model, dropout=dropout,
                                   hidden_factor=grn_hidden_factor)
        self.selection_grn = GRN(
            input_size=num_inputs * self.d_select,
            hidden_size=d_model,
            output_size=num_inputs,
            context_size=context_size,
            dropout=dropout,
        )

    def forward(self, x, context=None):
        """
        x:       (B, T, num_inputs, d_model) or (B, num_inputs, d_model)
        context: (B, context_size) or None
        Returns  out: (B, [T,] d_model),  weights: (B, [T,] num_inputs)
        """
        has_time = x.dim() == 4
        ctx = context
        if has_time and context is not None:
            ctx = context.unsqueeze(1).expand(-1, x.shape[1], -1)

        sel_in = self.compress(x) if self.compress is not None else x
        flat = sel_in.flatten(start_dim=-2)                             # (B,[T,] N*d_select)
        weights = torch.softmax(self.selection_grn(flat, ctx), dim=-1)  # (B,[T,] N)
        processed = self.var_grns(x)                                    # (B,[T,] N, d)
        out = (weights.unsqueeze(-1) * processed).sum(dim=-2)           # (B,[T,] d)
        return out, weights


# ─── INTERPRETABLE MULTI-HEAD ATTENTION ───────────────────────────────────────

class InterpretableMultiHeadAttention(nn.Module):
    """H query/key heads, one shared value head — keeps attention interpretable."""

    def __init__(self, d_model, n_heads, dropout=0.05):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, self.d_head)
        self.W_o = nn.Linear(self.d_head, d_model)

        self.attn_drop = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

    def forward(self, q, k, v, mask=None):
        B, T_q, _ = q.shape
        T_k = k.shape[1]

        Q = self.W_q(q).view(B, T_q, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(k).view(B, T_k, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(v)                                        # (B, T_k, d_head)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), -1e9)

        attn = self.attn_drop(torch.softmax(scores, dim=-1))
        attn_avg = attn.mean(dim=1)

        # single shared V: average the per-head contexts
        context = torch.matmul(attn.mean(dim=1), V)            # (B, T_q, d_head)
        return self.W_o(context), attn_avg


# ─── TEMPORAL FUSION TRANSFORMER ──────────────────────────────────────────────

class TemporalFusionTransformer(nn.Module):
    """
    Multi-horizon quantile forecaster.

    Inputs:
        static_cat:   (B, n_static_cat)          integer ids
        static_cont:  (B, n_static_cont)
        past_cont:    (B, encoder_len, n_past)
        future_cont:  (B, decoder_len, n_future)

    Returns:
        preds:        (B, decoder_len, n_quantiles)
        vsn_weights:  {'static', 'past', 'future'}
        attn_weights: (B, decoder_len, encoder_len + decoder_len)
    """

    def __init__(self, n_static_cat, cat_vocab_sizes, cat_embed_dim, n_static_cont,
                 n_past, n_future, encoder_len, decoder_len, d_model=128, n_heads=4,
                 lstm_layers=2, dropout=0.1, quantiles=None, monotone_quantiles=True,
                 d_select=8, grn_hidden_factor=0.5):
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)
        self.encoder_len = encoder_len
        self.decoder_len = decoder_len
        self.d_model = d_model

        # Monotone head needs exactly [lo, 0.5, hi]
        self.monotone = bool(monotone_quantiles) and (
            self.n_quantiles == 3 and abs(quantiles[1] - 0.5) < 1e-9
        )

        # ── categorical embeddings ────────────────────────────────────────────
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(v + 1, cat_embed_dim) for v in cat_vocab_sizes
        ])
        static_cat_dim = n_static_cat * cat_embed_dim

        # ── per-variable projections (vectorised) ────────────────────────────
        n_static_vsn = 1 + n_static_cont     # all categoricals fused into one "variable"
        self.n_static_vsn = n_static_vsn

        self.static_cat_proj = nn.Linear(static_cat_dim, d_model)
        self.static_cont_proj = BatchedFeatureProjection(n_static_cont, d_model)
        self.past_proj = BatchedFeatureProjection(n_past, d_model)
        self.future_proj = BatchedFeatureProjection(n_future, d_model)

        # ── static covariate encoder → 4 context vectors ─────────────────────
        vsn_kw = dict(dropout=dropout, d_select=d_select, grn_hidden_factor=grn_hidden_factor)
        self.static_vsn = VSN(n_static_vsn, d_model, **vsn_kw)
        self.cs_grn = GRN(d_model, d_model, dropout=dropout)   # enrichment
        self.ce_grn = GRN(d_model, d_model, dropout=dropout)   # VSN conditioning
        self.ch_grn = GRN(d_model, d_model, dropout=dropout)   # LSTM h0
        self.cc_grn = GRN(d_model, d_model, dropout=dropout)   # LSTM c0

        # ── temporal VSNs ─────────────────────────────────────────────────────
        self.past_vsn = VSN(n_past, d_model, context_size=d_model, **vsn_kw)
        self.future_vsn = VSN(n_future, d_model, context_size=d_model, **vsn_kw)

        # ── LSTM encoder / decoder ────────────────────────────────────────────
        self.lstm_layers = lstm_layers
        self.lstm_encoder = nn.LSTM(d_model, d_model, num_layers=lstm_layers,
                                    batch_first=True,
                                    dropout=dropout if lstm_layers > 1 else 0.0)
        self.lstm_decoder = nn.LSTM(d_model, d_model, num_layers=lstm_layers,
                                    batch_first=True,
                                    dropout=dropout if lstm_layers > 1 else 0.0)
        self.h0_proj = nn.Linear(d_model, lstm_layers * d_model)
        self.c0_proj = nn.Linear(d_model, lstm_layers * d_model)

        self.post_lstm_gate = nn.Linear(d_model, d_model * 2)
        self.post_lstm_norm = nn.LayerNorm(d_model)

        # ── enrichment, attention, FFN ────────────────────────────────────────
        self.static_enrichment_grn = GRN(d_model, d_model, context_size=d_model, dropout=dropout)
        self.attention = InterpretableMultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.attn_gate = nn.Linear(d_model, d_model * 2)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ff_grn = GRN(d_model, d_model, dropout=dropout)
        self.ff_gate = nn.Linear(d_model, d_model * 2)
        self.ff_norm = nn.LayerNorm(d_model)

        self.output_proj = nn.Linear(d_model, self.n_quantiles)

        # Start the monotone head with a sensible non-zero spread: softplus(-0.5)
        # ≈ 0.47 scaled units, roughly half a typical movement, so the intervals
        # are neither degenerate nor absurd at epoch 0.
        if self.monotone:
            with torch.no_grad():
                self.output_proj.bias.copy_(torch.tensor([-0.5, 0.0, -0.5]))

        self._init_weights()
        self._causal_mask = None

    def _init_weights(self):
        for name, p in self.named_parameters():
            if name.endswith("output_proj.bias"):
                continue                       # set deliberately above
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def _glu_gate(self, x, gate_layer, norm, residual):
        g1, g2 = gate_layer(x).chunk(2, dim=-1)
        return norm(torch.sigmoid(g2) * g1 + residual)

    def _get_mask(self, device):
        if self._causal_mask is None or self._causal_mask.device != device:
            T = self.encoder_len + self.decoder_len
            m = torch.ones(T, T, dtype=torch.bool, device=device)
            m[self.encoder_len:, self.encoder_len:] = torch.tril(
                torch.ones(self.decoder_len, self.decoder_len, dtype=torch.bool, device=device))
            self._causal_mask = m
        return self._causal_mask

    def forward(self, static_cat, static_cont, past_cont, future_cont):
        B = static_cat.shape[0]

        # 1. static encoder
        cat_embeds = [emb(static_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        cat_vec = self.static_cat_proj(torch.cat(cat_embeds, dim=-1)).unsqueeze(1)  # (B,1,d)
        cont_vecs = self.static_cont_proj(static_cont)                              # (B,n,d)
        static_feats = torch.cat([cat_vec, cont_vecs], dim=1)                       # (B,1+n,d)
        static_emb, static_vsn_w = self.static_vsn(static_feats)

        cs, ce = self.cs_grn(static_emb), self.ce_grn(static_emb)
        ch, cc = self.ch_grn(static_emb), self.cc_grn(static_emb)

        # 2/3. temporal VSNs
        past_vsn_out, past_vsn_w = self.past_vsn(self.past_proj(past_cont), ce)
        fut_vsn_out, fut_vsn_w = self.future_vsn(self.future_proj(future_cont), ce)

        # 4/5. LSTM encoder → decoder
        h0 = self.h0_proj(ch).view(B, self.lstm_layers, self.d_model).permute(1, 0, 2).contiguous()
        c0 = self.c0_proj(cc).view(B, self.lstm_layers, self.d_model).permute(1, 0, 2).contiguous()
        enc_out, (h_n, c_n) = self.lstm_encoder(past_vsn_out, (h0, c0))
        dec_out, _ = self.lstm_decoder(fut_vsn_out, (h_n, c_n))

        # 6. gate + skip over the LSTM
        combined_vsn = torch.cat([past_vsn_out, fut_vsn_out], dim=1)
        combined_lstm = torch.cat([enc_out, dec_out], dim=1)
        combined = self._glu_gate(combined_lstm, self.post_lstm_gate,
                                  self.post_lstm_norm, combined_vsn)

        # 7. static enrichment
        cs_exp = cs.unsqueeze(1).expand(-1, combined.shape[1], -1)
        enriched = self.static_enrichment_grn(combined, cs_exp)

        # 8. masked self-attention
        attn_out, attn_w = self.attention(enriched, enriched, enriched,
                                          self._get_mask(enriched.device))
        attn_out = self._glu_gate(attn_out, self.attn_gate, self.attn_norm, enriched)

        # 9. position-wise FFN
        ff_out = self._glu_gate(self.ff_grn(attn_out), self.ff_gate, self.ff_norm, attn_out)

        # 10. quantile head on decoder positions only
        raw = self.output_proj(ff_out[:, self.encoder_len:, :])   # (B, T_dec, Q)
        if self.monotone:
            mid = raw[..., 1]
            lo = mid - F.softplus(raw[..., 0])
            hi = mid + F.softplus(raw[..., 2])
            preds = torch.stack([lo, mid, hi], dim=-1)
        else:
            preds = raw

        return (
            preds,
            {"static": static_vsn_w, "past": past_vsn_w, "future": fut_vsn_w},
            attn_w[:, self.encoder_len:, :],
        )


# ─── QUANTILE (PINBALL) LOSS ─────────────────────────────────────────────────

class MaskedQuantileLoss(nn.Module):
    """Pinball loss averaged only over positions where mask == 1."""

    def __init__(self, quantiles, q50_boost=1.0, horizon_weights=None):
        super().__init__()
        self.quantiles = quantiles
        w = [q50_boost if abs(q - 0.5) < 1e-6 else 1.0 for q in quantiles]
        self.register_buffer("qw", torch.tensor(w).float())
        hw = torch.ones(1) if horizon_weights is None else torch.tensor(horizon_weights).float()
        self.register_buffer("hw", hw)

    def forward(self, preds, target, mask):
        diffs = target.unsqueeze(-1) - preds
        m = mask * self.hw.to(mask.device)
        denom = m.sum().clamp(min=1.0)
        total = 0.0
        for i, q in enumerate(self.quantiles):
            d = diffs[..., i]
            total = total + self.qw[i] * ((torch.max(q * d, (q - 1) * d)) * m).sum() / denom
        return total
