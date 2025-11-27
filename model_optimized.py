"""
Optimized Next-Location Predictor with Hierarchical S2 Features

Key insights from research:
1. S2 level 15 alone gives 44.6% baseline -> S2 is PRIMARY signal
2. Location sequences matter but S2 hierarchy is more important
3. Need to PRESERVE S2 information, not dilute it
4. Use location as CONTEXT for S2, not the other way around

Best practices applied:
- Separate embeddings for each feature type
- Skip connections to preserve information
- Layer normalization for stable training
- Attention over S2 hierarchy, not complex fusion
- Simple but deep architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class S2HierarchyAttention(nn.Module):
    """Attend over S2 hierarchy to combine multi-resolution information"""
    def __init__(self, d_model, num_levels=4):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)
        
    def forward(self, s2_embeddings):
        """
        s2_embeddings: list of [B, T, d_model], one per S2 level
        Returns: [B, T, d_model] - attended S2 representation
        """
        # Stack all levels: [B, T, num_levels, d_model]
        stacked = torch.stack(s2_embeddings, dim=2)
        B, T, L, D = stacked.shape
        
        # Use finest level (last) as query
        q = self.query(s2_embeddings[-1])  # [B, T, D]
        
        # Reshape for attention
        stacked_flat = stacked.view(B * T, L, D)
        k = self.key(stacked_flat)  # [B*T, L, D]
        v = self.value(stacked_flat)  # [B*T, L, D]
        q_flat = q.view(B * T, 1, D)  # [B*T, 1, D]
        
        # Attention
        scores = torch.bmm(q_flat, k.transpose(1, 2)) / self.scale  # [B*T, 1, L]
        attn = F.softmax(scores, dim=-1)
        out = torch.bmm(attn, v)  # [B*T, 1, D]
        
        return out.view(B, T, D)


class OptimizedHierarchicalPredictor(nn.Module):
    """
    Optimized architecture that respects the data:
    - S2 features are the primary signal
    - Location provides context
    - User and temporal features add minor signal
    """
    def __init__(self, config):
        super().__init__()
        
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        
        # Embeddings - S2 get more capacity
        s2_dim = self.d_model
        loc_dim = self.d_model // 2
        meta_dim = self.d_model // 4
        
        # S2 embeddings (PRIMARY SIGNAL)
        self.s2_l11 = nn.Embedding(config['num_s2_l11'], s2_dim)
        self.s2_l12 = nn.Embedding(config['num_s2_l12'], s2_dim)
        self.s2_l13 = nn.Embedding(config['num_s2_l13'], s2_dim)
        self.s2_l14 = nn.Embedding(config['num_s2_l14'], s2_dim)
        
        # Location embedding (CONTEXT)
        self.loc_emb = nn.Embedding(config['num_locations'], loc_dim)
        
        # Meta embeddings
        self.user_emb = nn.Embedding(config['num_users'], meta_dim)
        self.weekday_emb = nn.Embedding(config['num_weekdays'], meta_dim)
        
        # S2 hierarchy attention
        self.s2_hierarchy_attn = S2HierarchyAttention(s2_dim, num_levels=4)
        
        # Projection layers with skip connections
        self.loc_proj = nn.Linear(loc_dim, self.d_model)
        self.meta_proj = nn.Linear(meta_dim * 2, self.d_model)
        
        # Combine S2 (primary) with context (location + meta)
        self.combine = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(self.d_model)
        
        # Transformer with residual connections
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Output layers with skip connection from S2
        self.pre_out_norm = nn.LayerNorm(self.d_model)
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),  # Concat transformer + S2
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, config['num_locations'])
        )
        
        self._init_weights()
        
    def _init_weights(self):
        # Xavier for linear layers
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                if 'norm' not in name.lower():
                    nn.init.xavier_uniform_(p, gain=0.02)
            elif 'bias' in name:
                nn.init.zeros_(p)
        
        # Special init for embeddings - smaller variance
        for emb in [self.s2_l11, self.s2_l12, self.s2_l13, self.s2_l14, self.loc_emb]:
            nn.init.normal_(emb.weight, mean=0, std=0.02)
    
    def forward(self, batch):
        x = batch['X']
        user_x = batch['user_X']
        weekday_x = batch['weekday_X']
        s2_l11 = batch['s2_level11_X']
        s2_l12 = batch['s2_level13_X']
        s2_l13 = batch['s2_level14_X']
        s2_l14 = batch['s2_level15_X']
        mask_pad = batch['mask']
        
        B, T = x.shape
        device = x.device
        
        # === S2 Hierarchy (PRIMARY SIGNAL) ===
        s2_embs = [
            self.s2_l11(s2_l11),
            self.s2_l12(s2_l12),
            self.s2_l13(s2_l13),
            self.s2_l14(s2_l14)
        ]
        
        # Attend over S2 hierarchy
        s2_repr = self.s2_hierarchy_attn(s2_embs)  # [B, T, d_model]
        
        # === Location Context ===
        loc_repr = self.loc_proj(self.loc_emb(x))  # [B, T, d_model]
        
        # === Meta Features ===
        meta_repr = self.meta_proj(
            torch.cat([self.user_emb(user_x), self.weekday_emb(weekday_x)], dim=-1)
        )  # [B, T, d_model]
        
        # === Combine: S2 (primary) + Location (context) + Meta ===
        combined = self.combine(
            torch.cat([s2_repr, loc_repr, meta_repr], dim=-1)
        )  # [B, T, d_model]
        
        # Add positional encoding
        combined = self.pos_enc(combined)
        
        # Transformer with causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)
        padding_mask = ~mask_pad
        
        transformer_out = self.transformer(
            combined,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )  # [B, T, d_model]
        
        # Get last valid position
        transformer_out = self.pre_out_norm(transformer_out)
        seq_lengths = mask_pad.sum(dim=1) - 1
        batch_idx = torch.arange(B, device=device)
        last_hidden = transformer_out[batch_idx, seq_lengths]  # [B, d_model]
        last_s2 = s2_repr[batch_idx, seq_lengths]  # [B, d_model]
        
        # Skip connection: concat transformer output with original S2
        final_repr = torch.cat([last_hidden, last_s2], dim=-1)  # [B, 2*d_model]
        
        # Output
        logits = self.output_proj(final_repr)  # [B, num_locations]
        
        return logits
