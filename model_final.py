"""
FINAL MODEL: Simplified but robust architecture
Based on proven techniques but with numerical stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class FinalModel(nn.Module):
    """
    Robust model with proven components:
    1. Proper embeddings for all features
    2. S2 hierarchy as multi-resolution context
    3. User and time modeling
    4. Standard Transformer (stable)
    5. Multi-head attention over features
    """
    def __init__(self, config):
        super().__init__()
        
        d = config['d_model']
        num_locs = config['num_locations']
        
        # Feature embeddings
        self.loc_emb = nn.Embedding(num_locs, d)
        self.user_emb = nn.Embedding(config['num_users'], d)
        
        # S2 multi-resolution
        self.s2_l11 = nn.Embedding(config['num_s2_l11'], d // 4)
        self.s2_l12 = nn.Embedding(config['num_s2_l12'], d // 4)
        self.s2_l13 = nn.Embedding(config['num_s2_l13'], d // 4)
        self.s2_l14 = nn.Embedding(config['num_s2_l14'], d // 4)
        
        # Cyclical time
        self.weekday_sin = nn.Embedding(8, d // 8)
        self.weekday_cos = nn.Embedding(8, d // 8)
        
        # Project S2
        self.s2_proj = nn.Linear(d, d)
        
        # Combine all features
        self.feature_combine = nn.Sequential(
            nn.Linear(d * 2 + d // 4, d),  # loc + user + time
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(d)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=8,
            dim_feedforward=d * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # User-aware output
        self.output = nn.Sequential(
            nn.Linear(d * 2, d * 2),  # hidden + user
            nn.LayerNorm(d * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d * 2, num_locs)
        )
        
        self._init()
    
    def _init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.01)
    
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
        
        # Embeddings
        loc = self.loc_emb(x)
        user = self.user_emb(user_x)
        
        # S2 hierarchy
        s2 = torch.cat([
            self.s2_l11(s2_l11),
            self.s2_l12(s2_l12),
            self.s2_l13(s2_l13),
            self.s2_l14(s2_l14)
        ], dim=-1)
        s2 = self.s2_proj(s2)
        
        # Cyclical time
        time = torch.cat([
            self.weekday_sin(weekday_x),
            self.weekday_cos(weekday_x)
        ], dim=-1)
        
        # Combine
        features = self.feature_combine(torch.cat([loc, user, time], dim=-1))
        features = features + s2  # Add S2 as residual
        features = self.pos_enc(features)
        
        # Transformer
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)
        padding_mask = ~mask_pad
        
        hidden = self.transformer(
            features,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )
        
        # Get last position
        lengths = mask_pad.sum(dim=1) - 1
        batch_idx = torch.arange(B, device=device)
        last_hidden = hidden[batch_idx, lengths]
        last_user = user[batch_idx, 0]
        
        # Output
        logits = self.output(torch.cat([last_hidden, last_user], dim=-1))
        
        return logits
