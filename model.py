"""
USER-AWARE HIERARCHICAL TRANSFORMER
Target: ≥50% Acc@1 on test with <500K params
"""

import torch
import torch.nn as nn
import math


class UserAwareTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        d = cfg['d_model']
        
        # Core embeddings (USER is critical!)
        self.user_emb = nn.Embedding(cfg['n_users'], d)
        self.loc_emb = nn.Embedding(cfg['n_locations'], d)
        
        # S2 spatial (multi-resolution)
        self.s2_l11 = nn.Embedding(cfg['n_s2_l11'], d // 8)
        self.s2_l13 = nn.Embedding(cfg['n_s2_l13'], d // 8)
        self.s2_l14 = nn.Embedding(cfg['n_s2_l14'], d // 8)
        self.s2_l15 = nn.Embedding(cfg['n_s2_l15'], d // 8)
        self.s2_proj = nn.Linear(d // 2, d // 2)
        
        # Time
        self.weekday_emb = nn.Embedding(cfg['n_weekdays'], d // 4)
        
        # Combine features
        self.input_proj = nn.Sequential(
            nn.Linear(int(d * 2.75), d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(cfg['dropout'])
        )
        
        # Positional encoding
        self.register_buffer('pe', self._create_pe(cfg['max_pe'], d))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg['n_heads'],
            dim_feedforward=d * 4,
            dropout=cfg['dropout'],
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg['n_layers'])
        
        # Prediction head (user-aware)
        self.predictor = nn.Sequential(
            nn.Linear(d * 2, d * 2),
            nn.LayerNorm(d * 2),
            nn.GELU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(d * 2, cfg['n_locations'])
        )
        
        self._init_weights()
    
    def _create_pe(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.02 if 'emb' not in name else 1.0)
            elif 'bias' in name:
                nn.init.zeros_(p)
    
    def forward(self, batch):
        x = batch['X']
        user_x = batch['user_X']
        weekday_x = batch['weekday_X']
        s2_l11 = batch['s2_level11_X']
        s2_l13 = batch['s2_level13_X']
        s2_l14 = batch['s2_level14_X']
        s2_l15 = batch['s2_level15_X']
        mask_pad = batch['mask']
        
        B, T = x.shape
        device = x.device
        
        # Embeddings
        loc = self.loc_emb(x)
        user = self.user_emb(user_x)
        
        # S2
        s2 = torch.cat([
            self.s2_l11(s2_l11),
            self.s2_l13(s2_l13),
            self.s2_l14(s2_l14),
            self.s2_l15(s2_l15)
        ], dim=-1)
        s2 = self.s2_proj(s2)
        
        # Time
        weekday = self.weekday_emb(weekday_x)
        
        # Combine
        combined = torch.cat([loc, user, s2, weekday], dim=-1)
        features = self.input_proj(combined)
        features = features + self.pe[:T, :].unsqueeze(0)
        
        # Transformer
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)
        padding_mask = ~mask_pad
        
        hidden = self.transformer(features, mask=causal_mask, src_key_padding_mask=padding_mask)
        
        # Last position
        lengths = mask_pad.sum(dim=1) - 1
        batch_idx = torch.arange(B, device=device)
        last_hidden = hidden[batch_idx, lengths]
        last_user = user[batch_idx, 0]
        
        # Predict
        logits = self.predictor(torch.cat([last_hidden, last_user], dim=-1))
        
        return logits
