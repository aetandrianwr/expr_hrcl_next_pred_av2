"""
SIMPLE AND EFFECTIVE: Let S2 speak for itself

Key insight: S2 level 15 baseline = 44.6%
Our complex models = 24-35%

Problem: We're destroying information with complex architectures
Solution: Minimal processing, preserve S2 signal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleEffectivePredictor(nn.Module):
    """
    Ultra-simple: Just embed, attend, predict
    No complex hierarchies, just let the data speak
    """
    def __init__(self, config):
        super().__init__()
        
        d = config['d_model']
        
        # Simple embeddings
        self.loc_emb = nn.Embedding(config['num_locations'], d)
        self.s2_l11_emb = nn.Embedding(config['num_s2_l11'], d)
        self.s2_l12_emb = nn.Embedding(config['num_s2_l12'], d)
        self.s2_l13_emb = nn.Embedding(config['num_s2_l13'], d)
        self.s2_l14_emb = nn.Embedding(config['num_s2_l14'], d)
        
        # Tiny context embeddings
        self.user_emb = nn.Embedding(config['num_users'], d // 4)
        self.weekday_emb = nn.Embedding(config['num_weekdays'], d // 4)
        
        # Simple GRU for temporal modeling (more efficient than Transformer)
        self.gru = nn.GRU(
            input_size=d * 5 + d // 2,  # 5 S2 levels + loc + user + weekday
            hidden_size=d * 2,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        
        # Direct prediction
        self.classifier = nn.Sequential(
            nn.LayerNorm(d * 2),
            nn.Linear(d * 2, d * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d * 2, config['num_locations'])
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                if 'gru' in name:
                    nn.init.orthogonal_(p)
                else:
                    nn.init.xavier_uniform_(p, gain=0.01)
            elif 'bias' in name:
                nn.init.zeros_(p)
    
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
        
        # Embed everything
        loc = self.loc_emb(x)
        s11 = self.s2_l11_emb(s2_l11)
        s12 = self.s2_l12_emb(s2_l12)
        s13 = self.s2_l13_emb(s2_l13)
        s14 = self.s2_l14_emb(s2_l14)
        user = self.user_emb(user_x)
        weekday = self.weekday_emb(weekday_x)
        
        # Concat all features
        features = torch.cat([loc, s11, s12, s13, s14, user, weekday], dim=-1)
        
        # Pack for GRU (handle variable lengths)
        lengths = mask_pad.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            features, lengths, batch_first=True, enforce_sorted=False
        )
        
        # GRU
        _, hidden = self.gru(packed)  # hidden: [num_layers, B, hidden_size]
        
        # Use last layer's hidden state
        final_hidden = hidden[-1]  # [B, hidden_size]
        
        # Predict
        logits = self.classifier(final_hidden)
        
        return logits
