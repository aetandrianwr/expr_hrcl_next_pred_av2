"""
FOCUSED APPROACH: S2 Level 15 is the primary signal (44.6% baseline)
Let's not overcomplicate - just use it well
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocusedS2Predictor(nn.Module):
    """
    Key insight: S2 L15 alone = 44.6%
    Strategy: Use S2 L15 as primary, location as secondary, add minimal hierarchy
    """
    def __init__(self, config):
        super().__init__()
        
        d = config['d_model']
        
        # Primary: S2 Level 15 (most informative)
        self.s2_l15_emb = nn.Embedding(config['num_s2_l14'], d)
        
        # Secondary: Location  
        self.loc_emb = nn.Embedding(config['num_locations'], d // 2)
        
        # Hierarchical S2 context (coarser levels)
        self.s2_l14_emb = nn.Embedding(config['num_s2_l13'], d // 2)
        self.s2_l13_emb = nn.Embedding(config['num_s2_l12'], d // 4)
        self.s2_l12_emb = nn.Embedding(config['num_s2_l11'], d // 4)
        
        # Minimal metadata
        self.user_emb = nn.Embedding(config['num_users'], d // 8)
        self.weekday_emb = nn.Embedding(config['num_weekdays'], d // 8)
        
        # Total input size
        input_size = d + d // 2 + d // 2 + d // 4 + d // 4 + d // 8 + d // 8
        # = d + d//2 + d//2 + d//4 + d//4 + d//8 + d//8
        # = d * (1 + 0.5 + 0.5 + 0.25 + 0.25 + 0.125 + 0.125) = d * 2.75
        
        # Efficient LSTM (2 layers) - use d not d*2 for hidden
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=d,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        
        # Output with skip connection from S2 L15
        self.norm = nn.LayerNorm(d + d)  # LSTM hidden + S2 L15
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(d + d, config['num_locations'])
        
        self._init_weights()
    
    def _init_weights(self):
        # Careful initialization
        for name, p in self.named_parameters():
            if 'weight_ih' in name:  # LSTM input weights
                nn.init.xavier_uniform_(p)
            elif 'weight_hh' in name:  # LSTM hidden weights
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
            elif 'weight' in name and p.dim() > 1:  # Other weights
                nn.init.xavier_uniform_(p, gain=0.01)
    
    def forward(self, batch):
        x = batch['X']
        user_x = batch['user_X']
        weekday_x = batch['weekday_X']
        s2_l12 = batch['s2_level11_X']
        s2_l13 = batch['s2_level13_X']
        s2_l14 = batch['s2_level14_X']
        s2_l15 = batch['s2_level15_X']  # PRIMARY
        mask_pad = batch['mask']
        
        B, T = x.shape
        
        # Embed all features
        loc = self.loc_emb(x)
        s15 = self.s2_l15_emb(s2_l15)  # PRIMARY
        s14 = self.s2_l14_emb(s2_l14)
        s13 = self.s2_l13_emb(s2_l13)
        s12 = self.s2_l12_emb(s2_l12)
        user = self.user_emb(user_x)
        weekday = self.weekday_emb(weekday_x)
        
        # Concat: S2 L15 first (most important)
        features = torch.cat([s15, loc, s14, s13, s12, user, weekday], dim=-1)
        
        # LSTM
        lengths = mask_pad.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            features, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)  # hidden: [2, B, d*2]
        
        # Use last layer + skip connection from S2 L15 at last position
        lstm_out = hidden[-1]  # [B, d]
        last_s15 = s15[torch.arange(B), lengths - 1]  # [B, d]
        
        # Skip connection: preserve S2 L15 signal
        combined = torch.cat([lstm_out, last_s15], dim=-1)  # [B, d+d]
        combined = self.norm(combined)
        combined = self.dropout(combined)
        
        logits = self.classifier(combined)
        
        return logits
