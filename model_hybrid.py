"""
HYBRID APPROACH: Combine neural network with explicit pattern memory

Key idea: Neural networks should learn which patterns to attend to,
not try to memorize all patterns themselves.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridPatternPredictor(nn.Module):
    """
    Uses neural network to gate between different prediction strategies:
    1. Location transition patterns
    2. S2 cell patterns
    3. User patterns
    4. Sequence patterns
    """
    def __init__(self, config):
        super().__init__()
        
        d = config['d_model']
        num_locs = config['num_locations']
        
        # Embeddings
        self.loc_emb = nn.Embedding(num_locs, d)
        self.user_emb = nn.Embedding(config['num_users'], d)
        self.s2_l14_emb = nn.Embedding(config['num_s2_l14'], d)
        self.s2_l13_emb = nn.Embedding(config['num_s2_l13'], d // 2)
        self.weekday_emb = nn.Embedding(config['num_weekdays'], d // 4)
        
        # Pattern-specific predictors
        # 1. Location transition predictor
        self.transition_pred = nn.Sequential(
            nn.Linear(d * 2, d * 3),  # last_loc + user
            nn.LayerNorm(d * 3),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d * 3, num_locs)
        )
        
        # 2. S2 cell predictor
        self.s2_pred = nn.Sequential(
            nn.Linear(d + d // 2, d * 2),  # s2_l14 + s2_l13
            nn.LayerNorm(d * 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d * 2, num_locs)
        )
        
        # 3. User preference predictor
        self.user_pred = nn.Sequential(
            nn.Linear(d + d // 4, d * 2),  # user + weekday
            nn.LayerNorm(d * 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d * 2, num_locs)
        )
        
        # 4. Sequence predictor (LSTM)
        self.lstm = nn.LSTM(
            input_size=d,
            hidden_size=d,
            num_layers=2,
            dropout=0.15,
            batch_first=True
        )
        
        self.seq_pred = nn.Linear(d, num_locs)
        
        # Gating network - learns which predictor to trust
        gate_input = d * 3 + d // 2 + d // 4  # all context
        self.gate = nn.Sequential(
            nn.Linear(gate_input, d * 2),
            nn.LayerNorm(d * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d * 2, 4),  # 4 predictors
            nn.Softmax(dim=-1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(p)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
            elif 'weight' in name and p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.02)
    
    def forward(self, batch):
        x = batch['X']
        user_x = batch['user_X']
        weekday_x = batch['weekday_X']
        s2_l13 = batch['s2_level14_X']
        s2_l14 = batch['s2_level15_X']
        mask_pad = batch['mask']
        
        B, T = x.shape
        device = x.device
        lengths = mask_pad.sum(dim=1)
        
        batch_idx = torch.arange(B, device=device)
        last_idx = lengths - 1
        
        # Get last position features
        last_loc = x[batch_idx, last_idx]
        last_loc_emb = self.loc_emb(last_loc)
        
        user = user_x[batch_idx, 0]
        user_emb = self.user_emb(user)
        
        last_s14 = s2_l14[batch_idx, last_idx]
        last_s14_emb = self.s2_l14_emb(last_s14)
        
        last_s13 = s2_l13[batch_idx, last_idx]
        last_s13_emb = self.s2_l13_emb(last_s13)
        
        last_weekday = weekday_x[batch_idx, last_idx]
        weekday_emb = self.weekday_emb(last_weekday)
        
        # Get predictions from each expert
        # 1. Transition patterns
        transition_logits = self.transition_pred(
            torch.cat([last_loc_emb, user_emb], dim=-1)
        )
        
        # 2. S2 patterns
        s2_logits = self.s2_pred(
            torch.cat([last_s14_emb, last_s13_emb], dim=-1)
        )
        
        # 3. User patterns
        user_logits = self.user_pred(
            torch.cat([user_emb, weekday_emb], dim=-1)
        )
        
        # 4. Sequence patterns
        loc_seq = self.loc_emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            loc_seq, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        seq_logits = self.seq_pred(hidden[-1])
        
        # Compute gate weights
        context = torch.cat([last_loc_emb, user_emb, last_s14_emb, last_s13_emb, weekday_emb], dim=-1)
        gate_weights = self.gate(context)  # [B, 4]
        
        # Weighted combination
        all_logits = torch.stack([
            transition_logits,
            s2_logits,
            user_logits,
            seq_logits
        ], dim=1)  # [B, 4, num_locs]
        
        # Apply gates
        final_logits = (all_logits * gate_weights.unsqueeze(-1)).sum(dim=1)  # [B, num_locs]
        
        return final_logits
