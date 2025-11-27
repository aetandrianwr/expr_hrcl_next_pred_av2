"""
BREAKTHROUGH MODEL: Focus on what works
- Location transitions (50.5% baseline)
- User-specific patterns (64.75% per-user)
- S2 as supporting context

Strategy: Learn transition matrices per user with S2 context
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransitionAwarePredictor(nn.Module):
    """
    Model transition probabilities with user and S2 context
    """
    def __init__(self, config):
        super().__init__()
        
        d = config['d_model']
        
        # Core embeddings
        self.loc_emb = nn.Embedding(config['num_locations'], d)
        self.user_emb = nn.Embedding(config['num_users'], d)
        
        # S2 hierarchy - use ALL levels
        self.s2_l11_emb = nn.Embedding(config['num_s2_l11'], d // 4)
        self.s2_l12_emb = nn.Embedding(config['num_s2_l12'], d // 4)
        self.s2_l13_emb = nn.Embedding(config['num_s2_l13'], d // 4)
        self.s2_l14_emb = nn.Embedding(config['num_s2_l14'], d // 4)
        
        self.weekday_emb = nn.Embedding(config['num_weekdays'], d // 8)
        
        # Transition modeling: last location + user + S2 -> next location
        # This captures the 50.5% transition pattern + 64.75% user pattern
        
        transition_input = d * 2 + d + d // 8  # last_loc + user + s2_combined + weekday
        
        self.transition_net = nn.Sequential(
            nn.Linear(transition_input, d * 3),
            nn.LayerNorm(d * 3),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d * 3, d * 3),
            nn.LayerNorm(d * 3),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d * 3, config['num_locations'])
        )
        
        # Sequence context (temporal patterns)
        self.lstm = nn.LSTM(
            input_size=d + d // 4 * 4,  # loc + s2_all
            hidden_size=d,
            num_layers=1,
            batch_first=True
        )
        
        # Combine transition prediction with sequence context
        self.combiner = nn.Sequential(
            nn.Linear(config['num_locations'] + d, d * 2),
            nn.LayerNorm(d * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d * 2, config['num_locations'])
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
        lengths = mask_pad.sum(dim=1)
        
        # Get last position features
        batch_idx = torch.arange(B, device=device)
        last_idx = lengths - 1
        
        # Last location (transition source)
        last_loc = x[batch_idx, last_idx]
        last_loc_emb = self.loc_emb(last_loc)  # [B, d]
        
        # User (user-specific patterns)
        user = user_x[batch_idx, 0]  # User is constant
        user_emb = self.user_emb(user)  # [B, d]
        
        # S2 context at last position
        last_s11 = self.s2_l11_emb(s2_l11[batch_idx, last_idx])
        last_s12 = self.s2_l12_emb(s2_l12[batch_idx, last_idx])
        last_s13 = self.s2_l13_emb(s2_l13[batch_idx, last_idx])
        last_s14 = self.s2_l14_emb(s2_l14[batch_idx, last_idx])
        s2_combined = torch.cat([last_s11, last_s12, last_s13, last_s14], dim=-1)  # [B, d]
        
        # Weekday
        last_weekday = weekday_x[batch_idx, last_idx]
        weekday_emb = self.weekday_emb(last_weekday)  # [B, d//8]
        
        # BRANCH 1: Transition prediction (captures 50.5% + 64.75% patterns)
        transition_input = torch.cat([last_loc_emb, user_emb, s2_combined, weekday_emb], dim=-1)
        transition_logits = self.transition_net(transition_input)  # [B, num_locations]
        
        # BRANCH 2: Sequence context
        loc_seq = self.loc_emb(x)  # [B, T, d]
        s2_seq = torch.cat([
            self.s2_l11_emb(s2_l11),
            self.s2_l12_emb(s2_l12),
            self.s2_l13_emb(s2_l13),
            self.s2_l14_emb(s2_l14)
        ], dim=-1)  # [B, T, d]
        
        seq_features = torch.cat([loc_seq, s2_seq], dim=-1)
        
        # Pack and process with LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            seq_features, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        seq_context = hidden.squeeze(0)  # [B, d]
        
        # Combine both branches
        combined = torch.cat([transition_logits, seq_context], dim=-1)
        final_logits = self.combiner(combined)
        
        return final_logits
