"""
ULTRA-SIMPLE: Just learn location transitions with minimal abstraction
"""

import torch
import torch.nn as nn


class DirectTransitionModel(nn.Module):
    """
    Directly model P(next_loc | curr_loc, user, s2)
    No fancy architecture, just learn the patterns
    """
    def __init__(self, config):
        super().__init__()
        
        d = config['d_model']
        num_locs = config['num_locations']
        
        # Simple embeddings
        self.loc_emb = nn.Embedding(num_locs, d)
        self.user_emb = nn.Embedding(config['num_users'], d)
        self.s2_emb = nn.Embedding(config['num_s2_l14'], d)
        
        # Direct transition prediction - BIG capacity to memorize patterns
        self.predictor = nn.Sequential(
            nn.Linear(d * 3, d * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d * 4, d * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d * 4, num_locs)
        )
        
        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, batch):
        x = batch['X']
        user_x = batch['user_X']
        s2_l14 = batch['s2_level15_X']
        mask_pad = batch['mask']
        
        B = x.size(0)
        device = x.device
        lengths = mask_pad.sum(dim=1)
        
        batch_idx = torch.arange(B, device=device)
        last_idx = lengths - 1
        
        # Get last position
        last_loc = self.loc_emb(x[batch_idx, last_idx])
        user = self.user_emb(user_x[batch_idx, 0])
        s2 = self.s2_emb(s2_l14[batch_idx, last_idx])
        
        # Concatenate and predict
        features = torch.cat([last_loc, user, s2], dim=-1)
        logits = self.predictor(features)
        
        return logits
