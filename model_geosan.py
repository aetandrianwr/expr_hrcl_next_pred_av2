"""
Geography-Aware Hierarchical Transformer
Based on proven research: GeoSAN (AAAI 2020) + LSTPM (KDD 2020) + STAN (SIGIR 2020)

Key innovations:
1. Geography-aware attention (spatial distance modulates attention)
2. Long-short term preference separation
3. Hierarchical S2 encoding with spatial relationships
4. Cyclical time encoding
5. User preference modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CyclicalTimeEncoding(nn.Module):
    """Encode time cyclically for periodic patterns"""
    def __init__(self, d_model):
        super().__init__()
        self.weekday_linear = nn.Linear(2, d_model // 4)  # sin/cos for weekday
        
    def forward(self, weekday):
        # weekday: [B, T] with values 0-6
        # Convert to cyclical encoding
        angle = 2 * math.pi * weekday.float() / 7
        sin_weekday = torch.sin(angle).unsqueeze(-1)
        cos_weekday = torch.cos(angle).unsqueeze(-1)
        cyclical = torch.cat([sin_weekday, cos_weekday], dim=-1)
        return self.weekday_linear(cyclical)


class GeographyAwareAttention(nn.Module):
    """
    Self-attention that incorporates spatial distance
    Based on GeoSAN (AAAI 2020)
    
    attention_score = (QK^T / sqrt(d)) - alpha * spatial_distance
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Spatial distance weighting (learnable)
        self.spatial_weight = nn.Parameter(torch.tensor(0.5))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x, s2_distances, mask=None):
        """
        x: [B, T, d_model]
        s2_distances: [B, T, T] - pairwise S2 hierarchical distances
        mask: [B, T, T] - causal mask
        """
        B, T, _ = x.shape
        
        # Multi-head projections
        Q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, T, d_k]
        K = self.W_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        
        # Standard attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, T, T]
        
        # Geography-aware component: encode spatial distances
        # Simple linear transformation of distances
        dist_bias = -self.spatial_weight.abs() * s2_distances.unsqueeze(1)  # [B, 1, T, T]
        dist_bias = dist_bias.expand(B, self.num_heads, T, T)  # [B, H, T, T]
        
        # Combine: standard attention + spatial bias (not subtraction)
        attn_scores = attn_scores + dist_bias
        
        # Apply causal mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and apply to values
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        out = torch.matmul(attn_probs, V)  # [B, H, T, d_k]
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        
        return self.W_o(out)


class LongShortTermModule(nn.Module):
    """
    Separate modeling of long-term preferences and short-term transitions
    Based on LSTPM (KDD 2020)
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        
        # Long-term: user preferences (global)
        self.long_term_attn = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=dropout, batch_first=True
        )
        
        # Short-term: recent trajectory (local)
        self.short_term_attn = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=dropout, batch_first=True
        )
        
        # Fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        """
        x: [B, T, d_model]
        Returns: [B, T, d_model] with long+short term fusion
        """
        # Long-term: attend to all history
        long_term, _ = self.long_term_attn(x, x, x, key_padding_mask=~mask if mask is not None else None)
        
        # Short-term: attend to recent (last 5 positions)
        recent_mask = torch.zeros_like(mask) if mask is not None else None
        if recent_mask is not None:
            recent_mask[:, -5:] = mask[:, -5:]
        short_term, _ = self.short_term_attn(x, x, x, key_padding_mask=~recent_mask if recent_mask is not None else None)
        
        # Fusion with gating
        gate = self.fusion_gate(torch.cat([long_term, short_term], dim=-1))
        fused = gate * long_term + (1 - gate) * short_term
        
        return self.layer_norm(fused + x)


class GeographyAwareHierarchicalTransformer(nn.Module):
    """
    Complete model combining all proven techniques
    """
    def __init__(self, config):
        super().__init__()
        
        d = config['d_model']
        num_locs = config['num_locations']
        
        # Embeddings
        self.loc_emb = nn.Embedding(num_locs, d)
        self.user_emb = nn.Embedding(config['num_users'], d)
        
        # S2 hierarchical embeddings
        self.s2_l11_emb = nn.Embedding(config['num_s2_l11'], d // 4)
        self.s2_l12_emb = nn.Embedding(config['num_s2_l12'], d // 4)
        self.s2_l13_emb = nn.Embedding(config['num_s2_l13'], d // 4)
        self.s2_l14_emb = nn.Embedding(config['num_s2_l14'], d // 4)
        
        # Cyclical time encoding
        self.time_encoding = CyclicalTimeEncoding(d)
        
        # Project S2 hierarchy
        self.s2_proj = nn.Linear(d, d)
        
        # Combine features
        self.input_proj = nn.Linear(d * 2 + d // 4, d)  # loc + user + time
        
        # Geography-aware attention layers
        self.geo_attn1 = GeographyAwareAttention(d, num_heads=8, dropout=0.15)
        self.norm1 = nn.LayerNorm(d)
        
        self.geo_attn2 = GeographyAwareAttention(d, num_heads=8, dropout=0.15)
        self.norm2 = nn.LayerNorm(d)
        
        # Long-short term module
        self.long_short = LongShortTermModule(d, dropout=0.15)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d * 4, d)
        )
        self.norm_ffn = nn.LayerNorm(d)
        
        # Output
        self.output = nn.Sequential(
            nn.Linear(d * 2, d),  # Concat last_hidden + user
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d, num_locs)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.01)
    
    def compute_s2_distances(self, s2_levels):
        """
        Compute hierarchical S2 distances between all pairs
        s2_levels: list of [B, T] tensors for each S2 level
        Returns: [B, T, T] distance matrix
        """
        B, T = s2_levels[0].shape
        device = s2_levels[0].device
        
        # Hierarchical distance: count how many levels differ
        # If all levels match: distance = 0
        # If all differ: distance = 1
        
        distances = torch.zeros(B, T, T, device=device)
        
        for s2_level in s2_levels:
            # [B, T, T] - 1 if same, 0 if different
            same = (s2_level.unsqueeze(2) == s2_level.unsqueeze(1)).float()
            # Add 0.25 for each level that differs
            distances += (1 - same) * 0.25
        
        # Clamp to avoid extreme values
        distances = torch.clamp(distances, 0, 1.0)
        
        return distances
    
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
        loc_emb = self.loc_emb(x)  # [B, T, d]
        user_emb = self.user_emb(user_x)  # [B, T, d]
        time_emb = self.time_encoding(weekday_x)  # [B, T, d//4]
        
        # S2 hierarchical features
        s2_emb = torch.cat([
            self.s2_l11_emb(s2_l11),
            self.s2_l12_emb(s2_l12),
            self.s2_l13_emb(s2_l13),
            self.s2_l14_emb(s2_l14)
        ], dim=-1)  # [B, T, d]
        s2_emb = self.s2_proj(s2_emb)
        
        # Combine features
        combined = torch.cat([loc_emb, user_emb, time_emb], dim=-1)
        x_proj = self.input_proj(combined)  # [B, T, d]
        
        # Add S2 spatial context
        x_proj = x_proj + s2_emb
        
        # Compute S2-based spatial distances
        s2_distances = self.compute_s2_distances([s2_l11, s2_l12, s2_l13, s2_l14])
        
        # Causal mask
        causal_mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0)  # [1, T, T]
        
        # Geography-aware attention layer 1
        attn1 = self.geo_attn1(x_proj, s2_distances, causal_mask)
        x_proj = self.norm1(x_proj + attn1)
        
        # Geography-aware attention layer 2
        attn2 = self.geo_attn2(x_proj, s2_distances, causal_mask)
        x_proj = self.norm2(x_proj + attn2)
        
        # Long-short term module
        x_proj = self.long_short(x_proj, mask_pad)
        
        # FFN
        ffn_out = self.ffn(x_proj)
        x_proj = self.norm_ffn(x_proj + ffn_out)
        
        # Get last position
        lengths = mask_pad.sum(dim=1) - 1
        batch_idx = torch.arange(B, device=device)
        last_hidden = x_proj[batch_idx, lengths]  # [B, d]
        last_user = user_emb[batch_idx, 0]  # [B, d] - user is constant
        
        # Output
        final_repr = torch.cat([last_hidden, last_user], dim=-1)
        logits = self.output(final_repr)
        
        return logits
