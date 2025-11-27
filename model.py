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
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class HierarchicalCrossAttentionLayer(nn.Module):
    """
    Cross-attention where queries come from main stream, 
    keys and values from a specific S2 level stream.
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value, mask=None):
        # Cross-attention: Q from main, K,V from S2 level
        attn_out, _ = self.cross_attn(query, key_value, key_value, attn_mask=mask, need_weights=False)
        out = self.norm(query + self.dropout(attn_out))
        return out


class CausalTransformerBlock(nn.Module):
    """Standard causal transformer block with self-attention."""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.norm1(x + self.dropout1(attn_out))
        
        # Feedforward
        ff_out = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ff_out))
        return x


class HierarchicalTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Configuration
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.num_main_layers = config['num_main_layers']
        self.num_fusion_layers = config['num_fusion_layers']
        self.dropout = config['dropout']
        
        # Vocabulary sizes
        self.num_locations = config['num_locations']
        self.num_users = config['num_users']
        self.num_weekdays = config['num_weekdays']
        self.num_s2_l11 = config['num_s2_l11']
        self.num_s2_l12 = config['num_s2_l12']  # Using l13 as l12
        self.num_s2_l13 = config['num_s2_l13']  # Using l14 as l13
        self.num_s2_l14 = config['num_s2_l14']  # Using l15 as l14
        self.num_s2_l15 = config['num_s2_l15']  # Placeholder
        
        # Main stream embeddings (for X and auxiliary features)
        self.loc_embedding = nn.Embedding(self.num_locations, self.d_model)
        self.user_embedding = nn.Embedding(self.num_users, self.d_model // 2)
        self.weekday_embedding = nn.Embedding(self.num_weekdays, self.d_model // 4)
        
        # Project concatenated embeddings to d_model
        input_dim = self.d_model + self.d_model // 2 + self.d_model // 4
        self.input_proj = nn.Linear(input_dim, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        
        # S2 level embeddings
        s2_dim = self.d_model
        self.s2_l11_embedding = nn.Embedding(self.num_s2_l11, s2_dim)
        self.s2_l12_embedding = nn.Embedding(self.num_s2_l12, s2_dim)
        self.s2_l13_embedding = nn.Embedding(self.num_s2_l13, s2_dim)
        self.s2_l14_embedding = nn.Embedding(self.num_s2_l14, s2_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=100)
        self.s2_pos_encoder = PositionalEncoding(self.d_model, max_len=100)
        
        # Main transformer layers
        self.main_layers = nn.ModuleList([
            CausalTransformerBlock(self.d_model, self.nhead, self.d_model * 4, self.dropout)
            for _ in range(self.num_main_layers)
        ])
        
        # Cross-attention layers for each S2 level
        self.cross_attn_layers = nn.ModuleList([
            HierarchicalCrossAttentionLayer(self.d_model, self.nhead, self.dropout)
            for _ in range(4)  # 4 S2 levels
        ])
        
        # Fusion layers after concatenation
        # Input will be d_model (main) + 4 * d_model (cross-attention outputs)
        self.fusion_proj = nn.Linear(5 * self.d_model, self.d_model)
        self.fusion_norm = nn.LayerNorm(self.d_model)
        self.fusion_layers = nn.ModuleList([
            CausalTransformerBlock(self.d_model, self.nhead, self.d_model * 4, self.dropout)
            for _ in range(self.num_fusion_layers)
        ])
        
        # Output head with intermediate layer
        self.output_norm = nn.LayerNorm(self.d_model)
        self.pre_classifier = nn.Linear(self.d_model, self.d_model * 2)
        self.classifier = nn.Linear(self.d_model * 2, self.num_locations)
        
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _generate_causal_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, batch):
        # Unpack batch
        x = batch['X']  # [B, T]
        user_x = batch['user_X']  # [B, T]
        weekday_x = batch['weekday_X']  # [B, T]
        s2_l11 = batch['s2_level11_X']  # [B, T]
        s2_l12 = batch['s2_level13_X']  # [B, T] (using 13 as 12)
        s2_l13 = batch['s2_level14_X']  # [B, T] (using 14 as 13)
        s2_l14 = batch['s2_level15_X']  # [B, T] (using 15 as 14)
        mask_pad = batch['mask']  # [B, T]
        
        B, T = x.shape
        device = x.device
        
        # === Main stream ===
        # Embed main features
        loc_emb = self.loc_embedding(x)  # [B, T, d_model]
        user_emb = self.user_embedding(user_x)  # [B, T, d_model//2]
        weekday_emb = self.weekday_embedding(weekday_x)  # [B, T, d_model//4]
        
        # Concatenate and project
        main_features = torch.cat([loc_emb, user_emb, weekday_emb], dim=-1)
        main_stream = self.input_proj(main_features)  # [B, T, d_model]
        main_stream = self.input_norm(main_stream)
        main_stream = self.pos_encoder(main_stream)
        
        # Generate causal mask
        causal_mask = self._generate_causal_mask(T, device)
        
        # Pass through main transformer layers
        for layer in self.main_layers:
            main_stream = layer(main_stream, causal_mask)
        
        # === S2 level streams ===
        s2_levels = [s2_l11, s2_l12, s2_l13, s2_l14]
        s2_embeddings = [
            self.s2_l11_embedding(s2_l11),
            self.s2_l12_embedding(s2_l12),
            self.s2_l13_embedding(s2_l13),
            self.s2_l14_embedding(s2_l14),
        ]
        
        # Add positional encoding
        s2_streams = []
        for emb in s2_embeddings:
            stream = self.s2_pos_encoder(emb)
            s2_streams.append(stream)
        
        # === Cross-attention ===
        # Each S2 level attends to main stream as query, S2 as key/value
        cross_outputs = []
        for s2_stream, cross_attn in zip(s2_streams, self.cross_attn_layers):
            # Query from main_stream, Key/Value from s2_stream
            cross_out = cross_attn(main_stream, s2_stream, causal_mask)
            cross_outputs.append(cross_out)
        
        # === Fusion ===
        # Concatenate main stream with all cross-attention outputs
        fused = torch.cat([main_stream] + cross_outputs, dim=-1)  # [B, T, 5*d_model]
        fused = self.fusion_proj(fused)  # [B, T, d_model]
        fused = self.fusion_norm(fused)
        
        # Pass through fusion layers
        for layer in self.fusion_layers:
            fused = layer(fused, causal_mask)
        
        # === Output ===
        # Take last time step (accounting for padding)
        fused = self.output_norm(fused)
        
        # Get the last valid position for each sequence
        seq_lengths = mask_pad.sum(dim=1) - 1  # [B]
        batch_indices = torch.arange(B, device=device)
        final_repr = fused[batch_indices, seq_lengths]  # [B, d_model]
        
        # Classify with intermediate layer
        hidden = F.relu(self.pre_classifier(final_repr))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        logits = self.classifier(hidden)  # [B, num_locations]
        
        return logits
