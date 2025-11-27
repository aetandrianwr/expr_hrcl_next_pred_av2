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


class MultiResolutionFusion(nn.Module):
    """Fuse multi-resolution S2 features at each time step."""
    def __init__(self, d_model, num_levels=4, dropout=0.1):
        super().__init__()
        # Independent transformations for each S2 level
        self.level_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_levels)
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * num_levels, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, s2_features):
        """
        s2_features: list of [B, T, d_model] tensors, one per S2 level
        """
        # Transform each level
        transformed = [transform(feat) for transform, feat in zip(self.level_transforms, s2_features)]
        
        # Concatenate and fuse
        concatenated = torch.cat(transformed, dim=-1)  # [B, T, d_model * num_levels]
        fused = self.fusion(concatenated)  # [B, T, d_model]
        
        return fused


class HierarchicalTransformerV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Configuration
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        
        # Vocabulary sizes
        self.num_locations = config['num_locations']
        self.num_users = config['num_users']
        self.num_weekdays = config['num_weekdays']
        self.num_s2_l11 = config['num_s2_l11']
        self.num_s2_l12 = config['num_s2_l12']
        self.num_s2_l13 = config['num_s2_l13']
        self.num_s2_l14 = config['num_s2_l14']
        
        # Main embeddings
        self.loc_embedding = nn.Embedding(self.num_locations, self.d_model)
        self.user_embedding = nn.Embedding(self.num_users, self.d_model // 2)
        self.weekday_embedding = nn.Embedding(self.num_weekdays, self.d_model // 4)
        
        # S2 embeddings
        self.s2_l11_embedding = nn.Embedding(self.num_s2_l11, self.d_model)
        self.s2_l12_embedding = nn.Embedding(self.num_s2_l12, self.d_model)
        self.s2_l13_embedding = nn.Embedding(self.num_s2_l13, self.d_model)
        self.s2_l14_embedding = nn.Embedding(self.num_s2_l14, self.d_model)
        
        # Input projection
        input_dim = self.d_model + self.d_model // 2 + self.d_model // 4
        self.input_proj = nn.Linear(input_dim, self.d_model)
        
        # Multi-resolution fusion
        self.s2_fusion = MultiResolutionFusion(self.d_model, num_levels=4, dropout=self.dropout)
        
        # Combine main and S2 streams
        self.stream_combine = nn.Linear(self.d_model * 2, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=100)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Output head
        self.output_norm = nn.LayerNorm(self.d_model)
        self.classifier = nn.Linear(self.d_model, self.num_locations)
        
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _generate_causal_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    
    def forward(self, batch):
        # Unpack batch
        x = batch['X']  # [B, T]
        user_x = batch['user_X']  # [B, T]
        weekday_x = batch['weekday_X']  # [B, T]
        s2_l11 = batch['s2_level11_X']  # [B, T]
        s2_l12 = batch['s2_level13_X']  # [B, T]
        s2_l13 = batch['s2_level14_X']  # [B, T]
        s2_l14 = batch['s2_level15_X']  # [B, T]
        mask_pad = batch['mask']  # [B, T]
        
        B, T = x.shape
        device = x.device
        
        # Embed main features
        loc_emb = self.loc_embedding(x)  # [B, T, d_model]
        user_emb = self.user_embedding(user_x)  # [B, T, d_model//2]
        weekday_emb = self.weekday_embedding(weekday_x)  # [B, T, d_model//4]
        
        # Project main features
        main_features = torch.cat([loc_emb, user_emb, weekday_emb], dim=-1)
        main_stream = self.input_proj(main_features)  # [B, T, d_model]
        
        # Embed S2 features
        s2_features = [
            self.s2_l11_embedding(s2_l11),
            self.s2_l12_embedding(s2_l12),
            self.s2_l13_embedding(s2_l13),
            self.s2_l14_embedding(s2_l14),
        ]
        
        # Fuse S2 features
        s2_fused = self.s2_fusion(s2_features)  # [B, T, d_model]
        
        # Combine main and S2 streams
        combined = torch.cat([main_stream, s2_fused], dim=-1)  # [B, T, 2*d_model]
        combined = self.stream_combine(combined)  # [B, T, d_model]
        
        # Add positional encoding
        combined = self.pos_encoder(combined)
        
        # Generate causal mask
        causal_mask = self._generate_causal_mask(T).to(device)
        
        # Pass through transformer
        # Create padding mask (True for padding positions)
        padding_mask = ~mask_pad  # [B, T]
        
        transformer_out = self.transformer_encoder(
            combined,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )  # [B, T, d_model]
        
        # Get last valid position for each sequence
        transformer_out = self.output_norm(transformer_out)
        seq_lengths = mask_pad.sum(dim=1) - 1  # [B]
        batch_indices = torch.arange(B, device=device)
        final_repr = transformer_out[batch_indices, seq_lengths]  # [B, d_model]
        
        # Classify
        logits = self.classifier(final_repr)  # [B, num_locations]
        
        return logits
