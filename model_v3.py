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


class SimpleHierarchicalTransformer(nn.Module):
    """
    Simpler, more effective hierarchical approach: embed all features, 
    add them together with weights, and process with Transformer.
    """
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
        
        # All embeddings project to d_model
        self.loc_embedding = nn.Embedding(self.num_locations, self.d_model)
        self.user_embedding = nn.Embedding(self.num_users, self.d_model)
        self.weekday_embedding = nn.Embedding(self.num_weekdays, self.d_model)
        
        # S2 embeddings at different resolutions
        self.s2_l11_embedding = nn.Embedding(self.num_s2_l11, self.d_model)
        self.s2_l12_embedding = nn.Embedding(self.num_s2_l12, self.d_model)
        self.s2_l13_embedding = nn.Embedding(self.num_s2_l13, self.d_model)
        self.s2_l14_embedding = nn.Embedding(self.num_s2_l14, self.d_model)
        
        # Learnable weights for feature combination
        self.feature_weights = nn.Parameter(torch.ones(7))  # 7 features
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=100)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Output head
        self.output_norm = nn.LayerNorm(self.d_model)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.classifier = nn.Linear(self.d_model, self.num_locations)
        
        self._init_weights()
        
    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
    
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
        
        # Embed all features
        loc_emb = self.loc_embedding(x)  # [B, T, d_model]
        user_emb = self.user_embedding(user_x)
        weekday_emb = self.weekday_embedding(weekday_x)
        s2_l11_emb = self.s2_l11_embedding(s2_l11)
        s2_l12_emb = self.s2_l12_embedding(s2_l12)
        s2_l13_emb = self.s2_l13_embedding(s2_l13)
        s2_l14_emb = self.s2_l14_embedding(s2_l14)
        
        # Stack and weight features
        features = [loc_emb, user_emb, weekday_emb, s2_l11_emb, s2_l12_emb, s2_l13_emb, s2_l14_emb]
        weights = F.softmax(self.feature_weights, dim=0)
        
        # Weighted sum of all features
        combined = sum(w * f for w, f in zip(weights, features))  # [B, T, d_model]
        
        # Add positional encoding
        combined = self.pos_encoder(combined)
        combined = self.dropout_layer(combined)
        
        # Generate causal mask
        causal_mask = self._generate_causal_mask(T).to(device)
        padding_mask = ~mask_pad  # [B, T]
        
        # Pass through transformer
        transformer_out = self.transformer_encoder(
            combined,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )  # [B, T, d_model]
        
        # Get last valid position
        transformer_out = self.output_norm(transformer_out)
        seq_lengths = mask_pad.sum(dim=1) - 1  # [B]
        batch_indices = torch.arange(B, device=device)
        final_repr = transformer_out[batch_indices, seq_lengths]  # [B, d_model]
        
        # Classify
        final_repr = self.dropout_layer(final_repr)
        logits = self.classifier(final_repr)  # [B, num_locations]
        
        return logits
