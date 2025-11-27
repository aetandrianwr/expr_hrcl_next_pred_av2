# Hierarchical Transformer for Next-Location Prediction

This repository implements a hierarchical Transformer architecture for next-location prediction on the GeoLife dataset, using multi-resolution S2 spatial features.

## System Overview

The system predicts a single next-location label Y from a sequence of past locations X and associated Google S2 features at levels 11-15. The architecture uses hierarchical attention mechanisms to fuse information from multiple spatial resolutions.

## Results Summary

### Models Tested

1. **Initial Hierarchical Cross-Attention (V1)**: ~494K params
   - Test Acc@1: 30.53%
   - Architecture: Separate Transformer streams for main features and S2 levels with cross-attention

2. **Multi-Resolution Fusion (V2)**: ~964K params
   - Test Acc@1: 35.15%  
   - Architecture: Fusion of S2 features via dedicated fusion module, single Transformer encoder

3. **Weighted Feature Combination (V3)**: ~991K params
   - Test Acc@1: 27.84%
   - Architecture: Learnable weighted sum of all embeddings before Transformer

### Best Model: Multi-Resolution Fusion (V2)

**Test Performance:**
- Acc@1: 35.15%
- Acc@5: 55.88%
- Acc@10: 58.62%
- MRR: 44.76%
- NDCG: 47.85%

**Architecture Details:**
- d_model: 88
- num_heads: 4
- num_layers: 4
- Parameters: 964,448 (< 1M budget)
- Dropout: 0.25
- Label smoothing: 0.05

## Implementation

### Files

- `model.py`: Initial hierarchical cross-attention architecture
- `model_v2.py`: Multi-resolution fusion architecture (best performance)
- `model_v3.py`: Weighted feature combination architecture
- `dataset.py`: Data loading and batching
- `metrics.py`: Evaluation metrics (Acc@K, MRR, NDCG, F1)
- `train.py`: Training loop with early stopping

### Key Design Choices

1. **Multi-Resolution S2 Features**: Embeddings for S2 levels 11, 13, 14, 15 provide hierarchical spatial context
2. **Causal Masking**: Ensures predictions only use past information
3. **Label Smoothing**: Regularizes the classifier to prevent overfitting
4. **Early Stopping**: Monitors validation accuracy with patience=20

## Training

```bash
python train.py
```

Training uses:
- GPU acceleration with CUDA
- Fixed random seed (42) for reproducibility
- AdamW optimizer with ReduceLROnPlateau scheduler
- Batch size: 64-128
- Learning rate: 0.001-0.002

## Performance Analysis

The hierarchical Transformer approach shows modest improvements over baselines but struggles to reach 50% Acc@1. Key observations:

1. **S2 Feature Utility**: Multi-resolution spatial features provide useful context, improving performance over location-only models
2. **Architecture Trade-offs**: More complex hierarchical attention doesn't necessarily improve over simpler fusion approaches
3. **Dataset Difficulty**: GeoLife next-location prediction appears challenging with high location vocabulary (1186 classes) and variable sequence lengths

## Requirements

- PyTorch
- CUDA-capable GPU
- sklearn
- tqdm

## Model Selection

The best model (`model_v2.py`) balances performance and parameter efficiency through:
- Independent processing of S2 levels with dedicated transforms
- Fusion layer to combine multi-resolution information
- Standard Transformer encoder for temporal modeling
- Simple classification head

Total parameters: 964,448 (within 1M budget)
