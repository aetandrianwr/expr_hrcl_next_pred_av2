# Hierarchical Transformer Next-Location Prediction - Final Report

## Objective
Build a hierarchical Transformer for next-location prediction on GeoLife dataset achieving ≥50% Acc@1 with <1M parameters.

## Results Summary

### Best Model: Multi-Resolution Fusion (V2)
- **Test Acc@1: 35.15%** (vs 50% target)
- **Parameters: 964,448** (within 1M budget ✓)
- **Architecture**: Multi-resolution S2 fusion + Transformer encoder
- **File**: `model_v2.py`, `train.py`

### All Models Tested (10 iterations)

1. **Hierarchical Cross-Attention (V1)**: 494K params → 30.53% test
2. **Multi-Resolution Fusion (V2)**: 964K params → **35.15% test** ⭐
3. **Weighted Feature Combination (V3)**: 991K params → 27.84% test
4. **Optimized Hierarchical**: 980K params → 24.27% test
5. **Simple GRU**: 738K params → 29.81% test
6. **Focused S2**: 850K params → 30.64% test
7. **Transition-Aware**: 806K params → 30.33% test
8. **Hybrid Pattern**: 956K params → 28.98% test
9. **Direct Transition**: 962K params → 28.41% test

## Key Findings

### Critical Insights from Data Analysis

1. **Simple Baselines Outperform Neural Networks**:
   - Location transition patterns: **50.53%** (lookup table)
   - User-specific patterns: **64.75%** average per user
   - S2 Level 15 alone: **44.61%** (most common per cell)
   - S2+Location combination: **48.63%**

2. **Neural Networks Consistently Underperform**:
   - All 10 models: 24-35% range
   - **Gap**: Simple patterns (50%) vs Neural nets (35%) = **15% deficit**

### Why Neural Networks Struggle

1. **Over-parameterization**: Complex architectures destroy signal rather than enhance it
2. **Insufficient memorization**: 7,426 training samples insufficient for 1,093 location classes
3. **Wrong inductive bias**: Trying to generalize when task requires memorization
4. **S2 signal dilution**: Hierarchical attention dilutes the strong S2 L15 signal (44.6%)

## Architecture Details - Best Model (V2)

### Components
```
Input: [location, user, weekday, S2_L11, S2_L13, S2_L14, S2_L15]
├── Location embedding (d_model)
├── User embedding (d_model/2)
├── Weekday embedding (d_model/4)
├── S2 embeddings (d_model each)
└── Multi-Resolution Fusion Module
    ├── Independent transforms per S2 level
    ├── Concatenate + fusion layer
    └── Output: fused S2 representation

Combined features → Transformer Encoder (4 layers, 4 heads)
Last position → LayerNorm → Classifier
```

### Training Setup
- Batch size: 64
- Optimizer: AdamW (lr=0.001, weight_decay=0.01)
- Scheduler: ReduceLROnPlateau
- Label smoothing: 0.05
- Dropout: 0.25
- Early stopping: patience=20

## What Would Be Needed for 50%+

### Approach 1: Hybrid Lookup + Neural Network
Build explicit lookup tables for common patterns, use neural network only for rare cases.

### Approach 2: Ensemble
Combine multiple models with different architectures.

### Approach 3: More Data or Better Features
- 7.4K samples may be insufficient for 1K classes
- Need better spatial/temporal features
- Or significantly more training data

### Approach 4: Different Architecture Family
- Graph Neural Networks for spatial relationships
- Memory-Augmented Networks for explicit pattern storage
- Retrieval-based methods

## Techniques Applied

✅ Multi-resolution S2 hierarchy
✅ Skip connections
✅ Layer normalization (Pre-LN)
✅ Label smoothing
✅ Gradient clipping
✅ OneCycleLR / CosineAnnealing
✅ MixUp augmentation
✅ Careful weight initialization
✅ AdamW with weight decay
✅ Various learning rates (0.001-0.01)
✅ Various batch sizes (64-512)
✅ Different sequence encoders (Transformer, LSTM, GRU)
✅ Attention mechanisms
✅ Gated combinations
✅ Direct transition modeling

## Conclusion

Despite 10 different architectural approaches and extensive hyperparameter tuning, the hierarchical Transformer approach consistently achieves **~30-35% accuracy**, falling short of the **50% target** by **15-20 percentage points**.

The core issue is that **simple pattern matching outperforms learned representations** for this task, suggesting the dataset characteristics (high class count, low sample density) favor memorization over generalization.

**Best deliverable**: Model V2 with 35.15% test accuracy, 964K parameters, implementing hierarchical S2 feature fusion with causal Transformer encoder.

## Files

- `model_v2.py`: Best performing model
- `train.py`: Training script for V2
- `results.json`: Final results
- `best_model.pt`: Trained weights
- `training_log_v4.txt`: Training log for V2
- All experiments committed to git: https://github.com/aetandrianwr/expr_hrcl_next_pred_av2

