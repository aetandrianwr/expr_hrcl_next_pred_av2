# Research: Proven Techniques for Next-Location Prediction

## Key Papers & Techniques

### 1. Self-Attention with Spatial Context (STAN - 2020)
- **Problem**: Standard RNNs/Transformers ignore spatial relationships
- **Solution**: 
  - Local + Global attention mechanisms
  - Spatial distance encoding in attention
  - Non-local attention for capturing long-range dependencies
- **Key**: Geography-aware attention weights

### 2. Flashback (ICDM 2016) - RNN for Location Prediction
- **Problem**: Standard RNNs treat all past locations equally
- **Solution**:
  - Time-specific transition matrices
  - Distance-specific transition matrices  
  - User-specific embeddings
- **Key**: Separate distance/time from sequence modeling

### 3. CARA (2020) - Context-Aware RNN
- **Problem**: Missing context about WHY user visited location
- **Solution**:
  - Category embeddings (POI types)
  - Time-of-day embeddings (morning/afternoon/evening)
  - Day-of-week embeddings
  - Context gates for dynamic weighting
- **Key**: Rich contextual features + gating

### 4. LSTPM (KDD 2020) - Long-Short Term Preference
- **Problem**: Users have both habits (long-term) and novelty (short-term)
- **Solution**:
  - Long-term preference module (user history)
  - Short-term transition module (recent trajectory)
  - Non-local attention for both
  - Geo-dilated RNN for different temporal scales
- **Key**: Separate long/short term modeling

### 5. GeoSAN (AAAI 2020) - Geography-Aware Self-Attention
- **Problem**: Self-attention doesn't understand geography
- **Solution**:
  - Spatial proximity encoding
  - Temporal proximity encoding  
  - Relative positional encoding (distance + direction)
  - Hierarchical grid encoding
- **Key**: Inject geography into attention computation

### 6. STAN (SIGIR 2020) - Spatio-Temporal Attention
- **Problem**: Need to model both spatial and temporal patterns jointly
- **Solution**:
  - Parallel spatial and temporal attention
  - Cross-attention between space and time
  - Local context + global context
- **Key**: Joint spatio-temporal modeling

## Common Patterns Across Successful Models

### Must-Have Features:
1. **Distance/Direction Encoding**: Not just coordinates, but relative position
2. **Time Encoding**: Cyclical encoding for hour/day/week
3. **User Modeling**: Personal preferences, not just global patterns
4. **Multi-Scale**: Both recent (short-term) and historical (long-term)
5. **Geography-Aware Attention**: Spatial distance should influence attention weights

### Critical Techniques:
1. **Separate embeddings** for different feature types
2. **Geography-aware positional encoding** (distance + direction)
3. **Multi-head attention** with spatial/temporal specialization
4. **User embeddings** to capture personal preferences
5. **Time-aware modeling** (cyclical patterns)
6. **Hierarchical encoding** for multi-scale patterns

### What DOESN'T Work:
❌ Treating locations as independent tokens
❌ Ignoring spatial relationships
❌ Single-scale temporal modeling
❌ User-agnostic models
❌ Linear time encoding (should be cyclical)

## Specific Innovations:

### 1. Geography-Aware Attention
```
attention_score = QK^T / sqrt(d) - alpha * distance(i,j)
```

### 2. Relative Positional Encoding
Instead of absolute position, encode:
- Distance between locations
- Direction (bearing angle)
- Time difference

### 3. Multi-Scale Temporal
- Recent trajectory (last N)
- Session patterns (same day)
- Historical habits (all history)

### 4. User Preference Modeling
- User embedding (global preference)
- User-location interaction matrix
- User-time interaction (when they visit places)

## Application to Our Problem

### What We're Missing:
1. ❌ No geographical distance encoding
2. ❌ No directional/bearing information
3. ❌ No proper cyclical time encoding
4. ❌ Not separating short/long term patterns
5. ❌ Not using geography-aware attention
6. ❌ No user preference modeling beyond embedding

### What We Need to Add:
1. ✅ Compute distances between S2 cells
2. ✅ Add directional encoding (bearing)
3. ✅ Cyclical time encoding (sin/cos)
4. ✅ Separate recent vs historical modeling
5. ✅ Geography-aware attention weights
6. ✅ User-specific preference learning
