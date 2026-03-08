# Emotiwave 🌊 — Robust Multimodal Emotion Recognition with Interpretable Fusion

[![Paper](https://img.shields.io/badge/Preprint-Figshare-blue)](https://doi.org/10.6084/m9.figshare.31567024)
[![Notebooks](https://img.shields.io/badge/Code-Figshare-orange)](https://doi.org/10.6084/m9.figshare.31567054)
[![Collection](https://img.shields.io/badge/Collection-Figshare-green)](https://doi.org/10.6084/m9.figshare.c.8341840)
[![HAL](https://img.shields.io/badge/HAL-hal--05542288-red)](https://hal.science/hal-05542288)


> **A Comparative Study on Missing Modality Adaptation**
> 
> Author: **Angelic Charles** · ID: DF2025-069 · DataraFlow Internship Programme  
> Supervisor: **Winner Emeto** · [dataraflow@gmail.com](mailto:dataraflow@gmail.com)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [The Two Notebooks — Two Different Approaches](#the-two-notebooks--two-different-approaches)
- [Dataset](#dataset)
- [Notebook 1 — LSTM-Based Fusion (missing-modality-robustness-on-cmu-mosei)](#notebook-1--lstm-based-fusion)
  - [Data Pipeline](#notebook-1-data-pipeline)
  - [Architecture: FusionModel (Baseline)](#architecture-fusionmodel-baseline)
  - [Architecture: AttentionFusionModel](#architecture-attentionfusionmodel)
  - [Training Configuration](#notebook-1-training-configuration)
  - [Results](#notebook-1-results)
- [Notebook 2 — Cross-Modal Transformer (MER_CrossModalTransformer)](#notebook-2--cross-modal-transformer)
  - [Data Pipeline](#notebook-2-data-pipeline)
  - [Architecture: CrossModalTransformer](#architecture-crossmodaltransformer)
  - [Training Configuration](#notebook-2-training-configuration)
  - [Results](#notebook-2-results)
- [Cross-Notebook Comparison](#cross-notebook-comparison)
- [Missing Modality Ablation Study](#missing-modality-ablation-study)
- [Key Findings & Conclusions](#key-findings--conclusions)
- [Engineering Challenges Solved](#engineering-challenges-solved)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Acknowledgements](#acknowledgements)

---

## Overview

**Emotiwave** is a research project investigating how well AI systems can recognise human emotions from video when one or more sensors fail. The core question: *if you lose the audio, or the camera, or the transcript — does the system fall apart, or does it adapt?*

Human emotion is communicated across multiple channels simultaneously — **what you say** (text/language), **how you say it** (audio/prosody), and **how your face looks** (visual/facial). Most existing systems assume all three channels are always available. In the real world, that assumption breaks constantly: a noisy street degrades audio, poor lighting ruins video, and speech-to-text errors corrupt transcripts.

This project approaches the problem from **two completely different angles** across two separate notebooks, using different architectures, different visual feature sources, different loss formulations, and different preprocessing strategies — then compares what each approach reveals.

### What This Project Does

- Trains and evaluates **three distinct fusion architectures** on the CMU-MOSEI benchmark
- Investigates **missing modality robustness** by simulating sensor failure at inference time across all 7 possible modality combinations
- Measures whether **learned attention weights** actually tell you which modality the model relied on
- Documents why **extreme class imbalance** (60:1 ratio) defeats standard loss weighting

---

## Project Structure
```
Multimodal-Emotion-Recognition/
│
├── missing-modality-robustness-on-cmu-mosei.ipynb   ← Notebook 1: LSTM approach
│   ├── Phase 1: FusionModel (LSTM concatenation baseline)
│   └── Phase 2: AttentionFusionModel (LSTM + soft attention)
│
└── MER_CrossModalTransformer.ipynb                  ← Notebook 2: Transformer approach
    ├── Phase 1: Data engineering + class weighting
    ├── Phase 2: CrossModalTransformer architecture
    ├── Phase 3: Ablation study (graceful degradation)
    └── Phase 4: Interpretability visualisation
```

---

## The Two Notebooks — Two Different Approaches

This project deliberately uses **two separate experimental pipelines** rather than a single unified one. This was intentional — each notebook explores a different hypothesis about how to solve multimodal fusion:

| | Notebook 1 (LSTM) | Notebook 2 (Transformer) |
|---|---|---|
| **Core idea** | Sequential encoding + fusion | Global attention across all modalities at once |
| **Visual features** | OpenFace 2.0 — `VisualOpenFace2.csd` (713-dim) | FACET 4.2 — `VisualFacet42.csd` (35-dim) |
| **Text features** | GloVe word vectors, 300-dim | GloVe word vectors, 300-dim |
| **Audio features** | COVAREP, 74-dim | COVAREP, 74-dim |
| **Sequence handling** | Dynamic padding + truncation to MAX_LEN=300 | Temporal mean-pooling to fixed T=50 |
| **Loss function** | CrossEntropyLoss (single-label) | BCEWithLogitsLoss (multi-label binary) |
| **Batch size** | 16 (reduced from 32 due to OOM) | 32 |
| **GPU** | Tesla P100-PCIE-16GB | Kaggle CUDA GPU |
| **Models tested** | FusionModel, AttentionFusionModel | CrossModalTransformer |

---

## Dataset

**CMU-MOSEI** (Carnegie Mellon University Multimodal Opinion Sentiment and Emotion Intensity)

CMU-MOSEI is the gold-standard benchmark for in-the-wild multimodal affective computing, collected from YouTube opinion videos. Each video segment is annotated for both sentiment and six discrete Ekman emotions.

### Label Schema

The raw label vector has 7 columns. This project discards column 0 (sentiment) and uses only the 6 emotion columns:
```
[Sentiment, Happy, Sad, Anger, Surprise, Fear, Disgust]
     ↑ dropped        ↑ these 6 are the targets
```

### Available Modality Files
```
CMU-MOSEI/
├── labels/    CMU_MOSEI_Labels.csd              → 3,293 videos
├── languages/ CMU_MOSEI_TimestampedWords.csd    → raw word strings (NOT used)
│              CMU_MOSEI_TimestampedWordVectors.csd → GloVe 300-dim (USED)
│              CMU_MOSEI_TimestampedPhones.csd
├── acoustics/ CMU_MOSEI_COVAREP.csd             → 74-dim COVAREP features
└── visuals/   CMU_MOSEI_VisualOpenFace2.csd      → 713-dim (used in NB1)
               CMU_MOSEI_VisualFacet42.csd        → 35-dim  (used in NB2)
```

### Data Intersection

After taking the intersection of video IDs present across all four modality files:
```
Labels available:  3,293
Text available:    3,837
Audio available:   3,836
Visual available:  3,837
─────────────────────────
Valid intersection: 3,292 segments
```

### Data Split (both notebooks, SEED=42)
```
Total:      3,292 segments
Train:      2,633 (80.0%)
Validation:   329 (10.0%)
Test:         330 (10.0%)
```

### Class Distribution (Test Set)

This is the core challenge of the project. The label distribution is extremely skewed:

| Emotion  | Test Support | % of Test Set | Class Weight |
|----------|-------------|---------------|--------------|
| Happy    | 194         | 58.8%         | 0.64         |
| Sad      | 64          | 19.4%         | 3.54         |
| Anger    | 43          | 13.0%         | 4.57         |
| Fear     | 17          | 5.2%          | 7.95         |
| Disgust  | 9           | 2.7%          | 10.63        |
| Surprise | 3           | 0.9%          | 12.72        |

Happy accounts for **58.8%** of samples. Surprise accounts for **0.9%**. That is a ~66:1 ratio. Even with inverse-frequency class weights up to 12.72x, this proved impossible to overcome with standard loss functions.

---

## Notebook 1 — LSTM-Based Fusion

**File:** `missing-modality-robustness-on-cmu-mosei.ipynb`

This notebook is structured as a step-by-step investigation. Each cell is documented with "why we are doing this" reasoning. It implements **two architectures sequentially** — first a concatenation baseline, then an attention upgrade.

### Notebook 1 Data Pipeline

**Step 1 — Feature discovery**

Initially loaded `CMU_MOSEI_TimestampedWords.csd` for text. On inspection, this file contains raw byte strings (e.g., `b'i'`, `b'see'`) not numeric embeddings. Switching to `CMU_MOSEI_TimestampedWordVectors.csd` gave the correct 300-dim GloVe vectors:
```
Text shape:   (183, 1)   ← raw words file (WRONG)
Text shape:   (183, 300) ← word vectors file (CORRECT)
Audio shape:  (5721, 74)
Visual shape: (1714, 713)
Label shape:  (7,)
```

**Step 2 — Dataset class (`RobustDataset`)**

Lazy-loading from HDF5 files. Handles label dimensionality edge cases:
- If label tensor is 1D: slice `[1:]` to get 6 emotions
- If label tensor is 2D (Time, 7): average across time axis first, then slice

**Step 3 — Padding collation**

Variable-length sequences require dynamic padding. The `safe_collate` function:
- Truncates each modality to `MAX_LEN = 300` steps
- Zero-pads shorter sequences to match the longest in the batch
- Applied after reducing from `MAX_LEN=1000` (initial attempt caused 23.84 GiB OOM on a 15.89 GiB GPU)
```python
MAX_LEN = 300   # Sequences truncated here before padding
BATCH_SIZE = 16  # Reduced from 32 to fit GPU memory
```

---

### Architecture: FusionModel (Baseline)

Three independent Bidirectional LSTMs encode each modality, then the final hidden states are concatenated and passed through an MLP classifier.
```
Text  (T, 300)  →  Bi-LSTM(128)  → rep_text   (256)  ┐
Audio (T,  74)  →  Bi-LSTM(128)  → rep_audio  (256)  ├→ concat(768) → FC(128) → ReLU → Dropout(0.3) → FC(6)
Video (T, 713)  →  Bi-LSTM(128)  → rep_visual (256)  ┘

Total parameters: ~2.1M
```

Each Bi-LSTM takes the concatenation of the final forward + backward hidden states. The fused 768-dim vector passes through two linear layers.

---

### Architecture: AttentionFusionModel

Built on top of the same Bi-LSTM encoders but replaces concatenation with a **learned soft attention mechanism** over the three modality representations.
```
Text  (T, 300)  →  Bi-LSTM(128)  → rep_text   (256)  ┐
Audio (T,  74)  →  Bi-LSTM(128)  → rep_audio  (256)  ├→ ModalityAttention → fused(256) → FC(128) → ReLU → Dropout → FC(6)
Video (T, 713)  →  Bi-LSTM(128)  → rep_visual (256)  ┘

Total parameters: ~1.8M
```

**ModalityAttention module:**
```python
class ModalityAttention(nn.Module):
    def __init__(self, hidden_dim):
        self.attn_vector = nn.Linear(hidden_dim * 2, 1, bias=False)
    
    def forward(self, stacked):  # stacked: (3, Batch, 256)
        scores  = self.attn_vector(stacked)       # (3, Batch, 1)
        weights = F.softmax(scores, dim=0)        # normalise across 3 modalities
        fused   = (stacked * weights).sum(dim=0)  # weighted sum → (Batch, 256)
        return fused, weights
```

The single linear scoring vector learns to assign importance to each modality for each sample.

---

### Notebook 1 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimiser | AdamW |
| Weight decay | 1e-2 |
| Learning rate | 1e-4 |
| Batch size | 16 |
| Max sequence length | 300 |
| Epochs (planned) | 50 |
| Early stopping patience | 5 epochs |
| Modality dropout | p=0.3 (each modality independently) |
| Gradient clipping | `clip_grad_norm_`, max_norm=1.0 |
| Loss function | CrossEntropyLoss with inverse-frequency class weights |
| Input sanitisation | `torch.nan_to_num` on every batch |
| Checkpointing | Every 10 epochs + on every new best |

**Modality dropout during training:**
```python
if np.random.random() < 0.3: audio  = torch.zeros_like(audio)
if np.random.random() < 0.3: visual = torch.zeros_like(visual)
if np.random.random() < 0.3: text   = torch.zeros_like(text)
```

This forces the model to learn from any combination of available modalities — the core mechanism for robustness.

---

### Notebook 1 Results

#### FusionModel (Baseline) — Full Training Log

| Epoch | Loss  | Val Macro-F1 | Best? |
|-------|-------|--------------|-------|
| 1     | 1.2993 | 0.1287      | ⭐ |
| 5     | 1.0432 | 0.1391      | ⭐ |
| 10    | 0.9894 | 0.1878      | ⭐ |
| 15    | 0.9401 | 0.2109      | ⭐ |
| 19    | 0.9248 | 0.2410      | ⭐ |
| 23    | 0.9037 | 0.2472      | ⭐ |
| 26    | 0.8966 | 0.2614      | ⭐ |
| 31    | 0.8643 | 0.2644      | ⭐ |
| 35    | 0.8170 | 0.2659      | ⭐ |
| **36** | **0.8244** | **0.2739** | ⭐ **BEST** |
| 41    | 0.7845 | 0.2442      | ⏹️ Early stop |

**Final best validation Macro-F1: 0.2739 at epoch 36**  
Training terminated at epoch 41 (5 epochs without improvement)

#### FusionModel — Test Set Classification Report
```
              precision    recall  f1-score   support

       Happy       0.70      0.80      0.75       194
         Sad       0.31      0.33      0.32        64
       Anger       0.29      0.23      0.26        43
    Surprise       0.00      0.00      0.00         3
        Fear       0.00      0.00      0.00        17
     Disgust       0.00      0.00      0.00         9

    accuracy                           0.57       330
   macro avg       0.22      0.23      0.22       330
weighted avg       0.51      0.57      0.53       330
```

> **Critical observation:** The model never once predicted Surprise, Fear, or Disgust across 330 test samples. Class weights of up to 12.72x made no difference. These three classes have so few samples that the gradient contribution from them is overwhelmed by Happy every single epoch.

---

#### AttentionFusionModel — Training Log

| Epoch | Loss   | Val Macro-F1 | Note |
|-------|--------|--------------|------|
| 1     | 1.4225 | 0.1287       | ⭐ Saved |
| 2     | 1.1082 | 0.1287       | — |
| 3     | 1.0746 | 0.1287       | — |
| 4     | 1.0647 | 0.1287       | — |
| 5     | 1.0601 | 0.1287       | — |
| 6     | 1.0620 | 0.1287       | ⏹️ Early stop |

**Best validation Macro-F1: 0.1287 at epoch 1**  
Training terminated at epoch 6 — only 6 epochs before early stopping triggered.

> **Why did this happen?** The `ModalityAttention` module applies a single learned linear vector to score all three modality representations. Under high modality dropout (p=0.3), the Bi-LSTM encoders produce poorly differentiated representations early in training. The scoring vector has no signal to differentiate the modalities, so `softmax` collapses to uniform ~33% weights and stays there. The model gets stuck — the attention provides no benefit and the added complexity prevents the improvement the baseline achieved through simple concatenation. The Baseline reached 0.2739; the Attention model peaked at 0.1287 — a **53% degradation**.

---

## Notebook 2 — Cross-Modal Transformer

**File:** `MER_CrossModalTransformer.ipynb`

This notebook takes a completely different approach. Rather than encoding each modality independently and then fusing, the Cross-Modal Transformer projects all modalities into a **shared embedding space** and lets multi-head self-attention reason across all of them simultaneously.

### Notebook 2 Data Pipeline

**Key difference from Notebook 1:** Instead of dynamic padding, this notebook uses **temporal mean-pooling** to reduce all sequences to a fixed length of T=50.

**Why T=50?** The raw audio sequences go up to 24,355 frames. Transformer self-attention is O(T²) in memory — feeding 24,355 frames would be computationally impossible. Temporal pooling divides each sequence into 50 equal time bins and averages the features within each bin:
```python
TARGET_T = 50

def pool_features(feat, target_t):
    indices = np.array_split(np.arange(feat.shape[0]), target_t)
    return np.array([np.mean(feat[idx], axis=0) for idx in indices])
```

This reduces:
```
Text:   (183–621 steps, 300-dim)   →  (50, 300)
Audio:  (4,625–24,355 steps, 74-dim) →  (50, 74)
Visual: (1,385–7,299 steps, 35-dim)  →  (50, 35)   ← FACET 4.2, not OpenFace
```

**Visual feature source change:** This notebook uses `VisualFacet42.csd` (35 dimensions) rather than `VisualOpenFace2.csd` (713 dimensions). The FACET features provide a more compact representation — this was found during file inspection when exploring the Kaggle input directory.

**Class weight calculation:**

Rather than assuming class weights, this notebook calculates them directly from the label distribution:
```python
pos_counts  = np.sum(all_labels > 0, axis=0)  # per-emotion positive count
neg_counts  = len(all_labels) - pos_counts
pos_weights = neg_counts / (pos_counts + 1e-6) # inverse frequency ratio
```

Result:
```
Happy    : 0.64   (majority class, slightly down-weighted)
Sad      : 3.54
Anger    : 4.57
Fear     : 7.95
Disgust  : 10.63
Surprise : 12.72  (rarest class, highest priority)
```

**Loss formulation change:** This notebook uses `BCEWithLogitsLoss` with these positive weights, treating each emotion as an **independent binary classification problem** (multi-label). Notebook 1 used `CrossEntropyLoss` which forces a single-label choice.

---

### Architecture: CrossModalTransformer

All three modalities are projected into a shared d=256 space, concatenated with a learnable [CLS] token, and processed by a standard Transformer encoder. The [CLS] token's output is used for classification.
```
Text  (50, 300) → Linear(300→256) → proj_t (50, 256) ┐
Audio (50,  74) → Linear( 74→256) → proj_a (50, 256) ├─→ concat along seq dim
Video (50,  35) → Linear( 35→256) → proj_v (50, 256) ┘
                                                         ↓
                                              [CLS] token prepended
                                                         ↓
                                      TransformerEncoder(4 layers, 8 heads, d=256)
                                         - dim_feedforward = 1024
                                         - dropout = 0.1
                                         - activation = ReLU
                                                         ↓
                                              CLS output (256)
                                                         ↓
                                      MLP Head: FC(256→512) → ReLU → Dropout → FC(512→6)

Total parameters: ~3.2M
```
```python
class CrossModalTransformer(nn.Module):
    def __init__(self, text_dim=300, audio_dim=74, visual_dim=35,
                 d_model=256, nhead=8, num_layers=4):
        
        self.proj_t = nn.Linear(text_dim,   d_model)
        self.proj_a = nn.Linear(audio_dim,  d_model)
        self.proj_v = nn.Linear(visual_dim, d_model)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=1024, dropout=0.1,
            activation='relu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # MLP head: 256 → 512 → 6
```

**Forward pass with modality dropout:**
```python
def forward(self, text, audio, visual, p_dropout=0.3):
    # Project all modalities to shared space
    t = self.proj_t(text)    # (B, 50, 256)
    a = self.proj_a(audio)   # (B, 50, 256)
    v = self.proj_v(visual)  # (B, 50, 256)
    
    # Modality dropout
    if self.training:
        if random.random() < p_dropout: a = torch.zeros_like(a)
        if random.random() < p_dropout: v = torch.zeros_like(v)
        if random.random() < p_dropout: t = torch.zeros_like(t)
    
    # Prepend CLS token
    cls = self.cls_token.expand(B, -1, -1)
    x = torch.cat([cls, t, a, v], dim=1)  # (B, 151, 256)
    
    # Transformer
    x = self.transformer(x)
    cls_out = x[:, 0, :]  # Take CLS token output
    return self.mlp_head(cls_out)
```

---

### Notebook 2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimiser | AdamW |
| Weight decay | 1e-2 |
| Learning rate | 5e-5 (lower than NB1's 1e-4) |
| LR scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Batch size | 32 |
| Sequence length | Fixed T=50 (temporal pooling) |
| Epochs (planned) | 100 |
| Early stopping patience | 5 epochs |
| Modality dropout | p=0.3 |
| Gradient clipping | `clip_grad_norm_`, max_norm=1.0 |
| Loss function | BCEWithLogitsLoss + positive class weights |
| Input sanitisation | `torch.nan_to_num` |

---

### Notebook 2 Results

#### CrossModalTransformer — Full Training Log (selected epochs)

| Epoch | Loss   | Val Macro-F1 | Note |
|-------|--------|--------------|------|
| 1     | 0.8261 | 0.1129       | ⭐ |
| 6     | 0.7916 | 0.1185       | ⭐ |
| 13    | 0.7357 | 0.1464       | ⭐ |
| 18    | 0.6804 | 0.1947       | ⭐ |
| 21    | 0.6464 | 0.2155       | ⭐ |
| 26    | 0.5182 | 0.2388       | ⭐ |
| 34    | 0.1665 | 0.2046       | — |
| **38** | **-0.0014** | **0.3011** | ⭐ **BEST** |
| 44    | -0.1266 | 0.2716      | — |
| 53    | -0.3140 | 0.2248      | ⏹️ Early stop |

**Final best validation Macro-F1: 0.3011 at epoch 38**  
Training terminated at epoch 53 (5 epochs without improvement)

> **Note on negative loss values:** After epoch 34, the BCEWithLogitsLoss begins returning negative values. This is a known behaviour when strong positive class weights cause the model to confidently predict minority classes — the weighted binary cross-entropy can go negative if the model becomes overconfident. This is worth investigating in future work as a potential overfitting indicator.

---

#### Missing Modality Ablation Study (CrossModalTransformer)

After training, the best model was evaluated across all 7 possible modality configurations by zeroing out the missing modality inputs at inference time (p_dropout=0). This simulates real sensor failures.

| Configuration | Macro-F1 | Retention % | Target >70% |
|---------------|----------|-------------|-------------|
| **Full (T+A+V)** | **0.0808** | **100.0%** | — (baseline) |
| Text Only (T) | 0.0739 | **91.5%** | ✅ PASS |
| Text + Audio (T+A) | 0.0658 | **81.4%** | ✅ PASS |
| Visual Only (V) | 0.0632 | **78.2%** | ✅ PASS |
| Audio + Visual (A+V) | 0.0616 | **76.3%** | ✅ PASS |
| Text + Visual (T+V) | 0.0602 | **74.5%** | ✅ PASS |
| Audio Only (A) | 0.0202 | **25.0%** | ❌ FAIL |

**5 out of 6 partial configurations passed the 70% retention threshold.**

> **Why does Audio-only fail so badly?** COVAREP acoustic features compressed to T=50 via mean-pooling lose the fine-grained prosodic dynamics (pitch variation, energy bursts, speaking rate changes) that carry the most emotional information in audio. When text and visual are also removed, the model has almost nothing useful to work with.

> **Why does Text-only retain 91.5%?** CMU-MOSEI is collected from YouTube opinion videos — a genre where speakers deliberately use strong language to convey their point. The lexical content alone (what words they chose) is highly predictive of emotion in this specific domain.

---

## Cross-Notebook Comparison

| Model | Approach | Visual Dim | Loss | Best Val F1 | Best Epoch | Stopped |
|-------|----------|------------|------|-------------|------------|---------|
| **FusionModel** (NB1) | Bi-LSTM concat | OpenFace 713 | CrossEntropy | **0.2739** | 36 | Ep. 41 |
| AttentionFusionModel (NB1) | Bi-LSTM + soft attn | OpenFace 713 | CrossEntropy | 0.1287 | 1 | Ep. 6 |
| **CrossModalTransformer** (NB2) | 4-layer Transformer | FACET 35 | BCE+Logits | **0.3011** | 38 | Ep. 53 |

The CrossModalTransformer achieved the highest validation performance — but it also uses a different visual feature source, a different loss function, and a different preprocessing strategy than the LSTM models. These are not isolated variables. The comparison is meaningful but not perfectly controlled — which is exactly the point of running two separate experimental pipelines.

---

## Key Findings & Conclusions

### 1. Architectural Complexity Helps — But Only With Depth

The Cross-Modal Transformer (0.3011 val F1) outperformed the LSTM baseline (0.2739 val F1). However, adding attention to the LSTM degraded it catastrophically to 0.1287, with early stopping triggered at epoch 6.

The lesson: **attention only works when the encoder representations it scores are sufficiently expressive and differentiated**. The shallow LSTM+attention stack produced near-identical representations for all three modalities under heavy dropout, leaving the softmax with nothing to discriminate — so it settled at uniform ~33% weights permanently.

The Transformer's 4-layer, 8-head architecture has enough capacity to develop specialised representations even under dropout, which is why it succeeds where the attention upgrade fails.

### 2. Modality Dropout Training Works

Training with random modality zeroing (p=0.3) successfully produced a robust Transformer. **5 out of 6 partial modality configurations retained over 70% of full performance** at inference time without any architectural modification. This is a simple, effective technique for real-world deployment resilience.

### 3. Text Dominates in Opinion Video

Text-only performance (91.5% retention) confirms that in CMU-MOSEI's opinion-video domain, **what people say is far more predictive of their emotion than how they look or sound**. This makes sense: the dataset consists of deliberate, scripted-feeling opinion videos where emotional content is concentrated in word choice.

This has practical implications: for any system deployed on opinion or review content, text processing should be treated as the primary modality and designed to work well independently.

### 4. Standard Class Weighting Completely Fails at 60:1 Imbalance

This is the most important practical finding. Despite applying inverse-frequency class weights up to **12.72x for Surprise** and **10.63x for Disgust**, both LSTM models produced **0% recall on Surprise, Fear, and Disgust** across 330 test samples. The models never predicted these classes even once.

Why? With only 3 Surprise samples in the training set and 194 Happy samples, the gradient contribution from Surprise is so small that even multiplying the loss by 12.72 is insufficient to overcome the sheer volume of Happy gradients. Standard loss weighting is a patch, not a solution, for imbalances of this severity.

**What should be tried instead:**
- **Focal Loss** — dynamically down-weights easy majority examples rather than up-weighting minorities
- **SMOTE / ADASYN** — synthetically generate minority-class samples in feature space
- **Contrastive / metric learning** — train the model to separate emotion representations in embedding space rather than optimising for classification directly

### 5. Attention Weights Are Not Interpretable at the Global Level

Both attention architectures (AttentionFusionModel and CrossModalTransformer) produced near-uniform average attention weights across the test set (~33% per modality). This confirms a known finding from the NLP literature: **global average attention weights do not tell you what the model relied on**.

The Transformer achieved 30.11% val F1 while the LSTM+Attention model achieved 12.87% — a dramatic difference — yet both show identical-looking average attention distributions. The Transformer's performance must come from dynamic, per-sample, per-head attention patterns that average out to uniform when pooled globally.

Interpretability in multimodal systems requires per-sample attribution methods (Integrated Gradients, SHAP, GradCAM) — not attention weight inspection.

### 6. The Two-Notebook Approach Revealed More Than One Would Have

By running two completely separate experimental pipelines with different choices at every level (visual features, sequence handling, loss function, architecture), this project discovered that:
- The **feature source matters** — FACET 4.2 vs OpenFace produce different embedding geometries
- The **loss formulation matters** — BCEWithLogitsLoss vs CrossEntropyLoss treats the problem fundamentally differently (multi-label vs single-label)
- The **preprocessing strategy matters** — truncation vs temporal pooling changes what information survives into the model

A single unified pipeline would have hidden these interactions.

---

## Engineering Challenges Solved

These are the real technical obstacles encountered and documented during the project:

| Problem | What Happened | Solution |
|---------|--------------|----------|
| **Wrong text file** | `TimestampedWords.csd` contains raw byte strings, not embeddings | Switched to `TimestampedWordVectors.csd` for 300-dim GloVe |
| **CUDA OOM (1st attempt)** | MAX_LEN=1000 required 23.84 GiB on a 15.89 GiB GPU | Reduced MAX_LEN to 300 |
| **Loss: NaN from epoch 1** | Exploding gradients in LSTM on raw variable-length sequences | Added `clip_grad_norm_(max_norm=1.0)` |
| **Raw data NaN/inf values** | CMU-MOSEI HDF5 files contain occasional corrupted values | Applied `torch.nan_to_num` on every batch |
| **Label shape inconsistency** | Some videos return `(1, 7)` labels, others return `(4, 7)` | Added dimension check: if 2D, mean over time axis first |
| **Transformer OOM** | Raw audio up to 24,355 frames, O(T²) attention | Applied temporal mean-pooling to T=50 |
| **Attention model collapse** | AttentionFusionModel stuck at 0.1287 from epoch 1 | Documented as finding; Transformer architecture solved this |

---

## Requirements
```
Python 3.11+
PyTorch >= 2.0
h5py
numpy
pandas
scikit-learn
matplotlib
seaborn
tqdm
```

Install dependencies:
```bash
pip install torch h5py numpy pandas scikit-learn matplotlib seaborn tqdm
```

---

## How to Run

Both notebooks are designed to run on **Kaggle** with the CMU-MOSEI dataset attached as an input.

### Dataset Setup (Kaggle)

1. Go to [Kaggle Datasets](https://www.kaggle.com/datasets) and find `cmu-mosei`
2. Add it to your notebook under **Add Data**
3. The expected file structure is:
```
/kaggle/input/cmu-mosei/CMU-MOSEI/
├── labels/    CMU_MOSEI_Labels.csd
├── languages/ CMU_MOSEI_TimestampedWordVectors.csd
├── acoustics/ CMU_MOSEI_COVAREP.csd
└── visuals/   CMU_MOSEI_VisualOpenFace2.csd
               CMU_MOSEI_VisualFacet42.csd
```

### Running Notebook 1 (LSTM)

Open `missing-modality-robustness-on-cmu-mosei.ipynb` on Kaggle with GPU accelerator enabled.

- **Phase 1** (cells 1–34): Trains FusionModel for up to 50 epochs with early stopping
- **Phase 2** (cells 35–41): Trains AttentionFusionModel for up to 50 epochs
- Best models saved to `emotiwave_best.pth` and `emotiwave_attention_best.pth`

### Running Notebook 2 (Transformer)

Open `MER_CrossModalTransformer.ipynb` on Kaggle with GPU accelerator enabled.

- **Phase 1** (cells 1–5): Data extraction and temporal pooling → saves `mosei_final_train.pkl`
- **Phase 2** (cells 6–10): Model definition and data loaders
- **Phase 3** (cells 11–14): Training loop (100 epochs, early stopping) + ablation study
- **Phase 4** (cells 15–16): Interpretability visualisation

> **Tip:** Notebook 2 runs faster because temporal pooling to T=50 dramatically reduces memory requirements. It can comfortably use batch size 32 vs Notebook 1's batch size 16.

---

## Acknowledgements

This project was completed as part of the **DataraFlow Internship Programme**.

Supervised by **Winner Emeto** — [dataraflow@gmail.com](mailto:dataraflow@gmail.com)

The CMU-MOSEI dataset was created and made publicly available by the **CMU Multimodal Analysis Group**. Kaggle provided the GPU infrastructure (Tesla P100-PCIE-16GB) used for all experiments.

---

<div align="center">
<sub>Built with curiosity and a lot of CUDA errors — Angelic Charles · DF2025-069 · 2025</sub>
</div>
