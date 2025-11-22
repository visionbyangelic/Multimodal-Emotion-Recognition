## Robust Multimodal Emotion Recognition with Interpretable Fusion

This repository contains the official implementation for the paper: "**Robust Multimodal Emotion Recognition with Interpretable Fusion: A Comparative Study on Missing Modality Adaption**" by Angelic Charles.

The core contribution is a novel **Multimodal Fusion Architecture** engineered for stability and performance using available features in resource-constrained environments.

-----

## 💡 Key Contributions

  * [cite\_start]**Missing Modality Robustness:** The system uses **Modality Dropout** ($p=0.3$) to simulate real-world data loss, forcing the network to maintain predictive performance with incomplete inputs[cite: 13].
    
  * **High Stability:** Implementation includes specific engineering controls (Gradient Clipping and Input Sanitization) to eliminate common `NaN` loss and `CUDA OOM` errors during training.
    
  * **Interpretable Fusion:** The framework is designed to be upgraded to an **Attention-based Layer** to quantify the specific influence of Text, Audio, and Visual features on the final emotion prediction.
    
  * [cite\_start]**Evaluation:** The system is evaluated using the **Macro-Averaged F1-Score** on 3,292 verified segments of the CMU-MOSEI dataset, ensuring unbiased results[cite: 10].

-----

## 🛠️ Operational Methodology & Architecture

### 1\. Sequence Encoders (The Input Reality)

The system is built upon **Bi-directional LSTM encoders** to process the time-series nature of the pre-extracted features available in the dataset.

| Modality | Input Feature | Feature Dimension | Sequence Encoder | Time Truncation |
| :--- | :--- | :--- | :--- | :--- |
| **Visual (V)** | OpenFace Features | 713 | Bi-LSTM (Hidden 128) | 300 Steps (Max Length) |
| **Audio (A)** | COVAREP Features | 74 | Bi-LSTM (Hidden 128) | 300 Steps (Max Length) |
| **Text (T)** | GloVe Word Vectors | 300 | Bi-LSTM (Hidden 128) | 300 Steps (Max Length) |

### 2\. Fusion and Classification

  * [cite\_start]**Fusion Layer:** The final hidden states of the three LSTMs are **Concatenated** (joined side-by-side) into a 768-dimensional vector (Baseline Fusion)[cite: 13].
  * **Classification:** The fused vector is passed through a two-layer Multi-Layer Perceptron (MLP) to predict the 6 discrete emotion classes.

### 3\. Stability Engineering (The Fixes)

  * [cite\_start]**Truncation:** The maximum sequence length is set to **300 steps** to prevent `CUDA Out of Memory` (OOM) errors during padding and LSTM computation[cite: 15].
  * [cite\_start]**Gradient Clipping:** `torch.nn.utils.clip_grad_norm_` is implemented in the training loop to prevent the **Exploding Gradient Problem** and maintain stability[cite: 13].
  * **Checkpointing:** The entire training state (model weights, optimizer history, epoch number) is saved to a checkpoint file every 10 epochs to allow for seamless recovery from session disconnects.

-----

## 📊 Dataset and Execution

### Dataset

  * **Name:** Carnegie Mellon University Multimodal Opinion Sentiment and Emotion Intensity (**CMU-MOSEI**).
  * **Clean Scale:** 3,292 verified segments used for training and evaluation.
  * **Goal:** Six-category discrete emotion classification (happiness, sadness, anger, surprise, fear, disgust).

### Execution Instructions

1.  **Run Setup Cells:** Re-run all initial cells (Imports, Data Configuration, Splitting) to load `train_ids`, `val_ids`, and `test_ids` into memory.
2.  **Run Training Cell (Code Cell 13):** Execute the long training cell (`NUM_EPOCHS = 50`) to generate the final model weights and checkpoint files.
3.  **Run Evaluation Cells (Code Cells 14 & 15):** Execute the final evaluation script to load the best model and generate the **Classification Report, Confusion Matrix, and F1-Score Bar Chart** for the research paper.

-----

## ⚙️ Hyperparameters

  * **Batch Size:** 16
  * **Optimizer:** AdamW
  * **Learning Rate:** $1 \times 10^{-4}$
  * **Modality Dropout:** $p=0.3$

-----

