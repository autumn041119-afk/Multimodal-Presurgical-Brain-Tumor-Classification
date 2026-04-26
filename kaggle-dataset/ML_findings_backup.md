# ML Experiment Findings Backup

> Last updated: 2026-04-12
> STAT3612 Group 12 — Brain Tumor Classification

---

## 1. Dataset Overview

| Item | Value |
|------|-------|
| Total samples | 2,838 |
| Train / Val / Test | 1,983 / 283 / 572 |
| Classes | 5 |
| Class imbalance | 40:1 (Glioma 1056 vs Pineal 26) |

### Class Distribution (Train+Val)

| Class | Count | Proportion |
|-------|-------|------------|
| Glioma | 1,056 | 46.6% |
| Meningioma | 832 | 36.7% |
| Brain Metastase Tumour | 288 | 12.7% |
| Tumors of the sellar region | 64 | 2.8% |
| Pineal tumour and Choroid plexus tumour | 26 | 1.1% |

---

## 2. Data Sources

### 2.1 Clinical Information (27d)
- Sex (ordinal), Age (median fill + missing flag)
- Signal Intensity patterns (4 modalities × one-hot) — radiologist descriptions of MRI signal
- **Source**: `clinical_information/` CSVs
- **Solo performance**: ~66% Micro-F1

### 2.2 Radiology Reports (raw text)
- Radiologist-written medical text
- Encoded as TF-IDF + SVD or BioClinicalBERT frozen embeddings
- **Solo performance**: ~84-85% Micro-F1 ← **STRONGEST SOURCE**
- **Source**: `original_raw_report/` CSVs

### 2.3 Radiomics (20d raw, 49d enhanced)
- PyRadiomics features: Mean, Entropy, 90th Percentile, GLCM Contrast, GLCM Joint Entropy
- 5 features × 4 MRI modalities (T1, T1c, T2, T2-FLAIR) = 20 raw features
- Cross-modal ratios: T1c/T1, T2F/T2, T1/T2, deltas = 25 additional features
- **Solo performance**: ~51% Micro-F1 ← **≈ random baseline (46.6%)**
- **Source**: `radiomics_info/` CSVs

### 2.4 Deep Image Features (ResNet)
- ResNet-extracted 2048d features per modality
- PCA-reduced to 128d per modality (explains 94.2% variance)
- 4 modalities concatenated = 512d total
- **NOT mean-pooled** — per-modality concatenation preserves complementary information
  - T1/T1c: anatomy + vascularity
  - T2/T2F: edema pattern
- **Solo performance**: ~59% Micro-F1
- **Source**: `image_features/image_features/` `.npy` files

### 2.5 Tumor Location (1d label-encoded, 572d one-hot)
- 636 unique location values
- Label-encoded for embedding; one-hot for tree models
- **Solo performance**: ~65% Micro-F1 (label-encoded), ~75% (one-hot + GradientBoosting)
- **Source**: `clinical_information/` Tumor Location column

---

## 3. Experiments Summary

### 3.1 Radiomics Experiments

#### Raw Radiomics (20d, 5 features × 4 modalities)

| Model | Micro-F1 | Macro-F1 |
|-------|----------|-----------|
| XGBoost | 50.93% | 25.06% |
| LightGBM | 50.40% | 25.07% |
| RF | 52.96% | 24.91% |
| SVM-RBF | 53.22% | 23.93% |

**Per-modality best: T2F** (~46%) > T1 (~44%) > T2 (~42%) > T1c (~41%)

#### Enhanced Radiomics (49d: raw + cross-modal ratios)

| Feature Set | Dimension | Micro-F1 |
|-------------|-----------|----------|
| Raw radiomics | 20d | 51.68% |
| Enhanced (raw+ratios) | 49d | **51.32%** ← worse |
| Image+Rad eng | 561d | 61.17% |

**FINDING**: Radiomics feature engineering provides zero improvement. Even extensive cross-modal ratios, shape proxies, and texture interactions cannot extract discriminative signal beyond the baseline ~52%.

#### Data Augmentation (Gaussian noise, SMOTE, mixup)

| Setting | Micro-F1 |
|---------|----------|
| No augmentation | 52.30% |
| Augment 3x | 48.37% ← worse |
| Augment 5x | 48.90% ← worse |

**FINDING**: Data augmentation hurts radiomics. The features have no signal to augment.

### 3.2 Image Experiments

#### Mean-Pool vs Per-Modality PCA

| Method | Dimension | Micro-F1 |
|--------|-----------|----------|
| Mean-pool (4 modalities → 1) | 128d | 56.97% |
| Per-mod PCA (4 modalities) | 512d | **59.66%** |

**FINDING**: Mean-pooling destroys complementary physical information across MRI sequences. Per-modality concatenation is strictly better.

### 3.3 Report Experiments

#### TF-IDF + SVD Dimension Search

| SVD Dim | Variance Explained | Best Model | Micro-F1 |
|---------|------------------|------------|-----------|
| SVD-32 | 57.5% | LGB-d4-lr05 | 83.94% |
| SVD-64 | 71.1% | LGB-d4-lr05 | **84.42%** |
| SVD-128 | 83.4% | LGB-d4-lr05 | 84.69% |
| SVD-256 | 93.8% | LGB-d4-lr05 | 84.64% |

**FINDING**: SVD-64 to SVD-128 is optimal. More dimensions don't help.

#### BioClinicalBERT Frozen Embeddings + MLP

| Model | Micro-F1 | Macro-F1 |
|-------|----------|-----------|
| MLP-256 | 81.47% | 67.90% |
| MLP-256-128 | 81.82% | 66.83% |
| XGB-d3 | **82.57%** | 69.91% |

**FINDING**: Frozen BERT embeddings (~82%) are slightly worse than TF-IDF+SVD (~84%). Fine-tuning would likely improve BERT, but risks overfitting on 2K samples.

### 3.4 Unified Feature Model (All Sources Combined)

Combined dimension: **1,992d** = clinical(27) + location(572) + report_svd(64) + report_bert(768) + radiomics(49) + image(512)

| Model | Micro-F1 | Macro-F1 |
|-------|----------|-----------|
| Unified XGBoost | 69.37% | — |
| Unified LightGBM | 68.71% | — |
| Unified Ensemble | 68.84% | — |

**FINDING**: Unified model (69%) is far worse than Report alone (84%). Radiomics and Image drag down performance.

### 3.5 Class Weight Experiment

| Source | Without Class Weight | With Class Weight |
|--------|---------------------|------------------|
| Clinical | **66.0%** | 54.4% |
| Radiomics | 50.9% | 48.4% |
| Image | 59.4% | 58.8% |

**FINDING**: Class weights hurt Micro-F1 because Micro-F1 ≈ accuracy, dominated by majority classes. Class weights only help Macro-F1 for minority classes.

---

## 4. Blending Results

### Individual Source Performance

| Source | Micro-F1 | Macro-F1 |
|--------|----------|-----------|
| **Report (TF-IDF+SVD)** | **85.13%** | 73.04% |
| Report BERT | 82.48% | 69.92% |
| Clinical | 66.02% | 39.91% |
| Location | 62.67% | 52.56% |
| Image | 59.31% | 27.38% |
| Image+Rad eng | 61.17% | 29.78% |
| Radiomics | 50.66% | 24.73% |

### Best Blends (OOF Micro-F1)

| Configuration | Micro-F1 | Macro-F1 |
|--------------|----------|-----------|
| Report alone | 85.13% | 73.04% |
| Report + BERT (equal) | 85.57% | 72.82% |
| Report + BERT + Location | 85.75% | 74.72% |
| **image + report + report_bert + location (0.2, 0.3, 0.2, 0.3)** | **86.23%** | 70.14% |
| Unified + all sources | 86.01% | — |

---

## 5. Per-Class Accuracy (Best Blend)

| Class | Accuracy | Samples | Status |
|-------|----------|---------|--------|
| Glioma | 94.98% | 1,056 | ✅ |
| Meningioma | 89.78% | 832 | ✅ |
| Brain Metastase Tumour | **52.78%** | 288 | ❌ |
| Tumors of sellar region | 75.00% | 64 | 中 |
| Pineal tumour and CP tumour | **15.38%** | 26 | ❌ |

**Main error sources**: Brain Metastase (~135 errors) and Pineal (~22 errors) dominate the remaining ~14% misclassification.

---

## 6. Key Findings

### 6.1 Radiomics is Noise
- 5 PyRadiomics features per modality (20 total) carry almost no discriminative signal for tumor type classification
- This is expected: the features (Mean, Entropy, GLCM Contrast) capture global texture statistics but miss the clinically-relevant patterns radiologists use:
  - **Location** (where is the tumor?)
  - **Enhancement pattern** (ring? homogeneous? heterogeneous?)
  - **Edema extent** (surrounding edema distribution)
- Cross-modal ratios (T1c/T1, T2F/T2) are medically meaningful but still insufficient alone
- **Recommendation**: Do not invest more effort in radiomics feature engineering. The raw features have been thoroughly explored.

### 6.2 Report Dominates Everything
- Radiology reports alone reach ~85% Micro-F1
- Adding ALL other sources only improves to 86.23%
- The gap to 90% is NOT about adding more features or models
- The bottleneck is Brain Metastase (~53%) and Pineal (~15%) classification

### 6.3 Image Features Are Weak
- ResNet features (~60%) are much weaker than reports (~85%)
- Per-modality PCA is better than mean-pool, but still limited
- The learned features capture generic visual patterns, not tumor-type-specific discriminative information

### 6.4 Why Clinical > Radiomics?
Radiologists diagnose by:
- **Location** — directly captured by Location feature (~65%)
- **Enhancement** — captured by SI patterns in clinical notes
- **Edema extent** — captured by SI patterns + Location

The 5 PyRadiomics features don't capture these patterns.

### 6.5 Mean-Pooling Destroys Information
- T1/T1c capture anatomy + vascularity
- T2/T2F capture edema pattern
- Mean-pooling mixes these complementary signals into a single vector
- Per-modality concatenation preserves the information

---

## 7. Path to 90%+

### What's NOT the answer
- More radiomics features / engineering ❌
- Data augmentation ❌
- Larger unified models ❌
- Class weights (hurts Micro-F1) ❌
- Adding Image/Radiomics to ensemble ❌

### What's needed
1. **Report-centric**: Fine-tune BERT (not frozen) with careful regularization
2. **Brain Metastase focus**: This class accounts for most remaining errors. Needs targeted strategy.
3. **Post-processing**: Confidence threshold tuning for minority classes

### Projected performance
- Report alone: ~85%
- Best blend (current): ~86.23%
- With fine-tuned BERT + targeted Brain Metastase strategy: ~88-89%
- 90%+ requires either better Brain Metastase/Pineal classification or more data

---

## 8. Code Architecture

### Notebook: ML.ipynb

| Cell | Content |
|------|---------|
| 1 | Colab mount |
| 2 | Imports + library checks |
| 3 | Enhanced features module (inlined) |
| 4 | Data loading + feature extraction |
| 5 | Quick diagnostic (RF baseline) |
| 6 | `cv_evaluate()` function |
| 7 | `get_models_*()` — minimal model set (10 models) |
| 9-11 | Radiomics experiments |
| 13 | Clinical + feature engineering (Location OHE) |
| 14 | GradientBoosting tuning |
| 15 | Clinical experiments |
| 18 | Image + Radiomics combined experiments |
| 19 | Image-only baseline |
| 21 | Report TF-IDF+SVD experiments |
| 23 | Report BERT embeddings |
| 25 | Location experiments |
| 27 | Class weight experiment |
| 29 | Enhanced features + Unified XGB/LGB |
| 30-31 | Cross-source blending |
| 32 | All results summary |
| 34 | Final model training + submission |

### Files

| File | Purpose |
|------|---------|
| `data_loader.py` | Data loading + PCA + preprocessing |
| `enhanced_features.py` | Enhanced radiomics + image statistics (inlined into notebook) |
| `ML.ipynb` | Main experiment notebook |
| `model_training (1).ipynb` | Deep learning baseline (separate) |

### Model Count (Minimal)

| Source | Models |
|--------|--------|
| Radiomics | 2 (XGB, LGB) |
| Clinical | 2 (XGB, LGB) |
| Image | 2 (XGB, LGB) |
| Report | 4 (SVM×2, XGB, LGB) |
| Location | 2 (RF, XGB) |
| **Total** | **12** (down from 60+) |
