# STAT3612 Brain Tumor Classification - Project Notes

> Last updated: 2026-04-13
> Team: Group 12

---

## 1. Dataset Overview

| Item | Value |
|------|-------|
| Total samples | 2,838 |
| Train / Val / Test | 1,983 / 283 / 572 |
| Classes | 5 |
| Class imbalance | ~40:1 (Glioma 924 vs Pineal 23) |

### Class Distribution (Train+Val)

| Class | Count | Proportion |
|-------|-------|-----------|
| Glioma | 924 | 46.6% |
| Meningioma | 728 | 36.7% |
| Brain Metastase | 252 | 12.7% |
| Tumors of sellar region | 56 | 2.8% |
| Pineal/CP | 23 | 1.2% |

### Data Sources

| Source | Features | Solo Micro-F1 | Notes |
|--------|----------|--------------|-------|
| Report (TF-IDF+SVD) | 128d | **~85%** | Dominant source. Radiologist text. |
| Report (BERT frozen) | 768d | ~82% | Slightly worse than TF-IDF. |
| Clinical + SI | 27d | ~66% | Signal intensity patterns. |
| Location (label-encoded) | 1d | ~65% | Tumor location ≈ tumor type. |
| Location (one-hot) | 572d | ~75% | With GradientBoosting. |
| Image (ResNet PCA) | 512d | ~59% | Per-modality PCA, weak alone. |
| Radiomics (raw) | 20d | ~52% | Near random (~46%). |

### Key Finding: Radiomics is Noise

5 PyRadiomics features per modality (20 total) carry almost **zero discriminative signal** for tumor type classification. This is because:

- Features (Mean, Entropy, GLCM Contrast) capture **global texture statistics**
- They miss the clinically-relevant patterns radiologists use:
  - **Location** — directly captured by Location feature (~65%)
  - **Enhancement pattern** — captured by SI patterns in clinical notes
  - **Edema extent** — captured by SI patterns + Location

**Cross-modal ratios** (T1c/T1, T2F/T2, etc.) are medically meaningful (~77% when combined with Clinical+Location) but still insufficient alone.

---

## 2. Architecture Evolution

### v1: Original Unified Transformer

```
Per modality (T1/T1c/T2/T2F):
  [Image(2048) + Rad(5) + SI(6)] → Residual MLP → 512d

Clinical (Sex/Age/Location) → Residual MLP → 512d
Report (BERT 768d) → Residual MLP → 512d

6 tokens + CLS → 5-layer, 8-head Transformer
    → Classifier(512→256→5)
```

**Problem**: Image (~59%) and Radiomics (~52%) are noise but got equal weight as Report (~85%). Dragged down overall performance.

### v2: Dual-Path (No Image)

```
Path A: Report → BioClinicalBERT (top-3 fine-tune, CLS token) → 256d
Path B: Clinical + SI + Location + Cross-modal Ratios → MLP → 256d
    ↓ concat (512d) → Classifier (512→256→128→5)
```

**Key changes**:
- Removed Image entirely (noise source)
- Removed raw Radiomics (noise), kept cross-modal ratios
- Used CLS token instead of mean pooling
- Lightweight: ~10.5M trainable params (vs ~100M+ before)

### 5-Fold OOF Pipeline (Final)

See `model_training_5fold.ipynb`:

```
Cell 17: 5-Fold NN → oof_nn_probs (82%)
Cell 22: 5-Fold XGB/GBT on NN features → oof_xgb_probs, oof_gbt_probs (85%)
Cell 25: ML XGBoost on ALL sources (752d) → oof_ml_unified_probs (86%)
Cell 26: 6-model Stacking (Ridge/Logistic/Average)
```

---

## 3. Individual Model Results

### Neural Network

| Configuration | Micro-F1 | Macro-F1 | Notes |
|--------------|----------|---------|-------|
| NN (single train/val) | ~84% | - | Best val Micro-F1 |
| NN 5-Fold OOF | ~82% | - | More stable estimate |

### Tree Models on NN Features (512d)

| Model | Micro-F1 | Macro-F1 | Notes |
|-------|----------|---------|-------|
| XGBoost on NN features | ~85% | ~75% | Single split |
| GBT on NN features | ~85% | ~75% | Single split |
| XGBoost 5-Fold OOF | ~85% | - | Consistent |

### Pure ML (No Neural Network)

| Configuration | Features | Micro-F1 | Notes |
|--------------|----------|----------|-------|
| TF-IDF + LGB | Report SVD-64 | 84.4% | Best solo Report |
| XGBoost ALL sources | 752d | **86.45%** | Report+Clinical+LocOHE+Rad |
| LightGBM ALL sources | 752d | 86.19% | Slightly worse |
| CatBoost ALL sources | 752d | 83.94% | Worse |

**752d = Report SVD-128(128) + Clinical(27) + Location OHE(572) + Radiomics(49)**

### Blending Results

| Configuration | Micro-F1 | Notes |
|--------------|----------|-------|
| Report alone (TF-IDF) | 85.13% | Dominant |
| Report + BERT (equal) | 85.57% | Marginal improvement |
| Report + BERT + Location | 85.75% | Marginal improvement |
| Image + Report + BERT + Location | **86.23%** | Best old blend |
| 5-Fold 6-model Stacking | ~87-88% | Expected (unvalidated) |

---

## 4. Important Technical Notes

### 4.1 StratifiedKFold Statefulness

**Problem discovered**: `StratifiedKFold.split()` is a **stateful iterator**. If you iterate over it multiple times, you get **different fold splits each time**.

**Symptom**: OOF features and labels don't align → model trains on wrong data → 51% Micro-F1 (near random).

**Fix**: Recreate `skf` at the start of each cell that uses it:
```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr_idx, va_idx) in enumerate(skf.split(...)):
    ...
```

### 4.2 Why Location OHE > Label Encoding for Trees

| Encoding | Dimension | XGBoost Micro-F1 |
|----------|-----------|------------------|
| Label encoded | 1d | ~58% |
| One-hot | 572d | ~77% |

Location OHE with GradientBoosting (700 trees, depth=10, lr=0.2) reaches 77% — almost matching Report alone (85%). This is because different anatomical locations are highly predictive of tumor type.

### 4.3 Why Mean Pooling Destroys Information

Original image encoder used mean-pooling across 4 MRI modalities. This was harmful because:

- T1/T1c: anatomy + vascularity (contrast enhancement)
- T2/T2F: edema pattern

Mean-pooling mixes complementary signals. **Per-modality PCA** (128d × 4 = 512d) is strictly better (~59% vs ~57%).

### 4.4 Why Class Weights Hurt Micro-F1

Micro-F1 = accuracy for multi-class. Class weights boost minority classes but hurt majority classes (Glioma, Meningioma), resulting in lower overall accuracy.

- Without weights: Micro-F1 ~85%, Macro-F1 ~65%
- With weights: Micro-F1 ~82%, Macro-F1 ~70%

For Kaggle (Micro-F1 = accuracy), **don't use class weights**.

---

## 5. Bottleneck: Minority Classes

| Class | Accuracy | Samples | Status |
|-------|----------|---------|--------|
| Glioma | 95% | 1056 | ✅ Excellent |
| Meningioma | 90% | 832 | ✅ Good |
| Tumors of sellar | 75% | 64 | ⚠️ Moderate |
| Brain Metastase | **53%** | 288 | ❌ Poor |
| Pineal/CP | **15%** | 23 | ❌ Very Poor |

**Brain Metastase** and **Pineal/CP** account for ~95% of all errors. These need targeted strategies:

1. **Data augmentation** for minority classes
2. **Report-guided attention** to extract location/enhancement signals
3. **Confidence thresholding** with class-specific priors

---

## 6. Project Files

| File | Description |
|------|-------------|
| `ML.ipynb` | Pure ML experiments (TF-IDF, XGBoost, LightGBM, blending) |
| `model_training_new.ipynb` | Neural network training (original, single train/val) |
| `model_training_5fold.ipynb` | **Final**: 5-Fold OOF + 6-model stacking |
| `data_loader.py` | Data loading + PCA + preprocessing |
| `architecture_changes.md` | v1 → v2 architecture changes |
| `analysis_findings.md` | ML experiment findings summary |
| `experiment_design.md` | Planned experiments |
| `kaggle-dataset/` | Dataset (Clinical, Report, Radiomics, Image) |

---

## 7. Kaggle Competition Metric

**Verify**: Kaggle uses **Micro-F1** (= accuracy for multi-class). Do NOT use Macro-F1 or class-weighted metrics for model selection.

---

## 8. Known Issues & Resolutions

| Issue | Symptom | Resolution |
|-------|---------|-----------|
| `skf` stateful iterator | OOF Micro-F1 ~51% | Recreate `skf` in each cell |
| `patience=15` for 5-fold | NN barely trains | Use `patience=40+` or keep single train/val |
| Location OHE dims | ~572d | OK for GradientBoosting (uses all cols), bad for XGBoost (use `colsample_bytree=1.0`) |
| Mean pooling image | ~57% | Per-modality PCA is better |
| Class weights | Hurt Micro-F1 | Don't use for model selection |

---

## 9. Path to 90%+

### What's NOT the answer
- More radiomics features / engineering ❌ (radiomics is noise)
- Data augmentation ❌ (no signal to augment)
- Larger unified models ❌ (2K samples too small)
- Adding Image/Radiomics to ensemble ❌

### What's needed
1. **Better Report representations**: Fine-tune BERT (not frozen) with careful regularization
2. **Targeted Brain Metastase strategy**: 53% accuracy, 288 samples — biggest error source
3. **Cross-modal attention**: Let Report attend to clinical features for minority classes
4. **Pseudo-labeling**: Use high-confidence predictions on test set to augment training

---

## 10. Running the Final Notebook

```
# model_training_5fold.ipynb

Cells 0-16:  Setup (data, model, training functions)
Cell 17:      5-Fold NN → oof_nn_probs
Cell 22:      5-Fold XGB/GBT → oof_xgb_probs, oof_gbt_probs
Cell 25:      ML XGBoost on ALL sources → oof_ml_unified_probs
Cell 26:      6-model Stacking → final submission
```

Expected runtime: ~2-3 hours (5-fold NN is the bottleneck).
