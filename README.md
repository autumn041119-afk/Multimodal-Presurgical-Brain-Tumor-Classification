# Project Code Guide

## File Overview

| File | Type | Purpose |
|------|------|---------|
| `data_loader.py` | Utility | Data loading + preprocessing |
| `ML_experiments.ipynb` | Research | Solo modality experiments (trees) |
| `NN_experiments.ipynb` | Research | Neural network architecture experiments |
| `model_final.ipynb` | Production | Final ensemble model (NN + trees) |
| `report.pdf`  | Report | Final report for the whole project |
---

## 1. data_loader.py

**Purpose**: Centralized data loading and preprocessing for all modalities.

**What it does**:
- Loads radiology reports (text)
- Loads clinical features (Age, Sex, Signal Intensity)
- Loads location hierarchy (Region/Hemisphere/Lobe)
- Loads radiomics features (20 features × 4 modalities)
- Loads image features (ResNet-50 embeddings, PCA-compressed)
- Handles test set (original + new_test)
- Caches image features as `.npz` files for fast loading

**Key functions**:
```python
from data_loader import get_all_data

# Load all data for training
data = get_all_data('kaggle-dataset', pca_dim=128)

# Load new_test for submission
data = get_all_data('kaggle-dataset', pca_dim=128, test_data_dir='new_test')
```

**Returns**:
```python
{
    'train': {'report', 'clinical', 'merged', 'label', 'image'},
    'val':   {'report', 'clinical', 'merged', 'label', 'image'},
    'test':  {'report', 'clinical', 'merged', 'image'},
    'label_encoder': LabelEncoder,
    'clinical_cols': [column names]
}
```

**When to use**: Import this in any notebook that needs data.

---

## 2. ML_experiments.ipynb

**Purpose**: Exploratory analysis — "Which modality is most predictive?"

**What it does**:
- Tests each modality alone (solo performance)
- Compares tree models (XGBoost, LightGBM, CatBoost)
- Feature ablation experiments
- Class weight sensitivity analysis
- Cross-source blending experiments

**Key experiments**:
| Cell | Content | Output |
|------|---------|--------|
| Cell 6 | Solo modality baseline | `solo_modality_performance.png` |
| Cell 20 | Tree model comparison on radiomics/image | Table of Micro-F1 |
| Cell 25 | Report TF-IDF experiments | SVD-64/128/256 comparison |
| Cell 31 | Class weight experiment | Weight strategies comparison |
| Cell 34 | Cross-source blending | Best source combinations |

**Output**: 
- Understanding of which modalities matter
- Feature engineering insights
- Preliminary model configurations

**When to use**: 
- Before designing the final model
- To understand data characteristics
- To justify architecture decisions

**Not for**: Final model training (use model_final.ipynb)

---

## 3. NN_experiments.ipynb

**Purpose**: Neural network architecture exploration.

**What it does**:
- Tests different NN architectures
- Cross-modal gating experiments
- Ablation on BERT vs TF-IDF
- Image modality experiments (v1 architecture)

**Key findings** (historical):
- BERT embeddings outperform TF-IDF for NN
- Image features hurt NN when combined with text (distribution shift)
- CrossModalGate improves minority class performance

**When to use**:
- To understand why the final architecture was chosen
- To see alternative architectures that were tested
- Historical reference for design decisions

**Not for**: Final model training (model_final.ipynb has the final version)

---

## 4. model_final.ipynb

**Purpose**: Final production model — the one used for submission.

**What it does**:
- 5-fold OOF NN training (Dual-path: BERT + Tabular with CrossModalGate)
- 5-fold OOF tree models (XGBoost, LightGBM, CatBoost on TF-IDF + tabular)
- Ensemble weight optimization via grid search
- Generates submission.csv

**Pipeline**:
```
Cell 4:   Load data via data_loader.py
Cell 7-12: Pre-compute BioClinicalBERT embeddings
Cell 14:  Define DualPathBrainTumorClassifier
Cell 17:  Train NN (5-fold OOF) → oof_nn_probs
Cell 20:  Train trees (5-fold OOF) → oof_mlu_xgb/lgb/cb, oof_nnxgb, oof_imgtab
Cell 22:  Grid search ensemble weights
Cell 23:  Generate submission
```

**Output**:
- `submission.csv`
- OOF probabilities for all 6 models
- Per-class metrics (Micro-F1, Macro-F1, AUC)

**When to use**: 
- Training the final model
- Reproducing the submission
- Generating new test predictions

---

## How to Use This Project

### For Exploration (Understanding the data)
```
1. Open ML_experiments.ipynb
2. Run Cell 6 → See solo modality performance
3. Review findings → Understand which modalities matter
```

### For Architecture Decisions
```
1. Open NN_experiments.ipynb
2. Review experiments → See why certain choices were made
3. Open model_final.ipynb Cell 0 → See architecture summary
```

### For Final Model Training
```
1. Open model_final.ipynb
2. Run all cells in order (4 → 23)
3. Get submission.csv from Cell 23
```

### For Data Loading Only
```python
from data_loader import get_all_data

data = get_all_data('kaggle-dataset')
train_labels = data['train']['label']
train_reports = data['train']['report']
```

---

## Dependency Chain

```
data_loader.py
    │
    ├── ML_experiments.ipynb (uses get_all_data)
    │
    ├── NN_experiments.ipynb (uses get_all_data)
    │
    └── model_final.ipynb (uses get_all_data)
```

**No circular dependencies** — data_loader.py is standalone.

---

## File Sizes

| File | Size | Note |
|------|------|------|
| data_loader.py | 31 KB | Utility only |
| ML_experiments.ipynb | 237 KB | Exploratory research |
| NN_experiments.ipynb | 45 KB | Architecture exploration |
| model_final.ipynb | 88 KB | Final model (main output) |

---

## Quick Reference

| Need | Use |
|------|-----|
| Load data | `from data_loader import get_all_data` |
| Solo experiments | `ML_experiments.ipynb` Cell 6 |
| NN architecture | `NN_experiments.ipynb` |
| Final model | `model_final.ipynb` |
| Submission | `model_final.ipynb` Cell 23 |

---

## Reproducibility

All notebooks use:
- `random_state=42` for CV splits
- `random_state=42` for TF-IDF/SVD fitting (train only)
- No data leakage (features fitted on train, applied to val/test)

PCA and TF-IDF are fitted on training data only, then applied to val/test.
