"""
data_loader.py
Brain Tumor Classification - Data Loading & Preprocessing

Usage:
    from data_loader import get_all_data
    data = get_all_data()

    # Then access:
    data['train']['clinical']       # DataFrame: case_id + clinical features
    data['train']['radiomics']      # DataFrame: case_id + 20 radiomics features
    data['train']['image']          # dict: {case_id: {modality: np.array(2048,)}}
    data['train']['report']         # DataFrame: case_id + report text
    data['train']['label']          # np.array of encoded labels
    data['train']['merged']         # DataFrame: clinical + radiomics + label merged
    data['label_encoder']           # LabelEncoder for 5 tumor classes
    data['clinical_cols']           # list of clinical feature column names
"""

import json
import numpy as np
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x


MODALITIES = ['ax_t1', 'ax_t1c', 'ax_t2', 'ax_t2f']
SI_COLS = ['Signal Intensity (T1)', 'Signal Intensity (T1c)',
           'Signal Intensity (T2)', 'Signal Intensity (T2-FLAIR)']


# ---------------------------------------------------------------------------
# Step 1: Clinical Information
# ---------------------------------------------------------------------------
def load_clinical(data_dir, test_data_dir=None):
    ci_train = pd.read_csv(os.path.join(data_dir, 'clinical_information/train_patient_info.csv'), encoding='utf-8-sig')
    ci_val = pd.read_csv(os.path.join(data_dir, 'clinical_information/val_patient_info.csv'), encoding='utf-8-sig')
    # Test: use test_data_dir if provided, otherwise fall back to data_dir
    test_dir = test_data_dir if test_data_dir else data_dir
    test_clinical_path = (os.path.join(test_dir, 'clinical_information/test/test_patient_info.csv')
                          if test_data_dir
                          else os.path.join(data_dir, 'clinical_information/test_patient_info.csv'))
    ci_test = pd.read_csv(test_clinical_path, encoding='utf-8-sig')

    # Collect all signal intensity values for consistent one-hot columns
    all_si_values = set()
    for df in [ci_train, ci_val, ci_test]:
        for col in SI_COLS:
            all_si_values.update(df[col].dropna().unique())
    all_si_values = sorted(all_si_values)

    # Process train first, get median, apply to val/test
    ci_train, age_median = _process_clinical(ci_train, all_si_values)
    ci_val, _ = _process_clinical(ci_val, all_si_values, age_median)
    ci_test, _ = _process_clinical(ci_test, all_si_values, age_median)

    # Build column list (NOTE: column names include the full SI_COL name + '_' + value.
    # In the notebook Dataset, we match with `startswith(si_name + '_')` to avoid
    # the bug where 'Signal Intensity (T1)' accidentally matches 'Signal Intensity (T1c)'.
    si_cols_final = [f'{col}_{val}' for col in SI_COLS for val in all_si_values]
    clinical_cols = ['Sex_enc', 'Age_clean', 'Age_missing'] + si_cols_final

    # Build location encoder across all splits
    all_locations = pd.concat([ci_train['Location_enc'], ci_val['Location_enc'], ci_test['Location_enc']])
    loc_encoder = LabelEncoder()
    loc_encoder.fit(all_locations)
    ci_train['Location_enc'] = loc_encoder.transform(ci_train['Location_enc'])
    ci_val['Location_enc'] = loc_encoder.transform(ci_val['Location_enc'])
    ci_test['Location_enc'] = loc_encoder.transform(ci_test['Location_enc'])

    all_cols = clinical_cols + ['Location_enc', 'Tumor_Location_raw']

    ci_train_final = ci_train[['case_id'] + all_cols].copy()
    ci_val_final = ci_val[['case_id'] + all_cols].copy()
    ci_test_final = ci_test[['case_id'] + all_cols].copy()

    for df in [ci_train_final, ci_val_final, ci_test_final]:
        df['case_id'] = df['case_id'].astype(int)

    print(f'[Clinical] {len(clinical_cols)} features + 1 location (embedding), '
          f'age_median={age_median}, locations={len(loc_encoder.classes_)}')
    return ci_train_final, ci_val_final, ci_test_final, clinical_cols, loc_encoder


def _process_clinical(df, all_si_values, age_median=None):
    df = df.copy()
    # Sex: ordinal
    sex_map = {'male': 1, 'female': 0, 'unknown': -1}
    df['Sex_enc'] = df['Sex'].map(sex_map).fillna(-1).astype(int)
    # Age: fill with train median + missing flag
    if age_median is None:
        age_median = df['Age'].median()
    df['Age_missing'] = df['Age'].isna().astype(int)
    df['Age_clean'] = df['Age'].fillna(age_median)
    # Signal Intensity: one-hot
    for col in SI_COLS:
        col_values = df[col].fillna('unknown').values
        for val in all_si_values:
            df[f'{col}_{val}'] = (col_values == val).astype(int)
    # Tumor Location: normalize and label encode (for embedding in model)
    df['Location_enc'] = df['Tumor Location'].apply(_normalize_location)
    # Keep raw Tumor Location text for hierarchical parsing in the model
    df['Tumor_Location_raw'] = df['Tumor Location'].fillna('')
    return df, age_median


def _normalize_location(loc):
    """Normalize tumor location: lowercase, strip, clean Chinese chars."""
    if pd.isna(loc):
        return 'unknown'
    loc = str(loc).lower().strip()
    # Remove Chinese characters and trailing fragments
    import re
    loc = re.sub(r'[\u4e00-\u9fff]+.*$', '', loc).strip()
    # Remove leading Chinese prefixes like "位置: "
    loc = re.sub(r'^[^a-z]*', '', loc).strip()
    # Clean trailing commas/whitespace
    loc = loc.rstrip(',').strip()
    return loc if loc else 'unknown'


# ---------------------------------------------------------------------------
# Location Hierarchy Parser (for robustness to novel locations)
# ---------------------------------------------------------------------------
REGION_MAP = {
    'brain': ['brain', 'cerebr', 'cerebell', 'frontal', 'temporal', 'parietal',
              'occipital', 'pons', 'medulla', 'midbrain', 'thalamus', 'basal ganglia',
              'corpus callosum', 'ventricle', 'sella', 'pineal', 'cp angle', 'cpa'],
    'spine': ['spine', 'spinal', 'vertebra', 'cord', 'cervical', 'thoracic', 'lumbar'],
    'skull': ['skull', 'calvarium', 'cranial fossa', 'convexity'],
}

HEMISPHERE_MAP = {
    'left': ['left'],
    'right': ['right'],
    'bilateral': ['bilateral', 'both sides', 'multiple'],
    'midline': ['midline', 'central', 'sella', 'pineal', 'third ventricle', 'fourth ventricle'],
}

LOBE_MAP = {
    'frontal': ['frontal', 'front'],
    'temporal': ['temporal', 'temp'],
    'parietal': ['parietal'],
    'occipital': ['occipital'],
    'cerebellar': ['cerebell', 'cp angle', 'cpa'],
    'brainstem': ['brainstem', 'pons', 'medulla', 'midbrain'],
    'sellar': ['sella', 'pituitary', 'suprasellar'],
    'pineal': ['pineal'],
    'ventricular': ['ventricle'],
}


def parse_location_hierarchy(loc_text):
    """Parse a raw location string into hierarchical categories.

    Returns:
        (region_enc, hemisphere_enc, lobe_enc) as integers.
        Encodings:
          region: 0=brain, 1=spine, 2=skull, 3=other
          hemisphere: 0=left, 1=right, 2=bilateral, 3=midline, 4=unknown
          lobe: 0=frontal, 1=temporal, 2=parietal, 3=occipital,
                4=cerebellar, 5=brainstem, 6=sellar, 7=pineal,
                8=ventricular, 9=other
    """
    if pd.isna(loc_text):
        loc_text = ''
    loc = str(loc_text).lower().strip()

    # Determine anatomical region (brain, spine, skull, other)
    region = 'other'
    for r, keywords in REGION_MAP.items():
        if any(kw in loc for kw in keywords):
            region = r
            break

    # Determine hemisphere (left, right, bilateral, midline, unknown)
    hemisphere = 'unknown'
    for h, keywords in HEMISPHERE_MAP.items():
        if any(kw in loc for kw in keywords):
            hemisphere = h
            break

    # Determine specific anatomical lobe
    lobe = 'other'
    for l, keywords in LOBE_MAP.items():
        if any(kw in loc for kw in keywords):
            lobe = l
            break

    region_enc = {'brain': 0, 'spine': 1, 'skull': 2, 'other': 3}[region]
    hemisphere_enc = {'left': 0, 'right': 1, 'bilateral': 2, 'midline': 3, 'unknown': 4}[hemisphere]
    lobe_enc = {'frontal': 0, 'temporal': 1, 'parietal': 2, 'occipital': 3,
                'cerebellar': 4, 'brainstem': 5, 'sellar': 6, 'pineal': 7,
                'ventricular': 8, 'other': 9}[lobe]

    return region_enc, hemisphere_enc, lobe_enc


# ---------------------------------------------------------------------------
# Step 2: Raw Reports
# ---------------------------------------------------------------------------
def load_reports(data_dir, test_data_dir=None):
    splits = {}
    for split in ['train', 'val']:
        df = pd.read_csv(os.path.join(data_dir, f'original_raw_report/{split}_patient_info.csv'), encoding='utf-8-sig')
        df.columns = ['case_id', 'report']
        df['case_id'] = df['case_id'].astype(int)
        splits[split] = df
    # Test: use test_data_dir if provided, otherwise fall back to data_dir
    test_dir = test_data_dir if test_data_dir else data_dir
    test_path = os.path.join(test_dir, 'original_raw_report/test/test_patient_info.csv')
    df = pd.read_csv(test_path, encoding='utf-8-sig')
    df.columns = ['case_id', 'report']
    df['case_id'] = df['case_id'].astype(int)
    splits['test'] = df
    print(f'[Reports] train={len(splits["train"])}, val={len(splits["val"])}, test={len(splits["test"])}')
    return splits['train'], splits['val'], splits['test']


# ---------------------------------------------------------------------------
# Step 3: Radiomics Features
# ---------------------------------------------------------------------------
def load_radiomics(data_dir):
    rad_train = _load_radiomics_split(data_dir, 'train')
    rad_val = _load_radiomics_split(data_dir, 'val')
    rad_test = _load_radiomics_split(data_dir, 'test')

    feat_cols = [c for c in rad_train.columns if c != 'case_id']

    # Add missing flag per modality (1=missing, 0=present)
    for mod in MODALITIES:
        mod_cols = [c for c in feat_cols if c.startswith(f'{mod}_')]
        for df in [rad_train, rad_val, rad_test]:
            df[f'{mod}_missing'] = df[mod_cols].isnull().any(axis=1).astype(int)

    # Fill NaN with 0 (missing modality = no radiomics data)
    for df in [rad_train, rad_val, rad_test]:
        df[feat_cols] = df[feat_cols].fillna(0)

    # Update feature list to include missing flags
    missing_cols = [f'{mod}_missing' for mod in MODALITIES]
    print(f'[Radiomics] {len(feat_cols)} features + {len(missing_cols)} missing flags '
          f'= {len(feat_cols) + len(missing_cols)} total')
    return rad_train, rad_val, rad_test


def _load_radiomics_split(data_dir, split):
    dfs = []
    feat_names = ['rad_firstorder_Mean', 'rad_firstorder_Entropy',
                  'rad_firstorder_90Percentile', 'rad_glcm_Contrast', 'rad_glcm_JointEntropy']
    for mod in MODALITIES:
        fname = f'radiomics_info/{split}/{mod}_radiomics_{split}.csv'
        df = pd.read_csv(os.path.join(data_dir, fname), encoding='utf-8-sig')
        df['case_id'] = df['case_id'].astype(int)
        for col in feat_names:
            df.rename(columns={col: f'{mod}_{col}'}, inplace=True)
        # Only drop columns that exist (new_test radiomics lack 'sex'/'age')
        drop_cols = [c for c in ['sex', 'age', 'modality'] if c in df.columns]
        df.drop(columns=drop_cols, inplace=True)
        dfs.append(df)
    result = dfs[0]
    for df in dfs[1:]:
        result = result.merge(df, on='case_id', how='outer')
    return result


# ---------------------------------------------------------------------------
# Step 4: Image Features (ResNet)
# ---------------------------------------------------------------------------
def _load_one_case(args):
    image_base, cid, zero_vec = args
    case_path = os.path.join(image_base, str(cid))
    mod_features = {}
    for mod in MODALITIES:
        npy_path = os.path.join(case_path, mod, 'image.npy')
        if os.path.exists(npy_path):
            mod_features[mod] = np.load(npy_path).astype(np.float32)
        else:
            mod_features[mod] = zero_vec
    return cid, mod_features


def _load_image_split(image_base, case_ids):
    zero_vec = np.zeros(2048, dtype=np.float32)
    features = {}
    args_list = [(image_base, cid, zero_vec) for cid in case_ids]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for cid, mod_features in pool.map(_load_one_case, args_list):
            features[cid] = mod_features
    return features


# ---------------------------------------------------------------------------
# Image feature cache: merge all small .npy into a single compressed file
# for fast loading on Colab (network-mounted disk)
# ---------------------------------------------------------------------------
IMAGE_CACHE_PATH = os.path.join(os.path.dirname(__file__), 'image_features_cache.npz')
NEW_TEST_IMAGE_CACHE = os.path.join(os.path.dirname(__file__), 'new_test_image_cache.npz')


def _build_image_cache(data_dir, test_data_dir=None):
    """Merge all individual image.npy files into one (N, 4, 2048) array.

    Args:
        data_dir: Base directory for train/val data.
        test_data_dir: Optional separate directory for test data.
    """
    image_base = os.path.join(data_dir, 'image_features/image_features')

    # Collect all case_ids from JSON files
    all_cids = []
    for split in ['train', 'val']:
        json_path = os.path.join(data_dir, f'{split}.json')
        with open(json_path, encoding='utf-8') as f:
            all_cids.extend(json.load(f).keys())

    # Test: use test_data_dir if provided, otherwise fall back to data_dir
    test_dir = test_data_dir if test_data_dir else data_dir
    test_json_path = os.path.join(test_dir, 'test.json')
    with open(test_json_path, encoding='utf-8') as f:
        all_cids.extend(json.load(f).keys())

    all_cids = sorted(set(all_cids), key=int)
    cid_to_idx = {int(cid): i for i, cid in enumerate(all_cids)}

    # Load from individual .npy files
    zero_vec = np.zeros(2048, dtype=np.float32)
    all_arrays = np.zeros((len(all_cids), len(MODALITIES), 2048), dtype=np.float32)

    for i, cid in enumerate(tqdm(all_cids, desc='Building image cache')):
        case_path = os.path.join(image_base, str(cid))
        for j, mod in enumerate(MODALITIES):
            npy_path = os.path.join(case_path, mod, 'image.npy')
            if os.path.exists(npy_path):
                all_arrays[i, j] = np.load(npy_path).astype(np.float32)
            else:
                all_arrays[i, j] = zero_vec

    case_ids_arr = np.array(all_cids)
    np.savez_compressed(IMAGE_CACHE_PATH, case_ids=case_ids_arr, features=all_arrays)
    size_mb = os.path.getsize(IMAGE_CACHE_PATH) / (1024 * 1024)
    print(f'[Image] Cache saved to {IMAGE_CACHE_PATH} ({size_mb:.1f} MB)')


def _build_new_test_cache(test_data_dir, case_ids_test):
    """Build image cache for new_test from individual .npy files.

    Args:
        test_data_dir: Directory containing new_test image data.
        case_ids_test: List of case IDs to include in the cache.
    """
    img_base = os.path.join(test_data_dir, 'image_features/image_features')
    zero_vec = np.zeros(2048, dtype=np.float32)

    case_ids_arr = np.array(sorted(case_ids_test, key=int))
    all_arrays = np.zeros((len(case_ids_arr), len(MODALITIES), 2048), dtype=np.float32)

    for i, cid in enumerate(tqdm(case_ids_arr, desc='Building new_test image cache')):
        case_path = os.path.join(img_base, str(cid))
        for j, mod in enumerate(MODALITIES):
            npy_path = os.path.join(case_path, mod, 'image.npy')
            if os.path.exists(npy_path):
                all_arrays[i, j] = np.load(npy_path).astype(np.float32)
            else:
                all_arrays[i, j] = zero_vec

    np.savez_compressed(NEW_TEST_IMAGE_CACHE, case_ids=case_ids_arr, features=all_arrays)
    size_mb = os.path.getsize(NEW_TEST_IMAGE_CACHE) / (1024 * 1024)
    print(f'[Image] new_test cache saved to {NEW_TEST_IMAGE_CACHE} ({size_mb:.1f} MB)')


def _load_new_test_from_cache(case_ids_test, pca_model=None, pca_dim=None):
    """Load new_test images from dedicated cache.

    Args:
        case_ids_test: List of case IDs to load.
        pca_model: Optional PCA model to transform features.
        pca_dim: Target dimension if PCA is applied.
    """
    data = np.load(NEW_TEST_IMAGE_CACHE, allow_pickle=True)
    case_ids_all = data['case_ids']
    features_all = data['features']  # (N_test, 4, 2048)
    cid_to_idx = {int(cid): i for i, cid in enumerate(case_ids_all)}

    target_dim = pca_dim if pca_dim else 2048
    result = {}
    for cid in case_ids_test:
        idx = cid_to_idx.get(int(cid))
        if idx is not None:
            features_per_mod = {}
            for i, mod in enumerate(MODALITIES):
                feat = features_all[idx, i]  # (2048,)
                if pca_model is not None:
                    feat = pca_model.transform(feat.reshape(1, -1))[0]
                features_per_mod[mod] = feat.astype(np.float32)
            result[cid] = features_per_mod
        else:
            result[cid] = {mod: np.zeros(target_dim, dtype=np.float32) for mod in MODALITIES}
    return result


def _load_from_cache_with_pca(split_cids, pca_model=None, pca_dim=None):
    """Load from cache and optionally apply PCA."""
    data = np.load(IMAGE_CACHE_PATH, allow_pickle=True)
    case_ids_all = data['case_ids']
    features_all = data['features']  # (N_total, 4, 2048)
    cid_to_idx = {int(cid): i for i, cid in enumerate(case_ids_all)}

    target_dim = pca_dim if pca_dim else 2048

    result = {}
    for cid in split_cids:
        idx = cid_to_idx.get(int(cid))
        if idx is not None:
            features_per_mod = {}
            for i, mod in enumerate(MODALITIES):
                feat = features_all[idx, i]  # (2048,)
                if pca_model is not None:
                    # Apply PCA transformation
                    feat = pca_model.transform(feat.reshape(1, -1))[0]
                else:
                    # No PCA - keep original dim
                    pass  # feat remains (2048,)
                features_per_mod[mod] = feat.astype(np.float32)
            result[cid] = features_per_mod
        else:
            result[cid] = {mod: np.zeros(target_dim, dtype=np.float32) for mod in MODALITIES}
    return result


def _fit_pca_on_train(case_ids_train, n_components):
    """Fit PCA on training data only."""
    data = np.load(IMAGE_CACHE_PATH, allow_pickle=True)
    case_ids_all = data['case_ids']
    features_all = data['features']  # (N_total, 4, 2048)
    cid_to_idx = {int(cid): i for i, cid in enumerate(case_ids_all)}

    # Collect all training features across all modalities
    train_features = []
    for cid in case_ids_train:
        idx = cid_to_idx.get(int(cid))
        if idx is not None:
            for i, mod in enumerate(MODALITIES):
                train_features.append(features_all[idx, i])

    train_features = np.array(train_features)  # (N_train * 4, 2048)
    print(f'[PCA] Fitting PCA on {len(train_features)} samples ({len(case_ids_train)} cases × 4 modalities)')

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(train_features)
    explained = pca.explained_variance_ratio_.sum()
    print(f'[PCA] {n_components} components explain {explained*100:.2f}% variance')

    return pca


def _load_from_cache(split_cids):
    """Load from the single cached file and return dict format."""
    data = np.load(IMAGE_CACHE_PATH, allow_pickle=True)
    case_ids_all = data['case_ids']
    features_all = data['features']  # (N_total, 4, 2048)
    cid_to_idx = {int(cid): i for i, cid in enumerate(case_ids_all)}

    result = {}
    for cid in split_cids:
        idx = cid_to_idx.get(int(cid))
        if idx is not None:
            result[cid] = {mod: features_all[idx, i] for i, mod in enumerate(MODALITIES)}
        else:
            result[cid] = {mod: np.zeros(2048, dtype=np.float32) for mod in MODALITIES}
    return result


def load_image_features(data_dir, case_ids_train, case_ids_val, case_ids_test,
                         pca_dim=None, test_data_dir=None):
    """
    Load image features with optional PCA dimensionality reduction.

    Args:
        pca_dim: If specified, apply PCA to reduce each modality from 2048 to pca_dim.
                 PCA is fit on training data only to prevent leakage.
        test_data_dir: If specified, load test images from this directory
                       (e.g., 'new_test') instead of the old cache.
    """
    image_base = os.path.join(data_dir, 'image_features/image_features')

    # Helper: load a single split from a directory
    def _load_split_from_dir(case_ids, data_dir, split_name):
        img_base = os.path.join(data_dir, 'image_features/image_features')
        zero_vec = np.zeros(2048, dtype=np.float32)
        features = {}
        for cid in case_ids:
            mod_features = {}
            for mod in MODALITIES:
                npy_path = os.path.join(img_base, str(cid), mod, 'image.npy')
                if os.path.exists(npy_path):
                    mod_features[mod] = np.load(npy_path).astype(np.float32)
                else:
                    mod_features[mod] = zero_vec
            features[cid] = mod_features
        return features

    # Use cache if available (one file read vs 11k+ file reads)
    if os.path.exists(IMAGE_CACHE_PATH):
        if pca_dim is not None:
            print(f'[Image] Loading from cache with PCA({pca_dim})...')
            # Fit PCA on train only
            pca = _fit_pca_on_train(case_ids_train, pca_dim)
            # Apply to all splits
            img_train = _load_from_cache_with_pca(case_ids_train, pca, pca_dim)
            img_val = _load_from_cache_with_pca(case_ids_val, pca, pca_dim)
            # Test: from test_data_dir or old cache
            if test_data_dir:
                print(f'[Image] Loading test images from {test_data_dir}...')
                if os.path.exists(NEW_TEST_IMAGE_CACHE):
                    print('[Image] Loading new_test from cache...')
                    img_test = _load_new_test_from_cache(case_ids_test, pca, pca_dim)
                else:
                    print('[Image] No new_test cache found, building one...')
                    _build_new_test_cache(test_data_dir, case_ids_test)
                    img_test = _load_new_test_from_cache(case_ids_test, pca, pca_dim)
            else:
                img_test = _load_from_cache_with_pca(case_ids_test, pca, pca_dim)
            dim_info = f'{pca_dim}-dim (PCA from 2048)'
        else:
            print('[Image] Loading from cache...')
            img_train = _load_from_cache(case_ids_train)
            img_val = _load_from_cache(case_ids_val)
            if test_data_dir:
                print(f'[Image] Loading test images from {test_data_dir}...')
                if os.path.exists(NEW_TEST_IMAGE_CACHE):
                    print('[Image] Loading new_test from cache...')
                    img_test = _load_new_test_from_cache(case_ids_test)
                else:
                    print('[Image] No new_test cache found, building one...')
                    _build_new_test_cache(test_data_dir, case_ids_test)
                    img_test = _load_new_test_from_cache(case_ids_test)
            else:
                img_test = _load_from_cache(case_ids_test)
            dim_info = '2048-dim'
    else:
        # Build cache on first run (slow, but only once)
        print('[Image] No cache found, building image cache (one-time)...')
        _build_image_cache(data_dir, test_data_dir)
        # After building cache, reload with PCA if needed
        if pca_dim is not None:
            print(f'[Image] Loading from cache with PCA({pca_dim})...')
            pca = _fit_pca_on_train(case_ids_train, pca_dim)
            img_train = _load_from_cache_with_pca(case_ids_train, pca, pca_dim)
            img_val = _load_from_cache_with_pca(case_ids_val, pca, pca_dim)
            if test_data_dir:
                print(f'[Image] Loading test images from {test_data_dir}...')
                if os.path.exists(NEW_TEST_IMAGE_CACHE):
                    print('[Image] Loading new_test from cache...')
                    img_test = _load_new_test_from_cache(case_ids_test, pca, pca_dim)
                else:
                    print('[Image] No new_test cache found, building one...')
                    _build_new_test_cache(test_data_dir, case_ids_test)
                    img_test = _load_new_test_from_cache(case_ids_test, pca, pca_dim)
            else:
                img_test = _load_from_cache_with_pca(case_ids_test, pca, pca_dim)
            dim_info = f'{pca_dim}-dim (PCA from 2048)'
        else:
            print('[Image] Loading from cache...')
            img_train = _load_from_cache(case_ids_train)
            img_val = _load_from_cache(case_ids_val)
            if test_data_dir:
                print(f'[Image] Loading test images from {test_data_dir}...')
                if os.path.exists(NEW_TEST_IMAGE_CACHE):
                    print('[Image] Loading new_test from cache...')
                    img_test = _load_new_test_from_cache(case_ids_test)
                else:
                    print('[Image] No new_test cache found, building one...')
                    _build_new_test_cache(test_data_dir, case_ids_test)
                    img_test = _load_new_test_from_cache(case_ids_test)
            else:
                img_test = _load_from_cache(case_ids_test)
            dim_info = '2048-dim'

    print(f'[Image] {dim_info} x {len(MODALITIES)} modalities, '
          f'train={len(img_train)}, val={len(img_val)}, test={len(img_test)}')
    return img_train, img_val, img_test


# ---------------------------------------------------------------------------
# Step 5: Merge All with JSON Labels
# ---------------------------------------------------------------------------
def load_labels(data_dir):
    with open(os.path.join(data_dir, 'train.json'), encoding='utf-8') as f:
        train_json = json.load(f)
    with open(os.path.join(data_dir, 'val.json'), encoding='utf-8') as f:
        val_json = json.load(f)

    label_dict = {}
    for cid, info in {**train_json, **val_json}.items():
        if 'Overall_class' in info:
            label_dict[int(cid)] = info['Overall_class']

    label_encoder = LabelEncoder()
    label_encoder.fit(list(label_dict.values()))
    print(f'[Labels] {len(label_dict)} labeled samples, {len(label_encoder.classes_)} classes')
    return label_dict, label_encoder


def _merge_split(ci_df, rad_df, label_dict, label_encoder):
    merged = ci_df.merge(rad_df, on='case_id', how='inner')
    merged['Overall_class'] = merged['case_id'].map(label_dict)
    merged['label'] = merged['Overall_class'].map(
        lambda x: label_encoder.transform([x])[0] if pd.notna(x) else -1
    )
    return merged


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def get_all_data(data_dir='kaggle-dataset', pca_dim=None, test_data_dir=None):
    """
    Load and preprocess all data. Returns a dict with train/val/test splits.

    Args:
        data_dir: Base directory for train/val data (kaggle-dataset).
        pca_dim: Optional int. If specified, reduce image features from 2048 to pca_dim
                 using PCA fitted on training data only.
        test_data_dir: Optional. If specified, use this directory for test data
                       (e.g., 'new_test' for the new test set without data leakage).
                       If None, falls back to data_dir for test data.
                       Note: test_data_dir should contain the same subdirectory structure
                       as kaggle-dataset (clinical_information/, original_raw_report/, etc.)
    """
    print('=' * 50)
    print('Loading all data...')
    if pca_dim:
        print(f'Using PCA: {pca_dim} dimensions per modality (from 2048)')
    if test_data_dir:
        print(f'Test data: {test_data_dir}')
    print('=' * 50)

    # Step 1: Clinical
    ci_train, ci_val, ci_test, clinical_cols, loc_encoder = load_clinical(data_dir, test_data_dir)

    # Step 2: Reports
    rr_train, rr_val, rr_test = load_reports(data_dir, test_data_dir)

    # Step 3: Radiomics (train/val from data_dir, test from test_data_dir if provided)
    if test_data_dir:
        rad_train = _load_radiomics_split(data_dir, 'train')
        rad_val = _load_radiomics_split(data_dir, 'val')
        rad_test = _load_radiomics_split(test_data_dir, 'test')
        feat_cols = [c for c in rad_train.columns if c != 'case_id']
        for mod in MODALITIES:
            mod_cols = [c for c in feat_cols if c.startswith(f'{mod}_')]
            for df in [rad_train, rad_val, rad_test]:
                df[f'{mod}_missing'] = df[mod_cols].isnull().any(axis=1).astype(int)
        for df in [rad_train, rad_val, rad_test]:
            df[feat_cols] = df[feat_cols].fillna(0)
        print(f'[Radiomics] {len(feat_cols)} features + {len(MODALITIES)} missing flags')
    else:
        rad_train, rad_val, rad_test = load_radiomics(data_dir)

    # Step 4: Image features (train/val from data_dir cache, test from test_data_dir if provided)
    img_train, img_val, img_test = load_image_features(
        data_dir,
        ci_train['case_id'].tolist(),
        ci_val['case_id'].tolist(),
        ci_test['case_id'].tolist(),
        pca_dim=pca_dim,
        test_data_dir=test_data_dir,
    )

    # Step 5: Labels + merge
    label_dict, label_encoder = load_labels(data_dir)

    merged_train = _merge_split(ci_train, rad_train, label_dict, label_encoder)
    merged_val = _merge_split(ci_val, rad_val, label_dict, label_encoder)
    merged_test = _merge_split(ci_test, rad_test, label_dict, label_encoder)

    # Pack into dict
    data = {
        'train': {
            'clinical': ci_train,
            'radiomics': rad_train,
            'image': img_train,
            'report': rr_train,
            'label': merged_train['label'].values,
            'merged': merged_train,
        },
        'val': {
            'clinical': ci_val,
            'radiomics': rad_val,
            'image': img_val,
            'report': rr_val,
            'label': merged_val['label'].values,
            'merged': merged_val,
        },
        'test': {
            'clinical': ci_test,
            'radiomics': rad_test,
            'image': img_test,
            'report': rr_test,
            'merged': merged_test,
        },
        'label_encoder': label_encoder,
        'loc_encoder': loc_encoder,
        'clinical_cols': clinical_cols,
        'pca_dim': pca_dim,
    }

    print('=' * 50)
    print(f'Label classes: {list(label_encoder.classes_)}')
    print(f'Train: {len(merged_train)} | Val: {len(merged_val)} | Test: {len(merged_test)}')
    if pca_dim:
        print(f'Image dim: {pca_dim} (PCA-reduced from 2048)')
    if test_data_dir:
        print(f'Test data loaded from: {test_data_dir}')
    print('All data loaded successfully.')
    print('=' * 50)

    return data


if __name__ == '__main__':
    data = get_all_data()