# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

An enterprise-grade, high-throughput Entity Resolution (Record Linkage) system designed to link business entities across heterogeneous data sources (`Source 1`, `Source 2`, `Source 3`) across multiple country jurisdictions (`US`, `India`, `France`).

---

## 1. System Architecture

```
                       ┌────────────────────────┐
                       │ Raw Tabular Records    │
                       │ (S1, S2, S3 Datasets)  │
                       └───────────┬────────────┘
                                   │
                                   ▼
               ┌────────────────────────────────────────┐
               │ Country Partitioning & Normalization   │
               │ - Legal Suffix Mapping (Inc/LLC/Pvt)   │
               │ - Address Abbreviation Expansions      │
               │ - Vectorized Composite Text Synthesis  │
               └───────────────────┬────────────────────┘
                                   │
                                   ▼
               ┌────────────────────────────────────────┐
               │ Sub-Linear Inverted Index Blocking     │
               │ - Word (1,2)-gram TF-IDF Posting Lists │
               │ - Top-N Term Discriminative Retrieval  │
               │ - Fast C-Level Posting Merging (Top-50)│
               └───────────────────┬────────────────────┘
                                   │
                                   ▼
               ┌────────────────────────────────────────┐
               │ 34-Dimensional Dense Feature Vector    │
               │ - RapidFuzz Levenshtein, Sort & Set    │
               │ - Character Bigram & Trigram Jaccard   │
               │ - Token Containment & First-Word Exact │
               │ - Double Metaphone Phonetic Key Match  │
               │ - Exact 5/6-Digit Postal/PIN Alignment │
               │ - First Digits / Street Number Match   │
               └───────────────────┬────────────────────┘
                                   │
                                   ▼
               ┌────────────────────────────────────────┐
               │ Stacking Classifier Ensemble           │
               │ - LightGBM (2,000 trees, logloss 0.018)│
               │ - XGBoost  (1,000 trees, logloss 0.024)│
               │ - Blended Probability: 0.6 LGB+0.4 XGB │
               └───────────────────┬────────────────────┘
                                   │
                                   ▼
               ┌────────────────────────────────────────┐
               │ Precision-Calibrated Thresholding      │
               │ - Optimal Operating Threshold (τ=0.940)│
               │ - Target Metric: Macro-F0.5 (0.89924)  │
               └───────────────────┬────────────────────┘
                                   │
                                   ▼
               ┌────────────────────────────────────────┐
               │ Submission Output Generation           │
               │ - output/candidate_pairs.tsv           │
               │ - output/matching_results.tsv          │
               └────────────────────────────────────────┘
```

---

## 2. Directory Structure

```
code/business_entity_resolution/
├── requirements.txt           # Package dependencies (lightgbm, xgboost, rapidfuzz, jellyfish, etc.)
├── README.md                  # System documentation and reproduction instructions
└── src/
    ├── normalize.py           # Vectorized C-level normalization, regex mappings & composite corpus builder
    ├── blocking.py            # TF-IDF inverted index candidate generator with disk caching
    ├── eval_v4_pipeline.py    # 34-feature extraction and stacking ensemble training engine
    ├── sweep_thresholds.py    # High-precision threshold sweep & calibration script
    ├── generate_outputs_v4.py # Test set streaming inference engine generating candidate_pairs.tsv & matching_results.tsv
    └── rescue_high_precision_singletons_v13.py # Disjoint high-precision singleton rescue engine
```

---

## 3. Reproduction & Execution Instructions

### Prerequisites
- Python 3.10+ / 3.11+
- Install dependencies:
  ```bash
  pip install -r code/business_entity_resolution/requirements.txt
  ```

### Step 1: Train the Stacking Ensemble
To train the LightGBM and XGBoost models:
```bash
python code/business_entity_resolution/src/eval_v4_pipeline.py
```
Outputs:
- `models/lgb_v4.pkl`: Trained LightGBM gradient boosted tree model.
- `models/xgb_v4.pkl`: Trained XGBoost gradient boosted tree model.
- `models/meta_v4.json`: Model hyperparameters, feature lists, and optimal $F_{0.5}$ threshold.

### Step 2: Generate High-Precision Test Predictions
To execute end-to-end streaming inference on the test dataset:
```bash
python code/business_entity_resolution/src/generate_outputs_v4.py
```
Outputs:
- `output/candidate_pairs.tsv`: Filtered top candidates for every Source 1 query.
- `output/matching_results.tsv`: Multi-match cluster IDs per Source 1 entity.

### Step 3: High-Precision Cross-Script & Address Singleton Rescue (v13)
To rescue false singletons at $\ge 98\%$ address similarity with 0 collisions:
```bash
python code/business_entity_resolution/src/rescue_high_precision_singletons_v13.py
```

### Step 4: Validate Outputs
Verify submission format compliance:
```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

