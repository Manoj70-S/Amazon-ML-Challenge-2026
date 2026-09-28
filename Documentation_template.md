# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** EntityLink-AI  
**Submission Date:** September 2026

---

## 1. Executive Summary
We present a high-precision, sub-linear two-stage business entity resolution architecture engineered to link heterogeneous records across `Source 1`, `Source 2`, and `Source 3` spanning multiple international jurisdictions (`US`, `India`, `France`). Our system pairs an inverted-index posting-list candidate blocker with a high-capacity **Stacking Ensemble (LightGBM 2,000 trees + XGBoost 1,000 trees)** trained on 34 specialized syntactic, phonetic (Double Metaphone), character n-gram Jaccard, token containment, and structural address/ZIP features. This architecture is specifically calibrated for the competition's macro-averaged $F_{0.5}$ metric, achieving **$\text{Macro } F_{0.5} = 0.89924$** on validation data at the calibrated threshold $\tau^* = 0.940$.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across 12M+ records revealed key structural characteristics:
- **Exact Full Name Matches are Rare:** Only 10.69% of true positive pairs have identical full names; 89.31% feature jurisdictional abbreviations (*Pvt Ltd*, *SASU*, *LLC*, *GmbH*), typos, word reorderings, or legal suffixes.
- **Strong First-Word Anchor:** 68.85% of true positive pairs share an exact first word; 69.04% share the exact phonetic Double Metaphone key of the first word.
- **Postal / PIN Code Invariance:** Over 90.16% of true matching pairs share identical 5-digit / 6-digit postal codes when present in both records.
- **High Multi-Source Interlinkage:** 72% of ground truth entities match across BOTH Source 2 and Source 3 simultaneously.
- **Extreme Class Imbalance:** Cartesian comparison yields $>10^{13}$ pairs. Within country blocks, true linkages represent $<0.01\%$ of all pairs.
- **Asymmetric Precision Weighting in $F_{0.5}$:** Because Macro-$F_{0.5}$ weights precision four times as heavily as recall ($\beta^2 = 0.25$), false positive links on singleton queries immediately reduce that query's score from $1.0$ to $0.0$. Thus, high-confidence decision boundaries ($\tau \ge 0.940$) and near-zero false-positive rates maximize expected global score.

### 2.2 Solution Strategy

**Approach Type:** Sub-linear Posting-List Inverted Blocking + 34-Dimensional Stacking Ensemble (LightGBM + XGBoost) + Precision-Calibrated Threshold Optimization + Zero-Collision Disjoint Rescue Engine.

**Core Innovations:**
1. **Domain-Specific String Normalization:** Regex standardization unifying international corporate suffixes (`Incorporated` $\rightarrow$ `inc`, `Private Limited` $\rightarrow$ `pvt ltd`, `Société par actions simplifiée` $\rightarrow$ `sasu`) and geographic unit terms (`St` $\rightarrow$ `street`, `Rd` $\rightarrow$ `road`, `Apt` $\rightarrow$ `apartment`).
2. **Posting-List Inverted-Index Blocking:** Sub-linear word $(1,2)$-gram TF-IDF inverted index with posting-list filtering (pruning high-frequency terms with posting size $> 200\text{k}$) and fast C-level accumulation (`np.concatenate` + `np.unique`).
3. **34-Dimensional Feature Representation:** Granular features capturing RapidFuzz Levenshtein, Token Sort/Set, Jaro-Winkler, Character Bigram/Trigram Jaccard, Token Containment Ratio, Jellyfish Metaphone Phonetics, Exact ZIP/PIN Code Match, and Street Number Alignment.
4. **Stacking Ensemble:** 60/40 probability blend between a 2,000-tree LightGBM GBDT and a 1,000-tree XGBoost classifier, reducing variance on out-of-distribution entity names.
5. **High-Precision Thresholding:** Fine-grained grid search across $\tau \in [0.85, 0.99]$ establishing peak performance at $\tau = 0.940$.
6. **Dual-Signal Cross-Script & Address Singleton Rescue (v13):** Recovery of false singletons using an exact address similarity threshold ($\ge 98\%$), shared building/PIN number digits, and Indic script / name token matching with global 1-to-many disjoint collision resolution, rescuing 21,707 high-confidence matches at $>96.5\%$ ground-truth precision without introducing collisions.

---

## 3. Candidate Generation (Blocking)

To reduce the $12\text{M} \times 12\text{M}$ search space to tractable candidate sets:
- **Country Partitioning:** First-order exact blocking on the `country` attribute (`US`, `India`, `France`), eliminating cross-border comparison noise and enabling zero-shot generalisation to unseen country partitions.
- **Weighted Composite Representation:** A $2\times$ weighted composite string $S = \text{Name} \parallel \text{Name} \parallel \text{Address}$ is constructed to prioritize enterprise identity while anchoring geographic context.
- **Word $(1,2)$-Gram TF-IDF Inverted Index:** Vocabulary constrained to the top 200,000 discriminative word n-grams ($\text{min\_df}=3, \text{max\_df}=0.5$, sublinear TF scaling).
- **Top-N Term Selective Retrieval:** For each query entity, the top 10 highest TF-IDF weighted terms are retrieved, filtering candidates with score $\ge 0.05$.
- **Candidate Pool:** Retrieves the top $K=50$ candidates per query, capturing $>96\%$ blocking recall while keeping the downstream pair pool tractable.

---

## 4. Matching Model

### 4.1 Feature Engineering (34 Dense Features)
For each candidate pair $(S_1, S_{\text{cand}})$, we compute 34 dense similarity signals:

1. **Name-Level Features (15):**
   - Blocker TF-IDF Cosine Similarity
   - RapidFuzz Normalized Levenshtein Distance
   - Token Sort Ratio & Token Set Ratio
   - Jaro-Winkler String Similarity
   - Partial String Matching Ratio
   - Word Token Jaccard Distance
   - Character Bigram Jaccard Similarity
   - Character Trigram Jaccard Similarity
   - Token Containment Ratio ($\frac{|T_1 \cap T_2|}{\min(|T_1|, |T_2|)}$)
   - First-Word Jellyfish Double Metaphone Match (Binary)
   - Absolute Name Length Difference & Length Ratio
   - Exact Name Match Indicator (Binary)
   - Word Token Count Disparity

2. **Address-Level Features (11):**
   - Address RapidFuzz Levenshtein & Partial Alignment
   - Address Token Sort & Token Set Ratios
   - Address Word Jaccard Distance
   - Address Character Bigram Jaccard
   - First Street Number Match (Binary)
   - Exact 5/6-Digit ZIP/PIN Code Match (Binary)
   - Shared Digits Ratio in Address Strings
   - Address Length Ratio & Empty Address Indicator

3. **Blocker Context & Relative Ranking Features (8):**
   - Candidate Rank within Query's Retrieval Block ($1 \dots K$)
   - Delta from Query's Top TF-IDF Score ($\text{max\_sim} - \text{sim}$)
   - Maximum TF-IDF Score within Block
   - Candidate Pool Size ($K$)
   - `is_top1` Blocker Flag (Binary)
   - `is_top3` Blocker Flag (Binary)
   - Source Origin Flag (`is_s2` vs `is_s3`)

### 4.2 Classification & Metric Tuning
- **Models:**
  - **LightGBM:** `num_leaves=127`, `learning_rate=0.03`, `n_estimators=2000`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_samples=30`, `scale_pos_weight=1.0` (Validation Logloss: **0.01851**).
  - **XGBoost:** `max_depth=7`, `learning_rate=0.03`, `n_estimators=1000`, `subsample=0.8`, `colsample_bytree=0.8` (Validation Logloss: **0.02453**).
- **Stacking Blend:** $p_{\text{ens}} = 0.60 \cdot p_{\text{lgb}} + 0.40 \cdot p_{\text{xgb}}$.
- **Macro-$F_{0.5}$ Formulation:**
  $$\text{Macro } F_{0.5} = \frac{1 + 0.5^2}{\frac{0.5^2}{\text{Precision}} + \frac{1}{\text{Recall}}} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$
- **Threshold Calibration Curve:**
  - $\tau = 0.500 \rightarrow \text{Macro } F_{0.5} = 0.85360$
  - $\tau = 0.700 \rightarrow \text{Macro } F_{0.5} = 0.87741$
  - $\tau = 0.900 \rightarrow \text{Macro } F_{0.5} = 0.89648$
  - $\tau = 0.920 \rightarrow \text{Macro } F_{0.5} = 0.89813$
  - $\tau = 0.930 \rightarrow \text{Macro } F_{0.5} = 0.89878$
  - $\mathbf{\tau = 0.940 \rightarrow \text{Macro } F_{0.5} = 0.89924}$ (Peak Optimal Operating Point)
  - $\tau = 0.950 \rightarrow \text{Macro } F_{0.5} = 0.89893$
  - $\tau = 0.960 \rightarrow \text{Macro } F_{0.5} = 0.89828$

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score:** Achieved **0.89924** on out-of-fold validation data.
- **Precision on Accepted Clusters:** Exceeds 94.2% across US and India test partitions.
- **False Positive Mitigation:** Strict 5/6-digit postal code verification and phonetic Double Metaphone alignment successfully prevent false linkages in multi-tenant commercial centers.
- **Inference Efficiency:** Streaming chunked inference processes 1.73M queries in $<40$ minutes with memory consumption strictly under $2.5\text{ GB}$ RAM.

---

## 6. Conclusion
The combination of posting-list candidate blocking, 34 dense lexical-phonetic features, a LightGBM+XGBoost stacking ensemble, and precision-calibrated thresholding provides a highly scalable, robust entity resolution solution designed to win the Amazon ML Challenge 2026.

---

## Appendix

### A. Code Artefacts
The complete pipeline is structured under `code/business_entity_resolution/`:
- `requirements.txt`: Python package dependencies (`lightgbm`, `xgboost`, `rapidfuzz`, `jellyfish`, `scikit-learn`, `scipy`, `pandas`, `numpy`).
- `README.md`: Architecture overview and execution instructions.
- `src/normalize.py`: Vectorized text normalization and legal entity standardization.
- `src/blocking.py`: TF-IDF inverted index candidate generator with posting-list pruning.
- `src/eval_v4_pipeline.py`: 34-feature extraction and stacking ensemble training engine.
- `src/generate_outputs_v4.py`: High-precision test inference engine producing `output/candidate_pairs.tsv` and `output/matching_results.tsv`.
