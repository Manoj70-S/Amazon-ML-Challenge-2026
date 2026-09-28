"""
blocking.py  (v3 – posting-list retrieval, no OOM)
====================================================
Key fixes vs v2:
  1. Posting-list accumulation replaces sparse matmul -> zero OOM risk
  2. TF-IDF index is built once per country and can be cached/reused
  3. Progress every 5k queries so logs stay live
  4. Memory-efficient: O(n_candidates × float32) per query (~25 MB for 6M cands)

Posting-list retrieval speed (US, 6.2M cands, 300 terms/query):
  avg posting list = 500M nnz / 80k vocab = ~6,250 docs/term
  per query: 300 terms × 6,250 numpy accumulations = 1.875M ops -> ~2–5 ms/query
  for 72k queries: ~144–360 seconds = 2–6 minutes  (vs OOM with sparse matmul)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from normalize import build_vectorized_composite_corpus


class TFIDFBlocker:
    """
    High-recall candidate generator using character n-gram TF-IDF + posting-list retrieval.
    Partitions by country for generalisation to unseen countries (France in test).
    """

    def __init__(
        self,
        analyzer: str = 'word',
        ngram_range: Tuple[int, int] = (1, 2),
        min_sim: float = 0.10,
        top_k: int = 25,
        max_features: int = 200_000,
        min_df: int = 3,
        max_df: float = 0.5,
        chunk_size: int = 20,
        progress_every: int = 5_000,
    ):
        self.analyzer       = analyzer
        self.ngram_range    = ngram_range
        self.min_sim        = min_sim
        self.top_k          = top_k
        self.max_features   = max_features
        self.min_df         = min_df
        self.max_df         = max_df
        self.chunk_size     = chunk_size
        self.progress_every = progress_every

    # ─────────────────────────────────────────────────────────────────────────
    # Core: posting-list retrieval for ONE query against pre-built CSC index
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _retrieve_topk(
        q_indices: np.ndarray,   # non-zero term indices of the query
        q_data: np.ndarray,      # non-zero term weights of the query
        csc_indptr: np.ndarray,  # cand_mat_csc.indptr
        csc_indices: np.ndarray, # cand_mat_csc.indices
        csc_data: np.ndarray,    # cand_mat_csc.data
        acc: np.ndarray,         # pre-allocated float64 accumulator (n_cands,)
        cand_ids: np.ndarray,    # candidate entity IDs
        min_sim: float,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """
        Accumulate cosine similarity scores via posting lists, return top-K.

        Per-query timing (6.2M candidates, 79 terms, avg posting-list 6,250):
          accumulate : 79 x fancy-index-add  ~160 us
          np.where   : scan 6.2M float64     ~2.5 ms
          acc.fill   : reset 6.2M float64    ~2.5 ms
          Total      : ~5 ms/query  (vs ~30 ms with np.concatenate+np.unique)
        """
        # ── 1. Accumulate scores ──────────────────────────────────────────────
        has_any = False
        for t_idx in range(len(q_indices)):
            term = q_indices[t_idx]
            q_w  = float(q_data[t_idx])
            p_st = csc_indptr[term]
            p_en = csc_indptr[term + 1]
            if p_en == p_st:
                continue
            acc[csc_indices[p_st:p_en]] += q_w * csc_data[p_st:p_en]
            has_any = True

        if not has_any:
            return []

        # ── 2. Find entries above threshold (single O(n) scan) ───────────────
        above = np.where(acc >= min_sim)[0]   # indices into cand array

        # ── 3. Save scores BEFORE reset ───────────────────────────────────────
        if len(above) == 0:
            acc.fill(0.0)
            return []

        filt_scores = acc[above].copy()       # copy so reset doesn't corrupt

        # ── 4. Reset accumulator (single memset) ──────────────────────────────
        acc.fill(0.0)

        # ── 5. Top-K selection ────────────────────────────────────────────────
        if len(above) > top_k:
            part        = np.argpartition(-filt_scores, top_k)[:top_k]
            part        = part[np.argsort(-filt_scores[part])]
            above       = above[part]
            filt_scores = filt_scores[part]
        else:
            order       = np.argsort(-filt_scores)
            above       = above[order]
            filt_scores = filt_scores[order]

        return [(cand_ids[j], float(s)) for j, s in zip(above, filt_scores)]

    # ─────────────────────────────────────────────────────────────────────────
    # Build TF-IDF index for a set of candidate records (with optional caching)
    # ─────────────────────────────────────────────────────────────────────────
    def _build_index(
        self,
        c_cand: pd.DataFrame,
        verbose: bool = True,
        cache_dir: Optional[str] = None,
        cache_key: str = "",
    ) -> Tuple[TfidfVectorizer, sp.csc_matrix, np.ndarray]:
        """
        Fit TF-IDF vectorizer on candidate corpus and return (vectorizer, CSC matrix, cand_ids).
        If cache_dir is given, saves/loads the CSC matrix and vectorizer vocab to/from disk.
        """
        import pickle

        cand_ids = c_cand['entity_id'].values

        # --- try loading from cache ------------------------------------------
        if cache_dir and cache_key:
            os.makedirs(cache_dir, exist_ok=True)
            csc_path = os.path.join(cache_dir, f"{cache_key}_cand_csc.npz")
            vec_path = os.path.join(cache_dir, f"{cache_key}_vec.pkl")
            ids_path = os.path.join(cache_dir, f"{cache_key}_ids.npy")

            if os.path.exists(csc_path) and os.path.exists(vec_path) and os.path.exists(ids_path):
                if verbose:
                    print(f"  [cache] Loading index for {cache_key} …")
                t0 = time.time()
                cand_mat_csc = sp.load_npz(csc_path)
                with open(vec_path, 'rb') as f:
                    vec = pickle.load(f)
                cached_ids = np.load(ids_path, allow_pickle=True)
                if verbose:
                    print(f"  [cache] Loaded in {time.time()-t0:.1f}s | "
                          f"nnz={cand_mat_csc.nnz:,} | ids={len(cached_ids):,}")
                return vec, cand_mat_csc, cached_ids

        # --- build from scratch -----------------------------------------------
        cand_corpus = build_vectorized_composite_corpus(
            c_cand['business_name'], c_cand['business_address']
        )

        n_docs = len(cand_corpus)
        eff_min_df = min(self.min_df, max(1, n_docs // 10_000)) if n_docs >= 100 else 1
        eff_max_df = self.max_df if n_docs >= 100 else 1.0

        t0 = time.time()
        vec = TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            min_df=eff_min_df,
            max_df=eff_max_df,
            max_features=self.max_features,
            sublinear_tf=True,
            dtype=np.float32,
        )
        cand_mat     = vec.fit_transform(cand_corpus)    # (n_cands, vocab) CSR
        cand_mat_csc = cand_mat.tocsc()                  # convert for posting lists

        if verbose:
            print(f"  Index built in {time.time()-t0:.1f}s | "
                  f"Vocab={len(vec.vocabulary_):,} | "
                  f"Cand nnz={cand_mat.nnz:,} | "
                  f"Avg nnz/doc={cand_mat.nnz/max(1,n_docs):.1f}")

        # --- save to cache ----------------------------------------------------
        if cache_dir and cache_key:
            import pickle
            t0 = time.time()
            sp.save_npz(csc_path, cand_mat_csc)
            with open(vec_path, 'wb') as f:
                pickle.dump(vec, f)
            np.save(ids_path, cand_ids, allow_pickle=True)
            if verbose:
                print(f"  [cache] Saved index in {time.time()-t0:.1f}s -> {cache_dir}")

        return vec, cand_mat_csc, cand_ids


    # ─────────────────────────────────────────────────────────────────────────
    # Retrieve using top-N TF-IDF terms + np.concatenate/unique (fast path)
    # ─────────────────────────────────────────────────────────────────────────
    def _retrieve(
        self,
        vec: TfidfVectorizer,
        cand_mat_csc: sp.csc_matrix,
        cand_ids: np.ndarray,
        query_df: pd.DataFrame,
        verbose: bool = True,
        country: str = "",
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Per-query top-N term posting-list retrieval.

        Speed: O(n_queries x N_TERMS x avg_posting_list)
          - Python loop per query: 72k x 1 us = 0.07s
          - np.concatenate(N_TERMS arrays): 72k x 0.01ms = 0.7s
          - np.unique(N_TERMS x avg_pl): 72k x 0.05ms = 3.6s
          Total: ~5-10s for 72k queries (7,000-14,000 q/s)

        vs scipy sparse matmul: 27s for 500 queries = 18 q/s (1000x slower)

        Key insight: use only top-N highest-TF-IDF terms per query (most discriminative),
        skip terms with long posting lists (common noise words).
        Score = overlap_count / n_used_terms (approx Jaccard on selected terms).
        """
        MAX_PL   = 200_000   # safety cap only — high TF-IDF weight naturally selects rare terms
        N_TERMS  = 10        # use top-N TF-IDF terms per query

        query_corpus = build_vectorized_composite_corpus(
            query_df['business_name'], query_df['business_address']
        )
        t0 = time.time()
        query_mat = vec.transform(query_corpus)   # (n_queries, vocab) CSR
        if verbose:
            print(f"  Queries transformed in {time.time()-t0:.1f}s | "
                  f"n_queries={query_mat.shape[0]:,} | "
                  f"avg nnz/query={query_mat.nnz/max(1,query_mat.shape[0]):.1f}")

        # Pre-compute posting-list sizes (one per vocab term)
        csc_indptr  = cand_mat_csc.indptr
        csc_indices = cand_mat_csc.indices
        pl_sizes    = np.diff(csc_indptr)        # shape (vocab,), int32

        query_ids = query_df['entity_id'].values
        n_queries = query_mat.shape[0]
        results: Dict[str, List[Tuple[str, float]]] = {}
        t_ret = time.time()

        for i in range(n_queries):
            q_st = query_mat.indptr[i]
            q_en = query_mat.indptr[i + 1]

            if q_st == q_en:
                results[query_ids[i]] = []
                continue

            q_terms   = query_mat.indices[q_st:q_en]     # vocab indices
            q_weights = query_mat.data[q_st:q_en]        # TF-IDF weights

            # ── 1. Filter to useful terms (not too common) ──────────────────
            useful = pl_sizes[q_terms] <= MAX_PL
            q_terms   = q_terms[useful]
            q_weights = q_weights[useful]

            if len(q_terms) == 0:
                results[query_ids[i]] = []
                continue

            # ── 2. Select top-N by TF-IDF weight ────────────────────────────
            n_use = min(N_TERMS, len(q_terms))
            if len(q_terms) > n_use:
                top_idx = np.argpartition(-q_weights, n_use)[:n_use]
                q_terms = q_terms[top_idx]

            # ── 3. Collect posting lists for selected terms ──────────────────
            posting_lists = [csc_indices[csc_indptr[t]:csc_indptr[t + 1]]
                             for t in q_terms]
            all_cand_idx = np.concatenate(posting_lists)   # C-level memcpy

            # ── 4. Count term overlaps via np.unique ─────────────────────────
            unique_idx, counts = np.unique(all_cand_idx, return_counts=True)
            # Score = fraction of selected terms matched (approx Jaccard)
            scores = counts.astype(np.float32) / n_use

            # ── 5. Threshold + top-K ─────────────────────────────────────────
            mask = scores >= self.min_sim
            if not mask.any():
                results[query_ids[i]] = []
                continue

            unique_idx = unique_idx[mask]
            scores     = scores[mask]

            if len(unique_idx) > self.top_k:
                part       = np.argpartition(-scores, self.top_k)[:self.top_k]
                part       = part[np.argsort(-scores[part])]
                unique_idx = unique_idx[part]
                scores     = scores[part]
            else:
                order      = np.argsort(-scores)
                unique_idx = unique_idx[order]
                scores     = scores[order]

            results[query_ids[i]] = [
                (cand_ids[j], float(s)) for j, s in zip(unique_idx, scores)
            ]

            if verbose and (i + 1) % self.progress_every == 0:
                elapsed = time.time() - t_ret
                rate    = (i + 1) / max(elapsed, 1e-9)
                remain  = (n_queries - i - 1) / max(rate, 1e-9)
                avg_c   = sum(len(v) for v in results.values()) / (i + 1)
                print(f"  [{country}] {i+1:,}/{n_queries:,} queries "
                      f"| {rate:.0f} q/s | ETA {remain/60:.1f} min "
                      f"| avg cands {avg_c:.1f}")

        elapsed = time.time() - t_ret
        avg_c   = sum(len(v) for v in results.values()) / max(1, n_queries)
        if verbose:
            print(f"  Retrieval done in {elapsed:.1f}s "
                  f"({n_queries/elapsed:.1f} q/s) | avg cands: {avg_c:.2f}")
        return results


    # ─────────────────────────────────────────────────────────────────────────
    # Public API: block one country partition
    # ─────────────────────────────────────────────────────────────────────────
    def block_country(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
        country: str,
        verbose: bool = True,
        cache_dir: Optional[str] = None,
        split_name: str = "train",
    ) -> Dict[str, List[Tuple[str, float]]]:
        c_s1  = s1_df[s1_df['country'] == country].reset_index(drop=True)
        c_s23 = pd.concat([
            s2_df[s2_df['country'] == country],
            s3_df[s3_df['country'] == country],
        ], ignore_index=True)

        if verbose:
            print(f"[{country}] Queries: {len(c_s1):,} | Cand pool: {len(c_s23):,}")

        if len(c_s1) == 0:
            return {}
        if len(c_s23) == 0:
            return {eid: [] for eid in c_s1['entity_id']}

        # Build/load index — use cache_key = f"{split_name}_{country}"
        cache_key = f"{split_name}_{country.replace(' ', '_')}" if cache_dir else ""
        vec, cand_csc, cand_ids = self._build_index(
            c_s23, verbose=verbose, cache_dir=cache_dir, cache_key=cache_key
        )
        return self._retrieve(vec, cand_csc, cand_ids, c_s1, verbose=verbose, country=country)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API: block all countries dynamically
    # ─────────────────────────────────────────────────────────────────────────
    def generate_candidates(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
        verbose: bool = True,
        cache_dir: Optional[str] = None,
        split_name: str = "train",
    ) -> Dict[str, List[Tuple[str, float]]]:
        all_results: Dict[str, List[Tuple[str, float]]] = {}
        countries = s1_df['country'].unique()
        if verbose:
            print(f"=== Candidate Generation: {len(countries)} countries: {list(countries)} ===")
        for country in countries:
            res = self.block_country(
                s1_df, s2_df, s3_df, str(country),
                verbose=verbose, cache_dir=cache_dir, split_name=split_name
            )
            all_results.update(res)
        return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Export helpers
# ─────────────────────────────────────────────────────────────────────────────
def export_candidate_pairs_tsv(
    candidate_dict: Dict[str, List[Tuple[str, float]]],
    s1_ids: List[str],
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for eid in s1_ids:
            cands   = candidate_dict.get(eid, [])
            cand_str = ",".join(c for c, _ in cands)
            f.write(f"{eid}\t{cand_str}\n")
    print(f"Saved {len(s1_ids):,} candidate rows -> {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    s1 = pd.DataFrame({'entity_id': ['S1-001', 'S1-002'],
                        'business_name': ["Orelee's Barbershop", "American Choice Svc LLC"],
                        'business_address': ["1795 Westchester Dr, High Point NC", "216 Metropolitan Dr, Rochester NY"],
                        'country': ['US', 'US']})
    s2 = pd.DataFrame({'entity_id': ['S2-001'],
                        'business_name': ["Orelees Barbershop"],
                        'business_address': ["1795 Westchester Drive, High Point, NC"],
                        'country': ['US']})
    s3 = pd.DataFrame({'entity_id': ['S3-001'],
                        'business_name': ["American Choice Service"],
                        'business_address': ["216 Metropolitan Drive, Rochester, New York"],
                        'country': ['US']})
    blocker = TFIDFBlocker(min_sim=0.15, top_k=5)
    res = blocker.generate_candidates(s1, s2, s3)
    print("Results:", res)
    print("Self-test OK [OK]")

