# --- Method1: Dominance and Fold-change Based Hierarchical Assignment ---

import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import List, Optional

import logging
logger = logging.getLogger(__name__)

def assign_sgRNA_identity_dominance(
    sgrna_mtx: csr_matrix,
    min_umi: int = 10,
    min_ratio_1: float = 0.8,
    min_fold_diff_12: float = 4.0,
    min_fold_diff_23: float = 3.0,
    cumulative_threshold: float = 0.9,
    save_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Hierarchical sgRNA assignment based on dominance and fold-change metrics.
    
    Logic:
    1. QC: Total UMI >= min_umi.
    2. Tier 1 (Single): Top1 Ratio >= min_ratio_1 AND Top1/Top2 >= min_fold_diff_12.
    3. Tier 2 (Doublet): Top1+2 Ratio >= cumulative_threshold AND Top2/Top3 >= min_fold_diff_23.
    4. Tier 3 (Multiplet): Capture N sgRNAs required to reach cumulative_threshold.
    
    Args:
        sgrna_mtx: csr_matrix with .barcode_names and .feature_names attributes.
        min_umi: Minimum total UMI count per cell.
        min_ratio_1: Ratio threshold for Tier 1.
        min_fold_diff_12: Fold change threshold (Top1/Top2) for Tier 1.
        min_fold_diff_23: Fold change threshold (Top2/Top3) for Tier 2.
        cumulative_threshold: Target cumulative ratio for Tier 2 and Tier 3.
        save_dir: Optional file path to save the results as CSV.
        
    Returns:
        pd.DataFrame: [barcode, n_sgrnas, sgrnas, counts, total_counts]
    """
    
    # Extract metadata
    if not isinstance(sgrna_mtx, csr_matrix) or not hasattr(sgrna_mtx, 'barcode_names'):
        raise TypeError("sgRNA_matrix must be a scipy.sparse.csr_matrix constructed from generate_count_matrix().")
    
    barcodes = getattr(sgrna_mtx, 'barcode_names', [f"Cell_{i}" for i in range(sgrna_mtx.shape[0])])
    features = getattr(sgrna_mtx, 'feature_names', [f"sg_{i}" for i in range(sgrna_mtx.shape[1])])
    features = np.array(features)
    
    results = []

    for i in range(sgrna_mtx.shape[0]):
        row = sgrna_mtx.getrow(i)
        
        # Handle empty cells
        if row.nnz == 0:
            results.append([barcodes[i], 0, "", "", 0])
            continue

        data = row.data
        indices = row.indices
        sort_idx = np.argsort(data)[::-1]
        
        counts_sorted = data[sort_idx]
        features_sorted = features[indices[sort_idx]]
        total_counts = np.sum(counts_sorted)
        
        # QC Filter
        if total_counts < min_umi:
            results.append([barcodes[i], 0, "", "", total_counts])
            continue

        ratios = counts_sorted / total_counts
        r1 = ratios[0]
        
        # Calculate fold differences (Handling division by zero with high default)
        fold_12 = counts_sorted[0] / counts_sorted[1] if len(counts_sorted) > 1 else 999.0
        fold_23 = counts_sorted[1] / counts_sorted[2] if len(counts_sorted) > 2 else 999.0

        assigned_indices = []

        # Tier 1: Single assignment
        if r1 >= min_ratio_1 and fold_12 >= min_fold_diff_12:
            assigned_indices = [0]
            
        # Tier 2: Doublet assignment
        elif len(counts_sorted) >= 2 and (ratios[0] + ratios[1]) >= cumulative_threshold and fold_23 >= min_fold_diff_23:
            assigned_indices = [0, 1]
            
        # Tier 3: Multiplet (Cumulative fallback)
        else:
            cum_ratios = np.cumsum(ratios)
            cutoff_hits = np.where(cum_ratios >= cumulative_threshold)[0]
            n_to_take = cutoff_hits[0] + 1 if len(cutoff_hits) > 0 else len(counts_sorted)
            assigned_indices = list(range(n_to_take))

        # output
        res_sgrnas = "|".join(features_sorted[assigned_indices])
        res_counts = "|".join(counts_sorted[assigned_indices].astype(str))
        
        results.append([
            barcodes[i], 
            len(assigned_indices), 
            res_sgrnas, 
            res_counts, 
            total_counts
        ])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        df_out = pd.DataFrame(results, columns=['barcode', 'n_sgrnas', 'sgrnas', 'counts', 'total_counts'])
        filename = f"sgrna_assignment_dominance_minUMI{min_umi}_minR1_{min_ratio_1}_minFold12_{min_fold_diff_12}_minFold23_{min_fold_diff_23}_cumThresh_{cumulative_threshold}.csv"
        df_out.to_csv(f"{save_dir}/{filename}", index=False)
    
    # Results statistics
    df_out = pd.DataFrame(results, columns=['barcode', 'n_sgrnas', 'sgrnas', 'counts', 'total_counts'])

    print(
        f"sgRNA Assignment Summary:\n"
        f"Total CBCs Processed: {df_out.shape[0]}\n"
        f"Unassigned CBCs (n_sgrnas=0): {df_out[df_out['n_sgrnas'] == 0].shape[0]}\n"
        f"Single sgRNA Assigned CBCs (n_sgrnas=1): {df_out[df_out['n_sgrnas'] == 1].shape[0]}\n"
        f"Doublet sgRNA Assigned CBCs (n_sgrnas=2): {df_out[df_out['n_sgrnas'] == 2].shape[0]}\n"
        f"Multiplet sgRNA Assigned CBCs (n_sgrnas>2): {df_out[df_out['n_sgrnas'] > 2].shape[0]}"
    )
    
    return df_out

# --- Method2: GMM Assignment ---

from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.optimize import brentq
from joblib import Parallel, delayed

class GMMAssigner:
    """
    Core class for sgRNA assignment using GMM global modeling.
    Supports 2/3-peak switching, peak distance constraints, and global minimum threshold protection.
    """

    def __init__(self, min_cells=100, very_few_cells=10, force_two_peak_fit=False,
                 min_peak_distance=1.5, three_peak_init_means=[[3.], [6.], [9.]], posterior=0.5,
                 n_jobs=-1, random_state=16):
        """
        Initialize GMMAssigner.
        :param min_cells: Minimum cells to allow 3-component fit.
        :param very_few_cells: Threshold below which fitting is skipped. Using Global M only.
        :param force_two_peak_fit: Force 2-peak fit regardless of cell count.
        :param min_peak_distance: Min distance between left peaks in 3-comp model.
        :param three_peak_init_means: Initial means for 3-peak GMM fitting.
        :param posterior: Posterior probability for threshold calculation (0.5 = PDF intersection).
        :param n_jobs: Parallel worker count. Default -1 (all cores).
        """

        self.min_cells = min_cells
        self.very_few_cells = very_few_cells
        if self.very_few_cells < 2:
            raise ValueError("very_few_cells must be >= 2 for model-fitting! 10 is recommended.")
        if self.min_cells < self.very_few_cells:
            raise ValueError("min_cells must be >= very_few_cells")
        
        self.force_two_peak_fit = force_two_peak_fit
        self.min_peak_distance = min_peak_distance
        self.three_peak_init_means = three_peak_init_means
        self.posterior = posterior
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        self.feature_stats = {}  # Metadata for each sgRNA
        self.global_min_threshold = None # Global M value

    def fit(self, csr_matrix, ntc_features=None, global_threshold_method='top_n_median', top_n=10, force_global_min_threshold=3):
        """
        Main entry: Fit GMM models and calculate thresholds.
        :param csr_matrix: Sparse matrix with .feature_names and .barcode_names attributes.
        :param ntc_features: List of NTC sgRNA names.
        :param global_threshold_method: 'top_n_median', 'ntc_baseline' or 'force_value'.
        :param top_n: Number of top features to consider for global M calculation.
        :param force_global_min_threshold: Minimum allowed global M (in counts).
        """
        logger.debug("Preparing data for GMM fitting...")
        feature_names = csr_matrix.feature_names
        
        # Convert to CSC format for efficient column access
        data_matrix = csr_matrix.tocsc()

        # 1. Calculate Global Minimum Threshold (M)
        self.global_min_threshold = self._calculate_global_min_threshold(
            data_matrix, feature_names, ntc_features, global_threshold_method, top_n, force_global_min_threshold
        )
        logger.info(f"Global Minimum Threshold (M) set to: {self.global_min_threshold:.4f} (log2)")

        # 2. Parallel fitting for all features
        logger.info(f"Parallel GMM fitting for {len(feature_names)} features...")
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_single_feature)(
                feature_name=name,
                counts=data_matrix[:, i].toarray().flatten(),
                global_min_threshold=self.global_min_threshold
            )
            for i, name in enumerate(feature_names)
        )

        for res in results:
            self.feature_stats[res['sgrna']] = res
        
        logger.debug("GMM Fitting completed.")

    def _calculate_global_min_threshold(self, matrix, feature_names, ntc_features, method, top_n, force_value):
        """
        if method == 'force_value': return log2(force_value + 1).
        Otherwise, Calculate global threshold M based on top features or NTCs.
        if calculated threshold < log2(force_value + 1), use log2(force_value + 1).
        """
        if method == 'ntc_baseline' and ntc_features is not None:
            ref_indices = [i for i, name in enumerate(feature_names) if name in ntc_features]
        elif method == 'top_n_median' and top_n > 0:
            total_counts = np.array(matrix.sum(axis=0)).flatten()
            ref_indices = np.argsort(total_counts)[::-1][:top_n]
        elif method == 'force_value':
            return np.log2(force_value + 1)

        ref_thresholds = []
        for idx in ref_indices:
            counts = matrix[:, idx].toarray().flatten()
            res = self._fit_single_feature(feature_names[idx], counts, global_min_threshold=None)
            if res['thresholds']:
                ref_thresholds.append(res['thresholds'][0])

        threshold = np.median(ref_thresholds) if ref_thresholds else 1.0
        return max(threshold, np.log2(force_value + 1))

    def _fit_single_feature(self, feature_name, counts, global_min_threshold):
        """Fit GMM for a single feature and calculate intersection thresholds."""
        counts_nonzero = counts[counts > 0]
        n_cells = len(counts_nonzero)
        
        if n_cells < self.very_few_cells:
            return self._build_result(feature_name, n_cells, [], None, 0)

        log_counts = np.log2(counts_nonzero + 1).reshape(-1, 1)
        
        # Determine components and fit
        if n_cells < self.min_cells or self.force_two_peak_fit:
            model = GaussianMixture(n_components=2, random_state=self.random_state, n_init=5).fit(log_counts)
            n_components = 2
        else:
            # Try 3-peak fit with constraints
            try:
                m3 = GaussianMixture(n_components=3, means_init=self.three_peak_init_means, 
                                     random_state=self.random_state).fit(log_counts)
            except:
                m3 = GaussianMixture(n_components=3, random_state=self.random_state, n_init=5).fit(log_counts)
            
            means = np.sort(m3.means_.flatten())
            if (means[1] - means[0]) >= self.min_peak_distance:
                model, n_components = m3, 3
            else:
                model = GaussianMixture(n_components=2, random_state=self.random_state, n_init=5).fit(log_counts)
                n_components = 2

        thresholds = self._compute_posterior_thresholds(model, self.posterior)

        # Apply Global M protection to T_low
        if global_min_threshold is not None and thresholds:
            if thresholds[0] < global_min_threshold:
                thresholds[0] = global_min_threshold
        
        return self._build_result(feature_name, n_cells, thresholds, model, n_components)

    def _compute_posterior_thresholds(self, model, posterior):
        """
        Find thresholds where the posterior probability of the RIGHT Gaussian
        equals a given value (p in [0.5, 1)), considering only adjacent components.
        when posterior=0.5, this is equivalent to PDF intersection points.
        """
        if not (0.5 <= posterior < 1.0):
            raise ValueError("posterior must be in [0.5, 1).")

        means = model.means_.flatten()
        sigmas = np.sqrt(model.covariances_).flatten()
        weights = model.weights_.flatten()

        # sort by mean to ensure order
        idx = np.argsort(means)
        means, sigmas, weights = means[idx], sigmas[idx], weights[idx]

        thresholds = []
        ratio = posterior / (1.0 - posterior)

        for i in range(len(means) - 1):
            def diff_posterior(x):
                left = weights[i] * norm.pdf(x, means[i], sigmas[i])
                right = weights[i+1] * norm.pdf(x, means[i+1], sigmas[i+1])
                return right - ratio * left
            try:
                thresholds.append(
                    brentq(diff_posterior, means[i], means[i+1])
                )
            except ValueError:
                # fallback: midpoint if no root found
                thresholds.append((means[i] + means[i+1]) / 2)

        return thresholds

    # def _compute_intersection_thresholds(self, model):
    #     """Find PDF intersection points using brentq."""
    #     means = model.means_.flatten()
    #     sigmas = np.sqrt(model.covariances_).flatten()
    #     weights = model.weights_.flatten()
    #     idx = np.argsort(means)
    #     means, sigmas, weights = means[idx], sigmas[idx], weights[idx]
        
    #     thresholds = []
    #     for i in range(len(means) - 1):
    #         def diff_pdf(x):
    #             return (weights[i] * norm.pdf(x, means[i], sigmas[i]) - 
    #                     weights[i+1] * norm.pdf(x, means[i+1], sigmas[i+1]))
    #         try:
    #             thresholds.append(brentq(diff_pdf, means[i], means[i+1]))
    #         except ValueError:
    #             thresholds.append((means[i] + means[i+1]) / 2)
    #     return thresholds

    def generate_report(self, csr_matrix, output_dir=None):
        """
        Generate assignment summaries (Loose, Middle, Strict).
        :param csr_matrix: Sparse matrix with .feature_names and .barcode_names.
        :return: dict of DataFrames.
        """
        logger.info("Generating reports...")
        data_matrix = csr_matrix.tocsc()
        feature_names = csr_matrix.feature_names
        feature_names = np.asarray(feature_names)
        barcodes = csr_matrix.barcode_names

        os.makedirs(output_dir, exist_ok=True) if output_dir is not None else None
        
        # Pre-calculate T_low and T_high for each feature
        t_low_map = np.full(len(feature_names), np.inf)
        t_high_map = np.full(len(feature_names), np.inf)
        
        for i, name in enumerate(feature_names):
            if name in self.feature_stats:
                th = self.feature_stats[name]['thresholds']
                if len(th) == 1:
                    t_low_map[i] = t_high_map[i] = 2**th[0] - 1
                elif len(th) >= 2:
                    t_low_map[i] = 2**th[0] - 1
                    t_high_map[i] = 2**th[1] - 1

        results = {}
        modes = ['loose', 'strict', 'middle']
        
        for mode in modes:
            logger.info(f"Processing mode: {mode}")
            rows = []
            # Row-by-row summary (efficient on CSC indices)
            for j in range(len(barcodes)):
                # Get current cell's nonzero data
                row_data = data_matrix.getrow(j)
                indices = row_data.indices
                data = row_data.data
                
                # Filter by mode-specific thresholds
                if mode == 'loose':
                    mask = data > t_low_map[indices]
                elif mode == 'strict':
                    mask = data > t_high_map[indices]
                else: # middle
                    mask = (data > t_low_map[indices]) & (data <= t_high_map[indices])
                
                sel_idx = indices[mask]
                sel_data = data[mask]
                
                n_hit = len(sel_idx)
                f_str = "|".join(feature_names[sel_idx]) if n_hit > 0 else ""
                c_str = "|".join(map(str, sel_data.astype(int))) if n_hit > 0 else ""
                
                rows.append([barcodes[j], n_hit, f_str, c_str, int(data.sum())])
            
            res_df = pd.DataFrame(rows, columns=['barcode', 'n_sgrnas', 'sgrnas', 'counts', 'total_counts'])
            results[mode] = res_df
            if output_dir is not None:
                os.makedirs(os.path.dirname(output_dir), exist_ok=True)
                filename = os.path.join(output_dir, f"assignment_gmm_{mode}.csv")
                res_df.to_csv(filename, index=False)
                
        return results

    def _build_result(self, name, n, th, model, nc):
        return {'sgrna': name, 'n_cells': n, 'thresholds': th, 'model_obj': model, 'n_components': nc}
    
    