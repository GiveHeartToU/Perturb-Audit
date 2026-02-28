import numpy as np
import pandas as pd
from scipy.stats import entropy, norm, pearsonr
from scipy import sparse
import logging

class EvaluationHelper:
    """
    Helper class for quantifying the impact of Strict Discard on sgRNA assignment.
    Supports Identity Migration, Distribution Purity, and Model Quality metrics.
    """

    def __init__(self, raw_mtx, cleaned_mtx, gex_mtx,
                 raw_assign_df, cleaned_assign_df, 
                 raw_gmm=None, cleaned_gmm=None,
                 min_total_UMI=10):
        """
        :param raw_mtx: CSR matrix (raw counts)
        :param cleaned_mtx: CSR matrix (cleaned counts)
        :param raw_assign_df: DataFrame from assignment.py (raw)
        :param cleaned_assign_df: DataFrame from assignment.py (cleaned)
        :param gmm_assigner: Optional trained GMMAssigner instance
        :param min_total_UMI: Minimum total UMI to consider a cell
        """
        self.raw_mtx = raw_mtx
        self.cleaned_mtx = cleaned_mtx
        self.gex_mtx = gex_mtx
        self.raw_assign = raw_assign_df.set_index('barcode')
        self.cleaned_assign = cleaned_assign_df.set_index('barcode')
        self.raw_gmm = raw_gmm
        self.cleaned_gmm = cleaned_gmm
        # self.gex_mtx = gex_mtx
        
        # Align barcodes to ensure consistent comparison
        common_barcodes = self.raw_assign.index.intersection(self.cleaned_assign.index)
        self.raw_assign = self.raw_assign.loc[common_barcodes]
        self.cleaned_assign = self.cleaned_assign.loc[common_barcodes]
        print(f"Aligned to {len(common_barcodes)} common cells between raw and cleaned assignments.")

        # Filter cells based on min_total_UMI if needed
        if min_total_UMI > 0:
            valid_barcodes = self.raw_assign[self.raw_assign['total_counts'] >= min_total_UMI].index
            self.raw_assign = self.raw_assign.loc[valid_barcodes]
            self.cleaned_assign = self.cleaned_assign.loc[valid_barcodes]
            print(f"Filtered to {len(valid_barcodes)} cells with >= {min_total_UMI} total UMIs.")
        
    # --- Phase 1: Identity Migration ---

    def get_sankey_data(self):
        """
        Prepare data for Sankey diagram by tracking status changes.
        Categorize into: Negative (0), Singleton (1), Multi-plet (>=2)
        """
        def categorize(n):
            if n == 0: return "Negative"
            if n == 1: return "Singlet"
            if n == 2: return "Doublet"
            return "Multi-plet"

        status_raw = pd.Categorical(
            self.raw_assign['n_sgrnas'].apply(categorize),
            categories=["Negative", "Singlet", "Doublet", "Multi-plet"],
            ordered=True
        )
        status_clean = pd.Categorical(
            self.cleaned_assign['n_sgrnas'].apply(categorize),
            categories=["Negative", "Singlet", "Doublet", "Multi-plet"],
            ordered=True
        )
        migration = pd.DataFrame({'Raw': status_raw, 'Cleaned': status_clean})
        migration_print = pd.crosstab(migration['Raw'], migration['Cleaned'])
        print(migration_print)

        return migration

    def get_multiplet_stats(self):
        """Compare distribution of n_sgrnas per cell."""
        raw_counts = self.raw_assign['n_sgrnas'].value_counts().sort_index()
        clean_counts = self.cleaned_assign['n_sgrnas'].value_counts().sort_index()
        return pd.DataFrame({'Raw': raw_counts, 'Cleaned': clean_counts}).fillna(0)

    def get_singleton_identity_shift(self):
        """
        Analyze how many singleton cells changed their assigned sgRNA identity.
        """
        # 1. filter singleton cells both in raw and cleaned
        raw_singlets = self.raw_assign[self.raw_assign['n_sgrnas'] == 1]
        clean_singlets = self.cleaned_assign[self.cleaned_assign['n_sgrnas'] == 1]
        
        common_barcodes = raw_singlets.index.intersection(clean_singlets.index)
        
        raw_id = raw_singlets.loc[common_barcodes, 'sgrnas']
        clean_id = clean_singlets.loc[common_barcodes, 'sgrnas']
        
        # 2. DataFrame of identities
        shift_df = pd.DataFrame({
            'Raw_ID': raw_id,
            'Clean_ID': clean_id
        }, index=common_barcodes)
        
        # 3. Mark changes
        shift_df['is_changed'] = shift_df['Raw_ID'] != shift_df['Clean_ID']
        
        # 4. Crosstab matrix of changes
        changed_only = shift_df[shift_df['is_changed']]
        shift_matrix = pd.crosstab(changed_only['Raw_ID'], changed_only['Clean_ID'])
        
        change_rate = shift_df['is_changed'].mean() * 100
        logging.info(f"Singleton Identity Shift Rate: {change_rate:.2f}%")
        
        return shift_df, shift_matrix, change_rate
    
    # --- Phase 2: Distribution & Purity ---

    def get_gini_improvement(self):
        """Calculate Gini coefficient for each cell in raw vs cleaned matrix."""
        def gini(array):
            if np.sum(array) == 0: return 0
            array = np.sort(array)
            index = np.arange(1, array.shape[0] + 1)
            n = array.shape[0]
            return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))

        # We sample or iterate depending on matrix size
        # Using a subset if matrix is too large for speed
        barcodes = self.raw_assign.index
        gini_raw = []
        gini_clean = []
        
        # Accessing sparse rows
        raw_csc = self.raw_mtx.tocsr()
        clean_csc = self.cleaned_mtx.tocsr()
        
        # Optimization: only check cells with any counts
        for i in range(len(barcodes)):
            r_data = raw_csc.getrow(i).data
            c_data = clean_csc.getrow(i).data
            gini_raw.append(gini(r_data) if len(r_data) > 0 else 0)
            gini_clean.append(gini(c_data) if len(c_data) > 0 else 0)
            
        return pd.DataFrame({'Gini_Raw': gini_raw, 'Gini_Cleaned': gini_clean}, index=barcodes)

    def get_dominance_scores(self):
        """Calculate Top1 / Total UMI ratio for each cell."""
        # This reflects how 'sharp' the signal is
        raw_total = self.raw_assign['total_counts']
        # Extract Top 1 count from the 'counts' string column in assignment df
        def get_top1(s):
            if not s or pd.isna(s): return 0
            counts = [int(x) for x in str(s).split('|')]
            return max(counts) if counts else 0

        raw_top1 = self.raw_assign['counts'].apply(get_top1)
        clean_top1 = self.cleaned_assign['counts'].apply(get_top1)
        
        # Ratio of Top1 to total sgRNA UMIs in that cell
        score_raw = raw_top1 / self.raw_assign['total_counts']
        score_clean = clean_top1 / self.cleaned_assign['total_counts']
        
        return pd.DataFrame({'Score_Raw': score_raw, 'Score_Cleaned': score_clean}).fillna(0)

    # --- Phase 3: Model Quality (GMM specific) ---

    def _get_ashman_d(self, assigner):
        """
        Internal helper to compute Ashman's D for each feature in the GMMAssigner.
        """
        d_results = {}
        n_components_results = {}
        n_CBC_results = {}

        for feature, stats in assigner.feature_stats.items():
            model = stats['model_obj']
            if model is None or stats['n_components'] < 2:
                continue
            
            means = model.means_.flatten()
            sigmas = np.sqrt(model.covariances_.flatten())
            idx = np.argsort(means)
            
            mu0, mu1 = means[idx[0]], means[idx[1]]
            sigma0, sigma1 = sigmas[idx[0]], sigmas[idx[1]]
            
            # Ashman's D formula
            d = np.sqrt(2) * (abs(mu1 - mu0) / np.sqrt(sigma0**2 + sigma1**2))
            d_results[feature] = d
            
            n_components_results[feature] = stats['n_components']
            n_CBC_results[feature] = stats['n_cells']

        return d_results, n_components_results, n_CBC_results
    
    def compare_peak_separation(self):
        """
        Compare Ashman's D between raw and cleaned GMM models.
        Return DataFrame: [feature, raw_d, raw_n_components, raw_n_cells, cleaned_d, cleaned_n_components, cleaned_n_cells, d_improvement]
        """
        if self.raw_gmm is None or self.cleaned_gmm is None:
            raise ValueError("Both raw and cleaned GMMAssigner instances are required for this analysis.")

        raw_d_map, raw_ncomp_map, raw_ncbc_map = self._get_ashman_d(self.raw_gmm)
        clean_d_map, clean_ncomp_map, clean_ncbc_map = self._get_ashman_d(self.cleaned_gmm)
        
        common_features = set(raw_d_map.keys()) & set(clean_d_map.keys())
        print(f"Comparing Ashman's D for {len(common_features)} common features.")

        res = []
        for f in common_features:
            res.append({
                'feature': f,
                'raw_d': raw_d_map[f],
                'raw_n_components': raw_ncomp_map[f],
                'raw_n_cells': raw_ncbc_map[f],
                'cleaned_d': clean_d_map[f],
                'cleaned_n_components': clean_ncomp_map[f],
                'cleaned_n_cells': clean_ncbc_map[f],
                'd_improvement': clean_d_map[f] - raw_d_map[f]
            })

        return pd.DataFrame(res)

    def _compute_single_conf(self, mtx, assigner):
        """
        Calculate the posterior probability that each cell belongs to 'Signal' (non-background component).
        Logic: 1 - P(Background_Component)
        """
        # 1. check inputs
        if assigner is None:
            raise ValueError("GMMAssigner instance is needed")
        if not isinstance(mtx, sparse.csr_matrix):
            raise ValueError("matrix must be a scipy sparse matrix")
        if not hasattr(mtx, 'feature_names') or not hasattr(mtx, 'barcode_names'):
            raise AttributeError("csr_matrix must have 'feature_names' and 'barcode_names' attributes")

        # 2. transform matrix to CSC for efficient column access
        csc_mtx = mtx.tocsc()
        feature_names = mtx.feature_names
        barcode_names = np.array(mtx.barcode_names)
        
        confidence_results = []

        # 3. iterate over features
        for i, feature in enumerate(feature_names):
            
            if feature not in assigner.feature_stats:
                continue
                
            stats = assigner.feature_stats[feature]
            model = stats['model_obj']
            if model is None: 
                continue

            # Extract counts for this feature across all cells
            counts = csc_mtx[:, i].toarray().flatten()
            nonzero_mask = counts > 0
            if not np.any(nonzero_mask): 
                continue
            
            # log-transform non-zero counts for GMM input
            log_vals = np.log2(counts[nonzero_mask] + 1).reshape(-1, 1)
            
            # Get posterior probabilities for all components [n_cells, n_components]
            probs = model.predict_proba(log_vals)
            
            # find background component (lowest mean)
            means = model.means_.flatten()
            bg_idx = np.argmin(means)
            
            # Signal probability = 1 - P(Background)
            signal_probs = 1 - probs[:, bg_idx]
            
            confidence_results.append(pd.DataFrame({
                'barcode': barcode_names[nonzero_mask],
                'feature': feature,
                'counts': counts[nonzero_mask],
                'signal_prob': signal_probs
            }))

        if not confidence_results:
            return pd.DataFrame(columns=['barcode', 'feature', 'counts', 'signal_prob'])
        
        return pd.concat(confidence_results, ignore_index=True)

    def compare_assignment_confidence(self):
        """
        Compare assignment certainty for Raw and Cleaned tasks.
        It is used to observe whether the posterior probability distribution shifts to 0 or 1 after cleaning PCR-chimera.
        """
        if self.raw_gmm is None or self.cleaned_gmm is None:
            raise ValueError("Both raw and cleaned GMMAssigner instances are required for this analysis.")
        
        logging.info("Calculating Raw Data posterior probabilities...")
        raw_conf = self._compute_single_conf(self.raw_mtx, self.raw_gmm)
        
        logging.info("Calculating Cleaned Data posterior probabilities...")
        clean_conf = self._compute_single_conf(self.cleaned_mtx, self.cleaned_gmm)
        
        if raw_conf.empty or clean_conf.empty:
            logging.warning("One of the confidence DataFrames is empty. Cannot compare.")
            return pd.DataFrame(columns=['barcode', 'feature', 'counts_raw', 'signal_prob_raw', 'counts_clean', 'signal_prob_clean'])

        # inner join on (barcode, feature)
        comparison = pd.merge(
            raw_conf, clean_conf, 
            on=['barcode', 'feature'], 
            suffixes=('_raw', '_clean')
        )

        if comparison.empty:
            logging.warning("No common (barcode, feature) pairs found between raw and cleaned confidence data.")
            
        return comparison

    # --- Phase 4: Cross Modality Consistency ---

    def compute_library_correlation(self, gex_mtx):
        """
        Calculate Pearson R between GEX Total UMI and CRISPR Total UMI.
        To verify that technical positive correlations introduced by PCR amplification are removed.
        
        Args:
            gex_mtx: GEX(scipy.sparse.csr_matrix)  .barcode_names attribute is required.
        Returns:
            dict: {'raw_r': float, 'clean_r': float, 'raw_p': float, 'clean_p': float}
        """
        if not hasattr(gex_mtx, 'barcode_names'):
             raise AttributeError("gex_mtx must have 'barcode_names' attribute")

        # 1. Calculate log1p total UMI for GEX and CRISPR
        gex_barcodes = np.array(gex_mtx.barcode_names)
        gex_sums = np.log1p(np.array(gex_mtx.sum(axis=1)).flatten())

        raw_barcodes = np.array(self.raw_mtx.barcode_names)
        raw_sums = np.log1p(np.array(self.raw_mtx.sum(axis=1)).flatten())

        clean_barcodes = np.array(self.cleaned_mtx.barcode_names)
        clean_sums = np.log1p(np.array(self.cleaned_mtx.sum(axis=1)).flatten())

        # 2. Align Barcode (GEX and CRISPR cell numbers may not match)
        common_barcodes = np.intersect1d(
            np.intersect1d(gex_barcodes, raw_barcodes),
            clean_barcodes
        )
        
        if len(common_barcodes) == 0:
            raise ValueError("No common barcodes found between GEX and CRISPR data.\nTerminating correlation calculation. Check barcode naming consistency.")

        # Build index mapping
        gex_idx = {bc: i for i, bc in enumerate(gex_barcodes)}
        raw_idx = {bc: i for i, bc in enumerate(raw_barcodes)}
        clean_idx = {bc: i for i, bc in enumerate(clean_barcodes)}
        
        idx_gex = [gex_idx[bc] for bc in common_barcodes]
        idx_raw = [raw_idx[bc] for bc in common_barcodes]
        idx_clean = [clean_idx[bc] for bc in common_barcodes]

        vec_gex = gex_sums[idx_gex]
        vec_raw = raw_sums[idx_raw]
        vec_clean = clean_sums[idx_clean]
        
        # 3. valculate Pearson R
        r_raw, p_raw = pearsonr(vec_gex, vec_raw)
        r_clean, p_clean = pearsonr(vec_gex, vec_clean)
        
        return {
            'raw_r': r_raw,
            'clean_r': r_clean,
            'raw_p': p_raw,
            'clean_p': p_clean,
            # For plotting purpose
            'plot_data': pd.DataFrame({
                'gex_log_umi': vec_gex,
                'raw_log_umi': vec_raw,
                'clean_log_umi': vec_clean
            })
        }

