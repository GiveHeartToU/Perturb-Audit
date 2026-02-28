import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

class NTCQualityControl:
    """
    Evaluate the quality of Non-Targeting Controls (NTCs) based on their behavior in the latent space.
    The evaluation is performed in the context of singlet sgRNA assignments.
    """
    def __init__(self, adata, embedding_key='cellbender_embedding', ntc_label='NEG_CTRL'):
        self.adata = adata
        self.emb_key = embedding_key
        self.ntc_label = ntc_label
        self.metrics_df = pd.DataFrame() 
        self.pairwise_matrices = {}

    def evaluate_method(self, method_name, sin_sg_col, min_total_cells=30, min_sg_cells=5):
        """
        In the context of only singlet sgRNA assignments.
        Evaluate NTC quality for a given sgRNA assignment results.

        :param method_name: Name of the sgRNA assignment method (for labeling)
        :param sin_sg_col: Column name in adata.obs for singlet sgRNA assignments
        :param min_total_cells: Minimum number of NTC cells required to perform evaluation
        """
        # 1. Abtain NTC subset
        mask = (self.adata.obs[sin_sg_col] != 'NA') & \
               (self.adata.obs[sin_sg_col].str.contains(self.ntc_label))
        
        subset = self.adata[mask].copy()
        
        if len(subset) < min_total_cells:
            print(f"[{method_name}] Too few NTC cells ({len(subset)}). Skipping.")
            return

        X = subset.obsm[self.emb_key]
        sgrnas = subset.obs[sin_sg_col].values
        unique_sgrnas = np.unique(sgrnas)
        
        # 2. Calculate ensemble centroid with all NTC cells
        ensemble_centroid = np.mean(X, axis=0).reshape(1, -1)
        
        # 3. Calculate individual sgRNA centroids and their shifts
        centroids = []
        shifts = []
        counts = []
        valid_sgrnas = []
        
        for sg in unique_sgrnas:
            sg_mask = sgrnas == sg
            n_cells = np.sum(sg_mask)
            # if n_cells < 5: continue # skip sgRNAs with too few cells
            
            centroid = np.mean(X[sg_mask], axis=0).reshape(1, -1)
            dist_to_ensemble = pairwise_distances(centroid, ensemble_centroid)[0][0]
            
            centroids.append(centroid[0])
            shifts.append(dist_to_ensemble)
            counts.append(n_cells)
            valid_sgrnas.append(sg)
            
        if not valid_sgrnas:
            return

        # 4. Determine trustworthiness based on shift distances
        shifts = np.array(shifts)
        mean_shift = np.mean(shifts)
        std_shift = np.std(shifts)
        # outlier threshold: mean + 1.96 * std (95% CI)
        threshold = mean_shift + 1.96 * std_shift
        is_trusted = shifts <= threshold
        if min_sg_cells > 0:
            is_trusted = is_trusted & (np.array(counts) >= min_sg_cells)

        # 5. save metrics
        for i, sg in enumerate(valid_sgrnas):
            self.metrics_df = pd.concat([self.metrics_df, pd.DataFrame({
                'Method': [method_name],
                'sgRNA': [sg],
                'N_Cells': [counts[i]],
                'Shift_Distance': [shifts[i]],
                'Is_Trusted': [is_trusted[i]]
            })], ignore_index=True)
            
        # 6. Pairwise distance matrix among centroids (Heatmap)
        centroids_arr = np.array(centroids)
        pdist_mat = pairwise_distances(centroids_arr)
        df_dist = pd.DataFrame(pdist_mat, index=valid_sgrnas, columns=valid_sgrnas)
        self.pairwise_matrices[method_name] = df_dist
        
        print(f"[{method_name}] Evaluated {len(valid_sgrnas)} NTCs. {sum(is_trusted)} Trusted.")

    def get_consensus_ntc(self, threshold_ratio=0.75):
        """
        Consolidate trustworthiness across multiple methods.
        :param threshold_ratio: Minimum ratio of methods that must trust an NTC for it to be considered consensus trusted.
        """
        if self.metrics_df.empty:
            return []
            
        stats = self.metrics_df.groupby('sgRNA')['Is_Trusted'].mean()
        consensus_list = stats[stats >= threshold_ratio].index.tolist()
        
        print(f"Consensus Logic: NTC must be trusted in >{threshold_ratio*100}% of methods.")
        print(f"Final Consensus NTCs ({len(consensus_list)}): {consensus_list}")
        return consensus_list

    # --- Plot ---

    def plot_pairwise_heatmap(self, method_name, ax=None):
        """1: Pairwise Distance Heatmap (The 'Alien' Detector)"""
        if method_name not in self.pairwise_matrices:
            return
        
        df_dist = self.pairwise_matrices[method_name]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
            
        # clustered heatmap
        sns.heatmap(df_dist, cmap="viridis", annot=True, fmt=".1f", square=True, 
                    cbar_kws={'label': 'Euclidean Distance'}, ax=ax, cbar=False)
        ax.set_title(f"Pairwise Distances: {method_name}")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')

        
    def plot_shift_lollipop(self, ax=None):
        """2: Lollipop (Drift Plot)"""
        if self.metrics_df.empty: return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        # order by Method and Shift_Distance
        plot_data = self.metrics_df.sort_values(['Method', 'Shift_Distance'])
        
        sns.pointplot(data=plot_data, x='sgRNA', y='Shift_Distance', hue='Method', 
                      join=False, dodge=0.5, palette='tab10', scale=0.7, ax=ax)
        
        # reference line: global average shift
        avg_shift = plot_data['Shift_Distance'].mean()
        ax.axhline(avg_shift, ls='--', color='grey', alpha=0.5, label='Global Avg Shift')
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
        ax.set_xlabel("")
        ax.set_title("NTC Drift: Distance to Ensemble Centroid(Grey slash = Global Avg)")
        ax.set_ylabel("Euclidean Distance")
        ax.grid(axis='y', linestyle=':', alpha=0.5)

    def plot_trust_matrix(self, ax=None):
        """3: Consensus View with cell counts annotated"""
        if self.metrics_df.empty:
            return

        # Pivot: trust matrix (for color)
        trust_matrix = self.metrics_df.pivot(
            index='sgRNA',
            columns='Method',
            values='Is_Trusted'
        ).fillna(False)

        # Pivot: cell counts (for annotation)
        count_matrix = self.metrics_df.pivot(
            index='sgRNA',
            columns='Method',
            values='N_Cells'
        ).fillna(0).astype(int)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        # Heatmap: color by trust, annotate by N_Cells
        sns.heatmap(
            trust_matrix,
            cmap=["lightgrey", "#2ecc71"],
            linewidths=1,
            linecolor='white',
            cbar=False,
            annot=count_matrix,
            fmt='d',
            annot_kws=dict(weight='bold', color='black'),
            ax=ax
        )

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgrey', edgecolor='w', label='Unstable / Outlier'),
            Patch(facecolor='#2ecc71', edgecolor='w', label='Trusted')
        ]

        ax.legend(
            handles=legend_elements,
            ncol=2, 
            frameon=False,
            loc='upper left',
            bbox_to_anchor=(0.75, 1.08), 
            fontsize='small',
            handlelength=1.5,
            columnspacing=1.5
        )

        ax.set_title("NTC Trustworthiness Across Methods (with Cell Counts)")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
        # ax.set_yticklabels(ax.get_yticklabels(), rotation=45, ha='right')

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None
    
from scipy.stats import sem
from sklearn.metrics import silhouette_samples, pairwise_distances
from scipy.sparse import issparse

class TranscriptomeEvaluator:
    def __init__(self, adata, embedding_key='cellbender_embedding', 
                 ntc_label='NEG_CTRL', 
                 ntc_sgrnas=None, reference_method_sgRNA_col=None
                 ):
        """
        Initializes the evaluator.
        :param adata: AnnData object containing embedding and assignment results.
        :param embedding_key: The latent space key to use (e.g., 'X_pca' or 'cellbender_embedding').
        :param ntc_label: Keyword for NTCs (used to identify the control group).
        :param ntc_sgrnas: List of specific sgRNA names considered as NTCs (overrides ntc_label if provided). (e.g. ['NEG-1', 'NEG-3'])
        :param reference_method_sgRNA_col: Column name in adata.obs to use as reference for Distance to NTC calculation. (e.g. 'sin_sg_Domin_cellbender_clean')
        """
        self.adata = adata
        self.emb_key = embedding_key
        self.ntc_label = ntc_label
        self.ntc_sgrnas = ntc_sgrnas
        
        # Pre-compute the neighbor graph (for kNN Purity) to avoid redundant calculations.
        if 'neighbors' not in adata.uns:
            print(f"Computing neighbors based on {embedding_key}...")
            sc.pp.neighbors(adata, use_rep=embedding_key, n_neighbors=30)
        else:
            print("Using existing neighbor graph in adata.uns['neighbors'].")

        self.global_ntc_centroid = None
        if ntc_sgrnas and reference_method_sgRNA_col:
            print(f"Calculating Global NTC Centroid using {reference_method_sgRNA_col}...")
            print(f"Trusted NTC sgRNAs: {ntc_sgrnas}")
            # Assuming reference_method_col saved sgRNA ID
            ref_series = adata.obs[reference_method_sgRNA_col].astype(str)
            ntc_mask = ref_series.isin(ntc_sgrnas)
            
            if ntc_mask.sum() < 10:
                print(f"WARNING: Only {ntc_mask.sum()} cells found for Global NTC in reference method!")
            else:
                self.global_ntc_centroid = np.mean(adata.obsm[embedding_key][ntc_mask], axis=0).reshape(1, -1)
                print(f"Global NTC Centroid fixed based on {ntc_mask.sum()} cells.")
        else:
            print("Warning: No NTC info provided. E-distance metrics will be skipped.")

    def calculate_cell_metrics(self, level_col, method_name):
        """
        Calculates cell-level metrics: Silhouette, kNN Purity and Distance to NTC.
        :param level_col: Column name in adata.obs for grouping (e.g., 'Gene' or 'sgRNA').
        :param method_name: Name of the current method (for output labeling).
        """
        # 1. Filter data: Keep only singlets (non-NA and no ｜).
        is_valid_gene = (self.adata.obs[level_col].astype(str).ne('NA')) & \
                (~self.adata.obs[level_col].astype(str).str.contains(self.ntc_label)) & \
               (~self.adata.obs[level_col].astype(str).str.contains('|', regex=False)) 
        # 2. filter trusted NTCs
        if self.ntc_sgrnas:
            is_trusted_ntc = self.adata.obs[f'sin_sg_{method_name}'].isin(self.ntc_sgrnas)
        else:
            is_trusted_ntc = False
        # 3. Union: Keep singlets and trusted NTCs
        mask = is_valid_gene | is_trusted_ntc
        
        subset = self.adata[mask].copy()
        print(f"Calculating metrics for {method_name} with \n{len(subset)} singlet out of {self.adata.n_obs} total cells.")
        
        if len(subset) < 50:
            print(f"Skipping {method_name}: Too few cells ({len(subset)}).")
            return None

        X = subset.obsm[self.emb_key]
        labels = subset.obs[level_col].values
        
        # --- A. Silhouette Score (Cell-level) ---
        sil_vals = silhouette_samples(X, labels, metric='euclidean')
        
        # --- B. kNN Purity (Cell-level) ---
        # Use subset indices to map back to the global adjacency matrix.
        global_indices = self.adata.obs_names.get_indexer(subset.obs_names)
        # Extract the subgraph for the subset of cells. This is a weight(0-1) matrix.
        connectivities = self.adata.obsp['connectivities'][global_indices, :][:, global_indices]
        
        # Convert to LIL format for efficient structural modification
        connectivities = connectivities.tolil()
        connectivities.setdiag(0)
        
        # Convert back to CSR for efficient matrix-vector products
        connectivities = connectivities.tocsr()
        connectivities.eliminate_zeros()

        # Vectorized purity calculation
        # Create a one-hot matrix where rows are cells and columns are unique labels.
        unique_labels, integer_labels = np.unique(labels, return_inverse=True)
        label_matrix = np.eye(len(unique_labels))[integer_labels]

        # For each cell, sum the labels of its neighbors. ()
        neighbor_label_sums = connectivities @ label_matrix
        
        # The purity is the proportion of neighbors with the same label.
        # This is the value in the column corresponding to the cell's own label. All CBC Vectorization loc trick:
        purity_scores = neighbor_label_sums[np.arange(len(labels)), integer_labels]
        
        # Normalize by the number of neighbors for each cell. 
        # (Slicing made Neighborhood size variable)
        n_neighbors = np.asarray(connectivities.sum(axis=1)).flatten()
        # Avoid division by zero for cells with no neighbors.
        purity_scores = np.divide(purity_scores, n_neighbors, where=n_neighbors > 0, out=np.zeros_like(purity_scores, dtype=float))

        # --- C. Calculate Distance to Method-specific NTC ---
        dists_to_ntc = np.full(len(labels), np.nan)
        if self.global_ntc_centroid is not None:
            # (N,1) array of distances to the fixed&trusted global NTC centroid
            d = pairwise_distances(X, self.global_ntc_centroid).flatten()
            dists_to_ntc = d
        
        # --- D. Construct results DataFrame ---
        res_df = pd.DataFrame({
            'Barcode': subset.obs_names,
            'Group': labels,
            'Silhouette': sil_vals,
            'kNN_Purity': purity_scores,
            'Dist_to_NTC': dists_to_ntc,
            'Method': method_name
        })    
        return res_df

    def aggregate_metrics(self, cell_metrics_df):
        """
        Aggregates cell-level metrics to the Group (Gene/sgRNA) level, calculating mean and SEM.
        """
        if cell_metrics_df is None or cell_metrics_df.empty:
            return pd.DataFrame()

        agg_funcs = {
            'Silhouette': ['mean', sem, 'count'],
            'kNN_Purity': ['mean', sem],
            'Dist_to_NTC': ['mean', sem]
        }
        
        # Filter out columns that might not exist (e.g., Dist_to_NTC)
        valid_agg_funcs = {k: v for k, v in agg_funcs.items() if k in cell_metrics_df.columns}
        
        agg_df = cell_metrics_df.groupby(['Method', 'Group']).agg(valid_agg_funcs)
        
        # Flatten multi-level column index
        agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
        agg_df.rename(columns={
            'Silhouette_count': 'Cell_Count'
        }, inplace=True)
        
        return agg_df.reset_index()
