import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, issparse, coo_matrix
from typing import Optional, List

def generate_count_matrix(
    umi_df: pd.DataFrame,
    barcode_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    collisions_df: Optional[pd.DataFrame] = None,
    library_id: int = 1,  # Target library index (e.g., GEX=0, sgRNA=1)
    apply_strict_discard: bool = False,
    min_reads_per_umi: int = 3,
    feature_name_col: str = 'name'  # Column from feature_df to use for matrix column names
) -> csr_matrix:
    """
    Generates a sparse UMI count matrix (CBCs x Features) for the specified library, 
    with optional Strict Discard correction applied.

    Args:
        umi_df (pd.DataFrame): all UMI records.
        barcode_df (pd.DataFrame): metadata for cell barcodes.
        feature_df (pd.DataFrame): metadata for features (genes or sgRNAs).
        collisions_df (Optional[pd.DataFrame]): necessary for Strict Discard; defines colliding molecules.
        library_id (int): Target library for matrix generation. Generally 0 for GEX, 1 for sgRNA libraries. 
        apply_strict_discard (bool): Whether to apply Strict Discard collision correction. Default is False.
        min_reads_per_umi (int): Minimum read support per UMI to be included. Default is 3.
        feature_name_col (str): Column in feature_df to use for matrix column names. Default is 'name'. ('id' also pemitted)

    Returns:
        csr_matrix: (CBCs x Features) UMI count sparse matrix.
        csr_matrix.feature_names: List of feature names corresponding to matrix columns.
        csr_matrix.barcode_names: List of barcode strings corresponding to matrix rows.
    """

    print("-"*50)
    if apply_strict_discard and library_id == 0:
        matrix_type = "GEX_cleaned"
    elif apply_strict_discard and library_id == 1:
        matrix_type = "sgRNA_cleaned"
    elif not apply_strict_discard and library_id == 0:
        matrix_type = "GEX_raw"
    else:
        matrix_type = "sgRNA_raw"
    print(f"Generating [{matrix_type}] UMI count matrix...")
    
    # --- 1. Initial Filtering based on library_id and min_reads ---
    # Filter for the target library (GEX or sgRNA)
    current_df = umi_df[umi_df['library_idx'] == library_id].copy()
    print(f"Filtered UMI records to library_idx={library_id}; Remaining records: {len(current_df)}")
    
    # Filter based on read support (optional, but standard QC)
    if min_reads_per_umi > 1:
        current_df = current_df[current_df['count'] >= min_reads_per_umi]
        print(f"Filtered UMI records to those with at least {min_reads_per_umi} reads; Remaining records: {len(current_df)}")

    # Ensure necessary columns are integer types for efficiency
    current_df[['barcode_idx', 'feature_idx']] = current_df[['barcode_idx', 'feature_idx']].astype(int)

    # --- 2. Apply Strict Discard (Collision Removal) ---
    if apply_strict_discard and collisions_df is not None and not collisions_df.empty:
        # Create a unique identifier for collision molecules: (barcode_idx, umi)
        # Using a MultiIndex for high-performance set operations
        collision_set = pd.MultiIndex.from_frame(collisions_df[['barcode_idx', 'umi']])
        current_multi_index = pd.MultiIndex.from_frame(current_df[['barcode_idx', 'umi']])
        mask_is_collision = current_multi_index.isin(collision_set)
        current_df = current_df[~mask_is_collision].copy()
        print(f"Applied Strict Discard collision correction; Remaining records: {len(current_df)}")

    # --- 3. Data Aggregation (UMI Counting) ---
    # Only terms with non-zero counts are the basis for constructing sparse matrices.
    umi_counts = current_df.groupby(['barcode_idx', 'feature_idx']).size().reset_index(name='umi_count')

    # --- 4. Matrix Building (Sparse Matrix Generation) ---
    
    # --- Prepare Column Mapping (Features) ---
    # Identify features present in the current library (by feature_idx)
    present_feature_indices = umi_counts['feature_idx'].unique()
    # Locate through feature_df.index == feature_idx
    target_features = feature_df.loc[feature_df.index.intersection(present_feature_indices)]
    # Create the {feature_idx: column index} map
    feature_map = {idx: col_idx for col_idx, idx in enumerate(target_features.index)}
    
    # --- Prepare Row Mapping (CBCs) ---
    # Identify unique barcodes present in the counts. Sort for consistent ordering.
    unique_barcodes = np.sort(umi_counts['barcode_idx'].unique())
    # Create the {Barcode ID: Row Index} map
    barcode_map_series = pd.Series(
        data=np.arange(len(unique_barcodes)),
        index=unique_barcodes
    )
    barcode_map = barcode_map_series.to_dict()

    # --- Final Data Assembly ---
    # Map the barcode_idx and feature_idx values to the sparse matrix indices (rows and columns)
    rows = umi_counts['barcode_idx'].map(barcode_map).values
    cols = umi_counts['feature_idx'].map(feature_map).values
    data = umi_counts['umi_count'].values
    
    # Check for mapping issues
    if np.any(pd.isnull(rows)) or np.any(pd.isnull(cols)):
        # This error handling remains critical
        raise ValueError("Barcode or Feature mapping failed during matrix creation.")

    # Create Sparse Matrix (CSR format is ideal for row-wise operations/filtering)
    matrix_sparse = csr_matrix(
        (data, (rows, cols)), 
        shape=(len(unique_barcodes), len(target_features))
    )
    
    # --- Optional Metadata for User/Debugging ---
    # Store the actual names/IDs for columns and rows
    matrix_sparse.feature_names = target_features[feature_name_col].tolist()
    # Validate uniqueness of feature names
    final_feature_names = matrix_sparse.feature_names
    if len(final_feature_names) != len(set(final_feature_names)):
        duplicates = pd.Series(final_feature_names).value_counts()
        duplicates = duplicates[duplicates > 1].index.tolist()
    
        # Reraise the exception for safety, as non-unique columns can break downstream analysis
        raise ValueError(
            f"Critical Error: Feature names in column '{feature_name_col}' are not unique. "
            f"Duplicates found: {duplicates}"
        )
    else:
        print(f"Feature names in column '{feature_name_col}' are unique. Proceeding.")
    
    # Get the human-readable barcode strings
    barcode_mask = barcode_df['barcode_idx'].isin(unique_barcodes) & barcode_df['library_idx'] == library_id
    barcode_names = barcode_df[barcode_mask].sort_values(by='barcode_idx')['barcode'].tolist()
    matrix_sparse.barcode_names = barcode_names
    # validate uniqueness of barcode names
    final_barcode_names = matrix_sparse.barcode_names
    if len(final_barcode_names) != len(set(final_barcode_names)):
        duplicates = pd.Series(final_barcode_names).value_counts()
        duplicates = duplicates[duplicates > 1].index.tolist()
    
        # generally should not happen in one batch of single-cell data
        raise ValueError(
            f"Critical Error: Barcode names are not unique. "
            f"Duplicates found: {duplicates}"
        )
    else:
        print("Barcode names are unique. Proceeding.")

    library_dict = {0: "GEX", 1: "sgRNA"}
    if apply_strict_discard:
        print(f"Sparse matrix ({library_dict[library_id]}) generated of with Strict Discard collision correction applied.")
    else:
        print(f"Sparse matrix ({library_dict[library_id]}) generated without collision correction.")
    print(f"shape {matrix_sparse.shape} (CBCs x Features).")
    print("\n")

    return matrix_sparse

def align_sparse_matrix(
    target_matrix: csr_matrix,
    reference_matrix: csr_matrix
) -> csr_matrix:
    """
    Aligns a sparse UMI count matrix (target_matrix) to the exact row (barcode) 
    and column (feature) dimensions of a reference matrix (reference_matrix), 
    performing zero-padding for missing entries.

    This function assumes the target matrix and reference matrix have 'barcode_names' and 'feature_names' attributes attached.

    Args:
        target_matrix (csr_matrix): The matrix to be aligned (e.g., sgrna_mtx_cleaned).
        reference_matrix (csr_matrix): The matrix providing the target dimensions (e.g., sgrna_mtx_raw).

    Returns:
        csr_matrix: The newly aligned matrix with the correct shape, zero-padding, and metadata.
    """
    
    print("-"*50)
    # 1. Parameter Validation
    if not issparse(target_matrix) or not issparse(reference_matrix):
        raise TypeError("Both target_matrix and reference_matrix must be sparse matrices produced by 'generate_count_matrix'.")
        
    if not hasattr(reference_matrix, 'barcode_names') or not hasattr(reference_matrix, 'feature_names'):
        raise AttributeError("Reference matrix must have 'barcode_names' and 'feature_names' attributes.")
        
    ref_barcodes: List[str] = reference_matrix.barcode_names
    ref_features: List[str] = reference_matrix.feature_names
    
    # Check if target matrix is already aligned
    # if (len(target_matrix.barcode_names) == len(ref_barcodes) and 
    #     all(target_matrix.barcode_names == ref_barcodes) and
    #     len(target_matrix.feature_names) == len(ref_features) and
    #     all(target_matrix.feature_names == ref_features)):
        
    #     return target_matrix.copy()
    if (
        len(target_matrix.barcode_names) == len(ref_barcodes) and
        target_matrix.barcode_names == ref_barcodes and
        len(target_matrix.feature_names) == len(ref_features) and
        target_matrix.feature_names == ref_features
    ):
        return target_matrix.copy()


    # 2. Conversion to Dense/DataFrame for Reindexing
    # Convert target matrix to COO(Coordinate Format) for easy extraction of non-zero entries and their indices
    coo = target_matrix.tocoo()
    
    # Create temporary DataFrame from non-zero entries
    temp_df = pd.DataFrame({
        'UMI_Count': coo.data,
        'barcode_name': np.array(target_matrix.barcode_names)[coo.row],
        'feature_name': np.array(target_matrix.feature_names)[coo.col]
    })
    
    if temp_df.empty:
        print("Warning: Target matrix is empty. Creating an empty aligned matrix.")
        # If empty, return a zero matrix of the reference shape
        aligned_matrix = csr_matrix((len(ref_barcodes), len(ref_features)), dtype=target_matrix.dtype)
        aligned_matrix.barcode_names = ref_barcodes
        aligned_matrix.feature_names = ref_features
        return aligned_matrix

    # 3. Create Full Index Maps for Reindexing (Faster lookup)
    # Use pandas Categorical type for efficient mapping if necessary, but simple Series mapping is usually sufficient
    barcode_to_idx = {name: i for i, name in enumerate(ref_barcodes)}
    feature_to_idx = {name: i for i, name in enumerate(ref_features)}

    # Check for features/barcodes in target that are *not* in reference (shouldn't happen, but good check)
    if not temp_df['barcode_name'].isin(barcode_to_idx).all():
        missing_barcodes = temp_df.loc[~temp_df['barcode_name'].isin(barcode_to_idx), 'barcode_name'].unique()
        print(f"Warning: Check your data! Some barcodes in target_matrix are not in reference_matrix and will be discarded: {missing_barcodes}")
        temp_df = temp_df[temp_df['barcode_name'].isin(barcode_to_idx)]
    if not temp_df['feature_name'].isin(feature_to_idx).all():
        missing_features = temp_df.loc[~temp_df['feature_name'].isin(feature_to_idx), 'feature_name'].unique()
        print(f"Warning: Check your data! Some features in target_matrix are not in reference_matrix and will be discarded: {missing_features}")
        temp_df = temp_df[temp_df['feature_name'].isin(feature_to_idx)]
    
    # 4. Rebuild COO components based on reference indices (Zero-padding occurs here implicitly) 
    # Map names back to the new row/column indices based on the reference matrix
    new_row_ind = temp_df['barcode_name'].map(barcode_to_idx).values
    new_col_ind = temp_df['feature_name'].map(feature_to_idx).values
    
    # Check if mapping failed (NaNs introduced)
    if np.any(pd.isnull(new_row_ind)) or np.any(pd.isnull(new_col_ind)):
        raise ValueError("Mapping failed: Barcodes or features in target_matrix were not found in reference_matrix.")

    # 5. Reconstruct the Aligned Sparse Matrix
    
    aligned_matrix = coo_matrix(
        (temp_df['UMI_Count'].values, (new_row_ind, new_col_ind)), 
        shape=(len(ref_barcodes), len(ref_features)),
        dtype=target_matrix.dtype
    ).tocsr()
    
    # 6. Attach Metadata
    aligned_matrix.barcode_names = ref_barcodes
    aligned_matrix.feature_names = ref_features

    print(f"Original target matrix shape: {target_matrix.shape}, Reference matrix shape: {reference_matrix.shape}")
    print(f"Aligned matrix shape: {aligned_matrix.shape} (CBCs x Features).")
    print("-"*50 + "\n")
    return aligned_matrix

def generate_count_matrix_from_adata(
    adata,
    feature_type: str,
    feature_name_col: str = 'name',
    layer: Optional[str] = None) -> csr_matrix:
    """
    Generate a csr_matrix from an AnnData object for a given feature type and layer.

    Args:
        adata: AnnData object.
        feature_type (str): Value from adata.var['feature_type'], e.g.
            'Gene Expression' or 'CRISPR Guide Capture'.
        faeture_name_col (str): Column in adata.var to use as feature names,
            e.g. 'gene_name' or 'gene_id'. Default: 'name'.
        layer (Optional[str]): Layer in adata.layers to use. If None, use
            adata.X. Typical choices: 'raw', 'cellbender'.

    Returns:
        csr_matrix: Sparse matrix (cells x features) with
            `.barcode_names` and `.feature_names` attributes attached. (same as `generate_count_matrix()`)
    """
    # subset var by feature_type
    if 'feature_type' not in adata.var.columns:
        raise ValueError("AnnData.var must contain a 'feature_type' column.")
    if feature_type not in adata.var['feature_type'].unique():
        raise ValueError(f"feature_type '{feature_type}' not found in adata.var['feature_type'].")

    var_mask = adata.var['feature_type'] == feature_type
    var_subset = adata.var[var_mask]

    if var_subset.empty:
        raise ValueError(f"No features found for feature_type '{feature_type}'.")

    # select matrix layer
    if layer is None:
        mtx = adata.X
    else:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers.")
        mtx = adata.layers[layer]

    # ensure sparse csr
    if issparse(mtx):
        mtx = mtx.tocsr()
    else:
        mtx = csr_matrix(mtx)

    # slice columns by feature_type
    col_idx = np.where(var_mask.values)[0]
    sub_mtx = mtx[:, col_idx].tocsr()

    # attach feature_names
    if feature_name_col not in adata.var.columns:
        feature_names = adata.var.index.astype(str).tolist()
        print(
            f"Warning: feature_name_col '{feature_name_col}' not found in adata.var. "
            f"Using adata.var.index as feature names."
        )
    else:    
        feature_names = var_subset[feature_name_col].astype(str).tolist()

    if len(feature_names) != len(set(feature_names)):
        dup = pd.Series(feature_names).value_counts()
        dup = dup[dup > 1].index.tolist()
        raise ValueError(
            f"Feature names in column '{feature_name_col}' are not unique. "
            f"Duplicates: {dup}"
        )

    sub_mtx.feature_names = feature_names

    # attach barcode_names from obs.index
    barcode_names = adata.obs.index.astype(str).tolist()
    if len(barcode_names) != len(set(barcode_names)):
        dup = pd.Series(barcode_names).value_counts()
        dup = dup[dup > 1].index.tolist()
        raise ValueError(
            f"Barcode (cell) names in adata.obs.index are not unique. "
            f"Duplicates: {dup}"
        )

    sub_mtx.barcode_names = barcode_names

    return sub_mtx