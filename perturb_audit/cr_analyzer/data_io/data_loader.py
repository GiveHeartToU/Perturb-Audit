import h5py
import pandas as pd
import numpy as np
import json
import os
import sys

# --- Memory Reminder Module ---

def get_current_memory_usage():
    """Get current process memory usage in MB."""
    try:
        import psutil
        HAS_PSUTIL = True
    except ImportError:
        HAS_PSUTIL = False
    
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        mem_usage_mb = process.memory_info().rss / (1024 * 1024)
        mem_usage_gb = mem_usage_mb / 1024
        return f"{mem_usage_mb:.2f} MB = ({mem_usage_gb:.2f} GB)"
    return "N/A (Install psutil)"

# --- Core Loader Function (DataInput Package Component) ---

def check_gem_groups(f):
    """
    Checks the gem_group values in the library_info dataset to confirm 
    if all libraries belong to the same group (gem_group=1).
    Args:
        f (h5py.File): Opened HDF5 file object.
    Returns:
        bool: True if all libraries have gem_group=1, False otherwise.
    """
    lib_info_raw = f['/library_info'][0].decode()
    lib_data = json.loads(lib_info_raw)
    
    gem_groups = [entry.get('gem_group') for entry in lib_data]
    
    # library gem_group should all be 1
    all_is_one = all(g == 1 for g in gem_groups)
    
    if all_is_one:
        print("All libraries have gem_group=1, cellbarcodes are appended with -1.")
    else:
        print(f"Warning: Detected multiple gem_groups in library_info: {gem_groups}")
        
    return all_is_one

def load_molecule_info(h5_path: str, 
                       mode: str = 'pass_filter'):
    """
    Loads the molecule_info.h5 file and returns structured data subsets based on cell filtering status.

    Args:
        h5_path (str): Full path to the molecule_info.h5 file.
        mode (str): Determines which barcode subset to focus on:
                    - 'pass_filter': Returns data corresponding to cells that passed quality filtering.
                    - 'ambient': Returns data corresponding to molecules associated with 
                                 barcodes that did NOT pass quality filtering (ambient/background).

    Returns:
        tuple: (barcode_info_df, feature_info_df, subset_umi_df)
               - barcode_info_df: Contextual info for the selected mode. Sparse for 'ambient'.
               - feature_info_df: Universal feature metadata.
               - subset_umi_df: Raw UMI records ONLY for the selected barcode subset.
    """
    print("-" * 50)
    print(f"Starting Data Loading for: {os.path.basename(h5_path)} | Mode: {mode.upper()}")
    print(f"Memory usage before load: {get_current_memory_usage()}")

    try:
        f = h5py.File(h5_path, 'r')
    except (FileNotFoundError, Exception) as e:
        print(f"Error loading H5 file {h5_path}: {e}")
        return None, None, None

    # --- Pre-load Mappings ---
    library_info = f['/library_info']
    print(f"Library info: {library_info[:]}") 
    all_is_one = check_gem_groups(f) # Check gem_group consistency
    if not all_is_one:
        print("Warning: Detected multiple gem_groups. Ensure it is only 1 batch of 10X reactions")
        return None, None, None
    else:
        all_barcodes_raw = f['/barcodes'][:]
        suffix = "-1"
        all_barcodes = []
        for b in all_barcodes_raw:
            bc_str = b.decode()
            if '-' not in bc_str:
                all_barcodes.append(f"{bc_str}{suffix}")
            else:
                all_barcodes.append(bc_str)
        barcode_map_array = np.array(all_barcodes)
    
    genome_list = [g.decode() for g in f['/barcode_info/genomes'][:]]
    genome_map_array = pd.Series(genome_list).to_numpy()
    
    # --- Raw UMI Records (Always loaded first, used for subsetting) ---
    raw_umi_df = pd.DataFrame({
        'barcode_idx':  f['/barcode_idx'][:],
        'umi':          f['/umi'][:],
        'count':        f['/count'][:],
        'feature_idx':  f['/feature_idx'][:],
        'gem_group':    f['/gem_group'][:],
        'library_idx':  f['/library_idx'][:],
        'umi_type':     f['/umi_type'][:],
    })
    raw_umi_df_libCBCcounts = raw_umi_df.groupby('library_idx')['barcode_idx'].nunique().to_dict()
    print(f"DataInput: Loaded {len(raw_umi_df)} total raw molecule records.")

    # --- Feature Information (Universal) ---
    feat_grp = f['/features']
    feature_info_df = pd.DataFrame({
        'feature_type': [ft.decode() for ft in feat_grp['feature_type'][:]],
        'genome':       [g.decode() for g in feat_grp['genome'][:]],
        'id':           [i.decode() for i in feat_grp['id'][:]],
        'name':         [n.decode() for n in feat_grp['name'][:]],
        # Safe access for potentially missing fields (CRISPR-specific)
        'pattern':      [p.decode() for p in feat_grp.get('pattern', [])[:]] if feat_grp.get('pattern') is not None else [''] * len(feat_grp['id'][:]),
        'read':         [r.decode() for r in feat_grp.get('read', [])[:]] if feat_grp.get('read') is not None else [''] * len(feat_grp['id'][:]),
        'sequence':     [s.decode() for s in feat_grp.get('sequence', [])[:]],
    })
    feature_types = feature_info_df['feature_type'].value_counts().to_dict()
    print(f"DataInput: Feature type distribution: {feature_types}")

    # --- Barcode Subset Logic ---
    if mode == 'pass_filter':
        if '/barcode_info/pass_filter' not in f:
            print("Error: 'pass_filter' dataset not found in H5 file.")
            f.close()
            return None, feature_info_df, pd.DataFrame()
            
        # 1. Get Filtered Indices and Context
        pf = f['/barcode_info/pass_filter'][:]
        pf_df = pd.DataFrame(pf, columns=['barcode_idx', 'library_idx', 'genome_idx'])
        
        pf_df['barcode'] = barcode_map_array[pf_df['barcode_idx'].values]
        pf_df['genome'] = genome_map_array[pf_df['genome_idx'].values]
        barcode_info_df = pf_df.copy()
        
        # 2. Filter raw_umi_df to only include these cell barcodes
        filtered_indices = set(barcode_info_df['barcode_idx'])
        subset_umi_df = raw_umi_df[raw_umi_df['barcode_idx'].isin(filtered_indices)].copy()
        subset_umi_df_libCBCcounts = subset_umi_df.groupby('library_idx')['barcode_idx'].nunique().to_dict()
        
        print(f"DataInput: Subset Pass Filter barcodes (Cells) records count: {len(barcode_info_df)} in barcode_info_df.")
        print(f"DataInput: Subset UMI records count: {len(subset_umi_df)} in subset_umi_df, {len(subset_umi_df)/len(raw_umi_df)*100:.2f}% of raw.")
        print(f"DataInput: Unique cell barcodes per library in raw data: {raw_umi_df_libCBCcounts}")
        print(f"DataInput: Pass Filter barcodes per library: {subset_umi_df_libCBCcounts}")
        
        
    elif mode == 'ambient':
        # 1. Identify all unique barcodes present in the raw UMI data
        all_umi_indices = set(raw_umi_df['barcode_idx'].unique())
        
        # 2. Identify the indices that passed the filter
        if '/barcode_info/pass_filter' in f:
            pass_filter_indices = set(f['/barcode_info/pass_filter'][:, 0])
        else:
             pass_filter_indices = set() # Assume no passing cells if matrix is missing
        
        # 3. Ambient indices are those present in UMI data but NOT in pass_filter
        ambient_indices = list(all_umi_indices - pass_filter_indices)
        
        # 4. Create sparse barcode context DF for ambient
        barcode_info_df = pd.DataFrame({'barcode_idx': ambient_indices})
        barcode_info_df['mode'] = 'ambient' # Contextual column showing its nature
        
        # 5. Filter raw_umi_df to only include these ambient records
        subset_umi_df = raw_umi_df[raw_umi_df['barcode_idx'].isin(ambient_indices)].copy()
        subset_umi_df_libCBCcounts = subset_umi_df.groupby('library_idx')['barcode_idx'].nunique().to_dict()
        
        print(f"DataInput: subset Ambient barcodes records count: {len(ambient_indices)} in barcode_info_df.")
        print(f"DataInput: Subset UMI records count: {len(subset_umi_df)} in subset_umi_df, {len(subset_umi_df)/len(raw_umi_df)*100:.2f}% of raw.")
        print(f"DataInput: Unique cell barcodes per library in raw data: {raw_umi_df_libCBCcounts}")
        print(f"DataInput: Ambient barcodes per library: {subset_umi_df_libCBCcounts}")

    else:
        print(f"Error: Invalid mode '{mode}'. Use 'pass_filter' or 'ambient'.")
        f.close()
        return None, feature_info_df, pd.DataFrame()

    f.close()
    
    print(f"Memory usage after load: {get_current_memory_usage()}")
    print("-" * 50)
    
    return barcode_info_df, feature_info_df, subset_umi_df

# --- Additional Utility Functions for CellBender h5 Loading (if needed) ---
import h5py
import numpy as np
import anndata as ad
from scipy.sparse import csc_matrix
import scanpy as sc

def _decode(arr):
    """Decode byte strings to str if necessary."""
    if isinstance(arr[0], bytes):
        return np.array([x.decode("utf-8") for x in arr])
    return arr

def load_cellbender_minimal(raw_h5: str, cb_h5: str) -> ad.AnnData:
    """
    Minimal CellBender loader without cellbender dependency.
    the full cellbender h5 -> adata could be load through 
    `from cellbender.remove_background.downstream import load_anndata_from_input_and_output`

    Parameters
    ----------
    raw_h5 : str
        Path to 10x raw_feature_bc_matrix.h5
    cb_h5 : str
        Path to cellbender output .h5
    
    Returns
    -------
    AnnData
        X = cellbender cleaned counts
        layers['raw'] = raw counts
        layers['cellbender'] = cleaned counts
        obsm['cellbender_embedding'] = latent embedding
    """

    with h5py.File(cb_h5, "r") as f:
        data = f["matrix/data"][:]
        indices = f["matrix/indices"][:]
        indptr = f["matrix/indptr"][:]
        shape = f["matrix/shape"][:]
        clean_matrix = csc_matrix((data, indices, indptr), shape=shape).T.tocsr()

        barcodes = _decode(f["matrix/barcodes"][:])

        gene_ids = _decode(f["matrix/features/id"][:]) # unique
        gene_names = _decode(f["matrix/features/name"][:]) # may duplicate, not unique
        feature_types = _decode(f["matrix/features/feature_type"][:])
        genomes = _decode(f["matrix/features/genome"][:])

        embedding = f["droplet_latents/gene_expression_encoding"][:]

    # Construct AnnData with clean counts
    adata = ad.AnnData(X=clean_matrix)
    adata.obs_names = barcodes
    adata.var_names = gene_names
    adata.var["gene_id"] = gene_ids
    adata.var["feature_type"] = feature_types
    adata.var["genome"] = genomes
    # Load raw matrix and align
    raw_adata = sc.read_10x_h5(raw_h5, gex_only=False)
    raw_adata.var = raw_adata.var.rename(columns={"gene_ids": "gene_id", "feature_types": "feature_type", "genome": "genome"})

    # bacode alignment
    raw_adata = raw_adata[adata.obs_names].copy()
    # feature alignment by gene_id (not by gene_name, since gene_name may not be unique)
    geneid2idx = {gid: i for i, gid in enumerate(raw_adata.var['gene_id'])}
    col_indices = [geneid2idx[gid] for gid in adata.var['gene_id']]
    raw_adata = raw_adata[:, col_indices].copy()
    
    adata.layers["raw"] = raw_adata.X.astype(np.int64).copy()
    adata.layers["cellbender"] = clean_matrix.copy()

    adata.X = adata.layers["cellbender"]
    adata.obsm["cellbender_embedding"] = embedding

    adata.var.index.name = "gene_name" # set var index name to gene_name for consistency with cellbender var_names
    adata.obs.index.name = "barcode" # set obs index name to barcode for consistency with cellbender obs_names

    return adata

