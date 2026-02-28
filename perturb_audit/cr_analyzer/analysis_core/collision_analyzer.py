# analysis_core/collision_analyzer.py
import pandas as pd
from typing import Tuple, Set

def identify_cross_library_collisions(raw_umi_df: pd.DataFrame) -> Tuple[Set[int], pd.DataFrame]:
    """
    Identifies molecular records that represent cross-library PCR chimeras 
    based purely on the co-occurrence of (barcode_idx, umi) in both library_idx 0 and 1.
    """
    print("-"*50)
    print("AnalysisCore: Identifying cross-library collisions using vectorized operations...")
    
    # 1. Determine all unique (barcode_idx, umi) pairs present in each library
    # Library 0 (GEX) UMI Pairs
    gex_umi_pairs = raw_umi_df[raw_umi_df['library_idx'] == 0][['barcode_idx', 'umi']]
    if gex_umi_pairs.duplicated().any():
        print("Warning: Duplicated CBC-UMI pairs found in GEX library data; This is unexpected.")
    else:
        print("AnalysisCore: No CBC-UMI duplicates found in GEX library data. This is 10X expected.")

    # Library 1 (sgRNA) UMI Pairs
    sg_umi_pairs = raw_umi_df[raw_umi_df['library_idx'] == 1][['barcode_idx', 'umi']]
    if sg_umi_pairs.duplicated().any():
        sg_umi_pairs = sg_umi_pairs.drop_duplicates()
        print("Warning: Duplicated CBC-UMI pairs found in sgRNA library data; This is unexpected.")
    else:
        print("AnalysisCore: No CBC-UMI duplicates found in sgRNA library data. This is 10X expected.")

    # 2. Identify Colliding Keys by Merging
    # Merge on (barcode_idx, umi). An inner merge only keeps records present in BOTH.
    colliding_df = pd.merge(gex_umi_pairs, sg_umi_pairs, on=['barcode_idx', 'umi'], how='inner')
    print(f"AnalysisCore: Found {len(colliding_df)} unique (Cell, UMI) pairs involved in cross-library collision.")

    # 3. Determine the set of affected barcode_idx
    collision_barcodes = set(colliding_df['barcode_idx'].unique())
    print(f"AnalysisCore: {len(collision_barcodes)} unique cell barcodes are affected by collisions.")
    print(f"AnalysisCore: {len(collision_barcodes) / raw_umi_df['barcode_idx'].nunique() * 100:.3f}% of total cell barcodes affected.")
    print(f"AnalysisCore: {len(collision_barcodes) / sg_umi_pairs['barcode_idx'].nunique() * 100:.3f}% of sgRNA library cell barcodes affected.")

    # 4. Gather necessary read counts for reporting (Vectorized approach)
    # Filter raw_umi_df to only include records involved in a cross-library collision
    collision_mask = raw_umi_df.set_index(['barcode_idx', 'umi']).index.isin(
        colliding_df.set_index(['barcode_idx', 'umi']).index
    )
    collision_records = raw_umi_df[collision_mask].copy()
    print(f"AnalysisCore: Extracted {len(collision_records)} UMI records involved in cross-library collisions for detailed analysis.")
    print(f"AnalysisCore: {len(collision_records) / len(raw_umi_df) * 100:.4f}% of total UMI records involved in collisions.")
    
    # 5. Define aggregation function for features: collect all unique feature_idx into a list
    def aggregate_unique_features(series):
        return sorted(series.unique().tolist())
        
    # The pivot_table will use library_idx as columns, and aggregate 'count' (sum) and 'feature_idx' (list of unique features)
    collision_summary_df = collision_records.pivot_table(
        index=['barcode_idx', 'umi'], 
        columns='library_idx', 
        values=['count', 'feature_idx'],
        aggfunc={'count': 'sum', 'feature_idx': aggregate_unique_features}, 
        fill_value=0 # Fills missing counts (though shouldn't happen here due to the inner merge key)
    )

    # Library Index 0 = GEX; Library Index 1 = sgRNA
    collision_summary_df.columns = [
        f"{col}_{'GEX' if lib == 0 else 'sgRNA'}" if col != 'count' else f"{'GEX' if lib == 0 else 'sgRNA'}_Collision_Reads"
        for col, lib in collision_summary_df.columns
    ]
    collision_summary_df.reset_index(inplace=True)

    print("AnalysisCore: Cross-library collision identification complete.")
    print("-"*50)

    return collision_barcodes, collision_summary_df

