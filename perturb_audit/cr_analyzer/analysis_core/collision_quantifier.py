# analysis_core/collision_quantifier.py

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from ..data_io.data_loader import get_current_memory_usage

def generate_collision_summary_stats(
    umi_df: pd.DataFrame, 
    barcode_df: pd.DataFrame, 
    feature_df: pd.DataFrame, 
    collisions_df: pd.DataFrame, 
    collision_barcodes: set,
    LIB_GEX: int = 0,
    LIB_SGRNA: int = 1,
    TYPE_GEX: str = 'Gene Expression',
    TYPE_SGRNA: str = 'CRISPR Guide Capture',
    FEATURE_NAME_COL: str = 'name'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generates detailed cell-level and feature-level summary statistics to quantify 
    cross-library collisions stastistics.

    Args:
        umi_df (pd.DataFrame): The full raw molecule records for CBCs.
        barcode_df (pd.DataFrame): Contextual information for CBCs.
        feature_df (pd.DataFrame): Universal feature metadata (for mapping names).
        collisions_df (pd.DataFrame): Summary of colliding (CBC, UMI) pairs.
        collision_barcodes (set): Set of barcode_idx involved in any collision.
        LIB_GEX (int): Library ID for GEX library (default: 0) in umi_df['library_idx'].
        LIB_SGRNA (int): Library ID for sgRNA library (default: 1) in umi_df['library_idx'].
        TYPE_GEX (str): Feature type string for GEX features in feature_df['feature_type'].
        TYPE_SGRNA (str): Feature type string for sgRNA features in feature_df['feature_type'].
        FEATURE_NAME_COL (str): Column name in feature_df for feature names. Readable "name" by default. "id" can be used for Unique IDs.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
            1. cell_summary_df: CBC-level summary (18 column metrics).
            2. gex_feature_impact_df: Feature-level impact for GEX features.
            3. sgrna_feature_impact_df: Feature-level impact for sgRNA features.
    """
    print("\n" + "-"*50)
    print("AnalysisCore: Generating collision summary statistics (Finalized)...")
    print(f"Memory usage before analysis: {get_current_memory_usage()}")
    
    # Initialize the final cell summary table with all pass_filter barcodes
    all_barcodes = barcode_df['barcode_idx'].unique()
    cell_summary_df = pd.DataFrame({'barcode_idx': all_barcodes})
    
    # --- Helper to map Library ID to Suffix ---
    def get_lib_suffix(lib_id):
        return 'GEX' if lib_id == LIB_GEX else 'sgRNA'

    # --------------------------------------------------------------------------
    # 1.1 Background Stats (A) - Total UMI, Reads, Features per CBC per Library
    # --------------------------------------------------------------------------
    print("  -> Calculating background (Total) stats...")
    
    def aggregate_metrics(df, prefix):
        # Use generic names UMI, Reads, Features for easy mapping later
        agg_data = df.groupby(['barcode_idx', 'library_idx']).agg(
            UMIs=('umi', 'nunique'),
            Reads=('count', 'sum'),
            Features=('feature_idx', 'nunique')
        ).reset_index()
        
        pivot_metrics = agg_data.pivot(
            index='barcode_idx', 
            columns='library_idx', 
            values=['UMIs', 'Reads', 'Features']
        ).fillna(0)
        
        # Flatten and name MultiIndex columns: Total_UMI_GEX, Total_Reads_sgRNA, etc.
        pivot_metrics.columns = [
            f"{prefix}_{col}_{get_lib_suffix(lib)}"
            for col, lib in pivot_metrics.columns
        ]
        return pivot_metrics.reset_index()

    total_stats = aggregate_metrics(umi_df, prefix='Total')
    cell_summary_df = pd.merge(cell_summary_df, total_stats, on='barcode_idx', how='left').fillna(0)
    cell_summary_df['Is_Colliding_Cell'] = cell_summary_df['barcode_idx'].isin(collision_barcodes)

    # --------------------------------------------------------------------------
    # 1.2 Collision Stats (B) - Colliding UMI, Reads, Features per CBC per Library
    # --------------------------------------------------------------------------
    print("  -> Calculating collision stats...")
    
    collision_metrics = collisions_df.copy()
    
    # 1. Create GEX and sgRNA component rows for aggregation
    # GEX Component
    gex_collision_rows = collision_metrics[['barcode_idx', 'umi', 'GEX_Collision_Reads', 'feature_idx_GEX']].copy()
    gex_collision_rows.rename(columns={'GEX_Collision_Reads': 'Reads', 'feature_idx_GEX': 'Feature_Indices'}, inplace=True)
    gex_collision_rows['Library'] = LIB_GEX
    # sgRNA Component
    sg_collision_rows = collision_metrics[['barcode_idx', 'umi', 'sgRNA_Collision_Reads', 'feature_idx_sgRNA']].copy()
    sg_collision_rows.rename(columns={'sgRNA_Collision_Reads': 'Reads', 'feature_idx_sgRNA': 'Feature_Indices'}, inplace=True)
    sg_collision_rows['Library'] = LIB_SGRNA
    # collision_agg_df now has the Feature_Indices list for each collision event
    collision_agg_df = pd.concat([gex_collision_rows, sg_collision_rows])
    
    # Define custom aggregation function for Features: Flatten lists and count unique elements
    def count_unique_features_from_list(list_of_lists):
        # Flattens all lists into a single set to count unique elements
        all_features = set()
        for feature_list in list_of_lists:
            all_features.update(feature_list)
        return len(all_features)

    # 2. Group by CBC and Library to get collision totals (B)
    collision_totals = collision_agg_df.groupby(['barcode_idx', 'Library']).agg(
        UMIs=('umi', 'nunique'),
        Reads=('Reads', 'sum'),
        Features=('Feature_Indices', count_unique_features_from_list) 
    ).reset_index()

    # 3. Pivot and merge
    collision_totals_pivot = collision_totals.pivot(
        index='barcode_idx', 
        columns='Library', 
        values=['UMIs', 'Reads', 'Features']
    ).fillna(0)

    # Flatten and name MultiIndex columns: Colliding_UMI_GEX, Colliding_Reads_sgRNA, etc.
    final_cols = []
    for col_name, lib in collision_totals_pivot.columns:
        final_cols.append(f"Colliding_{col_name}_{get_lib_suffix(lib)}")

    collision_totals_pivot.columns = final_cols 
    collision_totals_pivot.reset_index(inplace=True)
    
    cell_summary_df = pd.merge(cell_summary_df, collision_totals_pivot, on='barcode_idx', how='left').fillna(0)
    
    # --------------------------------------------------------------------------
    # 1.3 Ratio Calculation (C) - Collision Impact Ratios
    # --------------------------------------------------------------------------
    print("  -> Calculating collision_stats ratios...")
    metrics = ['UMIs', 'Reads', 'Features']
    libraries = ['GEX', 'sgRNA']
    
    for metric in metrics:
        for lib in libraries:
            col_ratio = f"Collision_{metric}_Ratio_{lib}"          # Collision_UMIs_Ratio_GEX
            col_colliding = f"Colliding_{metric}_{lib}"            # Colliding_UMIs_GEX
            col_total = f"Total_{metric}_{lib}"                    # Total_UMI_GEX
            
            # Ratio = Colliding / Total. Handle division by zero.
            cell_summary_df[col_ratio] = np.where(
                cell_summary_df[col_total] > 0,
                cell_summary_df[col_colliding] / cell_summary_df[col_total],
                0
            )

    # --------------------------------------------------------------------------
    # Part 2: Feature-Level Impact (D & E) - Feature Purity Analysis
    # --------------------------------------------------------------------------
    print("  -> Performing feature-level summary statistics analysis...")
    
    # 2.1 Total UMI Count for each (CBC, Feature)
    total_umi_per_feature = umi_df.groupby(['barcode_idx', 'feature_idx']).agg(
        Total_Feature_UMIs=('umi', 'nunique')
    ).reset_index()
    
    # 2.2 Colliding UMI Count for each (CBC, Feature)
    if collisions_df[['barcode_idx', 'umi']].duplicated().any() :
        print("Warning: Duplicated (barcode_idx, umi) pairs should not be found in collisions_df; Dropping duplicates for feature impact analysis.")
        collisions_df = collisions_df.drop_duplicates(subset=['barcode_idx', 'umi'])
    
    colliding_umi_keys = collisions_df[['barcode_idx', 'umi']].set_index(['barcode_idx', 'umi'])
    collision_mask = umi_df.set_index(['barcode_idx', 'umi']).index.isin(colliding_umi_keys.index)
    colliding_umi_records = umi_df[collision_mask].copy()

    colliding_umi_per_feature = colliding_umi_records.groupby(['barcode_idx', 'feature_idx']).agg(
        Colliding_Feature_UMIs=('umi', 'nunique')
    ).reset_index()

    # 2.3 Merge and Calculate Ratio
    feature_impact = pd.merge(
        total_umi_per_feature, 
        colliding_umi_per_feature, 
        on=['barcode_idx', 'feature_idx'], 
        how='left'
    ).fillna(0)
    
    print("  -> Calculating feature-level collision UMI ratios...(Where zero total UMIs, ratio set to 0)")
    feature_impact['Collision_UMI_Ratio_In_Feature'] = np.where(
        feature_impact['Total_Feature_UMIs'] > 0,
        feature_impact['Colliding_Feature_UMIs'] / feature_impact['Total_Feature_UMIs'],
        0
    )

    # 2.4 Feature Name Mapping and Splitting (E)
    # Create the mapping DataFrame: Index is feature_idx
    # Use 'feature_type' for library distinction and default readable 'name' for feature name
    feature_map_df = feature_df[[FEATURE_NAME_COL, 'feature_type']].reset_index().rename(columns={'index': 'feature_idx'})
    feature_impact = pd.merge(feature_impact, feature_map_df, on='feature_idx', how='left')
    feature_impact.rename(columns={FEATURE_NAME_COL: 'Feature_Name', 'feature_type': 'Library_Type'}, inplace=True)

    # Final split based on Library Type (feature_type column)
    gex_feature_impact_df = feature_impact[feature_impact['Library_Type'] == TYPE_GEX].drop(columns=['Library_Type']).copy()
    sgrna_feature_impact_df = feature_impact[feature_impact['Library_Type'] == TYPE_SGRNA].drop(columns=['Library_Type']).copy()

    print("AnalysisCore: Summary statistics generation complete.")
    print(f"Memory usage after analysis: {get_current_memory_usage()}")
    print("-" * 50)
    return cell_summary_df, gex_feature_impact_df, sgrna_feature_impact_df

