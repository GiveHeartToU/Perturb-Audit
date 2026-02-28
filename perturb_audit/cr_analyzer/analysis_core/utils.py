# analysis_core/utils.py (Tooling for AnalysisCore)
import inspect
import pandas as pd

# Nucleotide to 2-bit mapping: A=0, C=1, G=2, T=3
NUC_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3} # 2-bit encoding
INT_TO_NUC = {v: k for k, v in NUC_TO_INT.items()} # Reverse mapping

def decode_umi(encoded_umi: int, umi_length: int = 10) -> str:
    """
    Decodes a 2-bit encoded UMI integer back into its DNA sequence string.

    The UMI sequences are 2-bit encoded: 
        0="A", 1="C", 2="G", 3="T"
    The least significant bits (LSB) represent the 3'-most nucleotide.

    Args:
        encoded_umi (int): The integer representation of the UMI.
        umi_length (int): Length of the UMI sequence (default=10 bases).

    Returns:
        str: The decoded DNA sequence string (5' -> 3').
    """
    if not isinstance(encoded_umi, int) or encoded_umi < 0:
        return ""

    decoded_sequence = []
    temp_umi = encoded_umi
    
    # Decode exactly umi_length bases
    for _ in range(umi_length):
        nuc_code = temp_umi & 0b11  # Get last 2 bits
        decoded_sequence.append(INT_TO_NUC.get(nuc_code, 'N'))
        temp_umi >>= 2  # Shift right by 2 bits

    # Reverse to get standard 5'->3' order
    return "".join(reversed(decoded_sequence))


def show_dataframes_columns(env=None):
    """
    Displays the names and columns of all pandas DataFrames in the given environment.
    Args:
        env (dict, optional): The environment (namespace) to inspect. 
                              Defaults to caller's globals() if None.
    """
    if env is None:
        # View the caller's global variable space
        frame = inspect.currentframe().f_back
        env = frame.f_globals

    dfs = {
        name: var
        for name, var in env.items()
        if isinstance(var, pd.DataFrame) and not name.startswith('_')
    }

    if not dfs:
        print("There are no pandas DataFrames in the provided environment.")
        return

    for name, df in dfs.items():
        print(f"DataFrame name: {name}")
        print(f"Columns: {list(df.columns)}\n")

from scipy.sparse import csr_matrix
from typing import List, Optional
def generate_sgRNA_list(
    sgRNA_matrix: csr_matrix,
    sort_by: str = 'total_umi',
    ascending: bool = False,
    top_n: Optional[int] = None,
    min_cells: int = 10,
    min_total_umi: int = 50
) -> List[str]:
    
    # 1. Data Extraction and Metric Calculation
    # Ensure the matrix has feature names
    if not hasattr(sgRNA_matrix, 'feature_names'):
        raise AttributeError("sgRNA_matrix must have a 'feature_names' attribute.")
    
    sgRNA_names = list(sgRNA_matrix.feature_names)
    
    # Calculate Total UMI (sum across rows, output is a dense array)
    total_umi = sgRNA_matrix.sum(axis=0).A.flatten()
    
    # Calculate Cell Count (number of non-zero entries in each column)
    # Using matrix greater than 0 creates a boolean matrix, sum gives cell counts
    cell_count = (sgRNA_matrix > 0).sum(axis=0).A.flatten()
    
    # 2. Create Metrics DataFrame
    metrics_df = pd.DataFrame({
        'sgRNA_name': sgRNA_names,
        'total_umi': total_umi,
        'cell_count': cell_count
    })
    
    # 3. Filtering
    # Filter based on minimum cell count and minimum total UMI
    metrics_df = metrics_df[metrics_df['cell_count'] >= min_cells]
    metrics_df = metrics_df[metrics_df['total_umi'] >= min_total_umi]
    
    if metrics_df.empty:
        print("Warning: No sgRNAs meet the specified filtering criteria.")
        return []
        
    # 4. Sorting
    sort_key = sort_by.lower()
    
    if sort_key == 'alphabetical':
        # Sort by the name itself
        sort_col = 'sgRNA_name'
        # Alphabetical sorting usually defaults to True for ascending
        current_ascending = ascending
    elif sort_key == 'total_umi':
        sort_col = 'total_umi'
        current_ascending = ascending
    elif sort_key == 'cell_count':
        sort_col = 'cell_count'
        current_ascending = ascending
    else:
        raise ValueError(f"Invalid sort_by option: '{sort_by}'. Must be 'total_umi', 'cell_count', or 'alphabetical'.")

    # Perform the sorting
    metrics_df = metrics_df.sort_values(by=sort_col, ascending=current_ascending)
    
    # 5. Top N Selection
    if top_n is not None and top_n > 0:
        metrics_df = metrics_df.head(top_n)
        
    # 6. Return 
    return metrics_df['sgRNA_name'].tolist()

import numpy as np
def extract_dominance_data(sgRNA_matrix: csr_matrix) -> pd.DataFrame:
    """
    Extract Top 3 sgRNA statistics and ratios for each CBC from sgRNA UMI sparse matrix.
    Used for Explortory Data Analysis (EDA) before dominance assignment.

    Args:
        sgRNA_matrix (csr_matrix): sgRNA UMI counts sparse matrix (from func "generate_count_matrix").

    Returns:
        pd.DataFrame: A DataFrame that contains all ratios and statistics for each CBC.
    """
    
    if not isinstance(sgRNA_matrix, csr_matrix) or not hasattr(sgRNA_matrix, 'barcode_names'):
        raise TypeError("sgRNA_matrix must be a scipy.sparse.csr_matrix constructed from generate_count_matrix().")

    n_cells = sgRNA_matrix.shape[0]
    barcode_names = list(sgRNA_matrix.barcode_names)
    
    # initialize arrays to hold Top 3 UMI counts and non-zero sgRNA counts
    top_umis = np.zeros((n_cells, 3), dtype=np.int32)
    n_nonzero_array = np.zeros(n_cells, dtype=np.int16)
    
    # calculate total sgRNA UMI counts for each CBC
    # Flatten (sum(axis=1)) (N, 1) matrix to (N,) array
    total_umi_array = sgRNA_matrix.sum(axis=1).A.flatten() # sgRNA_matrix.sum(axis=1).getA1() also works

    for i in range(n_cells):
        # Extract sgRNA UMI count for current CBC using row index
        row_data = sgRNA_matrix.data[sgRNA_matrix.indptr[i]:sgRNA_matrix.indptr[i+1]]
        
        # Count non-zero sgRNA entries
        n_nonzero = len(row_data)
        n_nonzero_array[i] = n_nonzero
        
        if n_nonzero > 0:
            # np.partition is more efficient than full sort for top-k selection
            if n_nonzero >= 3:
                partitioned = np.partition(row_data, -3)  # put the top 3 values at the right positions
                top_three = partitioned[-3:]  # extract the top 3 values
                sorted_top_three = np.sort(top_three)[::-1]  # descending sort
            else:
                sorted_top_three = np.sort(row_data)[::-1]

            top_umis[i, :len(sorted_top_three)] = sorted_top_three

    df = pd.DataFrame({
        'CBC_Name': barcode_names,
        'Total_sgRNA_UMI': total_umi_array,
        'n_nonzero': n_nonzero_array,
        'Top1_UMI': top_umis[:, 0],
        'Top2_UMI': top_umis[:, 1],
        'Top3_UMI': top_umis[:, 2]
    })

    # --- Calculate Ratios --- Pay attention to division by zero cases.
    # 1. Top N / Total
    # avoid dividing by zero Total_sgRNA_UMI
    safe_total_umi = np.where(df['Total_sgRNA_UMI'] > 0, df['Total_sgRNA_UMI'], 1)
    df['Ratio_1'] = df['Top1_UMI'] / safe_total_umi
    df['Ratio_2'] = df['Top2_UMI'] / safe_total_umi
    df['Ratio_3'] = df['Top3_UMI'] / safe_total_umi
    
    # 2. Log2(Top N / Top N+1)
    # R1/R2: filtering out Top2_UMI=0 cells (i.e., n_nonzero <= 1 cells)
    top2_valid = df['Top2_UMI'] > 0
    df['LogRatio_12'] = np.nan
    df.loc[top2_valid, 'LogRatio_12'] = np.log2(df.loc[top2_valid, 'Top1_UMI'] / df.loc[top2_valid, 'Top2_UMI'])
    
    # R2/R3: filtering out Top3_UMI=0 cells (i.e., n_nonzero <= 2 cells)
    top3_valid = df['Top3_UMI'] > 0
    df['LogRatio_23'] = np.nan
    df.loc[top3_valid, 'LogRatio_23'] = np.log2(df.loc[top3_valid, 'Top2_UMI'] / df.loc[top3_valid, 'Top3_UMI'])

    # R1/R3: filtering out Top3_UMI=0 cells (i.e., n_nonzero <= 2 cells)
    df['LogRatio_13'] = np.nan
    df.loc[top3_valid, 'LogRatio_13'] = np.log2(df.loc[top3_valid, 'Top1_UMI'] / df.loc[top3_valid, 'Top3_UMI'])
    
    # Set ratios to 0.0 where Total_sgRNA_UMI is zero
    df.loc[df['Total_sgRNA_UMI'] == 0, ['Ratio_1', 'Ratio_2', 'Ratio_3']] = 0.0
    
    return df