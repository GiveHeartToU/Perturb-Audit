# visualization/plotter.py
import os
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from pathlib import Path

try:
    # Python 3.9+
    from importlib.resources import files
except ImportError:
    # Python 3.7 / 3.8
    from importlib_resources import files  # backport

def get_font_dir(family="Arial") -> Path:
    """
    Returns the absolute path to the 'arial' fonts folder
    inside the visualization package.

    Further font families can be added in the future, but currently only 'Arial' is supported.
    """
    if family.lower() != "arial":
        raise ValueError(f"Unsupported font family: {family}. Only 'Arial' is supported in this function.")
    else:
        return files("perturb_audit.cr_analyzer.visualization").joinpath("arial")

def register_fonts_clean(font_dir, family="Arial", verbose=True):
    """
    Register fonts from a custom directory into Matplotlib without duplicates.
    Download .ttf font files into `font_dir`(and `fc-cache` the dir if on linux) before calling this function.
    'https://font.download/' is a good resource for free fonts.

    Parameters
    ----------
    font_dir : str
        Path to the directory containing .ttf font files.
    family : str, default "Arial"
        Font family name to set as default (if found).
    verbose : bool, default True
        Whether to print detailed registration information.

    Returns
    -------
    list
        List of unique font family names detected containing the `family` keyword.
    """
    import matplotlib
    import seaborn as sns
    print(f"Matplotlib version: {matplotlib.__version__}")
    print(f"Seaborn version: {sns.__version__}")

    if not os.path.isdir(font_dir):
        raise ValueError(f"Font directory not found: {font_dir}")

    # Find available fonts in the directory
    font_paths = fm.findSystemFonts(fontpaths=[font_dir])
    if not font_paths:
        raise ValueError(f"No fonts found in directory: {font_dir}")

    # Collect existing font paths to avoid duplicates
    existing_paths = {f.fname for f in fm.fontManager.ttflist}
    new_fonts = [p for p in font_paths if p not in existing_paths]

    # Register new fonts
    for fpath in new_fonts:
        fm.fontManager.addfont(fpath)

    # Rebuild or reload font manager()
    if hasattr(fm, "_load_fontmanager"):
        fm._load_fontmanager(try_read_cache=False)
    elif hasattr(fm, "_rebuild"):
        fm._rebuild()

    # List all available fonts containing the target family name
    available_fonts = sorted({f.name for f in fm.fontManager.ttflist if family.lower() in f.name.lower()})

    # Set matplotlib defaults if found
    if available_fonts:
        plt.rcParams["font.family"] = family
        plt.rcParams["svg.fonttype"] = "none"
        plt.rcParams["pdf.fonttype"] = 42
        if verbose:
            print(f"✅ Registered {len(new_fonts)} new fonts from: {font_dir}")
            print(f"✅ Found {len(available_fonts)} matching fonts for family '{family}':")
            for f in available_fonts:
                print("   •", f)
            print(f"✅ Matplotlib default font set to: {family}")
    else:
        if verbose:
            print(f"⚠️ No fonts matching '{family}' were found in {font_dir}.")

    return available_fonts

# Note: The actual plotting function should not be called here, only returned.
# The user will integrate this into the main execution flow.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List

# Define colors for consistency
COLOR_COLLIDING = '#f3793b'  # High saturation warm color (Orange/Red) for collision
COLOR_BACKGROUND = '#716db2' # Low saturation cool color (Blue/Purple) for background
COLOR_GEX = '#1d52a1'        # Deep Blue for GEX (if needed for axes/labels)
COLOR_SGRNA = '#E7872B'      # Green #72c15a for sgRNA (if needed for axes/labels)

def set_publication_style():
    """Sets Matplotlib/Seaborn style for high-quality publication output."""
    sns.set_theme(context='paper', style="ticks") 
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 5,
        'legend.fontsize': 5,
        'legend.title_fontsize': 6,
        'xtick.labelsize': 5,
        'ytick.labelsize': 5,
        'axes.labelsize': 6, 
        'axes.titlesize': 6,
        'lines.linewidth': 0.75,
        'lines.markersize': 3,
        'svg.fonttype': 'none' ,
        'pdf.fonttype': 42,
        'text.usetex': False,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })
    # plt.ion()

# --- Part 1: General Descriptive Plot ---

def _auto_label(col_name: str, log_scale: bool) -> str:
    """Helper to generate human-readable label from column name."""
    # Example: 'Total_Features_sgRNA' -> 'Total sgRNA Features Detected'
    parts = col_name.split('_')
    
    metric = parts[1] # UMI, Reads, or Features
    library = parts[2] # GEX or sgRNA
    
    label = f"Total {library} {metric} Detected"
    if log_scale:
        label += " (Log Scale)"
    return label

def plot_dual_metrics_scatter(
    cell_summary_df: pd.DataFrame,
    X_COL: str,
    Y_COL: str,
    figsize: Tuple[float, float] = (3.5, 3.5), # Increased size to accommodate margins
    log_scale: bool = True,
    # New parameters for marginal plots
    marginal_plot_kind='hist', 
    marginal_bins=50,
    marginal_color="#03051A"
) -> plt.Figure:
    """
    Generates a generic scatter plot comparing two total metrics with marginal plots, 
    highlighting colliding cells by top-zordering but not using hue.
    
    Args:
        cell_summary_df (pd.DataFrame): DataFrame containing cell-level summary statistics.
        X_COL (str): Column name for the X-axis metric.
        Y_COL (str): Column name for the Y-axis metric.
        figsize (Tuple[float, float]): Increased size recommended for JointPlot.
        log_scale (bool): Whether to apply log scale to both axes.
        marginal_plot_kind (str): Type of marginal plot (only 'hist' implemented at the moment).
        marginal_bins (int): Number of bins for the histogram.
        marginal_color (str): Color for the marginal plots.
    
    Returns:
        plt.Figure: The generated matplotlib Figure object.
    """
    
    # --- 1. Data Preparation ---
    plot_data = cell_summary_df.copy()
    plot_data = plot_data.sort_values(by='Is_Colliding_Cell', ascending=True)

    # --- 2. Plotting Setup and Styling (Initialize JointGrid) ---
    set_publication_style()
    
    # Use JointGrid for combined scatter and marginal plots
    g = sns.JointGrid(
        data=plot_data, 
        x=X_COL, 
        y=Y_COL, 
        height=figsize[0], # JointGrid uses 'height' instead of figsize
        ratio=5, # Ratio of joint plot to marginal plot size
    )

    # --- 3. Marginal Plots (Histograms) ---
    if marginal_plot_kind == 'hist':
        # prepare data for marginal plots
        temp_data = plot_data.copy()
        temp_data[X_COL] = temp_data[X_COL].clip(lower=1)
        temp_data[Y_COL] = temp_data[Y_COL].clip(lower=1)

        # Plot X-marginal (Top)
        sns.histplot(
            data=temp_data,
            x=X_COL,
            hue='Is_Colliding_Cell',
            palette={True: COLOR_COLLIDING, False: COLOR_BACKGROUND},
            ax=g.ax_marg_x,
            bins=marginal_bins,
            multiple='stack', # or 'layer' for overlapping
            edgecolor='black',
            legend=False,
            log_scale=log_scale # Apply log scale to histogram if set
        )
        # Plot Y-marginal (Right)
        sns.histplot(
            data=temp_data,
            y=Y_COL,
            hue='Is_Colliding_Cell',
            palette={True: COLOR_COLLIDING, False: COLOR_BACKGROUND},
            ax=g.ax_marg_y,
            bins=marginal_bins,
            multiple='stack', # or 'layer' for overlapping
            edgecolor='black',
            legend=False,
            log_scale=log_scale # Apply log scale to histogram if set
        )
        # sns.histplot(
        #     x=plot_data[X_COL], 
        #     ax=g.ax_marg_x, 
        #     bins=marginal_bins, 
        #     color=marginal_color,
        #     log_scale=log_scale # Apply log scale to histogram if set
        # )
        # sns.histplot(
        #     y=plot_data[Y_COL], 
        #     ax=g.ax_marg_y, 
        #     bins=marginal_bins, 
        #     color=marginal_color,
        #     log_scale=log_scale # Apply log scale to histogram if set
        # )
    # Add other marginal types (e.g., kde) if needed

    # --- 4. Scatter Plot Generation (Center) ---
    # A. Plot Background/Non-Colliding Cells (zorder=1)
    g.ax_joint.scatter(
        plot_data[~plot_data['Is_Colliding_Cell']][X_COL],
        plot_data[~plot_data['Is_Colliding_Cell']][Y_COL],
        color=COLOR_BACKGROUND,
        s=plt.rcParams['lines.markersize']**2,
        alpha=0.6,
        label=f'Non-Colliding CBCs (N={len(plot_data) - plot_data["Is_Colliding_Cell"].sum()})',
        edgecolors=None,
        zorder=1
    )

    # B. Plot Colliding Cells (zorder=2, on top)
    g.ax_joint.scatter(
        plot_data[plot_data['Is_Colliding_Cell']][X_COL],
        plot_data[plot_data['Is_Colliding_Cell']][Y_COL],
        color=COLOR_COLLIDING,
        s=plt.rcParams['lines.markersize']**2,
        alpha=0.8,
        label=f"Colliding CBCs (N={plot_data['Is_Colliding_Cell'].sum()})",
        edgecolors='black',
        linewidth=0.5,
        zorder=2
    )

    # --- 5. Customization and Refinement ---
    # Apply Log Scale to Joint Axes
    if log_scale:
        g.ax_joint.set_xscale('log')
        g.ax_joint.set_yscale('log')
        # Re-apply log scaling to marginal axes scales (JointGrid sometimes overrides this)
        g.ax_marg_x.set_xscale('log')
        g.ax_marg_y.set_yscale('log')
        # Adjust limits to avoid log(0)
        x_min = plot_data[X_COL].replace(0, np.nan).min() / 1.2
        y_min = plot_data[Y_COL].replace(0, np.nan).min() / 1.2
    else:
        x_min, y_min = -0.1, -0.1

    # Set Axes Labels using helper function
    g.ax_joint.set_xlabel(_auto_label(X_COL, log_scale))
    g.ax_joint.set_ylabel(_auto_label(Y_COL, log_scale))

    # Set dynamic limits
    g.ax_joint.set_xlim(x_min, plot_data[X_COL].max() * 1.2)
    g.ax_joint.set_ylim(y_min, plot_data[Y_COL].max() * 1.2)
    
    # Adjust Legend Location: Place below the plot area
    g.ax_joint.legend(
        frameon=False, 
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.15), 
        ncol=2 
    )

    # Clean up the plot boundaries
    sns.despine(ax=g.ax_joint, top=False, right=False) # Keep the boundary box for Joint
    sns.despine(ax=g.ax_marg_x, left=True, bottom=True) # Clean up marginal axes
    sns.despine(ax=g.ax_marg_y, left=True, bottom=True)

    # Remove axis labels and ticks from marginal plots to keep it clean
    g.ax_marg_x.set_ylabel('')
    g.ax_marg_x.set_xlabel('')
    g.ax_marg_x.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    g.ax_marg_x.tick_params(axis='y', which='both', left=True, labelleft=True)

    g.ax_marg_y.set_ylabel('')
    g.ax_marg_y.set_xlabel('')
    g.ax_marg_y.tick_params(axis='y', which='both', left=False, labelleft=False)
    g.ax_marg_y.tick_params(axis='x', which='both', bottom=True, labelbottom=True)

    # Manually adjust bottom margin to ensure the legend is not clipped
    g.figure.subplots_adjust(bottom=0.15, right=0.95, hspace=0.1, wspace=0.1)
    # g.figure.tight_layout()
    
    # --- Final Cleanup ---
    plt.close(g.figure) # Close the figure reference to prevent Jupyter double-output
    plt.rcParams.update(plt.rcParamsDefault)
    
    return g.figure # Return the Figure object of the JointGrid

def plot_collision_ratios(
    cell_summary_df: pd.DataFrame,
    min_features_GEX: Optional[int] = None,
    min_UMIs_GEX: Optional[int] = None,
    min_features_sgRNA: Optional[int] = None,
    min_UMIs_sgRNA: Optional[int] = None,
    figsize: Tuple[float, float] = (7.5, 4.5), # Wider figure for 6 subplots
) -> plt.Figure:
    """
    Generates a single figure containing six boxplots showing the distribution of 
    collision ratios (UMI, Reads, Features) across GEX and sgRNA libraries, 
    only for colliding cells.

    Args:
        cell_summary_df (pd.DataFrame): DataFrame containing cell-level summary statistics.
        min_features_GEX (Optional[int]): Minimum GEX features filter for colliding cells.
        min_UMIs_GEX (Optional[int]): Minimum GEX UMIs filter for colliding cells.
        min_features_sgRNA (Optional[int]): Minimum sgRNA features filter for colliding cells.
        min_UMIs_sgRNA (Optional[int]): Minimum sgRNA UMIs filter for colliding cells.
        figsize (Tuple[float, float]): Figure size in inches (7x5 recommended for 2x3 layout).

    
    Returns:
        plt.Figure: The generated matplotlib Figure object.
    """
    
    # --- 1. Data Preparation and Filtering ---
    # Filter only colliding cells, as non-colliding cells have ratio = 0
    colliding_cells_df = cell_summary_df[cell_summary_df['Is_Colliding_Cell']].copy()

    # Apply optional filtering based on minimum features/UMIs
    sub_title_parts: List[str] = []
    if min_features_GEX is not None:
        colliding_cells_df = colliding_cells_df[colliding_cells_df['Total_Features_GEX'] >= min_features_GEX]
        sub_title_parts.append(f"Min GEX Features:{min_features_GEX}")
    if min_UMIs_GEX is not None:
        colliding_cells_df = colliding_cells_df[colliding_cells_df['Total_UMIs_GEX'] >= min_UMIs_GEX]
        sub_title_parts.append(f"Min GEX UMIs:{min_UMIs_GEX}")
    if min_features_sgRNA is not None:
        colliding_cells_df = colliding_cells_df[colliding_cells_df['Total_Features_sgRNA'] >= min_features_sgRNA]
        sub_title_parts.append(f"Min sgRNA Features:{min_features_sgRNA}")
    if min_UMIs_sgRNA is not None:
        colliding_cells_df = colliding_cells_df[colliding_cells_df['Total_UMIs_sgRNA'] >= min_UMIs_sgRNA]
        sub_title_parts.append(f"Min sgRNA UMIs:{min_UMIs_sgRNA}")
    
    filter_str = " | ".join(sub_title_parts) if sub_title_parts else "None"
    sub_title = f"In Cross-library Colliding CBCs (Filters Applied: {filter_str}, N={len(colliding_cells_df)})"

    # Define the 6 columns to be plotted (order matters for the 2x3 layout)
    RATIO_COLS = [
        'Collision_UMIs_Ratio_GEX','Collision_Reads_Ratio_GEX','Collision_Features_Ratio_GEX',
        'Collision_UMIs_Ratio_sgRNA','Collision_Reads_Ratio_sgRNA','Collision_Features_Ratio_sgRNA'
    ]
    
    if colliding_cells_df.empty:
        print("Warning: No colliding cells found. Cannot generate ratio boxplots.")
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, "No Colliding Cells", ha='center', va='center')
        ax.axis('off')
        return fig

    # --- 2. Plotting Setup and Styling ---
    set_publication_style()
    
    # Create 2 rows and 3 columns for 6 subplots
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize)
    axes = axes.flatten() # Flatten the 2x3 array of axes for easy iteration

    # --- 3. Plotting Loop ---
    for i, col in enumerate(RATIO_COLS):
        ax = axes[i]
        
        # Determine color and library based on column name
        is_gex = 'GEX' in col
        color = COLOR_GEX if is_gex else COLOR_SGRNA
        library_name = "GEX" if is_gex else "sgRNA"
            
        # Determine metric (UMIs, Reads, or Features)
        metric_name = col.split('_')[1]

        sns.boxplot(
            y=colliding_cells_df[col], 
            ax=ax, 
            color=color,
            width=0.5,
            fliersize=1.5,
            medianprops={'color': 'black', 'linewidth': 1}
        )
        
        # --- Customization per Subplot ---
        
        # Title: Only use the Metric Name (Library is implied by the row/color)
        ax.set_title(metric_name, fontsize=plt.rcParams['axes.titlesize'])
        
        # Y-Label: Set for the entire figure in the main title, or selectively for the first column.
        # Here we apply a detailed label for the first column of each row for clarity.
        if i % 3 == 0: # Only apply Y-label to the first column of each row
            ax.set_ylabel(f"Collision Ratio - {library_name}", fontsize=plt.rcParams['axes.labelsize'])
        else:
            ax.set_ylabel("")

        ax.set_xlabel("") # Remove x-label 
        
        # Clean up x-ticks and y-axis appearance
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        sns.despine(ax=ax, top=True, right=True)
        
        # Add a horizontal line for the median value
        median_val = colliding_cells_df[col].median()
        ax.axhline(median_val, color='red', linestyle='--', linewidth=0.7, alpha=0.5)

        # Set Y-limit explicitly for ratio plots
        # ax.set_ylim(0, 1.05) # Allow slightly over 1 just in case, but visually max at 1

    # --- 4. Final Figure Refinement ---
    # Add a main super title for the entire figure
    fig.suptitle(
        f"Distribution of Collision Ratios at CBC Level\n{sub_title}",
        fontsize=plt.rcParams['axes.titlesize'] + 2, 
        y=1.0
    )
    
    # Adjust layout: Use tight_layout to clean up spacing, and rect to avoid suptitle overlap
    fig.tight_layout(rect=[0, 0, 1, 0.98]) 

    # --- Final Cleanup ---
    plt.close(fig) 
    plt.rcParams.update(plt.rcParamsDefault)
    
    return fig

def plot_feature_purity_distribution(
    feature_impact_df: pd.DataFrame,
    library_type: str, 
    top_n_features: int = 30,
    min_feature_umis: int = 3,
    figsize: Tuple[float, float] = (7.5, 3.5),
) -> plt.Figure:
    
    # --- 1. Data Preparation and Filtering ---
    library_type = library_type.upper()
    # A. Apply required filters
    plot_data = feature_impact_df[
        (feature_impact_df['Total_Feature_UMIs'] >= min_feature_umis) &
        (feature_impact_df['Colliding_Feature_UMIs'] > 0)
    ].copy()
    
    if plot_data.empty:
        # ... (Empty handling remains the same) ...
        print(f"Warning: No valid {library_type} feature collision records found after filtering.")
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, f"No Colliding {library_type} Features (After Filters)", ha='center', va='center')
        ax.axis('off')
        return fig

    # --- 2. Feature Ranking (Median Ratio) ---

    ranking_df = plot_data.groupby('Feature_Name', observed=True)[ 
        'Collision_UMI_Ratio_In_Feature'
    ].median().reset_index(name='Median_Ratio').sort_values(
        by='Median_Ratio', ascending=False
    )
    
    top_features = ranking_df.head(top_n_features)['Feature_Name'].tolist()
    
    # Filter the main data and set categorical order for plotting
    plot_data = plot_data[plot_data['Feature_Name'].isin(top_features)].copy()
    plot_data['Feature_Name'] = pd.Categorical(
        plot_data['Feature_Name'], 
        categories=top_features, 
        ordered=True
    )
    
    # --- 3. Plotting Setup and Styling ---
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)
    color = COLOR_GEX if library_type == 'GEX' else COLOR_SGRNA
    
    # --- 4. Plot Vertical Boxplot ---
    sns.boxplot(
        data=plot_data,
        x='Feature_Name',
        y='Collision_UMI_Ratio_In_Feature',
        ax=ax,
        color=color,
        width=0.6,
        fliersize=1.5,
        medianprops={'color': 'black', 'linewidth': 1}
    )
    
    # --- 5. Customization and Dynamic Refinement ---
    # A. Dynamic Y-Limit Calculation and Annotations
    # Calculate MAX ratio for each feature for annotation position
    # Use observed=True again
    max_ratios = plot_data.groupby('Feature_Name', observed=True)[
        'Collision_UMI_Ratio_In_Feature'
    ].max()
    overall_max_ratio = max_ratios.max()
    
    # Set fixed offset for annotations (small fixed value, e.g., 0.01)
    annotation_offset = 0.01
    # New Y-Limit: Max ratio + offset + buffer (e.g., 5% of max or 0.1, whichever is larger)
    buffer = max(0.05, overall_max_ratio * 0.05)
    final_ylim = overall_max_ratio + annotation_offset + buffer
    # Force the Y-limit setting AFTER the plot generation
    ax.set_ylim(0, final_ylim) 
    
    # B. Add CBC Count Annotations (Dynamic Position)
    cbc_counts = plot_data.groupby('Feature_Name', observed=True).size()
    
    for i, feature_name in enumerate(top_features):
        count = cbc_counts.get(feature_name, 0)
        
        # Use the calculated max_ratios + offset for dynamic positioning
        y_pos = max_ratios.get(feature_name, 0) + annotation_offset
        ax.text(
            x=i, 
            y=y_pos, 
            s=str(count), 
            ha='center', 
            va='bottom', 
            fontsize=plt.rcParams['font.size']
        )
    
    # C. Set Labels and Titles (rest remains the same)
    ax.set_title(
        f"Distribution of Collision UMI Ratio per Feature in CBCs\nTop {len(top_features)} {library_type} Features Ranked by Median Ratio (Min UMIs:{min_feature_umis}, CBCs Count labeled)",
        fontsize=plt.rcParams['axes.titlesize'] + 2,
        y=1.0
    )
    ax.set_ylabel(
        "Collision UMI Ratio in CBCs\n(Colliding UMIs / Total UMIs)",
        fontsize=plt.rcParams['axes.labelsize']
    )
    ax.set_xlabel(
        f"Feature ID ({library_type} Library)",
        fontsize=plt.rcParams['axes.labelsize']
    )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    sns.despine(ax=ax, top=True, right=True)
    
    fig.subplots_adjust(bottom=0.25, top=0.90) 
    fig.tight_layout()

    # --- Final Cleanup ---
    plt.close(fig) 
    plt.rcParams.update(plt.rcParamsDefault)
    
    return fig

def plot_top_colliding_features(
    feature_impact_df: pd.DataFrame,
    library_type: str, # Must be 'GEX' or 'sgRNA'
    top_n_features: int = 30, # Typically plot more features for total contribution
    min_feature_umis: int = 3,
    figsize: Tuple[float, float] = (7.5, 3.5), # Wide figure for 30 bars
) -> plt.Figure:
    """
    Generates a barplot showing the absolute total number of Colliding UMIs 
    contributed by the Top N features, ranked by their total contribution.

    Args:
        feature_impact_df (pd.DataFrame): DataFrame containing cell-feature level statistics.
        library_type (str): Specifies the library being analyzed ('GEX' or 'sgRNA').
        top_n_features (int): Number of top features to plot.
        min_feature_umis (int): Minimum Total_Feature_UMIs required for a cell-feature record 
                                (applied before aggregation).
        figsize (Tuple[float, float]): Figure size in inches.
    
    Returns:
        plt.Figure: The generated matplotlib Figure object.
    """
    
    # --- 1. Data Preparation and Filtering ---
    library_type = library_type.upper()
    
    # Apply required UMIs filter before aggregation
    filtered_data = feature_impact_df[
        feature_impact_df['Total_Feature_UMIs'] >= min_feature_umis
    ].copy()

    # Aggregate by Feature_Name to calculate Total Colliding UMIs
    ranking_df = filtered_data.groupby('Feature_Name', observed=True).agg(
        Total_Colliding_UMIs=('Colliding_Feature_UMIs', 'sum')
    ).reset_index()

    # Sort and select the Top N features based on total contribution (Sum)
    ranking_df = ranking_df.sort_values(
        by='Total_Colliding_UMIs', 
        ascending=False
    ).head(top_n_features)
    
    # Remove features with zero contribution (if they somehow made it through)
    ranking_df = ranking_df[ranking_df['Total_Colliding_UMIs'] > 0].copy()

    # Check for sufficient data
    if ranking_df.empty:
        print(f"Warning: No significant colliding {library_type} features found after filtering/aggregation.")
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, f"No Impacting {library_type} Features", ha='center', va='center')
        ax.axis('off')
        return fig

    # Set categorical order for plotting
    top_features = ranking_df['Feature_Name'].tolist()
    ranking_df['Feature_Name'] = pd.Categorical(
        ranking_df['Feature_Name'], 
        categories=top_features, 
        ordered=True
    )
    
    # --- 2. Plotting Setup and Styling ---
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)
    color = COLOR_GEX if library_type == 'GEX' else COLOR_SGRNA
    
    # --- 3. Plot Generation (Vertical Barplot) ---
    sns.barplot(
        data=ranking_df,
        x='Feature_Name',
        y='Total_Colliding_UMIs',
        ax=ax,
        color=color,
        edgecolor=None
    )
    
    # --- 4. Customization and Refinement ---
    # A. Titles and Labels
    ax.set_title(
        f"Total Colliding UMI Contribution of Top {len(top_features)} {library_type} Features",
        fontsize=plt.rcParams['axes.titlesize'] + 2,
        y=1.0
    )
    ax.set_ylabel(
        "Total Colliding UMIs (Across all CBCs)",
        fontsize=plt.rcParams['axes.labelsize']
    )
    ax.set_xlabel(
        f"Feature ID ({library_type} Library)",
        fontsize=plt.rcParams['axes.labelsize']
    )
    
    # B. Aesthetics and Cleanup
    # Rotate X-tick labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    sns.despine(ax=ax, top=True, right=True)
    
    # Adjust Y-limit slightly above the maximum bar height
    max_umis = ranking_df['Total_Colliding_UMIs'].max()
    ax.set_ylim(0, max_umis * 1.1)

    # Ensure there's enough bottom margin for the rotated labels
    fig.subplots_adjust(bottom=0.25, top=0.90) 
    fig.tight_layout()

    # --- Final Cleanup ---
    plt.close(fig) 
    plt.rcParams.update(plt.rcParamsDefault)
    
    return fig

# --- Part 2: Specific sgRNA Visualization Plots ---

from scipy.sparse import csr_matrix

def plot_sgRNA_density(
    sgRNA_matrix: csr_matrix,
    sgRNA_name_to_plot: str,
    output_dir: str,
    log_transform: bool = True,
    binwidth: Optional[float] = 0.25,
    show_kde: bool = True,
    max_count: Optional[int] = None, 
    figsize: Tuple[float, float] = (3, 2),
    hist_color: str = "grey",
    kde_color: str = "black",
) -> plt.Figure:
    """
    Generates and saves the UMI count density plot for a single sgRNA.
    
    Args:
        sgRNA_matrix (csr_matrix): Sparse matrix of sgRNA UMI counts (CBCs x Features).
        sgRNA_name_to_plot (str): The name of the sgRNA feature to plot.
        output_dir (str): Directory path to save the PDF and PNG files.
        log_transform (bool): If True, use log2(UMI + 1) for the x-axis.
        binwidth (Optional[float]): Width of histogram bins. Default is 0.25 for log scale.
        show_kde (bool): If True, overlay a KDE density curve.
        max_count (Optional[int]): Optional x-axis limit for UMI counts (pre-log integer).
        figsize (Tuple[float, float]): Size of the figure in inches.

    Returns:
        str: The file path of the saved PNG image, used for report integration.
    """
    set_publication_style()

    # 1. Data Extraction and Transformation
    # Checking
    if not hasattr(sgRNA_matrix, 'feature_names'):
        raise AttributeError("sgRNA_matrix must have a 'feature_names' attribute.")
    
    sgRNA_names = sgRNA_matrix.feature_names
    
    try:
        # Find the column index for the target sgRNA
        col_idx = sgRNA_names.index(sgRNA_name_to_plot)
    except ValueError:
        raise ValueError(f"sgRNA '{sgRNA_name_to_plot}' not found in feature_names.")

    # Extract and convert to a dense NumPy array (efficiently for CSR)
    umi_data = sgRNA_matrix[:, col_idx].toarray().flatten()

    # 2. Data Preparation and Filtering
    # Only cells where the sgRNA was detected (UMI > 0)
    data_for_plot = umi_data[umi_data > 0]
    
    if data_for_plot.size == 0:
        print(f"Warning: sgRNA '{sgRNA_name_to_plot}' has no detected UMI counts (>0). Check your data! Plot will be skipped.")
        return None

    if log_transform:
        plot_data = np.log2(data_for_plot + 1)
        x_label = "log2(UMI Count + 1)"
        if max_count is not None:
             # Calculate the max log value for consistent axis limits
            x_limit = np.log2(max_count + 1)
        else:
            x_limit = np.max(plot_data) * 1.05
    else:
        plot_data = data_for_plot
        x_label = "UMI Count"
        x_limit = max_count if max_count is not None else np.max(plot_data) * 1.05

    # 3. Plotting (Matplotlib/Seaborn)
    fig, ax = plt.subplots(figsize=figsize)
    
    # robust hist kde (Avoid all values being the same resulting in bins being 0)
    if np.all(plot_data == plot_data[0]):
        bin_width = 1.0
        center = plot_data[0]

        sns.histplot(plot_data, bins=[center - bin_width / 2, center + bin_width / 2],
                     color="red", stat="density", edgecolor='black', ax=ax,
                     kde=False, alpha=0.8)
        if log_transform:
            ax.set_xlim(0, 13)
        else:
            ax.set_xlim(0, x_limit)

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        ax.text(center_x, center_y, 'All values equal', ha='center', va='center',
                        fontsize=12, color='blue', weight='bold')
    else:
        sns.histplot(
            plot_data, binwidth=binwidth, stat="density", kde=False, 
            color=hist_color, edgecolor='black', 
            ax=ax, alpha=0.7
        )
    
    if show_kde and len(np.unique(plot_data)) > 1:
        sns.kdeplot(plot_data, ax=ax, color=kde_color, bw_adjust=1.5, lw=1.5, alpha=0.6)
    
    desc = pd.Series(data_for_plot).describe()
    stats_text = (
        f'n_CBCs = {int(desc["count"])}\n'
        f'   min = {int(desc["min"])}\n'
        f'  mean = {desc["mean"]:.1f}\n'
        f'median = {desc["50%"]:.1f}\n'
        f'   max = {int(desc["max"])}'
    )
    ax.text(0.75, 0.95, stats_text,
            transform=ax.transAxes,
            ha='left', va='top',
            font = 'monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.3))

    # Set plot title and labels
    ax.set_title(f"UMI Density for {sgRNA_name_to_plot}", fontsize=plt.rcParams['axes.titlesize'] + 2)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")

    if x_limit > 0:
        ax.set_xlim(0, x_limit)

    plt.tight_layout()

    # 4. Saving Outputs (PNG and PDF)
    os.makedirs(output_dir, exist_ok=True)
    
    # Define clean file names
    base_filename = f"{sgRNA_name_to_plot.replace('/', '_')}_density_log{log_transform}"
    png_filepath = os.path.join(output_dir, f"{base_filename}.png")
    pdf_filepath = os.path.join(output_dir, f"{base_filename}.pdf")

    plt.savefig(pdf_filepath, format='pdf')
    plt.savefig(png_filepath, format='png', dpi=300)

    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)

    return fig

def plot_sgRNA_scatter(
    sgrna_mtx_raw: csr_matrix,
    sgrna_mtx_cleaned: csr_matrix,
    sgRNA_name_to_plot: str,
    output_dir: str,
    log_transform: bool = True,
    show_purity_lines: bool = True,
    figsize: Tuple[float, float] = (2.8, 2.5),
    cmap: str = 'Reds',
    return_data: bool = False
) -> Tuple[Optional[plt.Figure], Optional[pd.DataFrame]]:
    """
    Generates a scatter plot for a single sgRNA, showing its UMI count vs. Total Cell sgRNA UMI Count,
    colored by the collision contamination ratio.

    Args:
        sgrna_mtx_raw (csr_matrix): Original sgRNA UMI count matrix (CBCs x Features).
        sgrna_mtx_cleaned (csr_matrix): Cleaned sgRNA UMI count matrix after Strict Discard.(Same shape as raw)
        sgRNA_name_to_plot (str): The name of the sgRNA feature to plot.
        output_dir (str): Directory path to save the PDF and PNG files.
        log_transform (bool): If True, both X and Y axes use log2(Count + 1).
        show_purity_lines (bool): If True, plot y=x (100% purity) and y=x+offset (e.g., 80% purity) lines.
        figsize (Tuple[float, float]): Size of the figure in inches.
        cmap (str): Colormap for the contamination ratio.
        return_data (bool): If True, returns a DataFrame with raw data and contamination ratio.

    Returns:
        Tuple[Optional[plt.Figure], Optional[pd.DataFrame]]: The generated Figure object 
        and an optional DataFrame containing the plot data. Returns (None, None) on failure.
    """
    
    set_publication_style()

    # Initialize return values
    df_out = None
    
    # 1. Data Integrity and Indexing Check
    if sgrna_mtx_raw.shape != sgrna_mtx_cleaned.shape:
        print("Error: Raw and cleaned matrices must have the same shape. Using Func 'align_sparse_matrix()' to fix this issue before plotting.")
        return None, None
        
    if not hasattr(sgrna_mtx_raw, 'feature_names') or not hasattr(sgrna_mtx_raw, 'barcode_names'):
        print("Error: sgRNA matrices must have 'feature_names' and 'barcode_names' attributes.")
        return None, None
    
    sgRNA_names = sgrna_mtx_raw.feature_names
    barcode_names = sgrna_mtx_raw.barcode_names
    
    try:
        col_idx = sgRNA_names.index(sgRNA_name_to_plot)
    except ValueError:
        print(f"Error: sgRNA '{sgRNA_name_to_plot}' not found in feature_names.")
        return None, None

    # 2. Core Data Calculation (X, Y, and Contamination Ratio)
    # --- Y-axis: Total sgRNA UMI Count (Raw) ---
    # Sum across rows (axis=1) and flatten to a dense 1D array
    total_umi_raw = sgrna_mtx_raw.sum(axis=1).A.flatten()

    # --- X-axis: Target sgRNA UMI Count (Raw) ---
    umi_k_raw = sgrna_mtx_raw[:, col_idx].toarray().flatten()

    # --- Collision UMI Count (Raw - Cleaned) ---
    # Collision_mtx = sgrna_mtx_raw - sgrna_mtx_cleaned
    # Since they are sparse, subtraction is efficient. Extract the target column:
    collision_umi_k = (sgrna_mtx_raw[:, col_idx] - sgrna_mtx_cleaned[:, col_idx]).toarray().flatten()
    
    # --- Contamination Ratio (C_i) ---
    # C_i = Collision_k / UMI_k_raw. Handle division by zero.
    contamination_ratio = np.divide(
        collision_umi_k, 
        umi_k_raw, 
        out=np.zeros_like(umi_k_raw, dtype=float), # Output array for results
        where=umi_k_raw != 0 # Only compute where denominator is non-zero
    )
    
    # Ratios greater than 1.0 (due to floating point precision or rare edge cases) should be capped at 1.0
    contamination_ratio = np.clip(contamination_ratio, 0.0, 1.0)
    
    # 3. Filtering: Only consider cells where the target sgRNA was detected (X_raw > 0)
    detected_mask = umi_k_raw > 0
    
    X_plot = umi_k_raw[detected_mask]
    Y_plot = total_umi_raw[detected_mask]
    C_plot = contamination_ratio[detected_mask]
    
    if len(X_plot) == 0:
        print(f"Warning: sgRNA '{sgRNA_name_to_plot}' was not detected in any CBCs. Plot skipped.")
        return None, None

    # 4. Data Transformation for Plotting
    X_final = X_plot
    Y_final = Y_plot
    x_label = f"{sgRNA_name_to_plot} UMI Count"
    y_label = "Total sgRNA UMI Count (CBC Depth)"

    if log_transform:
        # Apply log2(Count + 1) transformation to both axes
        X_final = np.log2(X_plot + 1)
        Y_final = np.log2(Y_plot + 1)
        x_label = f"log2({sgRNA_name_to_plot} UMI + 1)"
        y_label = "log2(Total sgRNA UMI + 1)"

    # 5. Plotting
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use scatter plot, colored by the contamination ratio (C_plot)
    # Set vmax in the condition of X = Y = 1 only
    if np.max(C_plot) == 1:
        scatter = ax.scatter(
            X_final, Y_final, 
            c=C_plot, 
            cmap=cmap, 
            s=6, # Dot size
            edgecolors='dimgray',
            linewidths=0.3,
            # alpha=0.8,
            vmin=0, vmax=0.5 # Ensure color bar spans 0 to 0.5
        )
    else:
        scatter = ax.scatter(
            X_final, Y_final, 
            c=C_plot, 
            cmap=cmap, 
            s=6, # Dot size
            edgecolors='dimgray',
            linewidths=0.3,
            # alpha=0.8,
            vmin=0, vmax=np.max(C_plot) # Dynamic max for color bar
        )
    
    # Add Color Bar
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03, aspect=30)
    cbar.set_label("Collision Contamination Ratio", rotation=270, labelpad=8)

    # Add Purity Lines (if requested)
    if show_purity_lines:
        # Determine shared axis range based on current data limits
        x_data_min, x_data_max = np.min(X_final), np.max(X_final)
        y_data_min, y_data_max = np.min(Y_final), np.max(Y_final)
        
        # Use a unified limit for the square plot
        # min_val = min(x_data_min, y_data_min)
        min_val = 0
        max_val = max(x_data_max, y_data_max) * 1.05

        # 80% Purity Line: log2(Y) = log2(X * (1/0.8)) in raw space, or log2(Y) = log2(X) + log2(1/0.8)
        offset = np.log2(1.25) # Since 1/0.8 = 1.25. log2(Y) = log2(X) + log2(1/Purity)
        ax.plot([min_val, max_val - offset], [min_val + offset, max_val], 
                ls='-.', color='gray', label=' 80% Purity', alpha=0.7)
        
        # 100% Purity Line: y = x (in log space)
        ax.plot([min_val, max_val], [min_val, max_val], 
                ls='--', color='gray', label='100% Purity (y=x)', alpha=0.7)
        
        ax.legend(loc='lower right', frameon=True)
        
        # Set unified axes limits to make the purity lines meaningful
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

    # add stats text box
    desc = pd.Series(contamination_ratio[contamination_ratio > 0]).describe()
    stats_text = (
        f'Detected CBCs = {int(len(X_plot))}\n'
        f' Contam. CBCs = {int(desc["count"])}\n'
        f'   Ratio min = {desc["min"]:.3f}\n'
        f'  Ratio mean = {desc["mean"]:.3f}\n'
        f'Ratio median = {desc["50%"]:.3f}\n'
        f'   Ratio max = {desc["max"]:.3f}'
    )
    ax.text(0.52, 0.385, stats_text,
            transform=ax.transAxes,
            ha='left', va='top',
            font = 'monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.3))
    
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"Contamination & Purity of {sgRNA_name_to_plot}", fontsize=plt.rcParams['axes.titlesize'] + 2)

    plt.tight_layout()

    # 6. Saving Outputs (PNG and PDF)
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = f"{sgRNA_name_to_plot.replace('/', '_')}_scatter_log{log_transform}"
    png_filepath = os.path.join(output_dir, f"{base_filename}.png")
    pdf_filepath = os.path.join(output_dir, f"{base_filename}.pdf")

    plt.savefig(pdf_filepath, format='pdf')
    plt.savefig(png_filepath, format='png', dpi=300)

    # 7. Prepare DataFrame Output (if requested)
    if return_data:
        # Get the names of the cells that were plotted
        barcode_names_plotted = np.array(barcode_names)[detected_mask]
        
        df_out = pd.DataFrame({
            'CBC_Name': barcode_names_plotted,
            'Raw_UMI_Count': X_plot.astype(int),
            'Total_sgRNA_UMI': Y_plot.astype(int),
            'Collision_UMI_Count': collision_umi_k[detected_mask].astype(int),
            'Contamination_Ratio': C_plot
        })
        # Sort by contamination ratio descending, as requested
        df_out = df_out.sort_values(by='Contamination_Ratio', ascending=False).reset_index(drop=True)

    # Close the figure to free memory
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)

    return fig, df_out

# --- Part 3: Assignment & Exploratory Data Analysis (EDA) Plots ---

# 3.1 Dominance EDA Dual-Metric JointGrid Plot

def plot_eda_dual_metrics(
    eda_df: pd.DataFrame,
    output_dir: str,
    X_COL: str,
    Y_COL: str,
    min_umi_threshold: int = 10,
    plot_height: float = 3.5, 
    main_plot_kind='kde', # 'scatter' or 'kde'
    marginal_bins=50,
    marginal_color: str = "#984EA3",
    kde_cmap: str = "mako_r",
    scatter_color: str = "#A3A3A3", 
    detailed_mode: bool = False
) -> str:
    """
    Plot EDA dual-metric JointGrid for sgRNA dominance analysis.
    metrics include Ratio_1, Ratio_2, Ratio_3, LogRatio_12, LogRatio_23, LogRatio_13, n_nonzero, Total_sgRNA_UMI, Total_sgRNA_UMI_10.
    color recommendation:
    1. #34495e (dark blue-gray) mako_r
    2. #e74c3c (red) rocket_r
    3. #27ae60 (green) viridis_r
    4. #8e44ad (purple) crest_r

    Args:
        eda_df (pd.DataFrame): DataFrame containing EDA statistics at CBC level(generated by extract_dominance_data()).
        output_dir (str): Directory to save output plots.
        X_COL (str): X-axis indicator column metric name.
        Y_COL (str): Y-axis indicator column metric name.
        min_umi_threshold (int): Minimum Total_sgRNA_UMI to include a CBC in the plot.
        plot_height (float): JointGrid plot height in inches.
        main_plot_kind (str): 'scatter' or 'kde'
        marginal_bins (int): Number of bins for the marginal histograms.
        marginal_color (str): Color for marginal histograms.
        kde_cmap (str): KDE colormap name.
        scatter_color (str): Color for scatter points.
        detailed_mode (bool): If True, includes more detailed print statements for debugging.

    Returns:
        str: The file path of the saved PNG image.
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 1. Data Preparation
    if detailed_mode:
        print(f"[{X_COL} vs {Y_COL}]")

    LABEL_MAP = {
    'Ratio_1': 'Ratio 1 (Top1 UMI / Total UMI)',
    'Ratio_2': 'Ratio 2 (Top2 UMI / Total UMI)',
    'Ratio_3': 'Ratio 3 (Top3 UMI / Total UMI)',
    'LogRatio_12': 'Log2(Top1 UMI / Top2 UMI)',
    'LogRatio_23': 'Log2(Top2 UMI / Top3 UMI)',
    'LogRatio_13': 'Log2(Top1 UMI / Top3 UMI)',
    'n_nonzero': 'Number of sgRNAs Detected (n_nonzero)',
    'Total_sgRNA_UMI': 'Total sgRNA UMI Count',
    'Log_Total_UMI': 'Log10(Total sgRNA UMI Count + 1)',
    }
    
    # x_y_label generation
    plot_data = eda_df[eda_df['Total_sgRNA_UMI'] > 0].copy()
    plot_data = plot_data[plot_data['Total_sgRNA_UMI'] >= min_umi_threshold].copy()
    x_col_plot, y_col_plot = X_COL, Y_COL
    
    # log transform for Total_sgRNA_UMI if needed
    if 'Total_sgRNA_UMI' in [X_COL, Y_COL]:
        plot_data['Log_Total_UMI'] = np.log10(plot_data['Total_sgRNA_UMI'] + 1)
        if X_COL == 'Total_sgRNA_UMI':
            x_col_plot = 'Log_Total_UMI'
        if Y_COL == 'Total_sgRNA_UMI':
            y_col_plot = 'Log_Total_UMI'
            
    # filtering invalid data
    plot_data = plot_data[
        plot_data[x_col_plot].notnull() 
        & plot_data[y_col_plot].notnull() 
        & (plot_data[x_col_plot] > 0)
        & (plot_data[y_col_plot] > 0)
    ].copy()
    
    if plot_data.empty:
        print(f"Warning: No valid data points to plot for {X_COL} vs {Y_COL} after filtering.")
        return ""

    # 2. setting style
    set_publication_style() 

    # 3. JointGrid
    g = sns.JointGrid(
        data=plot_data, 
        x=x_col_plot, 
        y=y_col_plot, 
        height=plot_height, # size equals both width and height
        ratio=5,
        space=0.1
    )

    # --- 3.1. Marginal Plots ---
    # Histplot
    sns.histplot(x=plot_data[x_col_plot], ax=g.ax_marg_x, bins=marginal_bins, color=marginal_color)
    sns.histplot(y=plot_data[y_col_plot], ax=g.ax_marg_y, bins=marginal_bins, color=marginal_color)

    # --- 3.2. Joint Plot ---
    if main_plot_kind == 'scatter':
        sns.scatterplot(
            x=plot_data[x_col_plot],
            y=plot_data[y_col_plot],
            ax=g.ax_joint,
            color=scatter_color,
            s=8, 
            alpha=0.6,
            linewidth=0 # No edge color
        )
    elif main_plot_kind == 'kde':
        sns.kdeplot(
            data=plot_data,
            x=x_col_plot,
            y=y_col_plot,
            fill=True,
            cmap=kde_cmap,
            thresh=0.05,
            levels=20,
            ax=g.ax_joint
        )
    else:
        print(f"Warning: Unknown main_plot_kind '{main_plot_kind}'. Skipping joint plot.")

    # 4. specific customization
    x_label = LABEL_MAP.get(x_col_plot, X_COL)
    y_label = LABEL_MAP.get(y_col_plot, Y_COL)
    
    g.ax_joint.set_xlabel(x_label)
    g.ax_joint.set_ylabel(y_label)

    # Add sample count annotation
    sample_count = len(plot_data)
    text_str = f"N = {sample_count:,} CBCs"
    g.ax_joint.text(
        0.40, 0.04,  # Lower left corner position (relative coordinates, between 0 and 1) 
        text_str,
        transform=g.ax_joint.transAxes,  # Use axis coordinate system
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
    )

    # Add reference lines for Ratio plots
    if X_COL in ['Ratio_1', 'Ratio_2', 'Ratio_3'] and Y_COL in ['Ratio_1', 'Ratio_2', 'Ratio_3']:

        slopes = [1, 2, 3, 4, 6, 8]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(slopes)))

        legend_lines = []
        legend_labels = []

        for slope, color in zip(slopes, colors):
            line = g.ax_joint.axline(
                (0, 0),
                slope=slope,
                color=color,
                linewidth=1.5,
                alpha=0.8,
                zorder=1,
            )
            legend_lines.append(line)
            legend_labels.append(f'k={slope}')

        leg = g.ax_joint.legend(
            legend_lines,
            legend_labels,
            loc='lower right',
            frameon=True,
            fontsize=5,
            title="Slope",
            title_fontsize=6
        )

        leg.get_frame().set_alpha(0.5)

    # Clean up the plot boundaries
    sns.despine(ax=g.ax_joint, top=False, right=False) # Keep the boundary box for Joint
    sns.despine(ax=g.ax_marg_x, left=True, bottom=True) # Clean up marginal axes
    sns.despine(ax=g.ax_marg_y, left=True, bottom=True)

    # Remove axis labels and ticks from marginal plots to keep it clean
    g.ax_marg_x.set_ylabel('')
    g.ax_marg_x.set_xlabel('')
    g.ax_marg_x.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    g.ax_marg_x.tick_params(axis='y', which='both', left=True, labelleft=True)

    g.ax_marg_y.set_ylabel('')
    g.ax_marg_y.set_xlabel('')
    g.ax_marg_y.tick_params(axis='y', which='both', left=False, labelleft=False)
    g.ax_marg_y.tick_params(axis='x', which='both', bottom=True, labelbottom=True)

    g.figure.tight_layout()
    
    # 5. save both PDF and PNG
    filename_base = f"EDA_DualMetrics_{Y_COL}_vs_{X_COL}_minUMI{min_umi_threshold}"
    
    def save_plot(filename_base, fig):
        """Helper to save both PDF and PNG"""
        pdf_path = os.path.join(output_dir, f"{filename_base}.pdf")
        png_path = os.path.join(output_dir, f"{filename_base}.png")
        
        fig.savefig(pdf_path, bbox_inches='tight', format='pdf')
        fig.savefig(png_path, bbox_inches='tight', format='png', dpi=300)
        return png_path
        
    png_filepath = save_plot(filename_base, g.figure)
    if detailed_mode:
        print(f"EDA plot saved: {png_filepath}")

    # 6. Final Cleanup
    plt.close(g.figure)
    plt.rcParams.update(plt.rcParamsDefault)

    return png_filepath

# 3.2 GMM Assignment Visualization Plots

from scipy.stats import norm
import logging
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('fontTools').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
# 3.2.1 Single sgRNA GMM Fit Plot
def _plot_gmm_single(ax, feature_name, counts, feature_stats, global_min_threshold=None):
    """
    Plot the GMM fit details for a single sgRNA on the specified Axes.
    
    Args:
        ax: matplotlib Axes object to plot on.
        feature_name (str): sgRNA name.
        counts (np.array): The sgRNA's original non-zero UMI count array across all CBCs.
        feature_stats (dict): Dictionary obtained from GMMAssigner.feature_stats[name] (contains 'model_obj',' thresholds','n_components')
        global_min_threshold (float, optional): Global Minimum Protection Threshold recorded in GMMAssigner.global_min_threshold (log2 scale)
    """
    # 1. prepare data
    counts_nonzero = counts[counts > 0]
    n_cells = len(counts_nonzero)
    
    # title with n_cells
    ax.set_title(f"{feature_name}\nn={n_cells}", fontsize=10)
    
    # Extreme case: no data
    if n_cells == 0:
        ax.text(0.5, 0.5, "No counts detected", ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return

    # log-transform counts
    log_counts = np.log2(counts_nonzero + 1)
    
    # 2. Draw histogram (Density=True to align with PDF)
    # Dynamic calculation of bins: ensuring that sparse data can also be seen
    bins = np.linspace(log_counts.min(), log_counts.max() + 1, 40)
    ax.hist(log_counts, bins=bins, density=True, color='lightgray', alpha=0.8, edgecolor='white')

    # 3. Dynamic Reconstruction of GMM Curve
    model = feature_stats.get('model_obj')
    
    # Check whether the model is valid (sometimes because too few cells store string information)
    if model is not None and hasattr(model, 'means_'):
        # Generate smooth X-axis data points for PDF plotting
        x_plot = np.linspace(log_counts.min() - 0.5, log_counts.max() + 1.5, 500)
        
        # get GMM parameters
        means = model.means_.flatten()
        covars = model.covariances_.flatten()
        weights = model.weights_.flatten()
        
        # Sort to assign colors (Mean smallest is noise, largest is signal)
        idx = np.argsort(means)
        means, covars, weights = means[idx], covars[idx], weights[idx]
        
        # Components PDFs Color mapping
        if len(means) == 2:
            colors = ['green', 'red'] # Noise, Signal
        else:
            colors = ['green', 'orange', 'red'] # Noise, Middle, Signal
            
        total_pdf = np.zeros_like(x_plot)
        
        for i, (m, c, w) in enumerate(zip(means, covars, weights)):
            sigma = np.sqrt(c)
            pdf = w * norm.pdf(x_plot, m, sigma)
            total_pdf += pdf
            
            # Draw component curves (thin dashed lines)
            color = colors[i] if i < len(colors) else 'gray'
            ax.plot(x_plot, pdf, linestyle='--', linewidth=1, color=color, alpha=0.7)
        
        # Plot the overall fit curve (solid black line)
        ax.plot(x_plot, total_pdf, color='black', linewidth=1.5, alpha=0.9)

    # 4. Plot Threshold Lines
    thresholds = feature_stats.get('thresholds', [])
    
    # Plot calculated physical boundaries/probability thresholds
    if thresholds:
        if len(thresholds) == 1:
            # 2peak: Single threshold (blue line)
            ax.axvline(thresholds[0], color='tab:blue', linestyle='--', linewidth=1.5)
        elif len(thresholds) >= 2:
            # 3峰: Low (Loose, Blue) 和 High (Strict, blue)
            ax.axvline(thresholds[0], color='tab:blue', linestyle='--', linewidth=1.5)
            ax.axvline(thresholds[1], color='tab:blue', linestyle='--', linewidth=1.5)
            
    # 5. Plot Global Minimum Protection Threshold (if provided)
    if global_min_threshold is not None:
        ax.axvline(global_min_threshold, color='green', linestyle='-.', linewidth=1.2, alpha=0.6)

    # 样式微调
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel(None)
    ax.set_yticks([]) # Hide Y-axis ticks
    ax.set_xlabel("log2(UMI+1)", fontsize=8)

# 3.2.2 GMM Report Image Generator
def create_gmm_report_images(
    assigner,
    csr_matrix,
    sgRNA_list,
    output_dir="temp_report_images"
):
    """
    Generator function that generates a single PNG image.
    This function does not generate PDF report, but generates matplotlib Figure one by one according to sgRNA_list.
    and saved to temp_dir for use by external report_generator puzzles.
    
    Args:
        assigner (GMMAssigner): GMM Assigner object containing model and stats.
        csr_matrix: Sparse matrix of sgRNA UMI counts (CBCs x Features).
        sgRNA_list (list): List of orderd sgRNA names to generate plots for.
        output_dir (str): temporary directory to save generated PNG files.
    
    Returns:
        image_paths (list): png file paths list generated for report integration.
    """
    import os
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    image_paths = []
    
    # For quick reads, convert to CSC or utilize index mapping
    # Build mapping of feature_name -> column index
    feat_to_idx = {name: i for i, name in enumerate(csr_matrix.feature_names)}
    data_csc = csr_matrix.tocsc()
    
    logging.info(f"Generating {len(sgRNA_list)} plots for GMM report...")
    
    for i, name in enumerate(sgRNA_list):
        if name not in feat_to_idx:
            logging.warning(f"Skipping {name}: not found in matrix.")
            continue
            
        # 1. Access feature data
        col_idx = feat_to_idx[name]
        counts = data_csc[:, col_idx].toarray().flatten()
        
        # 2. Access feature GMM fited stats
        stats = assigner.feature_stats.get(name, {})
        
        # 3. Plotting
        fig, ax = plt.subplots(figsize=(4, 3), dpi=150) # 单图尺寸
        
        _plot_gmm_single(
            ax=ax, 
            feature_name=name, 
            counts=counts, 
            feature_stats=stats,
            global_min_threshold=assigner.global_min_threshold
        )
        
        # 4. save 
        save_path1 = os.path.join(output_dir, f"plot_{i:04d}_{name}.png")
        save_path2 = os.path.join(output_dir, f"plot_{i:04d}_{name}.pdf")
        plt.tight_layout()
        plt.savefig(save_path1, format='png', dpi=300)
        plt.savefig(save_path2, format='pdf')
        plt.close(fig)
        
        image_paths.append(save_path1)
        
    return image_paths