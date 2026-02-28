import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import scanpy as sc

# QC metric distribution plotting with peaks and troughs labeled, and optional CDF overlay
def plot_distribution_peaks_troughs(
    dataframe: pd.DataFrame,
    column: str = 'n_genes',
    bins=100,
    sigma=2,
    prominence=100,
    distance=3,
    height=None,
    show_labels=True,
    return_positions=False,
    show_cdf=True, 
    ax=None, 
    output_dir=None
):
    """
    Plot the distribution of n_genes and label the locations of peaks and valleys
    args:
        dataframe: DataFrame containing n_genes column
        column: The column name to be analyzed, default 'n_genes'
        bins: Number of bins in the histogram
        sigma: standard deviation of Gaussian smoothing
        prominence: peak prominence parameter
        distance: minimum distance between peaks
        height: minimum height of peak
        show_labels: whether to display the numerical labels of peaks and valleys
        return_positions: whether to return the peak and valley positions
        show_cdf: whether to display the cumulative distribution function (CDF) on the right axis
        ax: optional matplotlib axis to plot on; if None, a new figure and axis will be created
        output_dir: directory to save the png&pdf; if None, the figure will not be saved
    """

    n, bins = np.histogram(dataframe[column].dropna(), bins=bins)
    x = 0.5 * (bins[1:] + bins[:-1])   # bin centers

    # gaussian smoothing to reduce noise and make peak detection more robust
    y_smooth = gaussian_filter1d(n, sigma=sigma)

    # find peaks and troughs
    peak_args = {'prominence': prominence, 'distance': distance}
    if height is not None:
        peak_args['height'] = height

    peaks, _ = find_peaks(y_smooth, **peak_args)
    troughs, _ = find_peaks(-y_smooth, **peak_args)

    # main plot
    if ax is None:
        fig, ax1 = plt.subplots(figsize=(3, 2), layout='constrained')
    else:
        ax1 = ax
        fig = ax1.get_figure()

    ax1.plot(x, n, alpha=0.3, label="Histogram", drawstyle='steps-mid')
    ax1.plot(x, y_smooth, color='blue', label='Smoothed Curve')

    y_range = y_smooth.max() - y_smooth.min()
    offset = y_range * 0.05

    if show_labels:
        offset = (y_smooth.max() - y_smooth.min()) * 0.05
        for i in peaks: ax1.text(x[i], y_smooth[i] + offset, f"{x[i]:.1f}", ha='center', color='red')
        for i in troughs: ax1.text(x[i], y_smooth[i] - offset, f"{x[i]:.1f}", ha='center', color='green')

    ax1.set_xlabel(column)
    ax1.set_ylabel("Count")

    # CDF plot on the right axis
    if show_cdf:
        ax2 = ax1.twinx()
        cdf = np.cumsum(n) / np.sum(n)
        ax2.plot(x, cdf, color='orange', linewidth=2, label='Cumulative Density')
        ax2.set_ylabel("Cumulative Fraction")
        ax2.set_ylim(0, 1)

        # pct labels on y-axis
        ax2.set_yticks(np.linspace(0, 1, 6))
        ax2.set_yticklabels([f"{int(i*100)}%" for i in np.linspace(0,1,6)])

    # combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    if show_cdf:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2)
    else:
        ax1.legend()

    ax1.grid(True, alpha=0.3)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        png_path = os.path.join(output_dir, f"{column}_distribution.png")
        pdf_path = os.path.join(output_dir, f"{column}_distribution.pdf")
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')

    if return_positions:
        peak_positions = x[peaks]
        trough_positions = x[troughs]
    else:
        peak_positions = trough_positions = None

    plt.close(fig)

    if return_positions:
        return peak_positions, trough_positions, fig
    return fig

def plot_adata_qc(
        adata: sc.AnnData, 
        output_dir=None,
        prefix=""
        ) -> plt.Figure:

        """
        Plot multiple QC metric distributions with peaks and troughs labeled
        Draw it before and after filtering to show the effect of QC
        """
        qc_config = {
            'n_genes_by_counts': {'bins': 100, 'sigma': 2, 'prominence': 20},
            'total_counts': {'bins': 100, 'sigma': 2, 'prominence': 50},
            'pct_counts_mt': {'bins': 80, 'sigma': 1.5, 'prominence': 5}
        }
        
        metrics = [m for m in qc_config.keys() if m in adata.obs.columns]
        fig, axes = plt.subplots(1, len(metrics), figsize=(3 * len(metrics), 2), layout='constrained')
        
        if len(metrics) == 1: axes = [axes]

        for i, (ax, qc_metric) in enumerate(zip(axes, metrics)):  
            params = qc_config[qc_metric]
            plot_distribution_peaks_troughs(
                dataframe=adata.obs,
                column=qc_metric,
                ax=ax,
                **params
            )
            all_axes = ax.get_figure().get_axes()
            current_ax2 = None
            for other_ax in all_axes:
                if other_ax is not ax and other_ax.get_subplotspec() == ax.get_subplotspec():
                    current_ax2 = other_ax
                    break
            if i > 0:
                ax.set_ylabel("")
            if (current_ax2 is not None) & (i < len(metrics) - 1):
                current_ax2.set_ylabel("")
        
        # plt.tight_layout()
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            png_path = os.path.join(output_dir, f"{prefix}qc_metrics_distribution.png")
            pdf_path = os.path.join(output_dir, f"{prefix}qc_metrics_distribution.pdf")
            fig.savefig(png_path, dpi=300, bbox_inches='tight')
            fig.savefig(pdf_path, bbox_inches='tight')
        else:
            plt.close(fig)
            return fig
  
# density plot on umap
from anndata import AnnData
from typing import Tuple, List

def filter_and_embedding_density(
    adata: AnnData,
    group_key: str = "condition",
    basis: str = "umap",
    min_cells: int = 3,
    key_added: str = None,
) -> Tuple[AnnData, List[str]]:
    """
    Filter out groups whose number of cells is less than min_cells according to group_key, and 
    perform embedding_density calculation on the specified basis.

    args:
        adata: AnnData
        group_key: str The key in adata.obs to group cells.
        basis: str The embedding basis to use for density calculation.
        min_cells: int Minimum number of cells required for a group to be retained.
        key_added: str The key to store the density results in adata.obs.
    returns:
        adata_filtered: AnnData The filtered AnnData object.
        group_order: List[str] The list of groups sorted by cell count in descending order.
    """

    if group_key not in adata.obs.columns:
        raise ValueError(f"`{group_key}` not found in adata.obs columns.")

    cell_counts = adata.obs[group_key].value_counts()
    valid_groups = cell_counts[cell_counts >= min_cells].index
    filtered_groups = cell_counts[cell_counts < min_cells].index

    if len(filtered_groups) > 0:
        print(f"Filtered out groups with less than {min_cells} cells in '{group_key}':")
        print(filtered_groups.tolist())

    adata_filtered = adata[adata.obs[group_key].isin(valid_groups)].copy()
    sc.tl.embedding_density(adata_filtered, basis=basis, groupby=group_key, key_added=key_added)

    # counts and get sorted group order
    group_counts = (
        adata_filtered.obs[group_key]
        .value_counts()
        .sort_values(ascending=False)
    )
    group_order = group_counts.index.tolist()

    return adata_filtered, group_order

# two-method cross scatter plot for aggregated metrics
from adjustText import adjust_text

def plot_cross_scatter(
        agg_df, method_x, method_y, 
        metric='Silhouette', 
        top_n_labels=10, 
        show_NA=False, 
        scatter_color='coolwarm',
        ax=None
        ):
    """
    Cross-method scatter plot for aggregated metrics.
    :param agg_df: DataFrame containing aggregated metrics with columns: ['Method', 'Group', '{metric}_mean', '{metric}_sem', 'Cell_Count']
    :param method_x: Name of the method for x-axis.
    :param method_y: Name of the method for y-axis.
    :param metric: Metric to plot (e.g., 'Silhouette', 'kNN_Purity', 'Dist_to_NTC').
    :param top_n_labels: Number of top deviated points to label.
    :param show_NA: Whether to include 'NA' group in the plot.
    :param scatter_color: Colormap for scatter points.
    :param ax: Matplotlib Axes object to plot on. If None, a new figure and axes will be created.
    """
    # 1. subset data for the two methods
    df_x = agg_df[agg_df['Method'] == method_x].set_index('Group')
    df_y = agg_df[agg_df['Method'] == method_y].set_index('Group')
    
    # 2. intersect index
    common = df_x.index.intersection(df_y.index)
    if not show_NA:
        common = common[common != 'NA']
    df_x = df_x.loc[common]
    df_y = df_y.loc[common]
    
    # 3. Prepare data for plotting
    x = df_x[f'{metric}_mean']
    y = df_y[f'{metric}_mean']
    xerr = df_x[f'{metric}_sem']
    yerr = df_y[f'{metric}_sem']
    xcounts = df_x['Cell_Count']
    ycounts = df_y['Cell_Count']
    mean_counts = (xcounts + ycounts) / 2
    diff_counts = ycounts - xcounts

    # Object-Oriented
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3.3))
    else:
        fig = ax.get_figure()

    # diagonal line
    min_val, max_val = min(x.min(), y.min()), max(x.max(), y.max())
    padding = (max_val - min_val) * 0.1
    lims = [min_val - padding, max_val + padding]
    ax.plot(lims, lims, 'k--', alpha=0.1, zorder=0)
    
    # error bars
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='none', ecolor='grey', alpha=0.7, zorder=1)

    # size normalization and color normalization
    if diff_counts.min() < 0 < diff_counts.max():
        norm = plt.matplotlib.colors.TwoSlopeNorm(vmin=diff_counts.min(), vcenter=0, vmax=diff_counts.max())
    else:
        norm = plt.Normalize(vmin=diff_counts.min(), vmax=diff_counts.max())

    size_min, size_max = 20, 200
    sizes = np.interp(mean_counts, (mean_counts.min(), mean_counts.max()), (size_min, size_max))

    scatter = ax.scatter(x, y, 
                          s=sizes, 
                          c=diff_counts, 
                          cmap=scatter_color, 
                          norm=norm,
                          alpha=0.7, 
                          edgecolor='black', 
                          linewidth=0.5,
                          zorder=2)
    
    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(f'Diff Cell Count ({method_y} - {method_x})', fontsize='medium')
    cbar.ax.tick_params(labelsize='medium')
    
    # Legend
    min_c, max_c = mean_counts.min(), mean_counts.max()
    legend_counts = np.unique(np.percentile(mean_counts, [20, 50, 75, 97]).astype(int))
    legend_sizes = np.interp(legend_counts, (min_c, max_c), (size_min, size_max))
    
    legend_elements = [ax.scatter([], [], s=s, c='gray', alpha=0.7, edgecolor='black', linewidth=0.5, label=f'{c}') 
                       for c, s in zip(legend_counts, legend_sizes)]

    ax.legend(handles=legend_elements, 
               title="Mean counts", 
               fontsize='small', 
               loc='lower right',
               labelspacing=1.3,
               borderpad=1.2, 
               handletextpad=2)
    
    # Label top deviated points
    texts = []
    dist_from_diag = np.abs(x - y) / np.sqrt(2)
    top_n_labels = min(top_n_labels, len(common))
    top_indices = dist_from_diag.nlargest(top_n_labels).index
    
    count_threshold = xcounts.quantile(0.90)
    top_count_indices = xcounts[xcounts >= count_threshold].index
    top_indices = top_indices.union(top_count_indices)
    
    x_threshold_low, x_threshold_high = x.quantile(0.1), x.quantile(0.9)
    extreme_x_indices = x[(x <= x_threshold_low) | (x >= x_threshold_high)].index
    top_indices = top_indices.union(extreme_x_indices)
    
    for idx in top_indices:
        texts.append(ax.text(x[idx], y[idx], idx, fontsize=6, zorder=3, fontweight='bold', ha='center'))
        
    if adjust_text:
        adjust_text(texts, ax=ax, expand=(1.2, 2), arrowprops=dict(arrowstyle='-', color='black', lw=0.5))

    # Summary Text
    summary_text = f"Total points: {len(common)}\nAbove slash: {(y > x).sum()}\nBelow slash: {(x > y).sum()}"
    ax.text(0.03, 0.97, summary_text,
             transform=ax.transAxes,
             fontsize=5,
             verticalalignment='top',
             bbox=dict(boxstyle='round', fc='wheat', alpha=0.3))

    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel(f"{method_x} (Mean ± SEM)", fontsize=7)
    ax.set_ylabel(f"{method_y} (Mean ± SEM)", fontsize=7)
    ax.set_title(f"Metrics: {metric} (each point is a gene or sgRNA)", fontsize=7)
    
    return ax