import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from pySankey.sankey import sankey
import os

class EvaluationPlotter:
    """
    A collection of plotting functions for evaluating assignment after cleaning collisions.
    Each function generates specific plots based on provided DataFrames.
    """

    def __init__(self, 
                 output_dir: str = None,
                 migration_df=None, multiplet_stats_df=None, shift_df=None, shift_matrix=None,
                 dominance_df=None, gini_df=None, 
                 peak_sep_df=None, confidence_df=None,
                 cor_result=None, raw_cohesion_df=None, clean_cohesion_df=None):
        '''
        Initialize the EvaluationPlotter with necessary dataframes.
        Args:
            output_dir: Directory to save the plots.
            migration_df: DataFrame for identity migration analysis.
            multiplet_stats_df: DataFrame for multiplet statistics comparison.
            shift_df: DataFrame for singleton identity consistency.
            shift_matrix: DataFrame for identity swap matrix.
            dominance_df: DataFrame for dominance score distribution.
            gini_df: DataFrame for Gini coefficient comparison.
            peak_sep_df: DataFrame for GMM peak separation analysis.
            confidence_df: DataFrame for posterior probability and certainty analysis.
            corr_result: Dictionary for cross-modality correlation results.
            raw_cohesion_df: DataFrame for raw singlet cohesion analysis.
            clean_cohesion_df: DataFrame for cleaned singlet cohesion analysis.
        '''
        self.output_dir = output_dir
        self.migration_df = migration_df
        self.multiplet_stats_df = multiplet_stats_df
        self.shift_df = shift_df
        self.shift_matrix = shift_matrix
        self.dominance_df = dominance_df
        self.gini_df = gini_df
        self.peak_sep_df = peak_sep_df
        self.confidence_df = confidence_df
        self.corr_result = cor_result
        self.raw_cohesion_df = raw_cohesion_df
        self.clean_cohesion_df = clean_cohesion_df

    def _force_font_on_ax(ax, family="Arial"):
        """
        Force change font family on a matplotlib axis object,
        including title, axis labels, tick labels, text annotations, legend.
        """
        # Title and axis labels
        ax.title.set_fontfamily(family)
        ax.xaxis.label.set_fontfamily(family)
        ax.yaxis.label.set_fontfamily(family)

        # Tick labels (x and y)
        for tick_label in ax.get_xticklabels():
            tick_label.set_fontfamily(family)
        for tick_label in ax.get_yticklabels():
            tick_label.set_fontfamily(family)

        # Text objects added explicitly via ax.text()
        for text in ax.texts:
            text.set_fontfamily(family)

        # Legend texts
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontfamily(family)

    # --- 1. Identity Migration Plots ---
    def plot_identity_sankey(self):
        """
        Plot a Sankey diagram showing cell identity shifts from Raw to Cleaned.

        Args:
            output_dir: Optional directory to save the plot. If None, plot is not saved.
        """

        # pySankey needs long-form DataFrame
        df_long = self.migration_df.sort_values(['Raw', 'Cleaned'], ascending=[False, False])

        # sankey plot donnot support ax object directly
        sankey(
            left=df_long['Raw'],
            right=df_long['Cleaned'],
            aspect=20,
            fontsize=5,
        )

        # get current figure and axis
        ax = plt.gca()
        fig = plt.gcf()
        fig.set_size_inches(4, 3, forward=True)

        ax.set_title("Identity Migration Sankey: Raw to Cleaned")
        EvaluationPlotter._force_font_on_ax(ax, family="Arial")

        if self.output_dir is not None:
            output_dir = self.output_dir
            os.makedirs(output_dir, exist_ok=True)
            output_path1 = os.path.join(output_dir, "1.identity_migration_sankey.png")
            output_path2 = os.path.join(output_dir, "1.identity_migration_sankey.pdf")
            fig.savefig(output_path1, bbox_inches='tight', dpi=300)
            fig.savefig(output_path2, bbox_inches='tight')
            plt.close(fig)
            return output_path1
        else:
            plt.close(fig)
            return fig

    def plot_migration_heatmap(self):
        """
        An alternative to Sankey: A heatmap showing the percentage of 
        cells moving from one category to another.
        """
        # self.migration_df is expected to be in long form with 'Raw' and 'Cleaned' columns
        migration_ct = pd.crosstab(self.migration_df['Raw'], self.migration_df['Cleaned'])
        migration_pct = migration_ct.div(migration_ct.sum(axis=1), axis=0) * 100

        fig, ax = plt.subplots(figsize=(4,3))
        sns.heatmap(
            migration_pct,
            annot=True,
            fmt=".1f",
            cmap="YlGnBu",
            cbar_kws={'label': 'Percentage of Cells (%)'},
            ax=ax,
        )

        ax.set_title("Identity Fate Map: Raw to Clean Change Ratio")
        ax.set_xlabel("Cleaned Status")
        ax.set_ylabel("Raw Status")

        EvaluationPlotter._force_font_on_ax(ax, family="Arial")

        if self.output_dir is not None:
            output_dir = self.output_dir
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            output_path1 = os.path.join(output_dir, "identity_fate_heatmap.png")
            output_path2 = os.path.join(output_dir, "identity_fate_heatmap.pdf")
            fig.savefig(output_path1, bbox_inches='tight', dpi=300)
            fig.savefig(output_path2, bbox_inches='tight')

            plt.close(fig)
            return output_path1
        else:
            plt.close(fig)
            return fig

    def plot_multiplet_comparison(self):
        """
        Stacked or side-by-side bar plot for n_sgrnas distribution.
        Uses self.multiplet_stats_df.
        """
        fig, ax = plt.subplots(figsize=(4, 3))
        self.multiplet_stats_df.plot(
            kind='bar',
            ax=ax,
            color=['#A1CAF1', '#F49AC2']
        )
        ymax = self.multiplet_stats_df.values.max()
        ax.set_ylim(0, ymax * 1.1)

        ax.set_title("Distribution of sgRNAs per Cell (Raw vs Cleaned)")
        ax.set_xlabel("Number of sgRNAs Assigned")
        ax.set_ylabel("Number of Cells")

        for container in ax.containers:
            ax.bar_label(
                container,
                fmt='%d',
                label_type='edge',
                rotation=45,
                padding=1,
            )
        
        EvaluationPlotter._force_font_on_ax(ax, family="Arial")

        plt.tight_layout()

        if self.output_dir is not None:
            output_dir = self.output_dir
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            output_path1 = os.path.join(output_dir, "multiplet_comparison.png")
            output_path2 = os.path.join(output_dir, "multiplet_comparison.pdf")
            fig.savefig(output_path1, bbox_inches='tight', dpi=300)
            fig.savefig(output_path2, bbox_inches='tight')

            plt.close(fig)
            return output_path1
        else:
            plt.close(fig)
            return fig
    
    def plot_singleton_identity_consistency(self):
        """
        Plot identity consistency for singleton cells:
        Visualizing the consistency of Singleton identities enhances robustness to null data and data with little change.
        """
        # use internal data
        shift_df = self.shift_df
        shift_matrix = self.shift_matrix

        # unified style
        sns.set_style("whitegrid")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

        # --- Left: pie chart of consistency ---
        counts = shift_df['is_changed'].value_counts()
        plot_counts = [counts.get(False, 0), counts.get(True, 0)]
        labels = ['Consistent', 'Identity Swapped']
        colors = ['#66b3ff', '#ff9999']

        if sum(plot_counts) > 0:
            ax1.pie(
                plot_counts,
                labels=labels,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                explode=(0, 0.1 if plot_counts[1] > 0 else 0),
                textprops={'fontsize': 8}
            )
        else:
            ax1.text(
                0.5, 0.5,
                "No singleton cells\navailable!",
                ha='center', va='center', fontsize=12, color='gray'
            )
        ax1.set_title("Identity Consistency of Singleton Cells")

        # --- Right: heatmap of identity swaps ---
        if (shift_matrix is not None and not shift_matrix.empty
                and shift_matrix.sum().sum() > 0):
            # top 15 identities with most changes
            n_rows = min(15, len(shift_matrix.index))
            n_cols = min(15, len(shift_matrix.columns))
            top_rows = shift_matrix.sum(axis=1).sort_values(ascending=False).head(n_rows).index
            top_cols = shift_matrix.sum(axis=0).sort_values(ascending=False).head(n_cols).index

            sub_matrix = shift_matrix.loc[top_rows, top_cols]

            if sub_matrix.size > 0 and not sub_matrix.isna().all().all():
                sns.heatmap(
                    sub_matrix,
                    annot=True,
                    fmt='d',
                    cmap='Reds',
                    ax=ax2,
                    cbar_kws={'label': 'Cell Count'}
                )
                ax2.set_title("Identity Swap Matrix (Raw -> Cleaned)")
                ax2.set_xlabel("Cleaned Identity")
                ax2.set_ylabel("Raw Identity")
                plt.setp(
                    ax2.get_xticklabels(),
                    rotation=45,
                    ha="right",
                    rotation_mode="anchor"
                )
                plt.setp(ax2.get_yticklabels(), rotation=45)
            else:
                ax2.text(
                    0.5, 0.5,
                    "Identity swaps are too sparse to plot",
                    ha='center', va='center', color='gray', fontsize=12
                )
                ax2.set_axis_off()
        else:
            ax2.text(
                0.5, 0.5,
                "100% Consistency:\nNo Identity Swaps Detected",
                ha='center', va='center', fontsize=12, color='gray'
            )
            ax2.set_axis_off()

        if self.output_dir is not None:
            output_dir = self.output_dir
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            png_path = os.path.join(output_dir, "singleton_identity_consistency.png")
            pdf_path = os.path.join(output_dir, "singleton_identity_consistency.pdf")
            fig.savefig(png_path, bbox_inches='tight', dpi=300)
            fig.savefig(pdf_path, bbox_inches='tight')

            plt.close(fig)
            return png_path
        else:
            plt.close(fig)
            return fig


    # --- 2. Distribution & Purity Plots ---
    def plot_dominance_ecdf(self):
        """Empirical Cumulative Distribution Function (ECDF) for Dominance Score."""
        # sns.set_style("whitegrid")

        fig, ax = plt.subplots(figsize=(4,3))
        sns.ecdfplot(
            data=self.dominance_df,
            x='Score_Raw',
            label='Raw',
            color='gray',
            linestyle='--',
            ax=ax,
        )
        sns.ecdfplot(
            data=self.dominance_df,
            x='Score_Cleaned',
            label='Cleaned',
            color='red',
            linestyle='-',
            lw=0.5,
            ax=ax,
        )

        ax.set_title("Cumulative Distribution of Dominance Score (Top1/Total)")
        ax.set_xlabel("Dominance Score")
        ax.set_ylabel("Cumulative Probability")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if self.output_dir is not None:
            output_dir = self.output_dir
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            output_path1 = os.path.join(output_dir, "dominance_ecdf.png")
            output_path2 = os.path.join(output_dir, "dominance_ecdf.pdf")
            fig.savefig(output_path1, bbox_inches='tight', dpi=300)
            fig.savefig(output_path2, bbox_inches='tight')

            plt.close(fig)
            return output_path1
        else:
            plt.close(fig)
            return fig

    def plot_gini_improvement(self, cutoff=0.5):
        """
        Visualize Gini coefficient improvement using KDE and Scatter plots.

        Uses:
            self.gini_df with columns ['Gini_Raw', 'Gini_Cleaned'].
        """
        gini_df = self.gini_df
        sns.set_style("whitegrid")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

        # --- 1: KDE Distribution ---
        sns.kdeplot(
            gini_df['Gini_Raw'],
            fill=True,
            label='Raw',
            ax=ax1,
            color='gray'
        )
        sns.kdeplot(
            gini_df['Gini_Cleaned'],
            fill=True,
            label='Cleaned',
            ax=ax1,
            color='skyblue'
        )
        ax1.set_title("Gini Coefficient Distribution")
        ax1.set_xlabel("Gini Index (0=Equality, 1=Pure)")
        ax1.legend()

        # --- 2: Scatter Plot (Raw vs Cleaned) ---
        sample_df = gini_df.sample(n=min(len(gini_df), 8000))  # sample for visibility
        x = sample_df['Gini_Raw']
        y = sample_df['Gini_Cleaned']

        ax2.scatter(
            x,
            y,
            alpha=0.4,
            s=10,
            color='teal'
        )

        # Reference line y=x
        lims = [0, 1]
        ax2.plot(
            lims,
            lims,
            'r--',
            alpha=0.75,
            zorder=0,
            label='y=x (No Change)'
        )

        # Count points above and below y=x, split by y <= cutoff and y > cutoff
        mask_low = y <= cutoff
        mask_high = y > cutoff

        # for y <= cutoff
        n_above_low = ((y > x) & mask_low).sum()
        n_below_low = ((y < x) & mask_low).sum()

        # for y > cutoff
        n_above_high = ((y > x) & mask_high).sum()
        n_below_high = ((y < x) & mask_high).sum()

        total = len(sample_df)

        # y>x annotation
        ax2.text(
            0.03,
            0.90,
            (
                f'y<{cutoff} y>x:\n'
                f'{n_above_low} ({n_above_low/total:.1%})\n'
                f'y>={cutoff} y>x:\n'
                f'{n_above_high} ({n_above_high/total:.1%})'
            ),
            transform=ax2.transAxes,
            fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        # y<x annotation
        ax2.text(
            0.98,
            0.08,
            (
                f'y<{cutoff} y<x:\n'
                f'{n_below_low} ({n_below_low/total:.1%})\n'
                f'y>={cutoff} y<x:\n'
                f'{n_below_high} ({n_below_high/total:.1%})'
            ),
            transform=ax2.transAxes,
            fontsize=8,
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        ax2.set_title("Cell-level Gini Change")
        ax2.set_xlabel("Gini (Raw)")
        ax2.set_ylabel("Gini (Cleaned)")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.legend()

        plt.tight_layout()

        if self.output_dir is not None:
            output_dir = self.output_dir
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            output_path1 = os.path.join(output_dir, "gini_improvement.png")
            output_path2 = os.path.join(output_dir, "gini_improvement.pdf")
            fig.savefig(output_path1, bbox_inches='tight', dpi=300)
            fig.savefig(output_path2, bbox_inches='tight')

            plt.close(fig)
            return output_path1
        else:
            plt.close(fig)
            return fig

    # --- 3. GMM Sharpness Plots ---
    def plot_peak_separation_shift(self):
        """
        Ashman's D improvement scatter plot: raw vs cleaned.

        Uses:
            self.peak_sep_df: EvaluationHelper.compare_peak_separation() output DataFrame.
        """
        peak_sep_df = self.peak_sep_df
        if peak_sep_df is None or peak_sep_df.empty:
            return None

        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(4, 3.5))

        # 1. scatter plot
        sns.scatterplot(
            data=peak_sep_df,
            x='raw_d',
            y='cleaned_d',
            hue='d_improvement',
            palette='vlag',
            alpha=0.7,
            edgecolor=None,
            s=30,
            ax=ax,
        )

        # 2. add reference line y=x
        max_val = max(peak_sep_df['raw_d'].max(), peak_sep_df['cleaned_d'].max())
        min_val = min(peak_sep_df['raw_d'].min(), peak_sep_df['cleaned_d'].min())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='No Change')

        # 3. statistics
        improved_count = (peak_sep_df['d_improvement'] > 0).sum()
        total_count = len(peak_sep_df)
        ax.set_title(f"Ashman's D Value Shift\n({improved_count}/{total_count} features improved)")
        ax.set_xlabel("Raw Separation (D)")
        ax.set_ylabel("Cleaned Separation (D)")
        ax.legend(title="D Improvement", loc='lower right')

        plt.tight_layout()

        if self.output_dir is not None:
            output_dir = self.output_dir
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            png_path = os.path.join(output_dir, "peak_separation_shift.png")
            pdf_path = os.path.join(output_dir, "peak_separation_shift.pdf")
            fig.savefig(png_path, dpi=300, bbox_inches='tight')
            fig.savefig(pdf_path, bbox_inches='tight')

            plt.close(fig)
            return png_path
        else:
            plt.close(fig)
            return fig

    def plot_top_improved_features(self, top_n=20, figsize: tuple = (4, 3.5)):
        """
        [Local Accuracy] Draw the Top N features with the highest resolution improvement (lollipop plot).

        Uses:
            self.peak_sep_df with columns ['feature', 'd_improvement'].
        """
        peak_sep_df = self.peak_sep_df
        if peak_sep_df is None or peak_sep_df.empty:
            return None

        df_plot = peak_sep_df.sort_values('d_improvement', ascending=False).head(top_n).copy()
        if df_plot.empty:
            return None

        fig, ax = plt.subplots(figsize=figsize)
        sns.set_style("whitegrid")

        ax.hlines(
            y=range(len(df_plot)),
            xmin=0,
            xmax=df_plot['d_improvement'],
            color='skyblue',
            alpha=0.8
        )
        ax.scatter(
            df_plot['d_improvement'],
            range(len(df_plot)),
            color='steelblue',
            s=30,
            alpha=1,
            zorder=3
        )

        ax.set_yticks(range(len(df_plot)))
        ax.set_yticklabels(df_plot['feature'])
        ax.set_xlabel("Improvement in Ashman's D (Cleaned - Raw)")
        ax.set_title(f"Top {top_n} Features with Enhanced Signal Separation")
        ax.grid(axis='y', alpha=0.3)

        if self.output_dir is not None:
            output_dir = self.output_dir
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            png_path = os.path.join(output_dir, "top_improved_features.png")
            pdf_path = os.path.join(output_dir, "top_improved_features.pdf")
            fig.savefig(png_path, dpi=300, bbox_inches='tight')
            fig.savefig(pdf_path, bbox_inches='tight')

            plt.close(fig)
            return png_path
        else:
            plt.close(fig)
            return fig

    def plot_ambiguity_ecdf(self, raw_ambiguous_range=(0.1, 0.9)):
        """
        Posterior Probability uncertainty ECDF for ambiguous cells.
        Around 0.5 is ambiguous; both sides towards 0 or 1 is more certain.

        Uses:
            self.confidence_df with 'signal_prob_raw' and 'signal_prob_clean'.
        """
        confidence_df = self.confidence_df
        if confidence_df is None or confidence_df.empty:
            return None

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.set_style("ticks")

        low, high = raw_ambiguous_range
        mask_ambiguous = (
            (confidence_df['signal_prob_raw'] < high)
            & (confidence_df['signal_prob_raw'] > low)
        )
        subset = confidence_df[mask_ambiguous]

        if len(subset) == 0:
            plt.close(fig)
            return None

        sns.ecdfplot(
            data=subset,
            x='signal_prob_raw',
            label='Raw (Ambiguous Subset)',
            color='grey',
            linewidth=2,
            linestyle='--',
            ax=ax
        )
        sns.ecdfplot(
            data=subset,
            x='signal_prob_clean',
            label='Cleaned (Same Cells)',
            color='#E64B35',
            linewidth=2,
            ax=ax
        )

        ax.set_title("Certainty Improvement\n(Feature-Cell Pair Level)")
        ax.set_xlabel("Posterior Probability of Signal")
        ax.set_ylabel("Cumulative Proportion")
        ax.legend()
        ax.grid(alpha=0.3)

        ax.annotate(
            'Positive Certainty',
            xy=(0.85, 0.8),
            xytext=(0.72, 0.6),
            arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.7),
            fontsize=8
        )
        ax.annotate(
            'Negative Certainty',
            xy=(0.12, 0.2),
            xytext=(0, 0.4),
            arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.7),
            fontsize=8
        )

        plt.tight_layout()

        if self.output_dir is not None:
            output_dir = self.output_dir
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            png_path = os.path.join(output_dir, "ambiguity_ecdf.png")
            pdf_path = os.path.join(output_dir, "ambiguity_ecdf.pdf")
            fig.savefig(png_path, dpi=300, bbox_inches='tight')
            fig.savefig(pdf_path, bbox_inches='tight')

            plt.close(fig)
            return png_path
        else:
            plt.close(fig)
            return fig

    def plot_polarization_ecdf(self, raw_ambiguous_range=(0.1, 0.9)):
        """
        Max Posterior Distance (MPD) = |P - 0.5| * 2 mapped to [0, 1].
        0 = uncertain, 1 = certain. ECDF of polarization for ambiguous raw records.

        Uses:
            self.confidence_df with 'signal_prob_raw' and 'signal_prob_clean'.
        """
        confidence_df = self.confidence_df
        if confidence_df is None or confidence_df.empty:
            return None

        df = confidence_df.copy()
        df['polar_raw'] = (df['signal_prob_raw'] - 0.5).abs() * 2
        df['polar_clean'] = (df['signal_prob_clean'] - 0.5).abs() * 2

        low, high = raw_ambiguous_range
        mask = (df['signal_prob_raw'] >= low) & (df['signal_prob_raw'] <= high)
        subset = df[mask]

        if len(subset) == 0:
            return None

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.set_style("whitegrid")

        sns.ecdfplot(
            data=subset,
            x='polar_raw',
            label=f'Raw (Prob {low}-{high})',
            color='gray',
            linestyle='--',
            ax=ax
        )
        sns.ecdfplot(
            data=subset,
            x='polar_clean',
            label='Cleaned (Same Pairs)',
            color='#E64B35',
            linewidth=2,
            ax=ax
        )

        ax.set_title("Certainty Polarization Improvement\n(Feature-Cell Pair Level)")
        ax.set_xlabel("Polarization Score (|Prob - 0.5| * 2)")
        ax.set_ylabel("Cumulative Proportion")
        ax.legend()

        ax.annotate(
            'Shift to Higher Certainty',
            xy=(0.85, 0.8),
            xytext=(0.6, 0.5),
            arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.7),
            fontsize=8
        )

        plt.tight_layout()

        if self.output_dir is not None:
            output_dir = self.output_dir
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            png_path = os.path.join(output_dir, "polarization_ecdf.png")
            pdf_path = os.path.join(output_dir, "polarization_ecdf.pdf")
            fig.savefig(png_path, dpi=300, bbox_inches='tight')
            fig.savefig(pdf_path, bbox_inches='tight')

            plt.close(fig)
            return png_path
        else:
            plt.close(fig)
            return fig

    def plot_certainty_delta_distribution(self):
        """
        Delta Polarization (|P-0.5|*2) Distribution Plot.

        Uses:
            self.confidence_df with 'signal_prob_raw' and 'signal_prob_clean'.
        """
        confidence_df = self.confidence_df
        if confidence_df is None or confidence_df.empty:
            return None

        df = confidence_df.copy()
        df['polar_raw'] = (df['signal_prob_raw'] - 0.5).abs() * 2
        df['polar_clean'] = (df['signal_prob_clean'] - 0.5).abs() * 2
        df['delta_polar'] = df['polar_clean'] - df['polar_raw']

        improved_ratio = (df['delta_polar'] > 0.01).mean() * 100
        decreased_ratio = (df['delta_polar'] < -0.01).mean() * 100
        stable_ratio = 100 - improved_ratio - decreased_ratio

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.set_style("white")

        sns.histplot(
            df['delta_polar'],
            bins=100,
            kde=True,
            color='teal',
            edgecolor='w',
            alpha=0.7,
            ax=ax
        )

        ax.axvline(0, color='red', linestyle='--', linewidth=1, label='No Change')

        stats_text = (
            f"Improved (Δ>0): {improved_ratio:.1f}%\n"
            f"Stable: {stable_ratio:.1f}%\n"
            f"Decreased (Δ<0): {decreased_ratio:.1f}%"
        )

        ax.text(
            0.95,
            0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        ax.set_title("Distribution of Certainty Gain (Δ Polarization)")
        ax.set_xlabel("Δ Polarization Score (Cleaned - Raw)")
        ax.set_ylabel("Count of Feature-Cell Pairs")
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if self.output_dir is not None:
            output_dir = self.output_dir
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            png_path = os.path.join(output_dir, "certainty_delta_distribution.png")
            pdf_path = os.path.join(output_dir, "certainty_delta_distribution.pdf")
            fig.savefig(png_path, dpi=300, bbox_inches='tight')
            fig.savefig(pdf_path, bbox_inches='tight')

            plt.close(fig)
            return png_path
        else:
            plt.close(fig)
            return fig
            
    def plot_probability_migration_density(self):
        """
        Posterior Probability Migration Density Plot.

        Uses:
            self.confidence_df with 'signal_prob_raw' and 'signal_prob_clean'.
        """
        confidence_df = self.confidence_df
        if confidence_df is None or confidence_df.empty:
            return None

        fig, ax = plt.subplots(figsize=(4, 3))

        hb = ax.hexbin(
            y=confidence_df['signal_prob_raw'],
            x=confidence_df['signal_prob_clean'],
            gridsize=50,
            cmap='YlGnBu',
            bins='log',
            mincnt=1
        )
        cb = fig.colorbar(hb, ax=ax, label='log10(count of pairs)')

        ax.axvline(0.5, color='red', linestyle='--', alpha=0.4, label='Decision Boundary (0.5)')
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.4)

        ax.plot([0, 1], [0, 1], color='gray', linestyle=':', alpha=0.8, label='Diagonal (No Change)', lw=1.5)

        ax.set_title("Posterior Probability Migration (Raw vs Cleaned)")
        ax.set_ylabel("Raw Signal Probability")
        ax.set_xlabel("Cleaned Signal Probability")
        ax.legend(loc='lower right')

        plt.tight_layout()

        if self.output_dir is not None:
            output_dir = self.output_dir
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            png_path = os.path.join(output_dir, "probability_migration_density.png")
            pdf_path = os.path.join(output_dir, "probability_migration_density.pdf")
            fig.savefig(png_path, dpi=300, bbox_inches='tight')
            fig.savefig(pdf_path, bbox_inches='tight')

            plt.close(fig)
            return png_path
        else:
            plt.close(fig)
            return fig

    # --- 4. cross_modality ---
    def plot_correlation_shift(self):
        """
        Cross-Modality Library Size Correlation Check.

        Uses:
            self.corr_result: dict from EvaluationHelper.compute_library_correlation().
        """
        cor_result = self.cor_result
        if cor_result is None:
            return None

        data = cor_result['plot_data']
        r_raw = cor_result['raw_r']
        r_clean = cor_result['clean_r']

        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True, sharey=True)

        hex_params = {
            'gridsize': 50,
            'cmap': 'inferno',
            'mincnt': 1,
            'bins': 'log',
        }

        hb1 = axes[0].hexbin(data['gex_log_umi'], data['raw_log_umi'], **hex_params)
        axes[0].set_title(f"Raw: Pearson R = {r_raw:.5f}")
        axes[0].set_xlabel("log(GEX Total UMI + 1)")
        axes[0].set_ylabel("log(CRISPR Raw UMI + 1)")

        hb2 = axes[1].hexbin(data['gex_log_umi'], data['clean_log_umi'], **hex_params)
        axes[1].set_title(f"Cleaned: Pearson R = {r_clean:.5f}")
        axes[1].set_xlabel("log(GEX Total UMI + 1)")
        axes[1].set_ylabel("log(CRISPR Cleaned UMI + 1)")

        cb = fig.colorbar(hb1, ax=axes, orientation='vertical', fraction=0.05, pad=0.04)
        cb.set_label("log10(Count of Cells)")

        plt.suptitle("Cross-Modality Library Size Correlation Check")

        if self.output_dir is not None:
            output_dir = self.output_dir
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            png_path = os.path.join(output_dir, "correlation_shift.png")
            pdf_path = os.path.join(output_dir, "correlation_shift.pdf")
            fig.savefig(png_path, dpi=300, bbox_inches='tight')
            fig.savefig(pdf_path, bbox_inches='tight')

            plt.close(fig)
            return png_path
        else:
            plt.close(fig)
            return fig

