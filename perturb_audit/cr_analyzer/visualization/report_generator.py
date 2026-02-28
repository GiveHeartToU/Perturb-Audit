# report_generator.py (Redesigned for Single File Saving and Complex Layout)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from typing import Callable, List, Dict, Any, Optional, Tuple
import os
import io 

# --- CONSTANTS ---
PAGE_WIDTH_INCH1 = 8.5 # A4/Letter width approximation for PDF
DPI = 300 # Standard resolution for PNG conversion
PAGE_WIDTH_PX = int(PAGE_WIDTH_INCH1 * DPI)

# --- 1 Main Report Generator Function ---
# --- Helper Function for Image1 Combining ---
PAGE_HEIGHT_INCH1 = 16.5 # does not matter much as A4 width is standard

def _create_figure_list1(
    cell_summary_df: pd.DataFrame, 
    gex_feature_impact_df: pd.DataFrame, 
    sgrna_feature_impact_df: pd.DataFrame,
    plot_params: Dict[str, Any]
) -> List[Tuple[str, plt.Figure]]:
    """
    Coordinates the creation of all required Matplotlib Figures.
    
    Args:
        cell_summary_df, gex_feature_impact_df, sgrna_feature_impact_df: Input DataFrames.
        plot_params (Dict): Dictionary containing all required plotting parameters (e.g., top_n, min_umis).

    Returns:
        List[Tuple[str, plt.Figure]]: A list of (page_title, Figure) tuples.
    """
    
    figures = []
    
    # Extract common plotting parameters
    min_umi_sgRNA = plot_params.get('min_umi_sgRNA', 10)
    min_umi_feature = plot_params.get('min_feature_umis', 3)
    top_n_purity = plot_params.get('top_n_purity_features', 30)
    top_n_impact = plot_params.get('top_n_impact_features', 30)
    
    # ----------------------------------------------------------------------
    # A. CBC Level Qualitative Analysis (Scatter Plots)
    # ----------------------------------------------------------------------
    print("Generating Qualitative Scatter Plots...")
    
    # 1. Features GEX vs sgRNA
    fig_feat = plot_dual_metrics_scatter(
        cell_summary_df, 'Total_Features_GEX', 'Total_Features_sgRNA', 
        log_scale=False, figsize=(3.0, 3.0)
    )
    figures.append(("1. CBC - Total Features Scatter", fig_feat))

    # 2. UMIs GEX vs sgRNA
    fig_umi = plot_dual_metrics_scatter(
        cell_summary_df, 'Total_UMIs_GEX', 'Total_UMIs_sgRNA', 
        log_scale=True, figsize=(3.0, 3.0)
    )
    figures.append(("2. CBC - Total UMIs Scatter", fig_umi))

    # 3. Reads GEX vs sgRNA (Assuming Reads column exists and is named consistently)
    fig_read = plot_dual_metrics_scatter(
        cell_summary_df, 'Total_Reads_GEX', 'Total_Reads_sgRNA', 
        log_scale=True, figsize=(3.0, 3.0)
    )
    figures.append(("3. CBC - Total Reads Scatter", fig_read))
    
    # ----------------------------------------------------------------------
    # B. CBC Level Quantitative Analysis (Ratio Boxplots)
    # ----------------------------------------------------------------------
    print("Generating Quantitative CBC Ratio Plots...")
    
    fig_ratios = plot_collision_ratios(
        cell_summary_df, 
        min_UMIs_sgRNA=min_umi_sgRNA,
        figsize=(7.0, 4.0) # Use the standard 2x3 size
    )
    figures.append(("4. CBC - Collision Ratio Distribution", fig_ratios))

    # ----------------------------------------------------------------------
    # C. Feature Level Quantitative Analysis
    # ----------------------------------------------------------------------
    print("Generating Feature Level Plots...")
    
    # 5. GEX Feature Purity Distribution (Boxplot)
    fig_gex_purity = plot_feature_purity_distribution(
        gex_feature_impact_df, 'GEX', 
        top_n_features=top_n_purity, 
        min_feature_umis=min_umi_feature,
        figsize=(7.0, 2.0)
    )
    figures.append((f"5. GEX - Top {top_n_purity} Feature Purity (Ratio Median)", fig_gex_purity))

    # 6. sgRNA Feature Purity Distribution (Boxplot)
    fig_sgrna_purity = plot_feature_purity_distribution(
        sgrna_feature_impact_df, 'sgRNA', 
        top_n_features=top_n_purity, 
        min_feature_umis=min_umi_feature,
        figsize=(7.0, 2.0)
    )
    figures.append((f"6. sgRNA - Top {top_n_purity} Feature Purity (Ratio Median)", fig_sgrna_purity))

    # 7. GEX Feature Total Impact (Barplot)
    fig_gex_impact = plot_top_colliding_features(
        gex_feature_impact_df, 'GEX', 
        top_n_features=top_n_impact, 
        min_feature_umis=min_umi_feature,
        figsize=(7.0, 2.0)
    )
    figures.append((f"7. GEX - Top {top_n_impact} Feature Total UMI Impact", fig_gex_impact))

    # 8. sgRNA Feature Total Impact (Barplot)
    fig_sgrna_impact = plot_top_colliding_features(
        sgrna_feature_impact_df, 'sgRNA', 
        top_n_features=top_n_impact, 
        min_feature_umis=min_umi_feature,
        figsize=(7.0, 2.0)
    )
    figures.append((f"8. sgRNA - Top {top_n_impact} Feature Total UMI Impact", fig_sgrna_impact))

    return figures

def _combine_plots_to_page1(
    image_paths: List[str], 
    output_png_path: str,
    layout_inch: Tuple[float, float] = (PAGE_WIDTH_INCH1, PAGE_HEIGHT_INCH1),
    dpi: int = DPI
) -> None:
    """
    Combines a list of PNG images into a single image based on the required layout.
    
    Layout Requirement:
    - Row 1: Images 1, 2, 3 side-by-side (Horizontal)
    - Subsequent Rows: Remaining images (4, 5, 6, ...) each centered on its own row.
    """
    
    if not image_paths:
        return

    # Convert layout size (inches) to pixel size
    page_width_px = int(layout_inch[0] * dpi)
    page_height_px = int(layout_inch[1] * dpi)
    
    # 1. Load all images
    images = [Image.open(p) for p in image_paths]

    # 2. Prepare the canvas (White background)
    combined_img = Image.new('RGB', (page_width_px, page_height_px), 'white')
    current_y_px = int(0.5 * dpi) # Start with a small margin

    # --- Row 1: Images 1, 2, 3 (Horizontal) ---
    
    if len(images) >= 3:
        img1, img2, img3 = images[0:3]
        remaining_images = images[3:]

        # Calculate heights/widths for the first row
        # We need to scale them down to fit 3 horizontally with margins
        margin_px = int(0.50 * dpi) # 0.5 inch margin on sides
        spacing_px = int(0.0 * dpi) # 0.0 inch spacing between images
        usable_width_r1 = page_width_px - 2 * margin_px - 2 * spacing_px # 4 margins: left, between 1-2, between 2-3, right
        
        # Calculate maximum allowed width for one image in R1
        max_img_width_r1 = usable_width_r1 // 3 
        
        # Resize R1 images to a consistent size (maintaining aspect ratio, fitting max width/height)        
        r1_images_resized = []
        for img in [img1, img2, img3]:
            # Scale height to fit max_img_width_r1
            new_width = max_img_width_r1
            new_height = int(img.height * new_width / img.width)
            r1_images_resized.append(img.resize((new_width, new_height)))
            
        # Paste R1 images
        current_x_px = margin_px
        r1_height = r1_images_resized[0].height # Use the height of the first resized image
        
        for img_r in r1_images_resized:
            # Paste, aligned to the top of the row
            combined_img.paste(img_r, (current_x_px, current_y_px))
            current_x_px += img_r.width + spacing_px
        
        current_y_px += r1_height + int(0.1 * dpi) # Add R1 height and a vertical spacer
    elif len(images) > 0:
        print("Warning: Less than 3 images provided, checking available images!")
        return
    else:
        print("Warning: No images provided for combination, skipping combination.")
        return

    # --- Subsequent Rows: Remaining Images (Centered, Single Row) ---
    
    max_img_width_r2 = page_width_px - 2 * int(0.75 * dpi) 
    
    for img in remaining_images:
        # Check if we run out of vertical space
        if current_y_px > page_height_px - int(0.5 * dpi): # Leave .5 inch bottom margin
            print("Warning: Ran out of space for subsequent plots on the summary page.")
            break
            
        # Scale image to fit the maximum width for R2, maintaining aspect ratio
        scale_factor = max_img_width_r2 / img.width
        
        new_width = max_img_width_r2
        new_height = int(img.height * scale_factor)
        
        # If the resulting image is too tall to fit the remaining space
        if current_y_px + new_height > page_height_px - int(0.5 * dpi):
             print(f"Warning: Last image {img.filename} is too tall to fit.")
             # Re-scale based on remaining height
             remaining_height = page_height_px - int(1.0 * dpi) - current_y_px
             scale_factor = remaining_height / new_height
             new_height = remaining_height
             new_width = int(new_width * scale_factor)
        
        img_r = img.resize((new_width, new_height))
        
        # Paste centered
        center_x = (page_width_px - new_width) // 2
        combined_img.paste(img_r, (center_x, current_y_px))
        
        # Advance Y position for the next row
        current_y_px += new_height + int(0.1 * dpi) # Add image height and a vertical spacer

    # Save the combined image
    combined_img.save(output_png_path, dpi=(dpi, dpi))

# Assuming all plotting functions (plot_dual_metrics_scatter, etc.) 
# and constants (set_publication_style) are accessible, 
# either through import or by being in the same directory.
from .plotter import (
    plot_dual_metrics_scatter,
    plot_collision_ratios,
    plot_feature_purity_distribution,
    plot_top_colliding_features,
    set_publication_style
)

def generate_collision_report1(
    cell_summary_df: pd.DataFrame, 
    gex_feature_impact_df: pd.DataFrame, 
    sgrna_feature_impact_df: pd.DataFrame,
    output_dir: str,
    report_batch_id: str = 'TEST --GHTU',
    report_filename: str = 'Collision_Analysis_Report.pdf',
    plot_params: Dict[str, Any] = None
) -> None:
    
    if plot_params is None:
        plot_params = {}

    print(f"Starting report generation in directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate all Matplotlib figures (Figure 1 to 8)
    figure_list_with_titles = _create_figure_list1(
        cell_summary_df, 
        gex_feature_impact_df, 
        sgrna_feature_impact_df, 
        plot_params
    )
    # IMPORTANT: Ensure fonts are set for vector embedding (done in set_publication_style)
    set_publication_style() 

    # 2. Save individual files (Vector PDF & Raster PNG)
    single_plot_paths = [] # List to track all PNG paths for potential cleanup/combining
    figures_to_combine = [] # List of (title, png_path, fig) tuples for the summary page

    print("\n--- Saving individual plots (PDF) ---")

    for i, (title, fig) in enumerate(figure_list_with_titles):
        base_name = f"fig_{i+1}_{title.replace(' ', '_').replace('-', '_')}"
        pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
        png_path = os.path.join(output_dir, f"{base_name}.png")

        # A. Save as individual Vector PDF (for editing)
        fig.savefig(pdf_path, bbox_inches='tight')
        print(f"  Saved vector PDF: {os.path.basename(pdf_path)}")

        # B. Save as individual PNG (for combination)
        fig.savefig(png_path, dpi=DPI, bbox_inches='tight')
        # print(f"  Saved raster PNG: {os.path.basename(png_path)}") # PNG will be deleted later.
        
        single_plot_paths.append(png_path)
        figures_to_combine.append((title, png_path, fig))
        
        # Close the figure handle to free memory
        plt.close(fig) 
    
    # --- 3. Image Combination (Summary Page Creation) ---
    summary_page_png = os.path.join(output_dir, "Summary_Page_Layout.png")
    
    # Get the PNG paths for the figures that will be combined
    png_paths_to_combine = [item[1] for item in figures_to_combine]
    
    print("\n--- Creating Summary Page1 Layout ---")
    _combine_plots_to_page1(
        png_paths_to_combine, 
        summary_page_png,
        dpi=DPI
    )
    print(f"  Created Summary Page PNG: {os.path.basename(summary_page_png)}")

    # --- 4. Final PDF Output ---
    final_report_path = os.path.join(output_dir, report_filename)

    try:
        with PdfPages(final_report_path) as pdf:
            
            # Page 1: The Combined Summary Page (Raster Image)
            summary_img = Image.open(summary_page_png)
            
            # Convert size back to inches for PdfPages
            w_inch = summary_img.width / DPI
            h_inch = summary_img.height / DPI
            
            # Create a temporary figure to hold the raster image for PDF output
            fig_summary, ax_summary = plt.subplots(figsize=(w_inch, h_inch))
            ax_summary.imshow(summary_img)
            ax_summary.axis('off') # Hide axes
            ax_summary.set_title(f"Collision Analysis Summary ({report_batch_id})", y=0.98, fontsize=9)
            pdf.savefig(fig_summary, bbox_inches='tight')
            plt.close(fig_summary)
            print("  Added Page 1: Summary Layout (Raster)")
            
        print(f"\n✅ Report successfully saved to: {final_report_path}")
        # delete individual PNG files after final report creation
        for png_path in single_plot_paths:
            if os.path.exists(png_path):
                os.remove(png_path)

    except Exception as e:
        print(f"\n❌ Error saving final PDF report: {e}")
        
    finally:
        # Clean up the large temporary PNG files used for combining
        if os.path.exists(summary_page_png):
            # os.remove(summary_page_png) # Uncomment if you want cleanup
            pass
        print("Cleanup done.")

# --- 2 sgRNA Report Generator ---

from .plotter import plot_sgRNA_density, plot_sgRNA_scatter, create_gmm_report_images
import warnings

def create_analysis_report(
    sgRNA_list: List[str],
    plot_func: Callable,
    plot_func_kwargs: Dict[str, Any],
    report_path: str,
    cols: int = 3, # Images per row
    page_width_inch: float = 8.5,
    margin_inch: float = 0.4,
    temp_dir: str = "temp_report_images",
    keep_temp: bool = False
) -> Optional[str]:
    """
    Generates a multi-page PDF report by calling a specified plotter function 
    for each sgRNA in the list and assembling the resulting PNGs via Pillow.
    At present, supports plot_sgRNA_scatter, plot_sgRNA_density, and create_gmm_report_images...
    Args:
        sgRNA_list (List[str]): The ordered list of sgRNA names to plot.
        plot_func (Callable): The single-sgRNA plotting function to call (plot_sgRNA_scatter or plot_sgRNA_density).
        plot_func_kwargs (Dict[str, Any]): Keyword arguments to pass to plot_func (must include sgRNA_matrix etc.).
        report_path (str): The final path and filename for the multi-page PDF report.
        cols (int): Number of images per row in the report layout.
        page_width_inch (float): The width of the final PDF page in inches (default 8.5 for US Letter).
        margin_inch (float): The total margin width (left + right) in inches.
        temp_dir (str): Temporary directory to store PNGs generated during the process.
        keep_temp (bool): Whether to keep the temporary PNG/PDFs after report generation.

    Returns:
        Optional[str]: The path to the generated PDF report, or None on failure.
    """
    
    # 1. Setup and Pre-plotting: Generate all necessary PNGs
    # Create temporary directory
    temp_full_path = os.path.join(os.path.dirname(report_path), temp_dir)
    os.makedirs(temp_full_path, exist_ok=True)

    all_png_paths = []
    
    # Check a critical argument needed for file naming
    if 'output_dir' not in plot_func_kwargs:
        # Use the temporary directory for plotter output
        plot_func_kwargs['output_dir'] = temp_full_path
    
    print(f"--- 1. Generating {len(sgRNA_list)} individual PNGs in {temp_full_path} ---")

    if plot_func in [plot_sgRNA_scatter, plot_sgRNA_density]:
        for sgRNA_name in sgRNA_list:
            try:
                # Since our plot_sgRNA_scatter returns (fig, df), we must handle the return value based on the specific function.
                result = plot_func(sgRNA_name_to_plot=sgRNA_name, **plot_func_kwargs)
                
                if isinstance(result, tuple) and len(result) == 2:
                    # Assuming plot_sgRNA_scatter signature (fig, df)
                    # If the function returns a Figure object, the PNG must have been saved to disk.
                    # We need to calculate the expected PNG path based on the function's internal naming convention.
                    
                    # --- Assuming plotter functions save file and the name follows the pattern: ---
                    log_status = plot_func_kwargs.get('log_transform', True)
                    base_filename = f"{sgRNA_name.replace('/', '_')}_{plot_func.__name__.split('_')[-1]}_log{log_status}"
                    png_path = os.path.join(plot_func_kwargs['output_dir'], f"{base_filename}.png")
                    
                    if os.path.exists(png_path):
                        all_png_paths.append(png_path)
                    else:
                        print(f"Warning: Plot function did not save PNG at expected path: {png_path}")
                
                elif result is not None:
                    # Assuming plot_sgRNA_density returns Figure (or None on skip)
                    log_status = plot_func_kwargs.get('log_transform', True) # Density default is True
                    base_filename = f"{sgRNA_name.replace('/', '_')}_density_log{log_status}"
                    png_path = os.path.join(plot_func_kwargs['output_dir'], f"{base_filename}.png")
                    
                    if os.path.exists(png_path):
                        all_png_paths.append(png_path)
                    else:
                        print(f"Warning: Plot function did not save PNG at expected path: {png_path}")
                    
            except Exception as e:
                print(f"Error plotting {sgRNA_name}: {e}. Skipping.")

    elif plot_func == create_gmm_report_images:
        # Special handling for GMM report images
        try:
            gmm_png_paths = create_gmm_report_images(**plot_func_kwargs)
            all_png_paths.extend(gmm_png_paths)
        except Exception as e:
            print(f"Error generating GMM report images: {e}")

    if not all_png_paths:
        print("Error: No PNGs were successfully generated. Cannot create report.")
        return None

    print(f"Successfully generated {len(all_png_paths)} PNGs.")
    
    # 2. Layout Calculation (Based on fixed width and image aspect ratio)
    print("--- 2. Calculating Layout Parameters ---")

    # Read the first image to determine the fixed aspect ratio (width_px / height_px)
    try:
        sample_img = Image.open(all_png_paths[0])
        img_aspect_ratio = sample_img.width / sample_img.height
        sample_img.close()
    except Exception as e:
        print(f"Error reading the sample image for aspect ratio calculation: {e}")
        return None
    
    # Calculate usable width in inches
    usable_width_inch = page_width_inch - (2 * margin_inch)
    # Example equal internal gap between images
    horizontal_gap_inch = 0.1 
    # Total width used by gaps: (cols - 1) * horizontal_gap_inch
    img_width_inch = (usable_width_inch - (cols - 1) * horizontal_gap_inch) / cols
    # Calculate target height in inches based on aspect ratio
    img_height_inch = img_width_inch / img_aspect_ratio
    
    # --- Convert inches to Pixels (Crucial for Pillow) ---
    # We use a fixed DPI for the final assembly to ensure consistent scale
    ASSEMBLY_DPI = 300 
    
    # Target size of a single image in pixels
    img_width_px = int(img_width_inch * ASSEMBLY_DPI)
    img_height_px = int(img_height_inch * ASSEMBLY_DPI)
    
    # Layout dimensions in pixels
    margin_px = int(margin_inch * ASSEMBLY_DPI)
    horizontal_gap_px = int(horizontal_gap_inch * ASSEMBLY_DPI)
    page_width_px = int(page_width_inch * ASSEMBLY_DPI)
    
    # 3. PDF Integration Loop (using Matplotlib's PdfPages)
    print("--- 3. Assembling PNGs into One-page PDF Report ---")
    
    # We need to decide how many rows per page. Since we need AUTO-ADAPTIVE height:
    rows_per_page = np.ceil(len(all_png_paths) / cols).astype(int)

    # if Split into pages:
    # images_per_page = cols * rows_per_page
    # pages_list = [all_png_paths[i:i + images_per_page] for i in range(0, len(all_png_paths), images_per_page)]
    
    with PdfPages(report_path) as pdf:
        start_idx = 0
        while start_idx < len(all_png_paths):
            
            # Determine images for the current page
            current_page_paths = all_png_paths[start_idx:start_idx + rows_per_page * cols]
            current_page_count = len(current_page_paths)
            
            # Determine actual rows needed for this page
            actual_rows = np.ceil(current_page_count / cols).astype(int)
            
            # Calculate final page height for auto-adaptation
            # Height = Margin(Top) + Rows * ImgHeight + (Rows - 1) * VerticalGap + Margin(Bottom)
            vertical_gap_px = horizontal_gap_px # Use same gap size
            
            page_height_px = (2 * margin_px) + \
                            (actual_rows * img_height_px) + \
                            ((actual_rows - 1) * vertical_gap_px)
            
            page_height_inch = page_height_px / ASSEMBLY_DPI
            
            # Create the Pillow canvas for the current page
            combined_img = Image.new(
                'RGB', 
                (page_width_px, page_height_px), 
                'white'
            )
            
            # --- Paste Loop ---
            for i, png_path in enumerate(current_page_paths):
                row = i // cols
                col = i % cols
                
                # Calculate paste position
                x_pos = margin_px + col * (img_width_px + horizontal_gap_px)
                y_pos = margin_px + row * (img_height_px + vertical_gap_px)
                
                try:
                    img = Image.open(png_path)
                    
                    # Scale image to fit the target width/height
                    img_resized = img.resize((img_width_px, img_height_px), Image.Resampling.LANCZOS)
                    
                    combined_img.paste(img_resized, (x_pos, y_pos))
                    img.close()
                except Exception as e:
                    print(f"Error processing image {png_path}: {e}")
            
            # Use Matplotlib to add the temporary PNG to the PDF
            # This is the cleanest way to use PdfPages for external images
            fig = plt.figure(figsize=(page_width_inch, page_height_inch), dpi=ASSEMBLY_DPI)
            ax = fig.add_axes([0, 0, 1, 1]) # Use full canvas space [left, bottom, width, height]
            img_array = np.array(combined_img)
            ax.imshow(img_array) # Image.open(temp_page_png_path) needs more memory, so we use the array directly
            ax.axis('off') # Hide axes
            ax.set_title(f"sgRNA Analysis Report ({plot_func.__name__})", y=0.97, fontsize=9)
            
            pdf.savefig(fig)
            plt.close(fig)
            combined_img.close()
            del img_array
            
            start_idx += rows_per_page * cols

    print(f"Report successfully saved to {report_path}")
    
    # 4. Cleanup (Optional, but recommended)
    if not keep_temp:
        print(f"Temporary files kept at: {temp_full_path}")
        try:
            import shutil
            shutil.rmtree(temp_full_path)
            print(f"Cleaned up temporary directory: {temp_full_path}")
        except Exception as e:
            print(f"Warning: Could not clean up temp directory. {e}")
        
    return report_path

# --- 3 Exploratory EDA Report Generator ---

def create_dominance_eda_report(
    sgrna_mtx, # 接收 sgrna_mtx_raw
    output_pdf_path: str,
    min_umi_threshold: int = 10,
    margin_inch: float = 0.4, # Border margin in inches
    gap_inch: float = 0.1,    # Gap between subplots in inches
    plot_height_inch: float = 3, # JointPlot height parameter
    detailed_mode: bool = False
):
    """
    Generate sgRNA dominance diagnostic report for 4x3 layout. 
    Assuming the dominant Three sgRNAs per cell.
    Args:
        sgrna_mtx: sgRNA UMI counts sparse matrix (from func "generate_count_matrix").
        output_pdf_path (str): final PDF report path.
        min_umi_threshold (int): minimum UMI threshold for CBCs inclusion in EDA.
        border_inch (int): report border margin in inches.
        gap_inch (float): gap between subplots in inches.
        plot_height_inch (float): height parameter for each joint plot.
    """
    
    print("--- Starting 4x3 EDA Report Generation ---")
    temp_dir = os.path.join(os.path.dirname(output_pdf_path), "temp_eda_images")
    os.makedirs(temp_dir, exist_ok=True)

    COLOR_THEMES = {
    'row1_T1T2': {'marginal_color': '#D52E5F', 'kde_cmap': 'rocket_r'}, # red-redpurple
    'row2_T2T3': {'marginal_color': '#4daf4a', 'kde_cmap': 'mako_r'},   # green-purple
    'row3_T1T3': {'marginal_color': '#ff7f00', 'kde_cmap': 'plasma_r'},     # orange-lightblue
    'row4_Depth': {'marginal_color': '#984ea3', 'kde_cmap': 'viridis_r'},  # purple-yellow
    }

    PLOT_CONFIG = [
        # Row 1: Top 1 vs Top 2
        ('Ratio_2', 'Ratio_1', 'row1_T1T2'),      # Col 1: R1 vs R2 (Classic Doublet)
        ('LogRatio_12', 'Ratio_1', 'row1_T1T2'),  # Col 2: LogR12 vs R1 (KEY: Single/Assignment Threshold)
        ('LogRatio_12', 'Ratio_2', 'row1_T1T2'),  # Col 3: LogR12 vs R2
        
        # Row 2: Top 2 vs Top 3
        ('Ratio_3', 'Ratio_2', 'row2_T2T3'),      # Col 1: R2 vs R3 (Triplet Detection)
        ('LogRatio_23', 'Ratio_2', 'row2_T2T3'),  # Col 2: LogR23 vs R2
        ('LogRatio_23', 'Ratio_3', 'row2_T2T3'),  # Col 3: LogR23 vs R3
        
        # Row 3: Top 1 vs Top 3
        ('Ratio_3', 'Ratio_1', 'row3_T1T3'),      # Col 1: R1 vs R3 
        ('LogRatio_13', 'Ratio_1', 'row3_T1T3'),  # Col 2: LogR13 vs R1
        ('LogRatio_13', 'Ratio_3', 'row3_T1T3'),  # Col 3: LogR13 vs R3
        
        # Row 4: Depth Impact
        ('Total_sgRNA_UMI', 'Ratio_1', 'row4_Depth'), # Col 1: Log UMI vs R1
        ('Total_sgRNA_UMI', 'Ratio_2', 'row4_Depth'), # Col 2: Log UMI vs R2
        ('Total_sgRNA_UMI', 'Ratio_3', 'row4_Depth'),  # Col 3: Log UMI vs R3
    ]

    # 1. subplot width calculation
    border_px = int(margin_inch * DPI) # Convert margin from inches to pixels
    gap_px = int(gap_inch * DPI)       # Convert gap from inches to pixels
    usable_width_px = PAGE_WIDTH_PX - (2 * border_px) - (2 * gap_px) 
    single_plot_width_px = usable_width_px // 3
    
    # 2. generate individual plots
    from ..analysis_core.utils import extract_dominance_data
    eda_df = extract_dominance_data(sgrna_mtx)

    from .plotter import plot_eda_dual_metrics 
    png_paths = []
    rows = 4
    cols = 3
    
    for i, (x_col, y_col, theme_key) in enumerate(PLOT_CONFIG):
        
        theme = COLOR_THEMES[theme_key]
        if detailed_mode:
            print(f"\nGenerating Plot {i+1}/{rows*cols}: {y_col} vs {x_col}, min_umi_threshold={min_umi_threshold}")

        png_path = plot_eda_dual_metrics(
            eda_df=eda_df,
            output_dir=temp_dir,
            X_COL=x_col,
            Y_COL=y_col,
            min_umi_threshold=min_umi_threshold,
            plot_height=plot_height_inch, 
            main_plot_kind='kde', 
            marginal_bins=50,
            marginal_color=theme['marginal_color'],
            kde_cmap=theme['kde_cmap'],
            detailed_mode=detailed_mode,
            # scatter_color default as COLOR_BACKGROUND
        )
        
        if png_path:
            png_paths.append(png_path)
        else:
            print(f"Skipping plot {i+1} due to generation error.")
            
    if len(png_paths) == 0:
        print("Error: No plots were generated. Aborting report creation.")
        return

    # 3. Pillow integration for layout
    images = [Image.open(p) for p in png_paths]

    # resize height calculation based on first image aspect ratio
    first_image = images[0]
    ratio = single_plot_width_px / first_image.width
    single_plot_height_px = int(first_image.height * ratio)

    total_width_px = PAGE_WIDTH_PX
    total_height_px = (rows * single_plot_height_px) + (2 * border_px) + ((rows - 1) * gap_px)

    final_image = Image.new('RGB', (total_width_px, total_height_px), 'white')

    current_y = border_px 

    # paste loop (resize and paste)
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if idx < len(images):
                img_original = images[idx]
                
                # X position calculation
                x_pos = border_px + c * (single_plot_width_px + gap_px)
                
                # Y position calculation
                y_pos = current_y
                
                # resize image
                img_resized = img_original.resize(
                    (single_plot_width_px, single_plot_height_px), 
                    Image.Resampling.LANCZOS
                )
                final_image.paste(img_resized, (x_pos, y_pos))
                
        # move to next row
        current_y += single_plot_height_px + gap_px

    # 4. save final PDF
    temp_png_path = os.path.join(temp_dir, f"EDA_{rows}x{cols}_Layout.png")
    final_image.save(temp_png_path, dpi=(DPI, DPI))
    final_image.close()

    with PdfPages(output_pdf_path) as pdf:
        fig = plt.figure(figsize=(PAGE_WIDTH_INCH1, total_height_px / DPI), dpi=DPI)
        ax = fig.add_axes([0, 0, 1, 1]) # Use full canvas space [left, bottom, width, height]
        ax.imshow(Image.open(temp_png_path))
        ax.axis('off') # Hide axes
        ax.set_title(f"sgRNA Analysis Report ({rows}x{cols} EDA, min UMI={min_umi_threshold})", y=0.97, fontsize=9)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    print(f"--- EDA Report ({rows}x{cols}) successfully saved to: {output_pdf_path} ---")

    # # 5. delete temporary files
    # for p in png_paths:
    #      if os.path.exists(p):
    #         os.remove(p)
    
# --- 4 Evaluation Report Generator ---
# --- Helper Function for Evaluation Report ---

from ..analysis_core.evaluation import EvaluationHelper
from .evaluation_plots import EvaluationPlotter

from PIL import Image
from typing import List, Tuple
import os

PAGE_WIDTH_INCH = 8.5
DPI = 300

MARGIN_INCH = 0.4
SPACING_INCH = 0.2

PAGE_WIDTH_PX = int(PAGE_WIDTH_INCH * DPI)
MARGIN_PX = int(MARGIN_INCH * DPI)
SPACING_PX = int(SPACING_INCH * DPI)

def _create_evaluation_figure_list(
    helper: EvaluationHelper = None,
    plotter: EvaluationPlotter = None,
) -> List[Tuple[str, plt.Figure]]:
    """
    Coordinates the creation of all required Matplotlib Figures for evaluation report.
    
    Args:
        helper: An instance of EvaluationHelper containing evaluation data.
        plotter: An instance of EvaluationPlotter for plotting.
        output_dir (str): Directory to save any intermediate files(PN&PDF) if needed.

    Returns:
        List[Tuple[str, plt.Figure]]: A list of (page_title, Figure) tuples.
    """
    
    figures = []

    if not isinstance(helper, EvaluationHelper):
        print("Warning: Invalid EvaluationHelper instance provided, skipping evaluation plots.")
        return figures
    if not isinstance(plotter, EvaluationPlotter):
        print("Warning: Invalid EvaluationPlotter instance provided, skipping evaluation plots.")
        return figures
    
    print("Helper attributes:", helper.__dict__.keys())
    print(helper.__class__)
    print(helper.__class__.__init__.__code__.co_varnames)

    
    if helper.raw_gmm is None:
        report_mode = 0 # dominance assignment
    else:
        report_mode = 1 # GMM assignment
    
    print("Generating Evaluation Plots...")
    
    set_publication_style() 

    # 1. Migration Plot
    plotter.migration_df = helper.get_sankey_data()
    fig_sankey = plotter.plot_identity_sankey()
    figures.append(("1. Evaluation - Migration Sankey Plot", fig_sankey))

    fig_migration_heatmap = plotter.plot_migration_heatmap()
    figures.append(("2. Evaluation - Migration Heatmap", fig_migration_heatmap))

    plotter.multiplet_stats_df = helper.get_multiplet_stats()
    fig_multiplet_stats = plotter.plot_multiplet_comparison()
    figures.append(("3. Evaluation - Multiplet Stats Comparison", fig_multiplet_stats))

    plotter.shift_df, plotter.shift_matrix, _ = helper.get_singleton_identity_shift()
    fig_identity_shift = plotter.plot_singleton_identity_consistency()
    figures.append(("4. Evaluation - Singleton Identity Shift", fig_identity_shift))

    # 2. Distribution & Purity
    plotter.dominance_df = helper.get_dominance_scores()
    fig_dominance_ecdf = plotter.plot_dominance_ecdf()
    figures.append(("5. Evaluation - Dominance Score ECDF", fig_dominance_ecdf))

    plotter.gini_df = helper.get_gini_improvement()
    fig_gini_improvement = plotter.plot_gini_improvement()
    figures.append(("6. Evaluation - Gini Improvement", fig_gini_improvement))
    
    if report_mode == 1:
        # 3. GMM peak Separation
        plotter.peak_sep_df = helper.compare_peak_separation()
        fig_peak_separation = plotter.plot_peak_separation_shift()
        figures.append(("7. Evaluation - Peak Separation Improvement", fig_peak_separation))

        fig_separation_improve = plotter.plot_top_improved_features(top_n=15)
        figures.append(("8. Evaluation - Top Improved Features", fig_separation_improve))

        # 4. assign confidence
        plotter.confidence_df = helper.compare_assignment_confidence()
        fig_ambiguity_ecdf = plotter.plot_ambiguity_ecdf(raw_ambiguous_range=(0.25, 0.75))
        figures.append(("9. Evaluation - Assignment Ambiguity ECDF", fig_ambiguity_ecdf))
    
        fig_polarization_ecdf = plotter.plot_polarization_ecdf(raw_ambiguous_range=(0.25, 0.75))
        figures.append(("10. Evaluation - Assignment Polarization ECDF", fig_polarization_ecdf))

        fig_certainty_change = plotter.plot_certainty_delta_distribution()
        figures.append(("11. Evaluation - Assignment Certainty Change", fig_certainty_change))

        fig_probability_migration = plotter.plot_probability_migration_density()
        figures.append(("12. Evaluation - Probability Migration Density", fig_probability_migration))

    # 5. cross modality
    plotter.cor_result = helper.compute_library_correlation(gex_mtx=helper.gex_mtx)
    fig_library_correlation = plotter.plot_correlation_shift()
    figures.append(("13. Evaluation - Library Correlation Shift", fig_library_correlation))

    return figures

def _group_image_paths(
    image_paths: List[str],
    report_mode: int
) -> List[List[str]]:
    """
    Returns a list of rows, each row is a list of 1 or 2 image paths.
    Arranges images per row based on report mode.
    """
    print(f"Grouping {len(image_paths)} images for report mode {report_mode}...")
    rows = []
    if report_mode == 1:
        img_rows_order = [(0,1), (2,4), 3, 5, (6,7), (8,9), (10,11), 12,]
        for i in img_rows_order:
            if isinstance(i, tuple):
                rows.append([image_paths[i[0]], image_paths[i[1]]])
            else:
                rows.append([image_paths[i]])
        
    else:
        img_rows_order = [(0,1), (2,4), 3, 5, 6,]
        for i in img_rows_order:
            if isinstance(i, tuple):
                rows.append([image_paths[i[0]], image_paths[i[1]]])
            else:
                rows.append([image_paths[i]])

    return rows

def _compose_pdf_from_pngs(
    png_paths: List[tuple],  # [(title, path), ...]
    output_pdf_path: str,
    report_mode: int,
):
    assert len(png_paths) >= 6

    usable_width1 = PAGE_WIDTH_PX - 2 * MARGIN_PX
    usable_width2 = (PAGE_WIDTH_PX - 2 * MARGIN_PX - SPACING_PX) // 2

    rows = _group_image_paths(png_paths, report_mode)

    # --- 1. calculate resized images and row heights ---
    row_heights = []
    resized_images = []

    for row in rows:
        row_imgs = []
        max_height = 0

        # determine target width
        if len(row) == 2:
            target_width = usable_width2
        else:
            target_width = usable_width1

        for img_title, img_path in row:
            print(f"Processing image for PDF: {img_title}")
            print(f" - Path: {img_path}")

            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            new_h = int(h * target_width / w)
            img_resized = img.resize(
                (target_width, new_h),
                Image.Resampling.LANCZOS
            )

            row_imgs.append(img_resized)
            max_height = max(max_height, new_h)

        resized_images.append(row_imgs)
        row_heights.append(max_height)

    # --- 2. calculate total canvas height ---
    total_height = (
        2 * MARGIN_PX
        + sum(row_heights)
        + SPACING_PX * (len(rows) - 1)
    )

    # --- 3. create canvas ---
    canvas = Image.new(
        "RGB",
        (PAGE_WIDTH_PX, total_height),
        color="white"
    )

    # --- 4. paste pngs onto canvas ---
    y_cursor = MARGIN_PX

    for row_imgs, row_h in zip(resized_images, row_heights):

        if len(row_imgs) == 2:
            x_positions = [
                MARGIN_PX,
                MARGIN_PX + usable_width2 + SPACING_PX
            ]
        else:
            # Center single image
            x_positions = [
                (PAGE_WIDTH_PX - usable_width1) // 2
            ]

        for img, x in zip(row_imgs, x_positions):
            canvas.paste(img, (x, y_cursor))

        y_cursor += row_h + SPACING_PX

    # --- 5. save PDF ---
    canvas.save(output_pdf_path, "PDF", resolution=DPI)

def generate_evaluation_report(
    output_dir: str, 
    helper: EvaluationHelper = None,
    plotter: EvaluationPlotter = None,
    report_filename: str = "Evaluation_Report.pdf"
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    figures = _create_evaluation_figure_list(helper=helper, plotter=plotter)

    report_mode = 0 if helper.raw_gmm is None else 1

    output_pdf_path = os.path.join(output_dir, report_filename)

    _compose_pdf_from_pngs(
        png_paths=figures,
        output_pdf_path=output_pdf_path,
        report_mode=report_mode
    )

    print(f"Evaluation report generated: {output_pdf_path}")
