import os
import yaml
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from pathlib import Path
import traceback

import logging
import warnings

# --- Forcefully reset the Root Logger and undo the impact of basicConfig in the package ---
root = logging.getLogger()
if root.handlers:
    for handler in root.handlers[:]:
        root.removeHandler(handler)
root.setLevel(logging.CRITICAL) # Completely silence the root log unless a fatal crash occurs

# 1. Blocking subset warnings for Matplotlib PDF backends (Matplotlib version: 3.5.3 has this issue)
logging.getLogger('fontTools').setLevel(logging.ERROR)
logging.getLogger('fontTools.subset').setLevel(logging.ERROR) # suppress fontTools subset warnings
logging.getLogger('matplotlib.backends.backend_pdf').setLevel(logging.ERROR)
# logging.getLogger('matplotlib').setLevel(logging.ERROR)
# 2. (meta NOT subset) warnings from 
warnings.filterwarnings("ignore", message=".*meta NOT subset.*")

# ==============================================================================
from .data_io.data_loader import load_molecule_info, load_cellbender_minimal
from .analysis_core.utils import generate_sgRNA_list
from .analysis_core.collision_analyzer import identify_cross_library_collisions
from .analysis_core.collision_quantifier import generate_collision_summary_stats
from .visualization.plotter import get_font_dir, register_fonts_clean, set_publication_style
from .visualization.plotter import plot_sgRNA_density, plot_sgRNA_scatter, create_gmm_report_images
from .visualization.report_generator import generate_collision_report1, create_analysis_report, create_dominance_eda_report, generate_evaluation_report
from .analysis_core.matrix_generator import generate_count_matrix, align_sparse_matrix, generate_count_matrix_from_adata
from .analysis_core.assignment import assign_sgRNA_identity_dominance, GMMAssigner
from .analysis_core.evaluation import EvaluationHelper
from .visualization.evaluation_plots import EvaluationPlotter
from .visualization.transcriptome_eval_plots import plot_adata_qc, filter_and_embedding_density, plot_cross_scatter
from .analysis_core.transcriptome_eval import NTCQualityControl, TranscriptomeEvaluator
# ==============================================================================

class CollisionRunner:
    """
    CR-Collision-Analyzer main control class.
    It coordinates the entire process of data loading, cleaning, allocation, evaluation, and report generation.
    """

    def __init__(self, config_path: str, output_dir: str):
        print(f"Loading configuration from: {config_path}")
        self.config_path = config_path
        
        # 1. Config reading
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)

        # 2. directory initialization
        self._init_directories(output_dir)

        # 3. logger setup
        self.logger = self._setup_logger()
        self.logger.info("CollisionRunner initialized.")

        # 4. State container (holds intermediate data objects)
        self.barcode_df, self.feature_df, self.umi_df = None, None, None
        self.collision_barcodes, self.collisions_df = None, None
        self.cell_summary_df, self.gex_feature_impact_df, self.sgrna_feature_impact_df = None, None, None

        self.datasets = {
            'strict_discard': {'raw_sg': None, 'clean_sg': None, 'clean_sg_aligned': None, 'raw_gex': None, 'clean_gex': None},
            'cellbender': {'raw_sg': None, 'clean_sg': None, 'raw_gex': None, 'clean_gex': None},
        }
        
        # save assignment DataFrames
        self.domin_assignments = {
            'strict_discard': {'raw': None, 'clean': None},
            'cellbender': {'raw': None, 'clean': None}
        } 
        self.gmm_assignments = {
            'strict_discard': {'raw': None, 'clean': None},
            'cellbender': {'raw': None, 'clean': None}
        }

        # save GMMAssigner instances
        self.assigners = {
            'strict_discard': {'raw': None, 'clean': None},
            'cellbender': {'raw': None, 'clean': None}
        }

        # save external GEX data (optional)
        self.external = {'adata_cb': None, 'adata_qc': None}

    def _init_directories(self, output_dir: str):
        """Create a unified output folder structure based on Config."""
        base_dir = output_dir
        # subdirectories
        self.dirs = {
            'root': base_dir,
            '01_collision': os.path.join(base_dir, '01_collision_analysis'),
            '011_summary_reports': os.path.join(base_dir, '01_collision_analysis', 'summary_reports'),
            '02_single_feature': os.path.join(base_dir, '02_single_feature'),
            '03_assignment': os.path.join(base_dir, '03_assignment_analysis'),
            '04_evaluation': os.path.join(base_dir, '04_evaluation_analysis'),
            '05_transcriptomic_evaluation': os.path.join(base_dir, '05_transcriptomic_evaluation'),
            'logs': os.path.join(base_dir, 'logs'),
        }
        
        for path in self.dirs.values():
            os.makedirs(path, exist_ok=True)

    def _setup_logger(self):
        """Logging info setup based on Config."""
        logger = logging.getLogger("CR_Analyzer")
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        logger.handlers = [] # clear existing handlers

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # console Handler only INFO level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # log file Handler full detail bug
        if self.cfg['logging']['save_to_file']:
            log_file = os.path.join(
                self.dirs['logs'], 
                f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            fh = logging.FileHandler(log_file)
            fh.setLevel(getattr(logging, self.cfg['logging']['level'].upper()))
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        return logger

    # --------------------------------------------------------------------------
    # Pipeline Step 1: Data Loading & Collision Summary [Report 1]
    # --------------------------------------------------------------------------
    def run_loading(self, h5_path: str):
        self.logger.info(">>> STEP 1: Reading & Collision Identification")
        
        try:
            # 1. Load molecule_info.h5 data
            self.logger.info(f"Loading data from: {h5_path}")
            self.barcode_df, self.feature_df, self.umi_df = load_molecule_info(h5_path, mode='pass_filter')
            self.logger.info("Data loaded successfully.")

            # 2. collision finding 
            self.logger.info("\nIdentifying cross-library collisions...")
            self.collision_barcodes, self.collisions_df = identify_cross_library_collisions(self.umi_df)
            
            # 3. Report1: Collision Summary
            self.cell_summary_df, self.gex_feature_impact_df, self.sgrna_feature_impact_df = \
                generate_collision_summary_stats(
                        umi_df = self.umi_df,
                        barcode_df = self.barcode_df,
                        feature_df = self.feature_df,
                        collisions_df = self.collisions_df,
                        collision_barcodes = self.collision_barcodes
                )

            self.logger.info("Generating collision summary report...")
            
            self.logger.info(f"Report1 Data Shapes: Cell:{self.cell_summary_df.shape}, GEX:{self.gex_feature_impact_df.shape}, sgRNA:{self.sgrna_feature_impact_df.shape}")
            generate_collision_report1(
                self.cell_summary_df, self.gex_feature_impact_df, self.sgrna_feature_impact_df,
                output_dir=self.dirs['011_summary_reports'],
                report_filename="collision_summary_report.pdf",
                **self.cfg['report1_params'])
            self.logger.info("Collision identification and summary report generation complete.")
        
        except Exception as e:
            self.logger.error(f"Error in cleaning stage: {str(e)}")
            self.logger.error(traceback.format_exc())
        
    # --------------------------------------------------------------------------
    # Pipeline Step 2: Data Cleaning [Report 2 & 3]
    # --------------------------------------------------------------------------

    def run_cleaning(self):
        self.logger.info(">>> STEP 2.1: Collision Cleaning & Matrix Generation")
        
        try:
            self.logger.info("A. Running Internal Strategy: Strict Discard")
            self.datasets['strict_discard']['raw_sg'] = generate_count_matrix(
                umi_df=self.umi_df,
                barcode_df=self.barcode_df,
                feature_df=self.feature_df,
                collisions_df=self.collisions_df,
                library_id=1,
                apply_strict_discard=False,
                **self.cfg['cleaning_mtx']
            )
            self.datasets['strict_discard']['clean_sg'] = generate_count_matrix(
                umi_df=self.umi_df,
                barcode_df=self.barcode_df,
                feature_df=self.feature_df,
                collisions_df=self.collisions_df,
                library_id=1,
                apply_strict_discard=True,
                **self.cfg['cleaning_mtx']
            )
            self.datasets['strict_discard']['clean_sg_aligned'] = align_sparse_matrix(
                target_matrix=self.datasets['strict_discard']['clean_sg'],
                reference_matrix=self.datasets['strict_discard']['raw_sg']
            )
            self.datasets['strict_discard']['raw_gex'] = generate_count_matrix(
                umi_df=self.umi_df,
                barcode_df=self.barcode_df,
                feature_df=self.feature_df,
                collisions_df=self.collisions_df,
                library_id=0,
                apply_strict_discard=False,
                feature_name_col='id',
                min_reads_per_umi=1
            )
            self.datasets['strict_discard']['clean_gex'] = generate_count_matrix(
                umi_df=self.umi_df,
                barcode_df=self.barcode_df,
                feature_df=self.feature_df,
                collisions_df=self.collisions_df,
                library_id=0,
                apply_strict_discard=True,
                feature_name_col='id',
                min_reads_per_umi=1
            )
            self.logger.info(f"A. Strict Discard Matrix generation complete. "
                             f"\nRaw sgRNA matrix shape: {self.datasets['strict_discard']['raw_sg'].shape}, "
                             f"Cleaned sgRNA matrix shape: {self.datasets['strict_discard']['clean_sg'].shape}, "
                             f"Aligned cleaned sgRNA matrix shape: {self.datasets['strict_discard']['clean_sg_aligned'].shape}, "
                                f"Raw GEX matrix shape: {self.datasets['strict_discard']['raw_gex'].shape}, "
                                f"Cleaned GEX matrix shape: {self.datasets['strict_discard']['clean_gex'].shape}.")
            # B. External Strategy: CellBender
            # ------------------------------------------------
            cb_cfg = self.cfg.get('external_inputs', {}).get('cellbender', {})
            if cb_cfg.get('enabled', False):
                cb_path = cb_cfg.get('h5_path')
                raw_feature_bc_matrix_path = cb_cfg.get('raw_h5_path')
                self.logger.info(f"Loading External Strategy: CellBender from {cb_path}")
                self.logger.info(f"Using raw feature_bc_matrix.h5 from {raw_feature_bc_matrix_path}")
                
                if os.path.exists(cb_path) and os.path.exists(raw_feature_bc_matrix_path):
                    try:
                        # from cellbender.remove_background.downstream import load_anndata_from_input_and_output
                        # self.external['adata_cb'] = load_anndata_from_input_and_output(
                        #     input_file=self.cfg['external_inputs']['cellbender']['raw_h5_path'],
                        #     output_file=self.cfg['external_inputs']['cellbender']['h5_path'],
                        #     input_layer_key='raw',  # the layer of raw_feature_bc_matrix.h5 will be named 'raw'
                        # )
                        self.external['adata_cb'] = load_cellbender_minimal(
                            raw_h5=self.cfg['external_inputs']['cellbender']['raw_h5_path'],
                            cb_h5=self.cfg['external_inputs']['cellbender']['h5_path']
                        )

                        # Diff in barcode calls and take intersection for fair comparison
                        self.logger.info(f"Barcodes in cellbender calls: {len(self.external['adata_cb'].obs_names)}, Barcodes out of cellranger calls: {len(self.barcode_df)}")
                        inter_barcodes = set(self.external['adata_cb'].obs_names).intersection(set(self.barcode_df['barcode']))
                        self.logger.info(f"Using intersecting barcodes: {len(inter_barcodes)}")

                        # update strict_discard matrices to use 
                        # the same CBC intersection as the external adata_cb
                        def _subset_mtx_to_intersection(mtx, inter_barcodes_set):
                            """
                            Subset a matrix to the barcodes that are in inter_barcodes_set.

                            Assumes the matrix has an attribute `.barcode_names`
                            which is index‑aligned to its rows.
                            """
                            if mtx is None or not hasattr(mtx, "barcode_names") or not hasattr(mtx, "feature_names"):
                                return mtx

                            keep_idx = [
                                i for i, bc in enumerate(mtx.barcode_names)
                                if bc in inter_barcodes_set
                            ]
                            if not keep_idx:
                                return mtx  # nothing to keep; return as‑is or handle upstream

                            keep_idx = np.array(keep_idx, dtype=int)
                            mtx_sub = mtx[keep_idx, :]
                            mtx_sub.barcode_names = [mtx.barcode_names[i] for i in keep_idx]
                            mtx_sub.feature_names = mtx.feature_names  # features remain unchanged
                            return mtx_sub

                        inter_barcodes_set = set(inter_barcodes)
                        for key in ['raw_sg', 'clean_sg', 'clean_sg_aligned',
                                    'raw_gex', 'clean_gex']:
                            self.datasets['strict_discard'][key] = _subset_mtx_to_intersection(
                                self.datasets['strict_discard'][key],
                                inter_barcodes_set
                            )
                        self.datasets['strict_discard']['clean_sg_aligned'] = align_sparse_matrix(
                            target_matrix=self.datasets['strict_discard']['clean_sg'],
                            reference_matrix=self.datasets['strict_discard']['raw_sg']
                        )
                        self.logger.info(f"A. Strict Discard matrices subsetted to intersecting barcodes with CellBender. "
                                            f"\nRaw sgRNA matrix shape: {self.datasets['strict_discard']['raw_sg'].shape}, "
                                            f"Cleaned sgRNA matrix shape: {self.datasets['strict_discard']['clean_sg'].shape}, "
                                            f"Aligned cleaned sgRNA matrix shape: {self.datasets['strict_discard']['clean_sg_aligned'].shape}, "
                                            f"Raw GEX matrix shape: {self.datasets['strict_discard']['raw_gex'].shape}, "
                                            f"Cleaned GEX matrix shape: {self.datasets['strict_discard']['clean_gex'].shape}.")
                        
                        # update cellbender adata
                        self.external['adata_cb'] = self.external['adata_cb'][self.external['adata_cb'].obs_names.isin(inter_barcodes)].copy()
                        self.datasets['cellbender']['raw_sg'] = generate_count_matrix_from_adata(
                            self.external['adata_cb'], feature_name_col='gene_id', 
                            feature_type='CRISPR Guide Capture', layer='raw')
                        self.datasets['cellbender']['clean_sg'] = generate_count_matrix_from_adata(
                            self.external['adata_cb'], feature_name_col='gene_id', 
                            feature_type='CRISPR Guide Capture', layer='cellbender')
                        self.datasets['cellbender']['raw_gex'] = generate_count_matrix_from_adata(
                            self.external['adata_cb'], feature_name_col='gene_id',
                            feature_type='Gene Expression', layer='raw')
                        self.datasets['cellbender']['clean_gex'] = generate_count_matrix_from_adata(
                            self.external['adata_cb'], feature_name_col='gene_id',
                            feature_type='Gene Expression', layer='cellbender')
                        self.logger.info(f"B. CellBender matrix loaded successfully."
                                            f"\nRaw sgRNA matrix shape: {self.datasets['cellbender']['raw_sg'].shape}, "
                                            f"Cleaned sgRNA matrix shape: {self.datasets['cellbender']['clean_sg'].shape}, "
                                            f"Raw GEX matrix shape: {self.datasets['cellbender']['raw_gex'].shape}, "
                                            f"Cleaned GEX matrix shape: {self.datasets['cellbender']['clean_gex'].shape}."
                                            f"adata_cb: {self.external['adata_cb']}.")
                    except Exception as e:
                        self.logger.error(f"Failed to load CellBender: {e}")
                else:
                    self.logger.warning(f"CellBender enabled but file not found: {cb_path} or {raw_feature_bc_matrix_path}")
            
            # (Optional) Save matrices to disk for future use
            # save_matrix(self.dirs['matrix'], ...) 

        except Exception as e:
            self.logger.error(f"Error in STEP 2.1: cleaning stage: {str(e)}")
            raise e
    
    def single_feature_reports(self, method: str):
        """
        Generate single-feature density and scatter plots for sgRNA raw and given-method cleaned matrices.
        args:
            method: str, either 'strict_discard' or 'cellbender'
        """
        if method not in self.datasets:
            self.logger.error(f"Method {method} not recognized. Available methods: {list(self.datasets.keys())}")
            return
        elif self.datasets[method]['raw_sg'] is None or self.datasets[method]['clean_sg'] is None:
            self.logger.error(f"Raw or cleaned sgRNA matrix for method {method} is None. Please run run_cleaning() first.")
            return
        else:
            self.logger.info(f">>> Generating Single-Feature Reports for method: {method}")
            raw_sg = self.datasets[method]['raw_sg']
            clean_sg = self.datasets[method]['clean_sg']
            single_feature_dir = os.path.join(self.dirs['02_single_feature'], method)
            for sub in ['raw_sgRNA_density', 'cleaned_sgRNA_density', 'sgRNA_scatter']:
                os.makedirs(os.path.join(single_feature_dir, sub), exist_ok=True)

        try:
            self.logger.info("Preparing sgRNA list for single-feature reports...")
            sgRNA_list = generate_sgRNA_list(
                sgRNA_matrix=clean_sg,
                **self.cfg['sgRNA_list']
            )
            self.logger.info(f"sgRNA list generated with {len(sgRNA_list)} features for plotting.")
        except Exception:
            self.logger.error(f"Error generating sgRNA list for single-feature reports:")
            self.logger.error(traceback.format_exc())
            return

        try:
            self.logger.info("Generating single-feature density and scatter plots...")
            
            # Single-feature density plots
            dic_desity_raw = {"sgRNA_matrix": raw_sg}
            dic_desity_raw.update(self.cfg['report2_params']['plot_func_params'])
            create_analysis_report(
                sgRNA_list=sgRNA_list,
                plot_func=plot_sgRNA_density,
                plot_func_kwargs=dic_desity_raw,
                report_path=os.path.join(single_feature_dir, 'raw_sgRNA_density', 'single_feature_density_plots_raw.pdf'),
                cols=self.cfg['report2_params']['cols'],
                keep_temp=self.cfg['report2_params']['keep_temp'],
            )

            dic_desity_clean = {"sgRNA_matrix": clean_sg}
            dic_desity_clean.update(self.cfg['report2_params']['plot_func_params'])
            create_analysis_report(
                sgRNA_list=sgRNA_list,
                plot_func=plot_sgRNA_density,
                plot_func_kwargs=dic_desity_clean,
                report_path=os.path.join(single_feature_dir, 'cleaned_sgRNA_density', 'single_feature_density_plots_cleaned.pdf'),
                cols=self.cfg['report2_params']['cols'],
                keep_temp=self.cfg['report2_params']['keep_temp'],
            )
            self.logger.info("Single-feature density plots generated successfully.")

            # Single-feature scatter plots
            if method == 'strict_discard':
                clean_sg = self.datasets[method]['clean_sg_aligned']
            else:
                clean_sg = self.datasets[method]['clean_sg']

            dic_scatter = {
                "sgrna_mtx_raw": raw_sg,
                "sgrna_mtx_cleaned": clean_sg
            }
            dic_scatter.update(self.cfg['report3_params']['plot_func_params'])
            create_analysis_report(
                sgRNA_list=sgRNA_list,
                plot_func=plot_sgRNA_scatter,
                plot_func_kwargs=dic_scatter,
                report_path=os.path.join(single_feature_dir, 'sgRNA_scatter', 'single_feature_scatter_plots.pdf'),
                cols=self.cfg['report3_params']['cols'],
                keep_temp=self.cfg['report3_params']['keep_temp'],
            )
            self.logger.info("Single-feature reports generated successfully.")

        except Exception: 
            self.logger.error(f"Error in single-feature plotting stage:")
            self.logger.error(traceback.format_exc())

    def run_single_feature_reports(self):
        self.logger.info(">>> STEP 2.2: Generating Single-Feature Reports")
        
        active_strategies = [m for m, d in self.datasets.items() if d['clean_sg'] is not None]
        
        for strategy_name in active_strategies:
            try:
                self.single_feature_reports(method=strategy_name)
            except Exception as e:
                self.logger.error(f"Failed to generate reports for {strategy_name}: {e}")

    # --------------------------------------------------------------------------
    # Pipeline step 3: Assignment [Report 4(1-4) & 5(1,2)]
    # --------------------------------------------------------------------------

    def dominance_assignment(self, method: str):
        """
        Generate Exploratory Data Analysis (EDA) and perform dominance assignment 
        for raw and cleaned matrices using specified method.

        args:
            method: str, either 'strict_discard' or 'cellbender'
        """
        self.logger.info(">>> STEP 3.1: Running Dominance Assignment")
        if method not in self.datasets:
            self.logger.error(f"Method {method} not recognized. Available methods: {list(self.datasets.keys())}")
            return
        elif self.datasets[method]['raw_sg'] is None or self.datasets[method]['clean_sg'] is None:
            self.logger.error(f"Raw or cleaned sgRNA matrix for method {method} is None. Please run run_cleaning() first.")
            return
        else:
            self.logger.info(f"Starting dominance assignment for method: [{method}]")
            raw_sg = self.datasets[method]['raw_sg']
            clean_sg = self.datasets[method]['clean_sg']
            domin_dir = os.path.join(self.dirs['03_assignment'], method)
            for sub in ['031_dominance_eda']:
                os.makedirs(os.path.join(domin_dir, sub, 'raw', 'min_umi_treshold_0'), exist_ok=True)
                os.makedirs(os.path.join(domin_dir, sub, 'raw', f'min_umi_treshold_{self.cfg["report4_params"]["min_umi_threshold"]}'), exist_ok=True)
                os.makedirs(os.path.join(domin_dir, sub, 'cleaned', 'min_umi_treshold_0'), exist_ok=True)
                os.makedirs(os.path.join(domin_dir, sub, 'cleaned', f'min_umi_treshold_{self.cfg["report4_params"]["min_umi_threshold"]}'), exist_ok=True)

        if raw_sg is None or clean_sg is None:
            raise ValueError("Matrices not found. Run run_cleaning() first.")

        # Dominance EDA Report generation and testing dominance assignment on raw matrix
        try:
            create_dominance_eda_report(
                sgrna_mtx=raw_sg,
                min_umi_threshold=0,
                output_pdf_path=os.path.join(domin_dir, '031_dominance_eda', 'raw', 'min_umi_treshold_0', 'dominance_eda_report.pdf'),
            )
            create_dominance_eda_report(
                sgrna_mtx=raw_sg,
                min_umi_threshold=self.cfg['report4_params']['min_umi_threshold'],
                output_pdf_path=os.path.join(domin_dir, '031_dominance_eda', 'raw', f'min_umi_treshold_{self.cfg["report4_params"]["min_umi_threshold"]}', 'dominance_eda_report.pdf'),
            )
            try: 
                self.domin_assignments[method]['raw'] = assign_sgRNA_identity_dominance(
                    sgrna_mtx=raw_sg,
                    save_dir=os.path.join(domin_dir, '032_assign_dominance', 'raw'),
                    **self.cfg['dominance_assignment']
                )
            except Exception:
                self.logger.error(f"Error testing dominance assignment on raw matrix:")
                self.logger.error(traceback.format_exc())
        except Exception:
            self.logger.error(f"Error in dominance EDA stage:")
            self.logger.error(traceback.format_exc())
        
        # Dominance EDA Report generation and testing dominance assignment on cleaned matrix
        try:
            create_dominance_eda_report(
                sgrna_mtx=clean_sg,
                min_umi_threshold=0,
                output_pdf_path=os.path.join(domin_dir, '031_dominance_eda', 'cleaned', 'min_umi_treshold_0', 'dominance_eda_report.pdf'),
            )
            create_dominance_eda_report(
                sgrna_mtx=clean_sg,
                min_umi_threshold=self.cfg['report4_params']['min_umi_threshold'],
                output_pdf_path=os.path.join(domin_dir, '031_dominance_eda', 'cleaned', f'min_umi_treshold_{self.cfg["report4_params"]["min_umi_threshold"]}', 'dominance_eda_report.pdf'),
            )
            try: 
                
                self.domin_assignments[method]['clean'] = assign_sgRNA_identity_dominance(
                    sgrna_mtx=clean_sg,
                    save_dir=os.path.join(domin_dir, '032_assign_dominance', 'cleaned'),
                    **self.cfg['dominance_assignment']
                )
            except Exception:
                self.logger.error(f"Error testing dominance assignment on cleaned matrix:")
                self.logger.error(traceback.format_exc())
        except Exception:
            self.logger.error(f"Error in dominance EDA stage:")
            self.logger.error(traceback.format_exc())
        
        self.logger.info(f"Dominance assignment for method [{method}] complete.")

    def run_dominance_assignment(self):
        active_strategies = [m for m, d in self.datasets.items() if d['clean_sg'] is not None]
        
        for strategy_name in active_strategies:
            try:
                self.dominance_assignment(method=strategy_name)
            except Exception as e:
                self.logger.error(f"Failed to run dominance assignment for {strategy_name}: {e}")

    def gmm_assignment(self, method: str):
        """
        Perform GMM assignment for raw and cleaned matrices using specified method.
        args:
            method: str, either 'strict_discard' or 'cellbender'
        """
        self.logger.info(">>> STEP 3.2: Running GMM Assignment")
        if method not in self.datasets:
            self.logger.error(f"Method {method} not recognized. Available methods: {list(self.datasets.keys())}")
            return
        elif self.datasets[method]['raw_sg'] is None or self.datasets[method]['clean_sg'] is None:
            self.logger.error(f"Raw or cleaned sgRNA matrix for method {method} is None. Please run run_cleaning() first.")
            return
        else:
            self.logger.info(f"Starting GMM assignment for method: [{method}]")
            raw_sg = self.datasets[method]['raw_sg']
            clean_sg = self.datasets[method]['clean_sg']
            gmm_dir = os.path.join(self.dirs['03_assignment'], method, '033_assign_gmm')
            os.makedirs(os.path.join(gmm_dir, 'raw'), exist_ok=True)
            os.makedirs(os.path.join(gmm_dir, 'cleaned'), exist_ok=True)

        if raw_sg is None or clean_sg is None:
            raise ValueError("Matrices not found. Run run_cleaning() first.")
        
        M_method = self.cfg['gmm_assignment']['fit_params1']['global_threshold_method']

        if M_method == 'ntc_baseline':
            NTC_sgRNAs = [x for x in raw_sg.feature_names if self.cfg['non_targeting_control']['pattern'] in x]
            print(f"Using NTC sgRNAs for global thresholding: {NTC_sgRNAs}")
        else:
            NTC_sgRNAs = None
            print("No NTC sgRNAs specified for global thresholding.")

        sgRNA_list = generate_sgRNA_list(
            sgRNA_matrix=clean_sg,
            **self.cfg['sgRNA_list']
        )

        # GMM Assignment on raw matrix
        if method == 'strict_discard':
            self.cfg['gmm_assignment']['fit_params'] = self.cfg['gmm_assignment']['fit_params1']
        elif method == 'cellbender':
            self.cfg['gmm_assignment']['fit_params'] = self.cfg['gmm_assignment']['fit_params2']

        try:
            gmm_assigner_raw = GMMAssigner(**self.cfg['gmm_assignment']['init_params'],)
            gmm_assigner_raw.fit(
                csr_matrix=raw_sg,
                ntc_features=NTC_sgRNAs,
                **self.cfg['gmm_assignment']['fit_params']
            )
            gmm_assigner_raw.generate_report(
                csr_matrix=raw_sg,
                output_dir=os.path.join(gmm_dir, 'raw')
            )
            
            self.assigners[method]['raw'] = gmm_assigner_raw

            create_analysis_report(
                sgRNA_list=sgRNA_list,
                plot_func=create_gmm_report_images,
                plot_func_kwargs={
                    "assigner": gmm_assigner_raw,
                    "csr_matrix": raw_sg,
                    "sgRNA_list": sgRNA_list
                },
                report_path=os.path.join(gmm_dir, 'raw', 'gmm_report_images.pdf'),
                **self.cfg['report5_params']
            )

        except Exception:
            self.logger.error(f"Error in GMM assignment on raw matrix:")
            self.logger.error(traceback.format_exc())
            
        
        # GMM Assignment on cleaned matrix
        try:
            gmm_assigner_clean = GMMAssigner(**self.cfg['gmm_assignment']['init_params'],)
            gmm_assigner_clean.fit(
                csr_matrix=clean_sg,
                ntc_features=NTC_sgRNAs,
                **self.cfg['gmm_assignment']['fit_params']
            )
            gmm_assigner_clean.generate_report(
                csr_matrix=clean_sg,
                output_dir=os.path.join(gmm_dir, 'cleaned')
            )
            self.assigners[method]['clean'] = gmm_assigner_clean

            create_analysis_report(
                sgRNA_list=sgRNA_list,
                plot_func=create_gmm_report_images,
                plot_func_kwargs={
                    "assigner": gmm_assigner_clean,
                    "csr_matrix": clean_sg,
                    "sgRNA_list": sgRNA_list
                },
                report_path=os.path.join(gmm_dir, 'cleaned', 'gmm_report_images.pdf'),
                **self.cfg['report5_params']
            )

        except Exception:
            self.logger.error(f"Error in GMM assignment on cleaned matrix:")
            self.logger.error(traceback.format_exc())
        
        self.logger.info("GMM assignment complete.")

    def run_gmm_assignment(self):
        active_strategies = [m for m, d in self.datasets.items() if d['clean_sg'] is not None]
        
        for strategy_name in active_strategies:
            try:
                self.gmm_assignment(method=strategy_name)
            except Exception as e:
                self.logger.error(f"Failed to run GMM assignment for {strategy_name}: {e}")
    
    # --------------------------------------------------------------------------
    # Pipeline step 4: evaluation [Report 6(1,2)]
    # --------------------------------------------------------------------------

    def evaluation(self, method: str):
        """
        Perform evaluation for dominance and GMM assignments using specified method.
        args:
            method: str, either 'strict_discard' or 'cellbender'
        """
        self.logger.info(">>> STEP 4: Running Evaluation Analysis")
        if method not in self.datasets:
            self.logger.error(f"Method {method} not recognized. Available methods: {list(self.datasets.keys())}")
            return
        elif self.datasets[method]['raw_sg'] is None or self.datasets[method]['clean_sg'] is None:
            self.logger.error(f"Raw or cleaned sgRNA matrix for method {method} is None. Please run run_cleaning() first.")
            return
        else:
            self.logger.info(f"Starting evaluation analysis for method: [{method}]")
            raw_sg = self.datasets[method]['raw_sg']
            clean_sg = self.datasets[method]['clean_sg']
            raw_gex = self.datasets[method]['raw_gex']
            clean_gex = self.datasets[method]['clean_gex']
            eva_dir = os.path.join(self.dirs['04_evaluation'], method)
            for sub in ['041_dominance_evaluation', '042_gmm_evaluation']:
                os.makedirs(os.path.join(eva_dir, sub, 'plots'), exist_ok=True)

        if raw_sg is None or clean_sg is None:
            raise ValueError("Matrices not found. Run run_cleaning() first.")
        

        # Evaluation for Dominance Assignment    
        self.logger.info(">>> STEP 4.1: Running Evaluation for Dominance Assignment")
        try:
            domin_eva = EvaluationHelper(
                raw_mtx=raw_sg,
                cleaned_mtx=clean_sg,
                gex_mtx=raw_gex,
                raw_assign_df=self.domin_assignments[method]['raw'],
                cleaned_assign_df=self.domin_assignments[method]['clean'],
                **self.cfg['evaluation_params']
            )
            domin_eva_plotter = EvaluationPlotter(
                output_dir=os.path.join(eva_dir, '041_dominance_evaluation', 'plots'),
            )
            generate_evaluation_report(
                output_dir=os.path.join(eva_dir, '041_dominance_evaluation'),
                helper=domin_eva,
                plotter=domin_eva_plotter,
                report_filename="dominance_evaluation_report.pdf"
            )
        except Exception:
            self.logger.error(f"Error in dominance evaluation stage:")
            self.logger.error(traceback.format_exc())
        
        # Evaluation for GMM Assignment
        self.logger.info(">>> SETP 4.2: Running Evaluation for GMM Assignment")
        try:
            if self.cfg['strict_or_loose_gmm'] == 'strict':
                self.gmm_assignments[method]['raw'] = pd.read_csv(
                    os.path.join(self.dirs['03_assignment'], method, '033_assign_gmm', 'raw', 'assignment_gmm_strict.csv'))
                self.gmm_assignments[method]['clean'] = pd.read_csv(
                    os.path.join(self.dirs['03_assignment'], method, '033_assign_gmm', 'cleaned', 'assignment_gmm_strict.csv'))
            else:
                self.gmm_assignments[method]['raw'] = pd.read_csv(
                    os.path.join(self.dirs['03_assignment'], method, '033_assign_gmm', 'raw', 'assignment_gmm_loose.csv'))
                self.gmm_assignments[method]['clean'] = pd.read_csv(
                    os.path.join(self.dirs['03_assignment'], method, '033_assign_gmm', 'cleaned', 'assignment_gmm_loose.csv'))

            gmm_eva = EvaluationHelper(
                raw_mtx=raw_sg,
                cleaned_mtx=clean_sg,
                gex_mtx=raw_gex,
                raw_assign_df=self.gmm_assignments[method]['raw'],
                cleaned_assign_df=self.gmm_assignments[method]['clean'],
                raw_gmm=self.assigners[method]['raw'],
                cleaned_gmm=self.assigners[method]['clean'],
                **self.cfg['evaluation_params']
            )
            gmm_eva_plotter = EvaluationPlotter(
                output_dir=os.path.join(eva_dir, '042_gmm_evaluation', 'plots'),
            )
            generate_evaluation_report(
                output_dir=os.path.join(eva_dir, '042_gmm_evaluation'),
                helper=gmm_eva,
                plotter=gmm_eva_plotter,
                report_filename="gmm_evaluation_report.pdf"
            )
        except Exception:
            self.logger.error(f"Error in GMM evaluation stage:")
            self.logger.error(traceback.format_exc())

        self.logger.info(f"Evaluation analysis for method [{method}] complete.")

    def run_evaluation(self):
        active_strategies = [m for m, d in self.datasets.items() if d['clean_sg'] is not None]
        
        for strategy_name in active_strategies:
            try:
                self.evaluation(method=strategy_name)
            except Exception as e:
                self.logger.error(f"Failed to run evaluation for {strategy_name}: {e}")

    # --------------------------------------------------------------------------
    # Pipeline step 5: Transcriptomic Impact Evaluation
    # --------------------------------------------------------------------------
    def adata_preprocessing(self):
        """
        Preprocess external adata for QC and assignment mapping.
        """
        self.logger.info(">>> STEP 5: Preprocessing external adata for QC and assignment mapping...")
        adata = self.external['adata_cb']
        BATCH_ID = self.cfg['report1_params']['report_batch_id']
        if adata is None:
            self.logger.warning("No external adata_cb found, skip adata preprocessing.")
            return
        
        try:
            adata.obs['batch_group'] = BATCH_ID
            adata.obs['barcode'] = adata.obs.index.astype(str)
            adata.obs['batch_barcode'] = adata.obs['batch_group'] + '-' + adata.obs['barcode']
            adata.obs.set_index('batch_barcode', inplace=True, drop=True)
            adata = adata[:, adata.var['feature_type'] == 'Gene Expression']
            var_names_series = pd.Series(adata.var_names)
            dup_var = var_names_series[var_names_series.duplicated(keep=False)].unique()
            self.logger.info(f"Found {len(dup_var)} duplicated gene names in adata.var_names \n{dup_var}, Making var_names unique by appending '--' and a count.")
            adata.var_names_make_unique(join='--')

            if "genome" in adata.var.columns:
                if adata.var["genome"].str.contains("GRCh|hg", case=False, na=False).any():
                    organism = "human"
                elif adata.var["genome"].str.contains("GRCm|mm", case=False, na=False).any():
                    organism = "mouse"
                else:
                    organism = "unknown"
            elif "gene_id" in adata.var.columns:
                if adata.var["gene_id"].str.startswith("ENSG").any():
                    organism = "human"
                elif adata.var["gene_id"].str.startswith("ENSMUSG").any():
                    organism = "mouse"
                else:
                    organism = "unknown"
            else:
                organism = "unknown"
            self.logger.info(f"Inferred organism for adata: {organism}")
            self.logger.info(f"External adata preprocessing successful. Final shape: {adata.shape}, organism: {organism}.")
            if organism == "human":
                adata.var["mt"] = adata.var_names.str.startswith("MT-")
            elif organism == "mouse":
                adata.var["mt"] = adata.var_names.str.startswith("mt-")
            else: # Robust human + mouse
                adata.var["mt"] = (
                    adata.var_names.str.startswith("mt-") |
                    adata.var_names.str.startswith("MT-")
                )
            self.logger.info(f"MT genes identified: {adata.var['mt'].sum()}. \n{adata.var_names[adata.var['mt']].tolist()}")

            sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
            self.external['adata_cb'] = adata
            self.logger.info("External adata preprocessing complete.")
        except Exception:
            self.logger.error(f"Error in external adata preprocessing stage:")
            self.logger.error(traceback.format_exc())
    
    def adata_qc(self):
        """
        Transcriptomic QC for external adata (CellBender).
        """
        self.logger.info(">>> STEP 5.1: Generating QC plots for transcriptomic data (if available)...")
        adata = self.external['adata_cb']
        if adata is None:
            self.logger.warning("No external adata_cb found, skip adata_qc.")
            return

        # ---- read thresholds from config with defaults ----
        qc_cfg = self.cfg.get('transcriptomic_qc', {})
        umi_pct_upper = qc_cfg.get('umi_pct_upper', 95)      # percentile for UMI upper bound
        gene_pct_lower = qc_cfg.get('gene_pct_lower', 5)     # percentile for n_genes lower bound
        mt_upper = qc_cfg.get('mt_upper', 10.0)              # absolute MT% upper bound

        output_dir = os.path.join(self.dirs['05_transcriptomic_evaluation'], 'plots')
        os.makedirs(output_dir, exist_ok=True)

        try:
            print(f"Before filtering: {adata.n_obs} cells, {adata.n_vars} genes")
            plot_adata_qc(adata=adata, output_dir=output_dir, prefix="01before_")
            umi_upper = np.percentile(adata.obs['total_counts'], umi_pct_upper).astype(int)
            gene_lower = np.percentile(adata.obs['n_genes_by_counts'], gene_pct_lower).astype(int)

            adata = adata[
                (adata.obs['total_counts'] <= umi_upper) &
                (adata.obs['n_genes_by_counts'] >= gene_lower) &
                (adata.obs['pct_counts_mt'] <= mt_upper),
                :
            ].copy()

            print(
                f"Filtering thresholds (from config if provided): "
                f"UMI <= {umi_upper} (p{umi_pct_upper}), "
                f"Genes >= {gene_lower} (p{gene_pct_lower}), "
                f"MT% <= {mt_upper}%"
            )
            print(f"After filtering: {adata.n_obs} cells, {adata.n_vars} genes")
            plot_adata_qc(adata=adata, output_dir=output_dir, prefix="02after_")
            return adata
        
        except Exception:
            self.logger.error("Error in adata QC stage:")
            self.logger.error(traceback.format_exc())

    def adata_assign_mapping(self):
        """
        Map assignments to adata.obs for downstream transcriptomic evaluation.
        """
        self.logger.info(">>> STEP 5.2: Mapping sgRNA assignments to adata (if available)...")
        BATCH_ID = self.cfg['report1_params']['report_batch_id']
        sgRNA_linker = self.cfg['sgRNA_linker']  # e.g. '-'
        adata = self.external['adata_qc']
        if adata is None:
            self.logger.warning("No external adata_cb found, running adata_qc did not produce an adata object, skip assignment mapping.")
            return
        
        # If both dominance and GMM assignments are completely missing, skip
        if any(df is None for method in self.domin_assignments.values() for df in method.values()) and \
           any(df is None for method in self.gmm_assignments.values() for df in method.values()):
            self.logger.warning("No dominance or GMM assignments found, skipping assignment mapping.")
            return

        try:
            methods_cols = {}

            for method_name, states in self.domin_assignments.items():
                for state, df in states.items():
                    if df is not None:
                        df['batch'] = BATCH_ID
                        df['batch_barcode'] = df['batch'] + '-' + df['barcode'].astype(str)
                        df.set_index('batch_barcode', inplace=True, drop=True)
                        df = df[df['n_sgrnas'] > 0].copy() # Only keep cells with at least 1 sgRNA for mapping
                        df['singlet_sgRNA'] = df['sgrnas'].apply(lambda x: x if '|' not in x else 'NA')
                        df['singlet_gene'] = df['singlet_sgRNA'].apply(lambda x: x.split(sgRNA_linker)[0] if x != 'NA' else 'NA')
                        if any(df['singlet_gene'] == 'NA'):
                            self.logger.warning(f"Check the format of sgRNA names(config wants '{sgRNA_linker}' exits) in the assignment dataframe.")
                        print(f"Dominance assignment: {method_name} - {state}, single:{sum(df['singlet_sgRNA']!='NA')}")

                        sgRNA_col_name = f"Domin_{method_name}_{state}"
                        single_sgRNA_col_name = f"sin_sg_Domin_{method_name}_{state}"
                        single_gene_col_name = f"sin_gene_Domin_{method_name}_{state}"
                        NsgRNA_col_name = f"n_sgrnas_Domin_{method_name}_{state}"
                        methods_cols.update({sgRNA_col_name: (NsgRNA_col_name, single_sgRNA_col_name, single_gene_col_name)})

                        merged_df = df[['sgrnas', 'singlet_sgRNA', 'singlet_gene', 'n_sgrnas']].rename(
                            columns={
                                'n_sgrnas': NsgRNA_col_name,
                                'sgrnas': sgRNA_col_name,
                                'singlet_sgRNA': single_sgRNA_col_name,
                                'singlet_gene': single_gene_col_name
                            }
                        )
                        adata.obs = adata.obs.join(
                            merged_df,
                            how='left'
                        ).fillna({NsgRNA_col_name: 0, sgRNA_col_name: 'NA', single_sgRNA_col_name: 'NA', single_gene_col_name: 'NA'})
                        adata.obs[NsgRNA_col_name] = adata.obs[NsgRNA_col_name].astype(int)

            for method_name, states in self.gmm_assignments.items():
                for state, df in states.items():
                    if df is not None:
                        df['batch'] = BATCH_ID
                        df['batch_barcode'] = df['batch'] + '-' + df['barcode'].astype(str)
                        df.set_index('batch_barcode', inplace=True, drop=True)
                        df = df[df['n_sgrnas'] > 0].copy() # Only keep cells with at least 1 sgRNA for mapping
                        df['singlet_sgRNA'] = df['sgrnas'].apply(lambda x: x if '|' not in x else 'NA')
                        df['singlet_gene'] = df['singlet_sgRNA'].apply(lambda x: x.split(sgRNA_linker)[0] if x != 'NA' else 'NA')
                        if any(df['singlet_gene'] == 'NA'):
                            self.logger.warning(f"Check the format of sgRNA names(config wants '{sgRNA_linker}' exits) in the assignment dataframe.")
                        print(f"GMM assignment: {method_name} - {state}, single:{sum(df['singlet_sgRNA']!='NA')}")

                        sgRNA_col_name = f"GMM_{method_name}_{state}"
                        single_sgRNA_col_name = f"sin_sg_GMM_{method_name}_{state}"
                        single_gene_col_name = f"sin_gene_GMM_{method_name}_{state}"
                        NsgRNA_col_name = f"n_sgrnas_GMM_{method_name}_{state}"
                        methods_cols.update({sgRNA_col_name: (NsgRNA_col_name, single_sgRNA_col_name, single_gene_col_name)})
                        merged_df = df[['sgrnas', 'singlet_sgRNA', 'singlet_gene', 'n_sgrnas']].rename(
                            columns={
                                'n_sgrnas': NsgRNA_col_name,
                                'sgrnas': sgRNA_col_name,
                                'singlet_sgRNA': single_sgRNA_col_name,
                                'singlet_gene': single_gene_col_name
                            }
                        )
                        adata.obs = adata.obs.join(
                            merged_df,
                            how='left'
                        ).fillna({NsgRNA_col_name: 0, sgRNA_col_name: 'NA', single_sgRNA_col_name: 'NA', single_gene_col_name: 'NA'})
                        adata.obs[NsgRNA_col_name] = adata.obs[NsgRNA_col_name].astype(int)
            # After mapping all assignments, save the updated adata back to external for downstream use
            self.external['adata_cb'] = adata
            self.methods_cols = methods_cols

        except Exception:
            self.logger.error("Error in assignment mapping stage:")
            self.logger.error(traceback.format_exc())

    def adata_dimred_umap_density(self):
        """
        Perform dimensionality reduction and clustering for transcriptomic evaluation.
        Generate UMAP plots and save as both PDF and PNG.
        Generate Leiden clusters at specified resolutions and include in UMAP coloring.
        Generate UMAP density plots for specific features if specified in config (optional, can be implemented later).
        """
        self.logger.info(">>> STEP 5.3: Running Dimensionality Reduction and Clustering")
        adata = self.external.get('adata_qc', None)
        if adata is None:
            self.logger.warning("No adata_qc found, skip dimreduction_clustering.")
            return

        # ---- read config with defaults ----
        dr_cfg = self.cfg.get('transcriptomic_dr', {})
        use_rep = dr_cfg.get('use_rep', 'cellbender_embedding')
        n_neighbors = dr_cfg.get('n_neighbors', 30)
        n_pcs = dr_cfg.get('n_pcs', 30)
        n_pcs = min(n_pcs, adata.obsm[use_rep].shape[1] if use_rep in adata.obsm_keys() else adata.shape[1])
        umap_min_dist = dr_cfg.get('umap_min_dist', 0.5)
        umap_spread = dr_cfg.get('umap_spread', 1.0)
        leiden_resolutions = dr_cfg.get('leiden_resolutions', [0.3,0.5,0.8])
        leiden_key_prefix = dr_cfg.get('leiden_key_prefix', 'leiden_cb_res')
        figsize = tuple(dr_cfg.get('figsize', [3, 3]))
        dpi = dr_cfg.get('dpi', 100)
        dpi_save = dr_cfg.get('dpi_save', 150)
        fontsize = dr_cfg.get('fontsize', 5)
        colors_cfg = dr_cfg.get('colors', None)
        pdf_name = dr_cfg.get('pdf_name', '03umap_leiden_clusters.pdf')
        png_name = dr_cfg.get('png_name', '03umap_leiden_clusters.png')

        out_dir = os.path.join(self.dirs['05_transcriptomic_evaluation'], 'plots')
        os.makedirs(out_dir, exist_ok=True)

        # dimensionality reduction, clustering, and plotting
        try:
            sc.set_figure_params(scanpy=True, dpi=dpi, dpi_save=dpi_save,
                                 fontsize=fontsize, figsize=figsize)

            # ---- neighbors / embedding selection ----
            if use_rep == 'cellbender_embedding':
                if 'cellbender_embedding' not in adata.obsm_keys():
                    self.logger.warning(
                        "cellbender_embedding not found in adata.obsm, "
                        "fall back to PCA (X_pca)."
                    )
                    return
                else:
                    sc.pp.neighbors(adata, use_rep='cellbender_embedding',
                                    n_neighbors=n_neighbors, n_pcs=n_pcs)
            else:
                self.logger.warning("Other embedding specified in config, but currently only 'cellbender_embedding' is supported. ")

            sc.tl.umap(adata, min_dist=umap_min_dist, spread=umap_spread)

            # ---- Leiden clustering for each resolution ----
            leiden_keys = []
            for res in leiden_resolutions:
                key = f"{leiden_key_prefix}{res}"
                sc.tl.leiden(adata, key_added=key, resolution=res)
                leiden_keys.append(key)

            # ---- which colors to plot ----
            if colors_cfg is None:
                colors = leiden_keys
            else:
                # ensure list
                colors = list(colors_cfg)

            # PDF
            pdf_path = os.path.join(out_dir, pdf_name)
            sc.pl.umap(
                adata,
                color=colors,
                frameon=False,
                legend_loc='on data',
                show=False,
                save=None,
            )
            plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
            plt.close()

            # PNG
            png_path = os.path.join(out_dir, png_name)
            sc.pl.umap(
                adata,
                color=colors,
                frameon=False,
                legend_loc='on data',
                show=False,
                save=None,
            )
            plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"UMAP clustering plots saved to {pdf_path} and {png_path}")
            self.external['adata_cb'] = adata  
        except Exception:
            self.logger.error("Error in dimreduction_clustering stage:")
            self.logger.error(traceback.format_exc())

        # density plots for specific features can be implemented here
        density_config = self.cfg.get('density_params', None)
        group_level = density_config.get('group_level', None)
        assign_method = density_config.get('assign_method', None)
        denoise_method = density_config.get('denoise_method', None)
        min_cells_per_level = density_config.get('min_cells_per_level', 3)
        denoised = density_config.get('denoised', True)
        if denoised:
            group_col = f"sin_{group_level}_{assign_method}_{denoise_method}_clean"
        else:
            group_col = f"sin_{group_level}_{assign_method}_{denoise_method}_raw"

        try:
            adata_density, density_order = filter_and_embedding_density(adata, group_key=group_col, min_cells=min_cells_per_level, basis="umap")
            fig = sc.pl.embedding_density(
                adata_density,
                groupby=group_col,
                group=density_order,
                ncols=8,
                color_map='YlOrRd', # 'viridis',
                show=False,
                return_fig=True
            )

            # iterate through the axes and find the colorbar axes.
            for i, ax in enumerate(fig.axes): # number of axes per subplot is 2
                if i % 2 == 0: # This is a plot axis
                    ax.title.set_fontsize(14)
                    ax.title.set_fontweight('bold')
                    ax.set_xlabel('')
                    ax.set_ylabel('')
                else: # This is a colorbar axis
                    ax.tick_params(labelsize=10)

            density_pdf_path = os.path.join(out_dir, f"04umap_density_{group_col}.pdf")
            density_png_path = os.path.join(out_dir, f"04umap_density_{group_col}.png")
            fig.savefig(density_pdf_path, format='pdf', bbox_inches='tight')
            fig.savefig(density_png_path, format='png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            self.logger.info(f"UMAP density plots saved to {density_pdf_path} and {density_png_path}")
        except Exception:
            self.logger.error("Error in UMAP density plotting stage:")
            self.logger.error(traceback.format_exc())

    def negtive_control_evaluation(self):
        """
        Perform negative control evaluation for transcriptomic impact analysis.
        This includes checking the distribution of non-targeting control sgRNAs
        in the embedding space and summarizing trusted NTCs.
        """
        self.logger.info(">>> STEP 5.4: Running Negative Control Evaluation (if applicable)")
        adata = self.external.get('adata_qc', None)
        if adata is None:
            self.logger.warning("No adata_qc found, skip negative control evaluation.")
            return

        if not hasattr(self, "methods_cols") or not self.methods_cols:
            self.logger.warning("methods_cols is empty, skip negative control evaluation.")
            return

        NTC_PATTERN = self.cfg['non_targeting_control']['pattern']
        BATCH = self.cfg['report1_params']['report_batch_id']
        out_dir = os.path.join(self.dirs['05_transcriptomic_evaluation'], 'plots')
        os.makedirs(out_dir, exist_ok=True)

        try:
            self.logger.info(f"Starting NTC Quality Control with pattern: {NTC_PATTERN}")
            ntc_qc = NTCQualityControl(
                adata,
                embedding_key='cellbender_embedding',
                ntc_label=NTC_PATTERN
            )

            # run QC for each method
            for method_name, cols in self.methods_cols.items():
                sin_sg_col = cols[1]  # (NsgRNA_col, sin_sg_col, sin_gene_col)
                ntc_qc.evaluate_method(
                    method_name=method_name,
                    sin_sg_col=sin_sg_col,
                    min_total_cells=30,
                    min_sg_cells=5
                )

            # consensus NTC
            self.final_trusted_ntc = ntc_qc.get_consensus_ntc(threshold_ratio=0.9)

            pdf_path = os.path.join(out_dir, "05ntc_qc_summary.pdf")
            png_path = os.path.join(out_dir, "05ntc_qc_summary.png")

            fig = plt.figure(figsize=(8, 6))
            gs = fig.add_gridspec(2, 3)
            ax1 = fig.add_subplot(gs[0, 0])
            ntc_qc.plot_pairwise_heatmap('Domin_cellbender_clean', ax=ax1) # Representative method: CellBender Clean
            ax2 = fig.add_subplot(gs[0, 1:])
            ntc_qc.plot_trust_matrix(ax=ax2)
            ax3 = fig.add_subplot(gs[1, :])
            ntc_qc.plot_shift_lollipop(ax=ax3)

            fig.suptitle(
                f"NTC Quality Control Analysis ({BATCH})",
                fontsize=8,
                weight='bold',
                y=1.02
            )
            plt.tight_layout()
            fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
            fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
            self.logger.info(f"NTC QC plots saved to {pdf_path} and {png_path}")
            plt.close(fig)
        except Exception:
            self.logger.error("Error in negative control evaluation stage:")
            self.logger.error(traceback.format_exc())
        
        # Umap density of all NTC
        assign_method = self.cfg['density_params']['assign_method']
        denoise_method = self.cfg['density_params']['denoise_method']
        denoised = self.cfg['density_params']['denoised']
        min_cells_per_level = self.cfg['density_params']['min_cells_per_level']
        ntc_col = f'sin_sg_{assign_method}_{denoise_method}_{"clean" if denoised else "raw"}'
        try:
            ntc_col = ntc_col  # Representative method: CellBender Clean
            adata_ntc_density, ntc_density_order = filter_and_embedding_density(
                adata,
                group_key=ntc_col,
                min_cells=min_cells_per_level,
                basis="umap"
            )
            ntc_sgRNA = [sg for sg in ntc_density_order if NTC_PATTERN in sg]
            fig = sc.pl.embedding_density(
                adata_ntc_density,
                groupby=ntc_col,
                group=ntc_sgRNA,
                ncols=5,
                color_map='YlOrRd', # 'viridis',
                show=False,
                return_fig=True
            )

            for i, ax in enumerate(fig.axes): # number of axes per subplot is 2
                if i % 2 == 0: # This is a plot axis
                    ax.title.set_fontsize(14)
                    ax.title.set_fontweight('bold')
                    ax.set_xlabel('')
                    ax.set_ylabel('')
                else: # This is a colorbar axis
                    ax.tick_params(labelsize=10)
            ntc_density_pdf_path = os.path.join(out_dir, f"06umap_density_ntc_{ntc_col}.pdf")
            ntc_density_png_path = os.path.join(out_dir, f"06umap_density_ntc_{ntc_col}.png")
            fig.savefig(ntc_density_pdf_path, format='pdf', bbox_inches='tight')
            fig.savefig(ntc_density_png_path, format='png', dpi=300, bbox_inches='tight')
            self.logger.info(f"NTC UMAP density plots saved to {ntc_density_pdf_path} and {ntc_density_png_path}")
            plt.close(fig)
        except Exception:
            self.logger.error("Error in NTC UMAP density plotting stage:")
            self.logger.error(traceback.format_exc())
        
    def evalueate_transcriptomic_impact(self):
        """
        
        """
        self.logger.info(">>> STEP 5.5: Evaluating Transcriptomic Impact (if applicable)")
        adata = self.external.get('adata_qc', None)
        methods_cols = getattr(self, 'methods_cols', None)
        final_trusted_ntc = getattr(self, 'final_trusted_ntc', None)
        group_level = self.cfg['transcriptomic_impact']['group_level']
        if group_level == 'gene':
            col_idx = 2
        else:
            col_idx = 1 # sgRNA level

        # transcriptomic impact evaluation
        try:
            if adata is None or methods_cols is None or final_trusted_ntc is None:
                self.logger.warning("Missing adata_qc, methods_cols, or final_trusted_ntc, skip transcriptomic impact evaluation.")
                return
            
            evaluator = TranscriptomeEvaluator(
                adata=adata,
                embedding_key='cellbender_embedding',
                ntc_sgrnas=final_trusted_ntc,
                reference_method_sgRNA_col='sin_sg_Domin_cellbender_clean')
            
            all_cell_metrics = []

            for method_name, cols in methods_cols.items():
                df_res = evaluator.calculate_cell_metrics(
                    level_col=cols[col_idx], 
                    method_name=method_name
                )
                if df_res is not None:
                    all_cell_metrics.append(df_res)

            # Long Form DataFrame of all cell-level metrics
            if all_cell_metrics:
                final_cell_df = pd.concat(all_cell_metrics, axis=0, ignore_index=True)
                # Grouped mean Level with SEM
                final_agg_df = evaluator.aggregate_metrics(final_cell_df)
                self.logger.info("Transcriptomic impact evaluation complete.")
                # Save results
                final_agg_df.to_csv(self.dirs['05_transcriptomic_evaluation'] + '/transcriptomic_impact_summary.csv', index=False)
        except Exception:
            self.logger.error("Error in transcriptomic impact evaluation stage:")
            self.logger.error(traceback.format_exc())

        # plot comarison figures
        # Define the metrics and methods
        metrics = ['Silhouette', 'kNN_Purity', 'Dist_to_NTC']
        methods = ['Domin_strict_discard', 'GMM_strict_discard', 'Domin_cellbender', 'GMM_cellbender']
        method_display_names = [
            "Dominance Assign\n(Strict Discard)", 
            "GMM Assign\n(Strict Discard)", 
            "Dominance Assign\n(CellBender cleaned)", 
            "GMM Assign\n(CellBender cleaned)"
        ]
        out_dir = os.path.join(self.dirs['05_transcriptomic_evaluation'], 'plots')
        os.makedirs(out_dir, exist_ok=True)

        try:
            # Create a figure with a 4x3 grid
            fig1 = plt.figure(figsize=(12, 13.2), layout='constrained')
            gs = fig1.add_gridspec(4, 3, hspace=0.1, wspace=0.1)

            # Iterate over methods and metrics to create subplots
            for row, method in enumerate(methods):
                for col, metric in enumerate(metrics):
                    ax = fig1.add_subplot(gs[row, col])
                    plot_cross_scatter(
                        final_agg_df,
                        method_x=f'{method}_raw',
                        method_y=f'{method}_clean',
                        metric=metric,
                        top_n_labels=5,
                        show_NA=False,
                        ax=ax
                    )

                    if row == len(methods) - 1:
                        ax.set_xlabel("Raw (Mean ± SEM)", fontsize=7)
                    else:
                        ax.set_xlabel("")
                        
                    if col == 0:
                        ax.set_ylabel("Cleaned (Mean ± SEM)", fontsize=7)
                        
                        ax.annotate(method_display_names[row], 
                                    xy=(-0.25, 0.5), 
                                    xycoords='axes fraction', 
                                    rotation=90, 
                                    ha='center', va='center', 
                                    fontweight='bold', fontsize=10,
                                    # color='navy'
                                    )
                    else:
                        ax.set_ylabel("")

                    if row == 0:
                        ax.set_title(metric.replace('_', ' '), fontweight='bold', fontsize=10, pad=15)
                    else:
                        ax.set_title("")

                    # ax.set_ylabel(ax.get_ylabel(), fontsize=7)
                    # ax.set_xlabel(ax.get_xlabel(), fontsize=7)

            plt.suptitle("Cross-Method Transcriptome Evaluation", fontsize=12, fontweight='bold', y=1.02)
            cross_pdf_path = os.path.join(out_dir, "07transcriptomic_impact_cross_method.pdf")
            cross_png_path = os.path.join(out_dir, "07transcriptomic_impact_cross_method.png")
            fig1.savefig(cross_pdf_path, format='pdf', bbox_inches='tight')
            fig1.savefig(cross_png_path, format='png', dpi=300, bbox_inches='tight')
            self.logger.info(f"Transcriptomic impact cross-method plots saved to {cross_pdf_path} and {cross_png_path}")
            plt.close(fig1)
        except Exception:
            self.logger.error("Error in transcriptomic impact plotting stage:")
            self.logger.error(traceback.format_exc())

    # --------------------------------------------------------------------------
    # pipeline step 
    # --------------------------------------------------------------------------
    def run_pipeline(self, h5_path):
        self.logger.info(f"Pipeline started for: {h5_path}")
        font_dir: Path = get_font_dir(self.cfg['font'])
        register_fonts_clean(font_dir=font_dir, family=self.cfg['font'])
        set_publication_style()
        print(logging.root.manager.loggerDict.keys())
        
        self.run_loading(h5_path)
        self.run_cleaning()
        self.run_single_feature_reports()
        self.run_dominance_assignment()
        self.run_gmm_assignment()
        self.run_evaluation()
        
        if self.external['adata_cb'] is not None:
            self.logger.info(">>> STEP 5: Running Transcriptomic Evaluation")
            self.adata_preprocessing()
            self.external['adata_qc'] = self.adata_qc()
            self.adata_assign_mapping()
            self.adata_dimred_umap_density()
            self.negtive_control_evaluation()
            self.evalueate_transcriptomic_impact()
            
            adata_final = self.external['adata_qc']
            adata_final.write_h5ad(os.path.join(self.dirs['05_transcriptomic_evaluation'], 'adata_qc.h5ad'))
            self.logger.info("Transcriptomic evaluation complete. QC-ed adata saved.")
