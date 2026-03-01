# Perturb-Audit / CR-Collision-Analyzer

Perturb-Audit is an open-source diagnostic and denoising framework designed for CRISPR screening workflows using the 10x Genomics scRNA-seq platform with feature barcode technology. It audits the `CellRanger count` outputs, specifically the `molecule_info.h5` records, to identify and quantify cross-library collisions resulting from Post-GEM PCR chimeras. This framework is compatible with Cell Ranger versions 8.0.0 to 10.0.0.

The pipeline combines targeted physical removal (Strict-discard the cross-library collisions) with generative statistical modeling (CellBender) and adds two sgRNA assignment strategies.

## Table of contents
1. [Why this matters](#why-this-matters)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick start (CLI)](#quick-start-cli)
5. [Configuration](#configuration)
6. [Output layout](#output-layout)
7. [Working with the code](#working-with-the-code)
8. [Testing & validation](#testing--validation)
9. [Next steps](#next-steps)

## Why this matters

In the Perturb-seq workflow, mRNA and sgRNA molecules are captured in the same gel bead-in-emulsion (GEM) and jointly amplified. During the PCR amplification step, template switching can generate Post-GEM PCR chimeras where the CBC-UMI pair from one molecule is incorrectly fused to the transcript of another. Downstream tools such as Cell Ranger treat the CRISPR and GEX data as independent modalities, so collisions that reuse the same CBC-UMI pair in both libraries evade the standard Bayesian CBC correction and UMI deduplication logic. The result is widespread false-positive sgRNA assignments that bias cell identities and dilute biological signals.

Perturb-Audit plugs this gap by: auditing the raw `molecule_info.h5` file for cross-library collisions, quantifying the scope of the chimeric noise, and providing two denoising strategies (Strict-discard and CellBender). The toolkit generates diagnostic reports (Fig. 1 in the manuscript) that highlight collision statistics, single-feature behavior, assignment exploration, and evaluation metrics.

## Features

- Cross-library collision detection by inspecting CBC-UMI overlaps between GEX and sgRNA reads.
- Collision summary reporting with publication-style plots (`report1`), font 'Arial' is automatically registrated for matplotlib.
- Dual cleaning strategies: physical removal and statistical modeling.
- Assignment audits (dominance and GMM strategies) and transcriptomic evaluation to compare approaches.
- Evaluation helpers (GMM diagnostics, transcriptome QC, NTC QC, sgRNA density).
- All-phase orchestration via `cr_analyzer.runner.CollisionRunner`.

## Installation

```
git clone https://github.com/yourname/perturb-audit.git
cd perturb-audit

conda env create -f environment_analysis.yml
conda activate perturb_audit_env

pip install -e .

# check it
perturb-audit --help
```

## Quick start
### Running CellBender Denoising (Optional)

If you plan to run CellBender yourself, you must:
- Install CellBender separately (v3.0.0 tested).
- Run cellbender remove-background to generate a denoised .h5.
- Then use this package for downstream analysis.

Example workflow:
```
# In your CellBender environment
cellbender remove-background \
  --input raw_feature_bc_matrix.h5 \
  --output cellbender_denoised.h5 \
  --fpr 0.01 0.05
```

### Running Pertub-audit (CLI)
The entire pipeline can be executed through the command line via `main.py`. Example:

```bash
perturb-audit \
  -o ./02cr_analysis/A1_C1 \
  -c ./configs/KLF4-A1_C1.yaml \
  -i ./molecule_info.h5
```

Arguments:
- `-i / --input`: path to the `molecule_info.h5` file.
- `-o / --output`: directory where all reports, matrices, and logs are written.
- `-c / --config`: configuration YAML (defaults to `configs/default_config.yaml`).

## Configuration

All configuration is YAML-driven. `configs/default_config.yaml` contains the full set of tunable options: logging level, report parameters, cleaning thresholds, assignment criteria, and evaluation settings. Key sections include:

- `logging`: controls console verbosity and file logging (`save_to_file`, log level).
- `report*_params`: shape the plotting behavior for collision summary, single-feature reports, assignment EDA, and GMM reports.
- `cleaning_mtx`, `sgRNA_list`: thresholds for filtering matrices and selecting sgRNAs.
- `dominance_assignment` and `gmm_assignment`: tuning parameters for the two assignment strategies.
- `external_inputs`: optional hooks to compare CellBender output.
- `strict_or_loose_gmm`: determines whether to use strict or relaxed thresholds when calling GMM assignments.

The `configs/KLF2-A1_C1.yaml` file is recommended run in the first default run, and the threshold parameters should be adjusted based on subsequent results to suit your data characteristics.

## Output layout

The `CollisionRunner` initializes a structured directory hierarchy inside the output folder:

- `01_collision_analysis/summary_reports`: collision summary report PDF.
- `02_single_feature`: density plots for individual sgRNAs.
- `03_assignment_analysis`: dominance and GMM assignment details.
- `04_evaluation_analysis`: quantitative evaluation figures.
- `05_transcriptomic_evaluation`: embedding/QC visuals for transcriptome data.
- `logs`: console and file logs (per run timestamp).

Each subfolder stores figures, PDF reports, and intermediate CSVs for that stage. Expect per-strategy artifacts (`strict_discard`, `cellbender`) inside the cleaner/assignment subdirectories.

## Working with the code

- Entry point: `main.py` sets up `CollisionRunner`.
- Core runner: `cr_analyzer/runner.py` orchestrates loading, cleaning, assignment, and report generation.
- Modules under `cr_analyzer/analysis_core` and `cr_analyzer/visualization` implement the heavy lifting (data loaders, matrix builders, collision detectors, GMM helpers, plotters).
- Configurable fonts and plotting styles ensure publication-ready visuals (`visualization/plotter.py`).

If you need module-level details, refer to `cr_analyzer/runner.py` or `test.ipynb` for the full module-execution chain.

## Testing & validation

no automated test suite yet

## Next Steps
...
