# Carbon-allowance risk forecasting: CMD-MoE

Reproducible research code and archived out-of-sample forecasts for **Adaptive distributional forecasting of carbon-allowance risk with a causal multiscale mixture of experts**. The code implements the causal multiscale mixture-of-experts (CMD-MoE) forecasting system, distributional and GARCH-family benchmarks, ablations, sensitivity analyses, and the main tables and figures.

## Repository structure

- `code/run_cmd_moe_nested.py` — annual, strictly ordered training / validation / test folds for the core system.
- `code/run_ablation.py` — benchmark ablations and the intermediate fold arrays required by the expanded analysis.
- `code/garch_benchmarks.py` — Student-t GARCH, GJR-GARCH, and EGARCH, including 90%/95% interval scores and 1%/5% quantile losses.
- `code/extended_analysis.py` — comparator forecasts, sensitivity and inferential analyses, and additional risk diagnostics.
- `code/verify_reproducibility.py` — verify the stored daily scores, GARCH evaluation, HAC lag sensitivities, and bootstrap confidence intervals.
- `code/make_tables_figures.py` and `code/make_architecture.py` — generate descriptive tables and figures.
- `results/` — frozen daily out-of-sample predictions and aggregated results. These files are included so readers can examine the reported results without costly re-estimation.
- `data/` — place the appropriately licensed raw OHLCV input here for end-to-end retraining.

## Quick verification — no raw input needed

Use Python 3.11+ in an isolated environment. Install the dependencies and run:

```bash
python -m pip install -r requirements.txt
python code/verify_reproducibility.py
```

This audit checks each pooled daily distributional score against its archived prediction records; validates GARCH 90% interval scores and 1% pinball losses; recomputes HAC tests at lags 3, 7, 14 and 21; and checks 1,000-replication circular block bootstrap intervals for block sizes 10 and 20. There are 1,393 dated out-of-sample observations across six folds (2021–2026, with 2026 partial).

The verification uses only saved daily prediction records. It **does not retrain the models**, and is not a substitute for a computationally intensive full re-estimation.

## Where to find the fold-level forecasts

The CSV files have explicit observation dates and test years, and the comparative records also identify each model:

| File | Contents |
| --- | --- |
| `results/nested_predictions.csv` | CMD-MoE daily forecasts, component variances, and time-varying expert weights. |
| `results/comparative_predictions.csv` | Daily forecasts, risk quantiles, losses, and interval scores for CMD-MoE and alternative expert combinations. |
| `results/garch_predictions.csv` | GARCH/GJR-GARCH/EGARCH daily forecasts with 90% and 95% interval scores, 1% and 5% quantile losses, and calibration thresholds. |
| `results/ablation_inference.csv` | Mean loss gains, HAC tests across four lags, and block bootstrap intervals at both block sizes. |
| `results/proper_score_comparisons.csv` | Aggregated risk-score comparisons; the daily data remain available for independent averaging. |

The remaining CSV/JSON records in `results/` contain the calibration, robustness, ablation, hyperparameter, reserve-decision, and diagnostic summaries.

## Full end-to-end re-estimation

The original market-data source is the Investing.com continuous European Union Allowance (EUA) futures OHLCV history (historical instrument identifier `CFI2Zc1`), covering the source period used by the study through 11 June 2026. Its raw downloaded input, `data/eua.csv`, is **not distributed here** because market-data redistribution may require permission. Obtain a licensed copy of the matching input directly from the provider and save it to `data/eua.csv`. The scripts parse and clean the raw source data; a pre-cleaned input is not required. Different snapshots of a continuous futures series can produce non-identical results.

Then, in this order, from the repository root:

```bash
python code/run_cmd_moe_nested.py
python code/run_ablation.py
python code/garch_benchmarks.py
python code/extended_analysis.py
python code/make_tables_figures.py
python code/make_architecture.py
python code/verify_reproducibility.py
```

**Order matters:** `run_ablation.py` creates `results/extended_cache/fold_<year>.npz` intermediate arrays required by `extended_analysis.py`. Training is expensive and the scripts overwrite existing outputs; work in a copy of the repository and preserve a backup of the frozen `results/` folder. Slight numerical variation across CPU and library builds is possible. For a visual-only regeneration, the plotting scripts can use the distributed frozen results plus appropriately licensed raw inputs for price diagnostics.

## Data provenance and reuse

The saved prediction/results files are research outputs from the study. Some include return/target variables derived from externally sourced market data; confirm all applicable rights before publicly redistributing these outputs. The vendor OHLCV source file is intentionally omitted from this package. No license is granted for third-party market data, and no public DOI is represented by this archive. The manuscript should cite the **actual** repository URL and, if available, its DOI only after the repository is published.

For questions about the manuscript or data-access permissions, contact the corresponding author identified in the published paper.
