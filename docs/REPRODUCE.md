# Reproducing the paper

This page gives the details behind the short instructions in the README.

## Environment

Python 3.10 or newer. Everything runs on CPU.

`requirements.txt` pins the versions used for the paper (Python 3.10.14, numpy 2.2.6, pandas 2.3.3, scipy 1.15.3,
scikit-learn 1.7.2, scikit-survival 0.25.0, lifelines 0.30.0, xgboost 3.2.0, torch 2.14.1 CPU, matplotlib 3.10.9).
Install in editable mode (`pip install -e .`) and run the scripts from the repository root: the data, results and `expected/` directories are found relative to the checkout. The teaser figure is rendered to PDF with
headless Chrome (`npm install puppeteer`, Node.js); without it the vector `fig_teaser.svg` is still written.


## Data

Place the data in `data/` (override with `EVODUR_DATA`); results go to `results/` (override with `EVODUR_RESULTS`).
No data are part of this repository.

| Name in code | Dataset | Incidents (after cleaning, tau = 1 min) | Arrival time | Source |
|---|---|---|---|---|
| `ring4` | Beijing 4th Ring Road, 2008 | 5,388 | yes | on request |
| `citywide` | Beijing, whole city, 2008 | 40,120 | yes | on request |
| `ny511early` | 511 NY events, accidents created 2010-2015 | 206,009 | no | data.ny.gov `ah74-pg4w` |
| `sydney` | Sydney road traffic incidents (crashes) | 38,607 | no | Mendeley Data `cgnx2cs665`, version 5 |
| `seattle` | Seattle police CAD, motor-vehicle collisions | 253,758 | yes | data.seattle.gov `33kz-ixgy` |
| `sanfrancisco` | San Francisco police CAD, vehicle accidents | 88,186 | yes | data.sf.gov `2zdj-bwza` |

The two Beijing files are available from the corresponding author on request. Put them at `data/raw_data.xlsx` (4th Ring Road) and
`data/raw_data_extended_stata.xlsx` (citywide). The hourly weather of the Beijing data (NOAA ISD station 54511) is
downloaded with the public data and converted automatically.

**Public data.**

```bash
scripts/download_data.sh                  # sydney ny511 seattle sanfrancisco (add: weather)
python -m evodur.check_data               # row counts and checksums against expected/data_checks.json
```

The Socrata portals keep adding records, so each query is pinned to the snapshot used in the paper:

| Dataset | Pinned call-time window | Also excluded (late-arriving records dated inside the window) | Rows in the pinned download |
|---|---|---|---|
| `seattle` | 2009-06-02 03:54:42 to 2026-10-23 23:33:47 | event `2026000316760` | 267,982 events (253,758 after cleaning) |
| `sanfrancisco` | 2014-12-31 23:18:13 to 2026-10-26 23:33:34 | cad_number 262993680, 262992897, 262993594, 262993485 | 90,437 rows (88,186 after cleaning) |
| `ny511early` | calls created 2010-01-01 to 2015-12-31 | none | 225,731 accident events (206,009 after cleaning) |
| `sydney` | fixed release (Mendeley version 5) | none | `Final Road Traffic Incidents.xlsx`, sha256 `6a1ee43c...` |

The same window and exclusions are applied by the loaders (`evodur.public_data.WINDOWS`, `EXCLUDE`), so a larger local
file gives the same table. The SF download written by `download_data` was compared with the snapshot of the paper and
is byte-identical to it. `check_data` prints the number of incidents and two SHA-256 checksums of the processed records
(`content_hash`: the records sorted by call time, location, label and duration; `order_hash`: the same in table order,
which fixes the cross-validation folds) and compares them with `expected/data_checks.json`. If the portal has changed
records inside a window (late corrections), the numbers differ slightly from the paper. A message tells you which of the
two checks failed.

The NY 511 portal returns rows in no stable order. The folds are drawn by row position, so the loader first sorts the
rows canonically and then applies the permutation stored in `expected/ny511early_row_order.npz` (integers only) to
restore the order of the snapshot of the paper.

## Model names in the code

| Paper | Code (`--models`) |
|---|---|
| EvoDur | `Evidence-DeepMix-8`, `-8-het`, `-16`, `-16-het` (K = 8 or 16 components, shared or covariate-dependent shapes); `paper_results` keeps the one with the best validation likelihood |
| Neural CR mixture / with 2 or 4 per class | `DeepCAFT` / `DeepCAFT-M` |
| Linear CR mixture | `Li2015` |
| Weibull MDN (K selected / K = 4) | `WeibullMDN-sel` / `WeibullMDN-4` |
| DeepHit, Neural Fine-Gray, DeepSurv, RSF | `DeepHit-CR`, `NeuralFineGray`, `DeepSurv`, `RSF` |
| Linear AFT, XGBoost-AFT | `AFT`, `XGBoost-AFT` |

## Full reproduction

1. Get the data (above).
2. Make the job list and run it. Every command is one fold of one group and the commands are independent.

```bash
python scripts/make_jobs.py -o jobs.txt                       # all six datasets (790 commands); or: make_jobs.py seattle sanfrancisco
scripts/run_sequential.sh jobs.txt                            # one machine, one command after the other, restartable
mkdir -p logs && sbatch --array=1-$(wc -l < jobs.txt)%50 scripts/slurm_array.sbatch.example jobs.txt   # SLURM
```

   The commands reproduce the flags of the original runs: `--tau 1`, batch size 256 and three repeats of 5-fold
   cross-validation on `ring4`, batch size 1024 and one repeat on the other datasets, 5-seed ensembles for the
   ensemble rows. Groups (`--groups`): `final` (main comparison, ablation,
   ensembles, point and interval measures, evidence experiment), `learning` (learning curves, 10, 25 and 50% of the
   training data), `ops` (per-incident forecasts for the operational tables; only the four datasets with an arrival
   time), `masked`, and `interpret` (effects of call-time attributes and labels).
   An extra group, `sens`, repeats the main comparison without records shorter than 15 min (`--tau 15`); it is not part
   of the paper and runs only when asked for (`--groups sens`).
3. Make the tables and figures:

```bash
scripts/make_paper.sh results paper_out         # tables in paper_out/tables, figures in paper_out/figs
```

   `make_paper.sh <results_dir> <out_dir>` reads the results (never writes into them) and writes everything else to
   `out_dir`. The outputs that need the data (`tab_example`, `tab_labels`, `tab_beijing_vars`, `tab_public_vars`,
   `fig_data`, `fig_method_curves`) are made when the data are in `EVODUR_DATA` and skipped otherwise.

### Compute

CPU only; the paper's jobs used 4 threads each (`--threads 4`, set by `make_jobs.py`). Training time in hours
(the sum of the per-fold training times that `run.py` records, per group; evidence experiments, `ops` and `masked` runs
come on top and are of the same order as `final`):

| Dataset | final | learning |
|---|---|---|
| ring4 | 4.4 | 0.5 |
| citywide | 6.5 | 1.4 |
| ny511early | 50 | 6.0 |
| sydney | 6.1 | 1.3 |
| seattle | 113 | 9.2 |
| sanfrancisco | 13 | 2.1 |

Wall time of one job in minutes (median of the per-tag medians and maximum; `expected/compute_cost.csv`, from the scheduler
records, for the first four datasets; Seattle and San Francisco were not recorded there):

| Dataset | `run.py` jobs, median / max | `evidence.py` jobs, median / max |
|---|---|---|
| ring4 | 1.6 / 27 | 0.9 / 6 |
| citywide | 3.8 / 42 | 3.3 / 12 |
| ny511early | 25 / 391 | 21 / 58 |
| sydney | 3.4 / 32 | 2.8 / 6 |

The training times above sum to about 240 job-hours (4 threads each) for these three groups; the evidence, `ops` and `masked`
runs come on top (their wall times are in the second table). The recorded peak memory of a `run.py` job was 3.7 GB (ring4),
2.8 GB (sydney), 28 GB (citywide) and 36 GB (ny511early), so give jobs on the larger datasets generous memory (the SLURM example
asks for 40 GB, which is safe for all recorded jobs). Peak memory was not recorded for Seattle and San Francisco. `tab_cost` of the paper lists
the mean training time per fold of every model. An EvoDur update takes 1-3 ms per incident on one core; `python scripts/time_update.py` measures it.

## Tables and figures

Result groups are the ones of `scripts/make_jobs.py`. All commands are run by `scripts/make_paper.sh`; the table files are
`<out_dir>/tables/<name>.tex`, the figures `<out_dir>/figs/<name>.pdf` (and `.png`).

| Output | Made by | Needs |
|---|---|---|
| `tab_prior_work` | copied from `static/` (hand-written) | |
| `tab_main_single`, `tab_main_ens` | `analysis.paper_results` | final |
| `tab_ablation`, `tab_point`, `tab_point_minutes`, `tab_coverage`, `tab_cost` | `analysis.paper_results` | final |
| `tab_states`, `tab_arrival` | `analysis.paper_results` | final (evidence) |
| `tab_stage_minutes`, `tab_own`, `tab_delay` | `analysis.ops_analysis` | ops |
| `tab_class`, `fig_value`, `fig_value_beijing` | `analysis.application` | ops |
| `tab_masked` | `analysis.tables_extra` | final (evidence), masked |
| `tab_interpret_call`, `tab_interpret_label`, `tab_importance`, `fig_interpret` | `analysis.interpret_tables` | interpret |
| `tab_example` | `analysis.tables_extra` | ops, Seattle data |
| `tab_labels`, `tab_beijing_vars`, `tab_public_vars`, `fig_data` | `analysis.data_tables`, `figures.fig_data` | data |
| `fig_summary`, `fig_gain`, `fig_dynamic_known` | `analysis.paper_results` | final |
| `fig_learning`, `fig_learning_known` | `analysis.paper_results` | learning |
| `fig_method_curves` | `figures.fig_evidence method` | data |
| `fig_teaser` | `figures.teaser_svg`, `figures/render_svg.js` | nothing |

Also written (not in the paper): `tab_main`, `tab_point_known`, `tab_incident`, `tab_metrics_*`, `incident_tests.csv`,
`paired_wins.csv`, `fig_pit`, `fig_dynamic`. EvoDur's own results are derived from four stored prior variants (the one
with the lowest validation likelihood, as in the paper) and written to `<out_dir>/_derived`.

## Reproducibility check

`make_paper.sh` was run on the stored results of the paper, and every table it wrote was compared with the
corresponding table of the manuscript with `diff`; they are identical. Training is deterministic on a given machine:
re-running fold 0 of San Francisco and Sydney from a fresh clone, with data that pass `check_data`, gave the stored numbers to the
last digit. On other CPUs, floating-point sums can differ slightly. On fold 0 of NY/NJ/CT 511, run on different
hardware, three baselines (Weibull MDN with K = 4 and with selected K, DeepHit) differed by at most 0.0008 in log score
per incident, and EvoDur and all other models were identical.

## Notes

- **Row order of NY 511.** See above; without `expected/ny511early_row_order.npz` the folds, and hence the numbers, differ slightly.
- **Beijing data** are not public; the Beijing columns of the tables and `fig_data` need them.
- **`fig_teaser.pdf`** needs headless Chrome (puppeteer); without it only the SVG is written.

