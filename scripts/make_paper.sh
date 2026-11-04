#!/usr/bin/env bash
# Regenerate the tables and figures of the paper.
#
#   scripts/make_paper.sh <results_dir> <out_dir>
#
# <results_dir>  the result files written by evodur.run / evodur.evidence / evodur.masked (read only; never written to)
# <out_dir>      tables/*.tex, figs/*.pdf|png and intermediate csv files go here (created if needed)
#
# Needs only the results: tab_main_single, tab_main_ens, tab_ablation, tab_states, tab_arrival, tab_point, tab_point_minutes,
# tab_coverage, tab_cost, tab_stage_minutes, tab_own, tab_delay, tab_class, tab_masked, tab_prior_work, and the figures
# fig_summary, fig_gain, fig_dynamic_known, fig_learning, fig_learning_known, fig_value, fig_value_beijing, fig_teaser.
# Needs the data (EVODUR_DATA, all six datasets incl. the non-public Beijing data): tab_example, tab_labels, tab_beijing_vars,
# tab_public_vars, fig_data, fig_method_curves. These are skipped with a message when the data are missing.
#
# Environment: PYTHON (interpreter, default python), EVODUR_DATA (data directory).
set -uo pipefail
[ $# -eq 2 ] || { echo "usage: $0 <results_dir> <out_dir>" >&2; exit 2; }
RESULTS=$(cd "$1" && pwd); OUT=$2; mkdir -p "$OUT/tables" "$OUT/figs" "$OUT/cache"; OUT=$(cd "$OUT" && pwd)
PYTHON=${PYTHON:-python}
HERE=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
fail=0
run() { echo "+ $*"; "$@" || { echo "FAILED: $*" >&2; fail=1; }; }
have_data() {  # are the raw files of all six datasets present?
  $PYTHON -c "import sys; from evodur.check_data import DATA, FILES; sys.exit(0 if all((DATA / f).exists() for f in FILES.values()) else 1)"
}

# 1. tables and figures computed from the results
run $PYTHON -m evodur.analysis.paper_results --results "$RESULTS" --out "$OUT"
run $PYTHON -m evodur.analysis.ops_analysis --results "$RESULTS" --out "$OUT/tables"
run $PYTHON -m evodur.analysis.application --results "$RESULTS" --out "$OUT/tables" --figs "$OUT/figs"   # also rewrites app_example.csv
run $PYTHON -m evodur.analysis.tables_extra --results "$RESULTS" --out "$OUT/tables"
cp "$HERE/static/tab_prior_work.tex" "$OUT/tables/"                                   # hand-written literature table
ls "$RESULTS"/interpret_*_f[0-9].csv >/dev/null 2>&1 && \
  run $PYTHON -m evodur.analysis.interpret_tables --results "$RESULTS" --out "$OUT/tables" --figs "$OUT/figs"   # only if the interpret runs exist

# 2. teaser figure (vector SVG -> PDF with headless Chrome: npm install puppeteer)
run $PYTHON -m evodur.figures.teaser_svg "$OUT/figs"
if command -v node >/dev/null && node -e "require('puppeteer')" 2>/dev/null; then
  run node "$HERE/evodur/figures/render_svg.js" "$OUT/figs/fig_teaser.svg" "$OUT/figs/fig_teaser"
else
  echo "node/puppeteer not available: figs/fig_teaser.svg written, but not rendered to fig_teaser.pdf" >&2
fi

# 3. outputs that need the data
if have_data; then
  run $PYTHON -m evodur.analysis.data_tables --out "$OUT/tables"
  run $PYTHON -m evodur.figures.fig_data --out "$OUT/figs"
  run $PYTHON -m evodur.figures.fig_evidence method --out "$OUT/figs" --cache "$OUT/cache"
else
  echo "data missing (see 'python -m evodur.check_data'): skipped tab_example, tab_labels, tab_beijing_vars, tab_public_vars, fig_data, fig_method_curves" >&2
fi
exit $fail
