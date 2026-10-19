#!/bin/sh
# reproduce.sh record  - regenerate every table row and figure from scripts/results/ (minutes)
# reproduce.sh verify  - check every trained policy against the record (about 30 minutes)
set -e
ROOT=$(cd "$(dirname "$0")" && pwd)
case "${1:-record}" in
  record)
    cd "$ROOT/scripts"
    python3 paper_rows.py results/runs.csv > results/paper_rows.tex
    python3 stats_tools.py results/runs.csv --all > results/tables.tex
    python3 2_anytime.py --from-json
    python3 7_figures.py
    python3 6_behaviour.py --from-csv
    python3 stats_tools.py --curves
    echo "tables: scripts/results/paper_rows.tex, scripts/results/tables.tex; figures: scripts/fig_*.pdf" ;;
  verify)
    cd "$ROOT" && python3 tools/verify_policies.py ;;
  *) echo "usage: ./reproduce.sh [record|verify]"; exit 2 ;;
esac
