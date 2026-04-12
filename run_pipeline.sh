#!/bin/bash
# ============================================================
# Full reproducibility pipeline
# Convergent Timing, Divergent Trajectories
# IEEE VIS 2026 Short Papers (Anonymous Submission)
# ============================================================
# Usage:
#   bash run_pipeline.sh              # synthetic data (default)
#   bash run_pipeline.sh --semantic   # + semantic robustness
# ============================================================

set -e
SEMANTIC=false
for arg in "$@"; do [[ "$arg" == "--semantic" ]] && SEMANTIC=true; done

echo "========================================================"
echo " VIS 2026 Reproducibility Pipeline"
echo "========================================================"

# 1. Install dependencies
echo "[1/4] Installing dependencies..."
pip install -r requirements.txt -q

# 2. Generate synthetic data
echo "[2/4] Generating synthetic corpus..."
python generate_synthetic.py --n_sessions 500 --seed 42 --output_dir data/
echo "      -> data/synthetic_sessions.csv"
echo "      -> data/synthetic_turns.csv"

# 3. Keyword anchoring
echo "[3/4] Running keyword anchoring..."
if [ "$SEMANTIC" = true ]; then
    python analysis/keyword_anchoring.py \
        --sessions data/synthetic_turns.csv \
        --taxonomy data/keyword_taxonomy.csv \
        --semantic
else
    python analysis/keyword_anchoring.py \
        --sessions data/synthetic_turns.csv \
        --taxonomy data/keyword_taxonomy.csv
fi
echo "      -> results/anchored_sessions.csv"

# 4. Survival analysis
echo "[4/4] Running survival & hazard analysis..."
python analysis/survival_analysis.py \
    --sessions data/synthetic_sessions.csv
echo "      -> results/full_detection_table.csv"
echo "      -> results/marginal_gains.csv"
echo "      -> results/figure_traj.pdf"

echo ""
echo "========================================================"
echo " Done. Check results/ for all outputs."
echo " Open visualization/interactive_dashboard.html"
echo " in any browser for the interactive figure."
echo "========================================================"
