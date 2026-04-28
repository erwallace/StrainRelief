#!/bin/bash
# Submit all ligand strain evaluation jobs.
# Edit the checkpoint paths below before running.
#
# Usage: bash eval/launch_all.sh

set -euo pipefail

mkdir -p logs results

# ---------------------------------------------------------------------------
# eSEN checkpoints — add or remove lines as needed
# ---------------------------------------------------------------------------
sbatch eval/submit_eval.sh esen /path/to/esen_v1.pt     results/esen_v1.json
sbatch eval/submit_eval.sh esen /path/to/esen_v2.pt     results/esen_v2.json
sbatch eval/submit_eval.sh esen /path/to/esen_v3.pt     results/esen_v3.json

# ---------------------------------------------------------------------------
# MACE checkpoint
# ---------------------------------------------------------------------------
sbatch eval/submit_eval.sh mace /path/to/mace.model     results/mace.json

echo "All jobs submitted. Monitor with: squeue -u \$USER"
