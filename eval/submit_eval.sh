#!/bin/bash
# Submit a single ligand strain evaluation job to SLURM.
#
# Usage: sbatch eval/submit_eval.sh <model_type> <checkpoint_path> <output_path> [subset]
#
# Arguments:
#   model_type      "esen" or "mace"
#   checkpoint_path Path to the model checkpoint file (.pt or .model)
#   output_path     Path to write results JSON (e.g. results/esen_v1.json)
#   subset          Optional: number of systems to evaluate (for smoke testing)
#
# Example:
#   sbatch eval/submit_eval.sh esen /path/to/esen_v1.pt results/esen_v1.json

#SBATCH --job-name=ligstrain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

MODEL_TYPE=${1:?'model_type required: esen or mace'}
CHECKPOINT=${2:?'checkpoint_path required'}
OUTPUT=${3:?'output_path required'}
SUBSET=${4:-""}

echo "=== Ligand Strain Evaluation ==="
echo "Model type:  $MODEL_TYPE"
echo "Checkpoint:  $CHECKPOINT"
echo "Output:      $OUTPUT"
echo "Date:        $(date)"
echo "Host:        $(hostname)"
echo "================================"

SUBSET_ARG=""
if [ -n "$SUBSET" ]; then
    SUBSET_ARG="--subset $SUBSET"
fi

python eval/run_ligand_strain.py \
    --model-type "$MODEL_TYPE" \
    --checkpoint "$CHECKPOINT" \
    --data-path data/omol25/eval/ligand_strain_inputs.pkl \
    --output "$OUTPUT" \
    --device cuda \
    $SUBSET_ARG

echo "Done at $(date)"
