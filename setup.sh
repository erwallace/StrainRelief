#!/bin/bash
# Environment setup for OMol25 ligand strain evaluation.
#
# Install order matters:
#   1. Install strain-relief (which pulls in mace-torch)
#   2. Install fairchem-core, then force-reinstall e3nn==0.5
#      (fairchem-core pins e3nn>=0.5 but mace-torch may have pulled in a newer version)
#
# Run once from the repo root: bash setup.sh

set -euo pipefail

echo "=== Installing strain-relief and dependencies ==="
uv pip install -e ".[dev]"

echo "=== Installing fairchem-core ==="
uv pip install fairchem-core

echo "=== Pinning e3nn==0.5 (required for fairchem-core + mace-torch compatibility) ==="
uv pip install --force-reinstall "e3nn==0.5"

echo "=== Installing data download tool ==="
uv pip install huggingface_hub

echo "=== Done ==="
python -c "import strain_relief; print('strain_relief OK')"
python -c "from fairchem.core import FAIRChemCalculator; print('fairchem OK')"
python -c "from mace.calculators import MACECalculator; print('mace OK')"
