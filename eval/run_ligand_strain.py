"""Run the OMol25 ligand strain evaluation for a single model checkpoint.

Usage:
    python eval/run_ligand_strain.py \
        --model-type esen \
        --checkpoint /path/to/eSEN.pt \
        --data-path data/omol25/eval/ligand_strain_inputs.pkl \
        --output results/esen_v1.json

The evaluation data is a pickle file:
    {system_id: {"bioactive_conf": Atoms, "conformers": [Atoms, ...]}}

Results are saved as JSON:
    {system_id: {"bioactive": {"energy": float, "forces": [...]},
                 "gas_phase": {"0": {"initial": {...}, "final": {...}}, ...}}}
"""

import argparse
import json
import pickle
from pathlib import Path

from ase import Atoms
from loguru import logger as logging


def _annotate_atoms(atoms: Atoms) -> None:
    """Ensure atoms.info has charge and spin — required by FAIRChemCalculator (eSEN).

    Reads from atoms.info if already set; otherwise defaults to neutral singlet (0, 1).
    """
    if "charge" not in atoms.info or "spin" not in atoms.info:
        if "charge" not in atoms.info:
            logging.warning(
                "atoms.info missing 'charge'; defaulting to 0 (neutral). "
                "Set explicitly if the ligand is charged."
            )
            atoms.info["charge"] = 0
        if "spin" not in atoms.info:
            logging.warning(
                "atoms.info missing 'spin'; defaulting to 1 (singlet). "
                "Set explicitly if the molecule has radical electrons."
            )
            atoms.info["spin"] = 1


def annotate_charge_spin(data: dict) -> dict:
    """Inject charge and spin into every Atoms object in the evaluation dataset."""
    for system in data.values():
        _annotate_atoms(system["bioactive_conf"])
        for conf in system["conformers"]:
            _annotate_atoms(conf)
    return data


def load_calculator(model_type: str, checkpoint: str, device: str):
    """Load an ASE calculator using the StrainRelief calculator wrappers."""
    if model_type == "esen":
        from strain_relief.calculators import fairchem_calculator
        return fairchem_calculator(model_paths=checkpoint, device=device, default_dtype="float32")
    elif model_type == "mace":
        from strain_relief.calculators import mace_calculator
        return mace_calculator(model_paths=checkpoint, device=device, default_dtype="float32")
    raise ValueError(f"Unknown model_type: {model_type!r}. Must be 'esen' or 'mace'.")


def main():
    parser = argparse.ArgumentParser(description="Run OMol25 ligand strain evaluation.")
    parser.add_argument("--model-type", choices=["esen", "mace"], required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt/.model)")
    parser.add_argument("--data-path", required=True, help="Path to ligand_strain_inputs.pkl")
    parser.add_argument("--output", required=True, help="Path to write results JSON")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Only evaluate first N systems (for smoke testing)",
    )
    args = parser.parse_args()

    logging.info(f"Loading calculator: {args.model_type} from {args.checkpoint}")
    calc = load_calculator(args.model_type, args.checkpoint, args.device)

    logging.info(f"Loading evaluation data from {args.data_path}")
    with open(args.data_path, "rb") as f:
        input_data = pickle.load(f)

    if args.subset is not None:
        keys = list(input_data.keys())[: args.subset]
        input_data = {k: input_data[k] for k in keys}
        logging.info(f"Running subset of {args.subset} systems")

    logging.info(f"Annotating charge/spin for {len(input_data)} systems")
    input_data = annotate_charge_spin(input_data)

    logging.info("Running ligand_strain recipe...")
    from fairchem.core.components.calculate.recipes.omol import ligand_strain
    results = ligand_strain(input_data, calc)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f)
    logging.info(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
