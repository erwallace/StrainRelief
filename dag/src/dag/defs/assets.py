import os

import dagster as dg
import pandas as pd
from neural_optimiser.conformers import Conformer, ConformerBatch
from rdkit import Chem
from strain_relief.conformers import generate_conformers
from strain_relief.constants import MOL_COL_NAME
from strain_relief.data_types import MolsDict
from strain_relief.io import load_parquet, process_output, to_mols_dict
from strain_relief.optimisation import run_optimisation

from .configs import (
    ConformerConfig,
    GlobalOptimisationConfig,
    InputConfig,
    LocalOptimisationConfig,
    OutputConfig,
)
from .resources import DATA_DIR, GlobalOptimiserResource, LocalOptimiserResource


@dg.asset(io_manager_key="pandas_io_manager")
def input_df(config: InputConfig) -> pd.DataFrame:
    """Load the input parquet of docked molecules.

    Drop the live RDKit ``mol`` column before returning: it cannot be serialised to parquet.
    Molecules persist as ``mol_bytes`` and are reconstructed downstream.
    """
    df = load_parquet(parquet_path=config.parquet_path)
    return df.drop(columns=[config.mol_col_name or MOL_COL_NAME], errors="ignore")


@dg.asset
def docked_mols(input_df: pd.DataFrame, config: InputConfig) -> dict:
    """Convert the input DataFrame to a MolsDict."""
    mols: MolsDict = to_mols_dict(input_df, **config.model_dump())
    return mols


@dg.asset(io_manager_key="pytorch_io_manager")
def conformers(docked_mols: dict, config: ConformerConfig) -> ConformerBatch:
    """Generate conformers for the input molecules."""
    generated_mols = generate_conformers(docked_mols, **config.model_dump())
    generated_batch: ConformerBatch = ConformerBatch.cat(
        [ConformerBatch.from_rdkit(**generated_mols[id]) for id in generated_mols]
    )
    return generated_batch


@dg.asset(io_manager_key="pytorch_io_manager")
def local_optimisation(
    local_optimiser: LocalOptimiserResource,
    docked_mols: dict,
    config: LocalOptimisationConfig,
) -> ConformerBatch:
    """Run local optimisation on the docked conformers."""
    docked_batch = ConformerBatch.from_data_list(
        [Conformer.from_rdkit(**docked_mols[id]) for id in docked_mols]
    )
    local_minima = run_optimisation(
        docked_batch,
        local_optimiser.get_optimiser(),
        config.batch_size,
        config.num_workers,
        config.device,
    )
    return local_minima


@dg.asset(
    io_manager_key="pytorch_io_manager",
)
def global_optimisation(
    global_optimiser: GlobalOptimiserResource,
    conformers: ConformerBatch,
    config: GlobalOptimisationConfig,
) -> ConformerBatch:
    """Run global optimisation on the generated conformers."""
    global_minima = run_optimisation(
        conformers,
        global_optimiser.get_optimiser(),
        config.batch_size,
        config.num_workers,
        config.device,
    )
    return global_minima


@dg.asset(io_manager_key="pandas_io_manager")
def aggregate_results(
    context: dg.AssetExecutionContext,
    input_df: pd.DataFrame,
    docked_mols: dict,
    local_optimisation: ConformerBatch,
    global_optimisation: ConformerBatch,
    config: OutputConfig,
) -> pd.DataFrame:
    """Aggregate results from local and global optimisation and save to output parquet file."""
    docked_batch = ConformerBatch.from_data_list(
        [Conformer.from_rdkit(**docked_mols[id]) for id in docked_mols]
    )
    if MOL_COL_NAME not in input_df.columns:
        input_df[MOL_COL_NAME] = input_df["mol_bytes"].apply(Chem.Mol)

    cfg = config.model_dump()
    run_dir = f"{DATA_DIR}/{context.run_id}"
    os.makedirs(run_dir, exist_ok=True)
    cfg["parquet_path"] = f"{run_dir}/{os.path.basename(cfg['parquet_path'])}"

    output_df = process_output(
        input_df, docked_batch, local_optimisation, global_optimisation, **cfg
    )
    return output_df


@dg.asset()
def plot_results(aggregate_results: pd.DataFrame):
    """Placeholder for plotting results."""
    # Placeholder for plotting results logic
    pass
