import dagster as dg
import pandas as pd
from neural_optimiser.conformers import Conformer, ConformerBatch
from strain_relief.conformers import generate_conformers
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
from .resources import GlobalOptimiserResource, LocalOptimiserResource


@dg.asset
def preprocess_data(config: InputConfig) -> dict:
    """Load input data and convert to MolsDict."""
    df = load_parquet(parquet_path=config.parquet_path)
    docked_mols: MolsDict = to_mols_dict(df, **config.model_dump())
    return docked_mols


@dg.asset(io_manager_key="pytorch_io_manager")
def conformers(preprocess_data: dict, config: ConformerConfig) -> ConformerBatch:
    """Generate conformers for the input molecules."""
    generated_mols = generate_conformers(preprocess_data, **config.model_dump())
    generated_batch: ConformerBatch = ConformerBatch.cat(
        [ConformerBatch.from_rdkit(**generated_mols[id]) for id in generated_mols]
    )
    return generated_batch


@dg.asset(io_manager_key="pytorch_io_manager")
def local_optimisation(
    local_optimiser: LocalOptimiserResource,
    preprocess_data: ConformerBatch,
    config: LocalOptimisationConfig,
) -> ConformerBatch:
    """Run local optimisation on the docked conformers."""
    docked_batch = ConformerBatch.from_data_list(
        [Conformer.from_rdkit(**preprocess_data[id]) for id in preprocess_data]
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
    preprocess_data: ConformerBatch,
    local_optimisation: ConformerBatch,
    global_optimisation: ConformerBatch,
    config: OutputConfig,
) -> pd.DataFrame:
    """Aggregate results from local and global optimisation and save to output parquet file."""
    docked_batch = ConformerBatch.from_data_list(
        [Conformer.from_rdkit(**preprocess_data[id]) for id in preprocess_data]
    )
    output_df = process_output(
        pd.DataFrame(), docked_batch, local_optimisation, global_optimisation, **config.model_dump()
    )
    return output_df


@dg.asset()
def plot_results(aggregate_results: pd.DataFrame):
    """Placeholder for plotting results."""
    # Placeholder for plotting results logic
    pass
