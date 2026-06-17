import dagster as dg


class InputConfig(dg.Config):
    """Config for preprocess_data -> to_mols_dict()."""

    parquet_path: str
    include_charged: bool = True
    id_col_name: str | None = None
    mol_col_name: str | None = None


class ConformerConfig(dg.Config):
    """Config for conformers -> generate_conformers()."""

    randomSeed: int = -1
    numConfs: int = 10
    maxAttempts: int = 200
    pruneRmsThresh: float = 0.1
    clearConfs: bool = False
    numThreads: int = 0


class LocalOptimisationConfig(dg.Config):
    """Config for local_optimisation -> run_optimisation()."""

    batch_size: int
    num_workers: int
    device: str


class GlobalOptimisationConfig(dg.Config):
    """Config for global_optimisation -> run_optimisation()."""

    batch_size: int
    num_workers: int
    device: str


class OutputConfig(dg.Config):
    """Config for aggregate_results -> process_output()."""

    threshold: float
    parquet_path: str
    save_batch: bool = False
    molecule_attr: str | None = None
    id_col_name: str | None = None
    mol_col_name: str | None = None
