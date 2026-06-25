import os
from pathlib import Path
from typing import Annotated, Literal

import dagster as dg
import pandas as pd
import torch
from neural_optimiser.calculators import FAIRChemCalculator, MACECalculator, MMFF94Calculator
from neural_optimiser.calculators.base import Calculator
from neural_optimiser.optimisers import BFGS
from neural_optimiser.optimisers.base import Optimiser
from pydantic import Field

from .configs import (
    BFGSConfig,
    CalculatorConfig,
    MACEConfig,
)

DATA_DIR = str(Path(__file__).resolve().parents[4] / "data")
_CALCULATOR_CACHE: dict[str, Calculator] = {}


def _get_or_build_calculator(config: CalculatorConfig) -> Calculator:
    """Load-once calculator cache; keyed on resolved config."""
    key = config.model_dump_json()
    if key not in _CALCULATOR_CACHE:
        _CALCULATOR_CACHE[key] = config.build()
    return _CALCULATOR_CACHE[key]


class CalculatorResource(dg.ConfigurableResource):
    """Holds the calculator selection; `get_calculator()` is load-once (see cache above)."""

    spec: CalculatorConfig

    def get_calculator(self) -> Calculator:
        return _get_or_build_calculator(self.spec)


class OptimiserResource(dg.ConfigurableResource):
    """Base: builds a BFGS optimiser and attaches the shared calculator.

    Mirrors compute_strain.py, which instantiates one calculator and assigns it to both
    optimisers.
    """

    calculator: CalculatorResource
    spec: OptimiserConfig

    def get_calculator(self) -> Calculator:
        return self.calculator.get_calculator()

    def get_optimiser(self) -> Optimiser:
        optimiser = self.spec.build()
        optimiser.calculator = self.get_calculator()
        return optimiser


class LocalOptimiserResource(OptimiserResource):
    """Local optimiser (hydra default.yaml: local_optimiser); defaults bound at registration."""


class GlobalOptimiserResource(OptimiserResource):
    """Global optimiser (hydra default.yaml: global_optimiser); defaults bound at registration."""


class PyTorchIOManager(dg.ConfigurableIOManager):
    """Stores tensors at {base_path}/{run_id}/{asset}.pt (run-scoped)."""

    base_path: str

    def handle_output(self, context: dg.OutputContext, obj: object):
        output_path = f"{self.base_path}/{context.run_id}/{context.step_key}.pt"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(obj, output_path)

    def load_input(self, context: dg.InputContext) -> object:
        out = context.upstream_output
        return torch.load(f"{self.base_path}/{out.run_id}/{out.step_key}.pt")


class ParquetIOManager(dg.ConfigurableIOManager):
    """Stores DataFrames at {base_path}/{run_id}/{step}.parquet (run-scoped)."""

    base_path: str

    def handle_output(self, context: dg.OutputContext, obj: pd.DataFrame):
        output_path = f"{self.base_path}/{context.run_id}/{context.step_key}.parquet"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        obj.to_parquet(output_path)

    def load_input(self, context: dg.InputContext) -> pd.DataFrame:
        out = context.upstream_output
        return pd.read_parquet(f"{self.base_path}/{out.run_id}/{out.step_key}.parquet")


@dg.definitions
def resources():
    calculator = CalculatorResource(  # default
        spec=MACEConfig(model_paths="../models/MACE_SPICE2_NEUTRAL.model"),
    )
    return dg.Definitions(
        executor=dg.in_process_executor,  # required for load-once calculator cache
        resources={
            # IO Managers specify how the OUTPUT of an asset is handled.
            # The INPUT is handled by the upstream asset's output manager.
            "io_manager": dg.FilesystemIOManager(),  # default if not specified.
            "pandas_io_manager": ParquetIOManager(base_path=DATA_DIR),
            "pytorch_io_manager": PyTorchIOManager(base_path=DATA_DIR),
            # Calculator + optimisers (config-driven; calculator shared).
            "calculator": calculator,
            "local_optimiser": LocalOptimiserResource(
                calculator=calculator, spec=BFGSConfig(fmax=0.50, fexit=5)
            ),
            "global_optimiser": GlobalOptimiserResource(
                calculator=calculator, spec=BFGSConfig(fmax=0.05, fexit=25)
            ),
        },
    )
