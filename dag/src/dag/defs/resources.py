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

# Root directory for run outputs. Each run writes under DATA_DIR/<run_id>/ (see the IO
# managers below and aggregate_results' final parquet), so runs never overwrite each other.
# Absolute (resolved from the repo layout) so it is independent of the launch directory.
# resources.py -> defs -> dag -> src -> dag(project) -> repo root; data/ sits at the repo root.
DATA_DIR = str(Path(__file__).resolve().parents[4] / "data")


class MACEConfig(dg.Config):
    kind: Literal["mace"] = "mace"
    model_paths: str
    device: str = "cpu"
    default_dtype: Literal["float32", "float64"] = "float32"

    def build(self) -> Calculator:
        return MACECalculator(
            model_paths=self.model_paths, device=self.device, default_dtype=self.default_dtype
        )


class MMFF94Config(dg.Config):
    kind: Literal["mmff94"] = "mmff94"

    def build(self) -> Calculator:
        return MMFF94Calculator()


class FAIRChemConfig(dg.Config):
    kind: Literal["fairchem"] = "fairchem"
    model_paths: str
    device: str = "cpu"
    task_name: str = "omol"
    default_dtype: Literal["float32", "float64"] = "float32"

    def build(self) -> Calculator:
        return FAIRChemCalculator(
            model_paths=self.model_paths,
            device=self.device,
            task_name=self.task_name,
            default_dtype=self.default_dtype,
        )


CalculatorConfig = Annotated[
    MACEConfig | MMFF94Config | FAIRChemConfig, Field(discriminator="kind")
]


class BFGSConfig(dg.Config):
    kind: Literal["bfgs"] = "bfgs"
    max_step: float = 0.04
    steps: int = 250
    fmax: float | None = None
    fexit: float | None = None

    def build(self) -> Optimiser:
        return BFGS(max_step=self.max_step, steps=self.steps, fmax=self.fmax, fexit=self.fexit)


class _BaseOptimiserConfig(dg.Config):
    """Placeholder so OptimiserConfig is a discriminated union (i.e. renders as a selector,
    uniform with CalculatorConfig). The base Optimiser is abstract, so this is never selected;
    it exists only until a second concrete optimiser (e.g. LBFGS) is added."""

    kind: Literal["base"] = "base"

    def build(self) -> Optimiser:
        raise NotImplementedError(
            "Select a concrete optimiser (e.g. bfgs); 'base' is a placeholder."
        )


OptimiserConfig = Annotated[BFGSConfig | _BaseOptimiserConfig, Field(discriminator="kind")]


# --------------------------------------------------------------------------------------
# Load-once calculator cache. The model load (torch.load) dominates step time, so we build
# the calculator at most once per process, keyed on the resolved config.
# --------------------------------------------------------------------------------------
_CALCULATOR_CACHE: dict[str, Calculator] = {}


def _get_or_build_calculator(config: CalculatorConfig) -> Calculator:
    key = config.model_dump_json()
    if key not in _CALCULATOR_CACHE:
        _CALCULATOR_CACHE[key] = config.build()
    return _CALCULATOR_CACHE[key]


# --------------------------------------------------------------------------------------
# Resources
# --------------------------------------------------------------------------------------
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
