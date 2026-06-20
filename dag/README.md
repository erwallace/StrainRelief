# dag

## Getting started

### Installing dependencies

This project depends on the parent [`strain-relief`](../) package via a local
editable source (`[tool.uv.sources]` in `pyproject.toml`), so changes to the
parent's `src/strain_relief` are picked up immediately.

> **Why not a plain `uv sync`?** `strain-relief` pulls in `torch-cluster`, a
> native extension with **no prebuilt wheel for arm64 macOS**. It must be
> compiled from source, which requires `torch` to already be present and the
> build to run with isolation disabled. The steps below handle that ordering.

Ensure [`uv`](https://docs.astral.sh/uv/) is installed (see their
[installation docs](https://docs.astral.sh/uv/getting-started/installation/)).

```bash
# 1. Create the venv on Python 3.11 (strain-relief requires ==3.11.*)
uv venv --python 3.11

# 2. Install torch first — torch-cluster needs it at build time.
#    2.8.0 is the version torch-cluster 1.6.3 compiles cleanly against.
uv pip install "torch==2.8.0"

# 3. Build tools required because the install runs with --no-build-isolation
uv pip install hatchling editables

# 4. Install this project (editable) + strain-relief, building torch-cluster
#    from source against the installed torch.
uv pip install -e . --no-build-isolation

# 5. Install the dev tooling (dg CLI + webserver) needed to run Dagster,
#    declared as the `dev` extra in pyproject.toml.
uv pip install -e ".[dev]" --no-build-isolation
```

Then activate the virtual environment:

| OS | Command |
| --- | --- |
| MacOS | ```source .venv/bin/activate``` |
| Windows | ```.venv\Scripts\activate``` |

#### Troubleshooting

- **`ModuleNotFoundError: No module named 'torch'`** when building
  `torch-cluster` — run step 2 before step 4, and keep `--no-build-isolation`.
- **`libc++abi: terminating ... std::length_error`** on `import torch_cluster`
  — the cached `torch-cluster` binary was built against a different `torch`.
  Force a clean rebuild against the installed torch:

  ```bash
  uv pip install --force-reinstall --no-build-isolation --no-deps --no-cache "torch-cluster==1.6.3"
  ```

#### Verify the install

```bash
python -c "import dag, strain_relief, dagster, torch_cluster; print('OK')"
dagster definitions validate -m dag.definitions   # or: dg check defs
```

### Running the pipeline

The pipeline is exposed as a single job, `all_assets`, covering every asset:
`input_df → docked_mols → conformers → local_optimisation → global_optimisation →
aggregate_results → plot_results`.

Run from the `dag/` directory with the venv activated. `run.yaml` is a complete example
config (input path, calculator/optimiser selection, per-asset settings).

**Headless (one-shot):**

```bash
dg launch --job all_assets --config run.yaml
```

> Use `--config` (a YAML file), **not** `--config-file`. Launch the whole job with
> `--job all_assets`, not `--assets "*"` — the `"*"` selector path pulls in an antlr
> grammar that conflicts with the antlr version `omegaconf`/`hydra` pin.

**Interactive UI:**

```bash
dg dev   # then open http://localhost:3000
```

In the UI, select the assets → **Materialize** → paste the contents of `run.yaml` into the
Launchpad config editor → Launch.

#### Changing the input

Point `input_df` / `docked_mols` at your own parquet in `run.yaml` (paths are relative to
the `dag/` working directory):

```yaml
ops:
  input_df:
    config:
      parquet_path: ../data/example_ligboundconf_input.parquet
  docked_mols:
    config:
      parquet_path: ../data/example_ligboundconf_input.parquet   # same file
```

To switch calculator/optimiser, edit the `resources:` block (e.g. `mace` → `mmff94`).

### Where outputs are stored

Each asset's return value is persisted by its assigned IO manager (configured in
`resources.py`). The IO manager for an asset determines how *its output* is stored; a
downstream asset loads that input via the **upstream** asset's manager.

| Asset | IO manager | Location |
| --- | --- | --- |
| `input_df` | `pandas_io_manager` (DuckDB) | table in `../data/db.duckdb` |
| `docked_mols` | default (filesystem, pickle) | Dagster instance storage (see note) |
| `conformers` | `pytorch_io_manager` | `../data/conformers/result.pt` |
| `local_optimisation` | `pytorch_io_manager` | `../data/local_optimisation/result.pt` |
| `global_optimisation` | `pytorch_io_manager` | `../data/global_optimisation/result.pt` |
| `aggregate_results` | `pandas_io_manager` (DuckDB) | table in `../data/db.duckdb` |
| `plot_results` | default (returns `None`) | — |

The `pytorch_io_manager` writes to `{base_path}/{asset}/{output}.pt` with `base_path=../data`;
the `pandas_io_manager` writes each DataFrame to a table inside the DuckDB file at
`../data/db.duckdb`.

**Final result:** in addition to the IO-managed outputs above, `aggregate_results` calls
`process_output`, which writes the merged results parquet to the path in
`OutputConfig.parquet_path` — by default **`../data/example_output.parquet`** (columns
include `id`, `local_min_e`, `global_min_e`, `ligand_strain`, `passes_strain_filter`). This
is the human-readable end product.

> **Note on the default IO manager / Dagster home.** With no `DAGSTER_HOME` set, `dg launch`
> uses a temporary instance directory (auto-created, ephemeral), so `docked_mols` and run
> metadata do not persist between runs. To keep them, point `DAGSTER_HOME` at a directory:
> `export DAGSTER_HOME=$PWD/.dagster_home`.

## Learn more

To learn more about this template and Dagster in general:

- [Dagster Documentation](https://docs.dagster.io/)
- [Dagster University](https://courses.dagster.io/)
- [Dagster Slack Community](https://dagster.io/slack)
