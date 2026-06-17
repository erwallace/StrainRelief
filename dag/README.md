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

### Running Dagster

Start the Dagster UI web server:

```bash
dg dev
```

Open http://localhost:3000 in your browser to see the project.

## Learn more

To learn more about this template and Dagster in general:

- [Dagster Documentation](https://docs.dagster.io/)
- [Dagster University](https://courses.dagster.io/)
- [Dagster Slack Community](https://dagster.io/slack)
