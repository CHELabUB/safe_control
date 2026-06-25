# Examples — shared utilities

This folder holds runnable demos plus a small shared utility.

## `run_registry.py` — run registry (shared)

A standalone, dependency-free helper for **organizing example outputs into per-run
folders and de-duplicating runs by their configuration**. It is generic (no
project-specific assumptions, standard library only) and reusable by any example.

### What it does
- Each *run* is one set of outputs identified by a configuration dict.
- Runs are saved as `<output_dir>/run_001/`, `run_002/`, … each containing a
  `config.json` (the run's input configuration).
- An index `<output_dir>/register.csv` lists every run with a timestamp, a config
  **fingerprint** (a hash of the inputs), selected key columns, and the config path.
- Because the fingerprint is computed over the configuration only, you can ask
  whether a configuration has already been computed and **skip recomputing** it.

### API
```python
from run_registry import RunRegistry

reg = RunRegistry(output_dir, key_columns=['model', 'tau'])  # extra register columns

match = reg.find_match(config)            # -> RunInfo | None  (same-config run?)
if match is not None:
    print(f"already computed as {match.name} at {match.path}")
else:
    run = reg.create_run(config)          # makes output/run_NNN/ + writes config.json
    #   ... write figures / result files into run.path ...
    reg.commit(run, columns={'model': 'idm', 'tau': 1.0})   # append a register.csv row
```

- `config` is any JSON-serializable dict of **inputs only** (exclude results — they
  should not change the fingerprint). Floats are rounded before hashing so
  re-runs are matched despite float-repr noise.
- `RunInfo` has `.name` (e.g. `run_002`), `.path` (absolute folder), `.fingerprint`.
- `key_columns` are the extra `register.csv` columns; populate them via the
  `columns=` argument to `commit()`.
- Run folder naming is configurable: `RunRegistry(..., run_prefix='run_', pad=3)`.

### Typical integration in an example `main()`
```python
out_dir = os.path.join(os.path.dirname(__file__), 'output')
run_config = build_run_config(args)                 # resolved inputs only
reg = RunRegistry(out_dir, key_columns=['model', 'tau', 'note'])

match = reg.find_match(run_config)
if match is not None and not args.force:
    print(f"Configuration already computed as '{match.name}' at {match.path}")
    return
run = reg.create_run(run_config)
... produce figures/results into run.path ...
reg.commit(run, columns={'model': args.model, 'tau': args.tau, 'note': args.note})
```

### Importing it from another example
`examples/` is not a package, so add this folder to `sys.path` (the same pattern the
examples already use for local modules) before importing:
```python
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from run_registry import RunRegistry
```

### Notes
- Dedup is on **configuration**, not on plotting/formatting code. Cosmetic figure
  changes do not change the fingerprint, so re-running a matched config is skipped;
  pass a `--force` flag (or clear the run folder) to regenerate figures.
- `register.csv`'s header is preserved across invocations, so appended rows stay
  aligned even if `key_columns` are reordered later.

### Used by
- `car_following/run_car_following.py` — see `car_following/README.md` for the
  end-to-end example (writes to `car_following/output/`).
