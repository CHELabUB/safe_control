"""
Standalone run registry: organize result folders and de-duplicate runs by config.

Generic and reusable (no project-specific assumptions, standard library only). A
"run" is one set of outputs identified by a configuration dict. Runs are saved as

    <output_dir>/run_001/ , run_002/ , ...

each containing a `config.json` (the run's input configuration), and indexed in

    <output_dir>/register.csv

A run's *fingerprint* is a hash of its configuration (inputs only). `find_match`
reports whether a configuration has already been computed, so identical re-runs can
be skipped instead of recomputed.

Typical use
-----------
    reg = RunRegistry(output_dir, key_columns=['model', 'tau'])
    match = reg.find_match(config)
    if match is not None:
        print(f"already computed as {match.name} at {match.path}")
        # ... or reuse it in place: run = reg.overwrite_run(match, config)
    else:
        run = reg.create_run(config)         # makes output/run_NNN + config.json
        ... write figures/files into run.path ...
        reg.commit(run, columns={'model': 'idm', 'tau': 1.0})
"""

import os
import csv
import json
import hashlib
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RunInfo:
    """Handle for a single run."""
    name: str          # e.g. 'run_002'
    path: str          # absolute path to the run folder
    fingerprint: str   # config hash


def _canonical(obj):
    """Recursively round floats so hashing is stable across float repr noise."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: _canonical(obj[k]) for k in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    return obj


def config_fingerprint(config: dict, length: int = 12) -> str:
    """Stable short hash of a configuration dict (order-independent)."""
    blob = json.dumps(_canonical(config), sort_keys=True, separators=(',', ':'))
    return hashlib.sha1(blob.encode('utf-8')).hexdigest()[:length]


class RunRegistry:
    """Allocate, fingerprint, and index runs under a single output directory."""

    BASE_COLUMNS = ('run_name', 'timestamp', 'fingerprint')

    def __init__(self, output_dir, register_name='register.csv',
                 run_prefix='run_', pad=3, key_columns=()):
        """
        Args:
            output_dir: base directory for all runs and the register.
            register_name: CSV filename inside output_dir.
            run_prefix, pad: run folders are f'{run_prefix}{i:0{pad}d}'.
            key_columns: extra register columns (besides run_name/timestamp/
                fingerprint/config_path) populated from commit(columns=...).
        """
        self.output_dir = os.path.abspath(output_dir)
        self.register_path = os.path.join(self.output_dir, register_name)
        self.run_prefix = run_prefix
        self.pad = int(pad)
        self.key_columns = list(key_columns)
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Register reads
    # ------------------------------------------------------------------

    def _rows(self):
        if not os.path.exists(self.register_path):
            return []
        with open(self.register_path, newline='') as fh:
            return list(csv.DictReader(fh))

    def find_match(self, config: dict):
        """Return RunInfo of an existing run with the same config, else None."""
        fp = config_fingerprint(config)
        for row in self._rows():
            if row.get('fingerprint') == fp:
                name = row['run_name']
                return RunInfo(name, os.path.join(self.output_dir, name), fp)
        return None

    # ------------------------------------------------------------------
    # Run allocation / writing
    # ------------------------------------------------------------------

    def _next_index(self) -> int:
        """Next free run index, robust to register rows and stray folders."""
        idx = 0
        for row in self._rows():
            idx = max(idx, self._index_of(row.get('run_name', '')))
        if os.path.isdir(self.output_dir):
            for d in os.listdir(self.output_dir):
                if os.path.isdir(os.path.join(self.output_dir, d)):
                    idx = max(idx, self._index_of(d))
        return idx + 1

    def _index_of(self, name: str) -> int:
        if name.startswith(self.run_prefix):
            tail = name[len(self.run_prefix):]
            if tail.isdigit():
                return int(tail)
        return 0

    def create_run(self, config: dict, columns=None) -> RunInfo:
        """Allocate the next run folder and write its config.json (no register row
        yet — call commit() after the outputs are produced)."""
        name = f"{self.run_prefix}{self._next_index():0{self.pad}d}"
        path = os.path.join(self.output_dir, name)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'config.json'), 'w') as fh:
            json.dump(config, fh, indent=2)
        return RunInfo(name, path, config_fingerprint(config))

    def overwrite_run(self, match: RunInfo, config: dict) -> RunInfo:
        """Reuse an existing (matched) run folder in place instead of allocating a new
        one: rewrite its config.json and return its RunInfo. The caller regenerates the
        run's outputs (figures, series, results) into the same folder, and commit()
        refreshes the existing register row rather than appending a duplicate."""
        path = match.path
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'config.json'), 'w') as fh:
            json.dump(config, fh, indent=2)
        return RunInfo(match.name, path, config_fingerprint(config))

    def commit(self, run: RunInfo, columns=None) -> None:
        """Write a register row for `run` (replace-or-append by run_name).

        A row with the same run_name (e.g. from an overwrite_run) is replaced in place
        so re-committing a run does not create a duplicate entry; otherwise the row is
        added. Rows are written sorted by run_name index, so overriding a run keeps it in
        its natural position instead of moving it to the end. The header is preserved to
        avoid column drift across invocations.
        """
        columns = columns or {}
        header = list(self.BASE_COLUMNS) + self.key_columns + ['config_path']
        # Preserve an existing header; keep all other runs' rows as-is.
        existing_rows = []
        if os.path.exists(self.register_path):
            with open(self.register_path, newline='') as fh:
                existing = next(csv.reader(fh), None)
            if existing:
                header = existing
            existing_rows = [r for r in self._rows() if r.get('run_name') != run.name]
        row = {
            'run_name': run.name,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'fingerprint': run.fingerprint,
            'config_path': os.path.relpath(
                os.path.join(run.path, 'config.json'), self.output_dir),
        }
        for k in self.key_columns:
            row[k] = columns.get(k, '')

        all_rows = sorted(existing_rows + [row],
                          key=lambda r: self._index_of(r.get('run_name', '')))
        with open(self.register_path, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=header, extrasaction='ignore')
            writer.writeheader()
            for r in all_rows:
                writer.writerow(r)
