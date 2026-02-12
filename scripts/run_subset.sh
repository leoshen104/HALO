#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
from halo.config.loader import load_config
from halo.utils.provenance import write_manifest
from halo.utils.seeding import set_deterministic_seed

cfg = load_config('configs/default.json')
seed = set_deterministic_seed(cfg.runtime.seed)
path = write_manifest('artifacts/run_manifest.json', config=cfg.to_dict(), seed=seed)
print(f'Wrote {path}')
PY
