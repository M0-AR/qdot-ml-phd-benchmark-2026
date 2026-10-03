# Verification log (filled by `scripts/run_all.py`)

- `results/benchmark.json` — W1/W2/W3/W4 metrics + config hash + elapsed time.
- `results/live_manifest.json` — UTC timestamp + HTTP reachability for QDSD GitHub API,
  NIST QFlow PDR, QDarts PyPI + GitHub API. Status `live-verified` if any reachable.
- `results/example_csd.png`, `results/kpca_clusters.png`,
  `results/hidden_pattern_asymmetry.png` — visual checks.
- `pytest -q` — physics shapes + supervised smoke test.

No file was hand-tuned after measurement: all numbers come from executed code.
Re-run: `PYTHONPATH=src python scripts/run_all.py` or `docker compose up qdotml`.
