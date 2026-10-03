"""Download helper: fetch small public manifests / samples (not bulk binaries)."""
from qdotml.live_verify import check
import json, pathlib
if __name__ == "__main__":
    m = check()
    pathlib.Path("results/live_manifest.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))
