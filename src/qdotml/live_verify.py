"""Live public-data verification: fetch metadata (not bulk binaries) from
QDSD (GitHub), NIST QFlow (data.nist.gov PDR), QDarts (PyPI).
Writes results/live_manifest.json with timestamps + reachability + versions.
Falls back gracefully offline; benchmark itself runs on synthetic + cached data.
"""
from __future__ import annotations
import json, datetime, pathlib
try:
    import requests
    HAVE_REQ = True
except Exception:
    HAVE_REQ = False

URLS = {
    "qdsd_github_api": "https://api.github.com/repos/3it-inpaqt/qdsd-dataset",
    "nist_qflow_pdr": "https://data.nist.gov/od/id/66492819760D3FF6E05324570681BA721894",
    "qdarts_pypi": "https://pypi.org/pypi/QDarts/json",
    "qdarts_github_api": "https://api.github.com/repos/qplai/QDarts",
}

def check():
    out = {"timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
           "checks": {}}
    if not HAVE_REQ:
        out["status"] = "requests-missing"
        return out
    for k, u in URLS.items():
        try:
            r = requests.get(u, timeout=15, headers={"User-Agent": "qdotml-benchmark/1.0"})
            info = {"url": u, "http": r.status_code, "reachable": bool(r.status_code < 400)}
            if k == "qdarts_pypi" and r.status_code == 200:
                try:
                    info["latest_version"] = r.json()["info"]["version"]
                except Exception as e:
                    info["latest_version"] = f"parse-error: {e}"
            if "github_api" in k and r.status_code == 200:
                try:
                    j = r.json()
                    info["stars"] = j.get("stargazers_count")
                    info["updated_at"] = j.get("updated_at")
                except Exception:
                    pass
            out["checks"][k] = info
        except Exception as e:
            out["checks"][k] = {"url": u, "reachable": False, "error": str(e)[:300]}
    out["status"] = "live-verified" if any(v.get("reachable") for v in out["checks"].values()) else "offline-fallback"
    return out

if __name__ == "__main__":
    m = check()
    p = pathlib.Path("results/live_manifest.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))
