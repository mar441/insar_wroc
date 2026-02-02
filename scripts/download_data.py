import os
import sys
import requests

OWNER = os.getenv("GITHUB_OWNER", "mar441")
REPO = os.getenv("GITHUB_REPO", "insar_wroc")
RELEASE_TAG = os.getenv("RELEASE_TAG", "data-v1")
DATA_DIR = os.getenv("DATA_DIR", "/var/data")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

FILES = [
    "wroclaw_caly.csv",
    "wroclaw_geo.csv",
    "predictions_ml.csv",
    "anomaly_95.csv",
    "anomaly_99.csv",
]

def api_get(url):
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    return r

def download(url, out_path):
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    with requests.get(url, headers=headers, stream=True, timeout=600) as r:
        r.raise_for_status()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp = out_path + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, out_path)

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    api_url = f"https://api.github.com/repos/{OWNER}/{REPO}/releases/tags/{RELEASE_TAG}"
    rel = api_get(api_url).json()

    assets = {a["name"]: a for a in rel.get("assets", [])}

    missing = [fn for fn in FILES if fn not in assets]
    if missing:
        print("ERROR: Missing assets in release:", missing)
        print("Found assets:", list(assets.keys()))
        sys.exit(1)

    for fn in FILES:
        out_path = os.path.join(DATA_DIR, fn)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"OK exists: {out_path}")
            continue

        asset = assets[fn]
        print(f"Downloading {fn} -> {out_path}")

        download(asset["browser_download_url"], out_path)

    print("DONE. Data ready in:", DATA_DIR)

if __name__ == "__main__":
    main()
