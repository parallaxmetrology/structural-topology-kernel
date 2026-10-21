"""Download data that is too big for the repository.

    python tools/fetch_data.py baselines    # HOG (7,200) + Haralick (196) features for UHCSDB, 22 MB

The baseline features were computed from the UHCS micrographs (CC BY 3.0 US, NIST
handle 11256/940) with the recipe in benchmarks/uhcsdb_baselines.py.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "baselines": ("https://github.com/parallaxmetrology/structural-topology-kernel/releases/download/data-v1/baselines_har_hog.npz",
                  ROOT / "data" / "uhcsdb" / "baselines_har_hog.npz", "3c53238fe1d71c48041dd6f33d58e38c2ece3358f2ca499ffecb0383c26fcfb7"),
}


def main(names):
    for name in names or FILES:
        url, dest, sha = FILES[name]
        if not dest.exists():
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, dest)
        if hashlib.sha256(dest.read_bytes()).hexdigest() != sha:
            print(f"checksum mismatch for {dest.name}; delete it and retry", file=sys.stderr)
            return 1
        print(f"ok: {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
