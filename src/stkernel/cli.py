"""stk measure IMAGE|FOLDER [-o out.csv]"""
from __future__ import annotations

import argparse
import json
import sys

from .core import measure, measure_many


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="stk", description="Structural Topology Kernel: 50 structural coordinates of a micrograph")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="one image (JSON) or a folder (CSV)")
    m.add_argument("path")
    m.add_argument("-o", "--out", help="CSV output for a folder")
    m.add_argument("--size", type=int, default=512)
    m.add_argument("--no-crop", action="store_true", help="don't remove a dark info bar at the bottom")
    a = ap.parse_args(argv)
    from pathlib import Path
    p = Path(a.path)
    if p.is_dir():
        df = measure_many(p, progress=True, size=a.size, crop_info_bar=not a.no_crop)
        if a.out:
            df.to_csv(a.out, index=False)
            print(f"{len(df)} images -> {a.out}")
        else:
            df.to_csv(sys.stdout, index=False)
    else:
        print(json.dumps(measure(p, size=a.size, crop_info_bar=not a.no_crop), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
