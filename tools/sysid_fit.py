"""python tools/sysid_fit.py runs/<sysid run> [--write]   -> model_fit.json, fig_sysid_fit.png (+ config/model.json)"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ftrack.sysid import identify_run, to_model_json
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--write", action="store_true")
a = ap.parse_args()
r = identify_run(a.run)
print(json.dumps({k: {kk: vv for kk, vv in v.items()} for k, v in r["axes"].items()}, indent=1))
if a.write:
    p = Path(__file__).resolve().parent.parent / "config" / "model.json"
    p.write_text(json.dumps(to_model_json(r, Path(a.run).name), indent=1)); print("wrote", p)
