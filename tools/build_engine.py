"""Compile every deploy/<run>/best.pt into a TensorRT FP16 engine (best.engine) for THIS GPU.

.venv/Scripts/python tools/build_engine.py [--runs final_det] [--force]

An engine only runs on the GPU model and TensorRT version that built it, so it is not shipped: build it once per
machine (5–8 min per model on an RTX 4050). ftrack/vision.py picks best.engine automatically when it exists.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def check_env():
    try:
        import torch
    except ImportError:
        sys.exit("torch is not installed in this Python (see README: install the CUDA build of torch first)")
    if not torch.cuda.is_available():
        sys.exit(f"torch {torch.__version__} has no CUDA: install the CUDA wheel (pip install torch --index-url https://download.pytorch.org/whl/cu126)")
    try:
        import tensorrt as trt
    except ImportError:
        sys.exit('TensorRT missing: pip install "tensorrt-cu12<11"   (CUDA-12 build; the default tensorrt 11 is CUDA-13 and crashes with cu12x torch)')
    major = int(trt.__version__.split(".")[0])
    cu = torch.version.cuda or ""
    if major >= 11 and cu.startswith("12"):
        sys.exit(f'TensorRT {trt.__version__} is a CUDA-13 build but torch uses CUDA {cu}: pip uninstall -y tensorrt tensorrt_cu13 tensorrt_cu13_bindings tensorrt_cu13_libs && pip install "tensorrt-cu12<11"')
    print(f"GPU {torch.cuda.get_device_name(0)} | torch {torch.__version__} (CUDA {cu}) | TensorRT {trt.__version__}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", default=str(ROOT / "models"))
    ap.add_argument("--runs", default="", help="comma-separated folder names (default: all)")
    ap.add_argument("--force", action="store_true", help="rebuild even if best.engine exists")
    a = ap.parse_args()
    check_env()
    from ultralytics import YOLO
    want = [r for r in a.runs.split(",") if r]
    for d in sorted(p for p in Path(a.deploy).iterdir() if (p / "tracker.json").exists() and (p / "best.pt").exists()):
        if want and d.name not in want:
            continue
        if (d / "best.engine").exists() and not a.force:
            print(f"[skip] {d.name}: best.engine exists (--force to rebuild)", flush=True)
            continue
        tc = json.loads((d / "tracker.json").read_text())
        t0 = time.time()
        print(f"[build] {d.name}: imgsz {tc['imgsz']}, FP16 ... (several minutes)", flush=True)
        p = YOLO(str(d / "best.pt")).export(format="engine", imgsz=tc["imgsz"], half=True, device=0, verbose=False)
        print(f"[done] {p}  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
