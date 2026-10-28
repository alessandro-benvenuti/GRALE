#!/usr/bin/env python3
"""Verify a module-backed SCRATCH environment and optionally exercise its GPU."""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    scratch = Path(os.environ["SCRATCH"]).resolve()
    prefix = Path(sys.prefix).resolve()
    if prefix == scratch or scratch not in prefix.parents or sys.prefix == sys.base_prefix:
        raise SystemExit(f"Expected an active virtual environment inside SCRATCH, found {prefix}")
    expected = os.environ["GRALE_TORCH_VERSION"]
    import torch
    if torch.__version__.split("+")[0] != expected:
        raise SystemExit(f"Expected module PyTorch {expected}, found {torch.__version__}")
    if prefix in Path(torch.__file__).resolve().parents:
        raise SystemExit("PyTorch was installed into the venv instead of inherited from its module")
    packages = ("lightning", "pytorch-lightning", "torchmetrics", "wandb", "h5py",
                "vtk", "PyYAML", "numpy", "scipy", "networkx", "joblib", "matplotlib")
    for module in ("lightning.pytorch", "wandb", "h5py", "vtkmodules.vtkIOXML", "scipy", "yaml"):
        importlib.import_module(module)
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    importlib.import_module("GRALE.main")
    result = {"python": sys.version, "executable": sys.executable, "prefix": str(prefix),
              "platform": platform.platform(), "torch": torch.__version__,
              "torch_file": str(torch.__file__), "cuda_runtime": torch.version.cuda,
              "packages": {name: importlib.metadata.version(name) for name in packages}}
    if args.gpu:
        if not torch.cuda.is_available():
            raise SystemExit("No CUDA device is available in the GPU smoke allocation")
        x = torch.randn(64, 64, device="cuda", requires_grad=True)
        x.square().mean().backward()
        torch.cuda.synchronize()
        if not bool(torch.isfinite(x.grad).all()):
            raise SystemExit("CUDA backward produced non-finite gradients")
        result["gpu"] = torch.cuda.get_device_name(0)
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
