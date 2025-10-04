"""
Package enabling use of Intel GPUs (XPUs) with PyTorch Lightning.

Licensed under the version 2.0 of the Apache License (Apache-2.0):
- https://www.apache.org/licenses/LICENSE-2.0.html

This package adds to, and modifies, the lightning package of PyTorch Lightning
as detailed in the modules:
- lightning_xpu.lightning.pytorch.accelerators.xpu
- lightning_xpu.lightning.pytorch.strategies.ddp
- lightning_xpu.lightning.pytorch.strategies.fsdp
- lightning_xpu.lightning.pytorch.strategies.model_parallel
- lightning_xpu.lightning.pytorch.trainer.setup
- lightning_xpu.lightning.pytorch.trainer.connectors.accelerator_connector
The package organisation under lighning_xpu mirrors the organisation of
the package of PyTorch Lightning.

PyTorch Lightning is licensed under version 2.0 of the Apache License:
- https://www.apache.org/licenses/LICENSE-2.0.html
"""
import sys

from lightning.pytorch.accelerators import AcceleratorRegistry
from lightning.fabric.utilities.registry import _register_classes
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning_xpu.lightning.pytorch.accelerators.xpu import XPUAccelerator
import lightning_xpu.lightning

_register_classes(AcceleratorRegistry, "register_accelerators", sys.modules[__name__], Accelerator)
