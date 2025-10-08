"""
Package enabling use of Intel GPUs (XPUs) with PyTorch Lightning.

Licensed under the version 2.0 of the Apache License (Apache-2.0):
- https://www.apache.org/licenses/LICENSE-2.0.html

This package adds to, and modifies, the lightning package of PyTorch Lightning
as detailed in the modules:
- lightning_xpu.pytorch.accelerators.xpu
- lightning_xpu.pytorch.strategies.ddp
- lightning_xpu.pytorch.strategies.fsdp
- lightning_xpu.pytorch.strategies.model_parallel
- lightning_xpu.pytorch.trainer.setup
- lightning_xpu.pytorch.trainer.connectors.accelerator_connector
- lightning_xpu.fabric.strategies.ddp
- lightning_xpu.fabric.strategies.fsdp
- lightning_xpu.fabric.strategies.model_parallel
- lightning_xpu.fabric.utilities.distributed
The package organisation under lightning_xpu mirrors the organisation of
under lightning of PyTorch Lightning.

PyTorch Lightning is licensed under version 2.0 of the Apache License:
- https://www.apache.org/licenses/LICENSE-2.0.html
"""
import sys

from lightning.pytorch.accelerators import AcceleratorRegistry
from lightning.fabric.utilities.registry import _register_classes
from lightning.pytorch.accelerators.accelerator import Accelerator
import lightning_xpu.fabric
import lightning_xpu.pytorch
from lightning_xpu.pytorch.accelerators.xpu import XPUAccelerator

_register_classes(AcceleratorRegistry, "register_accelerators", sys.modules[__name__], Accelerator)
