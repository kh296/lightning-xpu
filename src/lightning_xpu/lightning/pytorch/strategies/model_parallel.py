"""
Module enabling use of Intel GPUs (XPUs) with PyTorch Lightning.

On import, this module substitutes modified versions of some of the
methods of the class
lightning.pytorch.strategies.model_parallel.ModelParallelStrategy
of PyTorch Lightning, to include handling of XPUs:
- barrier():
  modified to allow "xccl" and "ccl" as backend for distributed processing
  (same as lightning.pytorch.strategies.fsdp.FSDPStrategy.barrier()):

Modified methods are based on the original methods
in the lightning package of PyTorch Lightning.

PyTorch Lightning is licensed under version 2.0 of the Apache License:
- https://www.apache.org/licenses/LICENSE-2.0.html
"""
from lightning_xpu.lightning.pytorch.accelerators.xpu import XPUAccelerator
from lightning_xpu.lightning.pytorch.strategies.fsdp import _xpu_barrier

from lightning.pytorch.strategies import ModelParallelStrategy

#
# Modifications to lightning.pytorch.strategies.ModelParallelStrategy
#

# Function _xpu_fdsp_barrier()
# defined in modifications to lightning.pytorch.strategies.FDSPStrategy
ModelParallelStrategy.barrier = _xpu_barrier
