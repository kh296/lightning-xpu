"""
Module enabling use of Intel GPUs (XPUs) with PyTorch Lightning.

On import, this module substitutes modified versions of some of the
methods of the class
lightning.pytorch.strategies.model_parallel.ModelParallelStrategy
of PyTorch Lightning, to include handling of XPUs:
- barrier():
  modified to allow "xccl" and "ccl" as backend for distributed processing;

Modified methods are based on the original methods
in the lightning package of PyTorch Lightning.

PyTorch Lightning is licensed under version 2.0 of the Apache License:
- https://www.apache.org/licenses/LICENSE-2.0.html
"""
from typing import Optional

import lightning_xpu.lightning.fabric.utilities.distributed

from lightning.pytorch.strategies import ModelParallelStrategy

#
# Modified methods for lightning.pytorch.strategies.ModelParallelStrategy.
#
def _xpu_barrier(self, name: Optional[str] = None) -> None:
    if not _distributed_is_initialized():
        return
    if torch.distributed.get_backend() in ["nccl", "xccl", "ccl"]:
        torch.distributed.barrier(device_ids=self._determine_device_ids())
    else:
        torch.distributed.barrier()

ModelParallelStrategy.barrier = _xpu_barrier
