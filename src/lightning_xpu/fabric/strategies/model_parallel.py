"""
Module enabling use of Intel GPUs (XPUs) with PyTorch Lightning.

On import, this module substitutes modified versions of one of the
methods of the class
lightning.fabric.strategies.model_parallel.ModelParallelStrategy
of PyTorch Lightning, to include handling of XPUs:
- barrier():
  modified to allow "xccl" and "ccl" as backend for distributed processing.

Modified method based on the original method
in the lightning package of PyTorch Lightning.

PyTorch Lightning is licensed under version 2.0 of the Apache License:
- https://www.apache.org/licenses/LICENSE-2.0.html
"""
from typing import Any

import lightning_xpu.fabric.utilities.distributed

from lightning.fabric.strategies import ModelParallelStrategy

#
# Modified method for lightning.fabric.strategies.ModelParallelStrategy.
#

def _xpu_barrier(self, *args: Any, **kwargs: Any) -> None:
    if not _distributed_is_initialized():
        return
    if torch.distributed.get_backend() in ["nccl", "xccl", "ccl"]:
        torch.distributed.barrier(device_ids=[self.root_device.index])
    else:
        torch.distributed.barrier()

ModelParallelStrategy.barrier = _xpu_barrier
