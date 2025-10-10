"""
Module enabling use of Intel GPUs (XPUs) with PyTorch Lightning.

On import, this module substitutes modified versions of some of the
methods of the class
lightning.pytorch.strategies.ddp.DDPStrategy
of PyTorch Lightning, to include handling of XPUs:
- barrier():
  modified to allow "xccl" and "ccl" as backend for distributed processing;
- _setup_model():
  modified to handle "xpu" device;

Modified methods are based on the original methods
in the lightning package of PyTorch Lightning.

PyTorch Lightning is licensed under version 2.0 of the Apache License:
- https://www.apache.org/licenses/LICENSE-2.0.html
"""
from contextlib import nullcontext
from typing import Optional

import lightning_xpu.fabric.utilities.distributed

import torch
from torch.nn import Module
from torch.nn.parallel.distributed import DistributedDataParallel

from lightning.fabric.utilities.distributed import _distributed_is_initialized
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.strategies.ddp import log

#
# Modified methods for lightning.pytorch.strategies.DDPStrategy.
#

def _xpu_setup_model(self, model: Module) -> DistributedDataParallel:
    """Wraps the model into a :class:`~torch.nn.parallel.distributed.DistributedDataParallel` module."""
    device_ids = self.determine_ddp_device_ids()
    log.debug(f"setting up DDP model with device ids: {device_ids}, kwargs: {self._ddp_kwargs}")
    if device_ids is None:
        ctx = nullcontext()
    else:
        if "cuda" == self.root_device.type:
            ctx = torch.cuda.stream(torch.cuda.Stream())
        elif "xpu" == self.root_device.type:
            ctx = torch.xpu.stream(torch.xpu.Stream())
        else: 
            ctx = nullcontext()
    with ctx:
        return DistributedDataParallel(module=model, device_ids=device_ids, **self._ddp_kwargs)

DDPStrategy._setup_model = _xpu_setup_model


def _xpu_barrier(self, name: Optional[str] = None) -> None:
    if not _distributed_is_initialized():
        return
    if torch.distributed.get_backend() in ["nccl", "xccl", "ccl"]:
        torch.distributed.barrier(device_ids=self.determine_ddp_device_ids())
    else:
        torch.distributed.barrier()

DDPStrategy.barrier = _xpu_barrier
