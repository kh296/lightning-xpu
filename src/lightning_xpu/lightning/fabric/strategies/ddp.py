"""
Module enabling use of Intel GPUs (XPUs) with PyTorch Lightning.

On import, this module substitutes modified versions of some of the
methods of the class
lightning.fabric.strategies.ddp.DDPStrategy:
of PyTorch Lightning, to include handling of XPUs:
- barrier():
  modified to allow "xccl" and "ccl" as backend for distributed processing;
- _get_process_group_backend():
  modified to set "xccl" (first choice) or "ccl" as default
  process-group backend for "xpu" device;
- setup_module(): modified to handle "xpu" device;
- _setup_distributed():
  modified to set default values for selected environment variables
  relevant to "xccl" and "ccl" distributed processing.

Modified methods are based on the original methods
in the lightning package of PyTorch Lightning.

PyTorch Lightning is licensed under version 2.0 of the Apache License:
- https://www.apache.org/licenses/LICENSE-2.0.html
"""
from contextlib import nullcontext
from typing import Any

import torch
from torch.nn import Module
from torch.nn.parallel.distributed import DistributedDataParallel

from lightning.fabric.utilities.distributed import _distributed_is_initialized
from lightning.fabric.strategies import DDPStrategy

#
# Modifications to lightning.fabric.strategies.DDPStrategy
#

# This is a modified version of lightning.fabric.utilities.distributed._get_default_process_group_backend_for_device()
def _xpu_get_default_process_group_backend_for_device(
        device: torch.device) -> str:
    """Return corresponding distributed backend for a given device."""
    device_backend_map = torch.distributed.Backend.default_device_backend_map
    if device.type in device_backend_map:
        return device_backend_map[device.type]
    if device.type == "xpu": return (
            "xccl" if torch.distributed.is_xccl_available() else "ccl")
    return "gloo"


def _xpu_get_process_group_backend(self) -> str:
            return self._process_group_backend or _xpu_get_default_process_group_backend_for_device(self.root_device)

DDPStrategy._get_process_group_backend = _xpu_get_process_group_backend


def _xpu_setup_module(self, module: Module) -> DistributedDataParallel:
    """Wraps the model into a :class:`~torch.nn.parallel.distributed.DistributedDataParallel` module."""
    device_ids = self._determine_ddp_device_ids()
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
        return DistributedDataParallel(module=module, device_ids=device_ids, **self._ddp_kwargs)

DDPStrategy.setup_module = _xpu_setup_module


def _xpu_barrier(self, *args: Any, **kwargs: Any) -> None:
    if not _distributed_is_initialized():
        return
    if torch.distributed.get_backend() in ["nccl", "xccl", "ccl"]:
        torch.distributed.barrier(device_ids=self._determine_ddp_device_ids())
    else:
        # Handle PyTorch bug where barrier() fails on CPU with "PrivateUse1HooksInterface" error
        try:
            torch.distributed.barrier()
        except RuntimeError as e:
            if "PrivateUse1HooksInterface" in str(e):
                # Fallback: Use all_reduce as barrier - all processes must participate
                # This achieves the same synchronization effect as barrier()
                dummy_tensor = torch.tensor(0.0, device=self.root_device)
                torch.distributed.all_reduce(dummy_tensor)
            else:
                raise

DDPStrategy.barrier = _xpu_barrier
