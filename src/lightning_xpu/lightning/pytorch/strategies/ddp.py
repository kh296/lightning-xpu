"""
Module enabling use of Intel GPUs (XPUs) with PyTorch Lightning.

On import, this module substitutes modified versions of some of the
methods of the class
lightning.pytorch.strategies.ddp.DDPStrategy:
of PyTorch Lightning, to include handling of XPUs:
- barrier():
  modified to allow "xccl" and "ccl" as backend for distributed processing;
- _get_process_group_backend():
  modified to set "xccl" (first choice) or "ccl" as process-group backend
  for "xpu" device;
- _setup_model():
  modified to handle "xpu" device;
- setup_distributed():
  modified to call modified version of _init_dist_connection(), and so
  set environment variables used to determine local rank and
  local world size when using XPU devices and CCL backend.

Modified methods are based on the original methods
in the lightning package of PyTorch Lightning.

PyTorch Lightning is licensed under version 2.0 of the Apache License:
- https://www.apache.org/licenses/LICENSE-2.0.html
"""
import os
from contextlib import nullcontext
from typing import Optional

from lightning_xpu.lightning.pytorch.accelerators.xpu import (
        XPUAccelerator,
        _get_all_visible_xpu_devices
        )
from lightning_xpu.lightning.fabric.utilities.distributed import (
        _xpu_get_process_group_backend,
        _xpu_init_dist_connection,
        )

import torch
from torch.nn import Module
from torch.nn.parallel.distributed import DistributedDataParallel

from lightning.fabric.utilities.distributed import _distributed_is_initialized
from lightning.fabric.utilities.seed import reset_seed
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_3
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.strategies.ddp import _DDP_FORK_ALIASES, log

#
# Modifications to lightning.pytorch.strategies.DDPStrategy
#

DDPStrategy._get_process_group_backend = _xpu_get_process_group_backend


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


def _xpu_setup_distributed(self) -> None:
    log.debug(f"{self.__class__.__name__}: setting up distributed...")
    reset_seed()
    self.set_world_ranks()
    self._process_group_backend = self._get_process_group_backend()
    assert self.cluster_environment is not None
    kwargs: dict[str, Any] = {"timeout": self._timeout}
    if _TORCH_GREATER_EQUAL_2_3:
        kwargs["device_id"] = self.root_device if self.root_device.type != "cpu" else None
    _xpu_init_dist_connection(self.cluster_environment, self._process_group_backend, **kwargs)

DDPStrategy.setup_distributed = _xpu_setup_distributed
