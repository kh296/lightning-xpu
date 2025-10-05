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
- _setup_model(): modified to handle "xpu" device;
- _setup_distributed():
  modified to set default values for selected environment variables
  relevant to "xccl" and "ccl" distributed processing.

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

import torch
from torch.nn import Module
from torch.nn.parallel.distributed import DistributedDataParallel

from lightning.fabric.utilities.distributed import (
        _distributed_is_initialized,
        _init_dist_connection,
        )
from lightning.fabric.utilities.seed import reset_seed
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.strategies.ddp import _DDP_FORK_ALIASES, log as ddp_log

#
# Modifications to lightning.pytorch.strategies.DDPStrategy
#

# This is a modified version of lightning.fabric.utilities.distributed._get_default_process_group_backend_for_device()
def _xpu_get_default_process_group_backend_for_device(
        device: torch.device) -> str:
    if device.type == "cuda": return "nccl"
    if device.type == "xpu": return (
            "xccl" if torch.distributed.is_xccl_available() else "ccl")
    return "gloo"


def _xpu_get_process_group_backend(self) -> str:
            return self._process_group_backend or _xpu_get_default_process_group_backend_for_device(self.root_device)

DDPStrategy._get_process_group_backend = _xpu_get_process_group_backend


def _xpu_setup_model(self, model: Module) -> DistributedDataParallel:
    """Wraps the model into a :class:`~torch.nn.parallel.distributed.DistributedDataParallel` module."""
    device_ids = self.determine_ddp_device_ids()
    ddp_log.debug(f"setting up DDP model with device ids: {device_ids}, kwargs: {self._ddp_kwargs}")
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


def _xpu_setup_distributed(self) -> None:
    ddp_log.debug(f"{self.__class__.__name__}: setting up distributed...")
    reset_seed()
    self.set_world_ranks()
    self._process_group_backend = self._get_process_group_backend()
    assert self.cluster_environment is not None
    _init_dist_connection(self.cluster_environment, self._process_group_backend, timeout=self._timeout)
    if self._process_group_backend in ["xccl", "ccl"]:
        os.environ.setdefault("CCL_WORKER_OFFLOAD", "0")
        # https://www.intel.com/content/www/us/en/docs/oneccl/developer-guide-reference/2021-9/environment-variables.html
        os.environ.setdefault("CCL_ATL_TRANSPORT", "ofi")
        # https://uxlfoundation.github.io/oneCCL/env-variables.html
        os.environ.setdefault("CCL_ZE_IPC_EXCHANGE", "pidfd")
        # https://www.intel.com/content/www/us/en/developer/articles/technical/flattening-gpu-tile-hierarchy.html
        os.environ.setdefault("ZE_FLAT_DEVICE_HIERARCHY", "FLAT")
        mask = ",".join(str(idx) for idx in _get_all_visible_xpu_devices())
        os.environ.setdefault("ZE_AFFINITY_MASK", mask)

DDPStrategy.setup_distributed = _xpu_setup_distributed


def _xpu_barrier(self, name: Optional[str] = None) -> None:
    if not _distributed_is_initialized():
        return
    if torch.distributed.get_backend() in ["nccl", "xccl", "ccl"]:
        torch.distributed.barrier(device_ids=self.determine_ddp_device_ids())
    else:
        torch.distributed.barrier()

DDPStrategy.barrier = _xpu_barrier
