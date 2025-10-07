"""
Module enabling use of Intel GPUs (XPUs) with PyTorch Lightning.

On import, this module substitutes modified versions of some of the
methods of the class
lightning.pytorch.strategies.model_parallel.ModelParallelStrategy
of PyTorch Lightning, to include handling of XPUs:
- _get_process_group_backend():
  modified to set "xccl" (first choice) or "ccl" as process-group backend
  for "xpu" device;
- barrier():
  modified to allow "xccl" and "ccl" as backend for distributed processing;
- _setup_distributed():
  modified to call modified version of _init_dist_connection(), and so
  set environment variables used to determine local rank and
  local world size when using XPU devices and CCL backend.

Modified methods are based on the original methods
in the lightning package of PyTorch Lightning.

PyTorch Lightning is licensed under version 2.0 of the Apache License:
- https://www.apache.org/licenses/LICENSE-2.0.html
"""
from lightning_xpu.lightning.pytorch.accelerators.xpu import XPUAccelerator
from lightning_xpu.lightning.pytorch.strategies.fsdp import _xpu_barrier
from lightning_xpu.lightning.fabric.utilities.distributed import (
        _xpu_get_process_group_backend,
        _xpu_init_dist_connection,
        )

from lightning.fabric.utilities.seed import reset_seed
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_3
from lightning.pytorch.strategies import ModelParallelStrategy

#
# Modifications to lightning.pytorch.strategies.ModelParallelStrategy
#

ModelParallelStrategy._get_process_group_backend = _xpu_get_process_group_backend
ModelParallelStrategy.barrier = _xpu_barrier


def _xpu_setup_distributed(self) -> None:
    super().setup_environment()
    reset_seed()
    self.set_world_ranks()
    self._process_group_backend = self._get_process_group_backend()
    assert self.cluster_environment is not None
    kwargs: dict[str, Any] = {"timeout": self._timeout}
    if _TORCH_GREATER_EQUAL_2_3:
        kwargs["device_id"] = self.root_device if self.root_device.type != "cpu" else None
    _xpu_init_dist_connection(self.cluster_environment, self._process_group_backend, **kwargs)

ModelParallelStrategy._setup_distributed = _xpu_setup_distributed
