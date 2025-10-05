"""
Module enabling use of Intel GPUs (XPUs) with PyTorch Lightning.

On import, this module substitutes modified versions of some of the
methods of the class
lightning.pytorch.strategies.fsdp.FSDPStrategy:
of PyTorch Lightning, to include handling of XPUs:
- methods substituted for class lightning.pytorch.strategies.fsdp.FSDPStrategy:
  - barrier():
    modified to allow "xccl" and "ccl" as backend for distributed processing;
  - setup_environment():
    modified to set "xpu" as device type for "xccl" or "ccl" as
    process-group backend;
  - _get_process_group_backend():
    modified to set "xccl" (first choice) or "ccl" as process-group backend
    for "xpu" device (same as
    lightning.pytorch.strategies.ddp.DDPStrategy._get_process_group_backend()).

Modified methods are based on the original methods
in the lightning package of PyTorch Lightning.

PyTorch Lightning is licensed under version 2.0 of the Apache License:
- https://www.apache.org/licenses/LICENSE-2.0.html
"""
from typing import Optional

from lightning_xpu.lightning.pytorch.accelerators.xpu import XPUAccelerator
from lightning_xpu.lightning.pytorch.strategies.ddp import _xpu_get_process_group_backend

from lightning.fabric.utilities.distributed import (
        _distributed_is_initialized,
        _init_dist_connection,
        )
from lightning.fabric.utilities.seed import reset_seed
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.strategies.fsdp import log as fsdp_log

#
# Modifications to lightning.pytorch.strategies.FSDPStrategy
#
def _xpu_barrier(self, name: Optional[str] = None) -> None:
    if not _distributed_is_initialized():
        return
    if torch.distributed.get_backend() in ["nccl", "xccl", "ccl"]:
        torch.distributed.barrier(device_ids=self._determine_device_ids())
    else:
        torch.distributed.barrier()

FSDPStrategy.barrier = _xpu_barrier


def _xpu_setup_environment(self) -> None:
    super(type(self), self).setup_environment()
    fsdp_log.debug(f"{self.__class__.__name__}: setting up distributed...")
    reset_seed()

    # determine which process we are and world size
    self.set_world_ranks()

    self._process_group_backend = self._get_process_group_backend()
    assert self.cluster_environment is not None 
    kwargs: dict[str, Any] = {"timeout": self._timeout}
    if _TORCH_GREATER_EQUAL_2_3:
        kwargs["device_id"] = self.root_device if self.root_device.type != "cpu" else None
    _init_dist_connection(self.cluster_environment, self._process_group_backend, **kwargs)

    # if 'device_mesh' in the `kwargs` is provided as a tuple, update it into the `DeviceMesh` object here
    if isinstance(self.kwargs.get("device_mesh"), tuple):
        from torch.distributed.device_mesh import init_device_mesh

        device_type = ("xpu" if self._process_group_backend in ["xccl", "ccl"]
                       else "cuda")
        self.kwargs["device_mesh"] = init_device_mesh(device_type, self.kwargs["device_mesh"])

FSDPStrategy.setup_environment = _xpu_setup_environment

# Function _xpu_get_process_group_backend()
# defined in modifications to lightning.pytorch.strategies.DDPStrategy
FSDPStrategy._get_process_group_backend = _xpu_get_process_group_backend
