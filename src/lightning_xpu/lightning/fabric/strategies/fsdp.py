"""
Module enabling use of Intel GPUs (XPUs) with PyTorch Lightning.

On import, this module substitutes modified versions of some of the
methods of the class
lightning.fabric.strategies.fsdp.FSDPStrategy:
of PyTorch Lightning, to include handling of XPUs:
- barrier():
modified to allow "xccl" and "ccl" as backend for distributed processing;
- setup_environment():
modified to set "xpu" as device type for "xccl" or "ccl" as
process-group backend;
- _get_process_group_backend():
modified to set "xccl" (first choice) or "ccl" as process-group backend
for "xpu" device (same as
lightning.fabric.strategies.ddp.DDPStrategy._get_process_group_backend()).

Modified methods are based on the original methods
in the lightning package of PyTorch Lightning.

PyTorch Lightning is licensed under version 2.0 of the Apache License:
- https://www.apache.org/licenses/LICENSE-2.0.html
"""
from typing import Any

from lightning_xpu.lightning.pytorch.accelerators.xpu import XPUAccelerator
from lightning_xpu.lightning.fabric.strategies.ddp import _xpu_get_process_group_backend

from lightning.fabric.utilities.distributed import _distributed_is_initialized
from lightning.fabric.utilities.seed import reset_seed
from lightning.fabric.strategies import FSDPStrategy

#
# Modifications to lightning.fabric.strategies.FSDPStrategy
#
def _xpu_barrier(self, *args: Any, **kwargs: Any) -> None:
    if not _distributed_is_initialized():
        return
    if torch.distributed.get_backend() in ["nccl", "xccl", "ccl"]:
        torch.distributed.barrier(device_ids=[self.root_device.index])
    else:
        torch.distributed.barrier()

FSDPStrategy.barrier = _xpu_barrier


def _xpu_setup_environment(self) -> None:
    super().setup_environment()
    self._setup_distributed()

    # if 'device_mesh' in the `_fsdp_kwargs` is provided as a tuple, update it into the `DeviceMesh` object here
    if isinstance(self._fsdp_kwargs.get("device_mesh"), tuple):
        from torch.distributed.device_mesh import init_device_mesh

        device_type = ("xpu" if self._process_group_backend in ["xccl", "ccl"]
                       else "cuda")
        self._fsdp_kwargs["device_mesh"] = init_device_mesh(device_type, self._fsdp_kwargs["device_mesh"])

FSDPStrategy.setup_environment = _xpu_setup_environment

# Function _xpu_get_process_group_backend()
# defined in modifications to lightning.fabric.strategies.DDPStrategy
FSDPStrategy._get_process_group_backend = _xpu_get_process_group_backend
