"""
Module enabling use of Intel GPUs (XPUs) with PyTorch Lightning.

On import, this module substitutes modified versions of some of the
methods of the class
lightning.pytorch.strategies.model_parallel.DeepSpeedStrategy
of PyTorch Lightning, to include handling of XPUs:
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
from lightning_xpu.lightning.pytorch.accelerators.xpu import XPUAccelerator
from lightning_xpu.lightning.pytorch.strategies.ddp import (
        _xpu_get_process_group_backend)

from lightning.pytorch.accelerators import CUDAAccelerator
from lightning.pytorch.strategies import DeepSpeedStrategy

#
# Modifications to lightning.pytorch.strategies.DeepSpeedStrategy
#
def _xpu_setup_environment(self) -> None:
    if not isinstance(self.accelerator, (CUDAAccelerator, XPUAccelerator)):
        raise RuntimeError(
            f"The DeepSpeed strategy is only supported on CUDA GPUs and on Intel  GPUs (XPUs) but `{self.accelerator.__class__.__name__}`" " is used."
        )
    super(type(self), self).setup_environment()

DeepSpeedStrategy.setup_environment = _xpu_setup_environment

# Function _xpu_get_process_group_backend()
# defined in modifications to lightning.pytorch.strategies.DDPStrategy
DeepSpeedStrategy._get_process_group_backend = _xpu_get_process_group_backend
