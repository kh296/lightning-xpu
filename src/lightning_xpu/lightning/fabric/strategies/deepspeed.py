"""
Module enabling use of Intel GPUs (XPUs) with PyTorch Lightning.

On import, this module substitutes modified versions of some of the
methods of the class
lightning.fabric.strategies.model_parallel.DeepSpeedStrategy
of PyTorch Lightning, to include handling of XPUs:
- setup_environment():
  modified to accpt XPUAccelerator as device class.

Modified methods are based on the original methods
in the lightning package of PyTorch Lightning.

PyTorch Lightning is licensed under version 2.0 of the Apache License:
- https://www.apache.org/licenses/LICENSE-2.0.html
"""
import lightning_xpu.lightning.fabric.utilities.distributed
from lightning_xpu.lightning.pytorch.accelerators.xpu import XPUAccelerator

from lightning.pytorch.accelerators import CUDAAccelerator
from lightning.fabric.strategies import DeepSpeedStrategy

#
# Modified methods for lightning.fabric.strategies.DeepSpeedStrategy.
#
def _xpu_setup_environment(self) -> None:
    if not isinstance(self.accelerator, (CUDAAccelerator, XPUAccelerator)):
        raise RuntimeError(
            f"The DeepSpeed strategy is only supported on CUDA GPUs and on Intel  GPUs (XPUs) but `{self.accelerator.__class__.__name__}`" " is used."
        )
    super(type(self), self).setup_environment()

DeepSpeedStrategy.setup_environment = _xpu_setup_environment
