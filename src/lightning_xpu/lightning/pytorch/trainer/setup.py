"""
Module for enabling use of Intel GPUs (XPUs) with PyTorch Lightning.

On import, this module substitutes modified versions of the function
lightning.pytorch.trainer.setup._log_device_info()
of PyTorch Lightning, to include handling of XPUs:
- function to recognise XPUAccelerator as representing a GPU,
  and to indicate whether available;

Modified function are based on the original function in the lightning
package of PyTorch Lightning.

PyTorch Lightning is licensed under version 2.0 of the Apache License:
- https://www.apache.org/licenses/LICENSE-2.0.html
"""
from lightning_xpu.lightning.pytorch.accelerators.xpu import XPUAccelerator

import lightning.pytorch as pl
from lightning.pytorch.accelerators import (
        CUDAAccelerator,
        MPSAccelerator,
        XLAAccelerator,
        )
from lightning.pytorch.trainer import setup
from lightning.pytorch.utilities.imports import _habana_available_and_importable
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn

#
# Modifications to lightning.pytorch.trainer.setup._log_device_info()
#

def _xpu_log_device_info(trainer: "pl.Trainer") -> None:
    if CUDAAccelerator.is_available():
        gpu_available = True
        gpu_type = " (cuda)"
    elif MPSAccelerator.is_available():
        gpu_available = True
        gpu_type = " (mps)"
    elif XPUAccelerator.is_available():
        gpu_available = True
        gpu_type = " (xpu)"
    else:
        gpu_available = False
        gpu_type = ""

    gpu_used = isinstance(trainer.accelerator,
                          (CUDAAccelerator, MPSAccelerator, XPUAccelerator))
    num_gpus = trainer.num_devices if gpu_used else 0
    rank_zero_info(f"GPU available: {gpu_available}{gpu_type}, using: "
                   f"{num_gpus} {'GPU' if 1 == num_gpus else 'GPUs'}{gpu_type}")

    num_tpu_cores = trainer.num_devices if isinstance(trainer.accelerator, XLAAccelerator) else 0
    rank_zero_info(f"TPU available: {XLAAccelerator.is_available()}, using: {num_tpu_cores} TPU cores")

    if _habana_available_and_importable():
        from lightning_habana import HPUAccelerator

        num_hpus = trainer.num_devices if isinstance(trainer.accelerator, HPUAccelerator) else 0
        hpu_available = HPUAccelerator.is_available()
    else:
        num_hpus = 0
        hpu_available = False
    rank_zero_info(f"HPU available: {hpu_available}, using: {num_hpus} HPUs")

    if (
        CUDAAccelerator.is_available()
        and not isinstance(trainer.accelerator, CUDAAccelerator)
        or MPSAccelerator.is_available()
        and not isinstance(trainer.accelerator, MPSAccelerator)
        or XPUAccelerator.is_available()
        and not isinstance(trainer.accelerator, XPUAccelerator)
    ):
        rank_zero_warn(
            "GPU available but not used. You can set it by doing `Trainer(accelerator='gpu')`.",
            category=PossibleUserWarning,
        )

    if XLAAccelerator.is_available() and not isinstance(trainer.accelerator, XLAAccelerator):
        rank_zero_warn("TPU available but not used. You can set it by doing `Trainer(accelerator='tpu')`.")

    if _habana_available_and_importable():
        from lightning_habana import HPUAccelerator

        if HPUAccelerator.is_available() and not isinstance(trainer.accelerator, HPUAccelerator):
            rank_zero_warn("HPU available but not used. You can set it by doing `Trainer(accelerator='hpu')`.")

setup._log_device_info = _xpu_log_device_info
