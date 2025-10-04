"""
Module for enabling use of Intel GPUs (XPUs) with PyTorch Lightning.

On import, this module substitutes modified versions of some of the
methods of the class
lightning.pytorch.trainer.connectors.accelerator_connector._AcceleratorConnector
of PyTorch Lightning, to include handling of XPUs:
- _check_strategy_and_fallback():
  modified to allow "fsdp" as strategy with "xpu" as accelerator type;
- _choose_auto_accelerator():
  modified to allow automatic selection of "xpu" as accelerator type;
- _choose_gpu_accelerator_backend():
  modified to allow selection of "xpu" as gpu backend;
- _choose_and_init_cluster_environment():
  modified to set default values for selected Slurm environment variables;
- _choose_strategy():
  modified to allow automatic selection of "single_device" or "ddp"
  as strategy for "xpu" device.

Modified methods are based on the original functions methods
in the lightning package of PyTorch Lightning.

PyTorch Lightning is licensed under version 2.0 of the Apache License:
- https://www.apache.org/licenses/LICENSE-2.0.html
"""
import os
from typing import Union

from lightning_xpu.lightning.pytorch.accelerators.xpu import XPUAccelerator

from lightning.fabric.plugins.environments import (
    ClusterEnvironment,
    LightningEnvironment,
    LSFEnvironment,
    MPIEnvironment,
    SLURMEnvironment,
    TorchElasticEnvironment,
)
from lightning.fabric.utilities.device_parser import _determine_root_gpu_device

from lightning.fabric.utilities.imports import _IS_INTERACTIVE
from lightning.pytorch.accelerators import (
        CUDAAccelerator,
        MPSAccelerator,
        XLAAccelerator,
        )
from lightning.pytorch.trainer.connectors.accelerator_connector import (
        _AcceleratorConnector)
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.imports import _habana_available_and_importable
from lightning.pytorch.strategies import (
        FSDPStrategy,
        SingleDeviceStrategy,
        Strategy
        )
from lightning.pytorch.strategies.ddp import _DDP_FORK_ALIASES

#
# Modifications to lightning.pytorch.trainer.connectors.accelerator_connector._AcceleratorConnector
#

@staticmethod
def _xpu_choose_auto_accelerator() -> str:
    """Choose the accelerator type (str) based on availability."""
    if _habana_available_and_importable():
        from lightning_habana import HPUAccelerator

        if HPUAccelerator.is_available():
            return "hpu"

    if XLAAccelerator.is_available():
        return "tpu"
    if MPSAccelerator.is_available():
        return "mps"
    if CUDAAccelerator.is_available():
        return "cuda"
    if XPUAccelerator.is_available():
        return "xpu"
    return "cpu"

_AcceleratorConnector._choose_auto_accelerator = _xpu_choose_auto_accelerator


@staticmethod
def _xpu_choose_gpu_accelerator_backend() -> str:
    if MPSAccelerator.is_available():
        return "mps"
    if CUDAAccelerator.is_available():
        return "cuda"
    if XPUAccelerator.is_available():
        return "xpu"
    raise MisconfigurationException("No supported gpu backend found!")

_AcceleratorConnector._choose_gpu_accelerator_backend = (
        _xpu_choose_gpu_accelerator_backend)


def _xpu_choose_and_init_cluster_environment(self) -> ClusterEnvironment:
    # Expect that setting Slurm environment variables
    # won't cause problems outside of Slurm environment...
    if "SLURM_NTASKS_PER_NODE" in os.environ:
        num_processes = min(len(self._devices_flag),
                int(os.environ["SLURM_NTASKS_PER_NODE"]))
        self._devices_flag = self._devices_flag[: num_processes]
        self._parallel_devices = self._parallel_devices[: num_processes]
    os.environ["SLURM_NTASKS_PER_NODE"] = str(len(self._devices_flag))

    if "SLURM_NNODES" in os.environ:
        self._num_nodes_flag = int(os.environ["SLURM_NNODES"])
    else:
        os.environ["SLURM_NNODES"] = str(int(self._num_nodes_flag))
    os.environ["SLURM_NTASKS"] = str(int(os.environ["SLURM_NNODES"])
            * int(os.environ["SLURM_NTASKS_PER_NODE"]))

    if isinstance(self._cluster_environment_flag, ClusterEnvironment):
        return self._cluster_environment_flag
    for env_type in (
        # TorchElastic has the highest priority since it can also be used inside SLURM
        TorchElasticEnvironment,
        SLURMEnvironment,
        LSFEnvironment,
        MPIEnvironment,
    ):
        if env_type.detect():
            return env_type()
    return LightningEnvironment()

_AcceleratorConnector._choose_and_init_cluster_environment = (
        _xpu_choose_and_init_cluster_environment)


def _xpu_choose_strategy(self) -> Union[Strategy, str]:
    if _habana_available_and_importable():
        from lightning_habana import HPUAccelerator

        if self._accelerator_flag == "hpu" or isinstance(self._accelerator_flag, HPUAccelerator):
            if self._parallel_devices and len(self._parallel_devices) > 1:
                from lightning_habana import HPUParallelStrategy

                return HPUParallelStrategy.strategy_name

            from lightning_habana import SingleHPUStrategy

            return SingleHPUStrategy(device=torch.device("hpu"))
    if self._accelerator_flag == "hpu" and not _habana_available_and_importable():
        raise ImportError(
            "You asked to run with HPU but you are missing a required dependency."
            " Please run `pip install lightning-habana` or seek further instructions"
            " in https://github.com/Lightning-AI/lightning-Habana/."
        )

    if self._accelerator_flag == "tpu" or isinstance(self._accelerator_flag, XLAAccelerator):
        if self._parallel_devices and len(self._parallel_devices) > 1:
            return XLAStrategy.strategy_name
        # TODO: lazy initialized device, then here could be self._strategy_flag = "single_xla"
        return SingleDeviceXLAStrategy(device=self._parallel_devices[0])
    if self._num_nodes_flag > 1:
        return "ddp"
    if len(self._parallel_devices) <= 1:
        if isinstance(self._accelerator_flag, (CUDAAccelerator, MPSAccelerator, XPUAccelerator)) or (
            isinstance(self._accelerator_flag, str) and self._accelerator_flag in ("cuda", "gpu", "mps", "xpu")
        ):
            device = _determine_root_gpu_device(self._parallel_devices)
        else:
            device = "cpu"
        # TODO: lazy initialized device, then here could be self._strategy_flag = "single_device"
        return SingleDeviceStrategy(device=device)  # type: ignore
    if len(self._parallel_devices) > 1 and _IS_INTERACTIVE:
        return "ddp_fork"
    return "ddp"

_AcceleratorConnector._choose_strategy = _xpu_choose_strategy


def _xpu_check_strategy_and_fallback(self) -> None:
    """Checks edge cases when the strategy selection was a string input, and we need to fall back to a different
    choice depending on other parameters or the environment."""
    # current fallback and check logic only apply to user pass in str config and object config
    # TODO this logic should apply to both str and object config
    strategy_flag = "" if isinstance(self._strategy_flag, Strategy) else self._strategy_flag

    if (
        strategy_flag in FSDPStrategy.get_registered_strategies() or type(self._strategy_flag) is FSDPStrategy
    ) and self._accelerator_flag not in ("cuda", "gpu", "xpu"):
        raise ValueError(
            f"The strategy `{FSDPStrategy.strategy_name}` requires a GPU accelerator, but got:"
            f" {self._accelerator_flag}"
        )
    if strategy_flag in _DDP_FORK_ALIASES and "fork" not in torch.multiprocessing.get_all_start_methods():
        raise ValueError(
            f"You selected `Trainer(strategy='{strategy_flag}')` but process forking is not supported on this"
            f" platform. We recommend `Trainer(strategy='ddp_spawn')` instead."
        )
    if strategy_flag:
        self._strategy_flag = strategy_flag

_AcceleratorConnector._check_strategy_and_fallback = _xpu_check_strategy_and_fallback
