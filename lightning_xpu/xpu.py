import os
from functools import lru_cache
from typing import Any, Dict, List, MutableSequence, Union
from typing_extensions import override

import torch
try:
    import intel_extension_for_pytorch as ipex
except ModuleNotFoundError:
    pass

try:
    import oneccl_bindings_for_pytorch
except ModuleNotFoundError:
    pass

from torch.nn import Module
from torch.nn.parallel.distributed import DistributedDataParallel

import lightning.pytorch as pl
from lightning.fabric.plugins.environments import (
    ClusterEnvironment,
    LightningEnvironment,
    LSFEnvironment,
    MPIEnvironment,
    SLURMEnvironment,
    TorchElasticEnvironment,
)
from lightning.fabric.utilities.device_parser import _determine_root_gpu_device
from lightning.fabric.utilities.distributed import _init_dist_connection
from lightning.fabric.utilities.imports import _IS_INTERACTIVE
from lightning.fabric.utilities.seed import reset_seed
from lightning.fabric.utilities.types import _DEVICE
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning.pytorch.trainer import setup, Trainer
from lightning.pytorch.trainer.connectors.accelerator_connector import (
        _AcceleratorConnector)
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn
from lightning.pytorch.strategies import (
        DDPStrategy,
        SingleDeviceStrategy,
        Strategy
        )
from lightning.pytorch.strategies.ddp import log as ddp_log


class XPUAccelerator(Accelerator):
    """Support for a hypothetical XPU, optimized for large-scale machine learning."""

    @staticmethod
    @override
    def parse_devices(devices: Any) -> Any:
        # Put parsing logic here how devices can be passed into the Trainer
        # via the `devices` argument
        from lightning.fabric.utilities.device_parser import (
                _check_data_type, _check_unique,
                _normalize_parse_gpu_string_input
                )

        # Check that devices param is None, Int, String or Sequence of Ints
        _check_data_type(devices)

        # Handle the case when no GPUs are requested
        if (devices is None or (isinstance(devices, int) and devices == 0)
                or str(devices).strip() in ("0", "[]")):
            return None

        # We know the user requested GPUs therefore if some of the
        # requested GPUs are not available an exception is thrown.
        gpus = _normalize_parse_gpu_string_input(devices)
        if isinstance(gpus, (MutableSequence, tuple)):
            gpus = list(gpus)

        if not gpus:
            raise MisconfigurationException(
                    "GPUs requested but none available.")

        all_available_gpus = _get_all_visible_xpu_devices()
        if -1 == gpus:
            return all_available_gpus
        elif isinstance(gpus, int):
            gpus = list(range(gpus))

        # Check that GPUs are unique.
        # Duplicate GPUs are not supported by the backend.
        _check_unique(gpus)

        for gpu in gpus:
            if gpu not in all_available_gpus:
                raise MisconfigurationException(
                    f"You requested gpu: {gpus}\n"
                    f"But your machine only has: {all_available_gpus}"
                )

        return gpus

    @override
    @staticmethod
    def get_parallel_devices(devices: Any) -> Any:
        # Here, convert the device indices to actual device objects
        return [torch.device("xpu", idx) for idx in devices]

    @override
    @staticmethod
    def auto_device_count() -> int:
        # Return a value for auto-device selection when `Trainer(devices="auto")`
        return num_xpu_devices()

    @override
    @staticmethod
    def is_available() -> bool:
        return torch.xpu.is_available()

    @override
    def get_device_stats(self, device: _DEVICE) -> Dict[str, Any]:
        # Return optional device statistics for loggers
        return torch.xpu.memory_stats(device)

    @override
    def setup_device(self, device: torch.device) -> None:
        pass

    @override
    def teardown(self) -> None:
        pass

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: Any) -> None:
        accelerator_registry.register(
            "xpu",
            cls,
            description=cls.__class__.__name__,
        )


@lru_cache(1)
def num_xpu_devices() -> int:
    """Returns the number of available CUDA devices.

    Unlike :func:`torch.cuda.device_count`, this function does its best not to create a CUDA context for fork support,
    if the platform allows it.

    """
    return torch.xpu.device_count()


def _get_all_visible_xpu_devices() -> List[int]:
    """Returns a list of all visible Intel XPU devices.

    Devices masked by the environment variabale ``ZE_AFFINITY_MASK`` won't be returned here. For example, assume you
    have 8 physical GPUs. If ``ZE_AFFINITY_MASK="1,3,6"``, then this function will return the list ``[0, 1, 2]``
    because these are the three visible GPUs after applying the mask ``ZE_AFFINITY_MASK``.

    """
    return list(range(num_xpu_devices()))


def _xpu_log_device_info(trainer: "pl.Trainer") -> None:
    if XPUAccelerator.is_available():
        gpu_available = True
        gpu_type = " (xpu)"
    else:
        gpu_available = False
        gpu_type = ""

    gpu_used = isinstance(trainer.accelerator, (XPUAccelerator))
    rank_zero_info(
            f"GPU available: {gpu_available}{gpu_type}, used: {gpu_used}")

    num_xpu_cores = trainer.num_devices if isinstance(
            trainer.accelerator, XPUAccelerator) else 0
    
    rank_zero_info(f"XPU available: {XPUAccelerator.is_available()}, "
            f"using: {num_xpu_cores} XPU "
            + ("core" if 1 == num_xpu_cores else "cores"))

setup._log_device_info = _xpu_log_device_info


@staticmethod
def _xpu_choose_auto_accelerator() -> str:
    """Choose the accelerator type (str) based on availability."""
    if XPUAccelerator.is_available():
        return "xpu"
    return "cpu"

_AcceleratorConnector._choose_auto_accelerator = _xpu_choose_auto_accelerator


@staticmethod
def _xpu_choose_gpu_accelerator_backend() -> str:
    if XPUAccelerator.is_available():
        return "xpu"
    raise MisconfigurationException("No supported gpu backend found!")

_AcceleratorConnector._choose_cpu_accelerator_backend = (
        _xpu_choose_gpu_accelerator_backend)


def _xpu_choose_and_init_cluster_environment(self) -> ClusterEnvironment:

    if not "SLURM_NTASKS_PER_NODE" in os.environ:
        os.environ["SLURM_NTASKS_PER_NODE"] = str(len(self._devices_flag))
    if "SLURM_NNODES" in os.environ:
        self._num_nodes_flag = int(os.environ["SLURM_NNODES"])
    else:
        os.environ["SLURM_NNODES"] = int(self._num_nodes_flag)
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
        if len(self._parallel_devices) <= 1:
            if isinstance(self._accelerator_flag, XPUAccelerator) or (
                    isinstance(self._accelerator_flag, str)
                    and self._accelerator_flag in ("xpu")
            ):
                device = _determine_root_gpu_device(self._parallel_devices)
            else:
                device = "cpu"
            # TODO: lazy initialized device, then here could be self._strategy_flag = "single_device"
            return SingleDeviceStrategy(device=device)  # type: ignore

        return "ddp"

_AcceleratorConnector._choose_strategy = _xpu_choose_strategy


def _xpu_get_default_process_group_backend_for_device(
        device: torch.device) -> str:
    return "ccl" if device.type == "xpu" else "gloo"


def _xpu_get_process_group_backend(self) -> str:
            return self._process_group_backend or _xpu_get_default_process_group_backend_for_device(self.root_device)

DDPStrategy._get_process_group_backend = _xpu_get_process_group_backend


def _xpu_setup_model(self, model: Module) -> DistributedDataParallel:
    """Wraps the model into a :class:`~torch.nn.parallel.distributed.DistributedDataParallel` module."""
    device_ids = self.determine_ddp_device_ids()
    ddp_log.debug(f"setting up DDP model with device ids: {device_ids}, kwargs: {self._ddp_kwargs}")
    # https://pytorch.org/docs/stable/notes/cuda.html#id5
    ctx = torch.xpu.stream(torch.xpu.Stream()) if device_ids is not None else nullcontext()
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
    os.environ.setdefault("CCL_WORKER_OFFLOAD", "0")
    # https://www.intel.com/content/www/us/en/docs/oneccl/developer-guide-reference/2021-9/environment-variables.html
    os.environ.setdefault("CCL_ATL_TRANSPORT", "ofi")
    # https://www.intel.com/content/www/us/en/developer/articles/technical/flattening-gpu-tile-hierarchy.html
    os.environ.setdefault("ZE_FLAT_DEVICE_HIERARCHY", "FLAT")
    mask = ",".join(str(idx) for idx in _get_all_visible_xpu_devices())
    os.environ.setdefault("ZE_AFFINITY_MASK", mask)

DDPStrategy.setup_distributed = _xpu_setup_distributed
