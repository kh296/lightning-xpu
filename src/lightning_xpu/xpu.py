import os
from contextlib import nullcontext
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
from lightning.pytorch.accelerators import (
        Accelerator,
        CPUAccelerator,
        CUDAAccelerator,
        MPSAccelerator,
        XLAAccelerator,
        )
from lightning.pytorch.trainer import setup, Trainer
from lightning.pytorch.trainer.connectors.accelerator_connector import (
        _AcceleratorConnector)
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.imports import _habana_available_and_importable
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn
from lightning.pytorch.strategies import (
        DDPStrategy,
        SingleDeviceStrategy,
        Strategy
        )
from lightning.pytorch.strategies.ddp import log as ddp_log


class XPUAccelerator(Accelerator):
    """Accelerator for Intel GPUs (XPUs)."""

    @override
    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            MisconfigurationException:
                If the selected device is not an Intel GPU (XPU).
        """
        if device.type != "xpu":
            raise MisconfigurationException(
                    f"Device should be XPU, got {device} instead")
        torch.xpu.set_device(device)

    @override
    def get_device_stats(self, device: _DEVICE) -> Dict[str, Any]:
        """Gets stats for the given GPU device.

        Args:
            device: GPU device for which to get stats

        Returns:
            A dictionary mapping the metrics to their values.

        Raises:
            FileNotFoundError:
                If xpu-smi installation not found
        """
        return torch.xpu.memory_stats(device)

    @override
    def teardown(self) -> None:
        pass

    # Parsing code largely adapted from:
    # lightning.pytorch.utilities.device_parser._parse_gpu_ids(),
    # and functions called from there.
    @staticmethod
    @override
    def parse_devices(devices: Any) -> Any:
        """Accelerator device parsing logic."""
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

        # If requested GPUs are not available, throw an exception.
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
        # Duplicate GPUs are not supported by the backend.
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
        """Gets parallel devices for the Accelerator."""
        return [torch.device("xpu", idx) for idx in devices]

    @override
    @staticmethod
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        return num_xpu_devices()

    @override
    @staticmethod
    def is_available() -> bool:
        return torch.xpu.is_available()

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
    """Return the number of available Intel GPU devices."""
    return torch.xpu.device_count()


def _get_all_visible_xpu_devices() -> List[int]:
    """
    Return a list of all visible Intel GPU devices.
    The devices returned depend on the values set for the environment
    variables ``ZE_FLAT_DEVICE_HIERARCHY`` and ``ZE_AFFINITY_MASK``
    """
    return list(range(num_xpu_devices()))

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
    rank_zero_info(f"GPU available: {gpu_available}{gpu_type}, used: {gpu_used}")

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

_AcceleratorConnector._choose_cpu_accelerator_backend = (
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

#
# Modifications to lightning.pytorch.strategies.DDPStrategy
#

# This is a modified version of lightning.fabric.utilities.distributed._get_default_process_group_backend_for_device()
def _xpu_get_default_process_group_backend_for_device(
        device: torch.device) -> str:
    if device.type == "cuda": return "nccl"
    if device.type == "xpu": return "ccl"
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
            ctx = torch.xpu.stream(torch.cuda.Stream())
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
    if "ccl" == elf._process_group_backend:
        os.environ.setdefault("CCL_WORKER_OFFLOAD", "0")
        # https://www.intel.com/content/www/us/en/docs/oneccl/developer-guide-reference/2021-9/environment-variables.html
        os.environ.setdefault("CCL_ATL_TRANSPORT", "ofi")
        # https://www.intel.com/content/www/us/en/developer/articles/technical/flattening-gpu-tile-hierarchy.html
        os.environ.setdefault("ZE_FLAT_DEVICE_HIERARCHY", "FLAT")
        mask = ",".join(str(idx) for idx in _get_all_visible_xpu_devices())
        os.environ.setdefault("ZE_AFFINITY_MASK", mask)

DDPStrategy.setup_distributed = _xpu_setup_distributed
