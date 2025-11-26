"""
Module that defines XPUAccelerator: Accelerator subclass for Intel GPUs (XPUs).

The class XPUAccelerator has been adapted from the PyTorch Lightning class:
- lightning.pytorch.accelerators.cuda.CUDAAccelerator

PyTorch Lightning is licensed under version 2.0 of the Apache License:
- https://www.apache.org/licenses/LICENSE-2.0.html
"""

import os
from functools import lru_cache
from typing import Any, Dict, List, MutableSequence
from typing_extensions import override

import torch
try:
    import intel_extension_for_pytorch as ipex
except ModuleNotFoundError:
    pass

# PyTorch XCCL backend enabled with Intel Extension for PyTorch v2.8.10+xpu.
if not hasattr(torch.distributed, "is_xccl_available"):
    torch.distributed.is_xccl_available = lambda: False
if not torch.distributed.is_xccl_available():
    try:
        import oneccl_bindings_for_pytorch
    except ModuleNotFoundError:
        pass

import lightning
from lightning.fabric.utilities.types import _DEVICE
from lightning.pytorch.accelerators import Accelerator


class XPUAccelerator(Accelerator):
    """Accelerator for Intel GPUs (XPUs)."""

    @override
    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            MisconfigurationException:
                If the selected device is not an XPU.
        """
        if device.type != "xpu":
            raise MisconfigurationException(
                    f"Device should be XPU, got {device} instead")
        torch.xpu.set_device(device)

        # Set default values for environment variables
        # relevant when using XPU devices in parallel processing.
        os.environ.setdefault("CCL_WORKER_OFFLOAD", "0")
        # https://www.intel.com/content/www/us/en/docs/oneccl/developer-guide-reference/2021-9/environment-variables.html
        os.environ.setdefault("CCL_ATL_TRANSPORT", "ofi")
        # https://uxlfoundation.github.io/oneCCL/env-variables.html
        os.environ.setdefault("CCL_ZE_IPC_EXCHANGE", "pidfd")
        # https://www.intel.com/content/www/us/en/developer/articles/technical/flattening-gpu-tile-hierarchy.html
        os.environ.setdefault("ZE_FLAT_DEVICE_HIERARCHY", "FLAT")
        mask = ",".join(str(idx) for idx in _get_all_visible_xpu_devices())
        os.environ.setdefault("ZE_AFFINITY_MASK", mask)

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
        torch.xpu.empty_cache()

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
        if isinstance(devices, int):
          devices = range(devices)
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

    @staticmethod
    @override
    def name() -> str:
        return "xpu"

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: Any) -> None:
        accelerator_registry.register(
            cls.name(),
            cls,
            description=cls.__class__.__name__,
        )


@lru_cache(1)
def num_xpu_devices() -> int:
    """Return the number of available XPU devices."""
    return torch.xpu.device_count()


def _get_all_visible_xpu_devices() -> List[int]:
    """
    Return a list of all visible XPU devices.
    The devices returned depend on the values set for the environment
    variables ``ZE_FLAT_DEVICE_HIERARCHY`` and ``ZE_AFFINITY_MASK``
    """
    return list(range(num_xpu_devices()))
