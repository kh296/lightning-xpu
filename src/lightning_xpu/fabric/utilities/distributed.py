"""
Module enabling use of Intel GPUs (XPUs) with PyTorch Lightning.

This module defines modified version of some of the functions of the module
lightning.fabric.utilities.distributed:
of PyTorch Lightning, to include handling of XPUs:
- _get_default_process_group_backend_for_device()
  modified to allow for both "xccl" and "ccl" as backend for "xpu" device;
- _init_dist_connection():
  modified to set environment variables used to determine local rank
  and local world size when using "xpu" devices and "xccl" or "ccl" backend.

Modified functions are based on the original functions
in the lightning package of PyTorch Lightning.

PyTorch Lightning is licensed under version 2.0 of the Apache License:
- https://www.apache.org/licenses/LICENSE-2.0.html
"""
import atexit
import os
from typing import Any, Optional

import torch
import lightning
from lightning.fabric.utilities.distributed import log
from lightning.fabric.utilities.rank_zero import rank_zero_info

#
# Modifications versions of functions
# defined in lightning.fabric.utilities.distributed.
#

def _xpu_get_default_process_group_backend_for_device(
        device: torch.device) -> str:
    """Return corresponding distributed backend for a given device."""
    device_backend_map = torch.distributed.Backend.default_device_backend_map
    if device.type in device_backend_map:
        return device_backend_map[device.type]
    if device.type == "xpu": return (
            "xccl" if torch.distributed.is_xccl_available() else "ccl")
    return "gloo"


def _xpu_init_dist_connection(
    cluster_environment: "ClusterEnvironment",
    torch_distributed_backend: str,
    global_rank: Optional[int] = None,
    world_size: Optional[int] = None,
    **kwargs: Any,
) -> None:
    """Utility function to initialize distributed connection by setting env variables and initializing the distributed
    process group.

    Args:
        cluster_environment: ``ClusterEnvironment`` instance
        torch_distributed_backend: Backend to use (includes `nccl` and `gloo`)
        global_rank: Rank of the current process
        world_size: Number of processes in the group
        kwargs: Kwargs for ``init_process_group``

    Raises:
        RuntimeError:
            If ``torch.distributed`` is not available

    """
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is not available. Cannot initialize distributed process group")
    if torch.distributed.is_initialized():
        log.debug("torch.distributed is already initialized. Exiting early")
        return
    global_rank = global_rank if global_rank is not None else cluster_environment.global_rank()
    world_size = world_size if world_size is not None else cluster_environment.world_size()
    os.environ["MASTER_ADDR"] = cluster_environment.main_address
    os.environ["MASTER_PORT"] = str(cluster_environment.main_port)

    # Set environment variables used to determine local rank
    # and local world size when using XPU devices and CCL backend.
    # Avoids warning:
    # |CCL_WARN| could not get local_idx/count from environment variables,
    # trying to get them from ATL
    if torch_distributed_backend in ["ccl", "xccl"]:
        local_world_size = torch.xpu.device_count()
        local_rank = cluster_environment.local_rank()
        ccl_process_launcher = os.environ.get("CCL_PROCESS_LAUNCHER", "hydra")
        if "hydra" == ccl_process_launcher:
            os.environ["MPI_LOCALNRANKS"] = str(local_world_size)
            os.environ["MPI_LOCALRANKID"] = str(local_rank)
        elif "torchrun" == ccl_process_launcher:
            os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)
            os.environ["LOCAL_RANK"] = str(local_rank)
        elif "none" == ccl_process_launcher:
            os.environ["CCL_LOCAL_SIZE"] = str(local_world_size)
            os.environ["CCL_LOCAL_RANK"] = str(local_rank)

    log.info(f"Initializing distributed: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}")
    torch.distributed.init_process_group(torch_distributed_backend, rank=global_rank, world_size=world_size, **kwargs)

    if torch_distributed_backend == "nccl":
        # PyTorch >= 2.4 warns about undestroyed NCCL process group, so we need to do it at program exit
        atexit.register(_destroy_dist_connection)

    # On rank=0 let everyone know training is starting
    rank_zero_info(
        f"{'-' * 100}\n"
        f"distributed_backend={torch_distributed_backend}\n"
        f"All distributed processes registered. Starting with {world_size} processes\n"
        f"{'-' * 100}\n"
    )


# Substitute functions modified in all modules where they're used.
modules = [
        lightning.fabric.strategies.ddp,
        lightning.fabric.strategies.deepspeed,
        lightning.fabric.strategies.fsdp,
        lightning.fabric.strategies.model_parallel,
        lightning.fabric.utilities.distributed,
        lightning.pytorch.strategies.ddp,
        lightning.pytorch.strategies.deepspeed,
        lightning.pytorch.strategies.fsdp,
        lightning.pytorch.strategies.model_parallel,
        ]

modified_functions = {
        "_init_dist_connection": _xpu_init_dist_connection,
        "_get_default_process_group_backend_for_device":
        _xpu_get_default_process_group_backend_for_device
        }

for module in modules:
    for function_name, modified_function in modified_functions.items():
        setattr(module, function_name, modified_function)
