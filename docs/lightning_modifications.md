# Modifications to lightning functions and methods

On import, `lightning_xpu` it substitutes modified versions of some of
the functions and methods of `lightning`:
- function: `lightning.fabric.utilities.distributed.`_get_default_process_group_backend_for_device()`
  modified to allow for both `"ccl"` and `"xccl"` as backend for `"xpu"` device;
- function: `lightning.fabric.utilities.distributed._init_dist_connection()`
  modified to set environment variables used to determine local rank
  and local world size when using `"xpu"` devices and `"xccl"` or `"ccl"` backend;
- function: `lightning.pytorch.trainer.setup._log_device_info()`
  modified to recognise `XPUAccelerator` as representing a GPU,
  and to report number of GPUs;  
- class: `lightning.fabric.strategies.ddp.DDPStrategy`
	- `barrier()`
       modified to allow `"xccl"` and `"ccl"` as backend for distributed processing;
	- `_setup_module`
      modified to handle `"xpu"` device;
- class: `lightning.fabric.strategies.fsdp.FSDPStrategy`
	- `barrier()`
      modified to allow `"xccl"` and `"ccl"` as backend for distributed processing;
	- `setup_environment()`
      modified to set `"xpu"` as device type for `"xccl"` or `"ccl"` as
      process-group backend;
- class: `lightning.fabric.strategies.model_parallel.ModelParallelStrategy`
	- `barrier()`
      modified to allow `"xccl"` and `"ccl"` as backend for distributed processing;
- class: `lightning.pytorch.strategies.ddp.DDPStrategy`
	- `barrier()`
       modified to allow `"xccl"` and `"ccl"` as backend for distributed processing;
	- `_setup_model()`
      modified to handle `"xpu"` device;
- class: `lightning.pytorch.strategies.fsdp.FSDPStrategy`
	- `barrier()`
       modified to allow `"xccl"` and `"ccl"` as backend for distributed processing;
	- `setup_environment()`
      modified to set `"xpu"` as device type for `"xccl"` or `"ccl"` as
      process-group backend;
- class: `lightning.pytorch.strategies.model_parallel.ModelParallelStrategy`
	- `barrier()`
       modified to allow `"xccl"` and `"ccl"` as backend for distributed processing;
- class: `lightning.pytorch.trainer.connectors.accelerator_connector._AcceleratorConnector`
	- `_check_strategy_and_fallback()`
      modified to allow `"fsdp"` as strategy with `"xpu"` as accelerator type;
	- `_choose_and_init_cluster_environment()`
      modified to set default values for selected Slurm environment variables;
	- `_choose_auto_accelerator()`
      modified to allow automatic selection of `"xpu"` as accelerator type;
	- `_choose_gpu_accelerator_backend()`
      modified to allow selection of `"xpu"` as gpu backend;
	- `_choose_strategy()`
      modified to allow automatic selection of `"single_device"` or `"ddp"
