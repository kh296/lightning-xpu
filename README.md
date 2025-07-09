# lightning-xpu

`lightning-xpu`  is a Python package to enable use of Intel GPUs (XPUs) with [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning).  It is a work in progress, tested for [Intel Data Center GPU Max 1550](https://www.intel.com/content/www/us/en/products/sku/232873/intel-data-center-gpu-max-1550/specifications.html) GPUs, on the [Dawn supercomputer](https://www.hpc.cam.ac.uk/d-w-n).

This package defines an `Accelerator` subclass, `XPUAccelerator` (accelerator name: `"xpu"`).  In addition, on import, it substitutes modified versions of a function, and of some of the methods of two classes:
- function: `lightning.pytorch.trainer.setup._log_device_info()`
- class: `lightning.pytorch.trainer.connectors.accelerator_connector._AcceleratorConnector`
	- `_choose_auto_accelerator()`
	- `_choose_gpu_accelerator_backend()`
	-  `_choose_and_init_cluster_environment()`
	-  `_choose_strategy()`
-  class: `lightning.pytorch.strategies.ddp.DDPStrategy`
	- `_get_process_group_backend()`
	- `_setup_model()`
	-  `_setup_distributed()`

The packages enables use of the `"single_device"` and `"ddp"` strategies with `"xpu"` accelerators, while maintaining compatibility with all other accelerator-strategy combinations supported by PyTorch Lightning.

Instructions are given below for user installation of `lightning_xpu` on Dawn, and for running a PyTorch Lightning toy example.  Basic information is then given for running other PyTorch Lightning applications on Dawn.

## Installation on Dawn

The software for XPU-enabled PyTorch Lightning can be installed as follows:
- Connect to Dawn:
```
ssh <username>@login-dawn.hpc.cam.ac.uk
```
- If not already installed in the user's area, install miniconda, following instructions at:
    https://www.anaconda.com/docs/getting-started/miniconda/install#linux
    To avoid running out of home disk space, it can be a good idea to install to
    `~/rds/hpc-work/miniconda3`, and create a link to this: 
  ```
  ln -s ~/rds/hpc-work/miniconda3 ~/miniconda3
  ```
- Clone this repository:
    ```
    git clone https://gitlab.developers.cam.ac.uk/kh296/lightning-xpu
    ```
- Within the cloned repository, go to the `install` directory:
    ```
    cd lightning-xpu/install
    ```
- Submit a batch job to perform the installation, using the script [install_lightning_2.7.sh](install/install_lightning_2.7.sh):
    ```
    sbatch install_lightning_2.7.sh
     ```
     This installs PyTorch Lightning based on [PyTorch 2.7](https://pytorch.org/blog/pytorch-2-7/).  The job writes to a log file `install2.7.log`, and creates a script `lightning-setup-2.7.sh` that can subsequently be sourced to perform environment setup.  The installation job should take about 30 minutes to complete.  If it's successful, the log file will end with information about the time taken for the installation.

## Running PyTorch Lightning toy example on Dawn

Continuing after installation of the software for XPU-enabled PyTorch Lightning, the [PyTorch Lightning toy example](https://github.com/Lightning-AI/pytorch-lightning?tab=readme-ov-file#pytorch-lightning-example) can be run on Dawn as follows:

- Within the cloned repository, go to the `examples` directory:
	```
	cd ../examples
	```
	This directory contains a Python application [toy_example.py](examples/toy_example.py), and a script for running it, [run_toy_example.sh](examples/run_toy_example.sh).   The application is a copy of the [PyTorch Lightning toy example](https://github.com/Lightning-AI/pytorch-lightning?tab=readme-ov-file#pytorch-lightning-example), but with addition of
	```
	import lightning_xpu
	```
	and with the number of training epochs set to 1.
- The toy example can be run on the batch system, or may be run interactively from a Dawn compute node.
	- To run  on the batch system, edit the script [run_toy_example.sh](examples/run_toy_example.sh) to set own project, then use:
		```
		sbatch run_toy_example.sh
		```
		By default, the job writes to `toy2.7.log`
	- To run interactively, obtain an allocation of, for example, two compute nodes:
		```
		# Substitute own project for <project>.
		sintr -t 01:00:00 -N 2 --gres=gpu:4 -A <project> -p pvc9
		./run_toy_example.sh
		```
	In both cases, the application will make use of all available GPU tiles (8, which is twice the number of GPUs), on the number of nodes requested.  Note that each initial import on a node of the modules on which the application depends can be quite slow; subsequent imports, until disconnect from the node, are faster.  A number of warnings from the Intel extensions to PyTorch are printed during initialisation, and are repeated for each GPU tile.  If the application runs successfully, the final output will indicate that training has completed for the requested number of epochs (one), and will give information both on the start-to-finish time (including the time for the initial imports), and the application run time.  The single-epoch training with the toy example is intended only as a check that the software is working.

## Running other PyTorch Lightning applications on Dawn

To run a PyTorch Lightning application on Dawn, the basic requirements are:

- Before any use of the `lightning` package, the application should include:
	```
	import lightning_xpu
	```
- The environment setup for running the application should be performed with:
	```
	# Substitue for <path_to_setup_script>
	# the path to the script lighting-setup-2.7.sh created during installation.
	# The user may move this script from its original location.
	source <path_to_setup_script>
	```

