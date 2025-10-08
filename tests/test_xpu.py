def test_import():
    # Test that lightning_xpu can be imported.
    try:
        import lightning_xpu
        lightning_xpu_imported = True
    except ModuleNotFoundError:
        lightning_xpu_imported = False

    assert lightning_xpu_imported == True

import torch.xpu
from lightning_xpu.lightning.pytorch.accelerators.xpu import XPUAccelerator
device = XPUAccelerator()

def test_xpu_available():
    # Test that device availability is as reported by pytorch.
    assert device.is_available() == torch.xpu.is_available()

def test_distributed_modifications():
    # Test that functions have been modified as intended.
    from lightning_xpu.lightning.fabric.utilities.distributed import (
            modules,
            modified_functions,
            )
    assert len(modules) > 0
    assert len(modified_functions) > 0
    for function_name, modified_function in modified_functions.items():
        assert callable(modified_function)
        for module in modules:
            assert getattr(module, function_name) is modified_function
