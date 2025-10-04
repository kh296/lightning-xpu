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
