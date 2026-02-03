"""
DEPRECATED: All old dataset classes have been DELETED.

BREAKING CHANGE: Complete migration to Grain-based data loading.

Old API (DELETED):
- BurgersEquationDataset - DELETED ✅
- DarcyFlowDataset - DELETED ✅
- DiffusionAdvectionDataset - DELETED ✅
- ShallowWaterEquationsDataset - DELETED ✅

New API (Use This):
    from opifex.data.loaders import (
        create_burgers_loader,
        create_darcy_loader,
        create_diffusion_loader,
        create_shallow_water_loader,
    )

This module is now empty - all dataset classes have been removed.
No backward compatibility is provided.
"""

__all__ = []
