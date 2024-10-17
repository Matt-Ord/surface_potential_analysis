from __future__ import annotations

import numpy as np
from slate.metadata import LabeledMetadata, SpacedLabeledMetadata


class MomentumMetadata(LabeledMetadata[np.float64]):
    """Metadata with the addition of momentum."""


class SpacedMomentumMetadata(SpacedLabeledMetadata, MomentumMetadata):
    """Metadata with the addition of momentum."""
