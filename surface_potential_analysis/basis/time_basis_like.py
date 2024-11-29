from __future__ import annotations

import numpy as np
from slate.metadata import LabeledMetadata, SpacedLabeledMetadata


class TimeMetadata(LabeledMetadata[np.float64]):
    """Metadata with the addition of length."""


class SpacedTimeMetadata(SpacedLabeledMetadata, TimeMetadata):
    """Metadata with the addition of length."""
