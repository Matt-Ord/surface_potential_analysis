# ruff: noqa: D102
from __future__ import annotations

import numpy as np
from slate.metadata import (
    ExplicitLabeledMetadata,
    LabeledMetadata,
    SpacedLabeledMetadata,
)


class BlochFractionMetadata(LabeledMetadata[np.float64]):
    """Metadata with the addition of bloch fracrtions."""


class ExplicitBlochFractionMetadata(
    ExplicitLabeledMetadata[np.float64], BlochFractionMetadata
):
    """Metadata with the addition of bloch fracrtions."""


class SpacedBlochFractionMetadata(SpacedLabeledMetadata, BlochFractionMetadata):
    """Metadata with the addition of bloch fracrtions."""
