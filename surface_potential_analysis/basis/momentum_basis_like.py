from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from surface_potential_analysis.basis.basis import FundamentalBasis

if TYPE_CHECKING:
    import numpy as np


class MomentumBasis(FundamentalBasis[Any]):  # noqa: D101
    def __init__(self, k_points: np.ndarray[Any, np.dtype[np.float64]]) -> None:
        self._k_points = k_points
        super().__init__(k_points.size)

    @property
    def k_points(self: Self) -> np.ndarray[Any, np.dtype[np.float64]]:  # noqa: D102
        return self._k_points
