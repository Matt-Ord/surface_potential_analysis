from __future__ import annotations

from typing import TYPE_CHECKING, Self, Sequence

import numpy as np
from matplotlib.scale import ScaleBase
from matplotlib.ticker import Locator
from matplotlib.transforms import Transform

if TYPE_CHECKING:
    from matplotlib.axis import Axis
    from matplotlib.pylab import ArrayLike


class SquaredLocator(Locator):  # noqa: D101
    def __call__(self: Self):  # noqa: D102, ANN204
        assert self.axis is not None
        min_val, max_val = self.axis.get_view_interval()  # type: ignore unknown lib type
        return self.tick_values(min_val, max_val)

    def tick_values(self: Self, vmin: float, vmax: float) -> Sequence[float]:
        """
        Return the locations of the ticks.

        .. note::

            Because the values are fixed, vmin and vmax are not used in this
            method.

        """
        locs = np.sqrt(np.linspace(vmin**2, vmax**2, 8)).tolist()
        return self.raise_if_exceeds(locs)


class SquaredScale(ScaleBase):  # noqa: D101
    name = "squared"

    def __init__(self, axis: Axis | None) -> None:
        super().__init__(axis)

    def get_transform(self: Self) -> Transform:  # noqa: D102
        return self._SquaredTransform()

    def set_default_locators_and_formatters(self: Self, axis: Axis) -> None:  # noqa: D102, PLR6301
        axis.set_major_locator(SquaredLocator())

    class _SquaredTransform(Transform):
        @property
        def input_dims(self: Self) -> int:
            return 1

        @property
        def output_dims(self: Self) -> int:
            return 1

        @property
        def is_separable(self: Self) -> bool:
            return True

        def transform_non_affine(self: Self, values: ArrayLike) -> ArrayLike:  # noqa: PLR6301
            return np.square(np.real(values))

        def inverted(self: Self) -> Transform:  # noqa: PLR6301
            return SquaredScale._InvertedSquaredTransform()

    class _InvertedSquaredTransform(Transform):
        @property
        def input_dims(self: Self) -> int:
            return 1

        @property
        def output_dims(self: Self) -> int:
            return 1

        @property
        def is_separable(self: Self) -> bool:
            return True

        def transform_non_affine(self: Self, values: ArrayLike) -> ArrayLike:  # noqa: PLR6301
            return np.sqrt(np.abs(values))

        def inverted(self: Self) -> Transform:  # noqa: PLR6301
            return SquaredScale._SquaredTransform()
