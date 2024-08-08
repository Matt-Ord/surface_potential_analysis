from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis_like import (
    AsFundamentalBasis,
    AsTransformedBasis,
    AxisVector,
    BasisLike,
    BasisWithLengthLike,
)
from surface_potential_analysis.util.util import slice_along_axis

if TYPE_CHECKING:
    from surface_potential_analysis.types import (
        IntLike_co,
    )

    _DT = TypeVar("_DT", bound=np.dtype[Any])
    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])


_N0_co = TypeVar("_N0_co", bound=int, covariant=True)
_N1_co = TypeVar("_N1_co", bound=int, covariant=True)
_N2_co = TypeVar("_N2_co", bound=int, covariant=True)
_ND0Inv = TypeVar("_ND0Inv", bound=int)


def _pad_sample_axis(
    vectors: np.ndarray[_S0Inv, _DT],
    step: IntLike_co,
    offset: IntLike_co,
    axis: IntLike_co = -1,
) -> np.ndarray[tuple[int, ...], _DT]:
    final_shape = np.array(vectors.shape)
    final_shape[axis] = step * final_shape[axis]
    padded = np.zeros(final_shape, dtype=vectors.dtype)
    # We could alternatively slice starting on zero
    # and roll at the end but this is worse for performance
    vectors = np.roll(vectors, offset // step, axis=axis)  # type: ignore cannot infer dtype
    padded[slice_along_axis(slice(offset % step, None, step), axis)] = vectors

    return padded  # type: ignore[no-any-return]


def _truncate_sample_axis(
    vectors: np.ndarray[_S0Inv, _DT],
    step: IntLike_co,
    offset: IntLike_co,
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], _DT]:
    truncated = vectors[slice_along_axis(slice(offset % step, None, step), axis)]  # type: ignore index type wrong
    # We could alternatively roll before we take the slice
    # and slice(0, None, ns) but this is worse for performance
    return np.roll(truncated, -(offset // step), axis=axis)  # type: ignore[no-any-return]


class EvenlySpacedBasis(
    AsFundamentalBasis[int, _N0_co],
    BasisLike[int, _N0_co],
    Generic[_N0_co, _N1_co, _N2_co],
):
    """A axis with vectors that are the fundamental position states."""

    def __init__(self, n: _N0_co, step: _N1_co, offset: _N2_co) -> None:
        self._n = n
        self._step = step
        self._offset = offset
        super().__init__()

    @property
    def n(self) -> _N0_co:
        return self._n

    @property
    def fundamental_n(self) -> int:
        return self.n * self._step

    @property
    def step(self) -> _N1_co:
        return self._step

    @property
    def offset(self) -> _N2_co:
        return self._offset

    def __as_fundamental__(
        self,
        vectors: np.ndarray[
            _S0Inv,
            np.dtype[np.complex128]
            | np.dtype[np.float64]
            | np.dtype[np.float64 | np.complex128],
        ],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
        casted = vectors.astype(np.complex128, copy=False)
        return _pad_sample_axis(casted, self.step, self.offset, axis)

    def __from_fundamental__(
        self,
        vectors: np.ndarray[
            _S0Inv,
            np.dtype[np.complex128]
            | np.dtype[np.float64]
            | np.dtype[np.float64 | np.complex128],
        ],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
        casted = vectors.astype(np.complex128, copy=False)
        return _truncate_sample_axis(casted, self.step, self.offset, axis)


class EvenlySpacedTransformedBasis(
    AsTransformedBasis[int, _N0_co],
    BasisLike[int, _N0_co],
    Generic[_N0_co, _N1_co, _N2_co],
):
    """A axis with vectors that are the fundamental position states."""

    def __init__(self, n: _N0_co, step: _N1_co, offset: _N2_co) -> None:
        self._n = n
        self._step = step
        self._offset = offset
        super().__init__()

    @property
    def n(self) -> _N0_co:
        return self._n

    @property
    def fundamental_n(self) -> int:
        return self.n * self._step  # type: ignore[return-value]

    @property
    def step(self) -> _N1_co:
        return self._step

    @property
    def offset(self) -> _N2_co:
        return self._offset

    def __as_transformed__(
        self,
        vectors: np.ndarray[
            _S0Inv,
            np.dtype[np.complex128]
            | np.dtype[np.float64]
            | np.dtype[np.float64 | np.complex128],
        ],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
        casted = vectors.astype(np.complex128, copy=False)
        return _pad_sample_axis(casted, self.step, self.offset, axis)

    def __from_transformed__(
        self,
        vectors: np.ndarray[
            _S0Inv,
            np.dtype[np.complex128]
            | np.dtype[np.float64]
            | np.dtype[np.float64 | np.complex128],
        ],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
        casted = vectors.astype(np.complex128, copy=False)
        return _truncate_sample_axis(casted, self.step, self._offset, axis)


# ruff: noqa: D102
class EvenlySpacedTransformedPositionBasis(
    EvenlySpacedTransformedBasis[_N0_co, _N1_co, _N2_co],
    BasisWithLengthLike[Any, _N0_co, _ND0Inv],
):
    """Basis used to represent a single eigenstate from a wavepacket."""

    def __init__(
        self,
        delta_x: AxisVector[_ND0Inv],
        *,
        n: _N0_co,
        step: _N1_co,
        offset: _N2_co,
    ) -> None:
        self._delta_x = delta_x
        super().__init__(n, step, offset)

    @property
    def delta_x(self) -> AxisVector[_ND0Inv]:
        return self._delta_x
