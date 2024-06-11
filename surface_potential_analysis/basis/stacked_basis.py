from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    TypeVar,
    TypeVarTuple,
    Union,
    Unpack,
    cast,
    overload,
    runtime_checkable,
)

import numpy as np

from surface_potential_analysis.basis.basis import (
    FundamentalBasis,
    FundamentalTransformedBasis,
)
from surface_potential_analysis.basis.basis_like import (
    BasisLike,
    convert_vector,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


_S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])

_B0 = TypeVarTuple("_B0")
_B0Inv = TypeVar("_B0Inv", bound=BasisLike[Any, Any])

_FN0_co = TypeVar("_FN0_co", bound=int, covariant=True)
_N0_co = TypeVar("_N0_co", bound=int, covariant=True)
_ND0_co = TypeVar("_ND0_co", bound=int, covariant=True)


# ruff: noqa: D102
@runtime_checkable
class StackedBasisLike(BasisLike[_FN0_co, _N0_co], Protocol[_N0_co, _FN0_co, _ND0_co]):
    """Represents a basis formed from two disjoint basis."""

    @property
    def ndim(self) -> int:
        return len(self.fundamental_shape)

    @property
    def n(self) -> _N0_co:
        return cast(_N0_co, np.prod(self.shape).item())

    @property
    def fundamental_n(self) -> _FN0_co:
        return cast(_FN0_co, np.prod(self.fundamental_shape).item())

    @property
    def shape(self) -> tuple[int, ...]:
        ...

    @property
    def fundamental_shape(
        self,
    ) -> tuple[int, ...]:
        ...


@runtime_checkable
class TupleBasisLike(StackedBasisLike[Any, Any, Any], Protocol[*_B0]):
    """Represents a basis formed from two disjoint basis."""

    @property
    def shape(self: TupleBasisLike[*tuple[_B0Inv, ...]]) -> tuple[int, ...]:
        return tuple(ax.n for ax in self)

    @property
    def fundamental_shape(
        self: TupleBasisLike[*tuple[_B0Inv, ...]],
    ) -> tuple[int, ...]:
        return tuple(ax.fundamental_n for ax in self)

    def __repr__(self: TupleBasisLike[*tuple[_B0Inv, ...]]) -> str:
        return f"{self.__class__.__name__}({', '.join(b.__repr__() for b in self.__iter__())})"

    def __iter__(self) -> Iterator[Union[*_B0]]:
        ...

    @overload
    def __getitem__(
        self: TupleBasisLike[*tuple[Any, Any, _B0Inv, Unpack[tuple[Any, ...]]]],
        index: Literal[2],
    ) -> _B0Inv:
        ...

    @overload
    def __getitem__(
        self: TupleBasisLike[*tuple[Any, _B0Inv, Unpack[tuple[Any, ...]]]],
        index: Literal[1],
    ) -> _B0Inv:
        ...

    @overload
    def __getitem__(
        self: TupleBasisLike[*tuple[_B0Inv, Unpack[tuple[Any, ...]]]],
        index: Literal[0],
    ) -> _B0Inv:
        ...

    @overload
    def __getitem__(self: TupleBasisLike[*tuple[_B0Inv, ...]], index: int) -> _B0Inv:
        ...

    @overload
    def __getitem__(self, index: slice) -> TupleBasisLike[*tuple[Union[*_B0], ...]]:
        ...


def _convert_tuple_basis_vector(
    vector: np.ndarray[_S0Inv, np.dtype[np.complex128] | np.dtype[np.float64]],
    initial_basis: TupleBasisLike[*tuple[BasisLike[Any, Any], ...]],
    final_basis: TupleBasisLike[*tuple[BasisLike[Any, Any], ...]],
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
    """
    Convert a vector, expressed in terms of the given basis from_config in the basis to_config.

    Parameters
    ----------
    vector : np.ndarray[tuple[int], np.dtype[np.complex_] | np.dtype[np.float_]]
        the vector to convert
    from_config : _B3d0Inv
    to_config : _B3d1Inv
    axis : int, optional
        axis along which to convert, by default -1

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.complex_]]
    """
    swapped = vector.swapaxes(axis, 0)
    stacked = swapped.reshape(*initial_basis.shape, *swapped.shape[1:])
    for ax, (initial, final) in enumerate(zip(initial_basis, final_basis, strict=True)):
        stacked = convert_vector(stacked, initial, final, ax)
    return (  # type: ignore[no-any-return]
        stacked.astype(np.complex128, copy=False)
        .reshape(-1, *swapped.shape[1:])
        .swapaxes(axis, 0)
    )


class TupleBasis(TupleBasisLike[Unpack[_B0]]):
    """Represents a basis formed from two disjoint basis."""

    _axes: tuple[Unpack[_B0]]

    def __init__(self, *args: Unpack[_B0]) -> None:
        self._axes = args  # type: ignore[assignment]
        super().__init__()

    def __from_fundamental__(
        self: TupleBasisLike[*tuple[BasisLike[Any, Any], ...]],
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex128] | np.dtype[np.float64]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
        basis = TupleBasis[Any](
            *tuple(FundamentalBasis(axis.fundamental_n) for axis in self)
        )
        return _convert_tuple_basis_vector(vectors, basis, self, axis)

    def __into_fundamental__(
        self: TupleBasisLike[*tuple[BasisLike[Any, Any], ...]],
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex128] | np.dtype[np.float64]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
        basis = TupleBasis[Any](
            *tuple(FundamentalBasis(axis.fundamental_n) for axis in self)
        )
        return _convert_tuple_basis_vector(vectors, self, basis, axis)

    def __into_transformed__(
        self: TupleBasisLike[*tuple[BasisLike[Any, Any], ...]],
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex128] | np.dtype[np.float64]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
        basis = TupleBasis[Any](
            *tuple(FundamentalTransformedBasis(axis.fundamental_n) for axis in self)
        )
        return _convert_tuple_basis_vector(vectors, self, basis, axis)

    def __from_transformed__(
        self: TupleBasisLike[*tuple[BasisLike[Any, Any], ...]],
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex128] | np.dtype[np.float64]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
        basis = TupleBasis[Any](
            *tuple(FundamentalTransformedBasis(axis.fundamental_n) for axis in self)
        )
        return _convert_tuple_basis_vector(vectors, basis, self, axis)

    def __convert_vector_into__(
        self: TupleBasisLike[*tuple[BasisLike[Any, Any], ...]],
        vector: np.ndarray[_S0Inv, np.dtype[np.complex128] | np.dtype[np.float64]],
        basis: BasisLike[Any, Any],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[np.complex128]]:
        assert basis.fundamental_n == self.fundamental_n
        if not isinstance(basis, TupleBasisLike):
            return super().__convert_vector_into__(vector, basis, axis)
        # We overload __convert_vector_into__, more likely to get the 'happy path'
        return _convert_tuple_basis_vector(vector, self, basis, axis)  # type: ignore unknown

    def __iter__(self) -> Iterator[Union[*_B0]]:
        return self._axes.__iter__()

    @overload
    def __getitem__(
        self: TupleBasisLike[*tuple[Any, Any, _B0Inv, Unpack[tuple[Any, ...]]]],
        index: Literal[2],
    ) -> _B0Inv:
        ...

    @overload
    def __getitem__(
        self: TupleBasisLike[*tuple[Any, _B0Inv, Unpack[tuple[Any, ...]]]],
        index: Literal[1],
    ) -> _B0Inv:
        ...

    @overload
    def __getitem__(
        self: TupleBasisLike[*tuple[_B0Inv, Unpack[tuple[Any, ...]]]],
        index: Literal[0],
    ) -> _B0Inv:
        ...

    @overload
    def __getitem__(self: TupleBasisLike[*tuple[_B0Inv, ...]], index: int) -> _B0Inv:
        ...

    @overload
    def __getitem__(self, index: slice) -> TupleBasisLike[*tuple[Union[*_B0], ...]]:
        ...

    def __getitem__(self, index: slice | int) -> Any:  # type: ignore typing ignores overload
        if isinstance(index, slice):
            return TupleBasis(*self._axes[index])
        return self._axes[index]
