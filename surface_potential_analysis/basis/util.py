from __future__ import annotations

from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeVar,
    TypeVarTuple,
    Union,
    Unpack,
    overload,
)

import numpy as np

from surface_potential_analysis.basis.basis import FundamentalBasis
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_basis,
)

from .basis_like import AxisVector, BasisLike, BasisWithLengthLike

if TYPE_CHECKING:
    from collections.abc import Iterator

    from surface_potential_analysis.basis.stacked_basis import (
        StackedBasisLike,
        StackedBasisWithVolumeLike,
    )
    from surface_potential_analysis.types import (
        ArrayFlatIndexLike,
        ArrayIndexLike,
        ArrayStackedIndexLike,
        FlatIndexLike,
        SingleFlatIndexLike,
        SingleIndexLike,
        SingleStackedIndexLike,
        StackedIndexLike,
    )

    from .stacked_basis import TupleBasisLike


_NF0Inv = TypeVar("_NF0Inv", bound=int)
_N0Inv = TypeVar("_N0Inv", bound=int)
_ND0Inv = TypeVar("_ND0Inv", bound=int)
_S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])
_B0_co = TypeVar("_B0_co", bound=BasisLike[Any, Any], covariant=True)
_B0Inv = TypeVar("_B0Inv", bound=BasisLike[Any, Any])
_BL0Inv = TypeVar("_BL0Inv", bound=BasisWithLengthLike[Any, Any, Any])
_BL0_co = TypeVar("_BL0_co", bound=BasisWithLengthLike[Any, Any, Any], covariant=True)
_B = TypeVarTuple("_B")
_TS = TypeVarTuple("_TS")


# ruff: noqa: D102, PLR0904
class BasisUtil(BasisLike[Any, Any], Generic[_B0_co]):
    """A class to help with the manipulation of an axis."""

    _basis: _B0_co

    def __init__(self, basis: _B0_co) -> None:
        self._basis = basis

    @property
    def n(self: BasisUtil[BasisLike[Any, _N0Inv]]) -> _N0Inv:
        return self._basis.n

    @property
    def fundamental_n(self: BasisUtil[BasisLike[_NF0Inv, _N0Inv]]) -> _NF0Inv:
        return self._basis.fundamental_n

    @property
    def vectors(
        self: BasisUtil[BasisLike[_NF0Inv, _N0Inv]],
    ) -> np.ndarray[tuple[_N0Inv, _NF0Inv], np.dtype[np.complex128]]:
        return self.__into_fundamental__(np.eye(self.n, self.n))  # type: ignore[return-value]

    def __into_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex128] | np.dtype[np.float64]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
        return self._basis.__into_fundamental__(vectors, axis)

    def __from_fundamental__(
        self,
        vectors: np.ndarray[_S0Inv, np.dtype[np.complex128] | np.dtype[np.float64]],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
        return self._basis.__from_fundamental__(vectors, axis)

    @property
    def nx_points(
        self: BasisUtil[BasisLike[Any, _N0Inv]],
    ) -> np.ndarray[tuple[_N0Inv], np.dtype[np.int_]]:
        return np.arange(0, self.n, dtype=int)  # type: ignore[no-any-return]

    @property
    def nk_points(
        self: BasisUtil[BasisLike[Any, _N0Inv]],
    ) -> np.ndarray[tuple[_N0Inv], np.dtype[np.int_]]:
        return np.fft.ifftshift(  # type: ignore[no-any-return]
            np.arange((-self.n + 1) // 2, (self.n + 1) // 2)
        )

    @property
    def fundamental_nk_points(
        self: BasisUtil[BasisLike[_NF0Inv, Any]],
    ) -> np.ndarray[tuple[_NF0Inv], np.dtype[np.int_]]:
        # We want points from (-self.Nk + 1) // 2 to (self.Nk - 1) // 2
        n = self.fundamental_n
        return np.fft.ifftshift(  # type: ignore[no-any-return]
            np.arange((-n + 1) // 2, (n + 1) // 2)
        )

    @property
    def fundamental_nx_points(
        self: BasisUtil[BasisLike[_NF0Inv, Any]],
    ) -> np.ndarray[tuple[_NF0Inv], np.dtype[np.int_]]:
        return np.arange(  # type: ignore[no-any-return]
            0,
            self.fundamental_n,
            dtype=int,  # type: ignore[misc]
        )

    @property
    def delta_x(
        self: BasisUtil[BasisWithLengthLike[Any, Any, _ND0Inv]],
    ) -> AxisVector[_ND0Inv]:
        return self._basis.delta_x

    @cached_property
    def dx(
        self: BasisUtil[BasisWithLengthLike[Any, Any, _ND0Inv]],
    ) -> AxisVector[_ND0Inv]:
        return self.delta_x / self.n  # type: ignore[no-any-return, misc]

    @cached_property
    def fundamental_dx(
        self: BasisUtil[BasisWithLengthLike[Any, Any, _ND0Inv]],
    ) -> AxisVector[_ND0Inv]:
        return self.delta_x / self.fundamental_n  # type: ignore[no-any-return,misc]

    @property
    def x_points(
        self: BasisUtil[BasisWithLengthLike[Any, _N0Inv, _ND0Inv]],
    ) -> np.ndarray[tuple[_ND0Inv, _N0Inv], np.dtype[np.int_]]:
        return self.dx[:, np.newaxis] * self.nx_points  # type: ignore[no-any-return]

    @property
    def fundamental_x_points(
        self: BasisUtil[BasisWithLengthLike[_NF0Inv, Any, _ND0Inv]],
    ) -> np.ndarray[tuple[_ND0Inv, _NF0Inv], np.dtype[np.int_]]:
        return self.fundamental_dx[:, np.newaxis] * self.fundamental_nx_points  # type: ignore[no-any-return]

    def __iter__(self: BasisUtil[TupleBasisLike[*_B]]) -> Iterator[Union[*_B]]:
        return self._basis.__iter__()

    @property
    def shape(
        self: BasisUtil[TupleBasisLike[*_TS]],
    ) -> tuple[int, ...]:
        return self._basis.shape

    @property
    def ndim(self: BasisUtil[TupleBasisLike[*_TS]]) -> int:
        return self._basis.ndim

    @property
    def fundamental_shape(
        self: BasisUtil[StackedBasisLike[Any, Any, Any]],
    ) -> tuple[int, ...]:
        return self._basis.fundamental_shape

    @property
    def stacked_nk_points(
        self: BasisUtil[TupleBasisLike[*_TS]],
    ) -> ArrayStackedIndexLike[tuple[int]]:
        nk_mesh = np.meshgrid(
            *[BasisUtil(xi_basis).nk_points for xi_basis in self],
            indexing="ij",
        )
        return tuple(nki.ravel() for nki in nk_mesh)

    @property
    def fundamental_stacked_nk_points(
        self: BasisUtil[StackedBasisLike[Any, Any, Any]],
    ) -> ArrayStackedIndexLike[tuple[int]]:
        fundamental = stacked_basis_as_fundamental_basis(self._basis)
        nk_mesh = np.meshgrid(
            *[BasisUtil(xi_basis).fundamental_nk_points for xi_basis in fundamental],
            indexing="ij",
        )
        return tuple(nki.ravel() for nki in nk_mesh)

    @property
    def stacked_nx_points(
        self: BasisUtil[TupleBasisLike[*_TS]],
    ) -> ArrayStackedIndexLike[tuple[int]]:
        nx_mesh = np.meshgrid(
            *[BasisUtil(xi_basis).nx_points for xi_basis in self],
            indexing="ij",
        )
        return tuple(nxi.ravel() for nxi in nx_mesh)

    @property
    def fundamental_stacked_nx_points(
        self: BasisUtil[StackedBasisLike[Any, Any, Any]],
    ) -> ArrayStackedIndexLike[tuple[int]]:
        shape = self.fundamental_shape
        nx_mesh = np.meshgrid(
            *[BasisUtil(FundamentalBasis(s)).fundamental_nx_points for s in shape],
            indexing="ij",
        )
        return tuple(nxi.ravel() for nxi in nx_mesh)

    @overload
    def get_flat_index(
        self: BasisUtil[TupleBasisLike[*_TS]],
        idx: SingleStackedIndexLike,
        *,
        mode: Literal["raise", "wrap", "clip"] = "raise",
    ) -> np.int_:
        ...

    @overload
    def get_flat_index(
        self: BasisUtil[TupleBasisLike[*tuple[_B0Inv, ...]]],
        idx: ArrayStackedIndexLike[Unpack[_TS]],
        *,
        mode: Literal["raise", "wrap", "clip"] = "raise",
    ) -> ArrayFlatIndexLike[Unpack[_TS]]:
        ...

    def get_flat_index(
        self: BasisUtil[TupleBasisLike[*tuple[_B0Inv, ...]]],
        idx: StackedIndexLike,
        *,
        mode: Literal["raise", "wrap", "clip"] = "raise",
    ) -> np.int_ | ArrayFlatIndexLike[Any]:
        """
        Given a stacked index, get the flat index into the Wigner-Seitz cell.

        Parameters
        ----------
        idx : tuple[int, int, int]
            The stacked index
        mode : Literal[&quot;raise&quot;, &quot;wrap&quot;, &quot;clip&quot;], optional
            Specifies how out-of-bounds indices are handled, by default "raise"

        Returns
        -------
        int
            the flattened index into the Wigner-Seitz cell
        """
        return np.ravel_multi_index(idx, self.shape, mode=mode)

    @overload
    def get_stacked_index(
        self: BasisUtil[TupleBasisLike[*tuple[_B0Inv, ...]]],
        idx: SingleFlatIndexLike,
    ) -> SingleStackedIndexLike:
        ...

    @overload
    def get_stacked_index(
        self: BasisUtil[TupleBasisLike[*tuple[_B0Inv, ...]]],
        idx: ArrayFlatIndexLike[Unpack[_TS]],
    ) -> ArrayStackedIndexLike[Unpack[_TS]]:
        ...

    def get_stacked_index(
        self: BasisUtil[TupleBasisLike[*tuple[_B0Inv, ...]]],
        idx: FlatIndexLike,
    ) -> StackedIndexLike:
        """
        Given a flat index, produce a stacked index.

        Parameters
        ----------
        idx : int

        Returns
        -------
        tuple[int, int, int]
        """
        return np.unravel_index(idx, self.shape)

    @overload
    def __getitem__(
        self: BasisUtil[TupleBasisLike[*tuple[_B0Inv, ...]]], index: int
    ) -> BasisLike[Any, Any]:
        ...

    @overload
    def __getitem__(
        self: BasisUtil[TupleBasisLike[*tuple[Any, ...]]], index: slice
    ) -> TupleBasisLike[*tuple[Any, ...]]:
        ...

    def __getitem__(
        self: BasisUtil[TupleBasisLike[*tuple[Any, ...]]], index: int | slice
    ) -> Any:
        return self._basis.__getitem__(index)

    @cached_property
    def volume(self) -> np.float64:
        return np.linalg.det(self.delta_x_stacked)  # type: ignore[no-any-return]

    @cached_property
    def reciprocal_volume(self) -> np.float64:
        return np.linalg.det(self.dk_stacked)  # type: ignore[no-any-return]

    @overload
    def get_k_points_at_index(
        self: BasisUtil[TupleBasisLike[*tuple[_BL0Inv, ...]]], idx: SingleIndexLike
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        ...

    @overload
    def get_k_points_at_index(
        self: BasisUtil[TupleBasisLike[*tuple[_BL0Inv, ...]]],
        idx: ArrayIndexLike[Unpack[_TS]],
    ) -> np.ndarray[tuple[int, Unpack[_TS]], np.dtype[np.float64]]:
        ...

    def get_k_points_at_index(
        self: BasisUtil[TupleBasisLike[*tuple[_BL0Inv, ...]]],
        idx: ArrayIndexLike[Unpack[_TS]] | SingleIndexLike,
    ) -> (
        np.ndarray[tuple[int, Unpack[_TS]], np.dtype[np.float64]]
        | np.ndarray[tuple[int], np.dtype[np.float64]]
    ):
        nk_points = idx if isinstance(idx, tuple) else self.get_stacked_index(idx)
        return np.tensordot(self.dk_stacked, nk_points, axes=(0, 0))  # type: ignore Return type is unknown

    @property
    def k_points(
        self: BasisUtil[TupleBasisLike[*tuple[_BL0Inv, ...]]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        return self.get_k_points_at_index(self.stacked_nk_points)

    @property
    def fundamental_stacked_k_points(
        self: BasisUtil[TupleBasisLike[*tuple[_BL0Inv, ...]]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        return np.tensordot(
            self.fundamental_dk_stacked, self.fundamental_stacked_nk_points, axes=(0, 0)
        )

    @overload
    def get_x_points_at_index(
        self: BasisUtil[TupleBasisLike[*tuple[Any, ...]]], idx: SingleIndexLike
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        ...

    @overload
    def get_x_points_at_index(
        self: BasisUtil[TupleBasisLike[*tuple[Any, ...]]],
        idx: ArrayIndexLike[Unpack[_TS]],
    ) -> np.ndarray[tuple[int, Unpack[_TS]], np.dtype[np.float64]]:
        ...

    def get_x_points_at_index(
        self: BasisUtil[TupleBasisLike[*tuple[_BL0_co, ...]]],
        idx: ArrayIndexLike[Unpack[_TS]] | SingleIndexLike,
    ) -> (
        np.ndarray[tuple[int, Unpack[_TS]], np.dtype[np.float64]]
        | np.ndarray[tuple[int], np.dtype[np.float64]]
    ):
        nx_points = idx if isinstance(idx, tuple) else self.get_stacked_index(idx)
        return np.tensordot(self.dx_stacked, nx_points, axes=(0, 0))  # type: ignore Return type is unknown

    @property
    def x_points_stacked(
        self: BasisUtil[TupleBasisLike[*tuple[_BL0Inv, ...]]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        return self.get_x_points_at_index(self.stacked_nx_points)

    @property
    def fundamental_x_points_stacked(
        self: BasisUtil[TupleBasisLike[*tuple[_BL0Inv, ...]]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        return np.tensordot(
            self.fundamental_dx_stacked, self.fundamental_stacked_nx_points, axes=(0, 0)
        )

    @property
    def delta_x_stacked(
        self: BasisUtil[StackedBasisWithVolumeLike[Any, Any, Any]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        return self._basis.delta_x_stacked

    @property
    def fundamental_delta_x_stacked(
        self: BasisUtil[StackedBasisWithVolumeLike[Any, Any, Any]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        return self._basis.delta_x_stacked

    @cached_property
    def dx_stacked(
        self: BasisUtil[TupleBasisLike[*tuple[_BL0Inv, ...]]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        return np.array([BasisUtil(axi).dx for axi in self])

    @cached_property
    def fundamental_dx_stacked(
        self: BasisUtil[TupleBasisLike[*tuple[_BL0Inv, ...]]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        return np.array([BasisUtil(axi).fundamental_dx for axi in self])

    @property
    def delta_k_stacked(
        self: BasisUtil[TupleBasisLike[*tuple[_BL0Inv, ...]]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        return np.array(self.shape)[:, np.newaxis] * self.dk_stacked

    @cached_property
    def fundamental_delta_k_stacked(
        self: BasisUtil[TupleBasisLike[*tuple[_BL0Inv, ...]]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        return np.array(self.fundamental_shape)[:, np.newaxis] * self.dk_stacked

    @cached_property
    def dk_stacked(
        self: BasisUtil[TupleBasisLike[*tuple[_BL0Inv, ...]]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Get dk as a list of dk for each axis."""
        return 2 * np.pi * np.linalg.inv(self.delta_x_stacked).T

    @property
    def fundamental_dk_stacked(
        self: BasisUtil[TupleBasisLike[*tuple[_BL0Inv, ...]]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        return self.dk_stacked


def _get_average_angles(
    angles: np.ndarray[Any, np.dtype[np.float64]], axis: int = -1
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Get the angles, averaged in a periodic sense.

    Parameters
    ----------
    angles : np.ndarray[Any, np.dtype[np.float64]]
    axis : int, optional
        axis, by default -1

    Returns
    -------
    np.ndarray[Any, np.dtype[np.float64]]
    """
    # Convert angles to unit circle coordinates
    x = np.cos(angles)
    y = np.sin(angles)

    # Compute the average of these coordinates
    avg_x = np.mean(x, axis=axis)
    avg_y = np.mean(y, axis=axis)

    # Convert the average coordinates back to an angle
    return np.where(
        np.logical_and(np.isclose(avg_x, 0), np.isclose(avg_y, 0)),
        np.nan,
        np.arctan2(avg_y, avg_x),
    )


def get_twice_average_nx(
    basis: StackedBasisLike[Any, Any, Any],
) -> tuple[np.ndarray[tuple[int, int], np.dtype[np.int_]], ...]:
    """
    Get a matrix of twice the average nx, taken in a periodic fashion.

    This should map 1 + (N-1) to 0, but (N//2 + N//2) to N //2

    Parameters
    ----------
    basis : StackedBasisLike[Any, Any, Any]

    Returns
    -------
    np.ndarray[tuple[int, int], np.dtype[np.int_]]
        _description_
    """
    util = BasisUtil(basis)
    # Interpret each x point as an angle
    angles = tuple(
        np.array(np.meshgrid(2 * np.pi * n_x_points / n, 2 * np.pi * n_x_points / n))
        for (n_x_points, n) in zip(
            util.fundamental_stacked_nx_points,
            util.fundamental_shape,
            strict=True,
        )
    )
    # Average each pair of angles
    average_angles = tuple(_get_average_angles(angle, axis=0) for angle in angles)
    # In this case, it is safe to do a normal average
    is_nan = np.isnan(average_angles)
    if np.any(is_nan):
        for i, angle in enumerate(angles):
            average_angles[i][is_nan[i]] = np.average(angle[:, is_nan[i]], axis=0)

    # Interpret this back as a coordinate. Note we multiply average by 2 here
    return tuple(
        np.rint(average * n / np.pi).astype(np.int_) % (2 * n)
        for (average, n) in zip(
            average_angles,
            util.fundamental_shape,
            strict=True,
        )
    )


def get_displacements_nx(
    basis: StackedBasisLike[Any, Any, Any],
) -> tuple[np.ndarray[tuple[int, int], np.dtype[np.int_]], ...]:
    """
    Get a matrix of displacements in nx, taken in a periodic fashion.

    Parameters
    ----------
    basis : StackedBasisLike[Any, Any, Any]

    Returns
    -------
    np.ndarray[tuple[int, int], np.dtype[np.int_]]
        _description_
    """
    util = BasisUtil(basis)
    return tuple(
        (n_x_points[:, np.newaxis] - n_x_points[np.newaxis, :] + n // 2) % n - (n // 2)
        for (n_x_points, n) in zip(
            util.fundamental_stacked_nx_points,
            util.fundamental_shape,
            strict=True,
        )
    )


def get_displacements_x(
    basis: StackedBasisLike[Any, Any, Any],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """
    Get a matrix of displacements in x, taken in a periodic fashion.

    Parameters
    ----------
    basis : StackedBasisLike[Any, Any, Any]
        _description_

    Returns
    -------
    np.ndarray[tuple[int, int], np.dtype[np.float64]]
        _description_
    """
    step = get_displacements_nx(basis)
    util = BasisUtil(basis)
    return np.linalg.norm(
        np.tensordot(step, util.dx_stacked, axes=(0, 0)),
        axis=2,
    )
