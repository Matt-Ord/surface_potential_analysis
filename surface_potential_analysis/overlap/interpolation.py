from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar, TypeVarTuple

import numpy as np

from surface_potential_analysis.basis.conversion import basis_as_single_point_basis
from surface_potential_analysis.basis.stacked_basis import TupleBasis
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.stacked_basis.util import (
    wrap_index_around_origin,
)
from surface_potential_analysis.util.decorators import timed

if TYPE_CHECKING:
    from collections.abc import Callable

    from surface_potential_analysis.basis.basis import FundamentalPositionBasis
    from surface_potential_analysis.basis.stacked_basis import TupleBasisLike
    from surface_potential_analysis.overlap.overlap import SingleOverlap
    from surface_potential_analysis.types import IntLike_co

    _L0Inv = TypeVar("_L0Inv", bound=int)
    _L1Inv = TypeVar("_L1Inv", bound=int)
    _L2Inv = TypeVar("_L2Inv", bound=int)

    FundamentalPositionOverlap = SingleOverlap[
        TupleBasisLike[
            FundamentalPositionBasis[_L0Inv, Literal[3]],
            FundamentalPositionBasis[_L1Inv, Literal[3]],
            FundamentalPositionBasis[_L2Inv, Literal[3]],
        ]
    ]
    Ts = TypeVarTuple("Ts")


_S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])

ArrayStackedIndexFractionLike = tuple[
    np.ndarray[_S0Inv, np.dtype[np.float64 | np.int_]],
    np.ndarray[_S0Inv, np.dtype[np.float64 | np.int_]],
    np.ndarray[_S0Inv, np.dtype[np.float64 | np.int_]],
]


def get_overlap_momentum_interpolator_k_fractions(
    overlap: SingleOverlap[
        TupleBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]
    ],
) -> Callable[
    [ArrayStackedIndexFractionLike[_S0Inv]],
    np.ndarray[_S0Inv, np.dtype[np.complex128]],
]:
    """
    Get an interpolator to calculate the overlap in momentum basis.

    Parameters
    ----------
    overlap : OverlapPosition[_L0Inv, _L1Inv, _L2Inv]

    Returns
    -------
    Callable[[np.ndarray[tuple[Literal[3], Unpack[_S0Inv]], np.dtype[np.float_]]], np.ndarray[_S0Inv, np.dtype[np.complex_]], ]
        Interpolator which takes a list of coordinates as fractions of k0, k1, k2 index and returns the
        overlap at this point
    """
    util = BasisUtil(overlap["basis"][0])
    nx_points_wrapped = wrap_index_around_origin(
        overlap["basis"][0], util.stacked_nx_points, axes=(0, 1)
    )
    x_fractions = np.asarray(nx_points_wrapped, dtype=float)[:, :, np.newaxis]
    x_fractions /= util.shape[0]

    def _interpolator(
        k_fractions: ArrayStackedIndexFractionLike[_S0Inv],
    ) -> np.ndarray[_S0Inv, np.dtype[np.complex128]]:
        phi = np.sum(x_fractions * np.asarray(k_fractions)[:, np.newaxis, :], axis=0)
        return np.sum(overlap["data"][:, np.newaxis] * np.exp(2j * np.pi * phi), axis=0)  # type: ignore[no-any-return]

    return _interpolator


def get_overlap_momentum_interpolator(
    overlap: FundamentalPositionOverlap[_L0Inv, _L1Inv, _L2Inv],
) -> Callable[
    [np.ndarray[tuple[Literal[3], *Ts], np.dtype[np.float64]]],
    np.ndarray[tuple[*Ts], np.dtype[np.complex128]],
]:
    """
    Given the overlap create an interpolator to calculate th interpolation in momentum basis.

    Parameters
    ----------
    overlap : OverlapPosition[_L0Inv, _L1Inv, _L2Inv]

    Returns
    -------
    Callable[[np.ndarray[tuple[Literal[3], Unpack[_S0Inv]], np.dtype[np.float_]]], np.ndarray[_S0Inv, np.dtype[np.complex_]], ]
        Interpolator, which takes a coordinate list on momentum basis
    """
    util = BasisUtil(overlap["basis"][0])
    nx_points_wrapped = wrap_index_around_origin(
        overlap["basis"][0], util.stacked_nx_points, axes=(0, 1)
    )
    x_points = util.get_x_points_at_index(nx_points_wrapped)

    def _interpolator(
        k_coordinates: np.ndarray[tuple[Literal[3], *Ts], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[*Ts], np.dtype[np.complex128]]:
        phi = np.tensordot(x_points, k_coordinates, axes=(0, 0))
        return np.tensordot(overlap["data"], np.exp(1j * phi), axes=(0, 0))  # type: ignore[no-any-return]

    return _interpolator


@timed
def get_overlap_momentum_interpolator_flat(
    overlap: SingleOverlap[
        TupleBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]
    ],
    n_points: IntLike_co | None = None,
) -> Callable[
    [np.ndarray[tuple[Literal[2], *Ts], np.dtype[np.float64]]],
    np.ndarray[tuple[*Ts], np.dtype[np.complex128]],
]:
    """
    Given the overlap create an interpolator to calculate the interpolation in momentum basis.

    This only works for kx2=0, and makes the assumption that x2 lies completely perpendicular to
    the x01 direction.

    Parameters
    ----------
    overlap : OverlapPosition[_L0Inv, _L1Inv, _L2Inv]

    Returns
    -------
    Callable[[np.ndarray[tuple[Literal[2], Unpack[_S0Inv]], np.dtype[np.float_]]], np.ndarray[_S0Inv, np.dtype[np.complex_]], ]
        Interpolator, which takes a coordinate list in momentum basis ignoring k2 axis
    """
    basis = TupleBasis[Any, Any, Any](
        *overlap["basis"][0][:-1],
        basis_as_single_point_basis(overlap["basis"][0][-1]),  # type: ignore cannot deal with *
    )
    util = BasisUtil(basis)
    nx_points_wrapped = wrap_index_around_origin(
        overlap["basis"][0], util.stacked_nx_points, axes=(0, 1)
    )
    x_points = util.get_x_points_at_index(nx_points_wrapped)[:2, :]

    vector = overlap["data"].reshape(overlap["basis"][0].shape)
    vector_transformed = np.fft.ifft(vector, axis=-1, norm="forward")[..., 0].ravel()

    relevant_slice = (
        slice(None)
        if n_points is None
        else np.argsort(np.abs(vector_transformed))[::-1][: int(n_points)]
    )
    relevant_x_points = x_points[:, relevant_slice]
    relevant_vector = vector_transformed[relevant_slice]

    def _interpolator(
        k_coordinates: np.ndarray[tuple[Literal[2], *Ts], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[*Ts], np.dtype[np.complex128]]:
        phi = np.tensordot(relevant_x_points, k_coordinates, axes=(0, 0))
        return np.tensordot(relevant_vector, np.exp(1j * phi), axes=(0, 0))  # type: ignore[no-any-return]

    return _interpolator


def get_angle_averaged_diagonal_overlap_function(
    interpolator: Callable[
        [np.ndarray[tuple[Literal[2], int], np.dtype[np.float64]]],
        np.ndarray[tuple[int], np.dtype[np.complex128]],
    ],
    abs_q: np.ndarray[_S0Inv, np.dtype[np.float64]],
    *,
    theta_samples: int = 50,
) -> np.ndarray[_S0Inv, np.dtype[np.float64]]:
    """
    Given an interpolator for the overlap, average over the angle.

    Parameters
    ----------
    interpolator : Callable[ [np.ndarray[tuple[Literal[2], int], np.dtype[np.float_]]], np.ndarray[tuple[int], np.dtype[np.complex_]], ]
    abs_q : np.ndarray[tuple[int], np.dtype[np.float_]]
    theta_samples : int, optional
        number of samples to average over, by default 50

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.float_]]
    """
    theta = np.linspace(0, 2 * np.pi, theta_samples)
    averages = list[np.float64]()
    for q in abs_q.ravel():
        k_points = q * np.array([np.cos(theta), np.sin(theta)])
        interpolated = interpolator(k_points)  # type: ignore[var-annotated]
        averages.append(np.average(np.square(np.abs(interpolated))))

    return np.array(averages).reshape(abs_q.shape)  # type: ignore[no-any-return]
