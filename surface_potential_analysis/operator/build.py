from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from surface_potential_analysis.basis.conversion import (
    basis_as_fundamental_position_basis,
)
from surface_potential_analysis.basis.legacy import StackedBasisLike
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.operator.operator_list import (
    SingleBasisOperatorList,
    operator_list_from_iter,
)
from surface_potential_analysis.stacked_basis.conversion import (
    tuple_basis_as_fundamental,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.legacy import (
        BasisLike,
        BasisWithLengthLike,
        FundamentalBasis,
        FundamentalPositionBasis,
        StackedBasisWithVolumeLike,
        TupleBasisWithLengthLike,
    )
    from surface_potential_analysis.operator.operator import (
        Operator,
        SingleBasisOperator,
    )
    from surface_potential_analysis.state_vector.eigenstate_list import ValueList
    from surface_potential_analysis.types import FloatLike_co


def _wrap_displacements(
    displacements: np.ndarray[Any, np.dtype[np.float64]], max_displacement: FloatLike_co
) -> np.ndarray[Any, np.dtype[np.float64]]:
    return (
        np.remainder((displacements + max_displacement), 2 * max_displacement)
        - max_displacement
    ).astype(np.float64)


def get_displacements_x(
    basis: BasisWithLengthLike, origin: float
) -> ValueList[FundamentalPositionBasis]:
    """Get the displacements from origin.

    Parameters
    ----------
    basis : BasisWithLengthLike
    origin : float

    Returns
    -------
    ValueList[FundamentalPositionBasis]
    """
    basis_x = basis_as_fundamental_basisbasis)
    distances = BasisUtil(basis_x).x_points - origin
    max_distance = np.linalg.norm(basis_x.delta_x) / 2
    data = _wrap_displacements(distances, max_distance)
    return {"basis": basis_x, "data": data.astype(np.complex128)}


def _get_displacements_x_along_axis(
    basis: StackedBasisWithVolumeLike,
    origin: float,
    axis: int,
) -> ValueList[TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],]:
    basis_x = tuple_basis_as_fundamental(basis)
    distances = BasisUtil(basis_x).x_points_stacked[axis] - np.real(origin)
    delta_x = np.linalg.norm(basis_x.delta_x_stacked[axis])
    max_distance = delta_x / 2
    data = _wrap_displacements(distances, max_distance)
    return {"basis": basis_x, "data": data.astype(np.complex128)}


def get_displacements_x_stacked(
    basis: StackedBasisWithVolumeLike, origin: tuple[float, ...]
) -> tuple[
    ValueList[TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],],
    ...,
]:
    """Get the displacements from origin."""
    return tuple(
        _get_displacements_x_along_axis(basis, o, axis)
        for (axis, o) in enumerate(origin)
    )


def get_displacements_matrix_nx_stacked(
    basis: StackedBasisLike,
) -> tuple[np.ndarray[tuple[int, int], np.dtype[np.int_]], ...]:
    """
    Get a matrix of displacements in nx, taken in a periodic fashion.

    Parameters
    ----------
    basis : StackedBasisLike

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


def get_displacements_matrix_nx(
    basis: BasisLike,
) -> np.ndarray[tuple[int, int], np.dtype[np.int_]]:
    """
    Get a matrix of displacements in nx, taken in a periodic fashion.

    Parameters
    ----------
    basis : StackedBasisLike

    Returns
    -------
    np.ndarray[tuple[int, int], np.dtype[np.int_]]
        _description_
    """
    util = BasisUtil(basis)
    n_x_points = util.fundamental_nx_points
    n = util.fundamental_n
    return (n_x_points[:, np.newaxis] - n_x_points[np.newaxis, :] + n // 2) % n - (
        n // 2
    )


def get_displacements_matrix_x(
    basis: BasisWithLengthLike,
    origin: float = 0.0,
) -> Operator[
    FundamentalPositionBasis,
    FundamentalPositionBasis,
]:
    """Get the displacements from origin.

    Parameters
    ----------
    basis : BasisWithLengthLike
    origin : float

    Returns
    -------
    ValueList[FundamentalPositionBasis]
    """
    basis_x = basis_as_fundamental_basisbasis)
    x_points = BasisUtil(basis_x).x_points
    distances = x_points[:, np.newaxis] - x_points[np.newaxis, :] - origin
    max_distance = np.linalg.norm(basis_x.delta_x) / 2
    data = np.remainder((distances + basis_x.delta_x), max_distance) - max_distance
    return {
        "basis": VariadicTupleBasis((basis_x, basis_x), None),
        "data": data.astype(np.complex128),
    }


def _get_displacements_matrix_x_along_axis(
    basis: StackedBasisWithVolumeLike,
    origin: float = 0,
    *,
    axis: int,
) -> SingleBasisOperator[
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
]:
    basis_x = tuple_basis_as_fundamental(basis)
    x_points = BasisUtil(basis_x).x_points_stacked[axis]
    distances = x_points[:, np.newaxis] - x_points[np.newaxis, :] - origin
    delta_x = np.linalg.norm(basis_x.delta_x_stacked[axis])
    max_distance = delta_x / 2
    data = _wrap_displacements(distances, max_distance)
    return {
        "basis": VariadicTupleBasis((basis_x, basis_x), None),
        "data": data.astype(np.complex128),
    }


def get_displacements_matrix_x_stacked(
    basis: StackedBasisWithVolumeLike, origin: tuple[float, ...] | None
) -> SingleBasisOperatorList[
    FundamentalBasis[BasisMetadata],
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
]:
    """Get the displacements from origin."""
    origin = tuple(0.0 for _ in basis.shape) if origin is None else origin
    return operator_list_from_iter(
        tuple(
            _get_displacements_matrix_x_along_axis(basis, o, axis=axis)
            for (axis, o) in enumerate(origin)
        )
    )


def get_total_displacements_matrix_x_stacked(
    basis: StackedBasisWithVolumeLike,
    origin: tuple[float, ...] | None = None,
) -> Operator[
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
]:
    """
    Get a matrix of displacements in x, taken in a periodic fashion.

    Parameters
    ----------
    basis : StackedBasisLike
        _description_

    Returns
    -------
    np.ndarray[tuple[int, int], np.dtype[np.float64]]
        _description_
    """
    displacements = get_displacements_matrix_x_stacked(basis, origin)
    return {
        "basis": displacements["basis"][1],
        "data": np.linalg.norm(
            displacements["data"].reshape(displacements["basis"].shape),
            axis=0,
        ).ravel(),
    }
