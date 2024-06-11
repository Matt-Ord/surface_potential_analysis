from __future__ import annotations

import json
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypedDict,
    TypeVar,
)

import numpy as np

from surface_potential_analysis.basis.basis import (
    FundamentalBasis,
    FundamentalPositionBasis,
    FundamentalPositionBasis2d,
    FundamentalPositionBasis3d,
)
from surface_potential_analysis.basis.stacked_basis import (
    TupleBasis,
    TupleBasisLike,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.util.interpolation import (
    interpolate_points_along_axis_spline,
    interpolate_points_rfftn,
)

if TYPE_CHECKING:
    from pathlib import Path

    from surface_potential_analysis.basis.basis_like import (
        BasisWithLengthLike,
    )


_L0_co = TypeVar("_L0_co", bound=int, covariant=True)
_L1_co = TypeVar("_L1_co", bound=int, covariant=True)
_L2_co = TypeVar("_L2_co", bound=int, covariant=True)

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)

# TODO: report bug in pylance - not possible to use TupleBasisLike[*tuple[Any, ...]]
_SB0 = TypeVar("_SB0", bound=TupleBasisLike)  # type: ignore use TupleBasisLike[*tuple[Any, ...]]


class Potential(TypedDict, Generic[_SB0]):
    """Represents a potential in an evenly spaced grid of points."""

    basis: _SB0
    data: np.ndarray[tuple[int], np.dtype[np.complex128]]


def load_potential(path: Path) -> Potential[Any]:
    """
    Load a potential from the npy format.

    Parameters
    ----------
    path : Path

    Returns
    -------
    Potential[Any, Any, Any]
        _description_
    """
    return np.load(path, allow_pickle=True)[()]  # type: ignore[no-any-return]


def load_potential_grid_json(
    path: Path,
) -> Potential[
    TupleBasisLike[
        BasisWithLengthLike[Any, Any, Literal[3]],
        BasisWithLengthLike[Any, Any, Literal[3]],
        BasisWithLengthLike[Any, Any, Literal[3]],
    ]
]:
    """
    Load a potential from the JSON format.

    Parameters
    ----------
    path : Path

    Returns
    -------
    Potential[Any, Any, Any]
    """

    class SurfacePotentialRaw(TypedDict):
        delta_x0: list[int]
        delta_x1: list[int]
        delta_x2: list[int]
        points: list[list[list[float]]]

    with path.open("r") as f:
        out: SurfacePotentialRaw = json.load(f)
        points = np.array(out["points"])
        return {
            "basis": TupleBasis(
                FundamentalPositionBasis(np.array(out["delta_x0"]), points.shape[0]),
                FundamentalPositionBasis(np.array(out["delta_x1"]), points.shape[1]),
                FundamentalPositionBasis(np.array(out["delta_x2"]), points.shape[2]),
            ),
            "data": np.array(out["points"]),
        }


class UnevenPotential3dZBasis(FundamentalBasis[_L2_co]):
    """Represents the z axis of an uneven potential."""

    def __init__(
        self, z_points: np.ndarray[tuple[_L2_co], np.dtype[np.float64]]
    ) -> None:
        self.z_points = z_points
        super().__init__(z_points.size)  # type:ignore[arg-type]


class UnevenPotential3d(TypedDict, Generic[_L0_co, _L1_co, _L2_co]):
    """Represents a potential unevenly spaced in the z direction."""

    basis: TupleBasisLike[
        FundamentalPositionBasis2d[_L0_co],
        FundamentalPositionBasis2d[_L1_co],
        UnevenPotential3dZBasis[_L2_co],
    ]
    data: np.ndarray[tuple[int], np.dtype[np.complex128]]


def load_uneven_potential(path: Path) -> UnevenPotential3d[Any, Any, Any]:
    """
    Load an uneven potential saved in the npy format.

    Parameters
    ----------
    path : Path

    Returns
    -------
    UnevenPotential[Any, Any, Any]
    """
    return np.load(path, allow_pickle=True)[()]  # type: ignore[no-any-return]


def load_uneven_potential_json(
    path: Path,
) -> UnevenPotential3d[Any, Any, Any]:
    """
    Load an uneven potential saved in the JSON format.

    Parameters
    ----------
    path : Path

    Returns
    -------
    UnevenPotential[Any, Any, Any]
    """

    class SurfacePotentialRaw(TypedDict):
        delta_x0: list[int]
        delta_x1: list[int]
        z_points: list[int]
        points: list[float]

    with path.open("r") as f:
        out: SurfacePotentialRaw = json.load(f)
        points = np.array(out["points"])

        return {
            "basis": TupleBasis(
                FundamentalPositionBasis(np.array(out["delta_x0"]), points.shape[0]),
                FundamentalPositionBasis(np.array(out["delta_x1"]), points.shape[1]),
                UnevenPotential3dZBasis(np.array(out["z_points"])),
            ),
            "data": points,
        }


def normalize_potential(data: Potential[_SB0]) -> Potential[_SB0]:
    """
    Set the minimum of the potential to 0.

    Parameters
    ----------
    data : _GPInv

    Returns
    -------
    _GPInv
    """
    points = data["data"]
    normalized_points = points - points.min()
    return {"data": normalized_points, "basis": data["basis"]}  # type: ignore[return-value,misc]


def truncate_potential(
    data: Potential[_SB0],
    *,
    cutoff: float = 2e-17,
    n: int = 1,
    offset: float = 2e-18,
) -> Potential[_SB0]:
    """
    Reduce the maximum energy by taking the transformation.

    :math:`cutoff * np.log(1 + ((E + offset) / cutoff) ** n) ** (1 / n) - offset`

    For :math:`E << Cutoff` the energy is left unchanged. This can be useful to
    prevent the energy interpolation process from producing rabid oscillations
    """
    points = (
        cutoff * np.log(1 + ((data["data"] + offset) / cutoff) ** n) ** (1 / n) - offset
    )
    return {"data": points, "basis": data["basis"]}  # type: ignore[return-value,misc]


def undo_truncate_potential(
    data: Potential[_SB0], *, cutoff: float = 2e-17, n: int = 1, offset: float = 2e-18
) -> Potential[_SB0]:
    """Reverses truncate_potential."""
    points = cutoff * (np.exp((data["data"] + offset) / cutoff) - 1) ** (1 / n) - offset
    return {"data": points, "basis": data["basis"]}  # type: ignore[return-value]


def interpolate_uneven_potential(
    data: UnevenPotential3d[int, int, int], shape: tuple[_L0Inv, _L1Inv, _L2Inv]
) -> Potential[
    TupleBasisLike[
        FundamentalPositionBasis3d[_L0Inv],
        FundamentalPositionBasis3d[_L1Inv],
        FundamentalPositionBasis3d[_L2Inv],
    ]
]:
    """
    Interpolate an energy grid using the fourier method.

    Makes use of a fourier transform to increase the number of points
    in the xy plane of the energy grid, and a cubic spline to interpolate in the z direction
    """
    util = BasisUtil(data["basis"])
    xy_interpolated = interpolate_points_rfftn(
        data["data"].reshape(util.shape).astype(np.float64),
        s=(shape[0], shape[1]),
        axes=(0, 1),
    )
    interpolated = interpolate_points_along_axis_spline(
        xy_interpolated, data["basis"][2].z_points, shape[2], axis=2
    )
    delta_x_0 = np.concatenate([data["basis"][0].delta_x, [0]])
    delta_x_1 = np.concatenate([data["basis"][1].delta_x, [0]])
    return {
        "basis": TupleBasis(
            FundamentalPositionBasis(delta_x_0, shape[0]),
            FundamentalPositionBasis(delta_x_1, shape[1]),
            FundamentalPositionBasis(
                np.array(
                    [
                        0,
                        0,
                        data["basis"][2].z_points[-1] - data["basis"][2].z_points[0],
                    ]
                ),
                shape[2],
            ),
        ),
        "data": interpolated.reshape(-1),  # type: ignore[typeddict-item]
    }


def mock_even_potential(
    uneven: UnevenPotential3d[_L0Inv, _L1Inv, _L2Inv],
) -> Potential[
    TupleBasisLike[
        FundamentalPositionBasis3d[_L0Inv],
        FundamentalPositionBasis3d[_L1Inv],
        FundamentalPositionBasis3d[_L2Inv],
    ]
]:
    """
    Generate a fake even potential from an uneven potential.

    Parameters
    ----------
    uneven : UnevenPotential[_L0Inv, _L1Inv, _L2Inv]

    Returns
    -------
    Potential[_L0Inv, _L1Inv, _L2Inv]
    """
    return {
        "basis": TupleBasis(
            FundamentalPositionBasis(
                np.array([*uneven["basis"][0].delta_x, 0]), uneven["basis"][0].n
            ),
            FundamentalPositionBasis(
                np.array([*uneven["basis"][1].delta_x, 0]), uneven["basis"][1].n
            ),
            FundamentalPositionBasis(
                np.array([0, 0, 1], dtype=float),
                uneven["basis"][2].fundamental_n,  # type:ignore[arg-type]
            ),
        ),
        "data": uneven["data"].reshape(-1),
    }
