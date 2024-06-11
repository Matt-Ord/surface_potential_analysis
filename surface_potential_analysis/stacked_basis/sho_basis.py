"""Utility functions to help with the generation of a sho basis."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal, TypedDict, TypeVar

import numpy as np
import scipy
import scipy.special
from scipy.constants import hbar

from surface_potential_analysis.basis.basis import (
    FundamentalPositionBasis,
)
from surface_potential_analysis.basis.explicit_basis import (
    ExplicitBasisWithLength,
)
from surface_potential_analysis.basis.stacked_basis import TupleBasis
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.stacked_basis.potential_basis import (
    PotentialBasisConfig,
    get_potential_basis_config_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis_like import BasisWithLengthLike3d

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_LF0Inv = TypeVar("_LF0Inv", bound=int)


def calculate_sho_wavefunction(
    x_points: np.ndarray[tuple[_L0Inv], np.dtype[np.complex128]],
    sho_omega: float,
    mass: float,
    n: int,
) -> np.ndarray[tuple[_L0Inv], np.dtype[np.complex128]]:
    """
    Calculate the value of a sho wavefunction at x_points.

    Parameters
    ----------
    x_points : np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]
        Points to evaluate
    sho_omega : float
        Omega in 1/2 m omega ** 2 x **2
    mass : float
        Mass of the oscillator
    n : int
        index of the wavefunction to calculate

    Returns
    -------
    np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]
        A list of values for the wavefunction in position basis
    """
    norm = (sho_omega * mass / hbar) ** 0.5
    normalized_x = x_points * norm

    prefactor = math.sqrt((norm / (2**n)) / (math.factorial(n) * math.sqrt(math.pi)))
    hermite = scipy.special.eval_hermite(n, normalized_x)  # type: ignore bad libary types
    exponential = np.exp(-np.square(normalized_x) / 2)
    return prefactor * hermite * exponential  # type: ignore[no-any-return]


def calculate_x_distances(
    parent: BasisWithLengthLike3d[_L0Inv, Any],
    x_origin: np.ndarray[tuple[Literal[3]], np.dtype[np.float64]],
) -> np.ndarray[tuple[_L0Inv], np.dtype[np.complex128]]:
    """Given a basis, calculate x distances with a projected value of zero at x_origin."""
    util = BasisUtil(parent)
    x_points = util.fundamental_x_points

    x0_norm = util.delta_x.copy() / np.linalg.norm(util.delta_x)
    distances_origin = np.dot(x0_norm, x_origin)
    return np.dot(x0_norm, x_points) + distances_origin  # type: ignore[no-any-return]


class SHOBasisConfig(TypedDict):
    """Configuration for a SHO Basis."""

    sho_omega: float
    mass: float
    x_origin: np.ndarray[tuple[Literal[3]], np.dtype[np.float64]]


def get_sho_potential_basis_config(
    parent: BasisWithLengthLike3d[_L0Inv, _LF0Inv], config: SHOBasisConfig, n: _L1Inv
) -> PotentialBasisConfig[FundamentalPositionBasis[_LF0Inv, Literal[1]], _L1Inv]:
    """
    Get a potential basis config assuming a SHO oscillator.

    Parameters
    ----------
    parent : FundamentalBasis[_L0Inv]
    config : SHOBasisConfig
    n : _L1Inv

    Returns
    -------
    PotentialTupleBasisLike[tuple[_L0Inv, _L1Inv]
    """
    delta_x1 = (
        np.array([0, 1, 0])
        if np.allclose([1, 0, 0], parent.delta_x)
        else np.array([1, 0, 0])
    )
    delta_x2 = np.cross(parent.delta_x, delta_x1)
    delta_x2 /= np.linalg.norm(delta_x2)

    basis = TupleBasis(
        FundamentalPositionBasis(np.array([np.linalg.norm(parent.delta_x)]), parent.n),
    )
    x_distances = calculate_x_distances(parent, config["x_origin"])

    return {
        "potential": {
            "basis": basis,
            "data": 0.5 * config["mass"] * config["sho_omega"] ** 2 * x_distances**2,
        },
        "n": n,
        "mass": config["mass"],
    }


def sho_basis_3d_from_config(
    parent: BasisWithLengthLike3d[_LF0Inv, _L0Inv], config: SHOBasisConfig, n: _L0Inv
) -> ExplicitBasisWithLength[_LF0Inv, _L0Inv, Literal[3]]:
    """
    Calculate the exact sho basis for a given basis, by directly diagonalizing the sho wavefunction in this basis.

    The resulting wavefunction
    is guaranteed to be orthonormal in this basis.

    Parameters
    ----------
    parent : _FBInv
    config : SHOBasisConfig
    n : _L0Inv

    Returns
    -------
    ExplicitBasis[_L0Inv, _FBInv]
    """
    potential_basis_config = get_sho_potential_basis_config(parent, config, n)
    axis = get_potential_basis_config_basis(potential_basis_config)
    return ExplicitBasisWithLength(parent.delta_x, axis.vectors)


def infinate_sho_basis_3d_from_config(
    parent: BasisWithLengthLike3d[_LF0Inv, _L1Inv], config: SHOBasisConfig, n: _L0Inv
) -> ExplicitBasis3d[_LF0Inv, _L0Inv]:
    """
    Generate an explicit sho basis assuming an infinate extent in the z direction.

    Parameters
    ----------
    parent : PositionBasis[_L1Inv]
    config : SHOBasisConfig
    n : _L0Inv

    Returns
    -------
    ExplicitBasis[_L0Inv, PositionBasis[_L1Inv]]
    """
    x_distances = calculate_x_distances(parent, config["x_origin"])

    vectors = np.array(
        [
            calculate_sho_wavefunction(
                x_distances, n=i, mass=config["mass"], sho_omega=config["sho_omega"]
            )
            for i in range(n)
        ]
    )
    util = BasisUtil(parent)
    vectors *= np.sqrt(np.linalg.norm(util.fundamental_dx))
    return ExplicitBasisWithLength(parent.delta_x, vectors)
