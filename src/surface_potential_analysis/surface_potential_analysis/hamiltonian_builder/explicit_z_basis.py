from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Literal, TypeVar

import numpy as np

from surface_potential_analysis.axis import (
    Axis3dUtil,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.hamiltonian_builder.momentum_basis import (
    hamiltonian_from_mass,
)

if TYPE_CHECKING:
    from surface_potential_analysis.axis.axis import (
        ExplicitAxis3d,
        MomentumAxis3d,
    )
    from surface_potential_analysis.basis.basis import (
        Basis3d,
    )
    from surface_potential_analysis.operator import HamiltonianWith3dBasis
    from surface_potential_analysis.potential import (
        FundamentalPositionBasisPotential3d,
    )

_N0Inv = TypeVar("_N0Inv", bound=int)
_N1Inv = TypeVar("_N1Inv", bound=int)
_N2Inv = TypeVar("_N2Inv", bound=int)
_NF0Inv = TypeVar("_NF0Inv", bound=int)
_NF1Inv = TypeVar("_NF1Inv", bound=int)
_NF2Inv = TypeVar("_NF2Inv", bound=int)


class PotentialSizeError(Exception):
    """Error thrown when the potential is too small."""

    def __init__(self, axis: int, required: int, actual: int) -> None:
        super().__init__(
            f"Potential does not have enough resolution in x{axis} direction"
            f"required {required} actual {actual}"
        )


class _SurfaceHamiltonianUtil(
    Generic[_N0Inv, _N1Inv, _N2Inv, _NF0Inv, _NF1Inv, _NF2Inv]
):
    _potential: FundamentalPositionBasisPotential3d[_NF0Inv, _NF1Inv, _NF2Inv]

    _basis: Basis3d[
        MomentumAxis3d[_NF0Inv, _N0Inv],
        MomentumAxis3d[_NF1Inv, _N1Inv],
        ExplicitAxis3d[_NF2Inv, _N2Inv],
    ]
    _mass: float

    def __init__(
        self,
        potential: FundamentalPositionBasisPotential3d[_NF0Inv, _NF1Inv, _NF2Inv],
        basis: Basis3d[
            MomentumAxis3d[_NF0Inv, _N0Inv],
            MomentumAxis3d[_NF1Inv, _N1Inv],
            ExplicitAxis3d[_NF2Inv, _N2Inv],
        ],
        mass: float,
    ) -> None:
        self._potential = potential
        self._basis = basis
        self._mass = mass
        if 2 * (self._basis[0].n - 1) > self._potential["basis"][0].n:
            raise PotentialSizeError(
                0, 2 * (self._basis[0].n - 1), self._potential["basis"][0].n
            )

        if 2 * (self._basis[1].n - 1) > self._potential["basis"][1].n:
            raise PotentialSizeError(
                1, 2 * (self._basis[1].n - 1), self._potential["basis"][1].n
            )

    @property
    def points(
        self,
    ) -> np.ndarray[tuple[_NF0Inv, _NF1Inv, _NF2Inv], np.dtype[np.complex_]]:
        return self._potential["vector"].reshape(  # type: ignore[no-any-return]
            BasisUtil(self._potential["basis"]).shape
        )

    @property
    def nx(self) -> int:
        return self.points.shape[0]  # type: ignore[no-any-return]

    @property
    def ny(self) -> int:
        return self.points.shape[1]  # type: ignore[no-any-return]

    @property
    def nz(self) -> int:
        return self.points.shape[2]  # type: ignore[no-any-return]

    @property
    def dz(self) -> float:
        util = Axis3dUtil(self._potential["basis"][2])
        return np.linalg.norm(util.fundamental_dx)  # type: ignore[return-value]

    def hamiltonian(
        self, _bloch_phase: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> HamiltonianWith3dBasis[
        MomentumAxis3d[_NF0Inv, _N0Inv],
        MomentumAxis3d[_NF1Inv, _N1Inv],
        ExplicitAxis3d[_NF2Inv, _N2Inv],
    ]:
        raise NotImplementedError

    def _calculate_diagonal_energy_fundamental_x2(
        self, bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float_]]:
        kinetic_xy = hamiltonian_from_mass(
            self._basis[0:2], self._mass, bloch_fraction[0:2]
        )

        return kinetic_xy["array"]

    def get_ft_potential(
        self,
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]:
        return np.fft.ifft2(self.points, axes=(0, 1, 2), norm="ortho")  # type: ignore[no-any-return]


def total_surface_hamiltonian(
    potential: FundamentalPositionBasisPotential3d[_NF0Inv, _NF1Inv, _NF2Inv],
    bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
    basis: Basis3d[
        MomentumAxis3d[_NF0Inv, _N0Inv],
        MomentumAxis3d[_NF1Inv, _N1Inv],
        ExplicitAxis3d[_NF2Inv, _N2Inv],
    ],
    mass: float,
) -> HamiltonianWith3dBasis[
    MomentumAxis3d[_NF0Inv, _N0Inv],
    MomentumAxis3d[_NF1Inv, _N1Inv],
    ExplicitAxis3d[_NF2Inv, _N2Inv],
]:
    """
    Calculate a hamiltonian using the given basis.

    Parameters
    ----------
    potential : Potential[_L0, _L1, _L2]
    bloch_fraction : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    basis : Basis3d[TruncatedBasis[_L3, MomentumBasis[_L0]], TruncatedBasis[_L4, MomentumBasis[_L1]], ExplicitBasis[_L5, MomentumBasis[_L2]]]
    mass : float

    Returns
    -------
    HamiltonianWithBasis[TruncatedBasis[_L3, MomentumBasis[_L0]], TruncatedBasis[_L4, MomentumBasis[_L1]], ExplicitBasis[_L5, MomentumBasis[_L2]]]
    """
    util = _SurfaceHamiltonianUtil(potential, basis, mass)
    bloch_phase = np.tensordot(
        BasisUtil(util._basis).fundamental_dk,  # noqa: SLF001
        bloch_fraction,
        axes=(0, 0),
    )
    return util.hamiltonian(bloch_phase)
