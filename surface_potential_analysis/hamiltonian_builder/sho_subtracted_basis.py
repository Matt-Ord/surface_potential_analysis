from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

import hamiltonian_generator
import numpy as np
from scipy.constants import hbar

from surface_potential_analysis.basis.basis import (
    FundamentalPositionBasis,
    FundamentalPositionBasis3d,
    TransformedPositionBasis,
    TransformedPositionBasis3d,
)
from surface_potential_analysis.basis.stacked_basis import (
    TupleBasis,
    TupleBasisLike,
)
from surface_potential_analysis.basis.util import (
    BasisUtil,
)
from surface_potential_analysis.stacked_basis.sho_basis import (
    SHOBasisConfig,
    calculate_x_distances,
    infinate_sho_basis_3d_from_config,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.explicit_basis import ExplicitBasis3d
    from surface_potential_analysis.operator import SingleBasisOperator
    from surface_potential_analysis.potential.potential import Potential


_N0Inv = TypeVar("_N0Inv", bound=int)
_N1Inv = TypeVar("_N1Inv", bound=int)
_N2Inv = TypeVar("_N2Inv", bound=int)
_NF0Inv = TypeVar("_NF0Inv", bound=int)
_NF1Inv = TypeVar("_NF1Inv", bound=int)
_NF2Inv = TypeVar("_NF2Inv", bound=int)


class _SurfaceHamiltonianUtil(
    Generic[_N0Inv, _N1Inv, _N2Inv, _NF0Inv, _NF1Inv, _NF2Inv]
):
    _potential: Potential[
        TupleBasisLike[
            FundamentalPositionBasis3d[_NF0Inv],
            FundamentalPositionBasis3d[_NF1Inv],
            FundamentalPositionBasis3d[_NF2Inv],
        ]
    ]

    _config: SHOBasisConfig

    _resolution: tuple[_N0Inv, _N1Inv, _N2Inv]

    def __init__(
        self,
        potential: Potential[
            TupleBasisLike[
                FundamentalPositionBasis3d[_NF0Inv],
                FundamentalPositionBasis3d[_NF1Inv],
                FundamentalPositionBasis3d[_NF2Inv],
            ]
        ],
        config: SHOBasisConfig,
        resolution: tuple[_N0Inv, _N1Inv, _N2Inv],
    ) -> None:
        self._potential = potential
        self._config = config
        self._resolution = resolution
        if 2 * (self._resolution[0] - 1) > self._potential["basis"][0].n:
            raise AssertionError(  # noqa: TRY003
                "Not have enough resolution in x0"  # noqa: EM101
            )

        if 2 * (self._resolution[1] - 1) > self._potential["basis"][1].n:
            raise AssertionError(  # noqa: TRY003
                "Not have enough resolution in x1"  # noqa: EM101
            )

    @property
    def points(
        self,
    ) -> np.ndarray[tuple[_NF0Inv, _NF1Inv, _NF2Inv], np.dtype[np.float64]]:
        return np.real(  # type: ignore[no-any-return]
            self._potential["data"].reshape(BasisUtil(self._potential["basis"]).shape)
        )

    @property
    def z_offset(self) -> float:
        return self._config["x_origin"][2]  # type: ignore[return-value]

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
        util = BasisUtil(self._potential["basis"][2])
        return np.linalg.norm(util.fundamental_dx)  # type: ignore[return-value]

    @property
    def z_distances(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return calculate_x_distances(
            self._potential["basis"][2],
            self._config["x_origin"],  # type: ignore[arg-type]
        )

    @property
    def basis(
        self,
    ) -> TupleBasis[
        TransformedPositionBasis3d[_NF0Inv, _N0Inv],
        TransformedPositionBasis3d[_NF1Inv, _N1Inv],
        ExplicitBasis3d[_NF2Inv, _N2Inv],
    ]:
        return TupleBasis(
            TransformedPositionBasis(
                self._potential["basis"][0].delta_x,
                self._resolution[0],
                self._potential["basis"][0].n,
            ),
            TransformedPositionBasis(
                self._potential["basis"][1].delta_x,
                self._resolution[1],
                self._potential["basis"][1].n,
            ),
            infinate_sho_basis_3d_from_config(
                FundamentalPositionBasis(
                    self._potential["basis"][2].delta_x,
                    self._potential["basis"][2].n,
                ),
                self._config,
                self._resolution[2],
            ),
        )

    def hamiltonian(
        self, bloch_phase: np.ndarray[tuple[Literal[3]], np.dtype[np.float64]]
    ) -> SingleBasisOperator[
        TupleBasisLike[
            TransformedPositionBasis3d[_NF0Inv, _N0Inv],
            TransformedPositionBasis3d[_NF1Inv, _N1Inv],
            ExplicitBasis3d[_NF2Inv, _N2Inv],
        ]
    ]:
        diagonal_energies = np.diag(self._calculate_diagonal_energy(bloch_phase))
        other_energies = self._calculate_off_diagonal_energies_fast()

        energies = diagonal_energies + other_energies

        return {
            "data": energies.reshape(-1),
            "basis": TupleBasis(self.basis, self.basis),
        }

    @cached_property
    def eigenstate_indexes(
        self,
    ) -> np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]]:
        x0t, x1t, zt = np.meshgrid(
            BasisUtil(self.basis[0]).nk_points,  # type: ignore[misc]
            BasisUtil(self.basis[1]).nk_points,  # type: ignore[misc]
            BasisUtil(self.basis[2]).nx_points,  # type: ignore[misc]
            indexing="ij",
        )
        return np.array([x0t.ravel(), x1t.ravel(), zt.ravel()])  # type: ignore[no-any-return]

    def _calculate_diagonal_energy(
        self, bloch_phase: np.ndarray[tuple[Literal[3]], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        k0_coords, k1_coords, nkz_coords = self.eigenstate_indexes

        util = BasisUtil(self.basis)

        dk0 = util.dk_stacked[0]
        dk1 = util.dk_stacked[1]
        mass = self._config["mass"]
        sho_omega = self._config["sho_omega"]

        kx_points = dk0[0] * k0_coords + dk1[0] * k1_coords + bloch_phase[0]
        x_energy = (hbar * kx_points) ** 2 / (2 * mass)
        ky_points = dk0[1] * k0_coords + dk1[1] * k1_coords + bloch_phase[1]
        y_energy = (hbar * ky_points) ** 2 / (2 * mass)
        z_energy = (hbar * sho_omega) * (nkz_coords + 0.5)
        return x_energy + y_energy + z_energy  # type: ignore[no-any-return]

    def get_sho_potential(self) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
        mass = self._config["mass"]
        sho_omega = self._config["sho_omega"]
        return 0.5 * mass * sho_omega**2 * np.square(self.z_distances)  # type: ignore[no-any-return]

    def get_sho_subtracted_points(
        self,
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
        return np.subtract(self.points, self.get_sho_potential())  # type: ignore[no-any-return]

    def get_ft_potential(
        self,
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]:
        subtracted_potential = self.get_sho_subtracted_points()
        return np.fft.ifft2(subtracted_potential, axes=(0, 1))  # type: ignore[no-any-return]

    def _calculate_off_diagonal_energies_fast(
        self,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
        mass = self._config["mass"]
        sho_omega = self._config["sho_omega"]
        return np.array(  # type: ignore[no-any-return]
            hamiltonian_generator.calculate_off_diagonal_energies(
                self.get_ft_potential().tolist(),
                self._resolution,
                self.dz,
                mass,
                sho_omega,
                self.z_offset,
            )
        )

    def _calculate_off_diagonal_entry(
        self, nz1: int, nz2: int, ndk0: int, ndk1: int
    ) -> float:
        """Calculate the off diagonal energy using the 'folded' points ndk0, ndk1."""
        ft_pot_points = self.get_ft_potential()[ndk0, ndk1]
        hermite1 = BasisUtil(self.basis[2]).vectors[nz1]
        hermite2 = BasisUtil(self.basis[2]).vectors[nz2]

        fourier_transform = float(np.sum(hermite1 * hermite2 * ft_pot_points))

        return self.dz * fourier_transform

    def _calculate_off_diagonal_energies(
        self,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
        n_coordinates = len(self.eigenstate_indexes.T)
        hamiltonian = np.zeros(shape=(n_coordinates, n_coordinates))

        for index1, [nk0_0, nk1_0, nz1] in enumerate(self.eigenstate_indexes.T):
            for index2, [nk0_1, nk1_1, nz2] in enumerate(self.eigenstate_indexes.T):
                # Number of jumps in units of dkx for this matrix element

                # As k-> k+ Nx * dkx the ft potential is left unchanged
                # Therefore we 'wrap round' ndkx into the region 0<ndkx<Nx
                # Where Nx is the number of x points

                # In reality we want to make sure ndkx < Nx (ie we should
                # make sure we generate enough points in the interpolation)
                ndk0 = (nk0_1 - nk0_0) % self.nx
                ndk1 = (nk1_1 - nk1_0) % self.ny

                hamiltonian[index1, index2] = self._calculate_off_diagonal_entry(
                    nz1, nz2, ndk0, ndk1
                )

        return hamiltonian  # type: ignore[no-any-return]


def total_surface_hamiltonian(
    potential: Potential[
        TupleBasisLike[
            FundamentalPositionBasis3d[_NF0Inv],
            FundamentalPositionBasis3d[_NF1Inv],
            FundamentalPositionBasis3d[_NF2Inv],
        ]
    ],
    config: SHOBasisConfig,
    bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float64]],
    resolution: tuple[_N0Inv, _N1Inv, _N2Inv],
) -> SingleBasisOperator[
    TupleBasisLike[
        TransformedPositionBasis3d[_NF0Inv, _N0Inv],
        TransformedPositionBasis3d[_NF1Inv, _N1Inv],
        ExplicitBasis3d[_NF2Inv, _N2Inv],
    ]
]:
    """
    Calculate a hamiltonian using the infinite sho basis.

    Parameters
    ----------
    potential : Potential[_L0, _L1, _L2]
    config : SHOBasisConfig
    bloch_fraction : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    resolution : tuple[_L3, _L4, _L5]

    Returns
    -------
    HamiltonianWithBasis[TruncatedBasis[_L3, MomentumBasis[_L0]], TruncatedBasis[_L4, MomentumBasis[_L1]], ExplicitBasis[_L5, PositionBasis[_L2]]]
    """
    util = _SurfaceHamiltonianUtil(potential, config, resolution)
    bloch_phase = np.tensordot(
        BasisUtil(util.basis).fundamental_dk_stacked,
        bloch_fraction,
        axes=(0, 0),
    )
    return util.hamiltonian(bloch_phase)
