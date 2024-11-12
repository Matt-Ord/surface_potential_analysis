from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.constants import hbar
from slate.basis._basis import Basis
from slate.basis.stacked._diagonal_basis import DiagonalBasis, diagonal_basis
from slate.basis.stacked._tuple_basis import as_tuple_basis
from slate.basis.transformed import fundamental_transformed_tuple_basis_from_metadata
from slate.metadata.length import fundamental_stacked_dk, fundamental_stacked_k_points
from slate.metadata.stacked import StackedMetadata

from surface_potential_analysis.basis.util import (
    BasisUtil,
)
from surface_potential_analysis.operator.operator import (
    Operator,
)

if TYPE_CHECKING:
    from slate.metadata import VolumeMetadata

    from surface_potential_analysis.potential.potential import (
        Potential,
    )


def hamiltonian_from_mass(
    basis: Basis[StackedMetadata[Any, Any], Any],
    mass: float,
    bloch_fraction: np.ndarray[Any, np.dtype[np.float64]] | None = None,
) -> Operator[Any, DiagonalBasis[np.complex128, Any, Any, None]]:
    """
    Given a mass and a basis calculate the kinetic part of the Hamiltonian.

    Parameters
    ----------
    basis : _B0Inv
    mass : float
    bloch_fraction : np.ndarray[tuple[int], np.dtype[np.float_]] | None, optional
        bloch phase, by default None

    Returns
    -------
    Hamiltonian[_B0Inv]
    """
    bloch_fraction = np.zeros(basis.n_dim) if bloch_fraction is None else bloch_fraction
    BasisUtil(basis)

    metadata = basis.metadata

    bloch_phase = np.tensordot(
        fundamental_stacked_dk(metadata), bloch_fraction, axes=(0, 0)
    )
    k_points = fundamental_stacked_k_points(metadata) + bloch_phase[:, np.newaxis]
    energy = np.sum(
        np.square(hbar * k_points) / (2 * mass), axis=0, dtype=np.complex128
    )
    momentum_basis = fundamental_transformed_tuple_basis_from_metadata(metadata)

    return Operator(
        diagonal_basis((momentum_basis, momentum_basis.conjugate_basis()), None), energy
    )


def hamiltonian_from_mass_in_basis[_B0: Basis[StackedMetadata[Any, Any], Any]](
    basis: _B0,
    mass: float,
    bloch_fraction: np.ndarray[Any, np.dtype[np.float64]] | None = None,
) -> Operator[np.complex128, _B0]:
    """
    Given a mass and a basis calculate the kinetic part of the Hamiltonian.

    Parameters
    ----------
    basis : _B0Inv
    mass : float
    bloch_fraction : np.ndarray[tuple[int], np.dtype[np.float_]] | None, optional
        bloch phase, by default None

    Returns
    -------
    Hamiltonian[_B0Inv]
    """
    hamiltonian = hamiltonian_from_mass(basis[0], mass, bloch_fraction)
    return hamiltonian.with_basis(basis)


def total_surface_hamiltonian(
    potential: Potential[np.complex128],
    mass: float,
    bloch_fraction: np.ndarray[Any, np.dtype[np.float64]] | None = None,
) -> Operator[np.complex128, Basis[StackedMetadata[VolumeMetadata, None], Any]]:
    """
    Calculate the total hamiltonian in momentum basis for a given potential and mass.

    Parameters
    ----------
    potential : Potential[_L0, _L1, _L2]
    mass : float
    bloch_fraction : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]

    Returns
    -------
    MomentumBasisHamiltonian[_L0, _L1, _L2]
    """
    basis = as_tuple_basis(potential.basis.inner)
    potential_hamiltonian = potential.as_operator().with_basis(basis)
    kinetic_hamiltonian = hamiltonian_from_mass_in_basis(basis, mass, bloch_fraction)

    return kinetic_hamiltonian + potential_hamiltonian
