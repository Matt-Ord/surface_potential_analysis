from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from scipy.constants import hbar

from surface_potential_analysis.basis.legacy import (
    TupleBasisWithLengthLike,
)
from surface_potential_analysis.basis.util import (
    BasisUtil,
)
from surface_potential_analysis.operator.conversion import (
    convert_operator_to_basis,
)
from surface_potential_analysis.operator.operator import add_operator, as_operator
from surface_potential_analysis.potential.conversion import convert_potential_to_basis
from surface_potential_analysis.stacked_basis.conversion import (
    tuple_basis_as_fundamental,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.legacy import (
        FundamentalTransformedPositionBasis,
        TupleBasisLike,
    )
    from surface_potential_analysis.operator.operator import (
        DiagonalOperator,
        SingleBasisOperator,
    )
    from surface_potential_analysis.potential.potential import Potential

    _L0 = TypeVar("_L0", bound=int)
    _SB0 = TypeVar("_SB0", bound=TupleBasisWithLengthLike[*tuple[Any, ...]])


def hamiltonian_from_potential(
    potential: Potential[_SB0],
) -> SingleBasisOperator[_SB0]:
    """
    Given a potential in some basis get the hamiltonian in the same basis.

    Parameters
    ----------
    potential : Potential[_B0Inv]

    Returns
    -------
    Hamiltonian[_B0Inv]
    """
    converted = convert_potential_to_basis(
        potential, tuple_basis_as_fundamental(potential["basis"])
    )

    return convert_operator_to_basis(
        {
            "basis": VariadicTupleBasis((converted["basis"], converted["basis"]), None),
            "data": np.diag(converted["data"]).reshape(-1),
        },
        VariadicTupleBasis((potential["basis"], potential["basis"]), None),
    )


def hamiltonian_from_mass(
    basis: TupleBasisWithLengthLike[*tuple[Any, ...]],
    mass: float,
    bloch_fraction: np.ndarray[tuple[_L0], np.dtype[np.float64]] | None = None,
) -> DiagonalOperator[
    TupleBasisLike[*tuple[FundamentalTransformedPositionBasis, ...]],
    TupleBasisLike[*tuple[FundamentalTransformedPositionBasis, ...]],
]:
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
    util = BasisUtil(basis)

    bloch_phase = np.tensordot(util.fundamental_dk_stacked, bloch_fraction, axes=(0, 0))
    k_points = util.fundamental_stacked_k_points + bloch_phase[:, np.newaxis]
    energy = np.sum(
        np.square(hbar * k_points) / (2 * mass), axis=0, dtype=np.complex128
    )
    momentum_basis = stacked_basis_as_transformed_basis(basis)

    return {
        "basis": VariadicTupleBasis((momentum_basis, momentum_basis), None),
        "data": energy,
    }


def hamiltonian_from_mass_in_basis(
    basis: _SB0,
    mass: float,
    bloch_fraction: np.ndarray[tuple[_L0], np.dtype[np.float64]] | None = None,
) -> SingleBasisOperator[_SB0]:
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
    hamiltonian = hamiltonian_from_mass(basis, mass, bloch_fraction)
    return convert_operator_to_basis(
        as_operator(hamiltonian), VariadicTupleBasis((basis, basis), None)
    )


def total_surface_hamiltonian(
    potential: Potential[_SB0],
    mass: float,
    bloch_fraction: np.ndarray[tuple[_L0], np.dtype[np.float64]] | None = None,
) -> SingleBasisOperator[_SB0]:
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
    potential_hamiltonian = hamiltonian_from_potential(potential)
    kinetic_hamiltonian = hamiltonian_from_mass_in_basis(
        potential_hamiltonian["basis"][0], mass, bloch_fraction
    )

    return add_operator(kinetic_hamiltonian, potential_hamiltonian)
