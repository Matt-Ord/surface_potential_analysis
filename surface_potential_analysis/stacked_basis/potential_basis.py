from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Literal, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis.legacy import (
    BasisWithLengthLike,
    ExplicitStackedBasisWithLength,
    FundamentalBasis,
    FundamentalPositionBasis,
    FundamentalPositionBasis1d,
    StackedBasisWithVolumeLike,
    TupleBasis,
    TupleBasisWithLengthLike,
)
from surface_potential_analysis.hamiltonian_builder.momentum_basis import (
    total_surface_hamiltonian,
)
from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_eigenvectors_hermitian,
)

if TYPE_CHECKING:
    from surface_potential_analysis.potential.potential import (
        Potential,
    )
    from surface_potential_analysis.state_vector.eigenstate_list import (
        EigenstateList,
    )


_B1d0Inv = TypeVar("_B1d0Inv", bound=BasisWithLengthLike[Any, Any, Literal[1]])
_B3d0 = TypeVar("_B3d0", bound=BasisWithLengthLike[Any, Any, Literal[3]])
_L1_co = TypeVar("_L1_co", covariant=True, bound=int)


class PotentialBasisConfig(TypedDict, Generic[_B1d0Inv, _L1_co]):
    """Configures the generation of an explicit basis from a given potential."""

    potential: Potential[TupleBasisWithLengthLike[_B1d0Inv]]
    mass: float
    n: _L1_co


_N0Inv = TypeVar("_N0Inv", bound=int)


def get_potential_basis_config_eigenstates(
    config: PotentialBasisConfig[_B1d0Inv, _N0Inv],
    *,
    bloch_fraction: float | None = None,
) -> EigenstateList[FundamentalBasis[_N0Inv], StackedBasisWithVolumeLike]:
    """
    Get the eigenstates of the potential, as used in the final basis.

    Parameters
    ----------
    config : PotentialBasisConfig[_L0Inv]

    Returns
    -------
    EigenstateList[PositionTupleBasisLike[tuple[_L0Inv, Literal[1], Literal[1]]]
    """
    bloch_fraction = 0 if bloch_fraction is None else bloch_fraction
    hamiltonian = total_surface_hamiltonian(
        config["potential"], config["mass"], np.array([bloch_fraction])
    )
    return calculate_eigenvectors_hermitian(  # type: ignore FundamentalBasis[BasisMetadata] not FundamentalBasis[_N0Inv]
        hamiltonian,
        subset_by_index=(0, config["n"] - 1),  # type: ignore cannot infer type of hamiltonian properly
    )


def get_potential_basis_config_basis(
    config: PotentialBasisConfig[_B1d0Inv, _N0Inv],
) -> ExplicitStackedBasisWithLength[
    FundamentalBasis[_N0Inv],
    StackedBasisWithVolumeLike,
]:
    """
    Get the explicit basis for the potential basis config.

    Parameters
    ----------
    config : PotentialBasisConfig[_L0Inv, _L1Inv]

    Returns
    -------
    ExplicitBasis[_L1Inv, PositionBasis[_L0Inv]]
    """
    eigenstates = get_potential_basis_config_eigenstates(config)
    return ExplicitStackedBasisWithLength(eigenstates)


def select_minimum_potential_3d(
    potential: Potential[TupleBasisWithLengthLike[Any, Any, _B3d0]],
) -> Potential[TupleBasisWithLengthLike[FundamentalPositionBasis1d[Any]]]:
    """
    Given a 3D potential in the standard configuration select the minimum potential.

    Parameters
    ----------
    potential : Potential[tuple[Any, Any, FundamentalPositionBasis3d[_L0]]]

    Returns
    -------
    Potential[tuple[FundamentalPositionBasis1d[_L0]]]
    """
    shape = potential["basis"].shape
    arg_min = np.unravel_index(np.argmin(potential["data"]), shape)
    return {
        "basis": TupleBasis(
            FundamentalPositionBasis(
                np.array([potential["basis"][2].delta_x[2]]), shape[2]
            ),
        ),
        "data": potential["data"].reshape(shape)[arg_min[0], arg_min[1], :],
    }
