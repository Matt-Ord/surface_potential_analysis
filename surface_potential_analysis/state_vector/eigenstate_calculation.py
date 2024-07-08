from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import scipy.linalg

from surface_potential_analysis.basis.basis import FundamentalBasis
from surface_potential_analysis.basis.basis_like import BasisLike, BasisWithLengthLike
from surface_potential_analysis.basis.explicit_basis import (
    ExplicitBasis,
    ExplicitStackedBasisWithLength,
)
from surface_potential_analysis.basis.stacked_basis import (
    TupleBasis,
    TupleBasisLike,
)
from surface_potential_analysis.operator.conversion import convert_operator_to_basis
from surface_potential_analysis.util.decorators import timed

if TYPE_CHECKING:
    from surface_potential_analysis.operator.operator import (
        Operator,
        SingleBasisOperator,
    )
    from surface_potential_analysis.state_vector.eigenstate_collection import (
        EigenstateList,
        ValueList,
    )
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )
    from surface_potential_analysis.types import IntLike_co

    from .state_vector import (
        StateDualVector,
        StateVector,
    )

_B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
_B1 = TypeVar("_B1", bound=BasisLike[Any, Any])
_B2 = TypeVar("_B2", bound=BasisLike[Any, Any])
_B3 = TypeVar("_B3", bound=BasisLike[Any, Any])


@timed
def calculate_eigenvectors_hermitian(
    hamiltonian: SingleBasisOperator[_B0],
    subset_by_index: tuple[IntLike_co, IntLike_co] | None = None,
) -> EigenstateList[FundamentalBasis[int], _B0]:
    """Get a list of eigenstates for a given operator, assuming it is hermitian."""
    eigenvalues, vectors = scipy.linalg.eigh(
        hamiltonian["data"].reshape(hamiltonian["basis"].shape),
        subset_by_index=subset_by_index,
    )
    return {
        "basis": TupleBasis(
            FundamentalBasis(np.size(eigenvalues)), hamiltonian["basis"][0]
        ),
        "data": np.transpose(vectors).reshape(-1),
        "eigenvalue": np.array(eigenvalues),
    }


def calculate_eigenvectors(
    hamiltonian: SingleBasisOperator[_B0],
) -> EigenstateList[FundamentalBasis[int], _B0]:
    """Get a list of eigenstates for a given operator, assuming it is hermitian."""
    eigenvalues, vectors = np.linalg.eig(
        hamiltonian["data"].reshape(hamiltonian["basis"].shape),
    )
    return {
        "basis": TupleBasis(
            FundamentalBasis(eigenvalues.size), hamiltonian["basis"][0]
        ),
        "data": np.transpose(vectors).reshape(-1),
        "eigenvalue": eigenvalues,
    }


def calculate_expectation(
    operator: Operator[_B0, _B1], state: StateVector[_B2]
) -> complex:
    """
    Calculate the energy of the given eigenvector.

    Parameters
    ----------
    hamiltonian : Hamiltonian[_B3d0Inv]
    eigenstate : Eigenstate[_B3d0Inv]

    Returns
    -------
    complex
        The energy of the Eigenvector given Hamiltonian
    """
    converted = convert_operator_to_basis(
        operator, TupleBasis(state["basis"], state["basis"])
    )

    return np.einsum(
        "i,ij,j->",
        np.conj(state["data"]),
        converted["data"].reshape(converted["basis"].shape),
        state["data"],
    )


def calculate_expectation_list(
    operator: Operator[_B0, _B3],
    states: StateVectorList[_B1, _B2],
) -> ValueList[_B1]:
    """
    Calculate the energy of the given eigenvector.

    Parameters
    ----------
    hamiltonian : Hamiltonian[_B3d0Inv]
    eigenstate : Eigenstate[_B3d0Inv]

    Returns
    -------
    complex
        The energy of the Eigenvector given Hamiltonian
    """
    converted = convert_operator_to_basis(
        operator, TupleBasis(states["basis"][1], states["basis"][1])
    )
    data = np.einsum(
        "ij,jk,ik->i",
        np.conj(states["data"].reshape(states["basis"].shape)),
        converted["data"].reshape(converted["basis"].shape),
        states["data"].reshape(states["basis"].shape),
    )
    return {"basis": states["basis"][0], "data": data}


def calculate_operator_inner_product(
    dual_vector: StateDualVector[_B0],
    operator: Operator[_B0, _B1],
    vector: StateVector[_B1],
) -> complex:
    """
    Calculate the energy of the given eigenvector.

    Parameters
    ----------
    hamiltonian : Hamiltonian[_B3d0Inv]
    eigenstate : Eigenstate[_B3d0Inv]

    Returns
    -------
    complex
        The energy of the Eigenvector given Hamiltonian
    """
    return np.linalg.multi_dot(  # type:ignore[no-any-return]
        [
            dual_vector["data"],
            operator["data"].reshape(operator["basis"].shape),
            vector["data"],
        ]
    )


def get_eigenstate_basis_from_hamiltonian(
    hamiltonian: SingleBasisOperator[_B0],
    subset_by_index: tuple[IntLike_co, IntLike_co] | None = None,
) -> ExplicitBasis[Any, Any]:
    """
    Given a hamiltonian, get the basis of the eigenstates given by subset_by_index.

    Parameters
    ----------
    hamiltonian : SingleBasisOperator[_B0Inv]
    subset_by_index : tuple[IntLike_co, IntLike_co] | None, optional
        subset_by_index, by default None

    Returns
    -------
    ExplicitBasis[Any, Any]
    """
    eigenvectors = calculate_eigenvectors_hermitian(hamiltonian, subset_by_index)
    return ExplicitBasis[Any, Any].from_state_vectors(eigenvectors)


_SB0 = TypeVar(
    "_SB0", bound=TupleBasisLike[*tuple[BasisWithLengthLike[Any, Any, Any], ...]]
)


def get_eigenstate_basis_stacked_from_hamiltonian(
    hamiltonian: SingleBasisOperator[_SB0],
    subset_by_index: tuple[IntLike_co, IntLike_co] | None = None,
) -> ExplicitStackedBasisWithLength[FundamentalBasis[int], _SB0]:
    """
    Given a hamiltonian, get the basis of the eigenstates given by subset_by_index.

    Parameters
    ----------
    hamiltonian : SingleBasisOperator[_SB0]
    subset_by_index : tuple[IntLike_co, IntLike_co] | None, optional
        subset_by_index, by default None

    Returns
    -------
    ExplicitStackedBasisWithLength[Any, Any, Any]
    """
    eigenvectors = calculate_eigenvectors_hermitian(hamiltonian, subset_by_index)
    return ExplicitStackedBasisWithLength(eigenvectors)
