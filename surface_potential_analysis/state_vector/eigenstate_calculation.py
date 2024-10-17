from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import scipy.linalg  # type:ignore lib

from surface_potential_analysis.basis.legacy import (
    BasisLike,
    ExplicitBasis,
    ExplicitStackedBasisWithLength,
    FundamentalBasis,
    StackedBasisWithVolumeLike,
    TupleBasis,
)
from surface_potential_analysis.operator.conversion import convert_operator_to_basis
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_to_basis,
)
from surface_potential_analysis.util.decorators import timed

if TYPE_CHECKING:
    from surface_potential_analysis.operator.operator import (
        Operator,
        SingleBasisDiagonalOperator,
        SingleBasisOperator,
    )
    from surface_potential_analysis.state_vector.eigenstate_list import (
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

_B0 = TypeVar("_B0", bound=BasisLike)
_B1 = TypeVar("_B1", bound=BasisLike)
_B2 = TypeVar("_B2", bound=BasisLike)
_B3 = TypeVar("_B3", bound=BasisLike)


@timed
def calculate_eigenvectors_hermitian(
    operator: SingleBasisOperator[_B0],
    subset_by_index: tuple[IntLike_co, IntLike_co] | None = None,
) -> EigenstateList[FundamentalBasis[BasisMetadata], _B0]:
    """Get a list of eigenstates for a given operator, assuming it is hermitian."""
    eigenvalues, vectors = scipy.linalg.eigh(  # type:ignore lib
        operator["data"].reshape(operator["basis"].shape),
        subset_by_index=subset_by_index,
    )
    return {
        "basis": TupleBasis(
            FundamentalBasis(np.size(eigenvalues)),  # type:ignore lib
            operator["basis"][0],
        ),
        "data": np.transpose(vectors).reshape(-1),  # type:ignore lib
        "eigenvalue": np.array(eigenvalues),  # type:ignore lib
    }


def calculate_eigenvectors(
    operator: SingleBasisOperator[_B0],
) -> EigenstateList[FundamentalBasis[BasisMetadata], _B0]:
    """Get a list of eigenstates for a given operator, assuming it is hermitian."""
    eigenvalues, vectors = np.linalg.eig(
        operator["data"].reshape(operator["basis"].shape),
    )
    return {
        "basis": VariadicTupleBasis(
            (FundamentalBasis(eigenvalues.size), None), operator["basis"][0]
        ),
        "data": np.transpose(vectors).reshape(-1),
        "eigenvalue": eigenvalues,
    }


def operator_from_eigenstates(
    states: EigenstateList[_B1, _B0],
) -> SingleBasisOperator[_B0]:
    """Get an operator from eigenstates."""
    eigenvectors = states["data"].reshape(states["basis"].shape)
    data = np.einsum(  # type:ignore lib
        "ji,j,kj->ik",
        eigenvectors,
        states["eigenvalue"],
        np.linalg.inv(eigenvectors),
    )

    return {
        "basis": VariadicTupleBasis((states["basis"][1], states["basis"][1]), None),
        "data": data.ravel(),
    }


def calculate_expectation_diagonal(
    operator: SingleBasisDiagonalOperator[_B0], state: StateVector[_B2]
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
    converted = convert_state_vector_to_basis(state, operator["basis"][0])

    return np.einsum(  # type:ignore lib
        "j,j,j->",
        np.conj(converted["data"]),
        operator["data"],
        converted["data"],
    )


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
        operator, VariadicTupleBasis((state["basis"], state["basis"]), None)
    )

    return np.einsum(  # type:ignore lib
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
        operator, VariadicTupleBasis((states["basis"][1], states["basis"][1]), None)
    )
    data = np.einsum(  # type:ignore lib
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


_SB0 = TypeVar("_SB0", bound=StackedBasisWithVolumeLike)


def get_eigenstate_basis_stacked_from_hamiltonian(
    hamiltonian: SingleBasisOperator[_SB0],
    subset_by_index: tuple[IntLike_co, IntLike_co] | None = None,
) -> ExplicitStackedBasisWithLength[FundamentalBasis[BasisMetadata], _SB0]:
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
