from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import FundamentalBasis
from surface_potential_analysis.basis.basis_like import BasisLike
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasisLike,
    TupleBasis,
)
from surface_potential_analysis.operator.operator_list import (
    OperatorList,
    SingleBasisDiagonalOperatorList,
    as_operator_list,
)

if TYPE_CHECKING:
    from surface_potential_analysis.wavepacket.wavepacket import (
        BlochWavefunctionListBasis,
        BlochWavefunctionListList,
    )

_B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
_B1 = TypeVar("_B1", bound=BasisLike[Any, Any])
_B2 = TypeVar("_B2", bound=BasisLike[Any, Any])

LocalizationOperator = OperatorList[_B0, _B1, _B2]
"""
A list of operators, acting on each bloch k

List over the bloch k, each operator maps a series of
states at each band _B2 into the localised states made from
a mixture of each band _B1.

Note that the mixing between states of different bloch k
that is required to form the set of localised states is implicit.
The 'fundamental' loclised states are a sum of the contribution from
each bloch k, and all other states can be found by translating the
states by a unit cell.
"""

_SB0 = TypeVar("_SB0", bound=StackedBasisLike[Any, Any, Any])
_SB1 = TypeVar("_SB1", bound=StackedBasisLike[Any, Any, Any])


def get_localized_wavepackets(
    wavepackets: BlochWavefunctionListList[_B2, _SB1, _SB0],
    operator: LocalizationOperator[_SB1, _B1, _B2],
) -> BlochWavefunctionListList[_B1, _SB1, _SB0]:
    """
    Apply the LocalizationOperator to produce localized wavepackets.

    Parameters
    ----------
    wavepackets : WavepacketList[_B2, _SB1, _SB0]
        The unlocalized wavepackets
    operator : LocalizationOperator[_SB1, _B1, _B2]
        The operator used to localize the wavepackets

    Returns
    -------
    WavepacketList[_B1, _SB1, _SB0]
        The localized wavepackets
    """
    assert wavepackets["basis"][0][0] == operator["basis"][1][1]
    assert wavepackets["basis"][0][1] == operator["basis"][0]

    stacked_operator = operator["data"].reshape(
        operator["basis"][0].n, *operator["basis"][1].shape
    )
    vectors = wavepackets["data"].reshape(*wavepackets["basis"][0].shape, -1)

    # Sum over the bloch idx
    # data = np.einsum("kil,ljk->ijk", stacked_operator, vectors)
    data = np.einsum("jil,ljk->ijk", stacked_operator, vectors)

    return {
        "basis": TupleBasis(
            TupleBasis(operator["basis"][1][0], wavepackets["basis"][0][1]),
            wavepackets["basis"][1],
        ),
        "data": data.reshape(-1),
    }


def get_localized_hamiltonian_from_eigenvalues(
    hamiltonian: SingleBasisDiagonalOperatorList[_B2, _SB1],
    operator: LocalizationOperator[_SB1, _B1, _B2],
) -> OperatorList[_SB1, _B1, _B1]:
    """
    Localize the hamiltonian according to the Localization Operator.

    Parameters
    ----------
    hamiltonian : SingleBasisDiagonalOperatorList[_B2, _SB1]
    operator : LocalizationOperator[_SB1, _B1, _B2]

    Returns
    -------
    OperatorList[_SB1, _B1, _B1]
    """
    hamiltonian_stacked = hamiltonian["data"].reshape(
        hamiltonian["basis"][0].n, hamiltonian["basis"][1][0].n
    )
    operator_stacked = operator["data"].reshape(-1, *operator["basis"][1].shape)

    # Hamiltonian n, k; is diagonal
    # Operator k n'; n with k just along diagonal
    # Move k to end, and do tensordot on n and k. Since H is diagonal we dont need to sum
    converted_front = (
        np.moveaxis(operator_stacked, 0, -1)[:, :, :]
        * hamiltonian_stacked[np.newaxis, :, :]
    )

    # converted_front is n'; n, k
    # Operator k n'; n move axis to get n'; n k and conj to make dual basis
    # this time we sum over the n axis
    converted = np.sum(
        converted_front[:, np.newaxis, :, :]
        * np.conj(np.moveaxis(operator_stacked, 0, -1))[np.newaxis, :, :, :],
        axis=(2),
    )
    # we now have n'; n' k, so we need to move k to the front
    # this is why we return an operator list
    return {
        "basis": TupleBasis(
            hamiltonian["basis"][1][0],
            TupleBasis(operator["basis"][1][0], operator["basis"][1][0]),
        ),
        "data": np.moveaxis(converted, -1, 0).reshape(-1),
    }


def get_identity_operator(
    basis: BlochWavefunctionListBasis[_SB0, _SB1],
) -> LocalizationOperator[_SB1, FundamentalBasis[int], _SB0]:
    """
    Get the localization operator which is a simple identity.

    Parameters
    ----------
    basis : BlochWavefunctionListBasis[_SB0, _SB1]

    Returns
    -------
    LocalizationOperator[_SB1, FundamentalBasis[int], _SB0]
    """
    return as_operator_list(
        {
            "basis": TupleBasis(
                basis[1], TupleBasis(FundamentalBasis(basis[1].n), basis[0])
            ),
            "data": np.ones(basis.n, dtype=np.complex128),
        }
    )
