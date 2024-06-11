from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import (
    FundamentalTransformedPositionBasis,
    TransformedPositionBasis,
)
from surface_potential_analysis.basis.basis_like import (
    BasisWithLengthLike,
    convert_matrix,
)
from surface_potential_analysis.basis.stacked_basis import TupleBasis
from surface_potential_analysis.operator.operator import DiagonalOperator, as_operator
from surface_potential_analysis.operator.operator_list import as_operator_list
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.basis.stacked_basis import TupleBasisLike
    from surface_potential_analysis.operator.operator import (
        Operator,
        SingleBasisOperator,
    )
    from surface_potential_analysis.operator.operator_list import (
        DiagonalOperatorList,
        OperatorList,
        SingleBasisOperatorList,
    )

    _B0Inv = TypeVar("_B0Inv", bound=BasisLike[Any, Any])
    _B1Inv = TypeVar("_B1Inv", bound=BasisLike[Any, Any])
    _B2Inv = TypeVar("_B2Inv", bound=BasisLike[Any, Any])
    _B3Inv = TypeVar("_B3Inv", bound=BasisLike[Any, Any])
    _B4 = TypeVar("_B4", bound=BasisLike[Any, Any])


def convert_operator_to_basis(
    operator: Operator[_B0Inv, _B1Inv], basis: TupleBasisLike[_B2Inv, _B3Inv]
) -> Operator[_B2Inv, _B3Inv]:
    """
    Given an operator, convert it to the given basis.

    Parameters
    ----------
    eigenstate : Eigenstate[_B3d0Inv]
    basis : _B3d1Inv

    Returns
    -------
    Eigenstate[_B3d1Inv]
    """
    converted = convert_matrix(
        operator["data"].reshape(operator["basis"].shape),
        operator["basis"][0],
        basis[0],
        operator["basis"][1],
        basis[1],
    )
    return {"basis": basis, "data": converted.reshape(-1)}


def convert_diagonal_operator_to_basis(
    operator: DiagonalOperator[_B0Inv, _B1Inv],
    basis: TupleBasisLike[_B2Inv, _B3Inv],
) -> Operator[_B2Inv, _B3Inv]:
    """Given an operator, convert it to the given basis.

    Parameters
    ----------
    operator : OperatorList[_B4, _B0Inv, _B1Inv]
    basis : TupleBasisLike[_B2Inv, _B3Inv]

    Returns
    -------
    OperatorList[_B4, _B2Inv, _B3Inv]
    """
    full = as_operator(operator)
    return convert_operator_to_basis(full, basis)


def convert_operator_list_to_basis(
    operator: OperatorList[_B4, _B0Inv, _B1Inv], basis: TupleBasisLike[_B2Inv, _B3Inv]
) -> OperatorList[_B4, _B2Inv, _B3Inv]:
    """Given an operator, convert it to the given basis.

    Parameters
    ----------
    operator : OperatorList[_B4, _B0Inv, _B1Inv]
    basis : TupleBasisLike[_B2Inv, _B3Inv]

    Returns
    -------
    OperatorList[_B4, _B2Inv, _B3Inv]
    """
    converted = convert_matrix(
        operator["data"].reshape(operator["basis"][0].n, *operator["basis"][1].shape),
        operator["basis"][1][0],
        basis[0],
        operator["basis"][1][1],
        basis[1],
        axes=(1, 2),
    )
    return {
        "basis": TupleBasis(operator["basis"][0], basis),
        "data": converted.reshape(-1),
    }


def convert_diagonal_operator_list_to_basis(
    operator: DiagonalOperatorList[_B4, _B0Inv, _B1Inv],
    basis: TupleBasisLike[_B2Inv, _B3Inv],
) -> OperatorList[_B4, _B2Inv, _B3Inv]:
    """Given an operator, convert it to the given basis.

    Parameters
    ----------
    operator : OperatorList[_B4, _B0Inv, _B1Inv]
    basis : TupleBasisLike[_B2Inv, _B3Inv]

    Returns
    -------
    OperatorList[_B4, _B2Inv, _B3Inv]
    """
    full = as_operator_list(operator)
    return convert_operator_list_to_basis(full, basis)


def sample_operator(
    operator: Operator[
        TupleBasisLike[*tuple[BasisWithLengthLike[Any, Any, Any], ...]],
        TupleBasisLike[*tuple[BasisWithLengthLike[Any, Any, Any], ...]],
    ],
    *,
    sample: tuple[int, ...],
) -> SingleBasisOperator[
    TupleBasisLike[*tuple[FundamentalTransformedPositionBasis[Any, Any], ...]],
]:
    """
    Sample an operator, taking only the sample lowest states in k.

    Returns
    -------
    _type_
        _description_
    """
    basis_k = stacked_basis_as_fundamental_momentum_basis(operator["basis"][0])
    basis_k_sampled = TupleBasis(
        *tuple(
            TransformedPositionBasis(
                basis_k[i].delta_x,
                s,
                basis_k[i].fundamental_n,
            )
            for (i, s) in enumerate(sample)
        ),
    )

    operators_k = convert_operator_to_basis(
        convert_operator_to_basis(
            operator,
            TupleBasis(basis_k, basis_k),
        ),
        TupleBasis(TupleBasis(basis_k_sampled), TupleBasis(basis_k_sampled)),
    )

    basis_k_full = TupleBasis(
        *tuple(
            FundamentalTransformedPositionBasis(
                basis_k[i].delta_x,
                s,
            )
            for (i, s) in enumerate(sample)
        ),
    )

    data = operators_k["data"] * np.prod(
        np.sqrt(
            [basis_k[i].fundamental_n / s for (i, s) in enumerate(sample)],
        ),
    )

    return {
        "basis": TupleBasis(basis_k_full, basis_k_full),
        "data": data,
    }


def sample_operator_list(
    operators: OperatorList[
        _B4,
        TupleBasisLike[*tuple[BasisWithLengthLike[Any, Any, Any], ...]],
        TupleBasisLike[*tuple[BasisWithLengthLike[Any, Any, Any], ...]],
    ],
    *,
    sample: tuple[int, ...],
) -> SingleBasisOperatorList[
    _B4,
    TupleBasisLike[*tuple[FundamentalTransformedPositionBasis[Any, Any], ...]],
]:
    """
    Sample an operator list, staking only the sample lowest omentum states.

    Returns
    -------
    SingleBasisOperatorList[
    _B4,
    TupleBasisLike[*tuple[FundamentalTransformedPositionBasis[Any, Any], ...]],
    ]

    """
    basis_k = stacked_basis_as_fundamental_momentum_basis(operators["basis"][1][0])
    basis_k_sampled = TupleBasis(
        *tuple(
            TransformedPositionBasis(
                basis_k[i].delta_x,
                s,
                basis_k[i].fundamental_n,
            )
            for (i, s) in enumerate(sample)
        ),
    )

    operators_k = convert_operator_list_to_basis(
        convert_operator_list_to_basis(
            operators,
            TupleBasis(basis_k, basis_k),
        ),
        TupleBasis(TupleBasis(basis_k_sampled), TupleBasis(basis_k_sampled)),
    )

    basis_k_full = TupleBasis(
        *tuple(
            FundamentalTransformedPositionBasis(
                basis_k[i].delta_x,
                s,
            )
            for (i, s) in enumerate(sample)
        ),
    )

    data = operators_k["data"] * np.prod(
        np.sqrt(
            [basis_k[i].fundamental_n / s for (i, s) in enumerate(sample)],
        ),
    )

    return {
        "basis": TupleBasis(
            operators["basis"][0],
            TupleBasis(basis_k_full, basis_k_full),
        ),
        "data": data,
    }
