from __future__ import annotations

from typing import (
    Any,
    Generic,
    TypedDict,
    TypeVar,
    TypeVarTuple,
)

import numpy as np

from surface_potential_analysis.basis.basis_like import BasisLike
from surface_potential_analysis.basis.stacked_basis import (
    TupleBasis,
    TupleBasisLike,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.operator.operator import (
    DiagonalOperator,
    Operator,
)
from surface_potential_analysis.operator.operator_list import (
    DiagonalOperatorList,
    OperatorList,
    as_diagonal_operator_list,
    as_operator_list,
)

_B0_co = TypeVar("_B0_co", bound=BasisLike[Any, Any], covariant=True)
_B1_co = TypeVar("_B1_co", bound=BasisLike[Any, Any], covariant=True)
_B2_co = TypeVar("_B2_co", bound=BasisLike[Any, Any], covariant=True)
_B3_co = TypeVar("_B3_co", bound=BasisLike[Any, Any], covariant=True)

_B0 = TypeVar("_B0", bound=BasisLike[int, int])
_B1 = TypeVar("_B1", bound=BasisLike[Any, Any])
_B2 = TypeVar("_B2", bound=BasisLike[Any, Any])


class NoiseKernel(TypedDict, Generic[_B0_co, _B1_co, _B2_co, _B3_co]):
    r"""
    Represents a generic noise kernel in the given basis.

    Note this kernel has an implicit 'noise operator' in this basis

    ```latex
    Z_i,j = \ket{i}\bra{j}
    ```

    which we can diagonalize to get a DiagonalNoiseKernel. This noise kernel
    represents sources of noise which have the same frequency dependance (for example
    a set of noise operators which are all markovian)
    """

    basis: TupleBasisLike[
        TupleBasisLike[_B0_co, _B1_co], TupleBasisLike[_B2_co, _B3_co]
    ]
    """The basis of the underlying noise operator"""
    data: np.ndarray[tuple[int], np.dtype[np.complex128]]


SingleBasisNoiseKernel = NoiseKernel[_B0, _B0, _B0, _B0]


class DiagonalNoiseKernel(TypedDict, Generic[_B0_co, _B1_co, _B2_co, _B3_co]):
    r"""
    Represents a noise kernel, written in diagonal form.

    Note we assume that all sources of noise in this kernel have the same time/frequency dependance

    This is useful for noise such as that caused by a local (coulomb) interaction.

    Note this kernel has an implicit 'noise operator' in this basis

    ```latex
    Z_i,j = \ket{i}\bra{j} \delta{i,j}
    ```

    Since the noise operator is zero i!=j, we only care about the diagonal elements of the kernel

    """

    basis: TupleBasisLike[
        TupleBasisLike[_B0_co, _B1_co], TupleBasisLike[_B2_co, _B3_co]
    ]
    """The basis of the underlying noise operator"""
    data: np.ndarray[tuple[int], np.dtype[np.complex128]]


SingleBasisDiagonalNoiseKernel = DiagonalNoiseKernel[_B0, _B0, _B0, _B0]


class IsotropicNoiseKernel(TypedDict, Generic[_B0_co]):
    r"""
    Represents a noise kernel which is isotropic.

    In this case, the correllation between any pair of states depends only on
    the difference between the two states. We therefore store the kernel
    relating to only a single state.
    """

    basis: _B0_co
    """The basis of the underlying noise operator"""
    data: np.ndarray[tuple[int], np.dtype[np.complex128]]


class EigenOperator(Operator[_B0_co, _B1_co], TypedDict):
    """A State vector which is the eigenvector of some operator."""

    eigenvalue: complex | np.complex128


class EigenOperatorList(
    OperatorList[_B0_co, _B1_co, _B2_co],
    TypedDict,
):
    """Represents a collection of eigen-operators, each with the same basis."""

    eigenvalue: np.ndarray[tuple[int], np.dtype[np.complex128]]


class DiagonalEigenOperatorList(
    DiagonalOperatorList[_B0_co, _B1_co, _B2_co],
    TypedDict,
):
    """Represents a collection of eigen-operators, each with the same basis."""

    eigenvalue: np.ndarray[tuple[int], np.dtype[np.complex128]]


# A noise operator represents the noise in a system.
# Each index in the noise operator is multiplied by a separate
# random operator


NoiseOperator = Operator
SingleBasisNoiseOperator = NoiseOperator[_B0, _B0]

DiagonalNoiseOperator = DiagonalOperator
SingleBasisDiagonalNoiseOperator = DiagonalNoiseOperator[_B0, _B0]

NoiseOperatorList = EigenOperatorList[_B0, _B1, _B2]
SingleBasisNoiseOperatorList = EigenOperatorList[_B0, _B1, _B1]

DiagonalNoiseOperatorList = DiagonalEigenOperatorList[_B0, _B1, _B2]
SingleBasisDiagonalNoiseOperatorList = DiagonalEigenOperatorList[_B0, _B1, _B1]
DiagonalNoiseOperator = DiagonalOperator


def as_full_kernel_from_diagonal(
    diagonal: DiagonalNoiseKernel[_B0, _B1, _B0, _B1],
) -> NoiseKernel[_B0, _B1, _B0, _B1]:
    """
    Given a diagonal noise kernel, get the full noise kernel.

    Parameters
    ----------
    diagonal : DiagonalNoiseKernel[_B0, _B1, _B0, _B1]

    Returns
    -------
    NoiseKernel[_B0, _B1, _B0, _B1]
    """
    n = diagonal["basis"][0].shape[0]
    m = diagonal["basis"][1].shape[0]

    full_data = np.diag(diagonal["data"]).reshape(n, m, n, m).swapaxes(1, 2)

    return {"basis": diagonal["basis"], "data": full_data.ravel()}


def as_diagonal_kernel_from_full(
    kernel: NoiseKernel[_B0, _B1, _B0, _B1],
) -> DiagonalNoiseKernel[_B0, _B1, _B0, _B1]:
    """
    Given a diagonal noise kernel, get the full noise kernel.

    Parameters
    ----------
    diagonal : DiagonalNoiseKernel[_B0, _B1, _B0, _B1]

    Returns
    -------
    NoiseKernel[_B0, _B1, _B0, _B1]
    """
    n = kernel["basis"][0].shape[0]
    m = kernel["basis"][1].shape[0]
    diagonal = np.diag(
        kernel["data"].reshape(n, n, m, m).swapaxes(1, 2).reshape(n * m, n * m)
    )

    return {"basis": kernel["basis"], "data": diagonal.ravel()}


def as_diagonal_kernel_from_isotropic(
    kernel: IsotropicNoiseKernel[_B0],
) -> DiagonalNoiseKernel[_B0, _B0, _B0, _B0]:
    """
    Convert an isotropic kernel into a diagonal kernel.

    By convention, we take the kernel corresponding to state 0.
    We fill the diagonal elements by finding the corresponding value for each given displacement

    Parameters
    ----------
    kernel : DiagonalNoiseKernel[_B0, _B0, _B0, _B0]

    Returns
    -------
    IsotropicNoiseKernel[_B0]
    """
    old_data = kernel["data"].ravel()

    indices = BasisUtil(kernel["basis"]).nx_points
    displacement_matrix = np.mod(indices[:, None] - indices[None, :], old_data.size)
    data = old_data[displacement_matrix]

    return {
        "basis": TupleBasis(
            TupleBasis(kernel["basis"], kernel["basis"]),
            TupleBasis(kernel["basis"], kernel["basis"]),
        ),
        "data": data.ravel(),
    }


def as_isotropic_kernel_from_diagonal(
    kernel: DiagonalNoiseKernel[_B0, _B0, _B0, _B0], *, assert_isotropic: bool = True
) -> IsotropicNoiseKernel[_B0]:
    """
    Convert a diagonal kernel into an isotropic kernel.

    By convention, we take the kernel corresponding to state 0

    Parameters
    ----------
    kernel : DiagonalNoiseKernel[_B0, _B0, _B0, _B0]

    Returns
    -------
    IsotropicNoiseKernel[_B0]
    """
    data = kernel["data"].reshape(kernel["basis"][0].shape)[0]
    out: IsotropicNoiseKernel[_B0] = {"basis": kernel["basis"][0][0], "data": data}

    if assert_isotropic:
        np.testing.assert_almost_equal(
            as_diagonal_kernel_from_isotropic(out)["data"], kernel["data"]
        )

    return out


_B0s = TypeVarTuple("_B0s")


def as_diagonal_kernel_from_isotropic_stacked(
    kernel: IsotropicNoiseKernel[TupleBasisLike[*_B0s]],
) -> SingleBasisDiagonalNoiseKernel[TupleBasisLike[*_B0s]]:
    """
    Convert an isotropic kernel into a diagonal kernel.

    By convention, we take the kernel corresponding to state 0.
    We fill the diagonal elements by finding the corresponding value for each given displacement.
    For a Stacked Basis, this displacement should be calculated axes-wise

    Parameters
    ----------
    kernel : DiagonalNoiseKernel[_B0, _B0, _B0, _B0]

    Returns
    -------
    IsotropicNoiseKernel[_B0]
    """
    old_data = kernel["data"].reshape(*kernel["basis"].shape)

    util = BasisUtil(kernel["basis"])
    # Calculate the displacement on each axis seperately
    displacement_matrix = tuple(
        np.mod(indices[:, None] - indices[None, :], n)
        for (indices, n) in zip(
            util.stacked_nx_points,
            util.shape,
            strict=True,
        )
    )

    data = old_data[displacement_matrix]

    return {
        "basis": TupleBasis(
            TupleBasis(kernel["basis"], kernel["basis"]),
            TupleBasis(kernel["basis"], kernel["basis"]),
        ),
        "data": data.ravel(),
    }


def as_diagonal_noise_operators_from_full(
    operators: NoiseOperatorList[_B0, _B1, _B2],
) -> DiagonalNoiseOperatorList[_B0, _B1, _B2]:
    """
    Convert noise operators to diagonal noise operators.

    Parameters
    ----------
    operators : NoiseOperatorList[_B0, _B1, _B2]

    Returns
    -------
    DiagonalNoiseOperatorList[_B0, _B1, _B2]
    """
    operators_diagonal = as_diagonal_operator_list(operators)
    return {
        "basis": operators["basis"],
        "data": operators_diagonal["data"],
        "eigenvalue": operators["eigenvalue"],
    }


def as_noise_operators_from_diagonal(
    operators: DiagonalNoiseOperatorList[_B0, _B1, _B2],
) -> NoiseOperatorList[_B0, _B1, _B2]:
    """
    Convert diagonal operators into full operators.

    Parameters
    ----------
    operators : DiagonalNoiseOperatorList[_B0, _B1, _B2]

    Returns
    -------
    NoiseOperatorList[_B0, _B1, _B2]
    """
    operators_full = as_operator_list(operators)
    return {
        "basis": operators["basis"],
        "data": operators_full["data"],
        "eigenvalue": operators["eigenvalue"],
    }


def get_full_kernel_from_operators(
    operators: NoiseOperatorList[_B2, _B0, _B1],
) -> NoiseKernel[_B0, _B1, _B0, _B1]:
    """
    Build a full kernel from operators.

    Parameters
    ----------
    operators : NoiseOperatorList[FundamentalBasis[int], _B0, _B1]

    Returns
    -------
    NoiseKernel[_B0, _B1, _B0, _B1]
    """
    operators_data = operators["data"].reshape(
        operators["basis"][0].n, *operators["basis"][1].shape
    )

    data = np.einsum(  # type:ignore  unknown
        "a,aji,akl->ij kl",
        operators["eigenvalue"],
        np.conj(operators_data),
        operators_data,
    )
    return {
        "basis": TupleBasis(operators["basis"][1], operators["basis"][1]),
        "data": data.reshape(-1),
    }


def get_diagonal_kernel_from_diagonal_operators(
    operators: DiagonalNoiseOperatorList[_B2, _B0, _B1],
) -> DiagonalNoiseKernel[_B0, _B1, _B0, _B1]:
    """
    Build a diagonal kernel from operators.

    Parameters
    ----------
    operators : DiagonalNoiseOperatorList[BasisLike[Any, Any], _B0, _B1]

    Returns
    -------
    DiagonalNoiseKernel[_B0, _B1, _B0, _B1]
    """
    operators_data = operators["data"].reshape(operators["basis"][0].n, -1)
    data = np.einsum(  # type:ignore  unknown
        "a,ai,aj->ij",
        operators["eigenvalue"],
        np.conj(operators_data),
        operators_data,
    )
    return {
        "basis": TupleBasis(operators["basis"][1], operators["basis"][1]),
        "data": data.reshape(-1),
    }


def get_diagonal_kernel_from_operators(
    operators: NoiseOperatorList[_B2, _B0, _B1],
) -> DiagonalNoiseKernel[_B0, _B1, _B0, _B1]:
    """
    Build a diagonal kernel from operators.

    Parameters
    ----------
    operators : NoiseOperatorList[BasisLike[Any, Any], _B0, _B1]

    Returns
    -------
    DiagonalNoiseKernel[_B0, _B1, _B0, _B1]
    """
    return get_diagonal_kernel_from_diagonal_operators(
        as_diagonal_noise_operators_from_full(operators),
    )


def get_isotropic_kernel_from_diagonal_operators(
    operators: SingleBasisDiagonalNoiseOperatorList[
        _B0,
        _B1,
    ],
    *,
    assert_isotropic: bool = True,
) -> IsotropicNoiseKernel[_B1]:
    """
    Build a isotropic kernel from operators.

    Parameters
    ----------
    operators: SingleBasisDiagonalNoiseOperatorList[
        _B0,
        _B1,
    ]

    Returns
    -------
    IsotropicNoiseKernel[_B1]
    """
    return as_isotropic_kernel_from_diagonal(
        get_diagonal_kernel_from_operators(operators),
        assert_isotropic=assert_isotropic,
    )


def get_isotropic_kernel_from_operators(
    operators: SingleBasisNoiseOperatorList[
        _B0,
        _B1,
    ],
    *,
    assert_isotropic: bool = True,
) -> IsotropicNoiseKernel[_B1]:
    """
    Build a isotropic kernel from operators.

    Parameters
    ----------
    operators : SingleBasisNoiseOperatorList[
        _B0,
        _B1,
    ]

    Returns
    -------
    IsotropicNoiseKernel[_B1]
    """
    return get_isotropic_kernel_from_diagonal_operators(
        as_diagonal_noise_operators_from_full(operators),
        assert_isotropic=assert_isotropic,
    )
