from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, TypeVar

import numpy as np
from scipy.constants import hbar  # type:ignore lib

from surface_potential_analysis.basis.basis import FundamentalBasis
from surface_potential_analysis.basis.stacked_basis import TupleBasis
from surface_potential_analysis.kernel.build import (
    truncate_diagonal_noise_operator_list,
)
from surface_potential_analysis.kernel.solve._eigenvalue import (
    get_periodic_noise_operators_diagonal_eigenvalue,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.kernel.kernel import (
        DiagonalNoiseKernel,
        DiagonalNoiseOperatorList,
        NoiseOperatorList,
    )
    from surface_potential_analysis.operator.operator_list import (
        DiagonalOperatorList,
        OperatorList,
    )

    _B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
    _B1 = TypeVar("_B1", bound=BasisLike[Any, Any])
    _B2 = TypeVar("_B2", bound=BasisLike[Any, Any])
    _B3 = TypeVar("_B3", bound=BasisLike[Any, Any])


def sample_noise_from_diagonal_operators_split(
    operators: DiagonalNoiseOperatorList[_B0, _B1, _B2], *, n_samples: int
) -> DiagonalOperatorList[TupleBasis[FundamentalBasis[int], _B0], _B1, _B2]:
    """Generate noise from a set of diagonal noise operators.

    Parameters
    ----------
    operators : DiagonalNoiseOperatorList[_B0, _B1, _B2]
    n_samples : int

    Returns
    -------
    DiagonalOperatorList[FundamentalBasis[int], _B1, _B2]
    """
    n_operators = operators["basis"][0].n

    rng = np.random.default_rng()
    factors = (
        rng.standard_normal((n_samples, n_operators))
        + 1j * rng.standard_normal((n_samples, n_operators))
    ) / np.sqrt(2)

    data = np.einsum(  # type: ignore lib
        "ij,j,jk->ijk",
        factors,
        np.lib.scimath.sqrt(operators["eigenvalue"] * hbar),
        operators["data"].reshape(n_operators, -1),
    )
    return {
        "basis": TupleBasis(
            TupleBasis(FundamentalBasis(n_samples), operators["basis"][0]),
            operators["basis"][1],
        ),
        "data": data.ravel(),
    }


def diagonal_operator_list_from_diagonal_split(
    split: DiagonalOperatorList[TupleBasis[_B3, _B0], _B1, _B2],
) -> DiagonalOperatorList[_B3, _B1, _B2]:
    """
    Sum over the 'spit' axis.

    Parameters
    ----------
    split : DiagonalOperatorList[TupleBasis[_B3, _B0], _B1, _B2]

    Returns
    -------
    DiagonalOperatorList[_B3, _B1, _B2]
    """
    data = np.sum(split["data"].reshape(*split["basis"][0].shape, -1), axis=1)
    return {
        "basis": TupleBasis(split["basis"][0][0], split["basis"][1]),
        "data": data.ravel(),
    }


def get_diagonal_split_noise_components(
    split: DiagonalOperatorList[TupleBasis[_B3, _B0], _B1, _B2],
) -> list[DiagonalOperatorList[_B3, _B1, _B2]]:
    """Get the components of the noise.

    Parameters
    ----------
    split : DiagonalOperatorList[TupleBasis[_B3, _B0], _B1, _B2]

    Returns
    -------
    list[DiagonalOperatorList[_B3, _B1, _B2]]
    """
    data = split["data"].reshape(*split["basis"][0].shape, -1).swapaxes(0, 1)
    return [
        {
            "basis": TupleBasis(split["basis"][0][0], split["basis"][1]),
            "data": d.ravel(),
        }
        for d in data
    ]


def sample_noise_from_diagonal_operators(
    operators: DiagonalNoiseOperatorList[_B0, _B1, _B2], *, n_samples: int
) -> DiagonalOperatorList[FundamentalBasis[int], _B1, _B2]:
    """Generate noise from a set of diagonal noise operators.

    Parameters
    ----------
    operators : DiagonalNoiseOperatorList[_B0, _B1, _B2]
    n_samples : int

    Returns
    -------
    DiagonalOperatorList[FundamentalBasis[int], _B1, _B2]
    """
    split = sample_noise_from_diagonal_operators_split(operators, n_samples=n_samples)
    return diagonal_operator_list_from_diagonal_split(split)


def sample_noise_from_operators(
    operators: NoiseOperatorList[_B0, _B1, _B2], *, n_samples: int
) -> OperatorList[FundamentalBasis[int], _B1, _B2]:
    """Generate noise from a set of noise operators.

    Parameters
    ----------
    operators : NoiseOperatorList[_B0, _B1, _B2]
    n_samples : int

    Returns
    -------
    OperatorList[FundamentalBasis[int], _B1, _B2]
    """
    n_operators = operators["basis"][0].n

    rng = np.random.default_rng()
    factors = (
        rng.standard_normal((n_samples, n_operators))
        + 1j * rng.standard_normal((n_samples, n_operators))
    ) / np.sqrt(2)

    data = np.einsum(  # type: ignore lib
        "ij,j,jk->ik",
        factors,
        np.lib.scimath.sqrt(operators["eigenvalue"] * hbar),
        operators["data"].reshape(n_operators, -1),
    )
    return {
        "basis": TupleBasis(FundamentalBasis(n_samples), operators["basis"][1]),
        "data": data.ravel(),
    }


def sample_noise_from_diagonal_kernel(
    kernel: DiagonalNoiseKernel[_B0, _B1, _B0, _B1],
    *,
    n_samples: int,
    truncation: Iterable[int] | None,
) -> OperatorList[FundamentalBasis[int], _B0, _B1]:
    """Generate noise for a diagonal kernel.

    Parameters
    ----------
    kernel : DiagonalNoiseKernel[_B0, _B1, _B0, _B1]
    n_samples : int
    truncation : Iterable[int] | None

    Returns
    -------
    OperatorList[FundamentalBasis[int], _B0, _B1]
        _description_
    """
    operators = get_periodic_noise_operators_diagonal_eigenvalue(kernel)
    truncation = range(operators["basis"][0].n) if truncation is None else truncation
    truncated = truncate_diagonal_noise_operator_list(operators, truncation)
    return sample_noise_from_diagonal_operators(truncated, n_samples=n_samples)
