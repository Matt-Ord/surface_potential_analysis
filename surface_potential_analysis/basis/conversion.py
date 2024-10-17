from __future__ import annotations

from typing import TYPE_CHECKING

from slate.basis._basis import Basis
from slate.basis.transformed import TransformedBasis
from slate.metadata._metadata import BasisMetadata

from surface_potential_analysis.basis.legacy import (
    FundamentalBasis,
)

if TYPE_CHECKING:
    import numpy as np


def basis_as_fundamental_basis[M: BasisMetadata, DT: np.generic](
    basis: Basis[M, DT],
) -> FundamentalBasis[M]:
    """
    Get the fundamental position axis for a given axis.

    Parameters
    ----------
    axis : BasisLike[_NF0Inv, _N0Inv]

    Returns
    -------
    FundamentalPositionBasis[_NF0Inv]
    """
    return FundamentalBasis(basis.metadata)


def basis_as_transformed_basis[M: BasisMetadata, DT: np.generic](
    basis: Basis[M, DT],
) -> TransformedBasis[M]:
    """
    Get the fundamental momentum axis for a given axis.

    Parameters
    ----------
    axis : BasisLike[_NF0Inv, _N0Inv, _NDInv]

    Returns
    -------
    FundamentalMomentumBasis[_NF0Inv, _NDInv]
    """
    return TransformedBasis(FundamentalBasis(basis.metadata))
