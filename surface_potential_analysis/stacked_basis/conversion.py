from __future__ import annotations

import numpy as np
from slate.basis.stacked._tuple_basis import TupleBasis
from slate.basis.transformed import TransformedBasis
from slate.metadata._metadata import BasisMetadata

from surface_potential_analysis.basis.legacy import (
    FundamentalBasis,
)


def tuple_basis_as_transformed_fundamental[M: BasisMetadata, E, DT: np.generic](
    basis: TupleBasis[M, E, DT],
) -> TupleBasis[M, E, np.complex128]:
    """
    Get the fundamental momentum basis for a given basis.

    Parameters
    ----------
    basis : _ALB0Inv

    Returns
    -------
    tuple[FundamentalMomentumBasis[Any, Any], ...]
    """
    return TupleBasis(
        tuple(TransformedBasis(FundamentalBasis(ax.metadata)) for ax in basis.children),
        basis.metadata.extra,
    )


def tuple_basis_as_fundamental[M: BasisMetadata, E, DT: np.generic](
    basis: TupleBasis[M, E, DT],
) -> TupleBasis[M, E, DT]:
    """
    Get the fundamental position basis for a given basis.

    Parameters
    ----------
    self : BasisConfigUtil[tuple[BasisLike[_LF0Inv, _L0Inv], BasisLike[_LF1Inv, _L1Inv], BasisLike[_LF2Inv, _L2Inv]]]

    Returns
    -------
    StackedBasisWithVolumeLike[tuple[FundamentalPositionBasis[_LF0Inv], FundamentalPositionBasis[_LF1Inv], FundamentalPositionBasis[_LF2Inv]]
    """
    return TupleBasis(
        tuple(FundamentalBasis(ax.metadata) for ax in basis.children),
        basis.metadata.extra,
    )


def stacked_basis_as_fundamental_with_shape[M: BasisMetadata, E, DT: np.generic](
    basis: TupleBasis[M, E, DT], shape: tuple[int, ...]
) -> TupleBasis[BasisMetadata, E, DT]:
    """
    Get the fundamental position basis for a given basis.

    Parameters
    ----------
    self : BasisConfigUtil[tuple[BasisLike[_LF0Inv, _L0Inv], BasisLike[_LF1Inv, _L1Inv], BasisLike[_LF2Inv, _L2Inv]]]

    Returns
    -------
    StackedBasisWithVolumeLike[tuple[FundamentalPositionBasis[_LF0Inv], FundamentalPositionBasis[_LF1Inv], FundamentalPositionBasis[_LF2Inv]]
    """
    return TupleBasis(
        tuple(FundamentalBasis.from_shape((s,)) for s in shape),
        basis.metadata.extra,
    )
