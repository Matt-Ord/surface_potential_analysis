from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis_like import (
    BasisLike,
    BasisWithLengthLike,
    convert_vector,
)
from surface_potential_analysis.basis.stacked_basis import TupleBasis
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    stacked_basis_as_fundamental_position_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        FundamentalPositionBasis,
        FundamentalTransformedPositionBasis,
    )
    from surface_potential_analysis.basis.stacked_basis import TupleBasisLike
    from surface_potential_analysis.overlap.overlap import Overlap

    _B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
    _B1 = TypeVar("_B1", bound=BasisLike[Any, Any])
    _B2 = TypeVar("_B2", bound=BasisLike[Any, Any])
    _B3 = TypeVar("_B3", bound=BasisLike[Any, Any])
    _BL0 = TypeVar("_BL0", bound=BasisWithLengthLike[Any, Any, Literal[3]])


def convert_overlap_to_basis(
    overlap: Overlap[_B0, _B1, _B2], basis: _B3
) -> Overlap[_B3, _B1, _B2]:
    """
    Convert the overlap to the given basis.

    Note this just converts the list basis (ie not the wavepacket index basis).

    Parameters
    ----------
    overlap : Overlap[_B0, _B1, _B2]
    basis : _B3

    Returns
    -------
    Overlap[_B3, _B1, _B2]
    """
    converted = convert_vector(
        overlap["data"].reshape(overlap["basis"].shape),
        overlap["basis"][0],
        basis,
        axis=0,
    )
    return {
        "basis": TupleBasis(basis, overlap["basis"][1]),
        "data": converted.reshape(-1),
    }


def convert_overlap_to_momentum_basis(
    overlap: Overlap[
        TupleBasisLike[*tuple[_BL0, ...]],
        _B1,
        _B2,
    ],
) -> Overlap[
    TupleBasisLike[*tuple[FundamentalTransformedPositionBasis[Any, Literal[3]], ...]],
    _B1,
    _B2,
]:
    """
    Convert an overlap from position basis to momentum.

    Parameters
    ----------
    overlap : Overlap[PositionTupleBasisLike[tuple[_L0Inv, _L1Inv, _L2Inv]]

    Returns
    -------
    OverlapMomentum[_L0Inv, _L1Inv, _L2Inv]
    """
    transformed = np.fft.ifftn(
        overlap["data"].reshape(overlap["basis"].shape),
        axes=(0, 1, 2),
        s=overlap["basis"].fundamental_shape,
        norm="forward",
    )
    transformed.reshape(-1)

    return convert_overlap_to_basis(
        overlap, stacked_basis_as_fundamental_momentum_basis(overlap["basis"][0])
    )


def convert_overlap_to_position_basis(
    overlap: Overlap[
        TupleBasisLike[*tuple[_BL0, ...]],
        _B1,
        _B2,
    ],
) -> Overlap[
    TupleBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]],
    _B1,
    _B2,
]:
    """
    Convert an overlap from momentum basis to position.

    Parameters
    ----------
    overlap : OverlapMomentum[_L0Inv, _L1Inv, _L2Inv]

    Returns
    -------
    Overlap[PositionTupleBasisLike[tuple[_L0Inv, _L1Inv, _L2Inv]]
    """
    return convert_overlap_to_basis(
        overlap, stacked_basis_as_fundamental_position_basis(overlap["basis"][0])
    )
