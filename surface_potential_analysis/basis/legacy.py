from __future__ import annotations

from typing import Any, TypeVarTuple

import numpy as np
from slate.basis import Basis, FundamentalBasis, TransformedBasis, VariadicTupleBasis
from slate.basis import EvenlySpacedBasis as EvenlySpacedBasisNew
from slate.basis import TruncatedBasis as TruncatedBasisNew
from slate.basis import TupleBasis as TupleBasisNew
from slate.explicit_basis._explicit_basis import ExplicitUnitaryBasis
from slate.metadata import BasisMetadata
from slate.metadata.length import AxisDirections, LengthMetadata
from slate.metadata.stacked import StackedMetadata

from surface_potential_analysis.basis.block_fraction_basis import (
    BlochFractionMetadata,
    ExplicitBlochFractionMetadata,
)

TS = TypeVarTuple("TS")

type TupleBasis[*TS] = VariadicTupleBasis[*TS, None, np.complex128]
type TupleBasisLike[*TS] = VariadicTupleBasis[*TS, None, np.complex128]
type TupleBasisWithLengthLike[*TS] = VariadicTupleBasis[
    *TS, AxisDirections, np.complex128
]
type StackedBasisLike = TupleBasisNew[BasisMetadata, Any, np.complex128]
type StackedBasisWithVolumeLike = TupleBasisNew[
    LengthMetadata, AxisDirections, np.complex128
]
type BasisWithLengthLike = Basis[LengthMetadata, np.complex128]
type BasisLike = Basis[BasisMetadata, np.complex128]
type FundamentalPositionBasis = FundamentalBasis[LengthMetadata]
type FundamentalPositionBasis3d = FundamentalPositionBasis
type FundamentalTransformedPositionBasis = TransformedBasis[LengthMetadata]
type FundamentalTransformedPositionBasis3d = FundamentalTransformedPositionBasis
type FundamentalTransformedBasis = TransformedBasis[BasisMetadata]
type TruncatedBasis = TruncatedBasisNew[Any, np.complex128]
type EvenlySpacedBasis = EvenlySpacedBasisNew[Any, np.complex128]
type EvenlySpacedTransformedPositionBasis = EvenlySpacedBasisNew[
    LengthMetadata, np.complex128
]
type ExplicitBlockFractionBasis = Basis[ExplicitBlochFractionMetadata, np.complex128]
type BasisWithBlockFractionLike = Basis[BlochFractionMetadata, np.complex128]

type ExplicitBasis = ExplicitUnitaryBasis[BasisMetadata, np.complex128]
type ExplicitBasis3d = ExplicitUnitaryBasis[BasisMetadata, np.complex128]
type ExplicitStackedBasisWithLength = ExplicitUnitaryBasis[
    StackedMetadata[LengthMetadata, AxisDirections], np.complex128
]


def convert_vector(
    vector: np.ndarray[
        Any,
        np.dtype[np.complex128]
        | np.dtype[np.float64]
        | np.dtype[np.float64 | np.complex128],
    ],
    initial_basis: Basis[BasisMetadata, Any],
    final_basis: Basis[BasisMetadata, Any],
    axis: int = -1,
) -> np.ndarray[Any, np.dtype[np.complex128]]:
    """
    Convert a vector, expressed in terms of the given basis from_config in the basis to_config.

    Parameters
    ----------
    vector : np.ndarray[tuple[int], np.dtype[np.complex_] | np.dtype[np.float_]]
        the vector to convert
    from_config : _B3d0Inv
    to_config : _B3d1Inv
    axis : int, optional
        axis along which to convert, by default -1

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.complex_]]
    """
    return initial_basis.__convert_vector_into__(
        vector.astype(np.complex128), final_basis, axis
    )
