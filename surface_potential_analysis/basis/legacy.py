from __future__ import annotations

from typing import Any, Self, TypeVarTuple

import numpy as np
from slate.basis import Basis, FundamentalBasis, TransformedBasis, VariadicTupleBasis
from slate.basis import EvenlySpacedBasis as EvenlySpacedBasisNew
from slate.basis import TruncatedBasis as TruncatedBasisNew
from slate.basis import TupleBasis as TupleBasisNew
from slate.explicit_basis._explicit_basis import ExplicitUnitaryBasis
from slate.metadata import BasisMetadata
from slate.metadata._metadata import LabelSpacing
from slate.metadata.length import AxisDirections, LengthMetadata, SpacedLengthMetadata
from slate.metadata.stacked import StackedMetadata

from surface_potential_analysis.basis.block_fraction_basis import (
    BlochFractionMetadata,
    ExplicitBlochFractionMetadata,
)

TS = TypeVarTuple("TS")


class TupleBasis[*TS](VariadicTupleBasis[*TS, None, np.complex128]):
    def __init__(self: Self, *args: *TS) -> None: ...

    def __call__(self, *args: *TS) -> VariadicTupleBasis[*TS, None, np.complex128]:
        return VariadicTupleBasis(args, None)


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


class FundamentalPositionBasis(FundamentalBasis[LengthMetadata]):
    def __init__(self: Self, delta: np.ndarray[Any, Any], n: int) -> None: ...

    def __call__(
        self, delta: np.ndarray[Any, Any], n: int
    ) -> FundamentalBasis[LengthMetadata]:
        return FundamentalBasis(
            SpacedLengthMetadata(
                (n,), spacing=LabelSpacing(delta=np.linalg.norm(delta).item())
            )
        )


type FundamentalPositionBasis3d = FundamentalPositionBasis
type FundamentalTransformedPositionBasis = TransformedBasis[LengthMetadata]
type FundamentalTransformedPositionBasis3d = FundamentalTransformedPositionBasis


class FundamentalTransformedBasis(TransformedBasis[BasisMetadata]):
    def __init__(self: Self, n: int) -> None: ...

    def __call__(self, n: int) -> TransformedBasis[BasisMetadata]:
        return TransformedBasis(FundamentalBasis.from_shape((n,)))


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
