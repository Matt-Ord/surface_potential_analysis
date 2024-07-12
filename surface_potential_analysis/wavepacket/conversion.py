from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, overload

import numpy as np

from surface_potential_analysis.basis.basis_like import (
    BasisLike,
    BasisWithLengthLike,
    convert_vector,
)
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasisLike,
    StackedBasisWithVolumeLike,
    TupleBasis,
    TupleBasisWithLengthLike,
)
from surface_potential_analysis.stacked_basis.build import (
    fundamental_stacked_basis_from_shape,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    stacked_basis_as_fundamental_position_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        FundamentalPositionBasis,
        FundamentalTransformedPositionBasis,
    )
    from surface_potential_analysis.basis.stacked_basis import (
        TupleBasisLike,
    )
    from surface_potential_analysis.wavepacket.wavepacket import (
        BlochWavefunctionList,
    )

    _B0 = TypeVar("_B0", bound=StackedBasisWithVolumeLike[Any, Any, Any])
    _B1 = TypeVar("_B1", bound=StackedBasisWithVolumeLike[Any, Any, Any])

    _B2 = TypeVar("_B2", bound=StackedBasisLike[Any, Any, Any])
    _B3 = TypeVar("_B3", bound=StackedBasisLike[Any, Any, Any])

    _BL0 = TypeVar("_BL0", bound=BasisWithLengthLike[Any, Any, Any])


@overload
def convert_wavepacket_to_basis(
    wavepacket: BlochWavefunctionList[_B2, _B0],
    *,
    list_basis: _B3,
    basis: None = None,
) -> BlochWavefunctionList[_B3, _B0]:
    ...


@overload
def convert_wavepacket_to_basis(
    wavepacket: BlochWavefunctionList[_B2, _B0],
    *,
    list_basis: _B3,
    basis: _B1,
) -> BlochWavefunctionList[_B3, _B1]:
    ...


@overload
def convert_wavepacket_to_basis(
    wavepacket: BlochWavefunctionList[_B2, _B0],
    *,
    list_basis: None = None,
    basis: _B1,
) -> BlochWavefunctionList[_B2, _B1]:
    ...


@overload
def convert_wavepacket_to_basis(
    wavepacket: BlochWavefunctionList[_B2, _B0],
    *,
    list_basis: None = None,
    basis: None = None,
) -> BlochWavefunctionList[_B2, _B0]:
    ...


def convert_wavepacket_to_basis(
    wavepacket: BlochWavefunctionList[_B2, _B0],
    *,
    list_basis: BasisLike[Any, Any] | None = None,
    basis: BasisLike[Any, Any] | None = None,
) -> Any:
    """
    Given a wavepacket convert it to the given basis.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv,  _B0Inv]
    basis : _B1Inv

    Returns
    -------
    Wavepacket[_S0Inv, _B1Inv]
    """
    list_basis = wavepacket["basis"][0] if list_basis is None else list_basis
    basis = wavepacket["basis"][1] if basis is None else basis
    vectors = convert_vector(
        wavepacket["data"].reshape(wavepacket["basis"].shape),
        wavepacket["basis"][0],
        list_basis,
        axis=0,
    )
    vectors = convert_vector(vectors, wavepacket["basis"][1], basis, axis=1)
    return {"basis": TupleBasis(list_basis, basis), "data": vectors.reshape(-1)}


def convert_wavepacket_to_position_basis(
    wavepacket: BlochWavefunctionList[_B2, TupleBasisWithLengthLike[*tuple[_BL0, ...]]],
) -> BlochWavefunctionList[
    _B2, TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]
]:
    """
    Convert a wavepacket to the fundamental position basis.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, _B3d0Inv]

    Returns
    -------
    Wavepacket[_NS0Inv, _NS1Inv, TupleBasisLike[tuple[PositionBasis[int], PositionBasis[int], PositionBasis[int]]]
    """
    return convert_wavepacket_to_basis(
        wavepacket,
        basis=stacked_basis_as_fundamental_position_basis(wavepacket["basis"][1]),
    )


@overload
def convert_wavepacket_to_fundamental_momentum_basis(
    wavepacket: BlochWavefunctionList[_B2, _B1],
    *,
    list_basis: _B3,
) -> BlochWavefunctionList[
    _B3,
    TupleBasisWithLengthLike[
        *tuple[FundamentalTransformedPositionBasis[Any, Any], ...]
    ],
]:
    ...


@overload
def convert_wavepacket_to_fundamental_momentum_basis(
    wavepacket: BlochWavefunctionList[_B2, _B1],
    *,
    list_basis: None = None,
) -> BlochWavefunctionList[
    _B2,
    TupleBasisWithLengthLike[
        *tuple[FundamentalTransformedPositionBasis[Any, Any], ...]
    ],
]:
    ...


def convert_wavepacket_to_fundamental_momentum_basis(
    wavepacket: BlochWavefunctionList[_B2, TupleBasisWithLengthLike[*tuple[_BL0, ...]]],
    *,
    list_basis: TupleBasisLike[*tuple[Any, ...]] | None = None,
) -> Any:
    """
    Convert a wavepacket to the fundamental position basis.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, _B3d0Inv]

    Returns
    -------
    Wavepacket[_NS0Inv, _NS1Inv, TupleBasisLike[tuple[PositionBasis[int], PositionBasis[int], PositionBasis[int]]]
    """
    return convert_wavepacket_to_basis(
        wavepacket,
        basis=stacked_basis_as_fundamental_momentum_basis(wavepacket["basis"][1]),
        list_basis=wavepacket["basis"][0] if list_basis is None else list_basis,
    )


def convert_wavepacket_to_shape(
    wavepacket: BlochWavefunctionList[_B2, _B1], shape: tuple[int, ...]
) -> BlochWavefunctionList[Any, _B1]:
    """
    Convert the wavepacket to the given shape.

    Note that BasisUtil(wavepacket["list_basis"]).shape must be divisible by shape

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]
    shape : _S1Inv

    Returns
    -------
    Wavepacket[_S1Inv, _B0Inv]
    """
    old_shape = wavepacket["basis"][0].shape
    slices = tuple(
        slice(None, None, s0 // s1) for (s0, s1) in zip(old_shape, shape, strict=True)
    )
    np.testing.assert_array_almost_equal(
        old_shape,
        [s.step * s1 for (s, s1) in zip(slices, shape, strict=True)],
    )
    return {
        "basis": TupleBasis(
            fundamental_stacked_basis_from_shape(shape), wavepacket["basis"][1]
        ),
        "data": wavepacket["data"]
        .reshape(*old_shape, -1)[*slices, :]
        .reshape(np.prod(shape), -1),
    }
