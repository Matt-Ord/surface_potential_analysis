from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, overload

import numpy as np

from surface_potential_analysis.basis.legacy import (
    BasisLike,
    BasisWithLengthLike,
    StackedBasisLike,
    StackedBasisWithVolumeLike,
    TupleBasis,
    TupleBasisWithLengthLike,
    convert_vector,
)
from surface_potential_analysis.stacked_basis.build import (
    fundamental_stacked_basis_from_shape,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    tuple_basis_as_fundamental,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.legacy import (
        FundamentalPositionBasis,
        FundamentalTransformedPositionBasis,
        TupleBasisLike,
    )
    from surface_potential_analysis.wavepacket.wavepacket import (
        BlochWavefunctionList,
        BlochWavefunctionListWithEigenvaluesList,
    )

    _B0 = TypeVar("_B0", bound=StackedBasisWithVolumeLike)
    _B1 = TypeVar("_B1", bound=StackedBasisWithVolumeLike)

    _B2 = TypeVar("_B2", bound=StackedBasisLike)
    _B3 = TypeVar("_B3", bound=StackedBasisLike)
    _B4 = TypeVar("_B4", bound=BasisLike)

    _BL0 = TypeVar("_BL0", bound=BasisWithLengthLike)


@overload
def convert_wavepacket_to_basis(
    wavepacket: BlochWavefunctionList[_B2, _B0],
    *,
    list_basis: _B3,
    basis: None = None,
) -> BlochWavefunctionList[_B3, _B0]: ...


@overload
def convert_wavepacket_to_basis(
    wavepacket: BlochWavefunctionList[_B2, _B0],
    *,
    list_basis: _B3,
    basis: _B1,
) -> BlochWavefunctionList[_B3, _B1]: ...


@overload
def convert_wavepacket_to_basis(
    wavepacket: BlochWavefunctionList[_B2, _B0],
    *,
    list_basis: None = None,
    basis: _B1,
) -> BlochWavefunctionList[_B2, _B1]: ...


@overload
def convert_wavepacket_to_basis(
    wavepacket: BlochWavefunctionList[_B2, _B0],
    *,
    list_basis: None = None,
    basis: None = None,
) -> BlochWavefunctionList[_B2, _B0]: ...


def convert_wavepacket_to_basis(
    wavepacket: BlochWavefunctionList[_B2, _B0],
    *,
    list_basis: BasisLike | None = None,
    basis: BasisLike | None = None,
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
    return {
        "basis": VariadicTupleBasis((list_basis, basis), None),
        "data": vectors.reshape(-1),
    }


@overload
def convert_wavepacket_with_eigenvalues_to_basis(
    wavepacket: BlochWavefunctionListWithEigenvaluesList[_B4, _B2, _B0],
    *,
    list_basis: _B3,
    basis: None = None,
) -> BlochWavefunctionListWithEigenvaluesList[_B4, _B3, _B0]: ...


@overload
def convert_wavepacket_with_eigenvalues_to_basis(
    wavepacket: BlochWavefunctionListWithEigenvaluesList[_B4, _B2, _B0],
    *,
    list_basis: _B3,
    basis: _B1,
) -> BlochWavefunctionListWithEigenvaluesList[_B4, _B3, _B1]: ...


@overload
def convert_wavepacket_with_eigenvalues_to_basis(
    wavepacket: BlochWavefunctionListWithEigenvaluesList[_B4, _B2, _B0],
    *,
    list_basis: None = None,
    basis: _B1,
) -> BlochWavefunctionListWithEigenvaluesList[_B4, _B2, _B1]: ...


@overload
def convert_wavepacket_with_eigenvalues_to_basis(
    wavepacket: BlochWavefunctionListWithEigenvaluesList[_B4, _B2, _B0],
    *,
    list_basis: None = None,
    basis: None = None,
) -> BlochWavefunctionListWithEigenvaluesList[_B4, _B2, _B0]: ...


def convert_wavepacket_with_eigenvalues_to_basis(
    wavepacket: BlochWavefunctionListWithEigenvaluesList[_B4, _B2, _B0],
    *,
    list_basis: BasisLike | None = None,
    basis: BasisLike | None = None,
) -> BlochWavefunctionListWithEigenvaluesList[Any, Any, Any]:
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
    list_basis = wavepacket["basis"][0][1] if list_basis is None else list_basis
    basis = wavepacket["basis"][1] if basis is None else basis
    vectors = convert_vector(
        wavepacket["data"].reshape(
            *wavepacket["basis"][0].shape, wavepacket["basis"][1].size
        ),
        wavepacket["basis"][0][1],
        list_basis,
        axis=1,
    )
    vectors = convert_vector(vectors, wavepacket["basis"][1], basis, axis=2)

    eigenvalues = convert_vector(
        wavepacket["eigenvalue"].reshape(wavepacket["basis"][0].shape),
        wavepacket["basis"][0][1],
        list_basis,
        axis=1,
    )
    return {
        "basis": VariadicTupleBasis(
            (VariadicTupleBasis((wavepacket["basis"][0][0], list_basis), None), None),
            basis,
        ),
        "data": vectors.reshape(-1),
        "eigenvalue": eigenvalues,
    }


def convert_wavepacket_to_position_basis(
    wavepacket: BlochWavefunctionList[_B2, TupleBasisWithLengthLike[*tuple[_BL0, ...]]],
) -> BlochWavefunctionList[
    _B2, TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]]
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
        basis=tuple_basis_as_fundamental(wavepacket["basis"][1]),
    )


@overload
def convert_wavepacket_to_fundamental_momentum_basis(
    wavepacket: BlochWavefunctionList[_B2, _B1],
    *,
    list_basis: _B3,
) -> BlochWavefunctionList[
    _B3,
    TupleBasisWithLengthLike[*tuple[FundamentalTransformedPositionBasis, ...]],
]: ...


@overload
def convert_wavepacket_to_fundamental_momentum_basis(
    wavepacket: BlochWavefunctionList[_B2, _B1],
    *,
    list_basis: None = None,
) -> BlochWavefunctionList[
    _B2,
    TupleBasisWithLengthLike[*tuple[FundamentalTransformedPositionBasis, ...]],
]: ...


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
