from __future__ import annotations

import warnings
from itertools import starmap
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import (
    FundamentalBasis,
    FundamentalPositionBasis,
)
from surface_potential_analysis.basis.basis_like import (
    AxisVector,
    BasisLike,
    BasisWithLengthLike,
    convert_vector,
)
from surface_potential_analysis.basis.evenly_spaced_basis import (
    EvenlySpacedBasis,
    EvenlySpacedTransformedPositionBasis,
)
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasisLike,
    StackedBasisWithVolumeLike,
    TupleBasis,
    TupleBasisLike,
    TupleBasisWithLengthLike,
)
from surface_potential_analysis.basis.util import (
    BasisUtil,
)
from surface_potential_analysis.operator.operator import (
    SingleBasisDiagonalOperator,
    average_eigenvalues,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_basis,
    stacked_basis_as_fundamental_position_basis,
)
from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_eigenvectors_hermitian,
)
from surface_potential_analysis.state_vector.eigenstate_collection import (
    EigenstateList,
    get_eigenvalues_list,
)
from surface_potential_analysis.state_vector.state_vector_list import StateVectorList

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from surface_potential_analysis.operator.operator import SingleBasisOperator
    from surface_potential_analysis.types import (
        ShapeLike,
        SingleFlatIndexLike,
    )

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_ND0Inv = TypeVar("_ND0Inv", bound=int)


_B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
_ESB0 = TypeVar("_ESB0", bound=EvenlySpacedBasis[Any, Any, Any])
_BL0 = TypeVar("_BL0", bound=BasisWithLengthLike[Any, Any, Any])

_SB0 = TypeVar("_SB0", bound=StackedBasisLike[Any, Any, Any])
_SB1 = TypeVar("_SB1", bound=StackedBasisLike[Any, Any, Any])
_TB0 = TypeVar("_TB0", bound=TupleBasisLike[*tuple[Any, ...]])
_SBV0 = TypeVar("_SBV0", bound=StackedBasisWithVolumeLike[Any, Any, Any])


BlochWavefunctionListBasis = TupleBasisLike[_SB0, _SB1]

BlochWavefunctionList = StateVectorList[_SB0, _SB1]
"""represents an approximation of a Wannier function."""


BlochWavefunctionListWithEigenvalues = EigenstateList[_SB0, _SB1]
"""represents an approximation of a Wannier function."""


BlochWavefunctionListList = StateVectorList[TupleBasisLike[_B0, _SB0], _SB1]
"""represents a list of wavefunctions."""


BlochWavefunctionListWithEigenvaluesList = EigenstateList[
    TupleBasisLike[_B0, _SB0], _SB1
]
"""
Represents a collection of bloch wavefunction lists.

An individual wavefunction is stored per band and per sample in the first brillouin zone.

An EigenstateList[TupleBasisLike[_B0, _SB0], _SB1] where
- _B0   - The basis of Bands, this is essentially just
the basis of the 'list' of individual wavepackets
- _SB0  - The basis of the samples in the first brillouin zone
- _SB1  - The basis of the individual bloch wavefunctions

The underlying wavepackets have a basis of TupleBasisLike[_SB0, _SB1]
however for convenience we store the data as a list of bloch state for each
sample in the first brillouin zone
"""


def get_fundamental_unfurled_sample_basis_momentum(
    basis: BlochWavefunctionListBasis[_TB0, _SBV0],
    offsets: tuple[int, ...] | None = None,
) -> TupleBasis[*tuple[EvenlySpacedTransformedPositionBasis[int, int, int, int], ...]]:
    """
    Get the basis of an individual wavefunction from the wavepacket.

    This takes states from the fundamental list_basis, for the sample at offset
    """
    offsets = (0,) * basis[0].ndim if offsets is None else offsets
    basis_x = stacked_basis_as_fundamental_position_basis(basis[1])
    return TupleBasis(
        *tuple(
            EvenlySpacedTransformedPositionBasis[int, int, int, int](
                delta_x=b1.delta_x * b0.fundamental_n,
                n=b1.fundamental_n,
                step=b0.fundamental_n,
                offset=offset,
            )
            for (b0, b1, offset) in zip(basis[0], basis_x, offsets, strict=True)
        )
    )


def get_sample_basis(
    basis: BlochWavefunctionListBasis[_TB0, _SBV0],
) -> TupleBasis[*tuple[BasisWithLengthLike[Any, Any, Any], ...]]:
    """
    Given the basis for a wavepacket, get the basis used to sample the packet.

    Parameters
    ----------
    basis : Basis[_ND0Inv]
    shape : _S0Inv

    Returns
    -------
    Basis[_ND0Inv]
    """
    # TODO: currently only supports fundamental list_basis ...
    basis_x = stacked_basis_as_fundamental_position_basis(basis[1])
    return TupleBasis(
        *tuple(
            FundamentalPositionBasis(b1.delta_x * b0.n, b0.n)
            for (b0, b1) in zip(basis[0], basis_x, strict=True)
        )
    )


class UnfurledBasis(TupleBasis[_B0, BasisWithLengthLike[_L0Inv, _L1Inv, _ND0Inv]]):
    """Represent the basis of an unfurled wavepacket."""

    @property
    def delta_x(self) -> AxisVector[_ND0Inv]:  # noqa: D102
        return self[1].delta_x * self[0].n  # type: ignore[no-any-return]


def get_unfurled_basis(
    basis: BlochWavefunctionListBasis[
        _TB0, TupleBasisWithLengthLike[*tuple[_BL0, ...]]
    ],
) -> TupleBasisWithLengthLike[*tuple[UnfurledBasis[Any, Any, Any, Any], ...]]:
    """
    Given the basis for a wavepacket, get the basis for the unfurled wavepacket.

    Parameters
    ----------
    basis : Basis[_ND0Inv]
    shape : _S0Inv

    Returns
    -------
    Basis[_ND0Inv]
    """
    return TupleBasis(
        *tuple(starmap(UnfurledBasis, zip(basis[0], basis[1], strict=True)))
    )


def get_furled_basis(
    basis: TupleBasisLike[*tuple[_BL0, ...]],
    shape: ShapeLike,
) -> TupleBasis[*tuple[BasisWithLengthLike[Any, Any, Any], ...]]:
    """
    Given the basis for an eigenstate, get the basis used for the furled wavepacket.

    Parameters
    ----------
    basis : Basis[_ND0Inv]
    shape : _S0Inv

    Returns
    -------
    Basis[_ND0Inv]
    """
    return TupleBasis(
        *tuple(
            FundamentalPositionBasis(ax.delta_x // n, ax.n // n)
            for (ax, n) in zip(basis, shape, strict=True)
        )
    )


def get_wavepacket_sample_fractions(
    list_basis: StackedBasisLike[Any, Any, Any],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """
    Get the frequencies of the samples in a wavepacket, as a fraction of dk.

    Parameters
    ----------
    shape : np.ndarray[tuple[_NDInv], np.dtype[np.int_]]

    Returns
    -------
    np.ndarray[tuple[Literal[_NDInv], int], np.dtype[np.float_]]
    """
    util = BasisUtil(list_basis)
    fundamental_fractions = (
        util.fundamental_stacked_nk_points
        / np.array(util.fundamental_shape, dtype=np.int_)[:, np.newaxis]
    )
    fundamental_basis = stacked_basis_as_fundamental_basis(list_basis)
    with warnings.catch_warnings(
        category=np.exceptions.ComplexWarning, action="ignore"
    ):
        return convert_vector(
            fundamental_fractions, fundamental_basis, list_basis
        ).astype(np.float64)


def get_wavepacket_sample_frequencies(
    basis: BlochWavefunctionListBasis[_SB0, TupleBasisLike[*tuple[_BL0, ...]]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """
    Get the frequencies used in a given wavepacket.

    Parameters
    ----------
    basis : Basis[_ND0Inv]
    shape : tuple length _ND0Inv

    Returns
    -------
    np.ndarray[tuple[_ND0Inv, int], np.dtype[np.float_]]
    """
    # TODO: currently only supports fundamental list_basis ...
    sample_basis = get_sample_basis(basis)
    util = BasisUtil(sample_basis)
    return util.fundamental_stacked_k_points


def generate_wavepacket(
    hamiltonian_generator: Callable[
        [np.ndarray[tuple[_ND0Inv], np.dtype[np.float64]]],
        SingleBasisOperator[_SB1],
    ],
    list_basis: _SB0,
    save_bands: _ESB0,
) -> BlochWavefunctionListWithEigenvaluesList[_ESB0, _SB0, _SB1]:
    """
    Generate a wavepacket with the given number of samples.

    Parameters
    ----------
    hamiltonian_generator : Callable[[np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]], Hamiltonian[_B3d0Inv]]
    shape : _S0Inv
    save_bands : np.ndarray[tuple[int], np.dtype[np.int_]] | None, optional

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[Wavepacket[_NS0Inv, _NS1Inv, _B3d0Inv]]]
    """
    bloch_fractions = get_wavepacket_sample_fractions(list_basis)
    h = hamiltonian_generator(bloch_fractions[:, 0])
    assert list_basis.ndim == h["basis"][0].ndim
    basis_size = h["basis"][0].n

    subset_by_index = (
        save_bands.offset,
        save_bands.offset + save_bands.step * (save_bands.n - 1),
    )

    n_samples = list_basis.n
    vectors = np.empty((save_bands.n, n_samples, basis_size), dtype=np.complex128)
    energies = np.empty((save_bands.n, n_samples), dtype=np.complex128)

    for i in range(list_basis.n):
        h = hamiltonian_generator(bloch_fractions[:, i])
        eigenstates = calculate_eigenvectors_hermitian(h, subset_by_index)

        for b in range(save_bands.n):
            band_idx = save_bands.step * b
            vectors[b][i] = eigenstates["data"].reshape(-1, basis_size)[band_idx]
            energies[b][i] = eigenstates["eigenvalue"][band_idx]
    return {
        "basis": TupleBasis(TupleBasis(save_bands, list_basis), h["basis"][0]),
        "data": vectors.reshape(-1),
        "eigenvalue": energies.reshape(-1),
    }


def get_wavepacket_basis(
    wavepackets: BlochWavefunctionListList[_B0, _SB0, _SB1],
) -> BlochWavefunctionListBasis[_SB0, _SB1]:
    """
    Get the basis of the wavepacket.

    Parameters
    ----------
    wavepackets : WavepacketList[_B0, _SB0, _SB1]

    Returns
    -------
    WavepacketBasis[_SB0, _SB1]
    """
    return TupleBasis(wavepackets["basis"][0][1], wavepackets["basis"][1])


def get_wavepacket(
    wavepackets: BlochWavefunctionListList[_B0, _SB0, _SB1],
    idx: SingleFlatIndexLike,
) -> BlochWavefunctionList[_SB0, _SB1]:
    """
    Get the wavepacket at idx.

    Parameters
    ----------
    wavepackets : WavepacketList[_B0, _SB0, _SB1]
    idx : SingleFlatIndexLike

    Returns
    -------
    Wavepacket[_SB0, _SB1]
    """
    return {
        "basis": get_wavepacket_basis(wavepackets),
        "data": wavepackets["data"].reshape(wavepackets["basis"][0][0].n, -1)[idx],
    }


def get_wavepacket_with_eigenvalues(
    wavepackets: BlochWavefunctionListWithEigenvaluesList[_B0, _SB0, _SB1],
    idx: SingleFlatIndexLike,
) -> BlochWavefunctionListWithEigenvalues[_SB0, _SB1]:
    """
    Get the wavepacket at idx.

    Parameters
    ----------
    wavepackets : WavepacketList[_B0, _SB0, _SB1]
    idx : SingleFlatIndexLike

    Returns
    -------
    Wavepacket[_SB0, _SB1]
    """
    return {
        "basis": get_wavepacket_basis(wavepackets),
        "eigenvalue": wavepackets["eigenvalue"][idx],
        "data": wavepackets["data"].reshape(wavepackets["basis"][0][0].n, -1)[idx],
    }


def get_wavepackets(
    wavepackets: BlochWavefunctionListList[_B0, _SB0, _SB1],
    idx: slice,
) -> BlochWavefunctionListList[BasisLike[Any, Any], _SB0, _SB1]:
    """
    Get the wavepackets at the given slice.

    Parameters
    ----------
    wavepackets : WavepacketList[_B0, _SB0, _SB1]
    idx : slice

    Returns
    -------
    WavepacketList[BasisLike[Any, Any], _SB0, _SB1]
    """
    stacked = wavepackets["data"].reshape(wavepackets["basis"][0][0].n, -1)
    stacked = stacked[idx]
    return {
        "basis": TupleBasis(
            TupleBasis(FundamentalBasis(len(stacked)), wavepackets["basis"][0][1]),
            wavepackets["basis"][1],
        ),
        "data": stacked.reshape(-1),
    }


def get_wavepackets_with_eigenvalues(
    wavepackets: BlochWavefunctionListWithEigenvaluesList[_B0, _SB0, _SB1],
    idx: slice,
) -> BlochWavefunctionListWithEigenvaluesList[BasisLike[Any, Any], _SB0, _SB1]:
    """
    Get the wavepackets at the given slice.

    Parameters
    ----------
    wavepackets : WavepacketList[_B0, _SB0, _SB1]
    idx : slice

    Returns
    -------
    WavepacketList[BasisLike[Any, Any], _SB0, _SB1]
    """
    stacked = wavepackets["data"].reshape(wavepackets["basis"][0][0].n, -1)
    stacked = stacked[idx]
    return {
        "basis": TupleBasis(
            TupleBasis(FundamentalBasis(len(stacked)), wavepackets["basis"][0][1]),
            wavepackets["basis"][1],
        ),
        "data": stacked.reshape(-1),
        "eigenvalue": wavepackets["eigenvalue"]
        .reshape(wavepackets["basis"][0][0].n, -1)[idx]
        .ravel(),
    }


def as_wavepacket_list(
    wavepackets: Iterable[BlochWavefunctionList[_SB0, _SB1]],
) -> BlochWavefunctionListList[FundamentalBasis[int], _SB0, _SB1]:
    """
    Convert an iterable of wavepackets into a wavepacket list.

    Parameters
    ----------
    wavepackets : Iterable[Wavepacket[_SB0, _SB1]]

    Returns
    -------
    WavepacketList[FundamentalBasis[int], _SB0, _SB1]
    """
    wavepacket_0 = next(iter(wavepackets))
    vectors = np.array([w["data"] for w in wavepackets])
    return {
        "basis": TupleBasis(
            TupleBasis(FundamentalBasis(len(vectors)), wavepacket_0["basis"][0]),
            wavepacket_0["basis"][1],
        ),
        "data": vectors.reshape(-1),
    }


def wavepacket_list_into_iter(
    wavepackets: BlochWavefunctionListList[Any, _SB0, _SB1],
) -> Iterable[BlochWavefunctionList[_SB0, _SB1]]:
    """
    Iterate over wavepackets in the list.

    Parameters
    ----------
    wavepackets : WavepacketList[Any, _SB0, _SB1]

    Returns
    -------
    Iterable[Wavepacket[_SB0, _SB1]]
    """
    stacked = wavepackets["data"].reshape(wavepackets["basis"][0][0].n, -1)
    basis = get_wavepacket_basis(wavepackets)
    return [{"basis": basis, "data": data} for data in stacked]


def get_average_eigenvalues(
    wavepackets: BlochWavefunctionListWithEigenvaluesList[_B0, Any, Any],
) -> SingleBasisDiagonalOperator[_B0]:
    """
    Get the band averaged eigenvalues of a wavepacket.

    Parameters
    ----------
    wavepackets : WavepacketWithEigenvaluesList[_B0, Any, Any]

    Returns
    -------
    SingleBasisDiagonalOperator[_B0]
    """
    eigenvalues = get_eigenvalues_list(wavepackets)
    averaged = average_eigenvalues(eigenvalues, axis=(1,))
    return {
        "basis": TupleBasis(averaged["basis"][0][0], averaged["basis"][0][0]),
        "data": averaged["data"],
    }
