from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from slate.basis.evenly_spaced import EvenlySpacedBasis, Spacing
from slate.basis.stacked._tuple_basis import VariadicTupleBasis
from slate.metadata._metadata import BasisMetadata, LabelSpacing
from slate.metadata.length import SpacedLengthMetadata

from surface_potential_analysis.basis.legacy import (
    BasisLike,
    BasisWithBlockFractionLike,
    BasisWithLengthLike,
    ExplicitBlockFractionBasis,
    FundamentalBasis,
    FundamentalTransformedPositionBasis,
    StackedBasisLike,
    StackedBasisWithVolumeLike,
    TruncatedBasis,
    TupleBasis,
    TupleBasisLike,
    convert_vector,
)
from surface_potential_analysis.basis.util import (
    BasisUtil,
)
from surface_potential_analysis.operator.operator import (
    SingleBasisDiagonalOperator,
    average_eigenvalues,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    tuple_basis_as_fundamental,
    tuple_basis_as_transformed_fundamental,
)
from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_eigenvectors_hermitian,
)
from surface_potential_analysis.state_vector.eigenstate_list import (
    EigenstateList,
    get_eigenvalues_list,
)
from surface_potential_analysis.state_vector.state_vector_list import StateVectorList

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from surface_potential_analysis.operator.operator import SingleBasisOperator
    from surface_potential_analysis.types import (
        SingleFlatIndexLike,
    )


_ND0Inv = TypeVar("_ND0Inv", bound=int)


_B0 = TypeVar("_B0", bound=BasisLike)
_TRB0 = TypeVar("_TRB0", bound=TruncatedBasis | EvenlySpacedBasis)
_SB0 = TypeVar("_SB0", bound=StackedBasisLike)
_SB1 = TypeVar("_SB1", bound=StackedBasisLike)
_TB0 = TypeVar("_TB0", bound=TupleBasisLike[*tuple[Any, ...]])
_SBV0 = TypeVar("_SBV0", bound=StackedBasisWithVolumeLike)


BlochWavefunctionListBasis = TupleBasisLike[_SB0, _SB1]

BlochWavefunctionList = StateVectorList[_SB0, _SB1]
"""represents an approximation of a Wannier function."""


BlochWavefunctionListWithEigenvalues = EigenstateList[_SB0, _SB1]
"""represents an approximation of a Wannier function."""

BlochWavefunctionListListBasis = TupleBasisLike[TupleBasisLike[_B0, _SB0], _SB1]

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
) -> TupleBasis[*tuple[EvenlySpacedBasis, ...]]:
    """
    Get the basis of an individual wavefunction from the wavepacket.

    This takes states from the fundamental list_basis, for the sample at offset
    """
    offsets = (0,) * basis[0].n_dim if offsets is None else offsets
    basis_x = tuple_basis_as_fundamental(basis[1])
    return TupleBasis(
        tuple(
            EvenlySpacedBasis(
                Spacing(n = ),
                FundamentalBasis(basis_x.metadata.children[i]),
                # delta_x=basis_x[i].delta_x * basis[0][i].fundamental_n,
                # n=basis_x[i].fundamental_n,
                # step=basis[0][i].fundamental_n,
                # offset=offset,
            )
            for (i, offset) in enumerate(offsets)
        ),
        None,
    )


def get_fundamental_sample_basis(
    basis: BlochWavefunctionListBasis[_SB0, _SBV0],
) -> TupleBasis[*tuple[BasisWithLengthLike, ...]]:
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
    basis_x = stacked_basis_as_fundamental_momentum_basis(basis[1])
    return VariadicTupleBasis(
        tuple(
            FundamentalTransformedPositionBasis(
                basis_x[i].delta_x * basis[0].fundamental_shape[i],
                basis[0].fundamental_shape[i],
            )
            for (i) in range(basis[0].n_dim)
        ),
        None,
    )


def get_fundamental_unfurled_basis(
    basis: BlochWavefunctionListBasis[_SB0, _SBV0],
) -> TupleBasis[*tuple[FundamentalTransformedPositionBasis, ...]]:
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
    basis_0 = tuple_basis_as_transformed_fundamental(basis[0])
    basis_1 = stacked_basis_as_fundamental_momentum_basis(basis[1])

    return VariadicTupleBasis(
        tuple(
            FundamentalTransformedPositionBasis(
                basis_0[i].n * basis_1[i].delta_x,
                basis_0[i].n * basis_1[i].n,
            )
            for i in range(basis_0.n_dim)
        ),
        None,
    )


def get_wavepacket_sample_fractions(
    list_basis: StackedBasisLike,
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
    fundamental_basis = tuple_basis_as_transformed_fundamental(list_basis)
    with warnings.catch_warnings(
        category=np.exceptions.ComplexWarning, action="ignore"
    ):
        return convert_vector(
            fundamental_fractions, fundamental_basis, list_basis
        ).astype(np.float64)


def get_wavepacket_fundamental_sample_frequencies(
    basis: BlochWavefunctionListBasis[_SB0, _SBV0],
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
    sample_basis = get_fundamental_sample_basis(basis)
    util = BasisUtil(sample_basis)
    return util.fundamental_stacked_k_points


_BF0 = TypeVar("_BF0", bound=BasisWithBlockFractionLike[Any, Any])


def generate_uneven_wavepacket(
    hamiltonian_generator: Callable[
        [np.ndarray[tuple[_ND0Inv], np.dtype[np.float64]]],
        SingleBasisOperator[_SB1],
    ],
    list_basis: _BF0,
    band_basis: _TRB0,
) -> EigenstateList[TupleBasis[_TRB0, _BF0], _SB1]:
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
    bloch_fractions = list_basis.bloch_fractions
    h = hamiltonian_generator(bloch_fractions[:, 0])
    assert list_basis.n_dim == h["basis"][0].sizedim
    basis_size = h["basis"][0].size

    offset, step = (
        (band_basis.offset, band_basis.step)
        if isinstance(band_basis, EvenlySpacedBasis)
        else (0, 1)
    )
    subset_by_index = (
        offset,
        offset + step * (band_basis.n - 1),
    )

    n_samples = list_basis.n
    vectors = np.empty((band_basis.n, n_samples, basis_size), dtype=np.complex128)
    energies = np.empty((band_basis.n, n_samples), dtype=np.complex128)

    for i in range(list_basis.n):
        h = hamiltonian_generator(bloch_fractions[:, i])
        eigenstates = calculate_eigenvectors_hermitian(h, subset_by_index)

        for b in range(band_basis.n):
            band_idx = step * b
            vectors[b][i] = eigenstates["data"].reshape(-1, basis_size)[band_idx]
            energies[b][i] = eigenstates["eigenvalue"][band_idx]

    return {
        "basis": VariadicTupleBasis(
            (VariadicTupleBasis((band_basis, list_basis), None), None), h["basis"][0]
        ),
        "data": vectors.reshape(-1),
        "eigenvalue": energies.reshape(-1),
    }


def generate_wavepacket(
    hamiltonian_generator: Callable[
        [np.ndarray[tuple[_ND0Inv], np.dtype[np.float64]]],
        SingleBasisOperator[_SB1],
    ],
    list_basis: _SB0,
    band_basis: _TRB0,
) -> BlochWavefunctionListWithEigenvaluesList[_TRB0, _SB0, _SB1]:
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
    wavepacket = generate_uneven_wavepacket(
        hamiltonian_generator, ExplicitBlockFractionBasis(bloch_fractions), band_basis
    )
    return {
        "basis": VariadicTupleBasis(
            (VariadicTupleBasis((band_basis, list_basis), None), None),
            wavepacket["basis"][1],
        ),
        "data": wavepacket["data"].reshape(-1),
        "eigenvalue": wavepacket["eigenvalue"].reshape(-1),
    }


def get_wavepacket_basis(
    basis: BlochWavefunctionListListBasis[_B0, _SB0, _SB1],
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
    return VariadicTupleBasis((basis[0][1], basis[1]), None)


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
        "basis": get_wavepacket_basis(wavepackets["basis"]),
        "data": wavepackets["data"].reshape(wavepackets["basis"][0][0].n, -1)[idx],
    }


def get_wavepacket_at_band(
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
        "basis": get_wavepacket_basis(wavepackets["basis"]),
        "eigenvalue": wavepackets["eigenvalue"].reshape(
            wavepackets["basis"][0][0].n, -1
        )[idx],
        "data": wavepackets["data"].reshape(wavepackets["basis"][0][0].n, -1)[idx],
    }


def get_wavepackets(
    wavepackets: BlochWavefunctionListList[_B0, _SB0, _SB1],
    idx: slice,
) -> BlochWavefunctionListList[FundamentalBasis[BasisMetadata], _SB0, _SB1]:
    """
    Get the wavepackets at the given slice.

    Parameters
    ----------
    wavepackets : WavepacketList[_B0, _SB0, _SB1]
    idx : slice

    Returns
    -------
    WavepacketList[BasisLike, _SB0, _SB1]
    """
    stacked = wavepackets["data"].reshape(wavepackets["basis"][0][0].n, -1)
    stacked = stacked[idx]
    return {
        "basis": TupleBasis(
            VariadicTupleBasis(
                (FundamentalBasis(len(stacked), None)), wavepackets["basis"][0][1]
            ),
            wavepackets["basis"][1],
        ),
        "data": stacked.reshape(-1),
    }


def get_wavepackets_with_eigenvalues(
    wavepackets: BlochWavefunctionListWithEigenvaluesList[_B0, _SB0, _SB1],
    idx: slice,
) -> BlochWavefunctionListWithEigenvaluesList[
    FundamentalBasis[BasisMetadata], _SB0, _SB1
]:
    """
    Get the wavepackets at the given slice.

    Parameters
    ----------
    wavepackets : WavepacketList[_B0, _SB0, _SB1]
    idx : slice

    Returns
    -------
    WavepacketList[BasisLike, _SB0, _SB1]
    """
    stacked = wavepackets["data"].reshape(wavepackets["basis"][0][0].n, -1)
    stacked = stacked[idx]
    return {
        "basis": TupleBasis(
            VariadicTupleBasis(
                (FundamentalBasis(len(stacked), None)), wavepackets["basis"][0][1]
            ),
            wavepackets["basis"][1],
        ),
        "data": stacked.reshape(-1),
        "eigenvalue": wavepackets["eigenvalue"]
        .reshape(wavepackets["basis"][0][0].n, -1)[idx]
        .ravel(),
    }


def as_wavepacket_list(
    wavepackets: Iterable[BlochWavefunctionList[_SB0, _SB1]],
) -> BlochWavefunctionListList[FundamentalBasis[BasisMetadata], _SB0, _SB1]:
    """
    Convert an iterable of wavepackets into a wavepacket list.

    Parameters
    ----------
    wavepackets : Iterable[Wavepacket[_SB0, _SB1]]

    Returns
    -------
    WavepacketList[FundamentalBasis[BasisMetadata], _SB0, _SB1]
    """
    wavepacket_0 = next(iter(wavepackets))
    vectors = np.array([w["data"] for w in wavepackets])
    return {
        "basis": TupleBasis(
            VariadicTupleBasis(
                (FundamentalBasis(len(vectors), None)), wavepacket_0["basis"][0]
            ),
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
    basis = get_wavepacket_basis(wavepackets["basis"])
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
        "basis": VariadicTupleBasis(
            (averaged["basis"][0][0], averaged["basis"][0][0]), None
        ),
        "data": averaged["data"],
    }
