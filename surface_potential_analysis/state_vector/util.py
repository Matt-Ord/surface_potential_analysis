from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from surface_potential_analysis.basis.legacy import (
    FundamentalBasis,
    FundamentalPositionBasis,
    FundamentalTransformedPositionBasis,
    StackedBasisLike,
    StackedBasisWithVolumeLike,
    TransformedPositionBasis,
    TupleBasis,
    TupleBasisWithLengthLike,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.probability_vector.probability_vector import (
    average_probabilities,
    from_state_vector_list,
)
from surface_potential_analysis.stacked_basis.conversion import (
    tuple_basis_as_fundamental,
)
from surface_potential_analysis.wavepacket.eigenstate_conversion import (
    unfurl_wavepacket_list,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    BlochWavefunctionListList,
    get_fundamental_unfurled_basis,
    get_wavepackets,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.legacy import (
        BasisLike,
        TupleBasisLike,
    )
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )
    from surface_potential_analysis.types import (
        IntLike_co,
        SingleFlatIndexLike,
    )
    from surface_potential_analysis.wavepacket.wavepacket import (
        BlochWavefunctionListBasis,
    )

    _B0 = TypeVar("_B0", bound=BasisLike)
    _SBV0 = TypeVar("_SBV0", bound=StackedBasisWithVolumeLike)
    _SB0 = TypeVar("_SB0", bound=StackedBasisLike)
    _TBL0 = TypeVar("_TBL0", bound=TupleBasisWithLengthLike[*tuple[Any, ...]])
    _L0Inv = TypeVar("_L0Inv", bound=int)


def get_single_point_state_vector_excact(
    basis: _B0, idx: SingleFlatIndexLike
) -> StateVector[_B0]:
    """Get the state which is nonzero at idx."""
    data = np.zeros(basis.n, dtype=np.complex128)
    data[idx] = 1
    return {"basis": basis, "data": data}


def get_single_point_state_vectors(
    basis: BlochWavefunctionListBasis[_SB0, _SBV0],
    n_bands: _L0Inv,
) -> StateVectorList[
    FundamentalBasis[_L0Inv],
    TupleBasisLike[*tuple[FundamentalPositionBasis, ...]],
]:
    """Get the state which are nonzero at idx."""
    converted = tuple_basis_as_fundamental(
        get_fundamental_unfurled_basis(basis)
    )
    data = np.zeros((n_bands, converted.n), dtype=np.complex128)
    for i, n in enumerate(range(0, basis[1].n, n_bands)):
        data[i, n] = 1
    return {
        "basis": VariadicTupleBasis((FundamentalBasis(n_bands), None), converted),
        "data": data.reshape(-1),
    }


def get_most_localized_free_state_vectors(
    basis: BlochWavefunctionListBasis[_SB0, _TBL0],
    shape: tuple[IntLike_co, ...],
) -> StateVectorList[
    TupleBasisLike[*tuple[FundamentalBasis[BasisMetadata], ...]],
    TupleBasisWithLengthLike[*tuple[TransformedPositionBasis[Any, Any, Any], ...]],
]:
    """
    Get the most localized free states on the surface.

    A reasonable choice for the initial wavefunctions are the 'most localized'
    states we would find if we assume the potential is zero. In this case
    the states are evenly occupied upto some threshold frequency.

    Returns
    -------
    StateVectorList[StackedBasis[*tuple[FundamentalBasis[BasisMetadata], ...]], StackedBasis[Any]]
        The most localized states
    """
    n_bands = np.prod(np.asarray(shape))
    # TODO: properly deal with uneven sampled basis
    sample_basis = TupleBasis(
        *tuple(
            TransformedPositionBasis(
                basis[0].fundamental_shape[i] * basis[1].delta_x_stacked[i],
                # TODO: is this correct threshold k when not in 1D,
                # or do we need a bigger or smaller width than n_bands?
                # best in Cu when multiplying by 3
                basis[0].fundamental_shape[i] * s,
                basis[0].fundamental_shape[i] * basis[1].fundamental_shape[i],
            )
            for (i, s) in enumerate(shape)
        )
    )
    bands_basis = VariadicTupleBasis((*tuple(FundamentalBasis(int(n), None)) for n in shape))
    bands_util = BasisUtil(bands_basis)
    sample_fractions = BasisUtil(sample_basis).stacked_nx_points
    sample_fractions = tuple(
        f / n for (f, n) in zip(sample_fractions, sample_basis.shape, strict=True)
    )

    data = np.zeros((n_bands, sample_basis.n))
    data = np.exp(
        (-2j * np.pi)
        * np.tensordot(
            bands_util.stacked_nk_points,
            sample_fractions,
            axes=(0, 0),
        )
    )
    data /= np.sqrt(np.sum(np.abs(data) ** 2, axis=1))[:, np.newaxis]
    return {"basis": VariadicTupleBasis((bands_basis, sample_basis), None), "data": data.reshape(-1)}


def get_most_localized_state_vectors_from_probability(
    wavepackets: BlochWavefunctionListList[_B0, _SB0, _SBV0],
    fractions: tuple[np.ndarray[tuple[int], np.dtype[np.float64]], ...],
) -> StateVectorList[
    FundamentalBasis[BasisMetadata],
    TupleBasisLike[*tuple[FundamentalTransformedPositionBasis, ...]],
]:
    """
    Get the most localized free states on the surface.

    A reasonable choice for the initial wavefunctions are the 'most localized'
    states we would find if we assume the potential is zero. In this case
    the states are evenly occupied upto some threshold frequency.

    Returns
    -------
    StateVectorList[StackedBasis[*tuple[FundamentalBasis[BasisMetadata], ...]], StackedBasis[Any]]
        The most localized states
    """
    n_bands = fractions[0].size
    unfurled = unfurl_wavepacket_list(get_wavepackets(wavepackets, slice(n_bands)))
    probabilities = from_state_vector_list(unfurled)
    averaged = average_probabilities(probabilities)

    sample_basis = unfurled["basis"][1]
    sample_fractions = BasisUtil(unfurled["basis"][1]).stacked_nx_points
    sample_fractions = tuple(
        f / n
        for (f, n) in zip(
            sample_fractions, wavepackets["basis"][0][1].shape, strict=True
        )
    )

    data = np.zeros((n_bands, sample_basis.n))
    data = np.exp(
        (-2j * np.pi) * np.tensordot(fractions, sample_fractions, axes=(0, 0))
    )
    data *= np.sqrt(averaged["data"])
    data /= np.sqrt(np.sum(np.abs(data) ** 2, axis=1))[:, np.newaxis]
    return {
        "basis": VariadicTupleBasis((FundamentalBasis(n_bands), None), sample_basis),
        "data": data.reshape(-1),
    }
