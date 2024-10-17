from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from surface_potential_analysis.basis.legacy import (
    StackedBasisLike,
    StackedBasisWithVolumeLike,
    TupleBasisLike,
    TupleBasisWithLengthLike,
)
from surface_potential_analysis.stacked_basis.conversion import (
    tuple_basis_as_transformed_fundamental,
)
from surface_potential_analysis.state_vector.state_vector_list import (
    as_state_vector_list,
)
from surface_potential_analysis.wavepacket.conversion import (
    convert_wavepacket_to_fundamental_momentum_basis,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    BlochWavefunctionList,
    BlochWavefunctionListList,
    get_fundamental_unfurled_basis,
    wavepacket_list_into_iter,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.legacy import (
        BasisLike,
        FundamentalTransformedPositionBasis,
    )
    from surface_potential_analysis.state_vector.state_vector import (
        StateVector,
    )
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )

    _SBV0 = TypeVar("_SBV0", bound=StackedBasisWithVolumeLike)

    _SB0 = TypeVar("_SB0", bound=StackedBasisLike)
    _B0 = TypeVar("_B0", bound=BasisLike)


def _unfurl_momentum_basis_wavepacket(
    wavepacket: BlochWavefunctionList[
        TupleBasisLike[*tuple[Any, ...]], TupleBasisWithLengthLike[*tuple[Any, ...]]
    ],
) -> StateVector[
    TupleBasisWithLengthLike[*tuple[FundamentalTransformedPositionBasis, ...]]
]:
    list_shape = wavepacket["basis"][0].shape
    states_shape = wavepacket["basis"][1].shape
    final_shape = tuple(
        ns * nx for (ns, nx) in zip(list_shape, states_shape, strict=True)
    )
    stacked = wavepacket["data"].reshape(*list_shape, *states_shape)

    # Shift negative frequency componets to the start, so we can
    # add the frequencies when we unravel
    shifted = np.fft.fftshift(stacked)
    # The list of axis index n,0,n+1,1,...,2n-1,n-1
    nd = len(list_shape)
    locations = [x for y in zip(range(nd, 2 * nd), range(nd), strict=True) for x in y]
    # We now have nx0, ns0, nx1, ns1, ns2, ...
    swapped = np.transpose(shifted, axes=locations)
    # Ravel the samples into the eigenstates, since they have fractional frequencies
    # when we ravel the x0 and x1 axis we retain the order of the frequencies
    ravelled = swapped.reshape(*final_shape)
    # Shift the 0 frequency back to the start and flatten
    # Note the ifftshift would shift by (list_shape[i] * states_shape[i]) // 2
    # Which is wrong in this case
    shift = tuple(
        -(list_shape[i] // 2 + (list_shape[i] * (states_shape[i] // 2)))
        for i in range(nd)
    )
    unshifted = np.roll(ravelled, shift, tuple(range(nd)))
    flattened = unshifted.reshape(-1)

    basis = get_fundamental_unfurled_basis(wavepacket["basis"])
    return {
        "basis": stacked_basis_as_transformed_basis(basis),
        "data": flattened / np.sqrt(np.prod(list_shape)),
    }


def unfurl_wavepacket(
    wavepacket: BlochWavefunctionList[_SB0, _SBV0],
) -> StateVector[
    TupleBasisWithLengthLike[*tuple[FundamentalTransformedPositionBasis, ...]]
]:
    """
    Convert a wavepacket into an eigenstate of the irreducible unit cell.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0, _NS1, MomentumBasis[_L0], MomentumBasis[_L1], _BX2]
        The wavepacket to unfurl

    Returns
    -------
    Eigenstate[MomentumBasis[_NS0 * _L0], MomentumBasis[_NS1 * _L1], _BX2]
        The eigenstate of the larger unit cell. Note this eigenstate has a
        smaller dk (for each axis dk = dk_i / NS)
    """
    converted = convert_wavepacket_to_fundamental_momentum_basis(
        wavepacket,
        list_basis=tuple_basis_as_transformed_fundamental(wavepacket["basis"][0]),
    )
    # TDOO:! np.testing.assert_array_equal(converted["data"], wavepacket["data"])
    return _unfurl_momentum_basis_wavepacket(converted)


def unfurl_wavepacket_list(
    wavepackets: BlochWavefunctionListList[_B0, _SB0, _SBV0],
) -> StateVectorList[
    _B0,
    TupleBasisWithLengthLike[*tuple[FundamentalTransformedPositionBasis, ...]],
]:
    """
    Convert a wavepacket list into a StateVectorList.

    This produces a set of fundamental states, which are (usually, but not guaranteed)
    to be located at the origin

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0, _NS1, MomentumBasis[_L0], MomentumBasis[_L1], _BX2]
        The wavepacket to unfurl

    Returns
    -------
    Eigenstate[MomentumBasis[_NS0 * _L0], MomentumBasis[_NS1 * _L1], _BX2]
        The eigenstate of the larger unit cell. Note this eigenstate has a
        smaller dk (for each axis dk = dk_i / NS)
    """
    unfurled = as_state_vector_list(
        unfurl_wavepacket(w) for w in wavepacket_list_into_iter(wavepackets)
    )
    return {
        "basis": VariadicTupleBasis(
            (wavepackets["basis"][0][0], unfurled["basis"][1]), None
        ),
        "data": unfurled["data"],
    }
