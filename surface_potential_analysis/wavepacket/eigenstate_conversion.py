from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np

from surface_potential_analysis.basis.stacked_basis import (
    TupleBasis,
    TupleBasisLike,
    TupleBasisWithLengthLike,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.stacked_basis.build import (
    fundamental_stacked_basis_from_shape,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_basis,
    stacked_basis_as_fundamental_momentum_basis,
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
    get_furled_basis,
    get_unfurled_basis,
    wavepacket_list_into_iter,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        FundamentalBasis,
        FundamentalTransformedPositionBasis,
    )
    from surface_potential_analysis.basis.basis_like import (
        BasisLike,
        BasisWithLengthLike3d,
    )
    from surface_potential_analysis.state_vector.state_vector import (
        StateVector,
    )
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )

    _NS0Inv = TypeVar("_NS0Inv", bound=int)
    _NS1Inv = TypeVar("_NS1Inv", bound=int)

    _A3d2Inv = TypeVar("_A3d2Inv", bound=BasisWithLengthLike3d[Any, Any])

    _SB1 = TypeVar("_SB1", bound=TupleBasisWithLengthLike[*tuple[Any, ...]])
    _MB0 = TypeVar("_MB0", bound=FundamentalTransformedPositionBasis[Any, Any])
    _SB0 = TypeVar("_SB0", bound=TupleBasisLike[*tuple[Any, ...]])
    _FB0 = TypeVar("_FB0", bound=FundamentalBasis[Any])
    _B0 = TypeVar("_B0", bound=BasisLike[Any, Any])

    _L0Inv = TypeVar("_L0Inv", bound=int)
    _L1Inv = TypeVar("_L1Inv", bound=int)


def furl_eigenstate(
    eigenstate: StateVector[
        TupleBasisLike[
            FundamentalTransformedPositionBasis[_L0Inv, Literal[3]],
            FundamentalTransformedPositionBasis[_L1Inv, Literal[3]],
            _A3d2Inv,
        ]
    ],
    shape: tuple[_NS0Inv, _NS1Inv, Literal[1]],
) -> BlochWavefunctionList[
    TupleBasisLike[
        FundamentalBasis[_NS0Inv],
        FundamentalBasis[_NS1Inv],
        FundamentalBasis[Literal[1]],
    ],
    TupleBasisLike[
        FundamentalTransformedPositionBasis[_L0Inv, Literal[3]],
        FundamentalTransformedPositionBasis[_L1Inv, Literal[3]],
        _A3d2Inv,
    ],
]:
    """
    Convert an eigenstate into a wavepacket of a smaller unit cell.

    Parameters
    ----------
    eigenstate : Eigenstate[MomentumBasis[_L0], MomentumBasis[_L1], _BX2]
        The eigenstate of the larger unit cell.
    shape : tuple[_NS0, _NS1]
        The shape of samples in the wavepacket grid.
        Note _NS0 must be a factor of _L0

    Returns
    -------
    Wavepacket[_NS0, _NS1, MomentumBasis[_L0 // _NS0], MomentumBasis[_L1 // _NS1], _BX2]
        The wavepacket with a smaller unit cell
    """
    (ns0, ns1, _) = shape
    (nx0_old, nx1_old, nx2) = BasisUtil(eigenstate["basis"]).shape
    (nx0, nx1) = (nx0_old // ns0, nx1_old // ns1)

    # We do the opposite to unfurl wavepacket at each step
    stacked = eigenstate["data"].reshape(nx0 * ns0, nx1 * ns1, nx2)
    shifted = np.fft.fftshift(stacked, (0, 1))
    double_stacked = shifted.reshape(nx0, ns0, nx1, ns1, nx2)
    swapped = double_stacked.swapaxes(2, 3).swapaxes(0, 1).swapaxes(1, 2)
    unshifted = np.fft.ifftshift(swapped, axes=(0, 1, 2, 3))
    flattened = unshifted.reshape(ns0, ns1, -1)

    basis = get_furled_basis(eigenstate["basis"], shape)
    return {
        "basis": TupleBasis(
            fundamental_stacked_basis_from_shape(shape),
            stacked_basis_as_fundamental_momentum_basis(basis),  # type: ignore z ax is wrong here
        ),
        "data": flattened * np.sqrt(ns0 * ns1),
        "eigenvalue": np.zeros(flattened.shape[0:2]),
    }


def _unfurl_momentum_basis_wavepacket(
    wavepacket: BlochWavefunctionList[
        TupleBasisLike[*tuple[_FB0, ...]], TupleBasisWithLengthLike[*tuple[_MB0, ...]]
    ],
) -> StateVector[
    TupleBasisLike[*tuple[FundamentalTransformedPositionBasis[Any, Any], ...]]
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

    basis = get_unfurled_basis(wavepacket["basis"])
    return {
        "basis": stacked_basis_as_fundamental_momentum_basis(basis),
        "data": flattened / np.sqrt(np.prod(list_shape)),
    }


def unfurl_wavepacket(
    wavepacket: BlochWavefunctionList[_SB0, _SB1],
) -> StateVector[
    TupleBasisLike[*tuple[FundamentalTransformedPositionBasis[Any, Any], ...]]
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
        list_basis=stacked_basis_as_fundamental_basis(wavepacket["basis"][0]),
    )
    # TDOO:! np.testing.assert_array_equal(converted["data"], wavepacket["data"])
    return _unfurl_momentum_basis_wavepacket(converted)


def unfurl_wavepacket_list(
    wavepackets: BlochWavefunctionListList[_B0, _SB0, _SB1],
) -> StateVectorList[
    _B0, TupleBasisLike[*tuple[FundamentalTransformedPositionBasis[Any, Any], ...]]
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
        "basis": TupleBasis(wavepackets["basis"][0][0], unfurled["basis"][1]),
        "data": unfurled["data"],
    }
