from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

import numpy as np

from surface_potential_analysis.basis.basis import FundamentalBasis
from surface_potential_analysis.basis.basis_like import BasisLike, BasisWithLengthLike
from surface_potential_analysis.basis.stacked_basis import (
    TupleBasis,
    TupleBasisLike,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_position_basis,
)
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_list_to_basis,
    convert_state_vector_to_position_basis,
)
from surface_potential_analysis.util.decorators import timed
from surface_potential_analysis.wavepacket.eigenstate_conversion import (
    unfurl_wavepacket,
    unfurl_wavepacket_list,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import FundamentalPositionBasis
    from surface_potential_analysis.overlap.overlap import Overlap, SingleOverlap
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )
    from surface_potential_analysis.types import SingleIndexLike
    from surface_potential_analysis.wavepacket.wavepacket import (
        BlochWavefunctionList,
        BlochWavefunctionListList,
    )


_B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
_BL0 = TypeVar("_BL0", bound=BasisWithLengthLike[Any, Any, Any])
_B1 = TypeVar("_B1", bound=BasisLike[Any, Any])
_BL1 = TypeVar("_BL1", bound=BasisWithLengthLike[Any, Any, Any])
_SB0 = TypeVar("_SB0", bound=TupleBasisLike[*tuple[Any, ...]])


@timed
def calculate_wavepacket_overlap(
    wavepacket_0: BlochWavefunctionList[
        TupleBasisLike[*tuple[_B0, ...]], TupleBasisLike[*tuple[_BL0, ...]]
    ],
    wavepacket_1: BlochWavefunctionList[
        TupleBasisLike[*tuple[_B1, ...]], TupleBasisLike[*tuple[_BL1, ...]]
    ],
) -> SingleOverlap[TupleBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]]:
    """
    Given two wavepackets in (the same) momentum basis calculate the overlap factor.

    Parameters
    ----------
    wavepacket_0 : Wavepacket[_NS0Inv, _NS1Inv, TupleBasisLike[tuple[MomentumBasis[_L0Inv], MomentumBasis[_L1Inv], _A3d0Inv]]
    wavepacket_1 : Wavepacket[_NS0Inv, _NS1Inv, TupleBasisLike[tuple[MomentumBasis[_L0Inv], MomentumBasis[_L1Inv], _A3d0Inv]]

    Returns
    -------
    Overlap[TupleBasisLike[tuple[PositionBasis[int], PositionBasis[int], _A3d0Inv]]
    """
    eigenstate_0 = convert_state_vector_to_position_basis(
        unfurl_wavepacket(wavepacket_0)
    )
    eigenstate_1 = convert_state_vector_to_position_basis(
        unfurl_wavepacket(wavepacket_1)
    )

    vector = np.conj(eigenstate_0["data"]) * (eigenstate_1["data"])
    return {
        "basis": TupleBasis(
            eigenstate_0["basis"],
            TupleBasis(
                FundamentalBasis[Literal[1]](1), FundamentalBasis[Literal[1]](1)
            ),
        ),
        "data": vector,
    }


def calculate_state_vector_list_overlap(
    states: StateVectorList[_B1, _SB0],
    *,
    shift: SingleIndexLike = 0,
) -> Overlap[_SB0, _B1, _B1]:
    """
    Given a state vector list, calculate the overlap.

    Parameters
    ----------
    StateVectorList[_B1,TupleBasisLike[*tuple[_BL0, ...]]]

    Returns
    -------
    Overlap[TupleBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]], _B1, _B1].
    """
    stacked = states["data"].reshape(states["basis"].shape)
    shift = (
        BasisUtil(states["basis"][1]).get_flat_index(shift, mode="wrap")
        if isinstance(shift, tuple)
        else shift
    )
    stacked_shifted = np.roll(stacked, shift, axis=(1))
    # stacked = i, j where i indexes the state and j indexes the position
    data = np.conj(stacked)[np.newaxis, :, :] * (stacked_shifted[:, np.newaxis, :])
    return {
        "basis": TupleBasis(
            states["basis"][1],
            TupleBasis(states["basis"][0], states["basis"][0]),
        ),
        "data": data.swapaxes(-1, 0).ravel(),
    }


@overload
def calculate_wavepacket_list_overlap(
    wavepackets: BlochWavefunctionListList[
        _B1, TupleBasisLike[*tuple[_B0, ...]], TupleBasisLike[*tuple[_BL0, ...]]
    ],
    *,
    shift: SingleIndexLike = 0,
    basis: _SB0,
) -> Overlap[_SB0, _B1, _B1]:
    ...


@overload
def calculate_wavepacket_list_overlap(
    wavepackets: BlochWavefunctionListList[
        _B1, TupleBasisLike[*tuple[_B0, ...]], TupleBasisLike[*tuple[_BL0, ...]]
    ],
    *,
    shift: SingleIndexLike = 0,
    basis: None = None,
) -> Overlap[TupleBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]], _B1, _B1]:
    ...


@timed
def calculate_wavepacket_list_overlap(
    wavepackets: BlochWavefunctionListList[
        _B1, TupleBasisLike[*tuple[_B0, ...]], TupleBasisLike[*tuple[_BL0, ...]]
    ],
    *,
    shift: SingleIndexLike = 0,
    basis: _SB0
    | TupleBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]
    | None = None,
) -> Overlap[
    _SB0 | TupleBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]], _B1, _B1
]:
    """
    Given two wavepackets in (the same) momentum basis calculate the overlap factor.

    Parameters
    ----------
    wavepacket_0 : Wavepacket[_NS0Inv, _NS1Inv, TupleBasisLike[tuple[MomentumBasis[_L0Inv], MomentumBasis[_L1Inv], _A3d0Inv]]
    wavepacket_1 : Wavepacket[_NS0Inv, _NS1Inv, TupleBasisLike[tuple[MomentumBasis[_L0Inv], MomentumBasis[_L1Inv], _A3d0Inv]]

    Returns
    -------
    Overlap[TupleBasisLike[tuple[PositionBasis[int], PositionBasis[int], _A3d0Inv]]
    """
    states = unfurl_wavepacket_list(wavepackets)
    basis = (
        stacked_basis_as_fundamental_position_basis(states["basis"][1])
        if basis is None
        else basis
    )
    converted = convert_state_vector_list_to_basis(states, basis)
    return calculate_state_vector_list_overlap(converted, shift=shift)
