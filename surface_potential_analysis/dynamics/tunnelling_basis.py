from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import FundamentalBasis
from surface_potential_analysis.basis.basis_like import AxisVector2d, BasisLike
from surface_potential_analysis.basis.stacked_basis import (
    TupleBasis,
    TupleBasisLike,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.wavepacket.localization._tight_binding import (
    get_wavepacket_two_points,
)
from surface_potential_analysis.wavepacket.wavepacket import wavepacket_list_into_iter

if TYPE_CHECKING:
    from surface_potential_analysis.wavepacket.wavepacket import (
        BlochWavefunctionListWithEigenvaluesList,
    )


_L0_co = TypeVar("_L0_co", bound=int, covariant=True)

_AX0Inv = TypeVar("_AX0Inv", bound=BasisLike[Any, Any])
_AX1Inv = TypeVar("_AX1Inv", bound=BasisLike[Any, Any])


class TunnellingSimulationBandsBasis(FundamentalBasis[_L0_co]):
    """
    Represents the bands axis of the simulation.

    States are localized in their respective unit cells, with
    an approximate central location given by locations as a fraction of
    the unit cell axis vectors.
    """

    def __init__(
        self,
        locations: np.ndarray[tuple[Literal[2], _L0_co], np.dtype[np.float64]],
        unit_cell: tuple[AxisVector2d, AxisVector2d],
    ) -> None:
        self.locations = locations
        self.unit_cell = unit_cell
        super().__init__(self.locations.shape[1])  # type: ignore Argument of type "int" cannot be N0

    @classmethod
    def from_wavepackets(
        cls: type[Self],
        wavepackets: BlochWavefunctionListWithEigenvaluesList[Any, Any, Any],
    ) -> Self:
        """
        Generate a basis given a list of wavepackets.

        Returns
        -------
        Self
        """
        util = BasisUtil(wavepackets["basis"][1])

        locations = np.zeros((2, wavepackets["basis"][0][0].n))
        for i, w in enumerate(wavepacket_list_into_iter(wavepackets)):
            idx0, idx1 = get_wavepacket_two_points(w)
            location = np.average([idx0, idx1], axis=0)
            locations[:, i] = location[0:2]
        return cls(locations, tuple(util.delta_x_stacked[0:2, 0:2]))  # type: ignore[arg-type]


_AX2Inv = TypeVar("_AX2Inv", bound=TunnellingSimulationBandsBasis[Any])

TunnellingSimulationBasis = TupleBasisLike[_AX0Inv, _AX1Inv, _AX2Inv]
"""
Basis used to represent the tunnelling simulation state

First two axes represent the shape of the supercell, last axis is the band
"""

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)

_L3Inv = TypeVar("_L3Inv", bound=int)


def get_basis_from_shape(
    shape: tuple[_L0Inv, _L1Inv],
    n_bands: _L2Inv,
    bands_basis: TunnellingSimulationBandsBasis[_L3Inv],
) -> TupleBasisLike[
    FundamentalBasis[_L0Inv],
    FundamentalBasis[_L1Inv],
    TunnellingSimulationBandsBasis[_L2Inv],
]:
    """
    Get the simulation basis from the shape and TunnellingSimulationBandsBasis.

    Parameters
    ----------
    shape : tuple[_L0Inv, _L1Inv]
    bands_basis : TunnellingSimulationBandsBasis[_L2Inv]

    Returns
    -------
    tuple[FundamentalBasis[_L0Inv], FundamentalBasis[_L1Inv], TunnellingSimulationBandsBasis[_L2Inv]]
    """
    return TupleBasis(
        FundamentalBasis(shape[0]),
        FundamentalBasis(shape[1]),
        TunnellingSimulationBandsBasis(
            bands_basis.locations[:, :n_bands], bands_basis.unit_cell
        ),
    )
