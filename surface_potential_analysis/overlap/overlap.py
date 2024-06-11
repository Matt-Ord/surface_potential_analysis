from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

from surface_potential_analysis.basis.basis import FundamentalBasis
from surface_potential_analysis.basis.basis_like import BasisLike
from surface_potential_analysis.basis.stacked_basis import TupleBasis
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.operator.operator_list import OperatorList

if TYPE_CHECKING:
    from surface_potential_analysis.types import SingleIndexLike


_B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
_B1 = TypeVar("_B1", bound=BasisLike[Any, Any])
_B2 = TypeVar("_B2", bound=BasisLike[Any, Any])

Overlap = OperatorList[_B0, _B1, _B2]
SingleOverlap = Overlap[
    _B0, BasisLike[Literal[1], Literal[1]], BasisLike[Literal[1], Literal[1]]
]


def get_single_overlap(
    overlap: Overlap[_B0, Any, Any], idx: SingleIndexLike
) -> SingleOverlap[_B0]:
    """
    Select a specific overlap from the overlap operator.

    Parameters
    ----------
    overlap : Overlap[_B0, Any, Any]
    idx : SingleIndexLike

    Returns
    -------
    SingleOverlap[_B0]
    """
    idx = (
        BasisUtil(overlap["basis"][1]).get_flat_index(idx)
        if isinstance(idx, tuple)
        else idx
    )
    return {
        "basis": TupleBasis(
            overlap["basis"][0],
            TupleBasis(
                FundamentalBasis[Literal[1]](1), FundamentalBasis[Literal[1]](1)
            ),
        ),
        "data": overlap["data"].reshape(overlap["basis"].shape)[:, idx],
    }


def get_overlap_cache_filename(
    i: int,
    j: int,
    offset_i: tuple[int, int] = (0, 0),
    offset_j: tuple[int, int] = (0, 0),
) -> str | None:
    """
    Get filename for overlap cache.

    Parameters
    ----------
    i : int
    j : int
    offset_i : tuple[int, int], optional
        offset_i, by default (0, 0)
    offset_j : tuple[int, int], optional
        offset_j, by default (0, 0)

    Returns
    -------
    tuple[int, int, tuple[int, int], tuple[int, int]]
        _description_
    """
    dx0i, dx1i = offset_i
    dx0j, dx1j = offset_j
    too_large_i = not (-2 < dx0i < 2 and -2 < dx1i < 2)  # noqa: PLR2004
    too_large_j = not (-2 < dx0j < 2 and -2 < dx1j < 2)  # noqa: PLR2004
    if too_large_i or too_large_j or i > 1 or j > 1:
        return None
    dx0i, dx1i = dx0i % 3, dx1i % 3
    dx0j, dx1j = dx0j % 3, dx1j % 3
    return f"overlap/overlap_{i}_{dx0i}_{dx1i}_{j}_{dx0j}_{dx1j}.npy"
