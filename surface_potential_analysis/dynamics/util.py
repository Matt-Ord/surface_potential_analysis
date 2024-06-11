from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from surface_potential_analysis.basis.basis import FundamentalBasis
from surface_potential_analysis.basis.stacked_basis import TupleBasis
from surface_potential_analysis.basis.util import BasisUtil

if TYPE_CHECKING:
    from surface_potential_analysis.types import IntLike_co


def get_hop_shift(hop: int, ndim: int) -> tuple[int, ...]:
    """
    Given a hop index in ndim, get the hop shift vector.

    Parameters
    ----------
    hop : int
    ndim : int

    Returns
    -------
    tuple[int, ...]
    """
    util = BasisUtil(TupleBasis(*tuple(FundamentalBasis(3) for _ in range(ndim))))
    return tuple(x.item(hop) for x in util.stacked_nk_points)


def build_hop_operator(
    hop: int, shape: tuple[IntLike_co, ...]
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Given a hop index, build a hop operator in the given shape.

    Parameters
    ----------
    hop : int
        hop index
    shape : _S0Inv
        shape

    Returns
    -------
    np.ndarray[tuple[Unpack[_S0Inv], Unpack[_S0Inv]], np.dtype[np.real_]]
    """
    hop_shift = get_hop_shift(hop, len(shape))
    operator = np.identity(cast(int, np.prod(shape))).reshape(*shape, *shape)  # type: ignore shape not array like
    return np.roll(operator, hop_shift, tuple(range(len(shape))))  # type: ignore[no-any-return]
