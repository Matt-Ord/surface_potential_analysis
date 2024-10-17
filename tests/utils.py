from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import special_ortho_group

from surface_potential_analysis.basis.legacy import (
    ExplicitBasisWithLength,
    FundamentalBasis,
    FundamentalPositionBasis,
    StackedBasis,
)

rng = np.random.default_rng()


def get_random_explicit_basis(
    nd: int,
    fundamental_n: int | None = None,
    n: int | None = None,
) -> ExplicitBasisWithLength[FundamentalBasis[Any], FundamentalPositionBasis]:
    fundamental_n = (
        rng.integers(2 if n is None else n, 5)  # type: ignore bad libary types
        if fundamental_n is None
        else fundamental_n
    )
    n = rng.integers(1, fundamental_n) if n is None else n  # type: ignore bad libary types
    vectors = special_ortho_group.rvs(fundamental_n)[:n].astype(np.complex128)
    delta_x = np.zeros(nd)
    delta_x[0] = 1
    return ExplicitBasisWithLength(
        {
            "basis": StackedBasis(
                FundamentalBasis(n),
                FundamentalPositionBasis(delta_x, fundamental_n),
            ),
            "data": vectors,
        }
    )
