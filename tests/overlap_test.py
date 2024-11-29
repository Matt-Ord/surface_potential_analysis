from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from surface_potential_analysis.basis.legacy import FundamentalBasis, StackedBasis
from surface_potential_analysis.basis.util import (
    BasisUtil,
)
from surface_potential_analysis.overlap.conversion import (
    convert_overlap_to_momentum_basis,
)
from surface_potential_analysis.overlap.interpolation import (
    get_overlap_momentum_interpolator,
    get_overlap_momentum_interpolator_k_fractions,
)
from surface_potential_analysis.stacked_basis.build import position_basis_3d_from_shape

if TYPE_CHECKING:
    from surface_potential_analysis.overlap.overlap import SingleOverlap

rng = np.random.default_rng()


class OverlapTest(unittest.TestCase):
    def test_overlap_interpolation(self) -> None:
        shape = (10, 10, 10)
        overlap: SingleOverlap[Any] = {
            "basis": StackedBasis(
                position_basis_3d_from_shape(shape),
                StackedBasis(
                    FundamentalBasis[Literal[1]](1), FundamentalBasis[Literal[1]](1)
                ),
            ),
            "data": np.array(rng.random(np.prod(shape)), dtype=complex),
        }

        util = BasisUtil(overlap["basis"][0])
        overlap_momentum = convert_overlap_to_momentum_basis(overlap)
        expected = overlap_momentum["data"]
        actual = get_overlap_momentum_interpolator(overlap)(util.k_points)  # type: ignore[var-annotated,arg-type]

        np.testing.assert_array_almost_equal(expected, actual)

    def test_overlap_interpolation_fractions(self) -> None:
        shape = (10, 10, 10)
        overlap: SingleOverlap[Any] = {
            "basis": StackedBasis(
                position_basis_3d_from_shape(shape),
                StackedBasis(
                    FundamentalBasis[Literal[1]](1), FundamentalBasis[Literal[1]](1)
                ),
            ),
            "data": np.array(rng.random(np.prod(shape)), dtype=complex),
        }

        util = BasisUtil(overlap["basis"])
        overlap_momentum = convert_overlap_to_momentum_basis(overlap)
        expected = overlap_momentum["data"]
        actual = get_overlap_momentum_interpolator_k_fractions(overlap)(
            util.stacked_nk_points
        )  # type: ignore[var-annotated,arg-type]

        np.testing.assert_array_almost_equal(expected, actual)
