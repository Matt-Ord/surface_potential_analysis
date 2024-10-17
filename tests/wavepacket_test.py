from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from surface_potential_analysis.basis.legacy import (
    FundamentalBasis,
    FundamentalTransformedPositionBasis,
    StackedBasis,
)
from surface_potential_analysis.stacked_basis.build import (
    fundamental_stacked_basis_from_shape,
    momentum_basis_3d_from_resolution,
    position_basis_3d_from_shape,
)
from surface_potential_analysis.wavepacket.eigenstate_conversion import (
    _unfurl_momentum_basis_wavepacket,  # type: ignore this is test file
    furl_eigenstate,
    unfurl_wavepacket,
)
from surface_potential_analysis.wavepacket.localization._tight_binding import (
    _get_global_phases,  # type: ignore this is test file
)
from surface_potential_analysis.wavepacket.localization._wannier90 import (
    _parse_nnk_points_file,  # type: ignore this is test file
)
from surface_potential_analysis.wavepacket.wavepacket import (
    BlochWavefunctionList,
    get_wavepacket_sample_fractions,
)

rng = np.random.default_rng()


class WavepacketTest(unittest.TestCase):
    def test_get_global_phases(self) -> None:
        ns0 = rng.integers(1, 10)  # type: ignore bad libary types
        ns1 = rng.integers(1, 10)  # type: ignore bad libary types
        resolution = (
            rng.integers(2, 10),  # type: ignore bad libary types
            rng.integers(1, 10),  # type: ignore bad libary types
            rng.integers(1, 10),  # type: ignore bad libary types
        )
        wavepacket: BlochWavefunctionList[Any, Any] = {
            "basis": StackedBasis(
                fundamental_stacked_basis_from_shape((ns0, ns1, 1)),
                position_basis_3d_from_shape(resolution),
            ),
            "data": np.zeros((ns0 * ns1, np.prod(resolution))).reshape(-1),
        }

        idx = rng.integers(0, np.prod(resolution).item())  # type: ignore bad libary types
        actual = _get_global_phases(wavepacket, idx)
        np.testing.assert_array_equal(actual.shape, (ns0 * ns1,))
        np.testing.assert_equal(actual[0], 0)

        idx = 0
        actual = _get_global_phases(wavepacket, idx)
        np.testing.assert_array_equal(actual, np.zeros_like(actual))

        idx_array = rng.integers(0, np.prod(resolution).item(), size=(10, 10, 11))  # type: ignore bad libary types
        actual_large = _get_global_phases(wavepacket, idx_array)
        np.testing.assert_array_equal(actual_large.shape, (ns0 * ns1, *idx_array.shape))
        np.testing.assert_equal(actual_large[0], 0)
        np.testing.assert_equal(actual_large[:, idx_array == 0], 0)

    def test_unfurl_wavepacket(self) -> None:
        wavepacket: BlochWavefunctionList[Any, Any] = {
            "basis": StackedBasis(
                fundamental_stacked_basis_from_shape((3, 2, 1)),
                momentum_basis_3d_from_resolution((3, 3, 3)),
            ),
            "data": np.zeros((3, 2, 27)).reshape(-1),
        }

        wavepacket["data"][0][0][0] = 1
        wavepacket["data"][1][0][0] = 2
        wavepacket["data"][2][0][0] = 3
        wavepacket["data"][0][1][0] = 4
        wavepacket["data"][1][1][0] = 5
        wavepacket["data"][2][1][0] = 6

        expected = np.zeros(162)
        expected[np.ravel_multi_index((0, 0, 0), (9, 6, 3))] = 1
        expected[np.ravel_multi_index((1, 0, 0), (9, 6, 3))] = 2
        expected[np.ravel_multi_index((8, 0, 0), (9, 6, 3))] = 3
        expected[np.ravel_multi_index((0, 5, 0), (9, 6, 3))] = 4
        expected[np.ravel_multi_index((1, 5, 0), (9, 6, 3))] = 5
        expected[np.ravel_multi_index((8, 5, 0), (9, 6, 3))] = 6

        eigenstate = unfurl_wavepacket(wavepacket)
        np.testing.assert_array_equal(eigenstate["data"], expected / np.sqrt(2 * 3))

    def test_furl_eigenstate(self) -> None:
        wavepacket: BlochWavefunctionList[Any, Any] = {
            "basis": StackedBasis(
                fundamental_stacked_basis_from_shape((3, 2, 1)),
                momentum_basis_3d_from_resolution((3, 3, 3)),
            ),
            "data": np.array(rng.random((3, 2, 27)), dtype=complex),
        }
        eigenstate = unfurl_wavepacket(wavepacket)
        actual = furl_eigenstate(eigenstate, (3, 2, 1))  # type: ignore[arg-type,var-annotated]

        np.testing.assert_array_almost_equal(wavepacket["data"], actual["data"])
        np.testing.assert_array_almost_equal(
            wavepacket["basis"][1][0].delta_x, actual["basis"][1][0].delta_x
        )
        np.testing.assert_array_almost_equal(
            wavepacket["basis"][1][1].delta_x, actual["basis"][1][1].delta_x
        )
        np.testing.assert_array_almost_equal(
            wavepacket["basis"][1][2].delta_x,
            actual["basis"][1][2].delta_x,  # type: ignore bad inference
        )

    def test_get_wavepacket_sample_fractions(self) -> None:
        shape = tuple(rng.integers(1, 10, size=rng.integers(1, 5)))  # type: ignore bad libary types

        actual = get_wavepacket_sample_fractions(
            fundamental_stacked_basis_from_shape(shape)
        )
        meshgrid = np.meshgrid(
            *[np.fft.fftfreq(s, 1) for s in shape],
            indexing="ij",
        )
        expected = np.array([x.ravel() for x in meshgrid])
        np.testing.assert_array_almost_equal(expected, actual)

    # ! cSpell:disable
    def test_parse_nnkpts_block(self) -> None:
        block = """
File written on 18Aug2023 at 07:15:55

calc_only_A  :  F

begin real_lattice
   2.4890159   0.0000000   0.0000000
   1.2445079   2.1555510   0.0000000
   0.0000000   0.0000000   2.2000000
end real_lattice

begin recip_lattice
   2.5243653  -1.4574430   0.0000000
   0.0000000   2.9148860   0.0000000
   0.0000000   0.0000000   2.8559933
end recip_lattice

begin projections
     0
end projections

begin auto_projections
     1
     0
end auto_projections

begin nnkpts
   8
     1     2      0   0   0
     1    12      0   0   0
     1    13      0   0   0
     1    14      0   0   0
     1   133      0   0   0
     1   144      0   0   0
     1     1      0   0   1
     1     1      0   0  -1
     2     1      0   0   0
     2     3      0   0   0
     2    14      0   0   0
     2    15      0   0   0
     2   133      0   0   0
     2   134      0   0   0
     2     2      0   0   1
     2     2      0   0  -1
     3     2      0   0   0
     3     4      0   0   0
     3    15      0   0   0
     3    16      0   0   0
     3   134      0   0   0
     3   135      0   0   0
     3     3      0   0   1
     3     3      0   0  -1
   144   133      0   0   0
   144   143      0   0   0
   144   144      0   0   1
   144   144      0   0  -1
end nnkpts
        """
        expected = [
            (1, 2, 0, 0, 0),
            (1, 12, 0, 0, 0),
            (1, 13, 0, 0, 0),
            (1, 14, 0, 0, 0),
            (1, 133, 0, 0, 0),
            (1, 144, 0, 0, 0),
            (1, 1, 0, 0, 1),
            (1, 1, 0, 0, -1),
            (2, 1, 0, 0, 0),
            (2, 3, 0, 0, 0),
            (2, 14, 0, 0, 0),
            (2, 15, 0, 0, 0),
            (2, 133, 0, 0, 0),
            (2, 134, 0, 0, 0),
            (2, 2, 0, 0, 1),
            (2, 2, 0, 0, -1),
            (3, 2, 0, 0, 0),
            (3, 4, 0, 0, 0),
            (3, 15, 0, 0, 0),
            (3, 16, 0, 0, 0),
            (3, 134, 0, 0, 0),
            (3, 135, 0, 0, 0),
            (3, 3, 0, 0, 1),
            (3, 3, 0, 0, -1),
            (144, 133, 0, 0, 0),
            (144, 143, 0, 0, 0),
            (144, 144, 0, 0, 1),
            (144, 144, 0, 0, -1),
        ]
        _, actual = _parse_nnk_points_file(block)
        np.testing.assert_array_equal(expected, actual)

    def test_unfurl_momentum_basis_wavepacket(self) -> None:
        ns = rng.integers(1, 5)  # type: ignore bad libary types
        nf = rng.integers(1, 5)  # type: ignore bad libary types
        vectors = np.zeros((ns, nf), dtype=np.complex128)
        vectors[0, 0] = 1
        actual = _unfurl_momentum_basis_wavepacket(
            {
                "basis": StackedBasis(
                    StackedBasis(FundamentalBasis(ns)),
                    StackedBasis(
                        FundamentalTransformedPositionBasis(np.array([2]), nf)
                    ),
                ),
                "data": vectors,
            }
        )
        expected = np.zeros(ns * nf)
        expected[0] = 1 / np.sqrt(ns)
        np.testing.assert_equal(actual["data"], expected)
