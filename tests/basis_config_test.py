from __future__ import annotations

import unittest

import numpy as np

from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.stacked_basis.brillouin_zone import (
    decrement_brillouin_zone_3d,
    get_all_brag_point,
    get_all_brag_point_index,
    get_bragg_plane_distance,
)
from surface_potential_analysis.stacked_basis.build import (
    momentum_basis_3d_from_resolution,
    position_basis_3d_from_shape,
)
from surface_potential_analysis.stacked_basis.conversion import (
    tuple_basis_as_fundamental,
)
from surface_potential_analysis.stacked_basis.util import (
    _wrap_distance,  # type: ignore this is testing module
    calculate_cumulative_x_distances_along_path,
    get_fundamental_stacked_k_points_projected_along_axes,
    get_fundamental_stacked_x_points_projected_along_axes,
    project_k_points_along_axes,
    project_x_points_along_axes,
    wrap_index_around_origin,
    wrap_x_point_around_origin,
)
from surface_potential_analysis.util.util import slice_along_axis

rng = np.random.default_rng()


class TestBasisConfig(unittest.TestCase):
    def test_surface_volume_100(self) -> None:
        points = rng.random(3)
        basis = position_basis_3d_from_shape(
            (1, 1, 1),
            np.array([[points[0], 0, 0], [0, points[1], 0], [0, 0, points[2]]]),
        )
        util = BasisUtil(basis)

        np.testing.assert_almost_equal(util.volume, np.prod(points))
        np.testing.assert_almost_equal(
            util.reciprocal_volume, (2 * np.pi) ** 3 / np.prod(points)
        )

        util2 = BasisUtil(basis)
        np.testing.assert_almost_equal(util2.volume, np.prod(points))
        np.testing.assert_almost_equal(
            util2.reciprocal_volume, (2 * np.pi) ** 3 / np.prod(points)
        )

    def test_inverse_lattice_points_100(self) -> None:
        delta_x = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 1]])
        basis = position_basis_3d_from_shape((1, 1, 1), delta_x)
        util = BasisUtil(basis)

        np.testing.assert_array_equal(delta_x[0], util.delta_x_stacked[0])
        np.testing.assert_array_equal(delta_x[1], util.delta_x_stacked[1])
        np.testing.assert_array_equal(delta_x[2], util.delta_x_stacked[2])

        self.assertEqual(util.dk_stacked[0][0], 2 * np.pi)
        self.assertEqual(util.dk_stacked[0][1], 0)
        self.assertEqual(util.dk_stacked[1][0], 0)
        self.assertEqual(util.dk_stacked[1][1], np.pi)

        util2 = BasisUtil(basis)

        np.testing.assert_array_equal(delta_x[0], util2.delta_x_stacked[0])
        np.testing.assert_array_equal(delta_x[1], util2.delta_x_stacked[1])
        np.testing.assert_array_equal(delta_x[2], util2.delta_x_stacked[2])

        self.assertEqual(util2.dk_stacked[0][0], 2 * np.pi)
        self.assertEqual(util2.dk_stacked[0][1], 0)
        self.assertEqual(util2.dk_stacked[1][0], 0)
        self.assertEqual(util2.dk_stacked[1][1], np.pi)

    def test_inverse_lattice_points_111(self) -> None:
        delta_x = np.array([[1, 0, 0], [0.5, np.sqrt(3) / 2, 0], [0, 0, 1]])
        basis = position_basis_3d_from_shape((1, 1, 1), delta_x)
        util = BasisUtil(basis)

        np.testing.assert_array_equal(delta_x[0], util.delta_x_stacked[0])
        np.testing.assert_array_equal(delta_x[1], util.delta_x_stacked[1])
        np.testing.assert_array_equal(delta_x[2], util.delta_x_stacked[2])

        self.assertAlmostEqual(util.dk_stacked[0][0], 2 * np.pi)
        self.assertAlmostEqual(util.dk_stacked[0][1], -2 * np.pi / np.sqrt(3))
        self.assertAlmostEqual(util.dk_stacked[1][0], 0)
        self.assertAlmostEqual(util.dk_stacked[1][1], 4 * np.pi / np.sqrt(3))

        util2 = BasisUtil(basis)
        np.testing.assert_array_equal(delta_x[0], util2.delta_x_stacked[0])
        np.testing.assert_array_equal(delta_x[1], util2.delta_x_stacked[1])
        np.testing.assert_array_equal(delta_x[2], util2.delta_x_stacked[2])

        self.assertAlmostEqual(util2.dk_stacked[0][0], 2 * np.pi)
        self.assertAlmostEqual(util2.dk_stacked[0][1], -2 * np.pi / np.sqrt(3))
        self.assertAlmostEqual(util2.dk_stacked[1][0], 0)
        self.assertAlmostEqual(util2.dk_stacked[1][1], 4 * np.pi / np.sqrt(3))

    def test_reciprocal_lattice(self) -> None:
        delta_x = rng.random((3, 3))
        basis = position_basis_3d_from_shape((1, 1, 1), delta_x)
        util = BasisUtil(basis)

        np.testing.assert_array_almost_equal(delta_x[0], util.delta_x_stacked[0])
        np.testing.assert_array_almost_equal(delta_x[1], util.delta_x_stacked[1])
        np.testing.assert_array_almost_equal(delta_x[2], util.delta_x_stacked[2])

        reciprocal = stacked_basis_as_transformed_basis(basis)
        reciprocal_util = BasisUtil(reciprocal)

        np.testing.assert_array_almost_equal(
            reciprocal_util.delta_x_stacked[0], util.delta_x_stacked[0]
        )
        np.testing.assert_array_almost_equal(
            reciprocal_util.delta_x_stacked[1], util.delta_x_stacked[1]
        )
        np.testing.assert_array_almost_equal(
            reciprocal_util.delta_x_stacked[2], util.delta_x_stacked[2]
        )

        np.testing.assert_array_almost_equal(
            reciprocal_util.dk_stacked[0], util.dk_stacked[0]
        )
        np.testing.assert_array_almost_equal(
            reciprocal_util.dk_stacked[1], util.dk_stacked[1]
        )
        np.testing.assert_array_almost_equal(
            reciprocal_util.dk_stacked[2], util.dk_stacked[2]
        )

        np.testing.assert_array_almost_equal(reciprocal_util.volume, util.volume)
        np.testing.assert_array_almost_equal(
            reciprocal_util.reciprocal_volume, util.reciprocal_volume
        )

        reciprocal_2 = tuple_basis_as_fundamental(reciprocal)
        reciprocal_2_util = BasisUtil(reciprocal_2)

        np.testing.assert_array_almost_equal(
            reciprocal_2_util.delta_x_stacked[0], util.delta_x_stacked[0]
        )
        np.testing.assert_array_almost_equal(
            reciprocal_2_util.delta_x_stacked[1], util.delta_x_stacked[1]
        )
        np.testing.assert_array_almost_equal(
            reciprocal_2_util.delta_x_stacked[2], util.delta_x_stacked[2]
        )

        np.testing.assert_array_almost_equal(
            reciprocal_2_util.dk_stacked[0], util.dk_stacked[0]
        )
        np.testing.assert_array_almost_equal(
            reciprocal_2_util.dk_stacked[1], util.dk_stacked[1]
        )
        np.testing.assert_array_almost_equal(
            reciprocal_2_util.dk_stacked[2], util.dk_stacked[2]
        )

    def test_get_stacked_index(self) -> None:
        delta_x = np.array(([1, 0, 0], [0, 2, 0], [0, 0, 1]))
        resolution = (
            rng.integers(1, 10),  # type: ignore bad libary types
            rng.integers(1, 10),  # type: ignore bad libary types
            rng.integers(1, 10),  # type: ignore bad libary types
        )
        basis = position_basis_3d_from_shape(resolution, delta_x)
        util = BasisUtil(basis)
        for i in range(np.prod(resolution)):
            self.assertEqual(i, util.get_flat_index(util.get_stacked_index(i)))

        util2 = BasisUtil(basis)
        for i in range(np.prod(resolution)):
            self.assertEqual(i, util2.get_flat_index(util2.get_stacked_index(i)))

    def test_nx_points_simple(self) -> None:
        delta_x = rng.random((3, 3))
        basis = position_basis_3d_from_shape((2, 2, 2), delta_x)
        util = BasisUtil(basis)

        actual = util.fundamental_stacked_nx_points
        expected = [
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 1, 0, 1, 0, 1],
        ]
        np.testing.assert_array_equal(expected, actual)

        resolution = (
            rng.integers(1, 20),  # type: ignore bad libary types
            rng.integers(1, 20),  # type: ignore bad libary types
            rng.integers(1, 20),  # type: ignore bad libary types
        )
        basis = position_basis_3d_from_shape(resolution, delta_x)
        util = BasisUtil(basis)
        actual = util.fundamental_stacked_nx_points

        for axis in range(3):
            basis_for_axis = actual[axis].reshape(*resolution)
            for j in range(resolution[axis]):
                slice_j = basis_for_axis[slice_along_axis(j, axis)]
                np.testing.assert_equal(slice_j, j)

    def test_nk_points_simple(self) -> None:
        delta_x = rng.random((3, 3))
        basis = position_basis_3d_from_shape((2, 2, 2), delta_x)
        util = BasisUtil(basis)

        actual = util.fundamental_stacked_nk_points
        expected = [
            [0, 0, 0, 0, -1, -1, -1, -1],
            [0, 0, -1, -1, 0, 0, -1, -1],
            [0, -1, 0, -1, 0, -1, 0, -1],
        ]
        np.testing.assert_array_equal(expected, actual)

        resolution = (
            rng.integers(1, 20),  # type: ignore bad libary types
            rng.integers(1, 20),  # type: ignore bad libary types
            rng.integers(1, 20),  # type: ignore bad libary types
        )
        basis = position_basis_3d_from_shape(resolution, delta_x)
        util = BasisUtil(basis)
        actual = util.fundamental_stacked_nk_points

        for axis in range(3):
            basis_for_axis = actual[axis].reshape(*resolution)
            expected_for_axis = np.fft.fftfreq(resolution[axis], 1 / resolution[axis])

            for j, expected in enumerate(expected_for_axis):
                slice_j = basis_for_axis[slice_along_axis(j, axis)]
                np.testing.assert_equal(slice_j, expected)

    def test_x_points_100(self) -> None:
        delta_x = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)

        basis = position_basis_3d_from_shape((3, 3, 3), delta_x)
        util = BasisUtil(basis)

        actual = util.fundamental_x_points_stacked
        # fmt: off
        expected_x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]) / 3
        expected_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]) / 3
        expected_z = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]) / 3
        # fmt: on

        np.testing.assert_array_equal(expected_x, actual[0])
        np.testing.assert_array_equal(expected_y, actual[1])
        np.testing.assert_array_equal(expected_z, actual[2])

        delta_x = np.array([[0, 1, 0], [3, 0, 0], [0, 0, 5]], dtype=float)
        basis = position_basis_3d_from_shape((3, 3, 3), delta_x)
        util = BasisUtil(basis)
        actual = util.fundamental_x_points_stacked

        # fmt: off
        expected_x = np.array([0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 6.0, 6.0, 6.0]) / 3
        expected_y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]) / 3
        expected_z = np.array([0.0, 5.0, 10.0, 0.0, 5.0, 10.0, 0.0, 5.0, 10.0, 0.0, 5.0, 10.0, 0.0, 5.0, 10.0, 0.0, 5.0, 10.0, 0.0, 5.0, 10.0, 0.0, 5.0, 10.0, 0.0, 5.0, 10.0]) / 3
        # fmt: on

        np.testing.assert_array_equal(expected_x, actual[0])
        np.testing.assert_array_equal(expected_y, actual[1])
        np.testing.assert_array_equal(expected_z, actual[2])

    def test_k_points_100(self) -> None:
        dx = (
            np.array([1, 0, 0], dtype=float),
            np.array([0, 1, 0], dtype=float),
            np.array([0, 0, 1], dtype=float),
        )
        basis = momentum_basis_3d_from_resolution((3, 3, 3), dx)
        util = BasisUtil(basis)

        actual = util.fundamental_stacked_nk_points
        # fmt: off
        expected_x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        expected_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
        expected_z = np.array([0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0])
        # fmt: on

        np.testing.assert_array_equal(expected_x, actual[0])
        np.testing.assert_array_equal(expected_y, actual[1])
        np.testing.assert_array_equal(expected_z, actual[2])

        delta_x = (
            np.array([0, 1, 0], dtype=float),
            np.array([3, 0, 0], dtype=float),
            np.array([0, 0, 5], dtype=float),
        )
        basis = momentum_basis_3d_from_resolution((3, 3, 3), delta_x)
        util = BasisUtil(basis)
        actual = util.fundamental_stacked_nk_points

        np.testing.assert_array_equal(expected_x, actual[0])
        np.testing.assert_array_equal(expected_y, actual[1])
        np.testing.assert_array_equal(expected_z, actual[2])

        actual_k = util.fundamental_stacked_k_points
        expected_kx = 2 * np.pi * expected_y / 3.0
        expected_ky = 2 * np.pi * expected_x / 1.0
        expected_kz = 2 * np.pi * expected_z / 5.0

        np.testing.assert_array_almost_equal(expected_kx, actual_k[0])
        np.testing.assert_array_almost_equal(expected_ky, actual_k[1])
        np.testing.assert_array_almost_equal(expected_kz, actual_k[2])

    def test_wrap_distance(self) -> None:
        expected = [0, 1, -1, 0, 1, -1, 0]
        distances = [-3, -2, -1, 0, 1, 2, 3]
        for e, d in zip(expected, distances, strict=True):
            np.testing.assert_equal(_wrap_distance(d, 3), e, f"d={d}, l=3")
        np.testing.assert_array_equal(_wrap_distance(np.array(distances), 3), expected)

        expected = [0, 1, -2, -1, 0, 1, -2, -1, 0]
        distances = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        for e, d in zip(expected, distances, strict=True):
            np.testing.assert_equal(_wrap_distance(d, 4), e, f"d={d}, l=4")
        np.testing.assert_array_equal(_wrap_distance(np.array(distances), 4), expected)

    def test_wrap_nx_point_around_origin(self) -> None:
        delta_x = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]], dtype=float)
        basis = position_basis_3d_from_shape((3, 3, 3), delta_x)
        util = BasisUtil(basis)
        actual = wrap_index_around_origin(basis, util.fundamental_stacked_nx_points)
        expected = util.fundamental_stacked_nk_points
        np.testing.assert_array_almost_equal(actual, expected)

        basis = position_basis_3d_from_shape((4, 4, 4), delta_x)
        util = BasisUtil(basis)
        actual = wrap_index_around_origin(basis, util.fundamental_stacked_nx_points)
        expected = util.fundamental_stacked_nk_points
        np.testing.assert_array_almost_equal(actual, expected)

    def test_wrap_x_point_around_origin(self) -> None:
        delta_x = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]], dtype=float)
        basis = position_basis_3d_from_shape((3, 3, 3), delta_x)
        util = BasisUtil(basis)
        actual = wrap_x_point_around_origin(basis, util.fundamental_x_points_stacked)
        expected = util.get_x_points_at_index(
            wrap_index_around_origin(basis, util.fundamental_stacked_nx_points)
        )
        np.testing.assert_array_almost_equal(actual, expected)
        expected = util.get_x_points_at_index(util.fundamental_stacked_nx_points)
        np.testing.assert_array_almost_equal(actual, expected)

        delta_x = np.array([[3, 0, 0], [1, 3, 0], [0, 0, 4]], dtype=float)
        basis = position_basis_3d_from_shape((3, 3, 3), delta_x)
        util = BasisUtil(basis)
        actual = wrap_x_point_around_origin(basis, util.fundamental_x_points_stacked)
        expected = util.get_x_points_at_index(
            wrap_index_around_origin(basis, util.fundamental_stacked_nx_points)
        )
        np.testing.assert_array_almost_equal(actual, expected)
        expected = util.get_x_points_at_index(util.fundamental_stacked_nk_points)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_calculate_cumulative_distances_along_path(self) -> None:
        delta_x = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]], dtype=float)

        basis = position_basis_3d_from_shape((3, 3, 3), delta_x)

        distances = calculate_cumulative_x_distances_along_path(
            basis, np.array([[0, 1], [0, 0], [0, 0]])
        )
        np.testing.assert_array_equal(distances, [0, 1])

        distances = calculate_cumulative_x_distances_along_path(
            basis, np.array([[0, 2], [0, 0], [0, 0]])
        )
        np.testing.assert_array_equal(distances, [0, 2])

        distances = calculate_cumulative_x_distances_along_path(
            basis, np.array([[0, 2], [0, 0], [0, 0]]), wrap_distances=True
        )
        np.testing.assert_array_equal(distances, [0, 1])

    def test_get_all_brag_point_index(self) -> None:
        basis = position_basis_3d_from_shape((1, 1, 1))
        indexes = get_all_brag_point_index(basis, n_bands=1)
        # fmt: off
        expected = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1],
            [0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1]
        ]
        # fmt: on
        np.testing.assert_array_equal(indexes, expected)

        basis = position_basis_3d_from_shape((2, 3, 5))
        indexes = get_all_brag_point_index(basis, n_bands=1)
        # fmt: off
        expected = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
            [0, 0, 0, 3, 3, 3, -3, -3, -3, 0, 0, 0, 3, 3, 3, -3, -3, -3, 0, 0, 0, 3, 3, 3, -3, -3, -3],
            [0, 5, -5, 0, 5, -5, 0, 5, -5, 0, 5, -5, 0, 5, -5, 0, 5, -5, 0, 5, -5, 0, 5, -5, 0, 5, -5]
        ]
        # fmt: on
        np.testing.assert_array_equal(indexes, expected)

    def test_get_bragg_distance(self) -> None:
        distance = get_bragg_plane_distance(np.array([2, 2, 2]), np.array([1, 1, 1]))
        expected = 0
        np.testing.assert_array_almost_equal(distance, expected)

        coordinates = np.array([[1, 0, 1, 2], [1, 0, 1, 2], [1, 0, 0, 2]])
        distances = get_bragg_plane_distance(np.array([2, 2, 2]), coordinates)
        expected_distances = np.array([0, -np.sqrt(3), -1 / np.sqrt(3), np.sqrt(3)])
        np.testing.assert_array_almost_equal(distances, expected_distances)

    def test_decrement_brillouin_zone(self) -> None:
        basis = position_basis_3d_from_shape((1, 1, 1))

        actual = decrement_brillouin_zone_3d(basis, (0, 0, 0))
        expected = (0, 0, 0)
        np.testing.assert_array_equal(np.asarray(actual), expected)

        actual_arr = decrement_brillouin_zone_3d(
            basis, (np.array(0), np.array(0), np.array(0))
        )
        expected = (np.array(0), np.array(0), np.array(0))
        np.testing.assert_array_equal(actual_arr, expected)

        actual_arr = decrement_brillouin_zone_3d(
            basis, (np.array([[0], [0]]), np.array([[0], [0]]), np.array([[0], [0]]))
        )
        expected = (np.array([[0], [0]]), np.array([[0], [0]]), np.array([[0], [0]]))
        np.testing.assert_array_equal(actual_arr, expected)

    def test_decrement_brillouin_zone_many(self) -> None:
        basis = position_basis_3d_from_shape((3, 3, 3))
        util = BasisUtil(position_basis_3d_from_shape((3, 3, 3)))
        actual = decrement_brillouin_zone_3d(basis, util.fundamental_stacked_nk_points)  # type: ignore[arg-type,var-annotated]

        expected = util.fundamental_stacked_nk_points
        np.testing.assert_array_equal(np.array(actual, dtype=float), expected)

        basis = position_basis_3d_from_shape((6, 6, 6))
        util = BasisUtil(position_basis_3d_from_shape((6, 6, 1)))
        actual = decrement_brillouin_zone_3d(basis, util.fundamental_stacked_nk_points)  # type: ignore[arg-type]

        expected = util.fundamental_stacked_nk_points
        # Not too sure about this tbh
        expected[0][21] = 3
        expected[1][21] = 3
        expected[2][21] = 0
        np.testing.assert_array_equal(np.array(actual, dtype=float), expected)

    def test_get_all_brag_point(self) -> None:
        delta_x = np.array([[2 * np.pi, 0, 0], [0, 2 * np.pi, 0], [0, 0, 2 * np.pi]])

        basis = position_basis_3d_from_shape((2, 2, 2), delta_x)
        actual = get_all_brag_point(basis)
        # fmt: off
        expected = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
            [0, 0, 0, 2, 2, 2, -2, -2, -2, 0, 0, 0, 2, 2, 2, -2, -2, -2, 0, 0, 0, 2, 2, 2, -2, -2, -2],
            [0, 2, -2, 0, 2, -2, 0, 2, -2, 0, 2, -2, 0, 2, -2, 0, 2, -2, 0, 2, -2, 0, 2, -2, 0, 2, -2],
        ])
        # fmt: on
        np.testing.assert_array_equal(actual, expected)

    def test_project_k_points_along_axis(self) -> None:
        basis = position_basis_3d_from_shape((3, 3, 3))

        points = BasisUtil(basis).fundamental_stacked_k_points
        actual = project_k_points_along_axes(points, basis, axes=(1, 2))
        expected = get_fundamental_stacked_k_points_projected_along_axes(
            basis, axes=(1, 2)
        )
        np.testing.assert_array_equal(actual, expected)

        points = BasisUtil(basis).fundamental_stacked_k_points
        actual = project_k_points_along_axes(points, basis, axes=(2, 0))
        expected = get_fundamental_stacked_k_points_projected_along_axes(
            basis, axes=(2, 0)
        )
        np.testing.assert_array_equal(actual, expected)

        points = BasisUtil(basis).fundamental_stacked_k_points
        actual = project_k_points_along_axes(points, basis, axes=(0, 1))
        expected = get_fundamental_stacked_k_points_projected_along_axes(
            basis, axes=(0, 1)
        )
        np.testing.assert_array_equal(actual, expected)

    def test_project_x_points_along_axis(self) -> None:
        basis = position_basis_3d_from_shape((3, 3, 3))

        points = np.array([1, 0, 0])
        actual = project_x_points_along_axes(points, basis, axes=(1, 2))
        expected = np.array([0, 0])
        np.testing.assert_array_equal(actual, expected)

        points = np.array([[1], [0], [0]])
        actual = project_x_points_along_axes(points, basis, axes=(1, 2))
        expected = np.array([[0], [0]])
        np.testing.assert_array_equal(actual, expected)

        points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        actual = project_x_points_along_axes(points, basis, axes=(0, 1))
        expected = np.array([[1, 0, 0], [0, 1, 0]])
        np.testing.assert_array_equal(actual, expected)

        points = BasisUtil(basis).fundamental_x_points_stacked
        actual = project_x_points_along_axes(points, basis, axes=(0, 1))
        expected = get_fundamental_stacked_x_points_projected_along_axes(
            basis, axes=(0, 1)
        )
        np.testing.assert_array_equal(actual, expected)

        points = np.array([[0, 0], [1, 0], [0, 1]])
        actual = project_x_points_along_axes(points, basis, axes=(1, 2))
        expected = np.array([[1, 0], [0, 1]])
        np.testing.assert_array_equal(actual, expected)

        points = BasisUtil(basis).fundamental_x_points_stacked
        actual = project_x_points_along_axes(points, basis, axes=(1, 2))
        expected = get_fundamental_stacked_x_points_projected_along_axes(
            basis, axes=(1, 2)
        )
        np.testing.assert_array_equal(actual, expected)
