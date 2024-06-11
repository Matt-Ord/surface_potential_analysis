from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any

import hamiltonian_generator
import numpy as np
from scipy.constants import hbar

from surface_potential_analysis.basis.basis import (
    FundamentalPositionBasis3d,
    TransformedPositionBasis,
    TransformedPositionBasis3d,
)
from surface_potential_analysis.basis.explicit_basis import ExplicitBasisWithLength
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasis,
    TupleBasisLike,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.stacked_basis.sho_basis import (
    SHOBasisConfig,
    infinate_sho_basis_3d_from_config,
)
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_to_basis,
)
from tests.utils import get_random_explicit_basis

if TYPE_CHECKING:
    from surface_potential_analysis.basis.explicit_basis import ExplicitBasis3d
    from surface_potential_analysis.state_vector.state_vector import StateVector

_rng = np.random.default_rng()


def _get_random_sho_eigenstate(
    resolution: tuple[int, int, int], fundamental_resolution: tuple[int, int, int]
) -> StateVector[
    TupleBasisLike[
        TransformedPositionBasis3d[Any, Any],
        TransformedPositionBasis3d[Any, Any],
        ExplicitBasis3d[int, Any],
    ]
]:
    vector = np.array(_rng.random(np.prod(resolution)), dtype=np.complex128)
    vector /= np.linalg.norm(vector)

    x2_basis = ExplicitBasisWithLength(
        np.array([0, 0, 20]),
        get_random_explicit_basis(
            3, fundamental_n=fundamental_resolution[2], n=resolution[2]
        ).vectors,
    )
    return {
        "basis": StackedBasis(
            TransformedPositionBasis(
                np.array([1, 0, 0]), resolution[0], fundamental_resolution[0]
            ),
            TransformedPositionBasis(
                np.array([0, 1, 0]), resolution[1], fundamental_resolution[1]
            ),
            x2_basis,
        ),
        "data": vector,
    }


class EigenstateConversionTest(unittest.TestCase):
    def test_convert_sho_eigenstate_rust_simple(self) -> None:
        resolution = (5, 6, 9)
        eigenstate = _get_random_sho_eigenstate(resolution, (10, 10, 100))
        config: SHOBasisConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "x_origin": np.array([0, 0, -10]),
        }

        eigenstate["basis"] = StackedBasis(
            eigenstate["basis"][0],
            eigenstate["basis"][1],
            infinate_sho_basis_3d_from_config(
                eigenstate["basis"][2], config, resolution[2]
            ),
        )

        util = BasisUtil(eigenstate["basis"])

        for i in range(resolution[2]):
            vector = np.zeros_like(eigenstate["data"])
            vector[np.ravel_multi_index((0, 0, i), resolution)] = 1
            eigenstate["data"] = vector

            points = (
                util.fundamental_x_points_stacked + config["x_origin"][:, np.newaxis]
            )

            actual = hamiltonian_generator.get_eigenstate_wavefunction(
                resolution,
                (util.delta_x_stacked[0].item(0), 0),
                (util.delta_x_stacked[1].item(0), util.delta_x_stacked[1].item(1)),
                config["mass"],
                config["sho_omega"],
                0,
                0,
                eigenstate["data"].tolist(),
                points.T.tolist(),
            )

            basis = StackedBasis[Any](
                FundamentalPositionBasis3d(
                    util.delta_x_stacked[0], util.fundamental_shape[0]
                ),  # type: ignore[misc]
                FundamentalPositionBasis3d(
                    util.delta_x_stacked[1], util.fundamental_shape[1]
                ),  # type: ignore[misc]
                FundamentalPositionBasis3d(
                    util.delta_x_stacked[2], util.fundamental_shape[2]
                ),  # type: ignore[misc]
            )
            expected = convert_state_vector_to_basis(eigenstate, basis)
            np.testing.assert_allclose(
                expected["data"], np.array(actual) / np.linalg.norm(actual)
            )

    def test_convert_sho_eigenstate_rust(self) -> None:
        resolution = (5, 6, 9)
        eigenstate = _get_random_sho_eigenstate(resolution, (10, 10, 100))
        config: SHOBasisConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "x_origin": np.array([0, 0, -10]),
        }

        eigenstate["basis"] = StackedBasis(
            eigenstate["basis"][0],
            eigenstate["basis"][1],
            infinate_sho_basis_3d_from_config(
                eigenstate["basis"][2], config, resolution[2]
            ),
        )

        util = BasisUtil(eigenstate["basis"])

        points = util.fundamental_x_points_stacked + config["x_origin"][:, np.newaxis]
        actual = hamiltonian_generator.get_eigenstate_wavefunction(
            resolution,
            (util.delta_x_stacked[0].item(0), 0),
            (util.delta_x_stacked[1].item(0), util.delta_x_stacked[1].item(1)),
            config["mass"],
            config["sho_omega"],
            0,
            0,
            eigenstate["data"].tolist(),
            points.T.tolist(),
        )

        basis = StackedBasis[Any](
            FundamentalPositionBasis3d(
                util.delta_x_stacked[0], util.fundamental_shape[0]
            ),  # type: ignore[misc]
            FundamentalPositionBasis3d(
                util.delta_x_stacked[1], util.fundamental_shape[1]
            ),  # type: ignore[misc]
            FundamentalPositionBasis3d(
                util.delta_x_stacked[2], util.fundamental_shape[2]
            ),  # type: ignore[misc]
        )
        expected = convert_state_vector_to_basis(eigenstate, basis)
        np.testing.assert_allclose(
            expected["data"], np.array(actual) / np.linalg.norm(actual)
        )
