from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from surface_potential_analysis.eigenstate.eigenstate_collection import (
    EigenstateColllection3d,
    calculate_eigenstate_collection,
    save_eigenstate_collection,
)

from .s2_hamiltonian import generate_hamiltonian_sho
from .surface_data import get_data_path

if TYPE_CHECKING:
    from surface_potential_analysis.hamiltonian import Hamiltonian3d


def _calculate_eigenstate_collection_sho(
    bloch_fractions: np.ndarray[tuple[int, Literal[3]], np.dtype[np.float_]],
    resolution: tuple[int, int, int],
) -> EigenstateColllection3d[Any, Any]:
    def hamiltonian_generator(
        x: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> Hamiltonian3d[Any]:
        return generate_hamiltonian_sho(
            shape=(200, 200, 501),
            bloch_fraction=x,
            resolution=resolution,
        )

    return calculate_eigenstate_collection(
        hamiltonian_generator, bloch_fractions, subset_by_index=(0, 10)  # type: ignore[arg-type]
    )


def _generate_eigenstate_collection_sho(
    bloch_fractions: np.ndarray[tuple[int, Literal[3]], np.dtype[np.float_]],
    resolution: tuple[int, int, int],
) -> None:
    collection = _calculate_eigenstate_collection_sho(bloch_fractions, resolution)
    filename = f"eigenstates_{resolution[0]}_{resolution[1]}_{resolution[2]}.npy"
    path = get_data_path(filename)
    save_eigenstate_collection(path, collection)


def generate_eigenstates_data() -> None:
    """Generate data on the eigenstates and eigenvalues for a range of resolutions."""
    kx_points = np.linspace(0, 0.5, 5)
    ky_points = np.linspace(0, 0.5, 5)
    kz_points = np.zeros_like(kx_points)
    bloch_fractions = np.array([kx_points, ky_points, kz_points]).T

    # _generate_eigenstate_collection_sho(bloch_fractions, (10, 10, 5))  # noqa: ERA001

    _generate_eigenstate_collection_sho(bloch_fractions, (23, 23, 10))

    # _generate_eigenstate_collection_sho(bloch_fractions, (23, 23, 12)) # noqa: ERA001

    # _generate_eigenstate_collection_sho(bloch_fractions, (25, 25, 16)) # noqa: ERA001
