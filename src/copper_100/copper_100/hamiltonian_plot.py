from surface_potential_analysis.energy_data import as_interpolation, get_xy_points_delta
from surface_potential_analysis.energy_eigenstate import EigenstateConfig
from surface_potential_analysis.plot_sho_wavefunction import (
    plot_energy_with_sho_potential_at_hollow,
)
from surface_potential_analysis.plot_surface_hamiltonian import plot_nth_eigenstate

from .hamiltonian import generate_hamiltonian
from .potential import load_interpolated_copper_data


def plot_interpolation_with_sho_config() -> None:
    data = load_interpolated_copper_data()
    interpolation = as_interpolation(data)
    # 80% 99514067252307.23
    # 50% 117905964225836.06
    config: EigenstateConfig = {
        "mass": 1.6735575e-27,
        "sho_omega": 117905964225836.06,  # 1e14,
        "delta_x": get_xy_points_delta(data["x_points"]),
        "delta_y": get_xy_points_delta(data["y_points"]),
        "resolution": (1, 1, 1),
    }
    z_offset = -1.840551985155284e-10
    plot_energy_with_sho_potential_at_hollow(interpolation, config, z_offset)


def plot_copper_ground_eigenvector():
    h = generate_hamiltonian(resolution=(12, 12, 10))
    fig, _ = plot_nth_eigenstate(h)

    fig.show()
    input()