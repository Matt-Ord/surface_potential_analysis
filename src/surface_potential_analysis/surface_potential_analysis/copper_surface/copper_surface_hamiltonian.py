from typing import Tuple

from ..energy_data.energy_data import as_interpolation, get_xy_points_delta
from ..energy_data.sho_config import (
    EigenstateConfig,
    generate_sho_config_minimum,
    plot_interpolation_with_sho,
)
from ..hamiltonian import SurfaceHamiltonian
from .copper_surface_potential import load_interpolated_copper_data


def generate_hamiltonian(resolution: Tuple[int, int, int] = (1, 1, 1)):
    data = load_interpolated_copper_data()
    interpolation = as_interpolation(data)
    config: EigenstateConfig = {
        "mass": 1.6735575e-27,
        "sho_omega": 117905964225836.06,
        "delta_x": get_xy_points_delta(data["x_points"]),
        "delta_y": get_xy_points_delta(data["y_points"]),
    }

    z_offset = -1.840551985155284e-10
    return SurfaceHamiltonian(resolution, interpolation, config, z_offset)


def generate_sho_config():
    data = load_interpolated_copper_data()
    interpolation = as_interpolation(data)
    mass = 1.6735575e-27
    return generate_sho_config_minimum(interpolation, mass, initial_guess=1e14)


def plot_interpolation_with_sho_config():
    data = load_interpolated_copper_data()
    interpolation = as_interpolation(data)
    # 80% 99514067252307.23
    # 50% 117905964225836.06
    config: EigenstateConfig = {
        "mass": 1.6735575e-27,
        "sho_omega": 117905964225836.06,  # 1e14,
        "delta_x": get_xy_points_delta(data["x_points"]),
        "delta_y": get_xy_points_delta(data["y_points"]),
    }
    z_offset = -1.840551985155284e-10
    plot_interpolation_with_sho(interpolation, config, z_offset)