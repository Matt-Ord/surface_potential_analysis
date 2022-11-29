import math

import matplotlib.pyplot as plt
import numpy as np

from energy_data import (
    EnergyData,
    fill_surface_from_z_maximum,
    interpolate_energies_grid,
    load_raw_energy_data,
    normalize_energy,
    truncate_energy,
)


def plot_z_direction_energy_data(
    data: EnergyData, otherData: EnergyData | None = None
) -> None:
    fig, ax = plt.subplots()

    heights = data["z_points"]
    points = np.array(data["points"])
    top_energy = points[0, 0]
    middle_x_index = math.floor(points.shape[0] / 2)
    max_potential = 1e-18

    bridge_energy = points[middle_x_index, 0]
    hollow_energy = points[middle_x_index, math.ceil(points.shape[1] / 2)]

    ax.plot(heights, top_energy, label="Top Site")
    ax.plot(heights, bridge_energy, label="Bridge Site")
    ax.plot(heights, hollow_energy, label="Hollow Site")

    if otherData is not None:
        heights = otherData["z_points"]
        points = np.array(otherData["points"])
        top_energy = points[0, 0]
        middle_x_index = math.floor(points.shape[0] / 2)
        bridge_energy = points[middle_x_index, 0]
        hollow_energy = points[middle_x_index, math.ceil(points.shape[1] / 2)]

        ax.plot(heights, top_energy, label="Top Site", ls="--")
        ax.plot(heights, bridge_energy, label="Bridge Site", ls="--")
        ax.plot(heights, hollow_energy, label="Hollow Site", ls="--")

    ax.set_title("Plot of energy at the Top and Hollow sites")
    ax.set_ylabel("Energy / J")
    ax.set_xlabel("relative z position /m")
    ax.set_ylim(bottom=0, top=max_potential)
    ax.legend()

    fig.tight_layout()
    fig.show()
    fig.savefig("temp.png")


def plot_x_direction_energy_data(data: EnergyData) -> None:
    fig, ax = plt.subplots()

    heights = data["x_points"]
    points = np.array(data["points"])
    middle_x_index = math.floor(points.shape[0] / 2)
    middle_y_index = math.floor(points.shape[1] / 2)
    top_equilibrium = np.argmin(points[0, 0])
    hollow_equilibrium = np.argmin(points[middle_x_index, middle_y_index])

    hollow_eq_energy = points[:, middle_y_index, hollow_equilibrium]
    top_eq_energy = points[:, middle_y_index, top_equilibrium]
    hollow_max_energy = points[:, middle_y_index, 0]

    ax.plot(heights, top_eq_energy, label="Near Top Equilibrium")
    ax.plot(heights, hollow_eq_energy, label="Near Hollow Equilibrium")
    ax.plot(heights, hollow_max_energy, label="Near Hollow Maximum")

    ax.set_title("Plot of energy in the x direction")
    ax.set_ylabel("Energy / eV")
    ax.set_xlabel("relative z position /m")
    ax.legend()

    fig.tight_layout()
    fig.show()
    fig.savefig("temp.png")


def plot_xz_plane_energy(data: EnergyData) -> None:
    fig, axs = plt.subplots(nrows=2, ncols=3)

    x_points = np.array(data["x_points"])
    y_points = np.array(data["y_points"])
    z_points = np.array(data["z_points"])
    points = np.array(data["points"])
    middle_x_index = math.floor(points.shape[0] / 2)
    middle_y_index = math.floor(points.shape[1] / 2)
    max_potential = 1e-18

    bridge_energies = np.clip(points[::, 0, ::-1].transpose(), 0, max_potential)
    hollow_energies = np.clip(
        points[::, middle_x_index, ::-1].transpose(), 0, max_potential
    )
    top_hollow_energies = np.clip(points.diagonal()[::-1], 0, max_potential)

    extent = [x_points[0], x_points[-1], z_points[0], z_points[-1]]
    axs[0][0].imshow(bridge_energies, extent=extent)
    axs[0][2].imshow(hollow_energies, extent=extent)
    extent = [
        np.sqrt(2) * x_points[0],
        np.sqrt(2) * x_points[-1],
        data["z_points"][0],
        data["z_points"][-1],
    ]
    axs[0][1].imshow(top_hollow_energies, extent=extent)

    extent = [x_points[0], x_points[-1], y_points[0], y_points[-1]]
    bottom_energies = np.clip(points[::, ::, 0], 0, max_potential)
    axs[1][0].imshow(bottom_energies, extent=extent)
    equilibrium_z = np.argmin(points[middle_x_index, middle_y_index])
    equilibrium_energies = np.clip(points[::, ::, equilibrium_z], 0, max_potential)
    axs[1][2].imshow(equilibrium_energies, extent=extent)

    axs[0][1].sharey(axs[0][0])
    axs[0][2].sharey(axs[0][0])

    axs[0][0].set_xlabel("x Position")
    axs[0][0].set_ylabel("z position /m")

    axs[0][0].set_title("Top-Bridge Site")
    axs[0][1].set_title("Top-Hollow Site")
    axs[0][2].set_title("Bridge-Hollow Site")
    axs[0][0].set_title("Bottom Energies")
    axs[0][2].set_title("Equilibrium Energies")

    fig.suptitle("Plot of energy through several planes perpendicular to xy")
    fig.tight_layout()
    fig.show()
    fig.savefig("temp.png")


if __name__ == "__main__":
    data = normalize_energy(load_raw_energy_data())
    plot_xz_plane_energy(data)
    data = fill_surface_from_z_maximum(data)
    # data = fill_subsurface_from_hollow_sample(data)
    # Spline 1D v=600, n=6
    truncated_data = truncate_energy(data, cutoff=3e-18, n=6, offset=1e-20)

    plot_z_direction_energy_data(data, truncated_data)
    # plot_x_direction_energy_data(data)
    # plot_x_direction_energy_data(truncated_data)
    # plot_xz_plane_energy(data)
    # plot_xz_plane_energy(truncated_data)
    plot_xz_plane_energy(data)
    interpolated = interpolate_energies_grid(truncated_data, shape=(21, 21, 100))
    plot_z_direction_energy_data(data, interpolated)
    plot_xz_plane_energy(interpolated)
    plot_x_direction_energy_data(data)
    plot_x_direction_energy_data(interpolated)

    # truncated2 = truncate_energy(data, cutoff=6e-18, n=6, offset=1e-20)
    # interpolated2 = interpolate_energies_grid(truncated2, shape=(21, 21, 50))
    # plot_xz_plane_energy(interpolated2)
    # plot_z_direction_energy_data(data, interpolated2)
    # plot_z_direction_energy_data(data, truncated2)
    # plot_x_direction_energy_data(interpolated2)
    input()
