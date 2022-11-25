from pathlib import Path
import json
from typing import List, TypedDict
import matplotlib.pyplot as plt
import numpy as np


class EnergyData(TypedDict):
    x_points: List[float]
    y_points: List[float]
    z_points: List[float]
    points: List[List[List[float]]]


def load_raw_energy_data() -> EnergyData:
    path = Path(__file__).parent / "data" / "raw_energies.json"
    with path.open("r") as f:
        return json.load(f)


def plot_z_direction_energy_data(data: EnergyData) -> None:
    fig, ax = plt.subplots()

    heights = data["z_points"]
    top_energy = data["points"][0][0]
    hollow_energy = data["points"][5][5]

    ax.plot(heights[8:], top_energy[8:], label="Top Site")
    ax.plot(heights[1:], hollow_energy[1:], label="Hollow Site")

    ax.set_title("Plot of energy at the Top and Hollow sites")
    ax.set_ylabel("Energy / eV")
    ax.set_xlabel("relative z position /m")
    ax.legend()

    fig.tight_layout()
    fig.savefig("temp.png")


# TODO: Use Actual equilibrium Heights
def plot_x_direction_energy_data(data: EnergyData) -> None:
    fig, ax = plt.subplots()

    heights = data["x_points"]
    hollow_eq_energy = [data["points"][ix][5][8] for ix in range(len(heights))]
    top_eq_energy = [data["points"][ix][5][11] for ix in range(len(heights))]

    ax.plot(heights, top_eq_energy, label="Near Top Equilibrium")
    ax.plot(heights, hollow_eq_energy, label="Near Hollow Equilibrium")

    ax.set_title("Plot of energy in the x direction")
    ax.set_ylabel("Energy / eV")
    ax.set_xlabel("relative z position /m")
    ax.legend()

    fig.tight_layout()
    fig.savefig("temp.png")


def plot_xz_plane_energy(data: EnergyData) -> None:
    fig, axs = plt.subplots(nrows=1, ncols=3)

    points = np.array(data["points"])
    bridge_energies = points[::, 0, ::-1].transpose()
    hollow_energies = points[::, 5, ::-1].transpose()
    top_hollow_energies = points.diagonal()[::-1]

    cutoff_energy = np.max(data["points"][0][0][8:])

    axs[0].imshow(bridge_energies, vmax=cutoff_energy)
    axs[1].imshow(top_hollow_energies, vmax=cutoff_energy)
    axs[1].sharey(axs[0])
    axs[2].imshow(hollow_energies, vmax=cutoff_energy)
    axs[2].sharey(axs[0])

    axs[0].set_xlabel("x Position")
    axs[0].set_ylabel("z position /m")
    axs[0].set_title("", pad=0)

    fig.suptitle("Plot of energy through several planes perpendicular to xy")
    fig.tight_layout()
    fig.savefig("temp.png")


def interpolate_z_energies(data: EnergyData) -> EnergyData:
    return data


if __name__ == "__main__":
    data = load_raw_energy_data()
    # plot_z_direction_energy_data(data)
    plot_xz_plane_energy(data)
