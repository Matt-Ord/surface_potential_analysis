from pathlib import Path
import json
from typing import List, Tuple, TypedDict
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate


class EnergyData(TypedDict):
    x_points: List[float]
    y_points: List[float]
    z_points: List[float]
    points: List[List[List[float]]]


def load_raw_energy_data() -> EnergyData:
    path = Path(__file__).parent / "data" / "raw_energies.json"
    with path.open("r") as f:
        return json.load(f)


def plot_z_direction_energy_data(
    data: EnergyData, otherData: EnergyData | None = None
) -> None:
    fig, ax = plt.subplots()

    heights = data["z_points"]
    top_energy = data["points"][0][0]
    hollow_energy = data["points"][5][5]

    ax.plot(heights, top_energy, label="Top Site")
    ax.plot(heights, hollow_energy, label="Hollow Site")

    if otherData is not None:
        heights = otherData["z_points"]
        top_energy = otherData["points"][0][0]
        hollow_energy = otherData["points"][5][5]

        ax.plot(heights, top_energy, label="Top Site", ls="--")
        ax.plot(heights, hollow_energy, label="Hollow Site", ls="--")

    ax.set_title("Plot of energy at the Top and Hollow sites")
    ax.set_ylabel("Energy / eV")
    ax.set_xlabel("relative z position /m")
    ax.set_ylim(bottom=-10, top=100)
    ax.legend()

    fig.tight_layout()
    fig.show()
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
    fig.show()
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
    fig.show()
    fig.savefig("temp.png")


def normalize_energy(
    data: EnergyData,
) -> EnergyData:
    points = np.array(data["points"], dtype=float)
    normalized_points = points - points.min()
    return {
        "points": normalized_points.tolist(),
        "x_points": data["x_points"],
        "y_points": data["y_points"],
        "z_points": data["z_points"],
    }


def truncate_energy(data: EnergyData, v=100) -> EnergyData:
    points = np.array(data["points"], dtype=float)
    truncated_points = v * np.log(1 + (points / v))
    return {
        "points": truncated_points.tolist(),
        "x_points": data["x_points"],
        "y_points": data["y_points"],
        "z_points": data["z_points"],
    }


def interpolate_energies(
    data: EnergyData, shape: Tuple[int, int, int] = (40, 40, 100)
) -> EnergyData:
    x_points = list(np.linspace(data["x_points"][0], data["x_points"][-1], shape[0]))
    y_points = list(np.linspace(data["y_points"][0], data["y_points"][-1], shape[1]))
    z_points = list(np.linspace(data["z_points"][0], data["z_points"][-1], shape[2]))

    interpolator = scipy.interpolate.RegularGridInterpolator(
        [data["x_points"], data["y_points"], data["z_points"]], data["points"]
    )
    xt, yt, zt = np.meshgrid(x_points, y_points, z_points, indexing="ij")
    test_points = np.array([xt.ravel(), yt.ravel(), zt.ravel()]).T
    points: List[List[List[float]]] = (
        interpolator(test_points, method="quintic").reshape(*shape).tolist()
    )

    return {
        "points": points,
        "x_points": x_points,
        "y_points": y_points,
        "z_points": z_points,
    }


def interpolate_energies_spline(
    data: EnergyData, shape: Tuple[int, int, int] = (40, 40, 100)
) -> EnergyData:
    old_points = np.array(data["points"])
    x_points = list(np.linspace(data["x_points"][0], data["x_points"][-1], shape[0]))
    y_points = list(np.linspace(data["y_points"][0], data["y_points"][-1], shape[1]))
    z_points = list(np.linspace(data["z_points"][0], data["z_points"][-1], shape[2]))

    # tck = scipy.interpolate.splprep()
    # xt, yt, zt = np.meshgrid(x_points, y_points, z_points, indexing="ij")
    # test_points = np.array([xt.ravel(), yt.ravel(), zt.ravel()]).T
    # scipy.interpolate.splev(test_points)

    points = np.empty((old_points.shape[0], old_points.shape[1], shape[2]))
    xt, yt = np.meshgrid(
        range(old_points.shape[0]), range(old_points.shape[1]), indexing="ij"
    )
    old_xy_points = np.array([xt.ravel(), yt.ravel()]).T
    for (x, y) in old_xy_points:
        old_energies = data["points"][x][y]
        tck = scipy.interpolate.splrep(data["z_points"], old_energies, s=0)
        new_energy = scipy.interpolate.splev(z_points, tck, der=0)
        points[x, y] = new_energy

    # for z in range(len(z_points)):
    #     xt_initial, yt_initial = np.meshgrid(
    #         data["x_points"], data["y_points"], indexing="ij"
    #     )
    #     tck = scipy.interpolate.bisplrep(xt_initial, yt_initial, z, s=0)
    #     znew = scipy.interpolate.bisplev(xnew[:, 0], ynew[0, :], tck)

    return {
        "points": points.tolist(),
        "x_points": data["x_points"],
        "y_points": data["y_points"],
        "z_points": z_points,
    }


if __name__ == "__main__":
    data = normalize_energy(load_raw_energy_data())
    truncated_data = truncate_energy(data)
    plot_z_direction_energy_data(data)
    plot_z_direction_energy_data(data, truncated_data)
    # plot_xz_plane_energy(data)
    interpolated = interpolate_energies_spline(truncated_data)
    # interpolated = interpolate_energies(data)
    plot_z_direction_energy_data(data, interpolated)
    # plot_xz_plane_energy(interpolated)
    # plot_x_direction_energy_data(interpolated)
    input()
