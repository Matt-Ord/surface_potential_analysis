import json
from pathlib import Path
from typing import List, Set, Tuple, TypedDict

import numpy as np
import scipy.interpolate


class EnergyData(TypedDict):
    x_points: List[float]
    y_points: List[float]
    z_points: List[float]
    mass: float
    sho_omega: float
    points: List[List[List[float]]]


def load_raw_energy_data() -> EnergyData:
    path = Path(__file__).parent / "data" / "raw_energies.json"
    with path.open("r") as f:
        out = json.load(f)
        # TODO
        out["sho_omega"] = 1
        out["mass"] = 1
        return out


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
        "mass": data["mass"],
        "sho_omega": data["sho_omega"],
    }


# Attempt to fill from one corner.
# We don't have enough points (or the Hydrogen is not fixed enough)
# to be able to 'fill' the whole region we want
def fill_subsurface_from_corner(data: EnergyData) -> EnergyData:
    points = np.array(data["points"], dtype=float)
    points_to_fill: Set[Tuple[int, int, int]] = set([(0, 0, 0)])
    fill_level = 1.6

    while len(points_to_fill) > 0:
        current_point = points_to_fill.pop()

        if points[current_point] >= fill_level:
            continue

        points[current_point] = 1000
        for (dx, dy, dz) in [
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (-1, 0, 0),
            (0, -1, 0),
        ]:

            next_point = (
                (current_point[0] + dx) % points.shape[0],
                (current_point[1] + dy) % points.shape[1],
                current_point[2] + dz,
            )
            points_to_fill.add(next_point)

    return {
        "points": points.tolist(),
        "x_points": data["x_points"],
        "y_points": data["y_points"],
        "z_points": data["z_points"],
        "mass": data["mass"],
        "sho_omega": data["sho_omega"],
    }


def fill_subsurface_from_hollow_sample(data: EnergyData) -> EnergyData:
    points = np.array(data["points"], dtype=float)

    # Number of points to fill
    fill_height = 5
    hollow_sample = points[5, 5, :fill_height]

    points[:, :, :fill_height] = 0.5 * points[:, :, :fill_height] + 0.5 * hollow_sample

    return {
        "points": points.tolist(),
        "x_points": data["x_points"],
        "y_points": data["y_points"],
        "z_points": data["z_points"],
        "mass": data["mass"],
        "sho_omega": data["sho_omega"],
    }


# Fills the surface below the maximum pof the potential
# Since at the bridge point the maximum is at r-> infty we must take
# the maximum within the first half of the data
def fill_surface_from_z_maximum(data: EnergyData) -> EnergyData:
    points = np.array(data["points"], dtype=float)
    max_arg = np.argmax(points[:, :, :10], axis=2, keepdims=True)
    max_val = np.max(points[:, :, :10], axis=2, keepdims=True)

    z_index = np.indices(dimensions=points.shape)[2]
    should_use_max = z_index < max_arg
    points = np.where(should_use_max, max_val, points)

    return {
        "points": points.tolist(),
        "x_points": data["x_points"],
        "y_points": data["y_points"],
        "z_points": data["z_points"],
        "mass": data["mass"],
        "sho_omega": data["sho_omega"],
    }


def truncate_energy(
    data: EnergyData, *, cutoff=2e-17, n: int = 1, offset: float = 2e-18
) -> EnergyData:
    points = np.array(data["points"], dtype=float)
    truncated_points = (
        cutoff * np.log(1 + ((points + offset) / cutoff) ** n) ** (1 / n) - offset
    )
    return {
        "points": truncated_points.tolist(),
        "x_points": data["x_points"],
        "y_points": data["y_points"],
        "z_points": data["z_points"],
        "mass": data["mass"],
        "sho_omega": data["sho_omega"],
    }


def interpolate_energies_grid(
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
        "mass": data["mass"],
        "sho_omega": data["sho_omega"],
    }


# Uses spline interpolation to increase the Z resolution
def interpolate_energies_spline(
    data: EnergyData, shape: Tuple[int, int, int] = (40, 40, 1000)
) -> EnergyData:
    old_points = np.array(data["points"])
    z_points = list(np.linspace(data["z_points"][0], data["z_points"][-1], shape[2]))

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

    return {
        "points": points.tolist(),
        "x_points": data["x_points"],
        "y_points": data["y_points"],
        "z_points": z_points,
        "mass": data["mass"],
        "sho_omega": data["sho_omega"],
    }
