import json
from pathlib import Path
from typing import List, Set, Tuple, TypedDict

import numpy as np
import scipy.interpolate


class EnergyData(TypedDict):
    x_points: List[float]
    y_points: List[float]
    z_points: List[float]
    points: List[List[List[float]]]


class EnergyInterpolation(TypedDict):
    delta_x: float
    delta_y: float
    dz: float
    # Note - points should exclude the 'nth' point
    points: List[List[List[float]]]


def get_xy_points_delta(points: List[float]):
    # Note additional factor to account for 'missing' point
    return (len(points)) * (points[-1] - points[0]) / (len(points) - 1)


def as_interpolation(data: EnergyData) -> EnergyInterpolation:
    """
    Converts between energy data and energy interpolation,
    assuming the x,y,z points are evenly spaced
    """
    delta_x = get_xy_points_delta(data["x_points"])
    delta_y = get_xy_points_delta(data["y_points"])
    dz = data["z_points"][1] - data["z_points"][0]

    x_points = np.linspace(0, delta_x, len(data["x_points"]))
    if not np.allclose(x_points, data["x_points"]):
        raise AssertionError("X Points Not evenly spaced")

    y_points = np.linspace(0, delta_y, len(data["y_points"]))
    if not np.allclose(y_points, data["y_points"]):
        raise AssertionError("y Points Not evenly spaced")

    nz = len(data["z_points"])
    z_points = np.linspace(0, dz * (nz - 1), nz) + data["z_points"][0]
    if not np.allclose(z_points, data["z_points"]):
        raise AssertionError("z Points Not evenly spaced")

    return {"delta_x": delta_x, "delta_y": delta_y, "dz": dz, "points": data["points"]}


def load_raw_energy_data() -> EnergyData:
    path = Path(__file__).parent / "data" / "raw_energies.json"
    with path.open("r") as f:
        out = json.load(f)
        # TODO
        out["sho_omega"] = 1
        out["mass"] = 1
        return out


def normalize_energy(data: EnergyData) -> EnergyData:
    points = np.array(data["points"], dtype=float)
    normalized_points = points - points.min()
    return {
        "points": normalized_points.tolist(),
        "x_points": data["x_points"],
        "y_points": data["y_points"],
        "z_points": data["z_points"],
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
    }


def repeat_original_data(
    data: EnergyData, x_padding: int = 1, y_padding: int = 1
) -> EnergyData:
    """Repeat the original data using the x, y symmetry to improve the interpolation convergence"""

    points = np.array(data["points"])
    x_points = data["x_points"]
    y_points = data["y_points"]

    x_tile = 1 + 2 * x_padding
    y_tile = 1 + 2 * y_padding
    xy_extended_data = np.tile(points, (x_tile, y_tile, 1))

    delta_x = get_xy_points_delta(x_points)
    delta_y = get_xy_points_delta(y_points)

    # Note we still have a 'missing' nth point
    new_x_points = np.linspace(
        -x_padding * delta_x,
        (x_padding + 1) * delta_x,
        num=x_tile * (len(x_points)),
        endpoint=False,
    )
    new_y_points = np.linspace(
        -y_padding * delta_y,
        (y_padding + 1) * delta_y,
        num=x_tile * (len(x_points)),
        endpoint=False,
    )
    return {
        "points": xy_extended_data.tolist(),
        "x_points": new_x_points.tolist(),
        "y_points": new_y_points.tolist(),
        "z_points": data["z_points"],
    }


def extend_z_data(data: EnergyData, extend_by: int = 2) -> EnergyData:
    old_points = np.array(data["points"])
    old_shape = old_points.shape
    z_len = old_shape[2] + 2 * extend_by
    z_extended_points = np.zeros(shape=(old_shape[0], old_shape[1], z_len))

    z_extended_points[:, :, extend_by:-extend_by] = old_points
    z_extended_points[:, :, :extend_by] = old_points[:, :, 0]
    z_extended_points[:, :, -extend_by:] = old_points[:, :, -1]

    z_points = np.zeros(shape=(z_len))
    z_points[extend_by:-extend_by] = np.array(data["z_points"])

    dz0 = data["z_points"][1] - data["z_points"][0]
    z_points[:extend_by] = np.linspace(
        data["z_points"][0] - (extend_by * dz0),
        data["z_points"][0],
        extend_by,
        endpoint=False,
    )
    dz1 = data["z_points"][-1] - data["z_points"][-2]
    z_points[-extend_by:] = np.linspace(
        data["z_points"][-1],
        data["z_points"][-1] + (extend_by * dz1),
        extend_by + 1,
    )[1:]

    return {
        "points": z_extended_points.tolist(),
        "x_points": data["x_points"],
        "y_points": data["y_points"],
        "z_points": z_points.tolist(),
    }


def add_back_symmetry_points(data: EnergyData) -> EnergyData:
    points = np.array(data["points"])
    nx = points.shape[0] + 1
    ny = points.shape[1] + 1
    extended_shape = (nx, ny, points.shape[2])
    extended_points = np.zeros(shape=extended_shape)
    extended_points[:-1, :-1] = points
    extended_points[-1, :-1] = points[0, :]
    extended_points[:-1, -1] = points[:, 0]
    extended_points[-1, -1] = points[0, 0]

    delta_x = get_xy_points_delta(data["x_points"])
    delta_y = get_xy_points_delta(data["y_points"])

    new_x_points = np.linspace(0, delta_x, nx)
    new_y_points = np.linspace(0, delta_y, ny)
    return {
        "points": extended_points.tolist(),
        "x_points": new_x_points.tolist(),
        "y_points": new_y_points.tolist(),
        "z_points": data["z_points"],
    }


def generate_interpolator(
    data: EnergyData,
) -> scipy.interpolate.RegularGridInterpolator:
    fixed_data = add_back_symmetry_points(extend_z_data(repeat_original_data(data)))
    return scipy.interpolate.RegularGridInterpolator(
        [fixed_data["x_points"], fixed_data["y_points"], fixed_data["z_points"]],
        fixed_data["points"],
    )


# TODO: symmetry, extend beyond
def interpolate_energies_grid(
    data: EnergyData, shape: Tuple[int, int, int] = (40, 40, 100)
) -> EnergyData:

    x_points = list(np.linspace(data["x_points"][0], data["x_points"][-1], shape[0]))
    y_points = list(np.linspace(data["y_points"][0], data["y_points"][-1], shape[1]))
    z_points = list(np.linspace(data["z_points"][0], data["z_points"][-1], shape[2]))

    interpolator = generate_interpolator(data)
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
    }
