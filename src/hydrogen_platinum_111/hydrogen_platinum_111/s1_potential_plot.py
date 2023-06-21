from __future__ import annotations

import numpy as np
from surface_potential_analysis.potential.plot_point_potential import (
    get_point_potential_xy_locations,
    plot_point_potential_all_z,
    plot_point_potential_location_xy,
)

from .s1_potential import load_raw_data
from .surface_data import save_figure


def plot_raw_data_points() -> None:
    data = load_raw_data()
    fig, ax, _ = plot_point_potential_location_xy(data)

    locations = get_point_potential_xy_locations(data)
    e_min: list[float] = []
    for x, y in locations.T:
        idx = np.argwhere(
            np.logical_and(
                np.array(data["x_points"]) == x,
                np.array(data["y_points"]) == y,
            )
        )
        e_min.append(np.min(np.array(data["points"])[idx]))

    amin = np.argsort(e_min)
    x_min = locations[0][amin[0]]
    y_min = locations[1][amin[0]]
    ax.text(x_min, y_min, "FCC (lowest E)")

    x_min = locations[0][amin[1]]
    y_min = locations[1][amin[1]]
    ax.text(x_min, y_min, "HCP (second lowest E)")

    for i in range(2, 9):
        x_min = locations[0][amin[i]]
        y_min = locations[1][amin[i]]
        ax.text(x_min, y_min, f"({i})")

    x_min = locations[0][amin[-1]]
    y_min = locations[1][amin[-1]]
    ax.text(x_min, y_min, "Top (largest E)")

    fig.show()
    save_figure(fig, "raw_points.png")

    fig, ax = plot_point_potential_all_z(data)
    ax.set_ylim(0, 3 * 10**-19)

    ax.legend()
    fig.show()
    save_figure(fig, "raw_points_z.png")

    input()