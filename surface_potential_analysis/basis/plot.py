from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from surface_potential_analysis.util.plot import get_figure
from surface_potential_analysis.util.util import Measure, get_measured_data

from .util import BasisUtil

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis.basis.legacy import BasisWithLengthLike
    from surface_potential_analysis.types import SingleFlatIndexLike


def plot_explicit_basis_states_x(
    basis: BasisWithLengthLike,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
) -> tuple[Figure, Axes, list[Line2D]]:
    """Plot basis states against position."""
    fig, ax = get_figure(ax)
    util = BasisUtil(basis)

    x_points = np.linalg.norm(util.fundamental_x_points, axis=0)
    lines: list[Line2D] = []
    for i, vector in enumerate(util.vectors):
        data = get_measured_data(vector, measure)
        (line,) = ax.plot(x_points, data)  # type: ignore lib
        line.set_label(f"State {i}")
        lines.append(line)

    ax.set_xlabel("x / m")  # type: ignore lib
    ax.set_ylabel("Amplitude")  # type: ignore lib
    ax.set_title("Plot of the wavefunction of the explicit basis states")  # type: ignore lib
    return fig, ax, lines


def plot_explicit_basis_state_x(
    basis: BasisWithLengthLike,
    idx: SingleFlatIndexLike = 0,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
) -> tuple[Figure, Axes, Line2D]:
    """Plot basis states against position."""
    fig, ax = get_figure(ax)
    util = BasisUtil(basis)

    x_points = np.linalg.norm(util.fundamental_x_points, axis=0)
    data = get_measured_data(util.vectors[idx], measure)
    (line,) = ax.plot(x_points, data)  # type: ignore lib
    return fig, ax, line
