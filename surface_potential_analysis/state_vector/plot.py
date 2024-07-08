from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import scipy
import scipy.signal
from matplotlib.animation import ArtistAnimation

from surface_potential_analysis.basis.basis import BasisLike, FundamentalPositionBasis
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasisWithVolumeLike,
    TupleBasis,
    TupleBasisLike,
    TupleBasisWithLengthLike,
)
from surface_potential_analysis.basis.time_basis_like import (
    BasisWithTimeLike,
    EvenlySpacedTimeBasis,
)
from surface_potential_analysis.basis.util import (
    BasisUtil,
)
from surface_potential_analysis.operator.conversion import (
    convert_diagonal_operator_to_basis,
)
from surface_potential_analysis.operator.operator import (
    as_operator,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    stacked_basis_as_fundamental_position_basis,
)
from surface_potential_analysis.stacked_basis.util import (
    calculate_cumulative_x_distances_along_path,
)
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_list_to_basis,
    convert_state_vector_to_basis,
    convert_state_vector_to_momentum_basis,
    convert_state_vector_to_position_basis,
)
from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_eigenvectors_hermitian,
    calculate_expectation_list,
)
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_all_value_list_against_time,
    plot_average_value_list_against_time,
    plot_value_list_distribution,
)
from surface_potential_analysis.state_vector.state_vector_list import (
    as_state_vector_list,
    calculate_inner_products,
)
from surface_potential_analysis.util.plot import (
    animate_data_through_list_1d_k,
    animate_data_through_list_1d_x,
    animate_data_through_surface_x,
    get_figure,
    plot_data_1d_k,
    plot_data_1d_x,
    plot_data_2d_k,
    plot_data_2d_x,
)
from surface_potential_analysis.util.util import (
    Measure,
    get_measured_data,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis.basis.basis import FundamentalBasis
    from surface_potential_analysis.operator.operator import (
        Operator,
        SingleBasisOperator,
    )
    from surface_potential_analysis.state_vector.eigenstate_collection import ValueList
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )
    from surface_potential_analysis.types import (
        SingleStackedIndexLike,
    )
    from surface_potential_analysis.util.plot import Scale

    _B0Inv = TypeVar("_B0Inv", bound=BasisLike[Any, Any])


# ruff: noqa: PLR0913
def plot_state_1d_k(
    state: StateVector[StackedBasisWithVolumeLike[Any, Any, Any]],
    axes: tuple[int] = (0,),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot an state in 1d along the given axis.

    Parameters
    ----------
    state : StateVector[_B0Inv]
    idx : SingleStackedIndexLike, optional
        index in the perpendicular directions, by default (0,0)
    axis : int, optional
        axis along which to plot, by default 0
    ax : Axes | None, optional
        plot axis, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    converted = convert_state_vector_to_momentum_basis(state)

    fig, ax, line = plot_data_1d_k(
        converted["basis"],
        converted["data"],
        axes,
        idx,
        ax=ax,
        scale=scale,
        measure=measure,
    )
    ax.set_ylabel("State /Au")
    return fig, ax, line


def plot_state_1d_x(
    state: StateVector[StackedBasisWithVolumeLike[Any, Any, Any]],
    axes: tuple[int] = (0,),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot an state in 1d along the given axis.

    Parameters
    ----------
    state : StateVector[_B0Inv]
    idx : SingleStackedIndexLike, optional
        index in the perpendicular directions, by default (0,0)
    axis : int, optional
        axis along which to plot, by default 0
    ax : Axes | None, optional
        plot axis, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    converted = convert_state_vector_to_position_basis(state)

    fig, ax, line = plot_data_1d_x(
        converted["basis"],
        converted["data"],
        axes,
        idx,
        ax=ax,
        scale=scale,
        measure=measure,
    )
    ax.set_ylabel("State /Au")
    line.set_label(f"{measure} state")
    return fig, ax, line


def animate_state_over_list_1d_x(
    states: StateVectorList[_B0, _SB0],
    axes: tuple[int] = (0,),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Plot an state in 1d along the given axis, over time.

    Parameters
    ----------
    states : StateVectorList[BasisLike[Any, Any], TupleBasisLike[*tuple[Any, ...]]]
    idx : SingleStackedIndexLike, optional
        index in the perpendicular directions, by default (0,0)
    axis : int, optional
        axis along which to plot, by default 0
    ax : Axes | None, optional
        plot axis, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    converted = convert_state_vector_list_to_basis(
        states, stacked_basis_as_fundamental_position_basis(states["basis"][1])
    )

    fig, ax, ani = animate_data_through_list_1d_x(
        converted["basis"][1],
        converted["data"].reshape(converted["basis"].shape),
        axes,
        idx,
        ax=ax,
        scale=scale,
        measure=measure,
    )
    ax.set_ylabel("State /Au")
    return fig, ax, ani


def animate_state_over_list_1d_k(
    states: StateVectorList[_B0, _SB0],
    axes: tuple[int] = (0,),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Plot an state in 1d along the given axis, over time.

    Parameters
    ----------
    states : StateVectorList[BasisLike[Any, Any], TupleBasisLike[*tuple[Any, ...]]]
    idx : SingleStackedIndexLike, optional
        index in the perpendicular directions, by default (0,0)
    axis : int, optional
        axis along which to plot, by default 0
    ax : Axes | None, optional
        plot axis, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    converted = convert_state_vector_list_to_basis(
        states, stacked_basis_as_fundamental_momentum_basis(states["basis"][1])
    )

    fig, ax, ani = animate_data_through_list_1d_k(
        converted["basis"][1],
        converted["data"].reshape(converted["basis"].shape),
        axes,
        idx,
        ax=ax,
        scale=scale,
        measure=measure,
    )
    ax.set_ylabel("State /Au")
    return fig, ax, ani


def plot_state_2d_k(
    state: StateVector[_SB0],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot an state in 2d, perpendicular to kz_axis in momentum basis.

    Parameters
    ----------
    state : Eigenstate[_B3d0Inv]
    idx : SingleFlatIndexLike
        index along z_axis to plot
    kz_axis : Literal[0, 1, 2, -1, -2, -3]
        axis perpendicular to which to plot
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;, &quot;angle&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    converted = convert_state_vector_to_momentum_basis(state)

    return plot_data_2d_k(
        converted["basis"],
        converted["data"],
        axes,
        idx,
        ax=ax,
        scale=scale,
        measure=measure,
    )


def plot_state_difference_2d_k(
    state_0: StateVector[StackedBasisWithVolumeLike[Any, Any, Any]],
    state_1: StateVector[StackedBasisWithVolumeLike[Any, Any, Any]],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the difference between two eigenstates in k.

    Parameters
    ----------
    state_0 : StateVector[_B0Inv]
    state_1 : StateVector[_B1Inv]
    idx : SingleStackedIndexLike | None, optional
        index at each axis perpendicular to axis, by default None
    axis : int, optional
        axis to plot along, by default 0
    ax : Axes | None, optional
        plot axis, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    basis = stacked_basis_as_fundamental_momentum_basis(state_0["basis"])

    converted_0 = convert_state_vector_to_basis(state_0, basis)
    converted_1 = convert_state_vector_to_basis(state_1, basis)
    state: StateVector[Any] = {
        "basis": basis,
        "data": (converted_0["data"] - converted_1["data"])
        / np.max([np.abs(converted_0["data"]), np.abs(converted_1["data"])], axis=0),
    }
    return plot_state_2d_k(state, axes, idx, ax=ax, measure=measure, scale=scale)


def plot_state_2d_x(
    state: StateVector[StackedBasisWithVolumeLike[Any, Any, Any]],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot an state in 2d, perpendicular to z_axis.

    Parameters
    ----------
    state : Eigenstate[_B3d0Inv]
    idx : SingleFlatIndexLike
        index along z_axis to plot
    z_axis : Literal[0, 1, 2, -1, -2, -3]
        axis perpendicular to which to plot
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;, &quot;angle&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    converted = convert_state_vector_to_position_basis(state)

    return plot_data_2d_x(
        converted["basis"],
        converted["data"],
        axes,
        idx,
        ax=ax,
        scale=scale,
        measure=measure,
    )


def plot_state_difference_1d_k(
    state_0: StateVector[StackedBasisWithVolumeLike[Any, Any, Any]],
    state_1: StateVector[StackedBasisWithVolumeLike[Any, Any, Any]],
    axes: tuple[int] = (0,),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the difference between two eigenstates in k.

    Parameters
    ----------
    state_0 : StateVector[_B0Inv]
    state_1 : StateVector[_B1Inv]
    idx : SingleStackedIndexLike | None, optional
        index at each axis perpendicular to axis, by default None
    axis : int, optional
        axis to plot along, by default 0
    ax : Axes | None, optional
        plot axis, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    basis = stacked_basis_as_fundamental_momentum_basis(state_0["basis"])

    converted_0 = convert_state_vector_to_basis(state_0, basis)
    converted_1 = convert_state_vector_to_basis(state_1, basis)
    state: StateVector[Any] = {
        "basis": basis,
        "data": (converted_0["data"] - converted_1["data"])
        / np.max([np.abs(converted_0["data"]), np.abs(converted_1["data"])], axis=0),
    }
    return plot_state_1d_k(state, axes, idx, ax=ax, measure=measure, scale=scale)


def plot_state_difference_2d_x(
    state_0: StateVector[StackedBasisWithVolumeLike[Any, Any, Any]],
    state_1: StateVector[StackedBasisWithVolumeLike[Any, Any, Any]],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the difference between two eigenstates in 2d, perpendicular to z_axis.

    Parameters
    ----------
    state_0 : StateVector[_B0Inv]
    state_1 : StateVector[_B1Inv]
    idx : SingleStackedIndexLike
        index along each axis perpendicular to axes
    axes : tuple[int, int], optional
        axis to plot, by default (0, 1)
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    basis = stacked_basis_as_fundamental_position_basis(state_0["basis"])

    converted_0 = convert_state_vector_to_basis(state_0, basis)
    converted_1 = convert_state_vector_to_basis(state_1, basis)
    state: StateVector[Any] = {
        "basis": basis,
        "data": (converted_0["data"] - converted_1["data"])
        / np.max([np.abs(converted_0["data"]), np.abs(converted_1["data"])], axis=0),
    }
    return plot_state_2d_x(state, axes, idx, ax=ax, measure=measure, scale=scale)


def animate_state_3d_x(
    state: StateVector[StackedBasisWithVolumeLike[Any, Any, Any]],
    axes: tuple[int, int, int] = (0, 1, 2),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
    clim: tuple[float | None, float | None] = (None, None),
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Animate a state in 3d, perpendicular to z_axis.

    Parameters
    ----------
    state : Eigenstate[_B3d0Inv]
    z_axis : Literal[0, 1, 2, -1, -2, -3]
        axis perpendicular to which to plot
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    converted = convert_state_vector_to_position_basis(state)
    util = BasisUtil(converted["basis"])
    points = converted["data"].reshape(*util.shape)

    return animate_data_through_surface_x(
        converted["basis"],
        points,
        axes,
        idx,
        ax=ax,
        scale=scale,
        measure=measure,
        clim=clim,
    )


def plot_state_along_path(
    state: StateVector[StackedBasisWithVolumeLike[Any, Any, Any]],
    path: np.ndarray[tuple[int, int], np.dtype[np.int_]],
    *,
    wrap_distances: bool = False,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot an state in 1d along the given path in position basis.

    Parameters
    ----------
    state : Eigenstate[_B3d0Inv]
    path : np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]]
        path, as a list of [x0_coords, x1_coords, x2_coords]
    wrap_distances : bool, optional
        should the coordinates be wrapped into the unit cell, by default False
    ax : Axes | None, optional
        plot axis, by default None
    measure : Literal[&quot;real&quot;, &quot;imag&quot;, &quot;abs&quot;], optional
        measure, by default "abs"
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = get_figure(ax)
    converted = convert_state_vector_to_position_basis(state)  # type: ignore[var-annotated,arg-type]

    util = BasisUtil(converted["basis"])
    points = converted["data"].reshape(*util.shape)[*path]
    data = get_measured_data(points, measure)
    distances = calculate_cumulative_x_distances_along_path(
        converted["basis"],
        path,
        wrap_distances=wrap_distances,  # type: ignore[arg-type]
    )
    (line,) = ax.plot(distances, data)
    ax.set_yscale(scale)
    ax.set_xlabel("distance /m")
    return fig, ax, line


def _get_band_occupation(
    hamiltonian: SingleBasisOperator[_B0Inv],
    states: StateVectorList[BasisLike[Any, Any], _B0Inv],
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.float64]],
    Operator[BasisLike[Any, Any], FundamentalBasis[int]],
]:
    eigenstates = calculate_eigenvectors_hermitian(hamiltonian)
    energies = eigenstates["eigenvalue"].astype(np.float64)
    energies -= np.min(energies)
    occupations = calculate_inner_products(states, eigenstates)
    return (energies, occupations)


def plot_all_band_occupations(
    hamiltonian: SingleBasisOperator[_B0Inv],
    states: StateVectorList[BasisLike[Any, Any], _B0Inv],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes]:
    """
    Plot the occupation of each state against energy.

    Parameters
    ----------
    hamiltonian : SingleBasisOperator[_B0Inv]
    states : StateVectorList[BasisLike[Any, Any], _B0Inv]
    ax : Axes | None, optional
        axis, by default None
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes]
    """
    fig, ax = get_figure(ax)

    energies, occupations = _get_band_occupation(hamiltonian, states)

    n_states = states["basis"][0].n
    for i, occupation in enumerate(occupations["data"].reshape(n_states, -1)):
        measured = np.abs(occupation) ** 2
        (line,) = ax.plot(energies, measured)
        line.set_label(f"state {i} occupation")

    ax.set_yscale(scale)
    ax.set_xlabel("Occupation")
    ax.set_xlabel("Energy /J")
    ax.set_title("Plot of Occupation against Energy")

    return fig, ax


def animate_all_band_occupations(
    hamiltonian: SingleBasisOperator[_B0Inv],
    states: StateVectorList[BasisLike[Any, Any], _B0Inv],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Animate the occupation of each state against energy.

    Parameters
    ----------
    hamiltonian : SingleBasisOperator[_B0Inv]
    states : StateVectorList[BasisLike[Any, Any], _B0Inv]
    ax : Axes | None, optional
        axis, by default None
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]

    """
    fig, ax = get_figure(ax)

    energies, occupations = _get_band_occupation(hamiltonian, states)

    frames: list[list[Line2D]] = []
    n_states = states["basis"][0].n
    for i, occupation in enumerate(occupations["data"].reshape(n_states, -1)):
        measured = np.abs(occupation) ** 2
        (line,) = ax.plot(energies, measured)
        line.set_label(f"state {i} occupation")
        frames.append([line])
        line.set_color(frames[0][0].get_color())

    ani = ArtistAnimation(fig, frames)
    ax.set_yscale(scale)
    ax.set_xlabel("Occupation")
    ax.set_xlabel("Energy /J")
    ax.set_title("Plot of Occupation against Energy")

    return fig, ax, ani


def plot_band_occupation(
    hamiltonian: SingleBasisOperator[_B0Inv],
    state: StateVector[_B0Inv],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes]:
    """
    Plot the occupation of the state against energy.

    Parameters
    ----------
    hamiltonian : SingleBasisOperator[_B0Inv]
    state : StateVector[_B0Inv]
    ax : Axes | None, optional
        axis, by default None
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    return plot_all_band_occupations(
        hamiltonian, as_state_vector_list([state]), ax=ax, scale=scale
    )


def plot_average_band_occupation(
    hamiltonian: SingleBasisOperator[_B0Inv],
    states: StateVectorList[BasisLike[Any, Any], _B0Inv],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the occupation of the state against energy.

    Parameters
    ----------
    hamiltonian : SingleBasisOperator[_B0Inv]
    state : StateVector[_B0Inv]
    ax : Axes | None, optional
        axis, by default None
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = get_figure(ax)

    energies, occupations = _get_band_occupation(hamiltonian, states)

    n_states = states["basis"][0].n
    occupations["data"].reshape(n_states, -1)
    probabilities = np.abs(occupations["data"].reshape(n_states, -1)) ** 2
    average = np.average(probabilities, axis=0)

    (line,) = ax.plot(energies, average)

    ax.set_yscale(scale)
    ax.set_xlabel("Occupation")
    ax.set_xlabel("Energy /J")

    return fig, ax, line


_SB0 = TypeVar("_SB0", bound=StackedBasisWithVolumeLike[Any, Any, Any])
_B0 = TypeVar("_B0", bound=BasisLike[Any, Any])


def get_periodic_x_operator(
    basis: StackedBasisWithVolumeLike[Any, Any, Any],
    direction: tuple[int, ...] | None = None,
) -> SingleBasisOperator[
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]
]:
    """
    Generate operator for e^(2npi*x / delta_x).

    Parameters
    ----------
    basis : _SB0

    Returns
    -------
    SingleBasisOperator[_SB0]
    """
    direction = tuple(1 for _ in range(basis.ndim)) if direction is None else direction
    basis_x = stacked_basis_as_fundamental_position_basis(basis)
    util = BasisUtil(basis_x)
    dk = tuple(n / f for (n, f) in zip(direction, util.shape))

    phi = (2 * np.pi) * np.einsum(
        "ij,i->j",
        util.stacked_nx_points,
        dk,
    )
    return as_operator(
        {"basis": TupleBasis(basis_x, basis_x), "data": np.exp(1j * phi)}
    )


def _get_periodic_x(
    states: StateVectorList[
        _B0Inv,
        StackedBasisWithVolumeLike[Any, Any, Any],
    ],
    direction: tuple[int, ...] | None = None,
) -> ValueList[_B0Inv]:
    """
    Calculate expectation of e^(2pi*x / delta_x).

    Parameters
    ----------
    basis : _SB0

    Returns
    -------
    SingleBasisOperator[_SB0]
    """
    operator = get_periodic_x_operator(states["basis"][1], direction)
    return calculate_expectation_list(operator, states)


_BT0 = TypeVar("_BT0", bound=BasisWithTimeLike[Any, Any])


def _get_restored_x(
    states: StateVectorList[
        TupleBasisLike[Any, _BT0],
        StackedBasisWithVolumeLike[Any, Any, Any],
    ],
    axis: int,
) -> ValueList[TupleBasisLike[Any, _BT0]]:
    direction = tuple(1 if i == axis else 0 for i in range(states["basis"][1].ndim))
    periodic_x = _get_periodic_x(states, direction)
    unravelled = np.unwrap(
        np.angle(periodic_x["data"].reshape(states["basis"][0].shape)), axis=1
    )

    return {
        "basis": periodic_x["basis"],
        "data": (
            unravelled * states["basis"][1].delta_x_stacked[axis] / (2 * np.pi)
        ).ravel(),
    }


def plot_periodic_averaged_occupation_1d_x(
    states: StateVectorList[
        TupleBasisLike[Any, _BT0],
        StackedBasisWithVolumeLike[Any, Any, Any],
    ],
    axes: tuple[int] = (0,),
    *,
    ax: Axes | None = None,
    measure: Measure = "real",
) -> tuple[Figure, Axes]:
    """
    Plot the max occupation against time in 1d for each trajectory against time.

    Parameters
    ----------
    states : StateVectorList[ TupleBasisLike[Any, _BT0], TupleBasisLike[_
    axes : tuple[int], optional
        direction to plot along, by default (0,)
    ax : Axes | None, optional
        plot axis, by default None
    unravel : bool, optional
        should the trajectories be unravelled, by default False

    Returns
    -------
    tuple[Figure, Axes]
    """
    occupation_x = _get_restored_x(states, axes[0])
    fig, ax = plot_all_value_list_against_time(occupation_x, ax=ax, measure=measure)
    ax.set_ylabel("Distance /m")
    return fig, ax


def _get_x_operator(basis: _SB0, axis: int) -> SingleBasisOperator[_SB0]:
    """
    Generate operator for e^(2pi*x / delta_x).

    Parameters
    ----------
    basis : _SB0

    Returns
    -------
    SingleBasisOperator[_SB0]
    """
    basis_x = stacked_basis_as_fundamental_position_basis(basis)
    util = BasisUtil(basis_x)
    return convert_diagonal_operator_to_basis(
        {
            "basis": TupleBasis(basis_x, basis_x),
            "data": util.dx_stacked[axis] * util.stacked_nx_points[axis],
        },
        TupleBasis(basis, basis),
    )


def _get_average_x(
    states: StateVectorList[
        _B0Inv,
        StackedBasisWithVolumeLike[Any, Any, Any],
    ],
    axis: int,
) -> ValueList[_B0Inv]:
    """
    Calculate expectation of e^(2pi*x / delta_x).

    Parameters
    ----------
    basis : _SB0

    Returns
    -------
    SingleBasisOperator[_SB0]
    """
    operator = _get_x_operator(states["basis"][1], axis)

    return calculate_expectation_list(operator, states)


def plot_averaged_occupation_1d_x(
    states: StateVectorList[
        TupleBasisLike[Any, _BT0],
        StackedBasisWithVolumeLike[Any, Any, Any],
    ],
    axes: tuple[int] = (0,),
    *,
    ax: Axes | None = None,
    measure: Measure = "real",
) -> tuple[Figure, Axes]:
    """
    Plot the max occupation against time in 1d for each trajectory against time.

    Parameters
    ----------
    states : StateVectorList[ TupleBasisLike[Any, _BT0], TupleBasisLike[_
    axes : tuple[int], optional
        direction to plot along, by default (0,)
    ax : Axes | None, optional
        plot axis, by default None
    unravel : bool, optional
        should the trajectories be unravelled, by default False

    Returns
    -------
    tuple[Figure, Axes]
    """
    occupation_x = _get_average_x(states, axes[0])
    fig, ax = plot_all_value_list_against_time(occupation_x, ax=ax, measure=measure)
    ax.set_ylabel("Distance /m")
    return fig, ax


def _get_x_spread(
    states: StateVectorList[
        TupleBasisLike[Any, _BT0],
        StackedBasisWithVolumeLike[Any, Any, Any],
    ],
    axis: int,
) -> ValueList[TupleBasisLike[Any, _BT0]]:
    r"""
    Calculate the spread, \sigma_0 using the periodic x operator.

    For a gaussian wavepacket

    \ket{\psi} = A \exp{(-\frac{{(x - x_0)}^2}{2 \sigma_0} + ik_0(x-x_0))} \ket{x}

    the expectation is given by

    \braket{e^{iqx}} = e^{iq.x_0}\exp{(-\sigma_0^2q^2 / 4)}

    Parameters
    ----------
    states : StateVectorList[TupleBasisLike[Any, _BT0], StackedBasisWithVolumeLike[Any, Any, Any]]
    axis : int

    Returns
    -------
    ValueList[TupleBasisLike[Any, _BT0]]
    """
    direction = tuple(1 if i == axis else 0 for i in range(states["basis"][1].ndim))

    data = states["data"].reshape(states["basis"].shape)
    data /= np.linalg.norm(data, axis=1)[:, np.newaxis]

    states["data"] = data.ravel()
    periodic_x = _get_periodic_x(states, direction)
    norm = np.abs(periodic_x["data"].reshape(states["basis"][0].shape))
    q = 2 * np.pi / np.linalg.norm(states["basis"][1].delta_x_stacked[axis])
    sigma_0 = np.sqrt(-(4 / q**2) * np.log(norm))

    return {"basis": periodic_x["basis"], "data": sigma_0.ravel()}


def plot_spread_1d(
    states: StateVectorList[
        TupleBasisLike[Any, _BT0],
        StackedBasisWithVolumeLike[Any, Any, Any],
    ],
    axes: tuple[int] = (0,),
    *,
    ax: Axes | None = None,
    measure: Measure = "real",
) -> tuple[Figure, Axes]:
    """
    Plot the change in sigma_0 over time.

    Parameters
    ----------
    states : StateVectorList[
        TupleBasisLike[Any, _BT0],
        StackedBasisWithVolumeLike[Any, Any, Any],
    ]
    axes : tuple[int], optional
        direction to plot along, by default (0,)
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes]
    """
    spread_x = _get_x_spread(states, axes[0])
    fig, ax = plot_all_value_list_against_time(spread_x, ax=ax, measure=measure)

    ax.set_ylabel("Distance /m")
    return fig, ax


def plot_spread_distribution_1d(
    states: StateVectorList[
        TupleBasisLike[Any, _BT0],
        StackedBasisWithVolumeLike[Any, Any, Any],
    ],
    axes: tuple[int] = (0,),
    *,
    ax: Axes | None = None,
    measure: Measure = "real",
) -> tuple[Figure, Axes]:
    """
    Plot the distribution of sigma_0.

    Parameters
    ----------
    states : StateVectorList[
        TupleBasisLike[Any, _BT0],
        StackedBasisWithVolumeLike[Any, Any, Any],
    ]
    axes : tuple[int], optional
        direction to plot along, by default (0,)
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes]
    """
    spread_x = _get_x_spread(states, axes[0])
    fig, ax = plot_value_list_distribution(spread_x, ax=ax, measure=measure)

    ax.set_xlabel("Distance /m")
    return fig, ax


def _get_k_operator(basis: _SB0, axis: int) -> SingleBasisOperator[_SB0]:
    """
    Generate operator for e^(2pi*x / delta_x).

    Parameters
    ----------
    basis : _SB0

    Returns
    -------
    SingleBasisOperator[_SB0]
    """
    basis_k = stacked_basis_as_fundamental_momentum_basis(basis)
    util = BasisUtil(basis_k)
    return convert_diagonal_operator_to_basis(
        {
            "basis": TupleBasis(basis_k, basis_k),
            "data": util.dk_stacked[axis] * util.stacked_nk_points[axis],
        },
        TupleBasis(basis, basis),
    )


def _get_average_k(
    states: StateVectorList[
        _B0Inv,
        StackedBasisWithVolumeLike[Any, Any, Any],
    ],
    axis: int,
) -> ValueList[_B0Inv]:
    """
    Calculate expectation of e^(2pi*x / delta_x).

    Parameters
    ----------
    basis : _SB0

    Returns
    -------
    SingleBasisOperator[_SB0]
    """
    operator = _get_k_operator(states["basis"][1], axis)
    return calculate_expectation_list(operator, states)


def plot_spread_against_k(
    states: StateVectorList[
        TupleBasisLike[Any, _BT0],
        StackedBasisWithVolumeLike[Any, Any, Any],
    ],
    axes: tuple[int] = (0,),
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot the distribution of sigma_0.

    Parameters
    ----------
    states : StateVectorList[
        TupleBasisLike[Any, _BT0],
        StackedBasisWithVolumeLike[Any, Any, Any],
    ]
    axes : tuple[int], optional
        direction to plot along, by default (0,)
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes]
    """
    fig, ax = get_figure(ax)

    spread_x = _get_x_spread(states, axes[0])
    k = _get_average_k(states, axes[0])

    ax.plot(k["data"], spread_x["data"])

    ax.set_xlabel("Momentum /$m^{-1}$")
    ax.set_ylabel("Spread /m")
    return fig, ax


def plot_spread_against_x(
    states: StateVectorList[
        TupleBasisLike[Any, _BT0],
        StackedBasisWithVolumeLike[Any, Any, Any],
    ],
    axes: tuple[int] = (0,),
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot the distribution of sigma_0.

    Parameters
    ----------
    states : StateVectorList[
        TupleBasisLike[Any, _BT0],
        StackedBasisWithVolumeLike[Any, Any, Any],
    ]
    axes : tuple[int], optional
        direction to plot along, by default (0,)
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes]
    """
    fig, ax = get_figure(ax)

    spread_x = _get_x_spread(states, axes[0])
    x = _get_average_x(states, axes[0])

    ax.plot(x["data"], spread_x["data"])

    ax.set_xlabel("Displacement /m")
    ax.set_ylabel("Spread /m")
    return fig, ax


def plot_k_distribution_1d(
    states: StateVectorList[
        TupleBasisLike[Any, _BT0],
        StackedBasisWithVolumeLike[Any, Any, Any],
    ],
    axes: tuple[int] = (0,),
    *,
    ax: Axes | None = None,
    measure: Measure = "real",
) -> tuple[Figure, Axes]:
    """
    Plot the distribution of sigma_0.

    Parameters
    ----------
    states : StateVectorList[
        TupleBasisLike[Any, _BT0],
        StackedBasisWithVolumeLike[Any, Any, Any],
    ]
    axes : tuple[int], optional
        direction to plot along, by default (0,)
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes]
    """
    k_values = _get_average_k(states, axes[0])
    fig, ax = plot_value_list_distribution(k_values, ax=ax, measure=measure)

    ax.set_xlabel("Momentum /$m^{-1}$")
    return fig, ax


def plot_x_distribution_1d(
    states: StateVectorList[
        TupleBasisLike[Any, _BT0],
        StackedBasisWithVolumeLike[Any, Any, Any],
    ],
    axes: tuple[int] = (0,),
    *,
    ax: Axes | None = None,
    measure: Measure = "real",
) -> tuple[Figure, Axes]:
    """
    Plot the distribution of sigma_0.

    Parameters
    ----------
    states : StateVectorList[
        TupleBasisLike[Any, _BT0],
        StackedBasisWithVolumeLike[Any, Any, Any],
    ]
    axes : tuple[int], optional
        direction to plot along, by default (0,)
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes]
    """
    x_values = _get_average_x(states, axes[0])
    fig, ax = plot_value_list_distribution(
        x_values, ax=ax, measure=measure, plot_gaussian=False
    )

    ax.set_xlabel("Displacement /m$")
    return fig, ax


def _get_average_displacements(
    positions: ValueList[TupleBasisLike[_B0Inv, _BT0]],
) -> ValueList[TupleBasisLike[_B0Inv, EvenlySpacedTimeBasis[Any, Any, Any]]]:
    basis = positions["basis"]
    stacked = positions["data"].reshape(basis.shape)
    squared_positions = np.square(stacked)
    total = np.cumsum(squared_positions + squared_positions[:, ::-1], axis=1)[:, ::-1]

    convolution = np.apply_along_axis(
        lambda m: scipy.signal.correlate(m, m, mode="full")[basis.shape[1] - 1 :],
        axis=1,
        arr=stacked,
    ).astype(np.float64)

    squared_diff = (total - 2 * convolution) / (1 + np.arange(basis[1].n))[::-1]
    out_basis = EvenlySpacedTimeBasis(basis[1].n, 1, 0, basis[1].dt * (basis[1].n))
    return {"basis": TupleBasis(basis[0], out_basis), "data": squared_diff.ravel()}


def plot_average_displacement_1d_x(
    states: StateVectorList[
        TupleBasisLike[Any, _BT0],
        StackedBasisWithVolumeLike[Any, Any, Any],
    ],
    axes: tuple[int] = (0,),
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the average displacement in 1d.

    Parameters
    ----------
    states : StateVectorList[ TupleBasisLike[_B0Inv, _BT0], TupleBasisLike[
    axes : tuple[int], optional
        plot axes, by default (0,)
    ax : Axes | None, optional
        ax, by default None
    measure : Measure, optional
        measure, by default "abs"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    restored_x = _get_restored_x(states, axes[0])
    displacements = _get_average_displacements(restored_x)

    return plot_average_value_list_against_time(
        displacements, ax=ax, measure=measure, scale=scale
    )
