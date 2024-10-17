from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
import scipy  # type: ignore unknown
import scipy.signal  # type: ignore unknown
from matplotlib.animation import ArtistAnimation
from slate.basis.stacked._tuple_basis import VariadicTupleBasis

from surface_potential_analysis.basis.legacy import (
    BasisWithTimeLike,
    EvenlySpacedTimeBasis,
    FundamentalBasis,
    StackedBasisWithVolumeLike,
    TupleBasis,
    TupleBasisLike,
)
from surface_potential_analysis.basis.util import (
    BasisUtil,
)
from surface_potential_analysis.operator.conversion import (
    convert_diagonal_operator_to_basis,
)
from surface_potential_analysis.stacked_basis.conversion import (
    tuple_basis_as_fundamental,
)
from surface_potential_analysis.stacked_basis.util import (
    calculate_cumulative_x_distances_along_path,
)
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_list_to_basis,
    convert_state_vector_to_basis,
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
    animate_data_through_list_2d_k,
    animate_data_through_list_2d_x,
    animate_data_through_surface_x,
    get_figure,
    plot_data_1d,
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

    from surface_potential_analysis.basis.legacy import (
        BasisLike,
    )
    from surface_potential_analysis.operator.operator import (
        Operator,
        SingleBasisDiagonalOperator,
        SingleBasisOperator,
    )
    from surface_potential_analysis.state_vector.eigenstate_list import ValueList
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )
    from surface_potential_analysis.types import (
        SingleStackedIndexLike,
    )
    from surface_potential_analysis.util.plot import Scale
    from surface_potential_analysis.wavepacket.get_eigenstate import BlochBasis

    _B0Inv = TypeVar("_B0Inv", bound=BasisLike)
    _SBV0 = TypeVar("_SBV0", bound=StackedBasisWithVolumeLike)
    _SBV1 = TypeVar("_SBV1", bound=StackedBasisWithVolumeLike)

    _B0 = TypeVar("_B0", bound=BasisLike)
    _B1 = TypeVar("_B1", bound=BasisLike)


# ruff: noqa: PLR0913
def plot_state_1d_k(
    state: StateVector[_SBV0],
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
    fig, ax, line = plot_data_1d_k(
        state["basis"],
        state["data"],
        axes,
        idx,
        ax=ax,
        scale=scale,
        measure=measure,
    )
    ax.set_ylabel("State /Au")  # type: ignore unknown
    return fig, ax, line


def plot_state_1d_x(
    state: StateVector[_SBV0],
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
    fig, ax, line = plot_data_1d_x(
        state["basis"],
        state["data"],
        axes,
        idx,
        ax=ax,
        scale=scale,
        measure=measure,
    )
    ax.set_ylabel("State /Au")  # type: ignore unknown
    line.set_label(f"{measure} state")
    return fig, ax, line


def animate_state_over_list_1d_x(
    states: StateVectorList[_B0, _SBV0],
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
    states : StateVectorList[BasisLike, TupleBasisLike[*tuple[Any, ...]]]
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
        states, tuple_basis_as_fundamental(states["basis"][1])
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
    ax.set_ylabel("State /Au")  # type: ignore unknown
    return fig, ax, ani


def animate_state_over_list_2d_k(
    states: StateVectorList[_B0, _SBV0],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Plot an state in 2d along the given axis in momentum space, over time.

    Parameters
    ----------
    states : StateVectorList[BasisLike, TupleBasisLike[*tuple[Any, ...]]]
    axes : tuple[int, int, int], optional
        axes to plot in, by default (0, 1)
    idx : SingleStackedIndexLike | None, optional
        index to plot, by default None
    ax : Axes | None, optional
        plot axis, by default None
    scale : Scale, optional
        scale, by default "linear"
    measure : Measure, optional
        measure, by default "abs"

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    converted = convert_state_vector_list_to_basis(
        states, stacked_basis_as_transformed_basis(states["basis"][1])
    )

    fig, ax, ani = animate_data_through_list_2d_k(
        converted["basis"][1],
        converted["data"].reshape(converted["basis"].shape),
        axes,
        idx,
        ax=ax,
        scale=scale,
        measure=measure,
    )
    ax.set_ylabel("State /Au")  # type: ignore unknown
    return fig, ax, ani


def animate_state_over_list_2d_x(
    states: StateVectorList[_B0, _SBV0],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Plot an state in 2d along the given axis in position space, over time.

    Parameters
    ----------
    states : StateVectorList[BasisLike, TupleBasisLike[*tuple[Any, ...]]]
    axes : tuple[int, int, int], optional
        axes to plot in, by default (0, 1)
    idx : SingleStackedIndexLike | None, optional
        index to plot, by default None
    ax : Axes | None, optional
        plot axis, by default None
    scale : Scale, optional
        scale, by default "linear"
    measure : Measure, optional
        measure, by default "abs"

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    converted = convert_state_vector_list_to_basis(
        states, tuple_basis_as_fundamental(states["basis"][1])
    )

    return animate_data_through_list_2d_x(
        converted["basis"][1],
        converted["data"].reshape(converted["basis"].shape),
        axes,
        idx,
        ax=ax,
        scale=scale,
        measure=measure,
    )


def animate_state_over_list_1d_k(
    states: StateVectorList[_B0, _SBV0],
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
    states : StateVectorList[BasisLike, TupleBasisLike[*tuple[Any, ...]]]
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
        states, stacked_basis_as_transformed_basis(states["basis"][1])
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
    ax.set_ylabel("State /Au")  # type: ignore unknown
    return fig, ax, ani


def plot_state_2d_k(
    state: StateVector[_SBV0],
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
    return plot_data_2d_k(
        state["basis"],
        state["data"],
        axes,
        idx,
        ax=ax,
        scale=scale,
        measure=measure,
    )


def plot_state_difference_2d_k(
    state_0: StateVector[_SBV0],
    state_1: StateVector[_SBV1],
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
    basis = stacked_basis_as_transformed_basis(state_0["basis"])

    converted_0 = convert_state_vector_to_basis(state_0, basis)
    converted_1 = convert_state_vector_to_basis(state_1, basis)
    state: StateVector[Any] = {
        "basis": basis,
        "data": (converted_0["data"] - converted_1["data"])
        / np.max([np.abs(converted_0["data"]), np.abs(converted_1["data"])], axis=0),
    }
    return plot_state_2d_k(state, axes, idx, ax=ax, measure=measure, scale=scale)


def plot_state_2d_x(
    state: StateVector[_SBV0],
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
    return plot_data_2d_x(
        state["basis"],
        state["data"],
        axes,
        idx,
        ax=ax,
        scale=scale,
        measure=measure,
    )


def plot_state_difference_1d_k(
    state_0: StateVector[_SBV0],
    state_1: StateVector[_SBV1],
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
    basis = stacked_basis_as_transformed_basis(state_0["basis"])

    converted_0 = convert_state_vector_to_basis(state_0, basis)
    converted_1 = convert_state_vector_to_basis(state_1, basis)
    state: StateVector[Any] = {
        "basis": basis,
        "data": (converted_0["data"] - converted_1["data"])
        / np.max([np.abs(converted_0["data"]), np.abs(converted_1["data"])], axis=0),
    }
    return plot_state_1d_k(state, axes, idx, ax=ax, measure=measure, scale=scale)


def plot_state_difference_2d_x(
    state_0: StateVector[_SBV0],
    state_1: StateVector[_SBV1],
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
    basis = tuple_basis_as_fundamental(state_0["basis"])

    converted_0 = convert_state_vector_to_basis(state_0, basis)
    converted_1 = convert_state_vector_to_basis(state_1, basis)
    state: StateVector[Any] = {
        "basis": basis,
        "data": (converted_0["data"] - converted_1["data"])
        / np.max([np.abs(converted_0["data"]), np.abs(converted_1["data"])], axis=0),
    }
    return plot_state_2d_x(state, axes, idx, ax=ax, measure=measure, scale=scale)


def animate_state_3d_x(
    state: StateVector[_SBV0],
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
    state: StateVector[_SBV0],
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
    (line,) = ax.plot(distances, data)  # type: ignore unknown
    ax.set_yscale(scale)  # type: ignore unknown
    ax.set_xlabel("distance /m")  # type: ignore unknown
    return fig, ax, line


def _get_eigenstate_occupation(
    hamiltonian: SingleBasisOperator[_B0Inv],
    states: StateVectorList[BasisLike, _B0Inv],
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.float64]],
    Operator[BasisLike, FundamentalBasis[BasisMetadata]],
]:
    eigenstates = calculate_eigenvectors_hermitian(hamiltonian)
    energies = eigenstates["eigenvalue"].astype(np.float64)
    energies -= np.min(energies)
    occupations = calculate_inner_products(states, eigenstates)
    return (energies, occupations)


def plot_all_eigenstate_occupations(
    hamiltonian: SingleBasisOperator[_B0Inv],
    states: StateVectorList[BasisLike, _B0Inv],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes]:
    """
    Plot the occupation of each state against energy.

    Parameters
    ----------
    hamiltonian : SingleBasisOperator[_B0Inv]
    states : StateVectorList[BasisLike, _B0Inv]
    ax : Axes | None, optional
        axis, by default None
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes]
    """
    fig, ax = get_figure(ax)

    energies, occupations = _get_eigenstate_occupation(hamiltonian, states)

    n_states = states["basis"][0].size
    for i, occupation in enumerate(occupations["data"].reshape(n_states, -1)):
        measured = np.abs(occupation) ** 2
        (line,) = ax.plot(energies, measured)  # type: ignore unknown
        line.set_label(f"state {i} occupation")

    ax.set_yscale(scale)  # type: ignore unknown
    ax.set_xlabel("Occupation")  # type: ignore unknown
    ax.set_xlabel("Energy /J")  # type: ignore unknown
    ax.set_title("Plot of Occupation against Energy")  # type: ignore unknown

    return fig, ax


def animate_all_eigenstate_occupations(
    hamiltonian: SingleBasisOperator[_B0Inv],
    states: StateVectorList[BasisLike, _B0Inv],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Animate the occupation of each state against energy.

    Parameters
    ----------
    hamiltonian : SingleBasisOperator[_B0Inv]
    states : StateVectorList[BasisLike, _B0Inv]
    ax : Axes | None, optional
        axis, by default None
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]

    """
    fig, ax = get_figure(ax)

    energies, occupations = _get_eigenstate_occupation(hamiltonian, states)

    frames: list[list[Line2D]] = []
    n_states = states["basis"][0].size
    for i, occupation in enumerate(occupations["data"].reshape(n_states, -1)):
        measured = np.abs(occupation) ** 2
        (line,) = ax.plot(energies, measured)  # type: ignore unknown
        line.set_label(f"state {i} occupation")
        frames.append([line])
        line.set_color(frames[0][0].get_color())

    ani = ArtistAnimation(fig, frames)
    ax.set_yscale(scale)  # type: ignore unknown
    ax.set_xlabel("Occupation")  # type: ignore unknown
    ax.set_xlabel("Energy /J")  # type: ignore unknown
    ax.set_title("Plot of Occupation against Energy")  # type: ignore unknown

    return fig, ax, ani


def plot_eigenstate_occupation(
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
    return plot_all_eigenstate_occupations(
        hamiltonian, as_state_vector_list([state]), ax=ax, scale=scale
    )


def plot_average_eigenstate_occupation(
    hamiltonian: SingleBasisOperator[_B0Inv],
    states: StateVectorList[BasisLike, _B0Inv],
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

    energies, occupations = _get_eigenstate_occupation(hamiltonian, states)

    n_states = states["basis"][0].size
    occupations["data"].reshape(n_states, -1)
    probabilities = np.abs(occupations["data"].reshape(n_states, -1)) ** 2
    average = np.average(probabilities, axis=0)

    (line,) = ax.plot(energies, average)  # type: ignore unknown

    ax.set_yscale(scale)  # type: ignore unknown
    ax.set_xlabel("Occupation")  # type: ignore unknown
    ax.set_xlabel("Energy /J")  # type: ignore unknown

    return fig, ax, line


def get_average_band_energy(
    hamiltonian: SingleBasisDiagonalOperator[BlochBasis[_B0]],
) -> ValueList[_B0]:
    """
    Get the average energy of each band.

    Parameters
    ----------
    hamiltonian : SingleBasisDiagonalOperator[BlochBasis[_B0]]

    Returns
    -------
    ValueList[_B0]
    """
    basis = hamiltonian["basis"][1].wavefunctions["basis"][0][0]

    hamiltonian_data = np.real(hamiltonian["data"].reshape(basis.n, -1))
    band_energies = np.average(hamiltonian_data, axis=1)
    return {"basis": basis, "data": band_energies}


def plot_total_band_occupation_against_energy(
    hamiltonian: SingleBasisDiagonalOperator[BlochBasis[_B1]],
    state: StateVector[_B0],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the occupation of the state in each band against energy.

    Parameters
    ----------
    hamiltonian : SingleBasisDiagonalOperator[_ESB0]
        The hamiltonian in the explicit basis, stored as a list of states (bands, bloch k)
    state : StateVector[_B0Inv]
    ax : Axes | None, optional
        axis, by default None
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    band_energies = get_average_band_energy(hamiltonian)["data"].astype(np.float64)

    converted = convert_state_vector_to_basis(state, hamiltonian["basis"][1])

    n_bands = hamiltonian["basis"][1].wavefunctions["basis"][0][0].n
    occupation = np.square(np.abs(converted["data"])).reshape(n_bands, -1)

    total_band_occupation = np.asarray(np.sum(occupation, axis=1))
    fig, ax, line = plot_data_1d(
        total_band_occupation,
        band_energies,
        ax=ax,
        scale=scale,
        measure=measure,
    )
    ax.set_xlabel("Average Band Energy /J")  # type: ignore unknown
    ax.set_ylabel("Total Band Occupation")  # type: ignore unknown

    average_energy = np.average(band_energies, weights=total_band_occupation)
    average_line = ax.axvline(average_energy)  # type: ignore unknown
    average_line.set_color(line.get_color())
    average_line.set_linestyle("--")

    return fig, ax, line


def get_periodic_x_operator(
    basis: _SBV0,
    direction: tuple[int, ...] | None = None,
) -> SingleBasisOperator[_SBV0]:
    """
    Generate operator for e^(2npi*x / delta_x).

    Parameters
    ----------
    basis : _SBV0

    Returns
    -------
    SingleBasisOperator[_SBV0]
    """
    direction = tuple(1 for _ in range(basis.n_dim)) if direction is None else direction
    basis_x = tuple_basis_as_fundamental(basis)
    util = BasisUtil(basis_x)
    dk = tuple(n / f for (n, f) in zip(direction, util.shape))

    phi = (2 * np.pi) * np.einsum(  # type: ignore unknown
        "ij,i->j",
        util.stacked_nx_points,
        dk,
    )
    return convert_diagonal_operator_to_basis(
        {
            "basis": VariadicTupleBasis((basis_x, basis_x), None),
            "data": np.exp(1j * phi),
        },
        VariadicTupleBasis((basis, basis), None),
    )


def _get_periodic_x(
    states: StateVectorList[
        _B0Inv,
        StackedBasisWithVolumeLike,
    ],
    direction: tuple[int, ...] | None = None,
) -> ValueList[_B0Inv]:
    """
    Calculate expectation of e^(2pi*x / delta_x).

    Parameters
    ----------
    basis : _SBV0

    Returns
    -------
    SingleBasisOperator[_SBV0]
    """
    operator = get_periodic_x_operator(states["basis"][1], direction)
    return calculate_expectation_list(operator, states)


_BT0 = TypeVar("_BT0", bound=BasisWithTimeLike)


def _get_average_x_periodic(
    states: StateVectorList[
        _B0Inv,
        StackedBasisWithVolumeLike,
    ],
    axis: int,
) -> ValueList[_B0Inv]:
    direction = tuple(1 if i == axis else 0 for i in range(states["basis"][1].sizedim))
    periodic_x = _get_periodic_x(states, direction)
    angle = np.angle(periodic_x["data"]) % (2 * np.pi)

    return {
        "basis": periodic_x["basis"],
        "data": (angle * states["basis"][1].delta_x_stacked[axis] / (2 * np.pi)),
    }


def _get_restored_x(
    states: StateVectorList[
        TupleBasisLike[Any, _BT0],
        StackedBasisWithVolumeLike,
    ],
    axis: int,
) -> ValueList[TupleBasisLike[Any, _BT0]]:
    direction = tuple(1 if i == axis else 0 for i in range(states["basis"][1].sizedim))
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
        StackedBasisWithVolumeLike,
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
    ax.set_ylabel("Distance /m")  # type: ignore unknown
    return fig, ax


def _get_x_operator(basis: _SBV0, axis: int) -> SingleBasisOperator[_SBV0]:
    """
    Generate operator for x.

    Parameters
    ----------
    basis : _SBV0

    Returns
    -------
    SingleBasisOperator[_SBV0]
    """
    basis_x = tuple_basis_as_fundamental(basis)
    util = BasisUtil(basis_x)
    return convert_diagonal_operator_to_basis(
        {
            "basis": VariadicTupleBasis((basis_x, basis_x), None),
            "data": util.dx_stacked[axis] * util.stacked_nx_points[axis],
        },
        VariadicTupleBasis((basis, basis), None),
    )


def _get_average_x(
    states: StateVectorList[_B0Inv, _SBV0],
    axis: int,
) -> ValueList[_B0Inv]:
    """
    Calculate expectation of e^(2pi*x / delta_x).

    Parameters
    ----------
    basis : _SBV0

    Returns
    -------
    SingleBasisOperator[_SBV0]
    """
    operator = _get_x_operator(states["basis"][1], axis)
    return calculate_expectation_list(operator, states)


def plot_averaged_occupation_1d_x(
    states: StateVectorList[TupleBasisLike[Any, _BT0], _SBV0],
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
    ax.set_ylabel("Distance /m")  # type: ignore unknown
    return fig, ax


def _get_x_spread(
    states: StateVectorList[
        _B0Inv,
        StackedBasisWithVolumeLike,
    ],
    axis: int,
) -> ValueList[_B0Inv]:
    r"""
    Calculate the spread, \sigma_0 using the periodic x operator.

    For a gaussian wavepacket

    \ket{\psi} = A \exp{(-\frac{{(x - x_0)}^2}{2 \sigma_0} + ik_0(x-x_0))} \ket{x}

    the expectation is given by

    \braket{e^{iqx}} = e^{iq.x_0}\exp{(-\sigma_0^2q^2 / 4)}

    Parameters
    ----------
    states : StateVectorList[TupleBasisLike[Any, _BT0], StackedBasisWithVolumeLike]
    axis : int

    Returns
    -------
    ValueList[TupleBasisLike[Any, _BT0]]
    """
    direction = tuple(1 if i == axis else 0 for i in range(states["basis"][1].sizedim))

    data = states["data"].reshape(states["basis"].shape)
    data /= np.linalg.norm(data, axis=1)[:, np.newaxis]

    states["data"] = data.ravel()
    periodic_x = _get_periodic_x(states, direction)
    norm = np.abs(periodic_x["data"])
    q = 2 * np.pi / np.linalg.norm(states["basis"][1].delta_x_stacked[axis])
    sigma_0 = np.sqrt(-(4 / q**2) * np.log(norm))

    return {"basis": periodic_x["basis"], "data": sigma_0.ravel()}


def get_coherent_coordinates(
    states: StateVectorList[
        _B0Inv,
        StackedBasisWithVolumeLike,
    ],
    axis: int,
) -> ValueList[TupleBasisLike[FundamentalBasis[Literal[3]], _B0Inv]]:
    """Get the coherent wavepacket coordinates x0,k0,sigma0.

    Parameters
    ----------
    states : StateVectorList[ _B0Inv, StackedBasisWithVolumeLike, ]
    axis : int

    Returns
    -------
    ValueList[TupleBasisLike[FundamentalBasis[Literal[3]], _B0Inv]]
    """
    sigma_0 = _get_x_spread(states, axis)
    x0 = _get_average_x_periodic(states, axis)
    k0 = _get_average_k(states, axis)
    return {
        "basis": TupleBasis(
            FundamentalBasis[Literal[3]](3),
            states["basis"][0],
        ),
        "data": np.array([x0["data"], k0["data"], sigma_0["data"]]),
    }


def plot_spread_1d(
    states: StateVectorList[
        TupleBasisLike[Any, _BT0],
        StackedBasisWithVolumeLike,
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
        StackedBasisWithVolumeLike,
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

    ax.set_ylabel("Distance /m")  # type: ignore unknown
    return fig, ax


def plot_spread_distribution_1d(
    states: StateVectorList[
        TupleBasisLike[Any, _BT0],
        StackedBasisWithVolumeLike,
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
        StackedBasisWithVolumeLike,
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
    fig, ax = plot_value_list_distribution(
        spread_x, ax=ax, measure=measure, distribution="skew normal"
    )

    ax.set_xlabel("Distance /m")  # type: ignore unknown
    return fig, ax


def _get_k_operator(basis: _SBV0, axis: int) -> SingleBasisOperator[_SBV0]:
    """
    Generate operator for k.

    Parameters
    ----------
    basis : _SBV0

    Returns
    -------
    SingleBasisOperator[_SBV0]
    """
    basis_k = stacked_basis_as_transformed_basis(basis)
    util = BasisUtil(basis_k)
    return convert_diagonal_operator_to_basis(
        {
            "basis": VariadicTupleBasis((basis_k, basis_k), None),
            "data": util.dk_stacked[axis] * util.stacked_nk_points[axis],
        },
        VariadicTupleBasis((basis, basis), None),
    )


def _get_average_k(
    states: StateVectorList[_B0Inv, _SBV0],
    axis: int,
) -> ValueList[_B0Inv]:
    """
    Calculate expectation of k.

    Parameters
    ----------
    basis : _SBV0

    Returns
    -------
    SingleBasisOperator[_SBV0]
    """
    operator = _get_k_operator(states["basis"][1], axis)
    return calculate_expectation_list(operator, states)


def plot_averaged_occupation_1d_k(
    states: StateVectorList[TupleBasisLike[Any, _BT0], _SBV0],
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
    occupation_x = _get_average_k(states, axes[0])
    fig, ax = plot_all_value_list_against_time(occupation_x, ax=ax, measure=measure)
    ax.set_ylabel("momentum /m^-1")  # type: ignore unknown
    return fig, ax


def plot_spread_against_k(
    states: StateVectorList[
        TupleBasisLike[Any, _BT0],
        StackedBasisWithVolumeLike,
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
        StackedBasisWithVolumeLike,
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

    ax.plot(k["data"], spread_x["data"])  # type: ignore unknown

    ax.set_xlabel("Momentum /$m^{-1}$")  # type: ignore unknown
    ax.set_ylabel("Spread /m")  # type: ignore unknown
    return fig, ax


def plot_spread_against_x(
    states: StateVectorList[
        TupleBasisLike[Any, _BT0],
        StackedBasisWithVolumeLike,
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
        StackedBasisWithVolumeLike,
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
    x = _get_average_x_periodic(states, axes[0])

    ax.plot(x["data"], spread_x["data"])  # type: ignore unknown

    ax.set_xlabel("Displacement /m")  # type: ignore unknown
    ax.set_ylabel("Spread /m")  # type: ignore unknown
    return fig, ax


def plot_k_distribution_1d(
    states: StateVectorList[
        TupleBasisLike[Any, _BT0],
        StackedBasisWithVolumeLike,
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
        StackedBasisWithVolumeLike,
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
    fig, ax = plot_value_list_distribution(
        k_values, ax=ax, measure=measure, distribution="normal"
    )

    ax.set_xlabel("Momentum /$m^{-1}$")  # type: ignore unknown
    return fig, ax


def plot_x_distribution_1d(
    states: StateVectorList[
        TupleBasisLike[Any, _BT0],
        StackedBasisWithVolumeLike,
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
        StackedBasisWithVolumeLike,
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
    fig, ax = plot_value_list_distribution(x_values, ax=ax, measure=measure)

    ax.set_xlabel("Displacement /m$")  # type: ignore unknown
    return fig, ax


def plot_periodic_x_distribution_1d(
    states: StateVectorList[
        TupleBasisLike[Any, _BT0],
        StackedBasisWithVolumeLike,
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
        StackedBasisWithVolumeLike,
    ]
    axes : tuple[int], optional
        direction to plot along, by default (0,)
    ax : Axes | None, optional
        plot axis, by default None

    Returns
    -------
    tuple[Figure, Axes]
    """
    x_values = _get_average_x_periodic(states, axes[0])
    fig, ax = plot_value_list_distribution(x_values, ax=ax, measure=measure)

    ax.set_xlabel("Displacement /m$")  # type: ignore unknown
    return fig, ax


def _calculate_total_offsset_multiplications_real(
    lhs: np.ndarray[Any, np.dtype[Any]],
    rhs: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[Any]]:
    """Calculate sum_i^N-i A_i B_i+N for all N.

    Parameters
    ----------
    lhs : np.ndarray[Any, np.dtype[np.float64]]
    rhs : np.ndarray[Any, np.dtype[np.float64]]

    Returns
    -------
    np.ndarray[Any, np.dtype[np.float64]]
    """
    return scipy.signal.correlate(lhs, rhs, mode="full")[lhs.size - 1 :]  # type: ignore unknown


def _calculate_total_offsset_multiplications_complex(
    lhs: np.ndarray[Any, np.dtype[np.complex128]],
    rhs: np.ndarray[Any, np.dtype[np.complex128]],
) -> np.ndarray[Any, np.dtype[np.complex128]]:
    """Calculate sum_i^N-i A_i B_i+N for all N.

    Parameters
    ----------
    lhs : np.ndarray[Any, np.dtype[np.float64]]
    rhs : np.ndarray[Any, np.dtype[np.float64]]

    Returns
    -------
    np.ndarray[Any, np.dtype[np.float64]]

    """
    re_re = _calculate_total_offsset_multiplications_real(np.real(lhs), np.real(rhs))
    re_im = _calculate_total_offsset_multiplications_real(np.real(lhs), np.imag(rhs))
    im_re = _calculate_total_offsset_multiplications_real(np.imag(lhs), np.real(rhs))
    im_im = _calculate_total_offsset_multiplications_real(np.imag(lhs), np.imag(rhs))
    return re_re - im_im + 1j * (re_im + im_re)


def get_average_displacements(
    positions: ValueList[TupleBasisLike[_B0Inv, _BT0]],
) -> ValueList[TupleBasisLike[_B0Inv, EvenlySpacedTimeBasis]]:
    """Get the RMS displacement against time."""
    basis = positions["basis"]
    stacked = positions["data"].reshape(basis.shape)
    squared_positions = np.square(stacked)
    total = np.cumsum(squared_positions + squared_positions[:, ::-1], axis=1)[:, ::-1]

    convolution = np.apply_along_axis(
        lambda m: _calculate_total_offsset_multiplications_real(m, m),
        axis=1,
        arr=stacked,
    ).astype(np.float64)

    squared_diff = (total - 2 * convolution) / np.arange(1, basis[1].n + 1)[::-1]  # type: ignore unknown
    out_basis = EvenlySpacedTimeBasis(basis[1].n, 1, 0, basis[1].dt * (basis[1].n))  # type: ignore unknown
    return {
        "basis": VariadicTupleBasis((basis[0], out_basis), None),
        "data": squared_diff.ravel(),
    }  # type: ignore unknown


def get_average_isf(
    positions: ValueList[TupleBasisLike[_B0Inv, _BT0]],
    scattered_k: float,
) -> ValueList[TupleBasisLike[_B0Inv, EvenlySpacedTimeBasis[Any, Any, Any]]]:
    """Get the average ISF against time difference."""
    basis = positions["basis"]
    stacked = positions["data"].reshape(basis.shape)
    exp_positions = np.exp(1j * scattered_k * stacked)

    # convolution_j = \sum_i^N-j e^(ik.x_i+j) e^(-ik.x_i)
    convolution = np.apply_along_axis(
        lambda m: _calculate_total_offsset_multiplications_complex(np.conj(m), m),
        axis=1,
        arr=exp_positions,
    )

    isf = (convolution) / np.arange(1, basis[1].n + 1)[::-1]  # type: ignore unknown
    out_basis = EvenlySpacedTimeBasis(basis[1].n, 1, 0, basis[1].dt * (basis[1].n))  # type: ignore unknown
    return {"basis": TupleBasis(basis[0], out_basis), "data": isf.ravel()}  # type: ignore unknown


def get_average_normalized_drift(
    positions: ValueList[TupleBasisLike[_B0Inv, _BT0]],
    k_values: ValueList[TupleBasisLike[_B0Inv, _BT0]],
) -> ValueList[TupleBasisLike[_B0Inv, EvenlySpacedTimeBasis[Any, Any, Any]]]:
    basis = positions["basis"]
    positions["data"].reshape(basis.shape)
    stacked_x = positions["data"].reshape(basis.shape)
    stacked_k = 1 / k_values["data"].reshape(basis.shape)

    # Calculate a_i = sum_j^N-i x_j+i / k_j
    convolution = np.zeros((basis.shape[0], basis.shape[1]))
    for i in range(basis.shape[0]):
        convolution[i] = _calculate_total_offsset_multiplications_real(
            stacked_x[i], 1 / stacked_k[i]
        )
    # Calculate a_i = sum_j^N-i x_j / k_j
    cumulative = np.cumsum(stacked_x / stacked_k, axis=1)[:, ::-1]
    diff = cumulative - convolution / cumulative.shape[1]

    out_basis = EvenlySpacedTimeBasis(basis[1].n, 1, 0, basis[1].dt * (basis[1].n))  # type: ignore unknown
    return {"basis": TupleBasis(basis[0], out_basis), "data": diff.ravel()}  # type: ignore unknown


def get_normalized_average_drift(
    positions: ValueList[TupleBasisLike[_B0Inv, _BT0]],
    k_values: ValueList[TupleBasisLike[_B0Inv, _BT0]],
) -> ValueList[TupleBasisLike[_B0Inv, EvenlySpacedTimeBasis[Any, Any, Any]]]:
    basis = positions["basis"]
    positions["data"].reshape(basis.shape)
    stacked_x = positions["data"].reshape(basis.shape)
    stacked_k = 1 / k_values["data"].reshape(basis.shape)

    # Calculate a_i = sum_j^N-i x_j+i
    cumulative_rhs = np.cumsum(stacked_x[:, ::-1], axis=1)

    # Calculate a_i = sum_j^N-i x_j
    cumulative_lhs = np.cumsum(stacked_x, axis=1)
    # Calculate b_i = sum_j^N-i k_j
    cumulative_k = np.cumsum(stacked_k, axis=1)
    # DIvide through by the total k, equivalent to N * average k
    diff = ((cumulative_lhs - cumulative_rhs) / cumulative_k)[:, ::-1]

    out_basis = EvenlySpacedTimeBasis(basis[1].n, 1, 0, basis[1].dt * (basis[1].n))  # type: ignore unknown
    return {"basis": TupleBasis(basis[0], out_basis), "data": diff.ravel()}  # type: ignore unknown


def plot_average_displacement_1d_x(
    states: StateVectorList[
        TupleBasisLike[Any, _BT0],
        StackedBasisWithVolumeLike,
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
    displacements = get_average_displacements(restored_x)

    return plot_average_value_list_against_time(
        displacements, ax=ax, measure=measure, scale=scale
    )


def plot_average_drift_1d_x(
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
    k_points = _get_average_k(states, axis=0)
    drift = get_normalized_average_drift(restored_x, k_points)

    return plot_average_value_list_against_time(
        drift, ax=ax, measure=measure, scale=scale
    )


def plot_average_isf_1d_x(
    states: StateVectorList[
        TupleBasisLike[Any, _BT0],
        StackedBasisWithVolumeLike[Any, Any, Any],
    ],
    scattered_k: float,
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
    drift = get_average_isf(restored_x, scattered_k)

    return plot_average_value_list_against_time(
        drift, ax=ax, measure=measure, scale=scale
    )
