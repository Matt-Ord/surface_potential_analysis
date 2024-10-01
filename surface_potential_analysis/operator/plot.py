from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from surface_potential_analysis.basis.basis_like import BasisLike
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasisWithVolumeLike,
    TupleBasis,
)
from surface_potential_analysis.operator.conversion import convert_operator_to_basis
from surface_potential_analysis.operator.operator import (
    SingleBasisDiagonalOperator,
    as_diagonal_operator,
    as_operator,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_position_basis,
)
from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_eigenvectors,
    calculate_eigenvectors_hermitian,
)
from surface_potential_analysis.state_vector.eigenvalue_list_plot import (
    plot_eigenstate_occupations as plot_eigenstate_occupations_states,
)
from surface_potential_analysis.state_vector.eigenvalue_list_plot import (
    plot_eigenvalues as plot_eigenvalues_states,
)
from surface_potential_analysis.util.plot import (
    Scale,
    animate_data_through_list_1d_n,
    get_figure,
    plot_data_1d_n,
    plot_data_1d_x,
    plot_data_2d,
    plot_data_2d_x,
)
from surface_potential_analysis.util.util import (
    Measure,
    get_measured_data,
)

if TYPE_CHECKING:
    from matplotlib.animation import ArtistAnimation
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.operator.operator import (
        DiagonalOperator,
        SingleBasisOperator,
    )
    from surface_potential_analysis.operator.operator_list import DiagonalOperatorList
    from surface_potential_analysis.types import SingleStackedIndexLike

    from .operator import Operator

    _SB0 = TypeVar("_SB0", bound=StackedBasisWithVolumeLike[Any, Any, Any])

    _B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
    _B1 = TypeVar("_B1", bound=BasisLike[Any, Any])
    _B2 = TypeVar("_B2", bound=BasisLike[Any, Any])


def plot_operator_sparsity(
    operator: Operator[BasisLike[Any, Any], BasisLike[Any, Any]],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    scale: Scale = "linear",
) -> tuple[Figure, Axes]:
    """
    Given an operator, plot the sparisity as a cumulative sum.

    Parameters
    ----------
    operator : Operator[BasisLike[Any, Any], BasisLike[Any, Any]]
    ax : Axes | None, optional
        axis, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes]
    """
    fig, ax = get_figure(ax)
    measured = get_measured_data(operator["data"], measure)

    values, bins = np.histogram(
        measured,
        bins=np.logspace(-22, np.log10(np.max(measured)), 10000),  # type:ignore lib
    )

    cumulative = np.cumsum(values)
    ax.plot(bins[:-1], cumulative)  # type:ignore lib

    ax.set_yscale(scale)  # type:ignore lib
    ax.set_xlabel("Value")  # type:ignore lib
    ax.set_ylabel("Density")  # type:ignore lib
    ax.set_xscale("log")  # type:ignore lib

    return fig, ax


def _get_operator_diagonals(
    operator: Operator[BasisLike[Any, Any], BasisLike[Any, Any]],
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
    stacked = operator["data"].reshape(operator["basis"].shape)
    out = np.zeros_like(stacked)
    for i in range(stacked.shape[0]):
        out[i] = np.diag(np.roll(stacked, shift=i, axis=0))
    return out


def plot_operator_diagonal_sparsity(
    operator: Operator[BasisLike[Any, Any], BasisLike[Any, Any]],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes]:
    """
    Given an operator, plot the sparisity as a cumulative sum.

    Parameters
    ----------
    operator : Operator[BasisLike[Any, Any], BasisLike[Any, Any]]
    ax : Axes | None, optional
        axis, by default None
    measure : Measure, optional
        measure, by default "abs"
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes]
    """
    fig, ax = get_figure(ax)
    diagonals = _get_operator_diagonals(operator)
    size = np.linalg.norm(diagonals, axis=1)

    values, bins = np.histogram(
        size,
        bins=np.logspace(-22, np.log10(np.max(size)), 10000),  # type:ignore lib
    )

    cumulative = np.cumsum(values)
    ax.plot(bins[:-1], cumulative)  # type:ignore lib

    ax.set_yscale(scale)  # type:ignore lib
    ax.set_xlabel("Value")  # type:ignore lib
    ax.set_ylabel("Density")  # type:ignore lib
    ax.set_xscale("log")  # type:ignore lib

    return fig, ax


def plot_eigenstate_occupations(
    operator: SingleBasisOperator[BasisLike[Any, Any]],
    temperature: float,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the expected occupation of eigenstates at the given temperature.

    Parameters
    ----------
    eigenstates : EigenstateList[BasisLike[Any, Any], BasisLike[Any, Any]]
    temperature : float
    ax : Axes | None, optional
        ax, by default None
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    eigenstates = calculate_eigenvectors(operator)
    return plot_eigenstate_occupations_states(
        eigenstates, temperature, ax=ax, scale=scale
    )


def plot_eigenvalues(
    operator: SingleBasisOperator[BasisLike[Any, Any]],
    *,
    hermitian: bool = False,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the expected occupation of eigenstates at the given temperature.

    Parameters
    ----------
    eigenstates : EigenstateList[BasisLike[Any, Any], BasisLike[Any, Any]]
    temperature : float
    ax : Axes | None, optional
        ax, by default None
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    eigenstates = (
        calculate_eigenvectors_hermitian(operator)
        if hermitian
        else calculate_eigenvectors(operator)
    )
    return plot_eigenvalues_states(eigenstates, ax=ax, scale=scale, measure=measure)


def plot_diagonal_operator_along_diagonal(
    operator: DiagonalOperator[_B1, _B2],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the expected occupation of eigenstates at the given temperature.

    Parameters
    ----------
    eigenstates : EigenstateList[BasisLike[Any, Any], BasisLike[Any, Any]]
    temperature : float
    ax : Axes | None, optional
        ax, by default None
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax, line = plot_data_1d_n(
        TupleBasis(operator["basis"][0]), operator["data"], measure=measure, scale=scale
    )

    line.set_label(f"{measure} operator")
    return fig, ax, line


def animate_diagonal_operator_list_along_diagonal(
    operator: DiagonalOperatorList[_B0, _B1, _B2],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Plot the expected occupation of eigenstates at the given temperature.

    Parameters
    ----------
    eigenstates : EigenstateList[BasisLike[Any, Any], BasisLike[Any, Any]]
    temperature : float
    ax : Axes | None, optional
        ax, by default None
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    return animate_data_through_list_1d_n(
        TupleBasis(operator["basis"][1][0]),
        operator["data"].reshape(operator["basis"][0].n, -1),
        ax=ax,
        scale=scale,
        measure=measure,
    )


def plot_operator_along_diagonal(
    operator: SingleBasisOperator[BasisLike[Any, Any]],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the expected occupation of eigenstates at the given temperature.

    Parameters
    ----------
    eigenstates : EigenstateList[BasisLike[Any, Any], BasisLike[Any, Any]]
    temperature : float
    ax : Axes | None, optional
        ax, by default None
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    diagonal = as_diagonal_operator(operator)
    return plot_diagonal_operator_along_diagonal(
        diagonal, ax=ax, scale=scale, measure=measure
    )


def plot_operator_along_diagonal_1d_x(  # noqa:PLR0913
    operator: SingleBasisOperator[_SB0],
    axes: tuple[int] = (0,),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "real",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the expected occupation of eigenstates at the given temperature.

    Parameters
    ----------
    eigenstates : EigenstateList[BasisLike[Any, Any], BasisLike[Any, Any]]
    temperature : float
    ax : Axes | None, optional
        ax, by default None
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    basis_x = stacked_basis_as_fundamental_position_basis(operator["basis"][0])
    converted = convert_operator_to_basis(operator, TupleBasis(basis_x, basis_x))
    diagonal = as_diagonal_operator(converted)
    return plot_data_1d_x(
        diagonal["basis"][0],
        diagonal["data"],
        axes=axes,
        idx=idx,
        ax=ax,
        scale=scale,
        measure=measure,
    )


def plot_diagonal_operator_along_diagonal_1d_x(  # noqa:PLR0913
    operator: SingleBasisOperator[_SB0],
    axes: tuple[int] = (0,),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "real",
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot the expected occupation of eigenstates at the given temperature.

    Parameters
    ----------
    eigenstates : EigenstateList[BasisLike[Any, Any], BasisLike[Any, Any]]
    temperature : float
    ax : Axes | None, optional
        ax, by default None
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    return plot_operator_along_diagonal_1d_x(
        as_operator(operator),
        axes=axes,
        idx=idx,
        ax=ax,
        scale=scale,
        measure=measure,
    )


def plot_operator_along_diagonal_2d_x(  # noqa:PLR0913
    operator: SingleBasisOperator[_SB0],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "real",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the expected occupation of eigenstates at the given temperature.

    Parameters
    ----------
    eigenstates : EigenstateList[BasisLike[Any, Any], BasisLike[Any, Any]]
    temperature : float
    ax : Axes | None, optional
        ax, by default None
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    basis_x = stacked_basis_as_fundamental_position_basis(operator["basis"][0])
    converted = convert_operator_to_basis(operator, TupleBasis(basis_x, basis_x))
    diagonal = as_diagonal_operator(converted)
    return plot_data_2d_x(
        diagonal["basis"][0],
        diagonal["data"],
        axes=axes,
        idx=idx,
        ax=ax,
        scale=scale,
        measure=measure,
    )


def plot_diagonal_operator_along_diagonal_2d_x(  # noqa:PLR0913
    operator: SingleBasisOperator[_SB0],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "real",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the expected occupation of eigenstates at the given temperature.

    Parameters
    ----------
    eigenstates : EigenstateList[BasisLike[Any, Any], BasisLike[Any, Any]]
    temperature : float
    ax : Axes | None, optional
        ax, by default None
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    return plot_operator_along_diagonal_2d_x(
        as_operator(operator),
        axes=axes,
        idx=idx,
        ax=ax,
        scale=scale,
        measure=measure,
    )


def plot_operator_2d(
    operator: SingleBasisOperator[BasisLike[Any, Any]],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the expected occupation of eigenstates at the given temperature.

    Parameters
    ----------
    eigenstates : EigenstateList[BasisLike[Any, Any], BasisLike[Any, Any]]
    temperature : float
    ax : Axes | None, optional
        ax, by default None
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    data = operator["data"].reshape(operator["basis"].shape)
    return plot_data_2d(data, ax=ax, scale=scale, measure=measure)


def plot_operator_2d_diagonal(
    operator: SingleBasisDiagonalOperator[BasisLike[Any, Any]],
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the expected occupation of eigenstates at the given temperature.

    Parameters
    ----------
    eigenstates : EigenstateList[BasisLike[Any, Any], BasisLike[Any, Any]]
    temperature : float
    ax : Axes | None, optional
        ax, by default None
    scale : Scale, optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    return plot_operator_2d(as_operator(operator), ax=ax, measure=measure)
