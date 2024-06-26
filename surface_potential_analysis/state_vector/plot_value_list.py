from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from matplotlib import pyplot as plt

from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis

if TYPE_CHECKING:
    from surface_potential_analysis.state_vector.eigenstate_collection import ValueList

_AX0Inv = TypeVar("_AX0Inv", bound=EvenlySpacedTimeBasis[Any, Any, Any])


def plot_value_list_abs_against_time(
    values: ValueList[Any],
    times0: _AX0Inv,
) -> None:
    """Plot the ValueList against time."""
    plt.plot(times0.times, abs(values["data"]))
    plt.xlabel("Time")
    plt.show()

    input()
