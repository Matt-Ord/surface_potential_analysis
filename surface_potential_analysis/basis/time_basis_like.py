from __future__ import annotations

import abc
from typing import Any, Literal, Protocol, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import FundamentalBasis
from surface_potential_analysis.basis.basis_like import BasisLike
from surface_potential_analysis.basis.evenly_spaced_basis import EvenlySpacedBasis

_N0_co = TypeVar("_N0_co", bound=int, covariant=True)
_N1_co = TypeVar("_N1_co", bound=int, covariant=True)
_N2_co = TypeVar("_N2_co", bound=int, covariant=True)


# ruff: noqa: D102
class BasisWithTimeLike(BasisLike[_N1_co, _N0_co], Protocol[_N1_co, _N0_co]):  # type: ignore[misc]
    """A generic object that represents an axis with a corresponding axis vector."""

    @property
    @abc.abstractmethod
    def times(self) -> np.ndarray[tuple[_N0_co], np.dtype[np.float64]]:
        ...

    @property
    @abc.abstractmethod
    def fundamental_times(self) -> np.ndarray[tuple[_N1_co], np.dtype[np.float64]]:
        ...

    @property
    def delta_t(self) -> float:
        return self.fundamental_times[1] - self.fundamental_times[0]  # type: ignore[no-any-return]


class EvenlySpacedTimeBasis(
    EvenlySpacedBasis[_N0_co, _N1_co, _N2_co], BasisWithTimeLike[_N0_co, Any]
):
    """A axis with vectors that are the fundamental position states."""

    def __init__(self, n: _N0_co, step: _N1_co, offset: _N2_co, delta_t: float) -> None:
        self._delta_t = delta_t
        super().__init__(n, step, offset)  # type: ignore[arg-type]

    @property
    def times(self) -> np.ndarray[tuple[_N0_co], np.dtype[np.float64]]:
        return self.fundamental_dt * self.nt_points

    @property
    def fundamental_times(self) -> np.ndarray[tuple[Any], np.dtype[np.float64]]:
        return np.linspace(0, self._delta_t, self.fundamental_n)  # type: ignore[no-any-return]

    @property
    def delta_t(self) -> float:
        return self._delta_t

    @property
    def dt(self) -> float:
        return self._delta_t / self.n

    @property
    def fundamental_dt(self) -> float:
        return self._delta_t / self.fundamental_n

    @property
    def nt_points(self) -> np.ndarray[tuple[int], np.dtype[np.int_]]:
        return np.arange(self.offset, self.n * self.step + self.offset, self.step)


class FundamentalTimeBasis(EvenlySpacedTimeBasis[_N0_co, Literal[1], Literal[0]]):
    """A axis with vectors that are the fundamental position states."""

    def __init__(self, n: _N0_co, delta_t: float) -> None:
        self._delta_t = delta_t
        super().__init__(n, 1, 0, delta_t)


class ExplicitTimeBasis(FundamentalBasis[_N1_co], BasisWithTimeLike[_N1_co, _N1_co]):
    """A axis with vectors that are the fundamental position states."""

    def __init__(self, times: np.ndarray[tuple[_N1_co], np.dtype[np.float64]]) -> None:
        self._times = times
        super().__init__(times.size)  # type: ignore[arg-type]

    @property
    def times(self) -> np.ndarray[tuple[_N1_co], np.dtype[np.float64]]:
        return self._times

    @property
    def fundamental_times(self) -> np.ndarray[tuple[_N1_co], np.dtype[np.float64]]:
        return self.times
