# ruff: noqa: D102
from __future__ import annotations

import abc
from typing import Any, Literal, Protocol, TypeVar, cast

import numpy as np

from surface_potential_analysis.basis.basis import FundamentalBasis
from surface_potential_analysis.basis.evenly_spaced_basis import (
    EvenlySpacedBasis,
)

from .basis_like import BasisLike

_N0_co = TypeVar("_N0_co", bound=int, covariant=True)
_NF0_co = TypeVar("_NF0_co", bound=int, covariant=True)
_N1_co = TypeVar("_N1_co", bound=int, covariant=True)
_N2_co = TypeVar("_N2_co", bound=int, covariant=True)


class BasisWithBlockFractionLike(BasisLike[_NF0_co, _N0_co], Protocol[_NF0_co, _N0_co]):  # type: ignore[misc]
    """A generic object that represents an axis with a corresponding axis vector."""

    @property
    @abc.abstractmethod
    def bloch_fractions(self) -> np.ndarray[tuple[int, _N0_co], np.dtype[np.float64]]:
        ...

    @property
    def ndim(self) -> int:
        return self.bloch_fractions.shape[0]


class ExplicitBlockFractionBasis(
    FundamentalBasis[_NF0_co], BasisWithBlockFractionLike[_NF0_co, _NF0_co]
):
    """A axis with vectors that are the fundamental position states."""

    def __init__(
        self, bloch_fractions: np.ndarray[tuple[Any, _NF0_co], np.dtype[np.float64]]
    ) -> None:
        self._bloch_fractions = bloch_fractions
        super().__init__(cast(_NF0_co, bloch_fractions.shape[1]))

    @property
    def bloch_fractions(self) -> np.ndarray[tuple[int, _NF0_co], np.dtype[np.float64]]:
        return self._bloch_fractions


class EvenlySpacedBlockFractionBasis(
    EvenlySpacedBasis[_N0_co, _N1_co, _N2_co],
    BasisWithBlockFractionLike[Any, _N0_co],
):
    """A axis with vectors that are the fundamental position states."""

    def __init__(self, n: _N0_co, step: _N1_co, offset: _N2_co) -> None:
        super().__init__(n, step, offset)

    @property
    def bloch_fractions(self) -> np.ndarray[tuple[Any, _N0_co], np.dtype[np.float64]]:
        return self.__from_fundamental__(  # type: ignore can't infer shape
            np.linspace(-0.5, 0.5, self.fundamental_n, endpoint=False)
        )


class FundamentalBlockFractionBasis(
    EvenlySpacedBlockFractionBasis[_NF0_co, Literal[1], Literal[0]]
):
    """A axis with vectors that are the fundamental position states."""

    def __init__(self, n: _NF0_co) -> None:
        super().__init__(n, 1, 0)
