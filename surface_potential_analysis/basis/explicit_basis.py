from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis_like import (
    AxisVector,
    BasisLike,
    BasisWithLengthLike,
)
from surface_potential_analysis.basis.conversion import basis_as_fundamental_basis
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasisWithVolumeLike,
    TupleBasisLike,
)
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_list_to_basis,
)
from surface_potential_analysis.state_vector.state_vector_list import (
    StateVectorList,
    get_basis_states,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        FundamentalBasis,
    )


_S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])

_B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
_B1 = TypeVar("_B1", bound=BasisLike[Any, Any])
_BL1 = TypeVar("_BL1", bound=BasisWithLengthLike[Any, Any, Any])

_SBL1 = TypeVar("_SBL1", bound=StackedBasisWithVolumeLike[Any, Any, Any])


class ExplicitBasis(BasisLike[Any, Any], Generic[_B0, _B1]):
    """An basis with vectors given as explicit states."""

    def __init__(
        self,
        vectors: StateVectorList[_B0, _B1],
    ) -> None:
        self._vectors = vectors
        super().__init__()

    @property
    def vectors_basis(self) -> TupleBasisLike[_B0, _B1]:
        """Basis of the vector list."""
        return self.vectors["basis"]

    @property
    def n(self) -> int:
        """N states."""
        return self.vectors_basis[0].n

    @property
    def fundamental_n(self) -> int:
        """N states in fundamental basis."""
        return self.vectors_basis[1].fundamental_n

    @property
    def vectors(self) -> StateVectorList[_B0, _B1]:
        """The states that make up the basis."""
        return self._vectors

    @property
    def _raw_vectors(self) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
        """The states that make up the basis, in the current basis."""
        return self._vectors["data"].reshape(self.vectors_basis[0].n, -1)

    @property
    def fundamental_raw_vectors(
        self,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
        """States in fundamental basis."""
        return convert_state_vector_list_to_basis(
            self.vectors, basis_as_fundamental_basis(self.vectors_basis[1])
        )["data"].reshape(self.vectors_basis[0].n, -1)

    @classmethod
    def from_state_vectors(
        cls: type[ExplicitBasis[_B0, _B1]],
        vectors: StateVectorList[_B0, _B1],
    ) -> ExplicitBasis[_B0, _B1]:
        """Form a basis from a list of vectors."""
        return cls(vectors)

    @classmethod
    def from_basis(
        cls: type[ExplicitBasis[FundamentalBasis[int], _B1]],
        basis: _B1,
    ) -> ExplicitBasis[FundamentalBasis[int], _B1]:
        """Form an explicit basis from another basis."""
        return cls.from_state_vectors(get_basis_states(basis))

    def __from_state_basis__(  # noqa: PLW3201
        self,
        vectors: np.ndarray[
            _S0Inv,
            np.dtype[np.complex128]
            | np.dtype[np.float64]
            | np.dtype[np.float64 | np.complex128],
        ],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
        """Given a set of vectors convert them from state basis into this basis along the given axis.

        Parameters
        ----------
        vectors : np.ndarray[_S0Inv, np.dtype[np.complex128]  |  np.dtype[np.float64]]
        axis : int, optional
            axis, by default -1

        Returns
        -------
        np.ndarray[tuple[int, ...], np.dtype[np.complex128]]
        """
        transformed = np.tensordot(
            np.conj(self._raw_vectors), vectors, axes=([1], [axis])
        )
        return np.moveaxis(transformed, 0, axis)

    def __into_state_basis__(  # noqa: PLW3201
        self,
        vectors: np.ndarray[
            _S0Inv,
            np.dtype[np.complex128]
            | np.dtype[np.float64]
            | np.dtype[np.float64 | np.complex128],
        ],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
        """Given a set of vectors convert them to state basis along the given axis.

        Parameters
        ----------
        vectors : np.ndarray[_S0Inv, np.dtype[np.complex128]  |  np.dtype[np.float64]]
        axis : int, optional
            axis, by default -1

        Returns
        -------
        np.ndarray[tuple[int, ...], np.dtype[np.complex128]]
        """
        transformed = np.tensordot(vectors, self._raw_vectors, axes=([axis], [0]))
        return np.moveaxis(transformed, -1, axis)

    def __from_fundamental__(
        self,
        vectors: np.ndarray[
            _S0Inv,
            np.dtype[np.complex128]
            | np.dtype[np.float64]
            | np.dtype[np.float64 | np.complex128],
        ],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
        # The vectors in self.vectors["basis"][1] basis
        vectors_in_basis = self.vectors_basis[1].__from_fundamental__(vectors, axis)
        return self.__from_state_basis__(vectors_in_basis, axis)

    def __into_fundamental__(
        self,
        vectors: np.ndarray[
            _S0Inv,
            np.dtype[np.complex128]
            | np.dtype[np.float64]
            | np.dtype[np.float64 | np.complex128],
        ],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
        vectors_in_basis = self.__into_state_basis__(vectors, axis)
        return self.vectors_basis[1].__into_fundamental__(vectors_in_basis, axis)

    def __from_transformed__(
        self,
        vectors: np.ndarray[
            _S0Inv,
            np.dtype[np.complex128]
            | np.dtype[np.float64]
            | np.dtype[np.float64 | np.complex128],
        ],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
        # The vectors in self.vectors["basis"][1] basis
        vectors_in_basis = self.vectors_basis[1].__from_transformed__(vectors, axis)
        return self.__from_state_basis__(vectors_in_basis, axis)

    def __into_transformed__(
        self,
        vectors: np.ndarray[
            _S0Inv,
            np.dtype[np.complex128]
            | np.dtype[np.float64]
            | np.dtype[np.float64 | np.complex128],
        ],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
        vectors_in_basis = self.__into_state_basis__(vectors, axis)
        return self.vectors_basis[1].__into_transformed__(vectors_in_basis, axis)


class ExplicitBasisWithLength(
    ExplicitBasis[_B0, _BL1], BasisWithLengthLike[Any, Any, Any]
):
    """An basis with vectors given as explicit states."""

    def __init__(
        self,
        vectors: StateVectorList[_B0, _BL1],
    ) -> None:
        super().__init__(vectors)

    @property
    def delta_x(self) -> AxisVector[int]:
        """Length of axis."""
        return self.vectors_basis[1].delta_x

    @classmethod
    def from_state_vectors(
        cls: type[ExplicitBasisWithLength[_B0, _BL1]],
        vectors: StateVectorList[_B0, _BL1],
    ) -> ExplicitBasisWithLength[_B0, _BL1]:
        """Form a basis from a list of vectors."""
        return cls(vectors)

    @classmethod
    def from_basis(
        cls: type[ExplicitBasisWithLength[FundamentalBasis[int], _BL1]],
        basis: _BL1,
    ) -> ExplicitBasisWithLength[FundamentalBasis[int], _BL1]:
        """Explicit basis from a basis."""
        return cls.from_state_vectors(get_basis_states(basis))


class ExplicitStackedBasisWithLength(
    ExplicitBasis[_B0, _SBL1],
    StackedBasisWithVolumeLike[Any, Any, Any],
    Generic[_B0, _SBL1],
):
    """An basis with vectors given as explicit states."""

    def __init__(
        self,
        vectors: StateVectorList[_B0, _SBL1],
    ) -> None:
        super().__init__(vectors)

    @property
    def delta_x_stacked(
        self,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Length of all of the axis vectors stacked."""
        return self.vectors_basis[1].delta_x_stacked

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of basis."""
        return self.vectors_basis[1].shape

    @property
    def fundamental_shape(self) -> tuple[int, ...]:
        """Shape of fundamental basis."""
        return self.vectors_basis[1].fundamental_shape

    @classmethod
    def from_state_vectors(
        cls: type[ExplicitStackedBasisWithLength[_B0, _SBL1]],
        vectors: StateVectorList[_B0, _SBL1],
    ) -> ExplicitStackedBasisWithLength[_B0, _SBL1]:
        """Form a basis from a list of vectors."""
        return cls(vectors)

    @classmethod
    def from_basis(
        cls: type[ExplicitStackedBasisWithLength[FundamentalBasis[int], _SBL1]],
        basis: _SBL1,
    ) -> ExplicitStackedBasisWithLength[FundamentalBasis[int], _SBL1]:
        """Form a basis from another basis."""
        return cls.from_state_vectors(get_basis_states(basis))
