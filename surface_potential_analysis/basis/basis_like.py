from __future__ import annotations

import abc
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

import numpy as np

_N0_co = TypeVar("_N0_co", bound=int, covariant=True)
_NF0_co = TypeVar("_NF0_co", bound=int, covariant=True)

_ND0Inv = TypeVar("_ND0Inv", bound=int)
_N0Inv = TypeVar("_N0Inv", bound=int)
_NF0Inv = TypeVar("_NF0Inv", bound=int)

AxisVector = np.ndarray[tuple[_ND0Inv], np.dtype[np.float64]]
AxisVector1d = AxisVector[Literal[1]]
AxisVector2d = AxisVector[Literal[2]]
AxisVector3d = AxisVector[Literal[3]]


_S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])


# ruff: noqa: D102
@runtime_checkable
class FromFundamentalBasis(Protocol[_NF0_co, _N0_co]):
    """Represents an axis which can be converted from the fundamental axis."""

    @abc.abstractmethod
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
        """
        Given a set of vectors convert them to fundamental basis along the given axis.

        Parameters
        ----------
        vectors : np.ndarray[_S0Inv, np.dtype[np.complex_  |  np.float_]]
        axis : int, optional
            axis to transform, by default -1

        Returns
        -------
        np.ndarray[tuple[int, ...], np.dtype[np.complex_]]
            The vectors, converted along axis
        """
        ...


@runtime_checkable
class FromTransformedBasis(Protocol[_NF0_co, _N0_co]):
    """Represents an axis which can be converted from the transformed axis."""

    @abc.abstractmethod
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
        """
        Given a set of vectors convert them to fundamental basis along the given axis.

        Parameters
        ----------
        vectors : np.ndarray[_S0Inv, np.dtype[np.complex_  |  np.float_]]
        axis : int, optional
            axis to transform, by default -1

        Returns
        -------
        np.ndarray[tuple[int, ...], np.dtype[np.complex_]]
            The vectors, converted along axis
        """
        ...


@runtime_checkable
class IntoFundamentalBasis(Protocol[_NF0_co, _N0_co]):
    """Represents an axis which can be converted to fundamental axis."""

    @abc.abstractmethod
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
        ...


@runtime_checkable
class IntoTransformedBasis(Protocol[_NF0_co, _N0_co]):
    """Represents an axis which can be converted to transformed axis."""

    @abc.abstractmethod
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
        ...


@runtime_checkable
class AsFundamentalBasis(
    IntoFundamentalBasis[_NF0_co, _N0_co],
    IntoTransformedBasis[_NF0_co, _N0_co],
    Protocol[_NF0_co, _N0_co],
):
    """Represents an axis which can (inexpensively) be converted to fundamental axis."""

    @abc.abstractmethod
    def __as_fundamental__(
        self,
        vectors: np.ndarray[
            _S0Inv,
            np.dtype[np.complex128]
            | np.dtype[np.float64]
            | np.dtype[np.float64 | np.complex128],
        ],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
        ...

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
        return self.__as_fundamental__(vectors, axis)

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
        fundamental = self.__into_fundamental__(vectors, axis)
        return np.fft.fft(fundamental, axis=axis, norm="ortho")  # type: ignore[no-any-return]


@runtime_checkable
class AsTransformedBasis(
    IntoFundamentalBasis[_NF0_co, _N0_co],
    IntoTransformedBasis[_NF0_co, _N0_co],
    Protocol[_NF0_co, _N0_co],
):
    """Represents an axis which can (inexpensively) be converted to transformed axis."""

    @abc.abstractmethod
    def __as_transformed__(
        self,
        vectors: np.ndarray[
            _S0Inv,
            np.dtype[np.complex128]
            | np.dtype[np.float64]
            | np.dtype[np.float64 | np.complex128],
        ],
        axis: int = -1,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
        ...

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
        return self.__as_transformed__(vectors, axis)

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
        as_transformed = self.__into_transformed__(vectors, axis)
        return np.fft.ifft(as_transformed, axis=axis, norm="ortho")  # type: ignore[no-any-return]


@runtime_checkable
class BasisLike(
    FromFundamentalBasis[_NF0_co, _N0_co],
    IntoFundamentalBasis[_NF0_co, _N0_co],
    FromTransformedBasis[_NF0_co, _N0_co],
    IntoTransformedBasis[_NF0_co, _N0_co],
    Protocol[_NF0_co, _N0_co],
):
    """A generic object that represents an axis for a axis."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n={self.n.__repr__()}, fundamental_n={self.fundamental_n.__repr__()})"

    @property
    @abc.abstractmethod
    def n(self) -> _N0_co:
        ...

    @property
    @abc.abstractmethod
    def fundamental_n(self) -> _NF0_co:
        ...

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
        transformed = self.__into_transformed__(vectors, axis)
        return np.fft.ifft(transformed, axis=axis, norm="ortho")  # type: ignore[no-any-return]

    # !Can also be done like
    # !basis_vectors = self.__into_fundamental__(np.eye(self.n, self.n))
    # !transformed = np.tensordot(vectors, basis_vectors, axes=([axis], [1]))
    # !return np.moveaxis(transformed, -1, axis)  # type: ignore[no-any-return]
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
        fundamental = self.__into_fundamental__(vectors, axis)
        return np.fft.fft(fundamental, axis=axis, norm="ortho")  # type: ignore[no-any-return]

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
        transformed = np.fft.fft(vectors, self.fundamental_n, axis=axis, norm="ortho")
        return self.__from_transformed__(transformed, axis)

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
        fundamental = np.fft.ifft(vectors, self.fundamental_n, axis=axis, norm="ortho")
        return self.__from_fundamental__(fundamental, axis)

    def __convert_vector_into__(
        self,
        vector: np.ndarray[
            _S0Inv,
            np.dtype[np.complex128]
            | np.dtype[np.float64]
            | np.dtype[np.float64 | np.complex128],
        ],
        basis: BasisLike[Any, Any],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[np.complex128]]:
        assert basis.fundamental_n == self.fundamental_n
        if self == basis:
            return vector.astype(np.complex128)
        # Small speedup here, and prevents imprecision of fft followed by ifft
        # And two pad_ft_points
        if isinstance(self, AsTransformedBasis) and isinstance(
            basis, AsTransformedBasis
        ):
            # If initial axis and final axis are AsTransformedAxis
            # we (might) be able to prevent the need for a fft
            transformed = self.__into_transformed__(vector, axis)
            return basis.__from_transformed__(transformed, axis)
        fundamental = self.__into_fundamental__(vector, axis)
        return basis.__from_fundamental__(fundamental, axis)


def convert_vector(
    vector: np.ndarray[
        _S0Inv,
        np.dtype[np.complex128]
        | np.dtype[np.float64]
        | np.dtype[np.float64 | np.complex128],
    ],
    initial_basis: BasisLike[Any, Any],
    final_basis: BasisLike[Any, Any],
    axis: int = -1,
) -> np.ndarray[Any, np.dtype[np.complex128]]:
    """
    Convert a vector, expressed in terms of the given basis from_config in the basis to_config.

    Parameters
    ----------
    vector : np.ndarray[tuple[int], np.dtype[np.complex_] | np.dtype[np.float_]]
        the vector to convert
    from_config : _B3d0Inv
    to_config : _B3d1Inv
    axis : int, optional
        axis along which to convert, by default -1

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.complex_]]
    """
    return initial_basis.__convert_vector_into__(vector, final_basis, axis)


def convert_dual_vector(
    co_vector: np.ndarray[
        _S0Inv,
        np.dtype[np.complex128]
        | np.dtype[np.float64]
        | np.dtype[np.float64 | np.complex128],
    ],
    initial_basis: BasisLike[Any, Any],
    final_basis: BasisLike[Any, Any],
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
    """
    Convert a co_vector, expressed in terms of the given basis from_config in the basis to_config.

    Parameters
    ----------
    co_vector : np.ndarray[tuple[int], np.dtype[np.complex_]]
        the vector to convert
    from_config : _B3d0Inv
    to_config : _B3d1Inv
    axis : int, optional
        axis along which to convert, by default -1

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.complex_]]
    """
    return np.conj(convert_vector(np.conj(co_vector), initial_basis, final_basis, axis))  # type: ignore[no-any-return]


def convert_matrix(
    matrix: np.ndarray[
        tuple[int, int],
        np.dtype[np.complex128]
        | np.dtype[np.float64]
        | np.dtype[np.float64 | np.complex128],
    ],
    initial_basis: BasisLike[Any, Any],
    final_basis: BasisLike[Any, Any],
    initial_dual_basis: BasisLike[Any, Any],
    final_dual_basis: BasisLike[Any, Any],
    *,
    axes: tuple[int, int] = (0, 1),
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
    """
    Convert a matrix from initial_basis to final_basis.

    Parameters
    ----------
    matrix : np.ndarray[tuple[int, int], np.dtype[np.complex_]]
    initial_basis : _B3d0Inv
    final_basis : _B3d1Inv

    Returns
    -------
    np.ndarray[tuple[int, int], np.dtype[np.complex_]]
    """
    converted = convert_vector(matrix, initial_basis, final_basis, axis=axes[0])
    return convert_dual_vector(
        converted, initial_dual_basis, final_dual_basis, axis=axes[1]
    )  # type: ignore[return-value]


@runtime_checkable
class BasisWithLengthLike(
    BasisLike[_NF0_co, _N0_co], Protocol[_NF0_co, _N0_co, _ND0Inv]
):
    """A generic object that represents an axis with a corresponding axis vector."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(delta_x={self.delta_x.__repr__()}, n={self.n.__repr__()}, fundamental_n={self.fundamental_n.__repr__()})"

    @property
    @abc.abstractmethod
    def delta_x(self) -> AxisVector[_ND0Inv]:
        ...


BasisWithLengthLike1d = BasisWithLengthLike[_NF0Inv, _N0Inv, Literal[1]]
BasisWithLengthLike2d = BasisWithLengthLike[_NF0Inv, _N0Inv, Literal[2]]
BasisWithLengthLike3d = BasisWithLengthLike[_NF0Inv, _N0Inv, Literal[3]]
