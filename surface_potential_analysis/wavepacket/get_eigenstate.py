from __future__ import annotations

import itertools
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar, cast, overload

import numpy as np
from slate.basis.stacked._tuple_basis import VariadicTupleBasis

from surface_potential_analysis.basis.legacy import (
    BasisLike,
    EvenlySpacedTransformedPositionBasis,
    ExplicitStackedBasisWithLength,
    FundamentalBasis,
    FundamentalTransformedBasis,
    FundamentalTransformedPositionBasis,
    StackedBasisLike,
    StackedBasisWithVolumeLike,
    TupleBasis,
    TupleBasisLike,
    TupleBasisWithLengthLike,
    convert_vector,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.operator.conversion import (
    convert_diagonal_operator_to_basis,
)
from surface_potential_analysis.stacked_basis.conversion import (
    tuple_basis_as_fundamental,
    tuple_basis_as_transformed_fundamental,
)
from surface_potential_analysis.stacked_basis.util import (
    wrap_index_around_origin,
)
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_list_to_basis,
    convert_state_vector_to_position_basis,
)
from surface_potential_analysis.state_vector.state_vector_list import (
    StateVectorList,
    as_state_vector_list,
    get_state_vector,
)
from surface_potential_analysis.types import (
    IntLike_co,
    SingleFlatIndexLike,
)
from surface_potential_analysis.wavepacket.conversion import (
    convert_wavepacket_to_basis,
    convert_wavepacket_to_fundamental_momentum_basis,
)
from surface_potential_analysis.wavepacket.eigenstate_conversion import (
    unfurl_wavepacket_list,
)
from surface_potential_analysis.wavepacket.localization_operator import (
    get_localized_hamiltonian_from_eigenvalues,
    get_localized_wavepackets,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    BlochWavefunctionList,
    BlochWavefunctionListBasis,
    BlochWavefunctionListList,
    BlochWavefunctionListWithEigenvalues,
    BlochWavefunctionListWithEigenvaluesList,
    get_fundamental_sample_basis,
    get_fundamental_unfurled_basis,
    get_wavepacket_basis,
    wavepacket_list_into_iter,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.legacy import FundamentalPositionBasis
    from surface_potential_analysis.operator.operator import SingleBasisDiagonalOperator
    from surface_potential_analysis.operator.operator_list import (
        OperatorList,
        SingleBasisDiagonalOperatorList,
    )
    from surface_potential_analysis.state_vector.eigenstate_list import Eigenstate
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.types import (
        SingleIndexLike,
        SingleStackedIndexLike,
    )
    from surface_potential_analysis.wavepacket.localization_operator import (
        LocalizationOperator,
    )

    _FB0 = TypeVar("_FB0", bound=FundamentalBasis[Any])
    _B0Inv = TypeVar("_B0Inv", bound=BasisLike)

    _SB0 = TypeVar("_SB0", bound=StackedBasisLike)
    _SBV0 = TypeVar("_SBV0", bound=StackedBasisWithVolumeLike)
    _TB0 = TypeVar("_TB0", bound=TupleBasisLike[*tuple[Any, ...]])

    _B1 = TypeVar("_B1", bound=BasisLike)
    _B2 = TypeVar("_B2", bound=BasisLike)

_B0 = TypeVar("_B0", bound=BasisLike)


def _get_sampled_basis(
    basis: BlochWavefunctionListBasis[_TB0, _SBV0],
    offset: tuple[IntLike_co, ...],
) -> TupleBasisWithLengthLike[*tuple[EvenlySpacedTransformedPositionBasis, ...]]:
    basis_x = tuple_basis_as_fundamental(basis[1])

    return TupleBasis(
        *tuple(
            EvenlySpacedTransformedPositionBasis(
                state_ax.delta_x * list_ax.n,
                n=state_ax.n,
                step=list_ax.n,
                offset=wrap_index_around_origin(
                    VariadicTupleBasis((FundamentalBasis(list_ax.n), None)),
                    (o,),
                    origin=0,
                )[0],
            )
            for (list_ax, state_ax, o) in zip(basis[0], basis_x, offset, strict=True)
        )
    )


def get_wavepacket_state_vector(
    wavepacket: BlochWavefunctionList[_SB0, _SBV0], idx: SingleIndexLike
) -> StateVector[
    TupleBasisWithLengthLike[*tuple[EvenlySpacedTransformedPositionBasis, ...]]
]:
    """
    Get the eigenstate of a given wavepacket at a specific index.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]
    idx : SingleIndexLike

    Returns
    -------
    Eigenstate[_B0Inv].
    """
    converted = convert_wavepacket_to_fundamental_momentum_basis(
        wavepacket,
        list_basis=tuple_basis_as_transformed_fundamental(wavepacket["basis"][0]),
    )
    util = BasisUtil(converted["basis"][0])
    idx = util.get_flat_index(idx) if isinstance(idx, tuple) else idx
    offset = util.get_stacked_index(idx)

    basis = _get_sampled_basis(converted["basis"], offset)
    return {
        "basis": basis,
        "data": converted["data"].reshape(converted["basis"].shape)[idx],
    }


@overload
def get_bloch_state_vector(
    wavepacket: BlochWavefunctionList[_SB0, _SBV0], idx: SingleFlatIndexLike
) -> StateVector[_SBV0]: ...


@overload
def get_bloch_state_vector(
    wavepacket: BlochWavefunctionList[_TB0, _SBV0], idx: SingleIndexLike
) -> StateVector[_SBV0]: ...


def get_bloch_state_vector(
    wavepacket: BlochWavefunctionList[_SB0, _SBV0], idx: SingleIndexLike
) -> StateVector[_SBV0]:
    """
    Get the eigenstate of a given wavepacket at a specific index.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]
    idx : SingleIndexLike

    Returns
    -------
    Eigenstate[_B0Inv].
    """
    util = BasisUtil(cast(TupleBasisLike[*tuple[Any, ...]], wavepacket["basis"][0]))
    idx = util.get_flat_index(idx) if isinstance(idx, tuple) else idx
    return get_state_vector(wavepacket, idx)


def get_all_eigenstates(
    wavepacket: BlochWavefunctionListWithEigenvalues[_SB0, _SBV0],
) -> list[
    Eigenstate[TupleBasisLike[*tuple[EvenlySpacedTransformedPositionBasis, ...]]]
]:
    """
    Get the eigenstate of a given wavepacket at a specific index.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]

    Returns
    -------
    Eigenstate[_B0Inv].
    """
    converted = convert_wavepacket_to_fundamental_momentum_basis(
        wavepacket,
        list_basis=tuple_basis_as_transformed_fundamental(wavepacket["basis"][0]),
    )
    util = BasisUtil(get_fundamental_sample_basis(converted["basis"]))
    return [
        {
            "basis": _get_sampled_basis(
                converted["basis"], cast(tuple[int, ...], offset)
            ),
            "data": v,
            "eigenvalue": e,
        }
        for (v, e, *offset) in zip(
            converted["data"],
            wavepacket["eigenvalue"],
            *util.stacked_nk_points,
            strict=True,
        )
    ]


def get_all_wavepacket_states(
    wavepacket: BlochWavefunctionList[_SB0, _SBV0],
) -> list[StateVector[TupleBasisWithLengthLike[*tuple[Any, ...]]]]:
    """
    Get the eigenstate of a given wavepacket at a specific index.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]

    Returns
    -------
    Eigenstate[_B0Inv].
    """
    converted = convert_wavepacket_to_fundamental_momentum_basis(
        wavepacket,
        list_basis=tuple_basis_as_transformed_fundamental(wavepacket["basis"][0]),
    )
    util = BasisUtil(get_fundamental_sample_basis(converted["basis"]))
    return [
        {
            "basis": _get_sampled_basis(
                converted["basis"], cast(tuple[IntLike_co, ...], offset)
            ),
            "data": v,
        }
        for (v, *offset) in zip(
            converted["data"].reshape(converted["basis"].shape),
            *util.stacked_nk_points,
            strict=True,
        )
    ]


def get_tight_binding_state(
    wavepacket: BlochWavefunctionList[_SB0, _SBV0],
    idx: SingleIndexLike = 0,
    origin: SingleIndexLike | None = None,
) -> StateVector[TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]]]:
    """
    Given a wavepacket, get the state corresponding to the eigenstate under the tight binding approximation.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]
        The initial wavepacket
    idx : SingleIndexLike, optional
        The index of the state vector to use as reference, by default 0
    origin : SingleIndexLike | None, optional
        The origin about which to produce the localized state, by default the maximum of the wavefunction

    Returns
    -------
    StateVector[tuple[FundamentalPositionBasis, ...]]
        The localized state under the tight binding approximation
    """
    state_0 = convert_state_vector_to_position_basis(
        get_wavepacket_state_vector(wavepacket, idx)
    )
    util = BasisUtil(state_0["basis"])
    if origin is None:
        idx_0: SingleStackedIndexLike = util.get_stacked_index(
            int(np.argmax(np.abs(state_0["data"]), axis=-1))
        )
        origin = wrap_index_around_origin(
            wavepacket["basis"][1],
            idx_0,
        )
    # Under the tight binding approximation all state vectors are equal.
    # The corresponding localized state is just the state at some index
    # truncated to a single unit cell
    unit_cell_util = BasisUtil(wavepacket["basis"])
    relevant_idx = wrap_index_around_origin(
        wavepacket["basis"][1],
        unit_cell_util.fundamental_stacked_nx_points,
        origin=origin,
    )
    relevant_idx_flat = util.get_flat_index(relevant_idx, mode="wrap")
    out: StateVector[
        TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]]
    ] = {
        "basis": state_0["basis"],
        "data": np.zeros_like(state_0["data"]),
    }
    out["data"][relevant_idx_flat] = state_0["data"][relevant_idx_flat]
    return out


def get_tight_binding_states(
    wavepacket: BlochWavefunctionListList[_B0, _SB0, _SBV0],
    idx: SingleIndexLike = 0,
    origin: SingleIndexLike | None = None,
) -> StateVectorList[
    _B0, TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]]
]:
    """Get all tight binding states."""
    out = as_state_vector_list(
        [
            get_tight_binding_state(s, idx, origin)
            for s in wavepacket_list_into_iter(wavepacket)
        ]
    )
    return {
        "basis": VariadicTupleBasis((wavepacket["basis"][0][0], out["basis"][1]), None),
        "data": out["data"],
    }


def get_states_at_bloch_idx(
    wavepackets: BlochWavefunctionListList[
        _B0Inv,
        TupleBasisLike[*tuple[_FB0, ...]],
        _SBV0,
    ],
    idx: SingleIndexLike,
) -> StateVectorList[
    _B0Inv,
    TupleBasisWithLengthLike[*tuple[EvenlySpacedTransformedPositionBasis, ...]],
]:
    """
    Get all wavepacket states at the given bloch index.

    Returns
    -------
    StateVectorList[
    _B0Inv,
    TupleBasisLike[
        *tuple[EvenlySpacedTransformedPositionBasis, ...]
    ],
    ]
    """
    util = BasisUtil(wavepackets["basis"][0][1])
    idx = util.get_flat_index(idx) if isinstance(idx, tuple) else idx
    offset = util.get_stacked_index(idx)
    # we should support _SB0Inv not in fundamental, and make it so we dont need to convert to momentum basis

    converted = convert_state_vector_list_to_basis(
        wavepackets,
        stacked_basis_as_transformed_basis(wavepackets["basis"][1]),
    )
    return {
        "basis": VariadicTupleBasis(
            (
                converted["basis"][0][0],
                _get_sampled_basis(get_wavepacket_basis(converted["basis"]), offset),
            ),
            None,
        ),
        "data": converted["data"]
        .reshape(*converted["basis"][0].shape, -1)[:, idx]
        .reshape(-1),
    }


def _get_compressed_bloch_states_at_bloch_idx(
    wavepackets: BlochWavefunctionListList[_B0, _SB0, _SBV0], idx: int
) -> StateVectorList[_B0, _SBV0]:
    return {
        "basis": VariadicTupleBasis(
            (wavepackets["basis"][0][0], wavepackets["basis"][1]), None
        ),
        "data": wavepackets["data"]
        .reshape(*wavepackets["basis"][0].shape, -1)[:, idx, :]
        .ravel(),
    }


def _get_fundamental_wavepacket_basis(
    wavepackets: BlochWavefunctionListList[_B0, _SB0, _SBV0],
) -> TupleBasisLike[
    TupleBasisLike[*tuple[FundamentalTransformedBasis, ...]],
    TupleBasisWithLengthLike[*tuple[FundamentalTransformedPositionBasis, ...]],
]:
    converted_basis = stacked_basis_as_transformed_basis(wavepackets["basis"][1])
    converted_list_basis = tuple_basis_as_transformed_fundamental(
        wavepackets["basis"][0][1]
    )
    return VariadicTupleBasis((converted_list_basis, converted_basis), None)


def get_fundamental_wavepacket(
    wavepackets: BlochWavefunctionListList[_B0, _SB0, _SBV0],
) -> BlochWavefunctionListList[
    _B0,
    TupleBasisLike[*tuple[FundamentalTransformedBasis, ...]],
    TupleBasisWithLengthLike[*tuple[FundamentalTransformedPositionBasis, ...]],
]:
    """Get the wavepacket in the fundamental (transformed) basis."""
    basis = _get_fundamental_wavepacket_basis(wavepackets)
    converted_basis = basis[1]
    converted_list_basis = basis[0]
    return convert_wavepacket_to_basis(
        wavepackets,
        basis=converted_basis,
        list_basis=VariadicTupleBasis(
            (wavepackets["basis"][0][0], converted_list_basis), None
        ),
    )


def get_bloch_states(
    wavepackets: BlochWavefunctionListList[_B0, _SB0, _SBV0],
) -> StateVectorList[
    TupleBasisLike[_B0, TupleBasisLike[*tuple[FundamentalTransformedBasis, ...]]],
    TupleBasisWithLengthLike[*tuple[FundamentalTransformedPositionBasis, ...]],
]:
    """
    Uncompress bloch wavefunction list.

    A bloch wavefunction list is implicitly compressed, as each wavefunction in the list
    only stores the state at the relevant non-zero bloch k. This function undoes this implicit
    compression

    Parameters
    ----------
    wavepacket : BlochWavefunctionListList[_B0, _SB0, _SB1]
        The wavepacket to decompress

    Returns
    -------
    StateVectorList[
    TupleBasisLike[_B0, _SB0],
    TupleBasisLike[*tuple[FundamentalTransformedPositionBasis, ...]],
    ]
    """
    converted = get_fundamental_wavepacket(wavepackets)

    decompressed_basis = TupleBasis(
        *tuple(
            FundamentalTransformedPositionBasis(b1.delta_x * b0.n, b0.n * b1.n)
            for (b0, b1) in zip(
                converted["basis"][0][1], converted["basis"][1], strict=True
            )
        )
    )
    out = np.zeros(
        (*wavepackets["basis"][0].shape, decompressed_basis.n), dtype=np.complex128
    )

    util = BasisUtil(converted["basis"][0][1])
    # for each bloch k
    for idx in range(converted["basis"][0][1].n):
        offset = util.get_stacked_index(idx)

        states = _get_compressed_bloch_states_at_bloch_idx(converted, idx)
        # Re-interpret as a sampled state, and convert to a full state
        full_states = convert_state_vector_list_to_basis(
            {
                "basis": TupleBasis(
                    states["basis"][0],
                    _get_sampled_basis(
                        VariadicTupleBasis(
                            (converted["basis"][0][1], converted["basis"][1]), None
                        ),
                        offset,
                    ),
                ),
                "data": states["data"],
            },
            decompressed_basis,
        )

        out[:, idx, :] = full_states["data"].reshape(
            wavepackets["basis"][0][0].n, decompressed_basis.n
        )

    return {
        "basis": VariadicTupleBasis((converted["basis"][0], decompressed_basis), None),
        "data": out.ravel(),
    }


_S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])


class BlochBasis(
    ExplicitStackedBasisWithLength[
        TupleBasisLike[_B0, TupleBasisLike[*tuple[FundamentalTransformedBasis, ...]]],
        TupleBasisWithLengthLike[*tuple[FundamentalTransformedPositionBasis, ...]],
    ],
    Generic[_B0],
):
    """An basis with vectors given as explicit states."""

    def __init__(
        self,
        wavefunctions: BlochWavefunctionListList[_B0, _SB0, _SBV0],
    ) -> None:
        converted_list_basis = tuple_basis_as_transformed_fundamental(
            wavefunctions["basis"][0][1]
        )
        self._wavefunctions = convert_wavepacket_to_basis(
            wavefunctions,
            list_basis=VariadicTupleBasis(
                (wavefunctions["basis"][0][0], converted_list_basis), None
            ),
        )

    @cached_property
    def _vectors(
        self: Self,
    ) -> StateVectorList[
        TupleBasisLike[_B0, TupleBasisLike[*tuple[FundamentalTransformedBasis, ...]]],
        TupleBasisWithLengthLike[*tuple[FundamentalTransformedPositionBasis, ...]],
    ]:
        return get_bloch_states(self.wavefunctions)

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
        axis = axis % vectors.ndim

        # The vectors in fundamental unfurled basis [(momentum,list), (momentum,list),...]
        vectors_in_basis = self.vectors_basis[1].__from_fundamental__(vectors, axis)

        k_shape = self.wavefunctions["basis"][1].fundamental_shape
        list_shape = self.wavefunctions["basis"][0][1].fundamental_shape

        inner_shape = tuple((n_k, n_list) for (n_k, n_list) in zip(k_shape, list_shape))

        shift = tuple(
            (n_list // 2 + (n_list * (n_s // 2))) for (n_s, n_list) in inner_shape
        )
        shifted = np.roll(
            vectors_in_basis.reshape(
                *vectors_in_basis.shape[:axis],
                *(np.prod(x) for x in inner_shape),
                *vectors_in_basis.shape[axis + 1 :],
            ),
            shift,
            axis=tuple(range(axis, axis + len(inner_shape))),
        ).reshape(
            *vectors_in_basis.shape[:axis],
            *(itertools.chain(*inner_shape)),
            *vectors_in_basis.shape[axis + 1 :],
        )

        # Flip the vectors into basis [(list, list, ...), (momentum, momentum, ...)]
        flipped = shifted.transpose(
            *range(axis),
            *range(axis + 1, axis + 2 * len(inner_shape), 2),
            *range(axis, axis + 2 * len(inner_shape), 2),
            *range(axis + 2 * len(inner_shape), shifted.ndim),
        )
        unshifted = np.fft.ifftshift(
            flipped,
            axes=tuple(range(axis, axis + 2 * len(inner_shape))),
        )
        # Convert from fundamental to wavepacket basis
        converted = convert_vector(
            unshifted.reshape(vectors_in_basis.shape),
            _get_fundamental_wavepacket_basis(self.wavefunctions),
            get_wavepacket_basis(self.wavefunctions["basis"]),
            axis=axis,
        )

        if self.vectors_basis.n < 100 * 90:
            np.testing.assert_allclose(
                np.apply_along_axis(
                    lambda x: np.einsum(  # type: ignore lib
                        "jk,ijk->ij",
                        x.reshape(
                            self.wavefunctions["basis"][0][1].n,
                            self.wavefunctions["basis"][1].size,
                        ),
                        np.conj(
                            self.wavefunctions["data"].reshape(
                                *self.wavefunctions["basis"][0].shape,
                                self.wavefunctions["basis"][1].size,
                            )
                        ),
                    ).ravel(),
                    axis,
                    converted,
                ),
                ExplicitStackedBasisWithLength(self.vectors).__from_fundamental__(
                    vectors, axis
                ),
                atol=1e-10,
            )

        return np.apply_along_axis(
            lambda x: np.einsum(  # type: ignore lib
                "jk,ijk->ij",
                x.reshape(
                    self.wavefunctions["basis"][0][1].n,
                    self.wavefunctions["basis"][1].size,
                ),
                np.conj(
                    self.wavefunctions["data"].reshape(
                        *self.wavefunctions["basis"][0].shape,
                        self.wavefunctions["basis"][1].size,
                    )
                ),
            ).ravel(),
            axis,
            converted,
        )

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
        axis = axis % vectors.ndim

        vectors_in_basis = np.apply_along_axis(
            lambda x: np.einsum(  # type: ignore lib
                "ij,ijk->jk",
                x.reshape(
                    self.wavefunctions["basis"][0].shape,
                ),
                self.wavefunctions["data"].reshape(
                    *self.wavefunctions["basis"][0].shape,
                    self.wavefunctions["basis"][1].size,
                ),
            ).ravel(),
            axis,
            vectors,
        )
        # The vectors in basis [(list, list, ...), (momentum, momentum, ...)]
        converted = convert_vector(
            vectors_in_basis,
            get_wavepacket_basis(self.wavefunctions["basis"]),
            _get_fundamental_wavepacket_basis(self.wavefunctions),
            axis=axis,
        )

        k_shape = self.wavefunctions["basis"][1].shape
        list_shape = self.wavefunctions["basis"][0][1].shape
        ndim = len(k_shape)

        inner_shape = tuple(zip(list_shape, k_shape))
        shifted = np.fft.fftshift(
            converted.reshape(
                *vectors_in_basis.shape[:axis],
                *tuple(item for pair in zip(*inner_shape) for item in pair),
                *vectors_in_basis.shape[axis + 1 :],
            ),
            axes=tuple(range(axis, axis + 2 * ndim)),
        )

        # Convert the vectors into fundamental unfurled basis
        # ie [(momentum,list), (momentum,list),...
        transposed = shifted.transpose(
            *range(axis),
            *itertools.chain(*((axis + i + ndim, axis + i) for i in range(ndim))),
            *range(axis + 2 * len(inner_shape), shifted.ndim),
        )

        # Shift the 0 frequency back to the start and flatten
        # Note the ifftshift would shift by (list_shape[i] * states_shape[i]) // 2
        # Which is wrong in this case
        shift = tuple(
            -(n_list // 2 + (n_list * (n_s // 2))) for (n_list, n_s) in inner_shape
        )
        unshifted = np.roll(
            transposed.reshape(
                *vectors_in_basis.shape[:axis],
                *tuple(np.prod(x) for x in inner_shape),
                *vectors_in_basis.shape[axis + 1 :],
            ),
            shift,
            tuple(range(axis, axis + ndim)),
        )
        if self.vectors_basis.n < 100 * 90:
            np.testing.assert_allclose(
                self.vectors_basis[1].__into_fundamental__(
                    unshifted.reshape(vectors_in_basis.shape), axis
                ),
                ExplicitStackedBasisWithLength(self.vectors).__into_fundamental__(
                    vectors, axis
                ),
                atol=1e-10,
            )

        return self.vectors_basis[1].__into_fundamental__(
            unshifted.reshape(vectors_in_basis.shape), axis
        )

    @property
    def vectors_basis(
        self: Self,
    ) -> TupleBasis[
        TupleBasisLike[_B0, TupleBasisLike[*tuple[FundamentalTransformedBasis, ...]]],
        TupleBasisWithLengthLike[*tuple[FundamentalTransformedPositionBasis, ...]],
    ]:
        """Get the basis of the full states list."""
        return TupleBasis(
            self.wavefunctions["basis"][0],
            self.unfurled_basis,
        )

    @property
    def unfurled_basis(
        self: Self,
    ) -> TupleBasisWithLengthLike[*tuple[FundamentalTransformedPositionBasis, ...]]:
        """Get the basis of the full states."""
        return get_fundamental_unfurled_basis(
            get_wavepacket_basis(self.wavefunctions["basis"])
        )

    @property
    def wavefunctions(
        self: Self,
    ) -> BlochWavefunctionListList[
        _B0,
        TupleBasisLike[*tuple[FundamentalTransformedBasis, ...]],
        StackedBasisWithVolumeLike,
    ]:
        """Get the raw wavefunctions."""
        return self._wavefunctions

    def vectors_at_bloch_k(
        self: Self, idx: SingleIndexLike
    ) -> StateVectorList[
        _B0,
        StackedBasisWithVolumeLike,
    ]:
        """Get the states at bloch idx."""
        util = BasisUtil(self.wavefunctions["basis"][0][1])
        idx = util.get_flat_index(idx, mode="wrap") if isinstance(idx, tuple) else idx
        data = self.wavefunctions["data"].reshape(
            *self.wavefunctions["basis"][0].shape, -1
        )[:, idx, :]
        return {
            "basis": TupleBasis(
                self.wavefunctions["basis"][0][0], self.wavefunctions["basis"][1]
            ),
            "data": data.ravel(),
        }

    def basis_at_bloch_k(
        self: Self, idx: tuple[int, ...]
    ) -> ExplicitStackedBasisWithLength[
        _B0,
        StackedBasisWithVolumeLike,
    ]:
        """Get the basis at bloch idx."""
        return ExplicitStackedBasisWithLength(self.vectors_at_bloch_k(idx))


def get_bloch_basis(
    wavefunctions: BlochWavefunctionListList[_B0, _SB0, _SBV0],
) -> BlochBasis[_B0]:
    """
    Get the basis, with the bloch wavefunctions as eigenstates.

    Returns
    -------
    ExplicitStackedBasisWithLength[
        TupleBasisLike[_B0, _SB0],
        TupleBasisLike[*tuple[FundamentalTransformedPositionBasis, ...]],
    ]
    """
    return BlochBasis(wavefunctions)


def get_bloch_hamiltonian(
    wavepackets: BlochWavefunctionListWithEigenvaluesList[_B0, _SB0, _SBV0],
) -> SingleBasisDiagonalOperatorList[_B0, _SB0]:
    """
    Get the Hamiltonian in the Wavepacket basis.

    This is a list of Hamiltonian, one for each bloch k

    Parameters
    ----------
    wavepackets : WavepacketWithEigenvaluesList[_B0, _SB1, _SB0]

    Returns
    -------
    SingleBasisDiagonalOperatorList[_B0, _SB1]
    """
    return {
        "basis": TupleBasis(
            wavepackets["basis"][0][0],
            VariadicTupleBasis(
                (wavepackets["basis"][0][1], wavepackets["basis"][0][1]), None
            ),
        ),
        "data": wavepackets["eigenvalue"],
    }


def get_full_bloch_hamiltonian(
    wavefunctions: BlochWavefunctionListWithEigenvaluesList[_B0, _SB0, _SBV0],
) -> SingleBasisDiagonalOperator[BlochBasis[_B0]]:
    """
    Get the hamiltonian in the full bloch basis.

    Returns
    -------
    SingleBasisDiagonalOperator[
    ExplicitStackedBasisWithLength[
        TupleBasisLike[_B0, _SB0],
        TupleBasisLike[*tuple[FundamentalTransformedPositionBasis, ...]],
    ]
    ]
    """
    basis = get_bloch_basis(wavefunctions)

    return {
        "basis": VariadicTupleBasis((basis, basis), None),
        "data": wavefunctions["eigenvalue"],
    }


def get_wannier_states(
    wavefunctions: BlochWavefunctionListList[_B2, _SB0, _SBV0],
    operator: LocalizationOperator[_SB0, _B1, _B2],
) -> StateVectorList[
    TupleBasisLike[_B1, _SB0],
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
]:
    """Get the wannier states."""
    localized = get_localized_wavepackets(wavefunctions, operator)

    fundamental_states = unfurl_wavepacket_list(localized)

    converted_fundamental = convert_state_vector_list_to_basis(
        fundamental_states,
        tuple_basis_as_fundamental(fundamental_states["basis"][1]),
    )
    converted_stacked = converted_fundamental["data"].reshape(
        operator["basis"][1][0].n, *converted_fundamental["basis"][1].shape
    )
    data = np.zeros(
        (
            operator["basis"][1][0].n,  # Wannier idx
            operator["basis"][0].size,  # Translation
            *converted_fundamental["basis"][1].shape,  # Wavefunction
        ),
        dtype=np.complex128,
    )

    util = BasisUtil(operator["basis"][0])
    # for each translation of the wannier functions
    for idx in range(operator["basis"][0].size):
        offset = util.get_stacked_index(idx)
        shift = tuple(-n * o for (n, o) in zip(wavefunctions["basis"][1].shape, offset))

        tanslated = np.roll(
            converted_stacked, shift, axis=tuple(1 + x for x in range(util.ndim))
        )

        data[:, idx, :] = tanslated

    return {
        "basis": TupleBasis(
            VariadicTupleBasis((operator["basis"][1][0], operator["basis"][0]), None),
            converted_fundamental["basis"][1],
        ),
        "data": data.ravel(),
    }


def get_wannier_basis(
    wavefunctions: BlochWavefunctionListList[_B2, _SB0, _SBV0],
    operator: LocalizationOperator[_SB0, _B1, _B2],
) -> ExplicitStackedBasisWithLength[
    TupleBasisLike[_B1, _SB0],
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
]:
    """
    Get the basis, with the localized (wannier) states as eigenstates.

    Returns
    -------
    ExplicitStackedBasisWithLength[
        TupleBasisLike[_B0, _SB0],
        TupleBasisLike[*tuple[FundamentalTransformedPositionBasis, ...]],
    ]
    """
    return ExplicitStackedBasisWithLength(get_wannier_states(wavefunctions, operator))


def get_wannier_hamiltonian(
    wavefunctions: BlochWavefunctionListWithEigenvaluesList[_B2, _SB0, _SBV0],
    operator: LocalizationOperator[_SB0, _B1, _B2],
) -> OperatorList[_SB0, _B1, _B1]:
    """
    Get the hamiltonian of a wavepacket after applying the localization operator.

    Parameters
    ----------
    wavepackets : WavepacketWithEigenvaluesList[_B2, _SB1, _SB0]
    operator : LocalizationOperator[_SB1, _B1, _B2]

    Returns
    -------
    OperatorList[_SB1, _B1, _B1]
    """
    hamiltonian = get_bloch_hamiltonian(wavefunctions)
    return get_localized_hamiltonian_from_eigenvalues(hamiltonian, operator)


def get_full_wannier_hamiltonian(
    wavefunctions: BlochWavefunctionListWithEigenvaluesList[_B2, _SB0, _SBV0],
    operator: LocalizationOperator[_SB0, _B1, _B2],
) -> SingleBasisDiagonalOperator[
    ExplicitStackedBasisWithLength[
        TupleBasisLike[_B1, _SB0],
        TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
    ]
]:
    """
    Get the hamiltonian in the full bloch basis.

    Returns
    -------
    SingleBasisDiagonalOperator[
    ExplicitStackedBasisWithLength[
        TupleBasisLike[_B0, _SB0],
        TupleBasisLike[*tuple[FundamentalTransformedPositionBasis, ...]],
    ]
    ]
    """
    basis = get_wannier_basis(wavefunctions, operator)
    hamiltonian = get_wannier_hamiltonian(wavefunctions, operator)
    hamiltonian_2d = np.einsum(  # type: ignore lib
        "ik,ij->ijk",
        hamiltonian["data"].reshape(hamiltonian["basis"][0].size, -1),
        np.eye(hamiltonian["basis"][0].size),
    )

    hamiltonian_stacked = hamiltonian_2d.reshape(
        *hamiltonian["basis"][0].shape,
        *hamiltonian["basis"][0].shape,
        -1,
    )
    # is this correct...
    # I think because H is real symmetric, this ultimately doesn't matter
    data_stacked = np.fft.fftn(
        np.fft.ifftn(
            hamiltonian_stacked,
            axes=tuple(range(hamiltonian["basis"][0].sizedim)),
            norm="ortho",
        ),
        axes=tuple(
            range(hamiltonian["basis"][0].sizedim, 2 * hamiltonian["basis"][0].sizedim)
        ),
        norm="ortho",
    )
    # Re order to match the correct basis
    data = np.einsum(  # type: ignore lib
        "ijkl->ikjl",
        data_stacked.reshape(
            hamiltonian["basis"][0].size,
            hamiltonian["basis"][0].size,
            *hamiltonian["basis"][1].shape,
        ),
    )
    return {
        "basis": VariadicTupleBasis((basis, basis), None),
        "data": data.ravel(),
    }
    # TODO: We can probably just fourier transform the wannier hamiltonian.  # noqa: FIX002
    # ie hamiltonian = get_wannier_hamiltonian(wavefunctions, operator)
    # This will be faster and have less artifacts
    basis = get_wannier_basis(wavefunctions, operator)
    hamiltonian = get_full_bloch_hamiltonian(wavefunctions)
    return convert_diagonal_operator_to_basis(
        hamiltonian, VariadicTupleBasis((basis, basis), None)
    )
