from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np

from surface_potential_analysis.basis.basis import (
    FundamentalBasis,
    FundamentalTransformedPositionBasis,
)
from surface_potential_analysis.basis.evenly_spaced_basis import (
    EvenlySpacedTransformedPositionBasis,
)
from surface_potential_analysis.basis.explicit_basis import (
    ExplicitStackedBasisWithLength,
)
from surface_potential_analysis.basis.stacked_basis import (
    TupleBasis,
    TupleBasisWithLengthLike,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.operator.conversion import convert_operator_to_basis
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_basis,
    stacked_basis_as_fundamental_momentum_basis,
    stacked_basis_as_fundamental_position_basis,
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
    get_state_vector,
)
from surface_potential_analysis.types import (
    IntLike_co,
)
from surface_potential_analysis.wavepacket.conversion import (
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
    get_sample_basis,
    get_wavepacket_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import FundamentalPositionBasis
    from surface_potential_analysis.basis.basis_like import (
        BasisLike,
        BasisWithLengthLike,
    )
    from surface_potential_analysis.basis.stacked_basis import (
        TupleBasisLike,
    )
    from surface_potential_analysis.operator.operator import SingleBasisDiagonalOperator
    from surface_potential_analysis.operator.operator_list import (
        OperatorList,
        SingleBasisDiagonalOperatorList,
    )
    from surface_potential_analysis.state_vector.eigenstate_collection import Eigenstate
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.types import (
        SingleIndexLike,
        SingleStackedIndexLike,
    )
    from surface_potential_analysis.wavepacket.localization_operator import (
        LocalizationOperator,
    )

    _SBL1 = TypeVar(
        "_SBL1",
        bound=TupleBasisLike[*tuple[Any, ...]],
    )
    _BL0 = TypeVar(
        "_BL0",
        bound=BasisWithLengthLike[Any, Any, Any],
    )
    _FB0 = TypeVar("_FB0", bound=FundamentalBasis[Any])
    _FTB0 = TypeVar("_FTB0", bound=FundamentalTransformedPositionBasis[Any, Any])
    _B0Inv = TypeVar("_B0Inv", bound=BasisLike[Any, Any])
    _SB0 = TypeVar("_SB0", bound=TupleBasisLike[*tuple[Any, ...]])
    _SB1 = TypeVar("_SB1", bound=TupleBasisWithLengthLike[*tuple[Any, ...]])

    _B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
    _B1 = TypeVar("_B1", bound=BasisLike[Any, Any])
    _B2 = TypeVar("_B2", bound=BasisLike[Any, Any])


def _get_sampled_basis(
    basis: BlochWavefunctionListBasis[
        TupleBasisLike[*tuple[_B0, ...]], TupleBasisLike[*tuple[_BL0, ...]]
    ],
    offset: tuple[IntLike_co, ...],
) -> TupleBasisLike[
    *tuple[EvenlySpacedTransformedPositionBasis[Any, Any, Any, Any], ...]
]:
    return TupleBasis(
        *tuple(
            EvenlySpacedTransformedPositionBasis[Any, Any, Any, Any](
                state_ax.delta_x * list_ax.n,
                n=state_ax.n,
                step=list_ax.n,
                offset=wrap_index_around_origin(
                    TupleBasis(FundamentalBasis(list_ax.n)), (o,), 0
                )[0],
            )
            for (list_ax, state_ax, o) in zip(basis[0], basis[1], offset, strict=True)
        )
    )


def get_wavepacket_state_vector(
    wavepacket: BlochWavefunctionList[_SB0, _SB1], idx: SingleIndexLike
) -> StateVector[
    TupleBasisLike[
        *tuple[EvenlySpacedTransformedPositionBasis[Any, Any, Any, Any], ...]
    ]
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
        list_basis=stacked_basis_as_fundamental_basis(wavepacket["basis"][0]),
    )
    util = BasisUtil(converted["basis"][0])
    idx = util.get_flat_index(idx) if isinstance(idx, tuple) else idx
    offset = util.get_stacked_index(idx)

    basis = _get_sampled_basis(converted["basis"], offset)
    return {
        "basis": basis,
        "data": converted["data"].reshape(converted["basis"].shape)[idx],
    }


def get_bloch_state_vector(
    wavepacket: BlochWavefunctionList[_SB0, _SBL1], idx: SingleIndexLike
) -> StateVector[_SBL1]:
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
    util = BasisUtil(wavepacket["basis"][0])
    idx = util.get_flat_index(idx) if isinstance(idx, tuple) else idx
    return get_state_vector(wavepacket, idx)


def get_all_eigenstates(
    wavepacket: BlochWavefunctionListWithEigenvalues[_SB0, _SBL1],
) -> list[
    Eigenstate[
        TupleBasisLike[
            *tuple[EvenlySpacedTransformedPositionBasis[Any, Any, Any, Any], ...]
        ]
    ]
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
        list_basis=stacked_basis_as_fundamental_basis(wavepacket["basis"][0]),
    )
    util = BasisUtil(get_sample_basis(converted["basis"]))
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
    wavepacket: BlochWavefunctionList[_SB0, _SBL1],
) -> list[StateVector[TupleBasisLike[*tuple[Any, ...]]]]:
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
        list_basis=stacked_basis_as_fundamental_basis(wavepacket["basis"][0]),
    )
    util = BasisUtil(get_sample_basis(converted["basis"]))
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
    wavepacket: BlochWavefunctionList[_SB0, _SBL1],
    idx: SingleIndexLike = 0,
    origin: SingleIndexLike | None = None,
) -> StateVector[TupleBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]]:
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
    StateVector[tuple[FundamentalPositionBasis[Any, Any], ...]]
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
        origin = wrap_index_around_origin(wavepacket["basis"], idx_0, (0, 0, 0), (0, 1))
    # Under the tight binding approximation all state vectors are equal.
    # The corresponding localized state is just the state at some index
    # truncated to a single unit cell
    unit_cell_util = BasisUtil(wavepacket["basis"])
    relevant_idx = wrap_index_around_origin(
        wavepacket["basis"],
        unit_cell_util.fundamental_stacked_nx_points,
        origin,
        (0, 1),  # type: ignore[arg-type]
    )
    relevant_idx_flat = util.get_flat_index(relevant_idx, mode="wrap")
    out: StateVector[
        TupleBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]
    ] = {
        "basis": state_0["basis"],
        "data": np.zeros_like(state_0["data"]),
    }
    out["data"][relevant_idx_flat] = state_0["data"][relevant_idx_flat]
    return out


def get_states_at_bloch_idx(
    wavepackets: BlochWavefunctionListList[
        _B0Inv,
        TupleBasisLike[*tuple[_FB0, ...]],
        TupleBasisLike[*tuple[_FTB0, ...]],
    ],
    idx: SingleIndexLike,
) -> StateVectorList[
    _B0Inv,
    TupleBasisLike[
        *tuple[EvenlySpacedTransformedPositionBasis[Any, Any, Any, Any], ...]
    ],
]:
    """
    Get all wavepacket states at the given bloch index.

    Returns
    -------
    StateVectorList[
    _B0Inv,
    TupleBasisLike[
        *tuple[EvenlySpacedTransformedPositionBasis[Any, Any, Any, Any], ...]
    ],
    ]
    """
    util = BasisUtil(wavepackets["basis"][0][1])
    idx = util.get_flat_index(idx) if isinstance(idx, tuple) else idx
    offset = util.get_stacked_index(idx)
    # TODO: support _SB0Inv not in fundamental, and make it so we dont need to convert to momentum basis

    converted = convert_state_vector_list_to_basis(
        wavepackets,
        stacked_basis_as_fundamental_momentum_basis(wavepackets["basis"][1]),
    )
    return {
        "basis": TupleBasis(
            converted["basis"][0][0],
            _get_sampled_basis(get_wavepacket_basis(converted), offset),
        ),
        "data": converted["data"]
        .reshape(*converted["basis"][0].shape, -1)[:, idx]
        .reshape(-1),
    }


def _get_compressed_bloch_states_at_bloch_idx(
    wavepackets: BlochWavefunctionListList[_B0, _SB0, _SB1], idx: int
) -> StateVectorList[_B0, _SB1]:
    return {
        "basis": TupleBasis(wavepackets["basis"][0][0], wavepackets["basis"][1]),
        "data": wavepackets["data"]
        .reshape(*wavepackets["basis"][0].shape, -1)[:, idx, :]
        .ravel(),
    }


def get_bloch_states(
    wavepackets: BlochWavefunctionListList[_B0, _SB0, _SB1],
) -> StateVectorList[
    TupleBasisLike[_B0, _SB0],
    TupleBasisLike[*tuple[FundamentalTransformedPositionBasis[Any, Any], ...]],
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
    TupleBasisLike[*tuple[FundamentalTransformedPositionBasis[Any, Any], ...]],
    ]
    """
    util = BasisUtil(wavepackets["basis"][0][1])

    converted_basis = stacked_basis_as_fundamental_momentum_basis(
        wavepackets["basis"][1]
    )
    converted = convert_state_vector_list_to_basis(wavepackets, converted_basis)

    decompressed_basis = TupleBasis(
        *tuple(
            FundamentalTransformedPositionBasis[Any, Any](
                b1.delta_x * b0.n, b0.n * b1.n
            )
            for (b0, b1) in zip(
                wavepackets["basis"][0][1], wavepackets["basis"][1], strict=True
            )
        )
    )
    out = np.zeros(
        (*wavepackets["basis"][0].shape, decompressed_basis.n), dtype=np.complex128
    )

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
                        TupleBasis(wavepackets["basis"][0][1], converted_basis),
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
        "basis": TupleBasis(wavepackets["basis"][0], decompressed_basis),
        "data": out.ravel(),
    }


def get_bloch_basis(
    wavefunctions: BlochWavefunctionListList[_B0, _SB0, _SB1],
) -> ExplicitStackedBasisWithLength[
    TupleBasisLike[_B0, _SB0],
    TupleBasisLike[*tuple[FundamentalTransformedPositionBasis[Any, Any], ...]],
]:
    """
    Get the basis, with the bloch wavefunctions as eigenstates.

    Returns
    -------
    ExplicitStackedBasisWithLength[
        TupleBasisLike[_B0, _SB0],
        TupleBasisLike[*tuple[FundamentalTransformedPositionBasis[Any, Any], ...]],
    ]
    """
    return ExplicitStackedBasisWithLength(get_bloch_states(wavefunctions))


def get_bloch_hamiltonian(
    wavepackets: BlochWavefunctionListWithEigenvaluesList[_B0, _SB1, _SB0],
) -> SingleBasisDiagonalOperatorList[_B0, _SB1]:
    """
    Get the Hamiltonian in the Wavepacket basis.

    This is a list of hamiltonians, one for each bloch k

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
            TupleBasis(wavepackets["basis"][0][1], wavepackets["basis"][0][1]),
        ),
        "data": wavepackets["eigenvalue"],
    }


def get_full_bloch_hamiltonian(
    wavefunctions: BlochWavefunctionListWithEigenvaluesList[_B0, _SB0, _SB1],
) -> SingleBasisDiagonalOperator[
    ExplicitStackedBasisWithLength[
        TupleBasisLike[_B0, _SB0],
        TupleBasisLike[*tuple[FundamentalTransformedPositionBasis[Any, Any], ...]],
    ]
]:
    """
    Get the hamiltonian in the full bloch basis.

    Returns
    -------
    SingleBasisDiagonalOperator[
    ExplicitStackedBasisWithLength[
        TupleBasisLike[_B0, _SB0],
        TupleBasisLike[*tuple[FundamentalTransformedPositionBasis[Any, Any], ...]],
    ]
    ]
    """
    basis = get_bloch_basis(wavefunctions)

    return {"basis": TupleBasis(basis, basis), "data": wavefunctions["eigenvalue"]}


def get_wannier_states(
    wavefunctions: BlochWavefunctionListList[_B2, _SB0, _SB1],
    operator: LocalizationOperator[_SB0, _B1, _B2],
) -> StateVectorList[
    TupleBasisLike[_B1, _SB0],
    TupleBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]],
]:
    localized = get_localized_wavepackets(wavefunctions, operator)

    fundamental_states = unfurl_wavepacket_list(localized)
    converted_fundamental = convert_state_vector_list_to_basis(
        fundamental_states,
        stacked_basis_as_fundamental_position_basis(fundamental_states["basis"][1]),
    )
    converted_stacked = converted_fundamental["data"].reshape(
        operator["basis"][1][0].n, *converted_fundamental["basis"][1].shape
    )
    data = np.zeros(
        (
            operator["basis"][0].n,  # Translation
            operator["basis"][1][0].n,  # Wannier idx
            *converted_fundamental["basis"][1].shape,  # Wavefunction
        ),
        dtype=np.complex128,
    )
    util = BasisUtil(operator["basis"][0])
    # for each translation of the wannier functions
    for idx in range(operator["basis"][0].n):
        offset = util.get_stacked_index(idx)
        shift = tuple(-n * o for (n, o) in zip(wavefunctions["basis"][1].shape, offset))

        tanslated = np.roll(
            converted_stacked, shift, axis=tuple(1 + x for x in range(util.ndim))
        )

        data[idx, :, :] = tanslated

    return {
        "basis": TupleBasis(
            TupleBasis(operator["basis"][1][0], operator["basis"][0]),
            converted_fundamental["basis"][1],
        ),
        "data": data.ravel(),
    }


def get_wannier_basis(
    wavefunctions: BlochWavefunctionListList[_B2, _SB0, _SB1],
    operator: LocalizationOperator[_SB0, _B1, _B2],
) -> ExplicitStackedBasisWithLength[
    TupleBasisLike[_B1, _SB0],
    TupleBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]],
]:
    """
    Get the basis, with the localised (wannier) states as eigenstates.

    Returns
    -------
    ExplicitStackedBasisWithLength[
        TupleBasisLike[_B0, _SB0],
        TupleBasisLike[*tuple[FundamentalTransformedPositionBasis[Any, Any], ...]],
    ]
    """
    return ExplicitStackedBasisWithLength(get_wannier_states(wavefunctions, operator))


def get_wannier_hamiltonian(
    wavefunctions: BlochWavefunctionListWithEigenvaluesList[_B2, _SB0, _SB1],
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
    wavefunctions: BlochWavefunctionListWithEigenvaluesList[_B2, _SB0, _SB1],
    operator: LocalizationOperator[_SB0, _B1, _B2],
) -> SingleBasisDiagonalOperator[
    ExplicitStackedBasisWithLength[
        TupleBasisLike[_B1, _SB0],
        TupleBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]],
    ]
]:
    """
    Get the hamiltonian in the full bloch basis.

    Returns
    -------
    SingleBasisDiagonalOperator[
    ExplicitStackedBasisWithLength[
        TupleBasisLike[_B0, _SB0],
        TupleBasisLike[*tuple[FundamentalTransformedPositionBasis[Any, Any], ...]],
    ]
    ]
    """
    basis = get_wannier_basis(wavefunctions, operator)
    hamiltonian = get_wannier_hamiltonian(wavefunctions, operator)
    hamiltonian_2d = np.einsum(
        "ik,ij->ijk",
        hamiltonian["data"].reshape(hamiltonian["basis"][0].n, -1),
        np.eye(hamiltonian["basis"][0].n),
    )

    hamiltonian_stacked = hamiltonian_2d.reshape(
        *hamiltonian["basis"][0].shape,
        *hamiltonian["basis"][0].shape,
        -1,
    )
    # TODO: is this correct...
    # I think because H is real symmetric, this ultimately doesn't matter
    data_stacked = np.fft.fftn(
        np.fft.ifftn(
            hamiltonian_stacked,
            axes=tuple(range(hamiltonian["basis"][0].ndim)),
            norm="ortho",
        ),
        axes=tuple(
            range(hamiltonian["basis"][0].ndim, 2 * hamiltonian["basis"][0].ndim)
        ),
        norm="ortho",
    )
    # Re order to match the correct basis
    data = np.einsum(
        "ijkl->ikjl",
        data_stacked.reshape(
            hamiltonian["basis"][0].n,
            hamiltonian["basis"][0].n,
            *hamiltonian["basis"][1].shape,
        ),
    )
    return {
        "basis": TupleBasis(basis, basis),
        "data": data.ravel(),
    }
    # TODO: We can probably just fourier transform the wannier hamiltonian.  # noqa: FIX002
    # ie hamiltonian = get_wannier_hamiltonian(wavefunctions, operator)
    # This will be faster and have less artifacts
    hamiltonian = get_full_bloch_hamiltonian(wavefunctions)
    return convert_operator_to_basis(hamiltonian, StackedBasis(basis, basis))
