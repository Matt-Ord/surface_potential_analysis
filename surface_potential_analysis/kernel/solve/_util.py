from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from surface_potential_analysis.basis.basis_like import convert_vector
from surface_potential_analysis.kernel.kernel import (
    IsotropicNoiseKernel,
    as_axis_kernel_from_isotropic,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_position_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import FundamentalPositionBasis
    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.basis.stacked_basis import (
        StackedBasisWithVolumeLike,
    )

    _SBV0 = TypeVar("_SBV0", bound=StackedBasisWithVolumeLike[Any, Any, Any])
    _B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
    _B1 = TypeVar("_B1", bound=BasisLike[Any, Any])


def _convert_isotropic_kernel_to_basis(
    kernel: IsotropicNoiseKernel[_B0],
    basis: _B1,
) -> IsotropicNoiseKernel[_B1]:
    """Convert the kernel to the given basis.

    Parameters
    ----------
    kernel : NoiseKernel[_B0, _B1, _B0, _B1]
    basis : TupleBasisLike[_B2Inv, _B3Inv]

    Returns
    -------
    NoiseKernel[_B0, _B1, _B0, _B1]
    """
    data = convert_vector(kernel["data"], kernel["basis"], basis)
    return {"data": data, "basis": basis}


def get_fundamental_axis_kernels_from_isotropic(
    kernel: IsotropicNoiseKernel[_SBV0],
) -> tuple[IsotropicNoiseKernel[FundamentalPositionBasis[Any, Any]], ...]:
    converted = _convert_isotropic_kernel_to_basis(
        kernel, stacked_basis_as_fundamental_position_basis(kernel["basis"])
    )

    return as_axis_kernel_from_isotropic(converted)
