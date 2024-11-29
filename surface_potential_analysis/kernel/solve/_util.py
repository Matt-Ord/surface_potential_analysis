from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from surface_potential_analysis.basis.legacy import convert_vector
from surface_potential_analysis.kernel.kernel import (
    IsotropicNoiseKernel,
    as_axis_kernel_from_isotropic,
)
from surface_potential_analysis.stacked_basis.conversion import (
    tuple_basis_as_fundamental,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.legacy import (
        BasisLike,
        FundamentalPositionBasis,
        StackedBasisWithVolumeLike,
    )

    _SBV0 = TypeVar("_SBV0", bound=StackedBasisWithVolumeLike)
    _B0 = TypeVar("_B0", bound=BasisLike)
    _B1 = TypeVar("_B1", bound=BasisLike)


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
) -> tuple[IsotropicNoiseKernel[FundamentalPositionBasis], ...]:
    converted = _convert_isotropic_kernel_to_basis(
        kernel, tuple_basis_as_fundamental(kernel["basis"])
    )

    return as_axis_kernel_from_isotropic(converted)
