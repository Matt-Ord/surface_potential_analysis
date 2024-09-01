from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from surface_potential_analysis.kernel.conversion import (
    convert_isotropic_kernel_to_basis,
)
from surface_potential_analysis.kernel.kernel import (
    IsotropicNoiseKernel,
    as_axis_kernel_from_isotropic,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_position_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import FundamentalPositionBasis
    from surface_potential_analysis.basis.stacked_basis import (
        StackedBasisWithVolumeLike,
    )

    _SBV0 = TypeVar("_SBV0", bound=StackedBasisWithVolumeLike[Any, Any, Any])


def get_fundamental_axis_kernels_from_isotropic(
    kernel: IsotropicNoiseKernel[_SBV0],
) -> tuple[IsotropicNoiseKernel[FundamentalPositionBasis[Any, Any]], ...]:
    converted = convert_isotropic_kernel_to_basis(
        kernel, stacked_basis_as_fundamental_position_basis(kernel["basis"])
    )

    return as_axis_kernel_from_isotropic(converted)
