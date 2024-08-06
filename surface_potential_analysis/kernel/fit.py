from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import factorial


def get_cos_series_expansion(
    true_noise_coeff: np.ndarray[Any, np.dtype[np.float64]],
    d_k: float,
    *,
    n_polynomials: int = 1,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    i_ = np.arange(0, n_polynomials + 1)
    m_ = np.arange(0, n_polynomials + 1).reshape(-1, 1)
    coefficients_matrix = (((-1) ** m_) / (factorial(2 * m_))) * (
        (i_ * d_k) ** (2 * m_)
    )
    cos_series_coefficients = np.linalg.solve(coefficients_matrix, true_noise_coeff)
    return cos_series_coefficients.T
