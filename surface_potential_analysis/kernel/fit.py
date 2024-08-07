from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import factorial


def get_cos_series_coefficients(
    polynomial_coefficients: np.ndarray[Any, np.dtype[np.float64]],
    d_k: float,
    *,
    n_coses: int = 1,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    i_ = np.arange(0, n_coses + 1)
    m_ = np.arange(0, n_coses + 1).reshape(-1, 1)
    coefficients_matrix = (((-1) ** m_) / (factorial(2 * m_))) * (
        (i_ * d_k) ** (2 * m_)
    )
    cos_series_coefficients = np.linalg.solve(
        coefficients_matrix, polynomial_coefficients
    )
    return cos_series_coefficients.T


def get_trig_series_coefficients(
    polynomial_coefficients: np.ndarray[Any, np.dtype[np.float64]],
    d_k: float,
    *,
    n_coses: int = 1,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    cos_series_coefficients = get_cos_series_coefficients(
        polynomial_coefficients, d_k, n_coses=n_coses
    )
    sin_series_coefficients = cos_series_coefficients[1::][::-1]
    return np.concatenate([cos_series_coefficients, sin_series_coefficients])


def get_trig_series_data(
    k: float,
    nx_points: np.ndarray[tuple[tuple[int]], np.dtype[np.int_]],
    *,
    n: int = 1,
) -> np.ndarray[Any, np.dtype[np.complex128]]:
    sines = np.sin(
        np.arange(1, n + 1)[:, np.newaxis] * k * nx_points[np.newaxis, :]
    ).astype(np.complex128)
    sines = sines[::-1]
    coses = np.cos(
        np.arange(0, n + 1)[:, np.newaxis] * k * nx_points[np.newaxis, :]
    ).astype(np.complex128)
    return np.append(coses, sines)
