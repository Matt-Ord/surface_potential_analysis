from functools import cache
from typing import TypeVar

import numpy as np
from numpy.typing import ArrayLike


@cache
def hermite_coefficient(n: int, m: int) -> int:
    """Calculate the n,m Physics hermite polynomial coefficient"""
    match (n, m):
        case (0, 0):
            return 1
        case (1, 0):
            return 0
        case (1, 1):
            return 2
        case _:
            if n >= m and m >= 0:
                lhs = 2 * hermite_coefficient(n - 1, m - 1)
                rhs = (m + 1) * hermite_coefficient(n - 1, m + 1)
                return lhs - rhs
            return 0


T = TypeVar("T", bound=ArrayLike)
# scipy.special.hermite


def hermite_value(n: int, x: T) -> T:
    """Calculate the value of the nth physicist hermite polynomial for a given value of x"""
    m_values = np.arange(0, n + 1)
    return np.sum(
        [hermite_coefficient(n, m) * np.power(x, m) for m in m_values], axis=0
    )
