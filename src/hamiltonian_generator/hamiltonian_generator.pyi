from typing import List, Optional, Tuple

def get_hamiltonian(
    ft_potential: List[List[List[float]]],
    resolution: Tuple[float, float, float],
    dz: float,
    mass: float,
    sho_omega: float,
    z_offset: float,
) -> List[List[float]]:
    """Adds two numbers and returns the answer as a string"""

def get_sho_wavefunction(
    z_points: List[float], sho_omega: float, mass: float, n: int
) -> List[float]:
    """Adds two numbers and returns the answer as a string"""

def get_hermite_val(x: float, n: int) -> float:
    """Adds two numbers and returns the answer as a string"""

def get_eigenstate_wavefunction(
    resolution: Tuple[float, float, float],
    delta_x: float,
    delta_y: float,
    mass: float,
    sho_omega: float,
    kx: float,
    ky: float,
    vector: List[complex],
    points: List[Tuple[float, float, float]],
) -> List[complex]:
    """Adds two numbers and returns the answer as a string"""