"""
Represents a grid of eigenstates, uniformly sampling the 1st Brillouin zone.

These states can be used to produce a more localized Wannier basis states
"""
from __future__ import annotations

from .wavepacket import BlochWavefunctionList

__all__ = ["BlochWavefunctionList"]
