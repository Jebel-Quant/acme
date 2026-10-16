"""Common solver result type."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RawResult:
    """Raw solver output in standard-form coordinates.

    ``z`` holds the cone multipliers with ``P x + q + A^T z = 0``.
    """

    status: str
    x: np.ndarray | None
    z: np.ndarray | None
    iterations: int
    solve_time: float
    solver_name: str
