"""Late-bound hooks the expression layer needs from the layers built on top of it.

:class:`~acme.expressions.expression.Expression` overloads ``+``, ``**``, ``<=``, ``>>``, ...
whose results are affine expressions, atoms and constraints -- all defined in modules that
import ``expression``. Instead of importing them back (a cycle), ``expression`` calls
through :data:`ops`, and each defining module registers its implementation at import time:

- :mod:`acme.expressions.affine` -- ``affine``, ``constant``
- :mod:`acme.atoms` (the package ``__init__``) -- ``sum``, ``power``
- :mod:`acme.constraints` -- ``equality``, ``inequality``, ``psd``

``import acme`` imports all three, so the hooks are set before any user code runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from acme.constraints import Constraint
    from acme.expressions.affine import AffineExpr
    from acme.expressions.expression import Expression


class Ops:
    """Registry of the constructors :class:`~acme.expressions.expression.Expression` delegates to."""

    affine: type[AffineExpr]
    constant: Callable[[Any], AffineExpr]
    sum: Callable[..., Expression]
    power: Callable[[Any, float], Expression]
    equality: Callable[[Any, Any], Constraint]
    inequality: Callable[[Any, Any], Constraint]
    psd: Callable[[Any], Constraint]


#: The process-wide registry.
ops = Ops()
