"""The late-bound ``ops`` registry is complete once ``akme`` is imported."""

import akme  # noqa: F401  # importing the package performs the registrations
from akme.expressions._ops import Ops, ops


def test_every_ops_slot_is_registered():
    """Each constructor declared on :class:`Ops` has been set by its defining module."""
    missing = [name for name in Ops.__annotations__ if not hasattr(ops, name)]
    assert not missing, f"ops slots never registered: {missing}"
