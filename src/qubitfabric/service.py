"""Core service surface for QubitFabric.

Exposes process health plus the serializable quantum-circuit IR: callers
submit JSON-compatible objects to build canonical circuits, simplify them
with deterministic equivalence transforms, and bind parameters to concrete
angles. Keep the public surface here backward compatible.
"""

from __future__ import annotations

from . import __version__
from .circuit import (
    CircuitValidationError,
    ParameterBindingError,
    bind_circuit,
    normalize_circuit,
    simplify_circuit,
)

__all__ = [
    "Service",
    "CircuitValidationError",
    "ParameterBindingError",
]


class Service:
    """QubitFabric service: health reporting and circuit IR operations."""

    name = "qubitfabric"
    version = __version__

    def health(self) -> dict[str, str]:
        return {"status": "ok", "service": self.name, "version": self.version}

    def normalize(self, spec: dict) -> dict:
        """Validate a JSON-compatible circuit spec and return canonical IR."""
        return normalize_circuit(spec)

    # Alias for callers that prefer an explicit name.
    normalize_circuit = normalize

    def simplify(self, circuit: dict) -> dict:
        """Simplify a circuit; returns the canonical circuit plus counts."""
        return simplify_circuit(circuit)

    def bind(self, circuit: dict, bindings: dict) -> dict:
        """Substitute parameter bindings, returning parameter-free IR."""
        return bind_circuit(circuit, bindings)
