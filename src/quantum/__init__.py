"""Classical, synthetic signal-processing modules with QKD-motivated names.

Nothing in this package simulates or interacts with quantum states, photons,
qubits, or optical hardware. The package name follows the project
specification; see each module docstring for its actual scope.
"""

from .topological_ao import TopologicalAdaptiveOpticsFilter

__all__ = ["TopologicalAdaptiveOpticsFilter"]
