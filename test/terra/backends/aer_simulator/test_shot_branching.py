# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
AerSimulator Integration Tests
"""

from ddt import ddt
from test.terra.reference import ref_measure
from test.terra.reference import ref_reset
from test.terra.reference import ref_initialize
from test.terra.reference import ref_kraus_noise
from test.terra.reference import ref_pauli_noise
from test.terra.reference import ref_readout_noise
from test.terra.reference import ref_reset_noise
from test.terra.reference import ref_conditionals

from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import ReadoutError, depolarizing_error
from qiskit.circuit.library import QuantumVolume
from qiskit.quantum_info.random import random_unitary
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods

from qiskit_aer import noise

import qiskit.quantum_info as qi
from qiskit.circuit.library import QFT
from qiskit.circuit import QuantumCircuit, Reset
from qiskit.circuit.library.standard_gates import IGate, HGate
from qiskit.quantum_info.states.densitymatrix import DensityMatrix

from qiskit.circuit import Parameter, Qubit, Clbit, QuantumRegister, ClassicalRegister
from qiskit.circuit.controlflow import *
from qiskit_aer.library.default_qubits import default_qubits
from qiskit_aer.library.control_flow_instructions import AerMark, AerJump

import numpy as np


SUPPORTED_METHODS = [
    "statevector",
    "density_matrix",
]
# tensor_network is tested in other test cases by setting shot_branching_enable by default

SUPPORTED_METHODS_INITIALIZE = [
    "statevector",
]


@ddt
class TestShotBranching(SimulatorTestCase):
    """AerSimulator measure tests."""

    OPTIONS = {"seed_simulator": 41411}

    @supported_methods(SUPPORTED_METHODS)
    def test_shot_branching_reset_moving_qubits(self, method, device):
        """Test AerSimulator reset with for circuits where qubits have moved"""
        backend = self.backend(method=method, device=device)
        # count output circuits
        shots = 1000
        circuits = ref_reset.reset_circuits_with_entangled_and_moving_qubits(final_measure=True)
        targets = ref_reset.reset_counts_with_entangled_and_moving_qubits(shots)
        result = backend.run(circuits, shots=shots, shot_branching_enable=True).result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)
