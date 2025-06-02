# Copyright 2025 Manimit Haldar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import os
from abc import ABC, abstractmethod
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.circuit.library import RYGate, RXGate, RZGate
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np

# Abstract Base Class
class QuantumCircuitBase(ABC):
    """Abstract base class for quantum circuit implementations."""
    
    def __init__(self, num_qubits=2, shots=1024):
        """
        Initialize the quantum circuit base class.
        
        Args:
            num_qubits (int): Number of qubits for the circuit (default: 2).
            shots (int): Number of shots for execution (default: 1024).
        """
        self.num_qubits = num_qubits
        self.shots = shots
        self.circuits = []  # Changed to list to hold multiple circuits
        self.results_list = []  # Changed to list to hold multiple results
        self.runtimes = {}
        self.backend = self._setup_backend()

    @abstractmethod
    def _setup_backend(self):
        """Set up the backend for circuit execution."""
        pass

    def initialize_circuit(self, gates=None, circuit_type='bell', r_rotation_parameter_shift=False):
        """
        Initialize quantum circuits based on input parameters.
        
        Args:
            gates (list): List of tuples specifying gates, e.g., [('h', 0), ('cx', 0, 1)].
                         If None, uses circuit_type to determine default gates.
            circuit_type (str): Type of default circuit ('bell' or 'uniform').
            r_rotation_parameter_shift (bool): If True, create circuits with parameter shifts.
        
        Time Complexity: O(g * c) where g is the number of gates, c is the number of circuits (1 or 3).
        """
        start_time = time.time()
        
        if gates is None:
            if circuit_type == 'bell':
                gates = [('h', 0), ('cx', 0, 1)]
            elif circuit_type == 'uniform':
                gates = [('h', i) for i in range(self.num_qubits)]
            else:
                raise ValueError("Invalid circuit_type. Use 'bell' or 'uniform'.")
        
        if r_rotation_parameter_shift:
            # Create three circuits: original, +π/2 shift, -π/2 shift
            self.circuits = [
                self._build_circuit(gates),
                self._build_circuit(self._shift_gates(gates, np.pi / 2)),
                self._build_circuit(self._shift_gates(gates, -np.pi / 2))
            ]
        else:
            # Create a single circuit with original parameters
            self.circuits = [self._build_circuit(gates)]
        
        self.runtimes['initialize_circuit'] = time.time() - start_time
        return self.circuits

    def _build_circuit(self, gates):
        """Helper method to build a quantum circuit from a gates list."""
        circuit = QuantumCircuit(self.num_qubits, self.num_qubits, name="quantum_circuit")
        for gate in gates:
            gate_type, *params = gate
            if gate_type == 'h':
                circuit.h(params[0])
            elif gate_type == 'cx':
                circuit.cx(params[0], params[1])
            elif gate_type == 'ry':
                qubit, theta = params
                circuit.append(RYGate(theta), [qubit])
            elif gate_type == 'rz':
                qubit, theta = params
                circuit.append(RZGate(theta), [qubit])
            elif gate_type == 'rx':
                qubit, theta = params
                circuit.append(RXGate(theta), [qubit])
            elif gate_type == 'x':
                circuit.x(params[0])
            else:
                raise ValueError(f"Unsupported gate type: {gate_type}")
        circuit.measure(range(self.num_qubits), range(self.num_qubits))
        return circuit

    def _shift_gates(self, gates, shift):
        """Shift the theta parameters of rotation gates by a specified value."""
        shifted_gates = []
        for gate in gates:
            gate_type, *params = gate
            if gate_type in {'ry', 'rz', 'rx'}:
                qubit, theta = params
                shifted_gates.append((gate_type, qubit, theta + shift))
            else:
                shifted_gates.append(gate)
        return shifted_gates

    @abstractmethod
    def run_circuit(self):
        """Execute the circuits on the backend."""
        pass

    def visualize_circuit(self, filename='circuit.png'):
        """
        Visualize the quantum circuits and save them as images.
        
        Time Complexity: O(g * c) where g is gates, c is number of circuits.
        """
        if not self.circuits:
            raise ValueError("Circuits not initialized. Call initialize_circuit first.")
        
        start_time = time.time()
        dir_path = os.path.dirname(filename)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        for i, circuit in enumerate(self.circuits):
            circuit_filename = f"{filename.rsplit('.', 1)[0]}_{i}.png"
            circuit.draw(output='mpl', filename=circuit_filename)
        self.runtimes['visualize_circuit'] = time.time() - start_time
        print(f"Circuit visualizations saved with base name {filename}")

    @abstractmethod
    def visualize_results(self, filename):
        """Visualize the execution results as histograms."""
        pass

    def get_time_complexity(self):
        """
        Return the theoretical time complexity of key operations.
        
        Returns:
            dict: Time complexities for each component.
        """
        num_circuits = len(self.circuits) if self.circuits else 1
        return {
            'initialize_circuit': f"O(g * c) where g is gates ({len(self.circuits[0].data) if self.circuits else 'N/A'}), c is circuits ({num_circuits})",
            'run_circuit': f"O(s * n * c) where s is shots ({self.shots}), n is qubits ({self.num_qubits}), c is circuits ({num_circuits})",
            'visualize_circuit': f"O(g * c) where g is gates, c is circuits ({num_circuits})",
            'visualize_results': f"O(m * c) where m is unique outcomes, c is circuits ({num_circuits})"
        }
    
    def get_runtimes(self):
        """
        Return the measured runtimes of key operations.
        
        Returns:
            dict: Runtimes in seconds for each component.
        """
        return self.runtimes

# QASM Simulator Implementation
class QASMSimulatorCircuit(QuantumCircuitBase):
    """Quantum circuit implementation using Qiskit Aer simulator."""
    
    def _setup_backend(self):
        """Set up the AerSimulator backend."""
        return AerSimulator(method='automatic')

    def run_circuit(self):
        """
        Execute the circuits on the QASM simulator.
        
        Time Complexity: O(s * n * c) where s is shots, n is qubits, c is circuits.
        """
        if not self.circuits:
            raise ValueError("Circuits not initialized. Call initialize_circuit first.")
        
        start_time = time.time()
        self.results_list = [self.backend.run(circuit, shots=self.shots).result() for circuit in self.circuits]
        self.runtimes['run_circuit'] = time.time() - start_time
        return self.results_list
    
    def visualize_results(self, filename='qasm_results.png'):
        """
        Visualize the simulation results as histograms.
        
        Time Complexity: O(m * c) where m is unique outcomes, c is circuits.
        """
        if not self.results_list:
            raise ValueError("No results available. Call run_circuit first.")
        
        start_time = time.time()
        dir_path = os.path.dirname(filename)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        for i, result in enumerate(self.results_list):
            counts = result.get_counts()
            result_filename = f"{filename.rsplit('.', 1)[0]}_{i}.png"
            plot_histogram(counts).savefig(result_filename)
            plt.close()
        self.runtimes['visualize_results'] = time.time() - start_time
        print(f"Results histograms saved with base name {filename}")

# IBM Quantum Implementation
class IBMQuantumCircuit(QuantumCircuitBase):
    """Quantum circuit implementation using IBM Quantum backend."""
    
    def __init__(self, num_qubits=2, backend_name=None, shots=1024, api_token=None):
        """Initialize the IBM Quantum circuit class."""
        if api_token is None:
            raise ValueError("IBM Quantum API token is required.")
        self.api_token = api_token
        self.backend_name = backend_name
        super().__init__(num_qubits, shots)

    def _setup_backend(self):
        """Set up the IBM Quantum backend."""
        service = QiskitRuntimeService(channel="ibm_quantum", token=self.api_token)
        if self.backend_name:
            return service.backend(self.backend_name)
        return service.least_busy(
            min_num_qubits=self.num_qubits,
            simulator=False,
            operational=True
        )

    def run_circuit(self):
        """
        Execute the circuits on the IBM Quantum backend using Sampler.
        
        Time Complexity: O(s * n * c) where s is shots, n is qubits, c is circuits.
        """
        if not self.circuits:
            raise ValueError("Circuits not initialized. Call initialize_circuit first.")
        
        start_time = time.time()
        try:
            transpiled_circuits = transpile(self.circuits, backend=self.backend, optimization_level=1)
            sampler = Sampler(backend=self.backend)
            pubs = [(circuit,) for circuit in transpiled_circuits]
            job = sampler.run(pubs, shots=self.shots)
            print(f"Job submitted: {job.job_id()}")
            self.results_list = job.result()
            self.runtimes['run_circuit'] = time.time() - start_time
            return self.results_list
        except Exception as e:
            print(f"Error running job: {e}")
            self.runtimes['run_circuit'] = time.time() - start_time
            raise
    
    def visualize_results(self, filename='ibm_results.png'):
        """
        Visualize the execution results as histograms.
        
        Time Complexity: O(m * c) where m is unique outcomes, c is circuits.
        """
        if not self.results_list:
            raise ValueError("No results available. Call run_circuit first.")
        
        start_time = time.time()
        try:
            dir_path = os.path.dirname(filename)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            for i, result in enumerate(self.results_list):
                counts = result.data.c.get_counts()
                result_filename = f"{filename.rsplit('.', 1)[0]}_{i}.png"
                plot_histogram(counts).savefig(result_filename)
                plt.close()
            self.runtimes['visualize_results'] = time.time() - start_time
            print(f"Results histograms saved with base name {filename}")
        except Exception as e:
            print(f"Error visualizing results: {e}")
            self.runtimes['visualize_results'] = time.time() - start_time
            raise

# Final Class for User Interaction
class QuantumCircuitRunner:
    """Class to run quantum circuits on simulator or IBM Quantum backend."""
    
    def __init__(self, platform, num_qubits=2, shots=1024, circuit_type='bell', gates=None, 
                 r_rotation_parameter_shift=False, filename_prefix='quantum', api_token=None, backend_name=None):
        """Initialize the quantum circuit runner."""
        self.platform = platform.lower()
        self.num_qubits = num_qubits
        self.shots = shots
        self.circuit_type = circuit_type.lower()
        self.gates = gates
        self.r_rotation_parameter_shift = r_rotation_parameter_shift
        self.filename_prefix = filename_prefix
        self.api_token = api_token
        self.backend_name = backend_name
        self.circuit_instance = None
        
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate user inputs."""
        if self.platform not in ['simulator', 'ibm_quantum']:
            raise ValueError("Platform must be 'simulator' or 'ibm_quantum'.")
        if self.platform == 'ibm_quantum' and not self.api_token:
            raise ValueError("API token is required for IBM Quantum platform.")
        if not isinstance(self.num_qubits, int) or self.num_qubits < 1:
            raise ValueError("Number of qubits must be a positive integer.")
        if not isinstance(self.shots, int) or self.shots < 1:
            raise ValueError("Number of shots must be a positive integer.")
        if self.gates is None and self.circuit_type not in ['bell', 'uniform']:
            raise ValueError("Circuit type must be 'bell' or 'uniform' when gates is None.")
        if self.gates:
            valid_gates = {'h', 'cx', 'ry', 'rz', 'rx', 'x'}
            for gate in self.gates:
                if not isinstance(gate, tuple) or len(gate) < 2:
                    raise ValueError("Each gate must be a tuple, e.g., ('h', 0) or ('cx', 0, 1).")
                gate_type, *params = gate
                if gate_type not in valid_gates:
                    raise ValueError(f"Gate type '{gate_type}' not supported. Use {valid_gates}.")
                if gate_type in {'h', 'x'}:
                    if len(params) != 1:
                        raise ValueError(f"{gate_type} gate requires exactly one qubit.")
                    if any(not isinstance(q, int) or q < 0 or q >= self.num_qubits for q in params):
                        raise ValueError(f"Qubit indices must be between 0 and {self.num_qubits-1}.")
                elif gate_type == 'cx':
                    if len(params) != 2:
                        raise ValueError("CNOT gate requires exactly two qubits.")
                    if any(not isinstance(q, int) or q < 0 or q >= self.num_qubits for q in params):
                        raise ValueError(f"Qubit indices must be between 0 and {self.num_qubits-1}.")
                elif gate_type in {'ry', 'rz', 'rx'}:
                    if len(params) != 2:
                        raise ValueError(f"{gate_type} gate requires a qubit and a theta parameter.")
                    qubit, theta = params
                    if not isinstance(qubit, int) or qubit < 0 or qubit >= self.num_qubits:
                        raise ValueError(f"Qubit index must be between 0 and {self.num_qubits-1}.")
                    if not (isinstance(theta, float) or isinstance(theta, int)):
                        raise ValueError("Theta parameter must be a number.")

    def _create_circuit_instance(self):
        """Create the appropriate circuit instance based on platform."""
        if self.platform == 'simulator':
            return QASMSimulatorCircuit(num_qubits=self.num_qubits, shots=self.shots)
        else:  # ibm_quantum
            return IBMQuantumCircuit(
                num_qubits=self.num_qubits,
                backend_name=self.backend_name,
                shots=self.shots,
                api_token=self.api_token
            )

    def run(self):
        """
        Run the quantum circuits based on user configuration.
        
        Returns:
            dict: Results including circuits, results list, complexities, and runtimes.
        """
        try:
            self.circuit_instance = self._create_circuit_instance()
            
            # Initialize circuits
            self.circuit_instance.initialize_circuit(
                gates=self.gates,
                circuit_type=self.circuit_type,
                r_rotation_parameter_shift=self.r_rotation_parameter_shift
            )
            
            # Visualize circuits
            circuit_filename = f"results/{self.filename_prefix}_circuit.png"
            self.circuit_instance.visualize_circuit(circuit_filename)
            
            # Run circuits
            self.circuit_instance.run_circuit()
            
            # Visualize results
            results_filename = f"results/{self.filename_prefix}_results.png"
            self.circuit_instance.visualize_results(results_filename)
            
            # Collect results
            return {
                'results_list': self.circuit_instance.results_list,
                'circuits': self.circuit_instance.circuits,
                'time_complexities': self.circuit_instance.get_time_complexity(),
                'runtimes': self.circuit_instance.get_runtimes()
            }
        except Exception as e:
            raise RuntimeError(f"Failed to run circuit: {str(e)}")
