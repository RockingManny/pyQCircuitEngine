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
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Abstract Base Class (unchanged from previous implementation)
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
        self.circuit = None
        self.results = None
        self.runtimes = {}
        self.backend = self._setup_backend()

    @abstractmethod
    def _setup_backend(self):
        """Set up the backend for circuit execution."""
        pass

    def initialize_circuit(self, gates=None, circuit_type='bell'):
        """
        Initialize a quantum circuit based on input parameters.
        
        Args:
            gates (list): List of tuples specifying gates, e.g., [('h', 0), ('cx', 0, 1)].
                         If None, uses circuit_type to determine default gates.
            circuit_type (str): Type of default circuit ('bell' for Bell state, 'uniform' for all states).
        
        Time Complexity: O(g) where g is the number of gates.
        """
        start_time = time.time()
        self.circuit = QuantumCircuit(self.num_qubits, self.num_qubits, name="quantum_circuit")
        
        if gates is None:
            if circuit_type == 'bell':
                gates = [('h', 0), ('cx', 0, 1)]
            elif circuit_type == 'uniform':
                gates = [('h', i) for i in range(self.num_qubits)]
            else:
                raise ValueError("Invalid circuit_type. Use 'bell' or 'uniform'.")
        
        for gate in gates:
            gate_type, *qubits = gate
            if gate_type == 'h':
                self.circuit.h(qubits[0])
            elif gate_type == 'cx':
                self.circuit.cx(qubits[0], qubits[1])
        
        self.circuit.measure(range(self.num_qubits), range(self.num_qubits))
        
        self.runtimes['initialize_circuit'] = time.time() - start_time
        return self.circuit

    @abstractmethod
    def run_circuit(self):
        """Execute the circuit on the backend."""
        pass

    def visualize_circuit(self, filename='circuit.png'):
        """
        Visualize the quantum circuit and save it as an image.
        
        Time Complexity: O(g) where g is the number of gates (rendering complexity).
        """
        if self.circuit is None:
            raise ValueError("Circuit not initialized. Call initialize_circuit first.")
        
        start_time = time.time()
        dir_path = os.path.dirname(filename)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        self.circuit.draw(output='mpl', filename=filename)
        self.runtimes['visualize_circuit'] = time.time() - start_time
        print(f"Circuit visualization saved as {filename}")

    @abstractmethod
    def visualize_results(self, filename):
        """Visualize the execution results as a histogram."""
        pass

    def get_time_complexity(self):
        """
        Return the theoretical time complexity of key operations.
        
        Returns:
            dict: Time complexities for each component.
        """
        return {
            'initialize_circuit': f"O(g) where g is the number of gates ({len(self.circuit.data) if self.circuit else 'N/A'})",
            'run_circuit': f"O(s * n) where s is shots ({self.shots}) and n is qubits ({self.num_qubits})",
            'visualize_circuit': f"O(g) where g is the number of gates ({len(self.circuit.data) if self.circuit else 'N/A'})",
            'visualize_results': f"O(m) where m is the number of unique outcomes"
        }
    
    def get_runtimes(self):
        """
        Return the measured runtimes of key operations.
        
        Returns:
            dict: Runtimes in seconds for each component.
        """
        return self.runtimes

# QASM Simulator Implementation (unchanged)
class QASMSimulatorCircuit(QuantumCircuitBase):
    """Quantum circuit implementation using Qiskit Aer simulator."""
    
    def _setup_backend(self):
        """
        Set up the AerSimulator backend.
        
        Returns:
            AerSimulator: Configured simulator backend.
        """
        return AerSimulator(method='automatic')

    def run_circuit(self):
        """
        Execute the circuit on the QASM simulator.
        
        Time Complexity: O(s * n) where s is shots and n is number of qubits (simplified).
        """
        if self.circuit is None:
            raise ValueError("Circuit not initialized. Call initialize_circuit first.")
        
        start_time = time.time()
        self.results = self.backend.run(self.circuit, shots=self.shots).result()
        self.runtimes['run_circuit'] = time.time() - start_time
        return self.results
    
    def visualize_results(self, filename='qasm_results.png'):
        """
        Visualize the simulation results as a histogram.
        
        Time Complexity: O(m) where m is the number of unique measurement outcomes.
        """
        if self.results is None:
            raise ValueError("No results available. Call run_circuit first.")
        
        start_time = time.time()
        counts = self.results.get_counts()
        dir_path = os.path.dirname(filename)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        plot_histogram(counts).savefig(filename)
        plt.close()
        self.runtimes['visualize_results'] = time.time() - start_time
        print(f"Results histogram saved as {filename}")

# IBM Quantum Implementation (unchanged)
class IBMQuantumCircuit(QuantumCircuitBase):
    """Quantum circuit implementation using IBM Quantum backend."""
    
    def __init__(self, num_qubits=2, backend_name=None, shots=1024, api_token=None):
        """
        Initialize the IBM Quantum circuit class.
        
        Args:
            num_qubits (int): Number of qubits for the circuit (default: 2).
            backend_name (str): Name of the IBM Quantum backend (default: None, selects least busy).
            shots (int): Number of shots for execution (default: 1024).
            api_token (str): IBM Quantum API token (required).
        """
        if api_token is None:
            raise ValueError("IBM Quantum API token is required.")
        self.api_token = api_token
        self.backend_name = backend_name
        super().__init__(num_qubits, shots)

    def _setup_backend(self):
        """
        Set up the IBM Quantum backend.
        
        Returns:
            Backend: Configured IBM Quantum backend.
        """
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
        Execute the circuit on the IBM Quantum backend using the Sampler primitive.
        
        Time Complexity: O(s * n) where s is shots and n is qubits (simplified, excludes queue time).
        """
        if self.circuit is None:
            raise ValueError("Circuit not initialized. Call initialize_circuit first.")
        
        start_time = time.time()
        try:
            transpiled_circuit = transpile(
                self.circuit,
                backend=self.backend,
                optimization_level=1
            )
            sampler = Sampler(backend=self.backend)
            job = sampler.run([(transpiled_circuit,)], shots=self.shots)
            print(f"Job submitted: {job.job_id()}")
            self.results = job.result()
            self.runtimes['run_circuit'] = time.time() - start_time
            return self.results
        except Exception as e:
            print(f"Error running job: {e}")
            self.runtimes['run_circuit'] = time.time() - start_time
            raise
    
    def visualize_results(self, filename='ibm_results.png'):
        """
        Visualize the execution results as a histogram.
        
        Time Complexity: O(m) where m is the number of unique measurement outcomes.
        """
        if self.results is None:
            raise ValueError("No results available. Call run_circuit first.")
        
        start_time = time.time()
        try:
            counts = self.results[0].data.c.get_counts()
            dir_path = os.path.dirname(filename)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            plot_histogram(counts).savefig(filename)
            plt.close()
            self.runtimes['visualize_results'] = time.time() - start_time
            print(f"Results histogram saved as {filename}")
        except Exception as e:
            print(f"Error visualizing results: {e}")
            self.runtimes['visualize_results'] = time.time() - start_time
            raise

# Final Class for User Interaction
class QuantumCircuitRunner:
    """Class to run quantum circuits on simulator or IBM Quantum backend with user customization."""
    
    def __init__(self, platform, num_qubits=2, shots=1024, circuit_type='bell', gates=None, 
                 filename_prefix='quantum', api_token=None, backend_name=None):
        """
        Initialize the quantum circuit runner.
        
        Args:
            platform (str): Execution platform ('simulator' or 'ibm_quantum').
            num_qubits (int): Number of qubits (default: 2).
            shots (int): Number of shots (default: 1024).
            circuit_type (str): Circuit type ('bell', 'uniform') if gates is None (default: 'bell').
            gates (list): Custom gates as tuples, e.g., [('h', 0), ('cx', 0, 1)] (default: None).
            filename_prefix (str): Prefix for output files (default: 'quantum').
            api_token (str): IBM Quantum API token (required for ibm_quantum).
            backend_name (str): IBM Quantum backend name (default: None, selects least busy).
        
        Raises:
            ValueError: If invalid inputs are provided.
        """
        self.platform = platform.lower()
        self.num_qubits = num_qubits
        self.shots = shots
        self.circuit_type = circuit_type.lower()
        self.gates = gates
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
            valid_gates = {'h', 'cx'}
            for gate in self.gates:
                if not isinstance(gate, tuple) or len(gate) < 2:
                    raise ValueError("Each gate must be a tuple, e.g., ('h', 0) or ('cx', 0, 1).")
                gate_type, *qubits = gate
                if gate_type not in valid_gates:
                    raise ValueError(f"Gate type '{gate_type}' not supported. Use {valid_gates}.")
                if any(not isinstance(q, int) or q < 0 or q >= self.num_qubits for q in qubits):
                    raise ValueError(f"Qubit indices must be integers between 0 and {self.num_qubits-1}.")
                if gate_type == 'h' and len(qubits) != 1:
                    raise ValueError("Hadamard gate requires exactly one qubit.")
                if gate_type == 'cx' and len(qubits) != 2:
                    raise ValueError("CNOT gate requires exactly two qubits.")

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
        Run the quantum circuit based on user configuration.
        
        Returns:
            dict: Results including circuit instance, time complexities, and runtimes.
        
        Raises:
            RuntimeError: If circuit execution fails.
        """
        try:
            self.circuit_instance = self._create_circuit_instance()
            
            # Initialize circuit
            self.circuit_instance.initialize_circuit(gates=self.gates, circuit_type=self.circuit_type)
            
            # Visualize circuit
            circuit_filename = f"results/{self.filename_prefix}_circuit.png"
            self.circuit_instance.visualize_circuit(circuit_filename)
            
            # Run circuit
            self.circuit_instance.run_circuit()
            
            # Visualize results
            results_filename = f"results/{self.filename_prefix}_results.png"
            self.circuit_instance.visualize_results(results_filename)
            
            # Collect results
            return {
                'circuit_instance': self.circuit_instance,
                'time_complexities': self.circuit_instance.get_time_complexity(),
                'runtimes': self.circuit_instance.get_runtimes()
            }
        
        except Exception as e:
            raise RuntimeError(f"Failed to run circuit: {str(e)}")

