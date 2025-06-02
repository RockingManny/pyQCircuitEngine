# QuantumCircuitRunner: Documentation

**Date**: May 26, 2025  
**Environment**: Python 3.11, Qiskit 1.2.4  
**Author**: Manimit Haldar
**Description**: This is a documentation for the QuantumCircuitRunner class. This class is designed to run quantum circuits on a simulator or a real quantum device.

## 1. Introduction
`QuantumCircuitRunner` is a Python class designed to streamline the execution of quantum circuits on both local simulators and IBM Quantum hardware. It provides an intuitive, unified interface that abstracts the complexities of quantum backends, enabling users to focus on designing and analyzing quantum circuits. This documentation outlines the theoretical foundations, operational principles, prerequisites, installation steps, and practical use cases for `QuantumCircuitRunner`.

## 2. Theoretical Background

### 2.1 Quantum Computing Basics
Quantum computing harnesses quantum mechanics to perform computations in ways that classical computers cannot. Core concepts include:

- **Qubits**: Unlike classical bits, qubits can exist in a superposition of states \( |0\rangle \) and \( |1\rangle \), represented as \( |\psi\rangle = \alpha|0\rangle + \beta|1\rangle \), where \( \alpha \) and \( \beta \) are complex amplitudes.
- **Quantum Gates**: Operations that transform qubit states, such as the Hadamard gate (H) for superposition and the Controlled-NOT gate (CNOT or CX) for entanglement.
- **Measurement**: Collapses a qubit’s state into a classical outcome (0 or 1), with probabilities determined by \( |\alpha|^2 \) and \( |\beta|^2 \).
- **Entanglement**: A correlation between qubits where the state of one qubit depends on another, e.g., the Bell state \( |\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}} \).
- **Superposition**: Allows qubits to represent multiple states simultaneously, enabling parallelism in quantum algorithms.

### 2.2 Quantum Circuits
A quantum circuit is a sequence of quantum gates applied to a set of qubits, followed by measurements to obtain classical results. `QuantumCircuitRunner` supports:
- **Bell State Circuit**: Uses gates [H(0), CX(0,1)] to create an entangled state, yielding outcomes `00` and `11` with equal probability.
- **Uniform Superposition Circuit**: Applies Hadamard gates [H(i) for i in range(num_qubits)] to all qubits, producing a uniform distribution over all possible states.
- **Custom Circuits**: Users can define arbitrary circuits by providing a list of gates (e.g., `[('h', 0), ('cx', 0, 1)]`).

### 2.3 Quantum Backends
`QuantumCircuitRunner` operates on two platforms:
- **Local Simulator**: Employs Qiskit’s `AerSimulator` for fast, noise-free simulations on a local machine.
- **IBM Quantum**: Runs circuits on real quantum hardware via IBM Quantum’s cloud service, requiring circuit transpilation to match hardware constraints (e.g., native gate sets and qubit connectivity).

## 3. Principle and Workflow
The `QuantumCircuitRunner` class simplifies quantum circuit execution by:
1. **Initializing Circuits**: Constructs circuits based on user-specified parameters (e.g., number of qubits, circuit type, or custom gates).
2. **Transpiling (IBM Quantum)**: Adapts circuits for hardware compatibility when targeting IBM Quantum backends.
3. **Executing Circuits**: Runs the circuit on the chosen platform with a specified number of shots (repetitions).
4. **Visualizing Results**: Generates circuit diagrams and histograms of measurement outcomes, saved as PNG files.
5. **Analyzing Performance**: Reports time complexities and actual runtimes for key operations.

This abstraction ensures a seamless experience across platforms, shielding users from backend-specific details.

## 4. Prerequisites
To use `QuantumCircuitRunner`, ensure the following:

- **Hardware**: A computer with internet access (required for IBM Quantum).
- **Software**:
  - Python 3.11
  - Qiskit packages: `qiskit==1.2.4`, `qiskit-aer==0.15.1`, `qiskit-ibm-runtime==0.29.0`
  - Additional libraries: `numpy==1.26.4`, `matplotlib==3.9.2`, `pylatexenc>=2.10`
- **IBM Quantum Account**: Required for IBM Quantum execution. Obtain an API token from [https://quantum.ibm.com](https://quantum.ibm.com).
- **Working Directory**: A directory with write permissions (e.g., `D:\Quantum\Projects`) for saving output files.

## 5. Installation

### 5.1 Set Up Python Environment
1. **Install Python 3.11**:
   - Download from [https://www.python.org/downloads/release/python-31110/](https://www.python.org/downloads/release/python-31110/).
   - Verify installation:
     ```bash
     python3.11 --version
     ```
2. **Create a Virtual Environment**:
   ```bash
   cd <your_directory>
   python3.11 -m venv .venv_311
   .venv_311\Scripts\activate  # Windows
   # or source .venv_311/bin/activate  # Linux/Mac
   ```

### 5.2 Install Dependencies
1. **Create a `requirements.txt` File**:
   Save the following in [requirements.py](requirements.txt):
   ```
    annotated-types==0.7.0
    asttokens==3.0.0
    certifi==2025.4.26
    cffi==1.17.1
    charset-normalizer==3.4.2
    colorama==0.4.6
    comm==0.2.2
    contourpy==1.3.2
    cryptography==45.0.3
    cycler==0.12.1
    debugpy==1.8.14
    decorator==5.2.1
    dill==0.4.0
    executing==2.2.0
    fonttools==4.58.0
    ibm-cloud-sdk-core==3.23.0
    ibm-platform-services==0.66.0
    idna==3.10
    ipykernel==6.29.5
    ipython==9.2.0
    ipython_pygments_lexers==1.1.1
    jedi==0.19.2
    jupyter_client==8.6.3
    jupyter_core==5.7.2
    kiwisolver==1.4.8
    matplotlib==3.9.2
    matplotlib-inline==0.1.7
    mpmath==1.3.0
    nest-asyncio==1.6.0
    numpy==1.26.4
    packaging==25.0
    parso==0.8.4
    pbr==6.1.1
    pillow==11.2.1
    platformdirs==4.3.8
    prompt_toolkit==3.0.51
    psutil==7.0.0
    pure_eval==0.2.3
    pycparser==2.22
    pydantic==2.11.5
    pydantic_core==2.33.2
    Pygments==2.19.1
    PyJWT==2.10.1
    pylatexenc==2.10
    pyparsing==3.2.3
    pyspnego==0.11.2
    python-dateutil==2.9.0.post0
    pywin32==310
    pyzmq==26.4.0
    qiskit==1.2.4
    qiskit-aer==0.15.1
    qiskit-ibm-provider==0.11.0
    qiskit-ibm-runtime==0.29.0
    requests==2.32.3
    requests_ntlm==1.3.0
    rustworkx==0.16.0
    scipy==1.15.3
    six==1.17.0
    sspilib==0.3.1
    stack-data==0.6.3
    stevedore==5.4.1
    symengine==0.13.0
    sympy==1.14.0
    tornado==6.5.1
    traitlets==5.14.3
    typing-inspection==0.4.1
    typing_extensions==4.13.2
    urllib3==2.4.0
    wcwidth==0.2.13
    websocket-client==1.8.0
    websockets==15.0.1
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Verify Installation**:
   ```bash
   pip show qiskit qiskit-aer qiskit-ibm-runtime numpy matplotlib pylatexenc
   ```

### 5.3 Configure IBM Quantum API Token
- Obtain your API token from [https://quantum.ibm.com](https://quantum.ibm.com).
- Set it as an environment variable:
  ```bash
  set IBMQ_TOKEN=your_api_token  # Windows
  # or export IBMQ_TOKEN=your_api_token  # Linux/Mac
  ```

## 6. Use Cases
`QuantumCircuitRunner` supports a range of applications, from educational demonstrations to custom quantum experiments. Below are three examples.

### 6.1 Simulating a Bell State Locally
- **Objective**: Run a Bell state circuit on a local simulator to observe entanglement.
- **Code**:
  ```python
  from pyQC import QuantumCircuitRunner  # Assume this is the module name

  runner = QuantumCircuitRunner(
      platform='simulator',
      num_qubits=2,
      shots=1024,
      circuit_type='bell',
      filename_prefix='sim_bell'
  )
  results = runner.run()
  print("Time Complexities:", results['time_complexities'])
  print("Runtimes (seconds):", results['runtimes'])
  ```
- **Expected Output**:
  - Circuit diagram saved as `results/sim_bell_circuit.png`.
  - Histogram saved as `results/sim_bell_results.png`, showing ~50% `00` and ~50% `11`.

### 6.2 Running a Uniform Superposition on IBM Quantum
- **Objective**: Execute a uniform superposition circuit on IBM Quantum hardware.
- **Code**:
  ```python
  import os
  from pyQC import QuantumCircuitRunner

  runner = QuantumCircuitRunner(
      platform='ibm_quantum',
      num_qubits=2,
      shots=1024,
      circuit_type='uniform',
      filename_prefix='ibm_uniform',
      api_token=os.getenv("IBMQ_TOKEN")
  )
  results = runner.run()
  print("Time Complexities:", results['time_complexities'])
  print("Runtimes (seconds):", results['runtimes'])
  ```
- **Expected Output**:
  - Circuit diagram saved as `results/ibm_uniform_circuit.png`.
  - Histogram saved as `results/ibm_uniform_results.png`, showing ~25% for each state (`00`, `01`, `10`, `11`), with some noise due to hardware imperfections.

### 6.3 Executing a Custom Circuit
- **Objective**: Simulate a custom circuit with specific gates.
- **Code**:
  ```python
  from pyQC import QuantumCircuitRunner

  custom_gates = [('h', 0), ('cx', 0, 1), ('h', 1)]
  runner = QuantumCircuitRunner(
      platform='simulator',
      num_qubits=2,
      shots=1024,
      gates=custom_gates,
      filename_prefix='sim_custom'
  )
  results = runner.run()
  print("Time Complexities:", results['time_complexities'])
  print("Runtimes (seconds):", results['runtimes'])
  ```
- **Expected Output**:
  - Circuit diagram saved as `results/sim_custom_circuit.png`.
  - Histogram saved as `results/sim_custom_results.png`, with outcome distribution dependent on the gates applied.

### 6.4 Simulating Custom Circuits with Rotation Gates
- **Objective**: Simulate a custom circuit including rotation gates with specified angles.
- **Code**:
  ```python
  from pyQCircuitEngine.pyQCircuitEngine import QuantumCircuitRunner

  custom_gates = [
      ('h', 0),
      ('ry', 0, 1.5708),  # ry(pi/2) on qubit 0
  ]

  runner = QuantumCircuitRunner(
      platform='simulator',
      num_qubits=1,
      shots=1024,
      gates=custom_gates,
      filename_prefix='sim_rotation_test'
  )
  results = runner.run()
  print("Time Complexities:", results['time_complexities'])
  print("Runtimes (seconds):", results['runtimes'])
  ```
- **Expected Output**:
  - Circuit diagram saved as `results/sim_rotation_test_circuit.png`.
  - Histogram saved as `results/sim_rotation_test_results.png`, showing the effect of rotation gates on the qubit state.

## 7. Conclusion
`QuantumCircuitRunner` offers a versatile and user-friendly tool for running quantum circuits, bridging the gap between theoretical quantum computing and practical implementation. By supporting both simulators and real quantum hardware, it empowers users to explore quantum phenomena and develop quantum algorithms with ease.

## 8. License
pyQCircuitEngine is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
Copyright (c) 2025 Manimit Haldar.

## 9. Contributing
We welcome contributions to pyQCircuitEngine! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for instructions on submitting pull requests.