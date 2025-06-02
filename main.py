from pyQCircuitEngine import QuantumCircuitRunner  # Assume this is the module name

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