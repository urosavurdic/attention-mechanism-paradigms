"""Quantum arithmetic circuit model for gate-based quantum computing."""

import math


class QuantumArithmeticCircuit:
    """Models gate-based quantum computer for matrix arithmetic."""

    def __init__(self, gate_time_s, measurement_time_s, toffoli_t_count,
                 qubit_count, classical_clock_hz, precision_bits=16):
        self.gate_time = gate_time_s
        self.measurement_time = measurement_time_s
        self.toffoli_t_count = toffoli_t_count
        self.qubit_count = qubit_count
        self.classical_clock = classical_clock_hz
        self.precision_bits = precision_bits

    def gates_per_multiply(self):
        toffoli_count = self.precision_bits ** 2
        return toffoli_count * self.toffoli_t_count

    def gates_per_addition(self):
        return self.precision_bits * self.toffoli_t_count

    def parallel_multiplies(self):
        qubits_per_mul = 2 * self.precision_bits
        return max(1, self.qubit_count // qubits_per_mul)

    def matmul_time(self, M, K, N):
        total_multiplies = M * K * N
        total_additions = M * N * max(K - 1, 0)

        par = self.parallel_multiplies()
        sequential_rounds = math.ceil(total_multiplies / par)
        depth_per_round = self.gates_per_multiply()
        total_depth = sequential_rounds * depth_per_round

        add_depth = math.ceil(total_additions / par) * self.gates_per_addition()
        total_depth += add_depth

        measurement = self.measurement_time * M * N
        gate_time = total_depth * self.gate_time

        return gate_time + measurement, {
            "total_multiplies": total_multiplies,
            "total_additions": total_additions,
            "parallel_capacity": par,
            "sequential_rounds": sequential_rounds,
            "total_gate_depth": total_depth,
            "gates_per_multiply": self.gates_per_multiply(),
        }

    def softmax_time(self, seq_len):
        measurement = seq_len * seq_len * self.measurement_time
        classical_softmax = 5 * seq_len * seq_len / self.classical_clock
        state_prep_gates = seq_len * seq_len * self.precision_bits * self.toffoli_t_count
        state_prep_time = state_prep_gates * self.gate_time

        return measurement + classical_softmax + state_prep_time, {
            "measurement_time_s": measurement,
            "classical_softmax_s": classical_softmax,
            "state_prep_time_s": state_prep_time,
            "note": "hybrid quantum-classical: measure, classical softmax, re-encode",
        }
