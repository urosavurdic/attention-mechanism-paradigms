"""DNA strand displacement cascade model for biological computing."""

import math


class DNAComputer:
    """Models DNA strand displacement for arithmetic operations."""

    def __init__(self, displacement_time_s, cascade_depth_per_multiply,
                 max_parallelism, enzymatic_exp_steps):
        self.displacement_time = displacement_time_s
        self.cascade_depth = cascade_depth_per_multiply
        self.max_parallelism = max_parallelism
        self.enzymatic_exp_steps = enzymatic_exp_steps

    def multiply_time(self):
        return self.cascade_depth * self.displacement_time

    def addition_time(self):
        return self.displacement_time

    def matmul_time(self, M, K, N):
        per_element = self.multiply_time() + K * self.addition_time()
        batch_rounds = math.ceil(M * N / self.max_parallelism)
        total_time = batch_rounds * per_element

        return total_time, {
            "per_element_time_s": per_element,
            "multiply_time_s": self.multiply_time(),
            "accumulation_steps": K,
            "batch_rounds": batch_rounds,
            "parallel_elements": min(M * N, self.max_parallelism),
            "estimated_strand_species": M * K * N * 10,
        }

    def softmax_time(self, seq_len):
        exp_time = self.enzymatic_exp_steps * self.displacement_time
        max_time = math.ceil(math.log2(max(seq_len, 2))) * self.displacement_time
        div_time = self.cascade_depth * self.displacement_time
        per_row = exp_time + max_time + div_time
        total_time = seq_len * per_row

        return total_time, {
            "exp_time_per_row_s": exp_time,
            "max_time_per_row_s": max_time,
            "div_time_per_row_s": div_time,
            "enzymatic_steps": self.enzymatic_exp_steps,
            "note": "rows processed sequentially (separate solutions)",
        }
