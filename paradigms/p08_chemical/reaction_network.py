"""Chemical reaction network model for analog computation."""

import math


class ChemicalReactionNetwork:
    """Models CRN computation with bimolecular reactions and well-based parallelism."""

    def __init__(self, reaction_rate_hz, well_size, mixing_time_s,
                 exponential_rate_hz):
        self.reaction_rate = reaction_rate_hz
        self.well_size = well_size
        self.mixing_time = mixing_time_s
        self.exponential_rate = exponential_rate_hz

    def matmul_time(self, M, K, N):
        batch_rounds = math.ceil(M * N / self.well_size)
        time_per_batch = K / self.reaction_rate + self.mixing_time
        total_time = batch_rounds * time_per_batch

        return total_time, {
            "batch_rounds": batch_rounds,
            "time_per_batch_s": time_per_batch,
            "parallel_wells": min(M * N, self.well_size),
            "total_reactions": M * K * N,
        }

    def softmax_time(self, seq_len):
        exp_time_per_row = seq_len / self.exponential_rate
        max_steps = math.ceil(math.log2(max(seq_len, 2)))
        overhead_per_row = (max_steps + 2) / self.reaction_rate
        softmax_per_row = exp_time_per_row + overhead_per_row
        parallel_rows = min(seq_len, self.well_size)
        row_rounds = math.ceil(seq_len / parallel_rows)
        total_time = row_rounds * softmax_per_row

        return total_time, {
            "exp_time_per_row_s": exp_time_per_row,
            "max_reaction_steps": max_steps,
            "overhead_per_row_s": overhead_per_row,
            "parallel_rows": parallel_rows,
            "row_rounds": row_rounds,
        }
