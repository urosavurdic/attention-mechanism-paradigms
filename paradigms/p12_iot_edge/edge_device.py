"""ARM Cortex-M4F edge device model for cycle-accurate estimation."""

import math


class CortexM4Device:
    """Models ARM Cortex-M4F with hardware FPU and DSP MAC instruction."""

    def __init__(self, clock_hz, cycles_per_mac, sram_bytes, flash_bytes,
                 exp_lut_cycles, div_cycles):
        self.clock_hz = clock_hz
        self.cycles_per_mac = cycles_per_mac
        self.sram_bytes = sram_bytes
        self.flash_bytes = flash_bytes
        self.exp_lut_cycles = exp_lut_cycles
        self.div_cycles = div_cycles

    def max_tile_dim(self, dtype_bytes=4):
        available = self.sram_bytes - 1024
        return int(math.sqrt(available / (3 * dtype_bytes)))

    def needs_tiling(self, M, K, N, dtype_bytes=4):
        total = (M * K + K * N + M * N) * dtype_bytes
        return total > self.sram_bytes

    def matmul_cycles(self, M, K, N, dtype_bytes=4):
        mac_cycles = M * K * N * self.cycles_per_mac
        pipeline_startups = M * N * 3

        if not self.needs_tiling(M, K, N, dtype_bytes):
            return mac_cycles + pipeline_startups

        tile_dim = self.max_tile_dim(dtype_bytes)
        tiles_m = math.ceil(M / tile_dim)
        tiles_n = math.ceil(N / tile_dim)
        total_tiles = tiles_m * tiles_n

        words_per_tile_load = 2 * tile_dim * K
        flash_stall_cycles = total_tiles * words_per_tile_load * 2

        return mac_cycles + pipeline_startups + flash_stall_cycles

    def softmax_cycles_per_row(self, seq_len):
        cmp_cycles = seq_len - 1
        sub_cycles = seq_len
        exp_cycles = seq_len * self.exp_lut_cycles
        add_cycles = seq_len - 1
        div_cyc = seq_len * self.div_cycles
        return cmp_cycles + sub_cycles + exp_cycles + add_cycles + div_cyc

    def softmax_cycles(self, seq_len):
        return seq_len * self.softmax_cycles_per_row(seq_len)

    def cycles_to_time(self, cycles):
        return cycles / self.clock_hz
