"""Mach-Zehnder Interferometer mesh model for photonic matrix multiplication."""

import math


class MZIMesh:
    """Models a programmable MZI mesh for optical matrix multiplication."""

    def __init__(self, mesh_size, propagation_time_s, dac_rate_hz,
                 oeo_latency_s, electronic_softmax_clock_hz):
        self.mesh_size = mesh_size
        self.propagation_time = propagation_time_s
        self.dac_rate = dac_rate_hz
        self.oeo_latency = oeo_latency_s
        self.electronic_softmax_clock = electronic_softmax_clock_hz

    def tile_time(self):
        dac_time = self.mesh_size / self.dac_rate
        adc_time = self.mesh_size / self.dac_rate
        return max(dac_time, adc_time, self.propagation_time)

    def matmul_time(self, M, K, N):
        tiles_m = math.ceil(M / self.mesh_size)
        tiles_n = math.ceil(N / self.mesh_size)
        tiles_k = math.ceil(K / self.mesh_size)
        num_tiles = tiles_m * tiles_n * tiles_k
        return num_tiles * self.tile_time(), num_tiles

    def softmax_time(self, seq_len):
        electronic_per_row = 5 * seq_len / self.electronic_softmax_clock
        per_row = self.oeo_latency + electronic_per_row
        return seq_len * per_row
