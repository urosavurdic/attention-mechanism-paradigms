"""Systolic array simulator for matrix multiplication."""

import numpy as np


class SystolicArray:
    """Simulates a 2D systolic array for matrix multiplication.

    Data flow:
    - Matrix A rows feed from the left (staggered by 1 cycle per row)
    - Matrix B columns feed from the top (staggered by 1 cycle per column)
    - Each PE does: accumulator += left_in * top_in, passes data through
    """

    def __init__(self, rows: int = 16, cols: int = 16):
        self.rows = rows
        self.cols = cols

    def matmul_cycles(self, M: int, K: int, N: int) -> dict:
        """Calculate cycles for (M,K) @ (K,N) matmul on this array.

        Tiling: if M > rows or N > cols, we tile the computation.
        Each tile processes min(rows, remaining_M) x min(cols, remaining_N).
        """
        R, C = self.rows, self.cols

        # Number of tiles
        tiles_m = (M + R - 1) // R
        tiles_n = (N + C - 1) // C

        # Cycles per tile: K (streaming) + R + C - 2 (pipeline fill/drain)
        cycles_per_tile = K + R + C - 2

        # Total cycles (tiles are sequential for simplicity)
        total_tiles = tiles_m * tiles_n
        total_cycles = total_tiles * cycles_per_tile

        # PE utilization
        # Useful PEs per tile: min(R, M_remaining) * min(C, N_remaining)
        # But for simplicity, use average:
        avg_useful_pes = min(R, M) * min(C, N)
        total_pes = R * C
        pe_utilization = avg_useful_pes / total_pes

        # Steady-state utilization (during K streaming cycles)
        # All PEs active during streaming, idle during fill/drain
        streaming_cycles = K * total_tiles
        fill_drain_cycles = (R + C - 2) * total_tiles
        time_utilization = streaming_cycles / total_cycles if total_cycles > 0 else 0

        # Ops per cycle in steady state
        ops_per_cycle_peak = R * C * 2  # Each PE does 1 MUL + 1 ADD

        # Data reuse: each element of A is used C times, each element of B is used R times
        a_reuse_factor = C
        b_reuse_factor = R

        return {
            "total_cycles": total_cycles,
            "cycles_per_tile": cycles_per_tile,
            "total_tiles": total_tiles,
            "tiles_m": tiles_m,
            "tiles_n": tiles_n,
            "pe_utilization": pe_utilization,
            "time_utilization": time_utilization,
            "ops_per_cycle_peak": ops_per_cycle_peak,
            "total_ops": 2 * M * K * N,
            "a_reuse_factor": a_reuse_factor,
            "b_reuse_factor": b_reuse_factor,
        }

    def simulate_matmul(self, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, dict]:
        """Actually compute matmul tile-by-tile and return result + stats.

        This validates correctness while collecting cycle counts.
        """
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, f"Shape mismatch: A({M},{K}) @ B({K2},{N})"

        C_result = np.zeros((M, N), dtype=A.dtype)
        R, Cols = self.rows, self.cols

        total_cycles = 0
        total_tiles = 0

        for tile_m in range(0, M, R):
            for tile_n in range(0, N, Cols):
                m_end = min(tile_m + R, M)
                n_end = min(tile_n + Cols, N)
                tile_rows = m_end - tile_m
                tile_cols = n_end - tile_n

                # Compute this tile (actual matmul)
                A_tile = A[tile_m:m_end, :]
                B_tile = B[:, tile_n:n_end]
                C_result[tile_m:m_end, tile_n:n_end] = A_tile @ B_tile

                # Count cycles for this tile
                total_cycles += K + tile_rows + tile_cols - 2
                total_tiles += 1

        stats = {
            "total_cycles": total_cycles,
            "total_tiles": total_tiles,
            "array_size": f"{R}x{Cols}",
        }
        return C_result, stats
