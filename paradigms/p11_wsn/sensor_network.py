"""Wireless sensor network model with distributed computation and communication."""

import math


class SensorNetwork:
    """Models a distributed WSN with constrained MCU nodes and 802.15.4 wireless."""

    def __init__(self, num_nodes, mcu_clock_hz, cycles_per_mac, node_ram_bytes,
                 wireless_bps, hop_latency_s, sync_rounds_softmax):
        self.num_nodes = num_nodes
        self.mcu_clock = mcu_clock_hz
        self.cycles_per_mac = cycles_per_mac
        self.node_ram = node_ram_bytes
        self.wireless_bps = wireless_bps
        self.hop_latency = hop_latency_s
        self.sync_rounds = sync_rounds_softmax

    def tile_dim(self, dtype_bytes=4):
        return int(math.sqrt(self.node_ram / (3 * dtype_bytes)))

    def communication_time(self, data_bytes):
        return data_bytes * 8 / self.wireless_bps + self.hop_latency

    def matmul_time(self, M, K, N, dtype_bytes=4):
        td = self.tile_dim(dtype_bytes)
        tiles_m = math.ceil(M / td)
        tiles_n = math.ceil(N / td)
        total_tiles = tiles_m * tiles_n

        rounds = math.ceil(total_tiles / self.num_nodes)

        tile_macs = td * td * K
        compute_time = tile_macs * self.cycles_per_mac / self.mcu_clock

        distribute_bytes = (td * K + K * td) * dtype_bytes
        collect_bytes = td * td * dtype_bytes
        comm_distribute = self.communication_time(distribute_bytes)
        comm_collect = self.communication_time(collect_bytes)

        per_round = compute_time + comm_distribute + comm_collect
        total_time = rounds * per_round

        return total_time, {
            "tile_dim": td,
            "total_tiles": total_tiles,
            "rounds": rounds,
            "compute_per_round_s": compute_time,
            "comm_distribute_s": comm_distribute,
            "comm_collect_s": comm_collect,
        }

    def softmax_time(self, seq_len, dtype_bytes=4):
        local_rows = math.ceil(seq_len / self.num_nodes)
        local_ops = 5 * local_rows * seq_len
        local_compute = local_ops * self.cycles_per_mac / self.mcu_clock

        sync_data = self.num_nodes * dtype_bytes
        sync_round_time = self.communication_time(sync_data)
        total_sync = self.sync_rounds * sync_round_time

        return local_compute + total_sync, {
            "local_rows_per_node": local_rows,
            "local_compute_s": local_compute,
            "sync_rounds": self.sync_rounds,
            "sync_time_s": total_sync,
        }
