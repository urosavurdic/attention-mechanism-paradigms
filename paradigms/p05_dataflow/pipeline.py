"""Maxeler-style dataflow pipeline simulation."""

from dataclasses import dataclass, field


@dataclass
class PipelineNode:
    """A single node in the dataflow pipeline."""
    name: str
    latency: int = 1           # Cycles to produce output after input
    ops_per_element: int = 1   # FLOPs per element processed

    # Runtime state
    _buffer: list = field(default_factory=list, repr=False)
    _output_ready_at: int = -1

    def reset(self):
        self._buffer = []
        self._output_ready_at = -1


class DataflowPipeline:
    """Simulates a streaming dataflow pipeline (Maxeler-style).

    Data flows through a chain of nodes. Each node processes one element
    per cycle (once pipeline is full). Total time = depth + stream_length - 1.
    """

    def __init__(self, nodes: list[PipelineNode]):
        self.nodes = nodes
        self.depth = sum(n.latency for n in nodes)

    def simulate(self, stream_length: int) -> dict:
        """Simulate pipeline execution for a given stream length.

        Returns:
            dict with total_cycles, throughput, utilization, etc.
        """
        # Pipeline startup: depth cycles to fill
        # Steady state: stream_length - 1 cycles of full throughput
        # Pipeline drain: depth cycles
        # But in a well-designed pipeline, drain overlaps with next batch

        total_cycles = self.depth + stream_length - 1
        total_ops = sum(n.ops_per_element for n in self.nodes) * stream_length

        # Utilization: fraction of node-cycles that do useful work
        # Total node-cycles = len(nodes) * total_cycles
        # Useful node-cycles = stream_length * len(nodes) (steady state)
        total_node_cycles = len(self.nodes) * total_cycles
        useful_node_cycles = stream_length * len(self.nodes)
        utilization = useful_node_cycles / total_node_cycles if total_node_cycles > 0 else 0

        # Throughput: elements per cycle in steady state
        throughput = 1.0  # 1 element per cycle once pipeline is full

        return {
            "total_cycles": total_cycles,
            "pipeline_depth": self.depth,
            "stream_length": stream_length,
            "total_ops": total_ops,
            "utilization": utilization,
            "throughput_elem_per_cycle": throughput,
            "num_stages": len(self.nodes),
        }


def build_matmul_pipeline(K: int) -> DataflowPipeline:
    """Build a pipeline for dot product of length K.

    Pipeline: K stages of multiply-accumulate.
    Each stage: FMUL (3 cycles) + FADD (2 cycles) = 5 cycles latency,
    but pipelined so 1 element per cycle throughput.
    """
    nodes = []
    for i in range(K):
        nodes.append(PipelineNode(
            name=f"mac_{i}",
            latency=5,       # FMUL + FADD pipeline latency
            ops_per_element=2,  # 1 mul + 1 add
        ))
    return DataflowPipeline(nodes)


def build_softmax_pipeline(N: int) -> DataflowPipeline:
    """Build pipeline for softmax normalization.

    Stages: max-reduction, subtract-max, exp, sum-reduction, divide.
    Note: reductions require buffering the full row, adding latency.
    """
    nodes = [
        PipelineNode(name="max_reduce", latency=N, ops_per_element=1),
        PipelineNode(name="subtract_max", latency=1, ops_per_element=1),
        PipelineNode(name="exp", latency=8, ops_per_element=1),  # exp is expensive
        PipelineNode(name="sum_reduce", latency=N, ops_per_element=1),
        PipelineNode(name="divide", latency=10, ops_per_element=1),
    ]
    return DataflowPipeline(nodes)


def build_attention_pipelines(seq_len: int, embed_dim: int) -> dict:
    """Build all pipelines for full attention computation."""
    return {
        "qkv_projection": build_matmul_pipeline(embed_dim),
        "qk_matmul": build_matmul_pipeline(embed_dim),
        "softmax": build_softmax_pipeline(seq_len),
        "av_matmul": build_matmul_pipeline(seq_len),
        "output_projection": build_matmul_pipeline(embed_dim),
    }
