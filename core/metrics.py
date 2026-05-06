"""Metrics dataclasses used across all paradigms."""

from dataclasses import dataclass, field


@dataclass
class StepMetrics:
    """Metrics for a single attention step (e.g., QKT matmul, softmax)."""
    name: str
    flops: int = 0
    memory_reads_bytes: int = 0
    memory_writes_bytes: int = 0
    wall_time_seconds: float = 0.0
    estimated_hw_time_seconds: float = 0.0  # Estimated time on actual hardware
    is_real_execution: bool = False
    paradigm_specific: dict = field(default_factory=dict)

    @property
    def effective_time(self) -> float:
        """Best available time: wall time for real HW, hw estimate for sims."""
        if self.is_real_execution:
            return self.wall_time_seconds
        if self.estimated_hw_time_seconds > 0:
            return self.estimated_hw_time_seconds
        return self.wall_time_seconds

    @property
    def memory_total_bytes(self) -> int:
        return self.memory_reads_bytes + self.memory_writes_bytes

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs per byte of memory traffic."""
        total_bytes = self.memory_total_bytes
        if total_bytes == 0:
            return 0.0
        return self.flops / total_bytes


@dataclass
class ParadigmResult:
    """Full result from running attention on one paradigm + one config."""
    paradigm_name: str
    paradigm_short: str
    seq_len: int
    embed_dim: int
    steps: list[StepMetrics] = field(default_factory=list)
    paradigm_info: dict = field(default_factory=dict)

    @property
    def total_flops(self) -> int:
        return sum(s.flops for s in self.steps)

    @property
    def total_wall_time(self) -> float:
        return sum(s.wall_time_seconds for s in self.steps)

    @property
    def total_effective_time(self) -> float:
        """Best available time per paradigm (hw estimate or wall time)."""
        return sum(s.effective_time for s in self.steps)

    @property
    def total_estimated_hw_time(self) -> float:
        """Total theoretical/simulated hardware time."""
        return sum(s.estimated_hw_time_seconds for s in self.steps)

    @property
    def total_memory_bytes(self) -> int:
        return sum(s.memory_total_bytes for s in self.steps)

    @property
    def arithmetic_intensity(self) -> float:
        total_mem = self.total_memory_bytes
        if total_mem == 0:
            return 0.0
        return self.total_flops / total_mem

    @property
    def throughput_gflops(self) -> float:
        t = self.total_effective_time
        if t == 0:
            return 0.0
        return self.total_flops / t / 1e9

    def to_dict(self) -> dict:
        """Flat dict for DataFrame/CSV export."""
        d = {
            "paradigm": self.paradigm_name,
            "paradigm_short": self.paradigm_short,
            "seq_len": self.seq_len,
            "embed_dim": self.embed_dim,
            "total_flops": self.total_flops,
            "total_wall_time_s": self.total_wall_time,
            "total_effective_time_s": self.total_effective_time,
            "total_estimated_hw_time_s": self.total_estimated_hw_time,
            "total_memory_bytes": self.total_memory_bytes,
            "arithmetic_intensity": self.arithmetic_intensity,
            "throughput_gflops": self.throughput_gflops,
        }
        # Per-step times
        for step in self.steps:
            d[f"time_{step.name}"] = step.effective_time
            d[f"flops_{step.name}"] = step.flops
        # Paradigm info
        for k, v in self.paradigm_info.items():
            d[f"info_{k}"] = v
        return d

    def to_full_dict(self) -> dict:
        """Nested dict preserving all data including paradigm_specific. For JSON logging."""
        return {
            "paradigm_name": self.paradigm_name,
            "paradigm_short": self.paradigm_short,
            "seq_len": self.seq_len,
            "embed_dim": self.embed_dim,
            "total_flops": self.total_flops,
            "total_wall_time_s": self.total_wall_time,
            "total_effective_time_s": self.total_effective_time,
            "total_memory_bytes": self.total_memory_bytes,
            "arithmetic_intensity": self.arithmetic_intensity,
            "throughput_gflops": self.throughput_gflops,
            "paradigm_info": self.paradigm_info,
            "steps": [
                {
                    "name": s.name,
                    "flops": s.flops,
                    "memory_reads_bytes": s.memory_reads_bytes,
                    "memory_writes_bytes": s.memory_writes_bytes,
                    "wall_time_seconds": s.wall_time_seconds,
                    "estimated_hw_time_seconds": s.estimated_hw_time_seconds,
                    "effective_time": s.effective_time,
                    "arithmetic_intensity": s.arithmetic_intensity,
                    "paradigm_specific": s.paradigm_specific,
                }
                for s in self.steps
            ],
        }
