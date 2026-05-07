"""CdTe Turing Machine with 4 operations: SKIP, CSKIP, INC, ADD."""

from enum import IntEnum
from dataclasses import dataclass


class TmOp(IntEnum):
    SKIP = 0      # Move head right by 1
    CSKIP = 1     # Move head right by 1 if tape[head] == 0
    INC = 2       # tape[head] += 1
    ADD = 3       # tape[head] += tape[head + offset]


@dataclass
class TmInstruction:
    op: TmOp
    offset: int = 0       # Used by ADD (relative offset to source)
    comment: str = ""     # For debugging


class TuringMachine:
    """Turing Machine with minimal 4-instruction ISA."""

    def __init__(self, tape_size: int = 4096):
        self.tape = [0] * tape_size
        self.head = 0
        self.steps = 0
        self.op_counts = {op.name: 0 for op in TmOp}

    def reset(self):
        self.tape = [0] * len(self.tape)
        self.head = 0
        self.steps = 0
        self.op_counts = {op.name: 0 for op in TmOp}

    def execute(self, program: list[TmInstruction], max_steps: int = 10_000_000) -> int:
        """Execute program. Returns total steps."""
        self.steps = 0
        pc = 0

        while pc < len(program) and self.steps < max_steps:
            inst = program[pc]
            self.steps += 1
            self.op_counts[inst.op.name] += 1

            if inst.op == TmOp.SKIP:
                self.head += 1
                pc += 1

            elif inst.op == TmOp.CSKIP:
                if self.tape[self.head] == 0:
                    self.head += 1
                pc += 1

            elif inst.op == TmOp.INC:
                self.tape[self.head] += 1
                pc += 1

            elif inst.op == TmOp.ADD:
                src = self.head + inst.offset
                if 0 <= src < len(self.tape):
                    self.tape[self.head] += self.tape[src]
                pc += 1

        return self.steps


def program_multiply(a: int, b: int) -> list[TmInstruction]:
    """Multiply a * b using repeated addition.

    Tape layout: [a, b, result, temp]
    Result = add 'a' to result 'b' times.
    Since we only have INC and ADD, we implement repeated addition.
    """
    # Direct approach: result = 0, then add 'a' to result 'b' times
    # We pre-compute and use INC to build the result
    # This demonstrates the cost: O(a * b) INC operations
    instructions = []
    result = a * b
    # Move head to result position
    instructions.append(TmInstruction(TmOp.SKIP))  # pos 0 -> 1
    instructions.append(TmInstruction(TmOp.SKIP))  # pos 1 -> 2
    # Increment result times
    for _ in range(result):
        instructions.append(TmInstruction(TmOp.INC))
    return instructions


def program_dot_product(vec_a: list[int], vec_b: list[int]) -> list[TmInstruction]:
    """Compute dot product of two integer vectors.

    For each pair (a_i, b_i): compute a_i * b_i, add to accumulator.
    Total operations: sum(a_i * b_i) INC ops + overhead.
    """
    instructions = []
    K = len(vec_a)

    # Tape layout: [acc, temp, ...]
    # We compute each product and add to accumulator
    for i in range(K):
        product = vec_a[i] * vec_b[i]
        for _ in range(product):
            instructions.append(TmInstruction(TmOp.INC))

    return instructions


def estimate_matmul_ops(M: int, K: int, N: int, max_val: int = 10) -> dict:
    """Analytically estimate TM operations for integer matmul.

    Each multiply of values a*b costs O(a*b) INC operations.
    Average value is max_val/2, so average multiply cost = (max_val/2)^2.
    Total multiplies = M * K * N.
    """
    avg_val = max_val / 2
    avg_multiply_cost = avg_val * avg_val  # O(a*b) INCs per multiply
    total_multiplies = M * K * N
    total_additions = M * N * K           # accumulate K products per element
    skip_overhead = M * N * K * 2         # head movement

    return {
        "total_inc_ops": int(total_multiplies * avg_multiply_cost),
        "total_add_ops": int(total_additions),
        "total_skip_ops": int(skip_overhead),
        "total_ops": int(total_multiplies * avg_multiply_cost + total_additions + skip_overhead),
        "multiplies": int(total_multiplies),
        "avg_cost_per_multiply": avg_multiply_cost,
    }


def estimate_softmax_ops(N: int, max_val: int = 10) -> dict:
    """Analytically estimate TM operations for softmax on integers.

    Softmax requires: max, subtract, exp, sum, divide.
    On a TM with only INC and ADD there is no exp and no divide, so both are
    charged as series expansions over the integer-scaled value range: exp at
    ~10 operations per unit of magnitude and divide at ~5, with max costing a
    comparison sweep. These are order-of-magnitude coefficients chosen for the
    scale of max_val, not a derivation -- a faithful series expansion would be
    far more expensive and would swamp every other step.
    """
    avg_val = max_val / 2
    # Finding max: N comparisons, each O(max_val) ops
    max_ops = N * max_val
    # Exp approximation: we can't do real exp, so estimate cost
    exp_ops = N * avg_val * 10  # rough approximation
    # Sum: N additions
    sum_ops = N * avg_val
    # Division: repeated subtraction, O(value / divisor)
    div_ops = N * avg_val * 5

    total = max_ops + exp_ops + sum_ops + div_ops
    return {
        "total_ops": int(total),
        "max_ops": int(max_ops),
        "exp_ops": int(exp_ops),
        "sum_ops": int(sum_ops),
        "div_ops": int(div_ops),
    }
