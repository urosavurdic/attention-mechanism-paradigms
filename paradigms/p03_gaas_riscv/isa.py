"""RISC-V ISA definition - minimal subset for attention simulation."""

from enum import IntEnum
from dataclasses import dataclass


class Op(IntEnum):
    """Minimal RISC-V-like opcodes."""
    NOP = 0
    LW = 1       # Load word: rd = mem[rs1 + imm]
    SW = 2       # Store word: mem[rs1 + imm] = rs2
    ADD = 3      # rd = rs1 + rs2
    MUL = 4      # rd = rs1 * rs2
    FADD = 5     # rd = rs1 + rs2 (float)
    FMUL = 6     # rd = rs1 * rs2 (float)
    FDIV = 7     # rd = rs1 / rs2 (float)
    BEQ = 8      # Branch if rs1 == rs2, offset = imm
    BNE = 9      # Branch if rs1 != rs2, offset = imm
    JAL = 10     # Jump and link: rd = PC+1, PC = imm
    HALT = 11


@dataclass
class Instruction:
    op: Op
    rd: int = 0   # destination register
    rs1: int = 0  # source register 1
    rs2: int = 0  # source register 2
    imm: int = 0  # immediate value


# Cycle costs per operation (simplified pipeline model)
CYCLE_COSTS = {
    Op.NOP: 1,
    Op.LW: 4,       # Memory access latency
    Op.SW: 3,
    Op.ADD: 1,
    Op.MUL: 3,      # Multiply takes 3 cycles
    Op.FADD: 2,
    Op.FMUL: 3,
    Op.FDIV: 10,    # Division is expensive
    Op.BEQ: 1,
    Op.BNE: 1,
    Op.JAL: 1,
    Op.HALT: 1,
}


def compile_matmul(M: int, K: int, N: int) -> list[Instruction]:
    """Generate instructions for (M,K) @ (K,N) matmul.

    Uses naive triple loop. Returns instruction list.
    Register allocation:
      r0=zero, r1=i, r2=j, r3=k, r4=acc, r5=a_val, r6=b_val, r7=tmp
      r8=M, r9=K, r10=N
    """
    instructions = []

    # For counting purposes, we generate the logical instruction sequence
    # for the inner loops. Each element of C requires:
    #   - K loads of A[i][k]
    #   - K loads of B[k][j]
    #   - K multiplies
    #   - K adds (accumulate)
    #   - 1 store of C[i][j]
    #   - loop control branches

    for i in range(M):
        for j in range(N):
            # Zero accumulator
            instructions.append(Instruction(Op.ADD, rd=4, rs1=0, rs2=0))
            for k in range(K):
                instructions.append(Instruction(Op.LW, rd=5, rs1=0, imm=i * K + k))   # load A[i][k]
                instructions.append(Instruction(Op.LW, rd=6, rs1=0, imm=M * K + k * N + j))  # load B[k][j]
                instructions.append(Instruction(Op.FMUL, rd=7, rs1=5, rs2=6))          # tmp = a * b
                instructions.append(Instruction(Op.FADD, rd=4, rs1=4, rs2=7))          # acc += tmp
            instructions.append(Instruction(Op.SW, rd=0, rs1=4, imm=M * K + K * N + i * N + j))  # store C[i][j]

    instructions.append(Instruction(Op.HALT))
    return instructions


def compile_softmax(N: int) -> list[Instruction]:
    """Generate instructions for softmax over a vector of length N.

    Steps: find max, subtract max, exponentiate (approx), sum, divide.
    """
    instructions = []

    # Find max (N-1 comparisons)
    instructions.append(Instruction(Op.LW, rd=1, rs1=0, imm=0))  # max = data[0]
    for i in range(1, N):
        instructions.append(Instruction(Op.LW, rd=2, rs1=0, imm=i))
        # Approximate comparison via subtract + branch
        instructions.append(Instruction(Op.FADD, rd=3, rs1=2, rs2=1))  # proxy for comparison
        instructions.append(Instruction(Op.BNE, rs1=3, rs2=0, imm=1))  # branch
        instructions.append(Instruction(Op.ADD, rd=1, rs1=2, rs2=0))   # max = data[i]

    # Subtract max + exp (approximated as multiply chain) + sum
    instructions.append(Instruction(Op.ADD, rd=4, rs1=0, rs2=0))  # sum = 0
    for i in range(N):
        instructions.append(Instruction(Op.LW, rd=2, rs1=0, imm=i))
        instructions.append(Instruction(Op.FADD, rd=2, rs1=2, rs2=1))   # x - max (approx)
        # exp approximation: 3 multiply-adds (Taylor-like)
        instructions.append(Instruction(Op.FMUL, rd=3, rs1=2, rs2=2))
        instructions.append(Instruction(Op.FADD, rd=3, rs1=3, rs2=2))
        instructions.append(Instruction(Op.FADD, rd=3, rs1=3, rs2=0))   # ~exp(x)
        instructions.append(Instruction(Op.SW, rd=0, rs1=3, imm=i))      # store exp
        instructions.append(Instruction(Op.FADD, rd=4, rs1=4, rs2=3))   # sum += exp

    # Divide each by sum
    for i in range(N):
        instructions.append(Instruction(Op.LW, rd=2, rs1=0, imm=i))
        instructions.append(Instruction(Op.FDIV, rd=2, rs1=2, rs2=4))
        instructions.append(Instruction(Op.SW, rd=0, rs1=2, imm=i))

    instructions.append(Instruction(Op.HALT))
    return instructions


def count_instructions(program: list[Instruction]) -> dict:
    """Count instructions by type."""
    counts = {}
    for inst in program:
        name = inst.op.name
        counts[name] = counts.get(name, 0) + 1
    return counts


def estimate_cycles(program: list[Instruction]) -> int:
    """Estimate total cycles for a program."""
    return sum(CYCLE_COSTS[inst.op] for inst in program)
