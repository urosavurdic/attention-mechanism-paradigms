"""RISC-V simulator - fetch/decode/execute loop."""

from paradigms.p03_gaas_riscv.isa import Instruction, Op, CYCLE_COSTS
import numpy as np


class RiscVSimulator:
    """Minimal RISC-V simulator for instruction counting and cycle estimation."""

    def __init__(self, memory_size: int = 65536):
        self.registers = np.zeros(32, dtype=np.float64)  # 32 float registers
        self.memory = np.zeros(memory_size, dtype=np.float64)
        self.pc = 0
        self.halted = False

        # Counters
        self.total_instructions = 0
        self.total_cycles = 0
        self.instruction_counts = {}

    def load_data(self, data: np.ndarray, offset: int = 0):
        """Load data into simulator memory."""
        self.memory[offset:offset + len(data)] = data.flatten()

    def read_data(self, offset: int, size: int) -> np.ndarray:
        """Read data from simulator memory."""
        return self.memory[offset:offset + size].copy()

    def execute(self, program: list[Instruction], max_steps: int = 10_000_000):
        """Run program to completion or max_steps."""
        self.pc = 0
        self.halted = False
        self.total_instructions = 0
        self.total_cycles = 0
        self.instruction_counts = {}

        for _ in range(max_steps):
            if self.pc >= len(program) or self.halted:
                break

            inst = program[self.pc]
            self._execute_one(inst)
            self.total_instructions += 1
            self.total_cycles += CYCLE_COSTS.get(inst.op, 1)

            name = inst.op.name
            self.instruction_counts[name] = self.instruction_counts.get(name, 0) + 1

        return self.total_instructions, self.total_cycles

    def _execute_one(self, inst: Instruction):
        """Execute a single instruction."""
        op = inst.op

        if op == Op.NOP:
            self.pc += 1

        elif op == Op.LW:
            addr = int(self.registers[inst.rs1] + inst.imm)
            if 0 <= addr < len(self.memory):
                self.registers[inst.rd] = self.memory[addr]
            self.pc += 1

        elif op == Op.SW:
            addr = int(self.registers[inst.rs1] + inst.imm)  # Fixed: use rs1 for value
            if 0 <= addr < len(self.memory):
                self.memory[addr] = self.registers[inst.rd]
            self.pc += 1

        elif op == Op.ADD:
            self.registers[inst.rd] = self.registers[inst.rs1] + self.registers[inst.rs2]
            self.pc += 1

        elif op == Op.MUL:
            self.registers[inst.rd] = self.registers[inst.rs1] * self.registers[inst.rs2]
            self.pc += 1

        elif op == Op.FADD:
            self.registers[inst.rd] = self.registers[inst.rs1] + self.registers[inst.rs2]
            self.pc += 1

        elif op == Op.FMUL:
            self.registers[inst.rd] = self.registers[inst.rs1] * self.registers[inst.rs2]
            self.pc += 1

        elif op == Op.FDIV:
            denom = self.registers[inst.rs2]
            if denom != 0:
                self.registers[inst.rd] = self.registers[inst.rs1] / denom
            self.pc += 1

        elif op == Op.BEQ:
            if self.registers[inst.rs1] == self.registers[inst.rs2]:
                self.pc += inst.imm
            else:
                self.pc += 1

        elif op == Op.BNE:
            if self.registers[inst.rs1] != self.registers[inst.rs2]:
                self.pc += inst.imm
            else:
                self.pc += 1

        elif op == Op.JAL:
            self.registers[inst.rd] = self.pc + 1
            self.pc = inst.imm

        elif op == Op.HALT:
            self.halted = True

        else:
            self.pc += 1
