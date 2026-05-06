"""Experiment configuration parameters."""

BUILD_VERSION = "2026-05-19-v6-fp16-sdpa-backend"

# Scaling sweep
SEQ_LENS = [128, 256, 512, 1024, 2048]
EMBED_DIMS = [64, 128, 256]
NUM_HEADS = 8

# Benchmarking
WARMUP_RUNS = 2
BENCHMARK_RUNS = 5

# Simulator limits (for P03/P04 which are slow)
SIM_MAX_SEQ_LEN = 32
SIM_MAX_EMBED_DIM = 16

# Estimated clock frequencies for simulated architectures (Hz)
GAAS_CLOCK_HZ = 1.0e9         # GaAs: ~1 GHz (high-speed switching)
CDTE_CLOCK_HZ = 10.0e6        # CdTe: ~10 MHz (very few transistors)
DATAFLOW_CLOCK_HZ = 500.0e6   # Maxeler DFE: ~500 MHz
SYSTOLIC_CLOCK_HZ = 700.0e6   # TPU v1: ~700 MHz

# TPU systolic array sizing
TPU_ARRAY_SIZE = 128           # 128x128 PE array (TPU v1 MXU scale)
TPU_VECTOR_WIDTH = 128         # Vector unit width for softmax/non-matmul ops

# Dataflow parallelism
DATAFLOW_PARALLEL_PIPES = 4    # Parallel kernel instances on DFE

# CPU roofline model (typical desktop, e.g. Intel i7 with AVX2)
CPU_PEAK_TFLOPS_FP32 = 0.5       # ~500 GFLOPS (8 cores × 16 FP32 ops/cycle × ~3.8 GHz)
CPU_MEMORY_BW_GB_S = 50.0        # DDR4-3200 dual-channel: ~50 GB/s

# GPU roofline model (NVIDIA T4 - Google Colab free tier)
GPU_PEAK_TFLOPS_FP32 = 8.1       # T4: 8.1 TFLOPS FP32
GPU_MEMORY_BW_GB_S = 320.0       # T4: 320 GB/s memory bandwidth
GPU_KERNEL_LAUNCH_OVERHEAD_S = 10e-6  # ~10 µs per kernel launch

# TPU real execution
TPU_WARMUP_RUNS = 3

# P07: Optical Computing (Photonic MZI Mesh)
OPTICAL_MZI_MESH_SIZE = 64
OPTICAL_PROPAGATION_TIME_S = 0.1e-9
OPTICAL_DAC_RATE_HZ = 10.0e9
OPTICAL_OEO_LATENCY_S = 10.0e-9
OPTICAL_ELECTRONIC_SOFTMAX_CLOCK_HZ = 2.0e9

# P08: Chemical Computing (Reaction-Diffusion)
CHEM_REACTION_RATE_HZ = 1.0e3
CHEM_WELL_SIZE = 256
CHEM_MIXING_TIME_S = 1.0e-3
CHEM_EXPONENTIAL_RATE_HZ = 100.0

# P09: Biological Computing (DNA Strand Displacement)
DNA_HYBRIDIZATION_RATE_HZ = 0.1
DNA_STRAND_DISPLACEMENT_TIME_S = 10.0
DNA_CASCADE_DEPTH_PER_MULTIPLY = 6
DNA_MAX_PARALLELISM = 1024
DNA_ENZYMATIC_EXP_STEPS = 20

# P10: Quantum Computing (Gate-based)
QUANTUM_GATE_TIME_S = 50.0e-9
QUANTUM_MEASUREMENT_TIME_S = 1.0e-6
QUANTUM_TOFFOLI_T_COUNT = 7
QUANTUM_QUBIT_COUNT = 1000
QUANTUM_CLASSICAL_CLOCK_HZ = 3.0e9

# P11: Wireless Sensor Network
WSN_MCU_CLOCK_HZ = 16.0e6
WSN_MCU_CYCLES_PER_MAC = 4
WSN_NODE_RAM_BYTES = 10240
WSN_WIRELESS_BPS = 250.0e3
WSN_HOP_LATENCY_S = 5.0e-3
WSN_NUM_NODES = 64
WSN_SYNC_ROUNDS_SOFTMAX = 3

# P12: IoT Edge Computing (ARM Cortex-M4F)
IOT_CLOCK_HZ = 168.0e6
IOT_CYCLES_PER_MAC = 3
IOT_SRAM_BYTES = 256 * 1024
IOT_FLASH_BYTES = 1024 * 1024
IOT_EXP_LUT_CYCLES = 10
IOT_DIV_CYCLES = 14

# Output
RESULTS_DIR = "results"
LOG_DIR = "results/logs"
