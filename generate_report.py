"""Generate the results report: text with inline figures, built from the canonical CSV."""

import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import textwrap

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
pdf_path = os.path.join(RESULTS, 'SICAAI-2026-report.pdf')

# --- Helpers ---

def add_text_page(pdf, title, body, font_size=10):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.08, 0.94, title, ha='left', va='top',
             fontsize=15, fontweight='bold', family='sans-serif')
    fig.text(0.08, 0.91, body, ha='left', va='top',
             fontsize=font_size, family='sans-serif', linespacing=1.55,
             wrap=True, transform=fig.transFigure,
             verticalalignment='top',
             multialignment='left')
    pdf.savefig(fig)
    plt.close(fig)


def add_figure_page(pdf, image_path, caption, subtitle=None):
    fig = plt.figure(figsize=(8.5, 11))
    img = mpimg.imread(image_path)
    h, w = img.shape[:2]
    aspect = w / h
    fig_w = 0.88
    fig_h = min(fig_w / aspect, 0.55)
    top = 0.92 if not subtitle else 0.88
    ax = fig.add_axes([0.06, top - fig_h, fig_w, fig_h])
    ax.imshow(img)
    ax.axis('off')
    fig.text(0.5, top - fig_h - 0.02, caption,
             ha='center', va='top', fontsize=10, fontstyle='italic',
             family='sans-serif', color='#333')
    if subtitle:
        fig.text(0.08, 0.94, subtitle, ha='left', va='top',
                 fontsize=13, fontweight='bold', family='sans-serif')
    return fig, top - fig_h - 0.05


def add_text_and_figure(pdf, title, body, image_path, caption, body_size=9.5):
    fig = plt.figure(figsize=(8.5, 11))
    # Title
    fig.text(0.08, 0.95, title, ha='left', va='top',
             fontsize=14, fontweight='bold', family='sans-serif')
    # Body text in top portion
    lines = body.strip().split('\n')
    line_height = body_size * 1.6 / 72
    n_lines = len(lines)
    text_height = n_lines * line_height
    text_bottom = 0.95 - 0.03 - text_height
    text_bottom = max(text_bottom, 0.50)

    fig.text(0.08, 0.92, body, ha='left', va='top',
             fontsize=body_size, family='sans-serif', linespacing=1.55,
             wrap=True, transform=fig.transFigure)

    # Figure in bottom portion
    if os.path.exists(image_path):
        img = mpimg.imread(image_path)
        h, w = img.shape[:2]
        aspect = w / h
        fig_w = 0.84
        fig_h = min(fig_w / aspect, 0.38)
        bottom = max(text_bottom - fig_h - 0.02, 0.04)
        ax = fig.add_axes([0.08, bottom, fig_w, fig_h])
        ax.imshow(img)
        ax.axis('off')
        fig.text(0.5, bottom - 0.01, caption, ha='center', va='top',
                 fontsize=9, fontstyle='italic', family='sans-serif', color='#444')

    pdf.savefig(fig)
    plt.close(fig)


# --- Content sections ---

sections = [
    # --- Title page ---
    {
        'type': 'text',
        'title': '',
        'body': '',
        'custom': 'title_page',
    },

    # --- Setup ---
    {
        'type': 'text',
        'title': 'Experiment Setup',
        'body': (
            "17 implementations on 12 execution substrates, one attention computation:\n\n"
            "Real hardware (measured on Google Colab):\n"
            "  •  CPU (NumPy decomposed + PyTorch SDPA) — T4 runtime host, 2 Xeon vCPUs\n"
            "  •  GPU FP32 (decomposed + SDPA) — NVIDIA T4, cuBLAS, float32 inputs\n"
            "  •  GPU FP16 (decomposed + SDPA) — NVIDIA T4, Tensor Cores\n"
            "  •  TPU (five jits + one jit) — Colab TPU via JAX, float32 (6-pass bf16)\n"
            "  CPU and GPU: one session, PyTorch 2.11.0+cu130, NumPy 2.1.3.\n\n"
            "Analytical cost models (uncalibrated; checked against the literature,\n"
            "every one is optimistic, so read them as upper bounds):\n"
            "  •  Dataflow (Maxeler-style) — 4 parallel pipes, 500 MHz\n"
            "  •  Optical (photonic MZI mesh) — 64×64 tiles, electronic softmax\n"
            "  •  IoT Edge (ARM Cortex-M4F) — 168 MHz, 3 cycles per FP32 MAC\n"
            "  •  GaAs RISC-V — 1 GHz, scalar, no cache\n"
            "  •  WSN — 64 nodes, IEEE 802.15.4 (250 kbps)\n"
            "  •  Minimal-ISA machine (cdte_turing) — 4 instructions, 10 MHz\n"
            "  •  Quantum (gate-based) — 1000 qubits, 50 ns gates\n"
            "  •  Chemical (CRN) — 256 wells, 1 kHz reactions\n"
            "  •  Biological (DNA) — strand displacement, 1024 parallel strands\n\n"
            "Sequence lengths 128–2048, embedding dimensions 64–256, batch size one.\n"
            "Every row of the results table records whether it is measured or modelled,\n"
            "and the precision it ran in."
        ),
    },

    # --- 1 + Fig 1 ---
    {
        'type': 'text_and_figure',
        'title': '1. Architecture Spans Many Orders of Magnitude',
        'body': (
            "The same 5.39 GFLOP of attention (n=2048, d=256) runs at 38 GFLOP/s on the\n"
            "CPU (NumPy) and 15,772 GFLOP/s on the TPU with XLA fusion — about 410x,\n"
            "measured. The cost models extend the range by roughly seven more orders\n"
            "of magnitude at the same size, down to 0.0007 GFLOP/s for the minimal\n"
            "four-instruction machine.\n\n"
            "The algorithm, the FLOP count and the result are identical throughout.\n"
            "Hatched bars are models: calculations, not stopwatch readings."
        ),
        'image': 'fig1_unified_comparison.png',
        'caption': 'Figure 1: All implementations at n=2048, d=256 (hatched = modelled)',
    },

    # --- 2 + Fig 7 ---
    {
        'type': 'text_and_figure',
        'title': '2. Softmax Costs Far More Than Its FLOPs',
        'body': (
            "Softmax is 0.39% of the FLOPs at n=2048, d=256. Measured share of runtime:\n"
            "•  CPU (NumPy): 35% — five single-threaded passes; 59% at d=64\n"
            "•  GPU FP16: 20% — memory-bound (121 µs vs 52 µs to stream it once)\n"
            "•  GPU FP32: 13% — memory-bound (173 µs vs 105 µs)\n"
            "•  TPU, five jits: 21% — but every stage takes 157–252 µs: dispatch\n\n"
            "Tensor Cores shrink the matrix products, so softmax's share rises in FP16.\n"
            "In the models the share follows from the assumptions: 99% for the photonic\n"
            "mesh (electronic softmax), 0.5% for the scalar processors."
        ),
        'image': 'fig7_operation_breakdown.png',
        'caption': 'Figure 7: Operation time breakdown — decomposed paradigms (n=2048, d=256)',
    },

    # --- 3 + Fig 3 ---
    {
        'type': 'text_and_figure',
        'title': '3. At Batch Size One, the Software Path Matters as Much as the Device',
        'body': (
            "•  TPU: one jit is 3.2x faster than five separately dispatched ones (4.2x at\n"
            "   n=128, where data movement is negligible) — fewer dispatches and syncs.\n"
            "•  GPU FP32: SDPA's memory-efficient kernel is 1.7x SLOWER than five plain\n"
            "   ops. FlashAttention needs sm_80+, and with batch x heads = 8 the fused\n"
            "   kernel has too little parallel work.\n"
            "•  GPU FP16: SDPA and decomposed are within 3% — a tie, given that\n"
            "   sessions differ by up to ~24%.\n"
            "•  CPU: PyTorch SDPA is 1.9x faster than NumPy, but that compares a\n"
            "   multithreaded kernel with single-threaded NumPy passes.\n\n"
            "The fused TPU path and the best GPU path differ by only 1.7x."
        ),
        'image': 'fig3_fused_vs_decomposed.png',
        'caption': 'Figure 3: Fused vs decomposed attention — measured hardware',
    },

    # --- 4 + Fig 2 ---
    {
        'type': 'text_and_figure',
        'title': '4. Precision — and How Easily It Changes Silently',
        'body': (
            "On the T4, FP16 is 2.3x faster than FP32 decomposed and 4.0x through SDPA —\n"
            "far below the 8x ratio of nominal peaks, because fixed per-operation costs\n"
            "and softmax dominate at these sizes.\n\n"
            "Under NumPy 2, a float32 array times a NumPy float64 scalar is float64. An\n"
            "input generator that scales by 1/sqrt(d) therefore yields float64 data, and\n"
            "every path that does not cast runs in FP64: decomposed attention at 199\n"
            "GFLOP/s instead of 4,002, and SDPA falls to its unfused math path (104),\n"
            "because the fused kernels accept only FP16/FP32. The generator casts to\n"
            "float32, and every row records its precision."
        ),
        'image': 'fig2_gpu_precision_detail.png',
        'caption': 'Figure 2: GPU T4 — FP32 vs FP16, decomposed vs SDPA (float32 inputs)',
    },

    # --- 5 + Fig 8 ---
    {
        'type': 'text_and_figure',
        'title': '5. Accelerators Absorb Quadratic Growth at These Sizes',
        'body': (
            "From n=128 to n=2048 at d=256, FLOPs grow 64.2x. Execution time:\n"
            "•  CPU (NumPy): 81.0x — faster than the work, as the n×n matrix leaves cache\n"
            "•  GPU FP32: 4.4x, GPU FP16: 1.9x, TPU fused: 2.6x\n\n"
            "At batch size one the accelerators are dominated by fixed costs: at n=128\n"
            "each decomposed GPU stage takes 33–98 µs whatever its size. This is a\n"
            "statement about this size range; the algorithm is O(n²d) everywhere."
        ),
        'image': 'fig8_time_scaling.png',
        'caption': 'Figure 8: Execution time scaling — key paradigms (d=256)',
    },

    # --- 6 + Fig 6 ---
    {
        'type': 'text_and_figure',
        'title': '6. Larger Embeddings Use the Hardware Better',
        'body': (
            "At n=2048, raising d from 64 to 256:\n"
            "•  GPU FP16: 2,510 → 9,103 GFLOP/s = 3.63x\n"
            "•  TPU fused: 4,855 → 15,772 GFLOP/s = 3.25x\n"
            "•  CPU (NumPy): 14.2 → 38.2 GFLOP/s = 2.70x\n"
            "•  GPU FP32: 1,535 → 4,002 GFLOP/s = 2.61x\n\n"
            "Wider matrices fill matrix units and warp schedulers more completely."
        ),
        'image': 'fig6_embed_dim_effect.png',
        'caption': 'Figure 6: Embedding dimension effect on throughput (n=2048)',
    },

    # --- 7 + Fig 5 ---
    {
        'type': 'text_and_figure',
        'title': '7. Unconventional Substrates Meet a Dense-Matrix Wall (Models)',
        'body': (
            "All cost models, all optimistic against the literature:\n"
            "•  Optical: 510 GFLOP/s — optical products take 66 µs; the scalar 2 GHz\n"
            "   electronic softmax takes 10.5 ms (OEO only 20 µs). Mesh reprogramming for\n"
            "   K and V is not modelled.\n"
            "•  Quantum: 6.4e-4 GFLOP/s (n=32) — 50 ns is a physical gate time;\n"
            "   error-corrected operations take microseconds.\n"
            "•  Chemical: 2.3e-4 GFLOP/s (n=32) — programmable CRNs take seconds to hours\n"
            "   per reaction, not 1 ms; exp/softmax is not rate-independent.\n"
            "•  DNA: 1.2e-8 GFLOP/s (n=32) — real cascades take ~2 h per layer, not 10 s.\n"
            "•  WSN: 0.013 GFLOP/s — 97% of the time is radio communication.\n"
            "•  Cortex-M4F: 0.11 GFLOP/s — an issue bound; loads not counted."
        ),
        'image': 'fig5_throughput_heatmap.png',
        'caption': 'Figure 5: Throughput heatmap — paradigm × sequence length (d=256)',
    },

    # --- Key numbers ---
    {
        'type': 'text',
        'title': 'Key Numbers for Quick Reference',
        'body': (
            "At n=2048, d=256 (5.39 GFLOP of work):\n\n"
            "Implementation               Precision      GFLOP/s     Source\n"
            "─" * 72 + "\n"
            "TPU fused (one jit)          FP32 6-pass   15,771.71    measured\n"
            "GPU fused SDPA               FP16           9,368.01    measured\n"
            "GPU decomposed               FP16           9,102.71    measured\n"
            "TPU decomposed (five jits)   FP32 6-pass    4,908.19    measured\n"
            "GPU decomposed               FP32           4,001.98    measured\n"
            "GPU fused SDPA               FP32           2,354.38    measured\n"
            "Dataflow (Maxeler-style)     –                977.43    cycle model\n"
            "Optical (photonic MZI)       –                509.82    latency model\n"
            "CPU fused SDPA (PyTorch)     FP32              71.24    measured\n"
            "CPU decomposed (NumPy)       FP32              38.24    measured\n"
            "GaAs RISC-V                  –                  0.15    instruction model\n"
            "IoT edge (ARM M4F)           –                  0.11    MCU cycle model\n"
            "WSN (64 nodes)               –                 0.013    network model\n"
            "Minimal ISA (cdte_turing)    –                0.0007    operation model\n\n"
            "At n=32, d=16 (capped):\n"
            "Quantum 0.00064 | Chemical 0.00023 | Biological 1.2e-8 GFLOP/s (models)"
        ),
        'font_size': 8.5,
    },
]


# --- Generate PDF ---

with PdfPages(pdf_path) as pdf:
    for section in sections:
        if section.get('custom') == 'title_page':
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.68,
                     'Analysis of Attention Mechanisms\nin the Context of Computational Paradigms',
                     ha='center', va='center', fontsize=22, fontweight='bold',
                     family='sans-serif', linespacing=1.5)
            fig.text(0.5, 0.52,
                     'Experiment Report \u2014 Findings & Figures',
                     ha='center', va='center', fontsize=16, color='#555',
                     family='sans-serif')
            fig.text(0.5, 0.38,
                     'Uro\u0161 Savurdi\u0107\n'
                     'School of Electrical Engineering, University of Belgrade\n\n'
                     'SICAAI 2026 \u2014 Kragujevac, May 20\u201321, 2026',
                     ha='center', va='center', fontsize=12, color='#777',
                     family='sans-serif', linespacing=1.6)
            fig.text(0.5, 0.18,
                     '12 substrates, 17 implementations | Measured: CPU, NVIDIA T4, TPU (Colab)\n'
                     'FP32 + FP16 | Modelled: dataflow, optical, GaAs, minimal ISA,\n'
                     'IoT edge, WSN, quantum, chemical, biological\n'
                     '7 findings | 8 figures',
                     ha='center', va='center', fontsize=10, color='#999',
                     family='sans-serif', linespacing=1.5)
            pdf.savefig(fig)
            plt.close(fig)

        elif section['type'] == 'text':
            add_text_page(pdf, section['title'], section['body'],
                          font_size=section.get('font_size', 10))

        elif section['type'] == 'text_and_figure':
            img_path = os.path.join(RESULTS, section['image'])
            add_text_and_figure(pdf, section['title'], section['body'],
                                img_path, section['caption'])

print(f'PDF report saved: {pdf_path}')
