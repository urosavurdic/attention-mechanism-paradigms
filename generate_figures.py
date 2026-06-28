import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.family': 'sans-serif',
})

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

df = pd.read_csv(RESULTS + '/combined_results.csv')

# The canonical table renames two columns and adds provenance. Alias the old
# names so the plotting code below reads the same way it always did.
df['throughput_gflops'] = df['throughput_gflop_s']
df['info_sdpa_backend_used'] = df['sdpa_backend']

# Hatching marks a modelled result. Drive it from the data rather than a
# hardcoded list, so a figure can never disagree with the table it came from.
_real = df['is_real_hardware'].astype(str).str.strip().str.lower() == 'true'
IS_MODELLED = dict(zip(df['paradigm_short'], ~_real))


def hatch_for(short):
    return '//' if IS_MODELLED.get(short, False) else ''


maxseq = df['seq_len'].max()
maxdim = df['embed_dim'].max()
sub = df[(df['seq_len'] == maxseq) & (df['embed_dim'] == maxdim)]

# FIGURE 1: Full cross-platform bar chart
paradigm_config = [
    ('tpu_fused',       'TPU Fused (JAX)',              '#E91E63', ''),
    ('gpu_fp16',        'GPU FP16 Decomposed (T4)',     '#8BC34A', ''),
    ('gpu_fused_fp16',  'GPU FP16 Fused SDPA (T4)',     '#CDDC39', ''),
    ('tpu_systolic',    'TPU Decomposed (JAX)',         '#9C27B0', ''),
    ('dataflow',        'Dataflow (Maxeler sim)',        '#00BCD4', '//'),
    ('optical',         'Optical (Photonic MZI sim)',    '#009688', '//'),
    ('gpu',             'GPU FP32 Decomposed (T4)',     '#4CAF50', ''),
    ('gpu_fused',       'GPU FP32 Fused SDPA (T4)',     '#FF9800', ''),
    ('cpu',             'CPU Decomposed',                '#2196F3', ''),
    ('cpu_fused',       'CPU Fused SDPA',                '#03A9F4', ''),
    ('iot_edge',        'IoT Edge (ARM M4F sim)',        '#FFC107', '//'),
    ('wsn',             'WSN (Sensor Network sim)',      '#8D6E63', '//'),
    ('gaas_riscv',      'GaAs RISC-V (sim)',             '#795548', '//'),
    ('chemical',        'Chemical (CRN sim)',             '#26A69A', '//'),
    ('quantum',         'Quantum (Gate-based sim)',       '#7E57C2', '//'),
    ('cdte_turing',     'CdTe Turing (sim)',             '#616161', '//'),
    ('biological',      'Biological (DNA sim)',           '#D81B60', '//'),
]

fig1, ax1 = plt.subplots(figsize=(14, 10))
names, vals, colors, hatches_list, annotations = [], [], [], [], []

for short, label, color, hatch in paradigm_config:
    row = sub[sub['paradigm_short'] == short]
    if not row.empty:
        names.append(label)
        v = row['throughput_gflops'].values[0]
        vals.append(v)
        colors.append(color)
        hatches_list.append(hatch_for(short))
        backend = row['info_sdpa_backend_used'].values[0] if 'info_sdpa_backend_used' in row.columns else None
        ann = ''
        if pd.notna(backend) and 'math' in str(backend).lower():
            ann = 'math backend (no Flash/MemEfficient)'
        elif pd.notna(backend):
            ann = str(backend) + ' backend'
        annotations.append(ann)

bars = ax1.barh(range(len(names)), vals, color=colors, edgecolor='white', linewidth=1.5, height=0.7)
for bar, h in zip(bars, hatches_list):
    bar.set_hatch(h)

ax1.set_yticks(range(len(names)))
ax1.set_yticklabels(names, fontsize=10)
ax1.set_xlabel('Throughput (GFLOP/s)', fontsize=12)
ax1.set_xscale('log')
ax1.set_title(f'Unified Paradigm Comparison - All Architectures\n(n={maxseq}, d={maxdim} | hatched = modelled, solid = measured)',
              fontsize=13, fontweight='bold')
ax1.invert_yaxis()

for i, (bar, val, ann) in enumerate(zip(bars, vals, annotations)):
    label_text = f'{val:,.0f}' if val >= 1 else f'{val:.4f}'
    ax1.text(val * 1.4, bar.get_y() + bar.get_height()/2,
             label_text, va='center', fontsize=9, fontweight='bold')
    if ann:
        ax1.annotate(ann, xy=(val * 0.3, bar.get_y() + bar.get_height()/2),
                     fontsize=7, color='white', fontweight='bold', va='center', ha='center')

fig1.tight_layout()
fig1.savefig(f'{RESULTS}/fig1_unified_comparison.png', dpi=150, bbox_inches='tight')
print('Saved fig1_unified_comparison.png')
plt.close(fig1)


# FIGURE 2: GPU detail panel
gpu_sub = sub[sub['paradigm_short'].isin(['gpu', 'gpu_fused', 'gpu_fp16', 'gpu_fused_fp16'])]
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5.5))

x = np.arange(2)
width = 0.35
fp32_vals = [
    gpu_sub[gpu_sub['paradigm_short']=='gpu']['throughput_gflops'].values[0],
    gpu_sub[gpu_sub['paradigm_short']=='gpu_fused']['throughput_gflops'].values[0],
]
fp16_vals = [
    gpu_sub[gpu_sub['paradigm_short']=='gpu_fp16']['throughput_gflops'].values[0],
    gpu_sub[gpu_sub['paradigm_short']=='gpu_fused_fp16']['throughput_gflops'].values[0],
]

b1 = ax2a.bar(x - width/2, fp32_vals, width, label='FP32', color=['#4CAF50', '#FF9800'])
b2 = ax2a.bar(x + width/2, fp16_vals, width, label='FP16', color=['#8BC34A', '#CDDC39'])
ax2a.set_xticks(x)
ax2a.set_xticklabels(['Decomposed', 'Fused SDPA'])
ax2a.set_ylabel('Throughput (GFLOP/s)')
ax2a.set_yscale('log')
ax2a.set_title(f'GPU T4: FP32 vs FP16\n(n={maxseq}, d={maxdim})', fontweight='bold')

for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax2a.annotate(f'{h:,.0f}', xy=(bar.get_x() + bar.get_width()/2, h),
                  xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')

for i in range(2):
    ratio = fp16_vals[i] / fp32_vals[i]
    mid_y = (fp32_vals[i] * fp16_vals[i]) ** 0.5
    ax2a.annotate(f'{ratio:.0f}x', xy=(i, mid_y), fontsize=11, fontweight='bold',
                  color='#D32F2F', ha='center')
ax2a.legend(fontsize=10)

labels_b = ['FP32 Decomposed\n(cuBLAS)', 'FP32 Fused\n(math fallback)',
            'FP16 Decomposed\n(Tensor Cores)', 'FP16 Fused\n(mem_efficient)']
values_b = fp32_vals + fp16_vals
bar_colors_b = ['#4CAF50', '#FF9800', '#8BC34A', '#CDDC39']
bars_b = ax2b.barh(range(4), values_b, color=bar_colors_b, height=0.6, edgecolor='white', linewidth=1.5)
ax2b.set_yticks(range(4))
ax2b.set_yticklabels(labels_b, fontsize=9)
ax2b.set_xlabel('Throughput (GFLOP/s)')
ax2b.set_xscale('log')
ax2b.set_title(f'GPU T4: Backend Dispatch Impact\n(sm_75 Turing, n={maxseq}, d={maxdim})', fontweight='bold')
ax2b.invert_yaxis()
for bar, val in zip(bars_b, values_b):
    ax2b.text(val * 1.3, bar.get_y() + bar.get_height()/2,
              f'{val:,.0f}', va='center', fontsize=9, fontweight='bold')

fig2.tight_layout()
fig2.savefig(f'{RESULTS}/fig2_gpu_precision_detail.png', dpi=150, bbox_inches='tight')
print('Saved fig2_gpu_precision_detail.png')
plt.close(fig2)


# FIGURE 3: Fused vs Decomposed across all real HW
fig3, ax3 = plt.subplots(figsize=(11, 6))
pairs = [
    ('cpu', 'cpu_fused', 'CPU\n(Colab)'),
    ('gpu', 'gpu_fused', 'GPU T4\nFP32'),
    ('gpu_fp16', 'gpu_fused_fp16', 'GPU T4\nFP16'),
    ('tpu_systolic', 'tpu_fused', 'TPU\n(JAX)'),
]
xp = np.arange(len(pairs))
width = 0.35
decomp_v, fused_v, labels_hw = [], [], []
for decomp, fused, label in pairs:
    d_row = sub[sub['paradigm_short'] == decomp]
    f_row = sub[sub['paradigm_short'] == fused]
    decomp_v.append(d_row['throughput_gflops'].values[0] if not d_row.empty else 0)
    fused_v.append(f_row['throughput_gflops'].values[0] if not f_row.empty else 0)
    labels_hw.append(label)

bars1 = ax3.bar(xp - width/2, decomp_v, width, label='Decomposed', color='#FF7043')
bars2 = ax3.bar(xp + width/2, fused_v, width, label='Fused', color='#42A5F5')
ax3.set_xticks(xp)
ax3.set_xticklabels(labels_hw)
ax3.set_ylabel('Throughput (GFLOP/s)')
ax3.set_yscale('log')
ax3.set_title(f'Decomposed vs Fused Attention - Real Hardware\n(n={maxseq}, d={maxdim})',
              fontsize=13, fontweight='bold')
ax3.legend(fontsize=11)

for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    ax3.annotate(f'{h:,.0f}', xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 4), textcoords='offset points', ha='center', fontsize=8, fontweight='bold')

for i, (d, f) in enumerate(zip(decomp_v, fused_v)):
    if d > 0 and f > 0:
        ratio = max(f/d, d/f)
        winner = 'Fused' if f > d else 'Decomp'
        ax3.text(i, max(d, f) * 2.5, f'{winner}\n{ratio:.1f}x',
                 ha='center', fontsize=9, fontweight='bold', color='#333')

fig3.tight_layout()
fig3.savefig(f'{RESULTS}/fig3_fused_vs_decomposed.png', dpi=150, bbox_inches='tight')
print('Saved fig3_fused_vs_decomposed.png')
plt.close(fig3)


# FIGURE 4: Scaling curves
paradigm_style = [
    ('tpu_fused',       'TPU Fused',        '#E91E63', 's', '-'),
    ('gpu_fp16',        'GPU FP16 Decomp',  '#8BC34A', 'o', '-'),
    ('gpu_fused_fp16',  'GPU FP16 Fused',   '#CDDC39', 'v', '-'),
    ('tpu_systolic',    'TPU Decomposed',   '#9C27B0', 'D', '-'),
    ('dataflow',        'Dataflow (sim)',    '#00BCD4', '^', '--'),
    ('optical',         'Optical (sim)',     '#009688', 'p', '--'),
    ('gpu',             'GPU FP32 Decomp',  '#4CAF50', 'o', '-'),
    ('gpu_fused',       'GPU FP32 Fused',   '#FF9800', 'v', '-'),
    ('cpu',             'CPU Decomposed',   '#2196F3', 'o', '-'),
    ('cpu_fused',       'CPU Fused',        '#03A9F4', 'v', '-'),
    ('iot_edge',        'IoT Edge (sim)',    '#FFC107', 'h', '--'),
    ('wsn',             'WSN (sim)',         '#8D6E63', '*', '--'),
    ('gaas_riscv',      'GaAs RISC-V (sim)','#795548', 'P', '--'),
    ('chemical',        'Chemical (sim)',    '#26A69A', 'd', '--'),
    ('quantum',         'Quantum (sim)',     '#7E57C2', '8', '--'),
    ('cdte_turing',     'CdTe Turing (sim)','#616161', 'X', '--'),
    ('biological',      'Biological (sim)', '#D81B60', '+', '--'),
]

fig4, ax4 = plt.subplots(figsize=(12, 7))
sub4 = df[df['embed_dim'] == maxdim]
for short, label, color, marker, ls in paradigm_style:
    pdata = sub4[sub4['paradigm_short'] == short].sort_values('seq_len')
    if pdata.empty:
        continue
    ax4.plot(pdata['seq_len'], pdata['throughput_gflops'],
             marker=marker, linestyle=ls, color=color, label=label,
             linewidth=2, markersize=7)

ax4.set_xlabel('Sequence Length', fontsize=12)
ax4.set_ylabel('Throughput (GFLOP/s)', fontsize=12)
ax4.set_title(f'Throughput Scaling - All Paradigms (d={maxdim})', fontsize=13, fontweight='bold')
ax4.set_yscale('log')
ax4.set_xscale('log', base=2)
ax4.set_xticks(sorted(sub4['seq_len'].unique()))
ax4.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax4.legend(loc='upper left', fontsize=8, ncol=2)
fig4.tight_layout()
fig4.savefig(f'{RESULTS}/fig4_scaling_curves.png', dpi=150, bbox_inches='tight')
print('Saved fig4_scaling_curves.png')
plt.close(fig4)

# (previously rewrote a raw export here; the canonical table is built by build_results_table.py)


# FIGURE 5: Throughput heatmap (paradigm × seq_len, d=256)
heat_paradigms = [
    ('tpu_fused',    'TPU Fused',    '#E91E63'),
    ('gpu_fp16',     'GPU FP16',     '#8BC34A'),
    ('gpu_fused_fp16','GPU FP16 Fused','#CDDC39'),
    ('tpu_systolic', 'TPU Decomp',   '#9C27B0'),
    ('dataflow',     'Dataflow',     '#00BCD4'),
    ('optical',      'Optical',      '#009688'),
    ('gpu',          'GPU FP32',     '#4CAF50'),
    ('gpu_fused',    'GPU FP32 Fused','#FF9800'),
    ('cpu',          'CPU',          '#2196F3'),
    ('iot_edge',     'IoT Edge',     '#FFC107'),
    ('wsn',          'WSN',          '#8D6E63'),
    ('gaas_riscv',   'GaAs RISC-V',  '#795548'),
    ('chemical',     'Chemical',     '#26A69A'),
    ('quantum',      'Quantum',      '#7E57C2'),
    ('cdte_turing',  'CdTe Turing',  '#616161'),
    ('biological',   'Biological',   '#D81B60'),
]
seq_lens = sorted(df['seq_len'].unique())
heat_data = []
heat_labels = []
for short, label, _ in heat_paradigms:
    row_vals = []
    for sl in seq_lens:
        v = df[(df['paradigm_short']==short) & (df['seq_len']==sl) & (df['embed_dim']==maxdim)]
        row_vals.append(v['throughput_gflops'].values[0] if not v.empty else 0)
    heat_data.append(row_vals)
    heat_labels.append(label)

heat_arr = np.array(heat_data)
fig5, ax5 = plt.subplots(figsize=(14, 9))
im = ax5.imshow(np.log10(np.clip(heat_arr, 0.001, None)), cmap='RdYlBu_r', aspect='auto')
ax5.set_xticks(range(len(seq_lens)))
ax5.set_xticklabels(seq_lens)
ax5.set_yticks(range(len(heat_labels)))
ax5.set_yticklabels(heat_labels, fontsize=10)
ax5.set_xlabel('Sequence Length', fontsize=12)
ax5.set_title(f'Throughput Heatmap (GFLOP/s, d={maxdim}) — Log Scale', fontsize=13, fontweight='bold')
for i in range(len(heat_labels)):
    for j in range(len(seq_lens)):
        v = heat_arr[i, j]
        txt = f'{v:,.0f}' if v >= 1 else f'{v:.3f}'
        ax5.text(j, i, txt, ha='center', va='center', fontsize=8,
                 color='white' if np.log10(max(v,0.001)) > 2.5 else 'black', fontweight='bold')
cbar = fig5.colorbar(im, ax=ax5, shrink=0.8)
cbar.set_label('log₁₀(GFLOP/s)', fontsize=10)
fig5.tight_layout()
fig5.savefig(f'{RESULTS}/fig5_throughput_heatmap.png', dpi=150, bbox_inches='tight')
print('Saved fig5_throughput_heatmap.png')
plt.close(fig5)


# FIGURE 6: Embedding dimension effect (grouped bars, n=2048)
embed_paradigms = [
    ('tpu_fused',    'TPU Fused',   '#E91E63'),
    ('gpu_fp16',     'GPU FP16',    '#8BC34A'),
    ('tpu_systolic', 'TPU Decomp',  '#9C27B0'),
    ('gpu',          'GPU FP32',    '#4CAF50'),
    ('optical',      'Optical',     '#009688'),
    ('cpu',          'CPU',         '#2196F3'),
    ('iot_edge',     'IoT Edge',    '#FFC107'),
    ('wsn',          'WSN',         '#8D6E63'),
]
embed_dims = sorted(df['embed_dim'].unique())
fig6, ax6 = plt.subplots(figsize=(14, 7))
x6 = np.arange(len(embed_paradigms))
w6 = 0.25
for idx, ed in enumerate(embed_dims):
    vals = []
    for short, _, _ in embed_paradigms:
        row = df[(df['paradigm_short']==short) & (df['seq_len']==maxseq) & (df['embed_dim']==ed)]
        vals.append(row['throughput_gflops'].values[0] if not row.empty else 0)
    bars6 = ax6.bar(x6 + (idx - 1) * w6, vals, w6,
                    label=f'd={ed}', alpha=min(0.5 + 0.15*idx, 1.0),
                    color=[c for _, _, c in embed_paradigms], edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars6, vals):
        if val > 0:
            ax6.annotate(f'{val:,.0f}', xy=(bar.get_x() + bar.get_width()/2, val),
                         xytext=(0, 3), textcoords='offset points', ha='center', fontsize=7, fontweight='bold')
ax6.set_xticks(x6)
ax6.set_xticklabels([l for _, l, _ in embed_paradigms])
ax6.set_ylabel('Throughput (GFLOP/s)')
ax6.set_yscale('log')
ax6.set_title(f'Embedding Dimension Effect (n={maxseq})', fontsize=13, fontweight='bold')
ax6.legend(fontsize=10)
fig6.tight_layout()
fig6.savefig(f'{RESULTS}/fig6_embed_dim_effect.png', dpi=150, bbox_inches='tight')
print('Saved fig6_embed_dim_effect.png')
plt.close(fig6)


# FIGURE 7: Operation breakdown stacked bars (decomposed paradigms, n=2048, d=256)
op_cols = ['time_qkv_projection', 'time_qk_matmul', 'time_softmax', 'time_av_matmul', 'time_output_projection']
op_labels = ['QKV Projection', 'QK^T Matmul', 'Softmax', 'AV Matmul', 'Output Projection']
op_colors = ['#42A5F5', '#FF9800', '#EF5350', '#66BB6A', '#AB47BC']
decomp_paradigms = [
    ('cpu',          'CPU'),
    ('gpu',          'GPU FP32'),
    ('gpu_fp16',     'GPU FP16'),
    ('tpu_systolic', 'TPU Systolic'),
    ('dataflow',     'Dataflow'),
    ('optical',      'Optical'),
    ('iot_edge',     'IoT Edge'),
    ('wsn',          'WSN'),
    ('gaas_riscv',   'GaAs RISC-V'),
    ('quantum',      'Quantum'),
    ('chemical',     'Chemical'),
    ('cdte_turing',  'CdTe Turing'),
    ('biological',   'Biological'),
]

fig7, ax7 = plt.subplots(figsize=(14, 8))
y7 = np.arange(len(decomp_paradigms))
lefts = np.zeros(len(decomp_paradigms))
for col_idx, (col, label, color) in enumerate(zip(op_cols, op_labels, op_colors)):
    pcts = []
    for short, _ in decomp_paradigms:
        row = sub[sub['paradigm_short'] == short]
        if not row.empty and col in row.columns:
            total = sum(row[c].values[0] for c in op_cols if c in row.columns and pd.notna(row[c].values[0]))
            val = row[col].values[0] if pd.notna(row[col].values[0]) else 0
            pcts.append(val / total * 100 if total > 0 else 0)
        else:
            pcts.append(0)
    pcts = np.array(pcts)
    ax7.barh(y7, pcts, left=lefts, height=0.6, label=label, color=color, edgecolor='white', linewidth=0.5)
    for i, p in enumerate(pcts):
        if p > 5:
            ax7.text(lefts[i] + p/2, i, f'{p:.0f}%', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    lefts += pcts

ax7.set_yticks(y7)
ax7.set_yticklabels([l for _, l in decomp_paradigms], fontsize=10)
ax7.set_xlabel('% of Total Execution Time', fontsize=12)
ax7.set_title(f'Operation Breakdown — Decomposed Paradigms (n={maxseq}, d={maxdim})', fontsize=13, fontweight='bold')
ax7.legend(loc='lower right', fontsize=9)
ax7.set_xlim(0, 100)
ax7.invert_yaxis()
fig7.tight_layout()
fig7.savefig(f'{RESULTS}/fig7_operation_breakdown.png', dpi=150, bbox_inches='tight')
print('Saved fig7_operation_breakdown.png')
plt.close(fig7)


# FIGURE 8: Execution time scaling (d=256)
time_paradigms = [
    ('tpu_fused',    'TPU Fused',   '#E91E63', 's', '-'),
    ('gpu_fp16',     'GPU FP16',    '#8BC34A', 'o', '-'),
    ('tpu_systolic', 'TPU Decomp',  '#9C27B0', 'D', '-'),
    ('gpu',          'GPU FP32',    '#4CAF50', 'o', '-'),
    ('cpu',          'CPU',         '#2196F3', 'o', '-'),
    ('dataflow',     'Dataflow',    '#00BCD4', '^', '--'),
    ('optical',      'Optical',     '#009688', 'p', '--'),
    ('iot_edge',     'IoT Edge',    '#FFC107', 'h', '--'),
    ('wsn',          'WSN',         '#8D6E63', '*', '--'),
    ('gaas_riscv',   'GaAs RISC-V', '#795548', 'P', '--'),
    ('quantum',      'Quantum',     '#7E57C2', '8', '--'),
    ('chemical',     'Chemical',    '#26A69A', 'd', '--'),
    ('cdte_turing',  'CdTe Turing', '#616161', 'X', '--'),
    ('biological',   'Biological',  '#D81B60', '+', '--'),
]
fig8, ax8 = plt.subplots(figsize=(12, 7))
sub8 = df[df['embed_dim'] == maxdim]
for short, label, color, marker, ls in time_paradigms:
    pdata = sub8[sub8['paradigm_short'] == short].sort_values('seq_len')
    if pdata.empty:
        continue
    ax8.plot(pdata['seq_len'], pdata['total_effective_time_s'],
             marker=marker, linestyle=ls, color=color, label=label,
             linewidth=2, markersize=7)

ax8.set_xlabel('Sequence Length', fontsize=12)
ax8.set_ylabel('Execution Time (seconds)', fontsize=12)
ax8.set_title(f'Execution Time Scaling — d={maxdim}', fontsize=13, fontweight='bold')
ax8.set_yscale('log')
ax8.set_xscale('log', base=2)
ax8.set_xticks(sorted(sub8['seq_len'].unique()))
ax8.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax8.legend(loc='upper left', fontsize=9)
fig8.tight_layout()
fig8.savefig(f'{RESULTS}/fig8_time_scaling.png', dpi=150, bbox_inches='tight')
print('Saved fig8_time_scaling.png')
plt.close(fig8)

print('Done - 8 figures generated')
