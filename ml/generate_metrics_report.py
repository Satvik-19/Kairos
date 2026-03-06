"""
Generates ml/KAIROS_ML_Metrics.png — a publication-ready metrics card
showing only verified, factual numbers from the three ML model training runs.
Light color scheme matching the Kairos frontend palette.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Verified training numbers (hardcoded from actual run output) ──────────────

# Model 1 – Hybrid Autoencoder + Pendulum Motion Check (60 epochs)
# Input dim: 780 = window(60) x features(13)
AE_EPOCHS     = [0, 10, 20, 30, 40, 50]
AE_TRAIN_LOSS = [0.760322, 0.467747, 0.377178, 0.349338, 0.330577, 0.297792]
AE_VAL_LOSS   = [0.548636, 0.360701, 0.276331, 0.261066, 0.249660, 0.226858]
AE_BEST_VAL   = 0.218183
AE_THRESHOLD  = 0.626959

# Per-mode detection rates (hybrid: autoencoder OR motion check)
DET_MODES  = ['frozen\npendulum', 'lorenz\nrunaway', 'pool\nstarvation', 'rd\nuniform']
DET_RATES  = [100.0, 100.0, 100.0, 7.2]
DET_OVERALL = 76.8

# Model 2 – LSTM Predictor (40 epochs)
LSTM_EPOCHS   = [0, 10, 20, 30]
LSTM_TRAIN    = [0.214227, 0.084653, 0.084458, 0.084436]
LSTM_VAL      = [0.148295, 0.090053, 0.085939, 0.084913]
LSTM_BEST_VAL = 0.084799
LSTM_NAIVE    = 0.084380
LSTM_CEIL     = 1 / 12        # ~0.0833 = Var(Uniform[0,1])
LSTM_RES_MEAN = 0.08479934
LSTM_RES_STD  = 0.01377534

# Model 4 – MLP Classifier (8 features: entropy x4 + lorenz_z/r + omega_std x2)
CLS_CLASSES  = ['critical', 'degraded', 'excellent', 'good']
CLS_PRECISION = [1.00, 0.87, 0.83, 1.00]
CLS_RECALL    = [1.00, 0.71, 0.93, 1.00]
CLS_F1        = [1.00, 0.78, 0.88, 1.00]
CLS_SUPPORT   = [1924, 1925, 2889, 1000]
CLS_CV_ACC    = 0.786
CLS_CV_STD    = 0.127
CLS_TRAIN_ACC = 0.90

# Dataset
DATA_SPLITS = [
    'normal\n(excellent)', 'degraded\n(frozen /\nrd_uniform)',
    'critical\n(lorenz /\npool starv.)', 'sequence\npairs (LSTM)',
]
DATA_COUNTS  = [2889, 1925, 1924, 4813]
EDA_OVERLAP  = 70.8
EDA_AUTOCORR = 0.9862
EDA_CHISQ    = 287.24

# ── Color palette — Kairos frontend (warm light theme) ───────────────────────
BG     = '#F4F2EC'   # warm cream body background
PANEL  = '#FFFFFF'   # white cards
HEADER = '#EBE8E0'   # section header / divider tone
BORDER = '#D8D4CC'   # borders
INK    = '#1A1A1A'   # primary text (body/dark header)
DARK   = '#374151'   # secondary text
MUTED  = '#6B7280'   # captions / labels
FAINT  = '#A3A3A3'   # extra-muted

AMBER  = '#F59E0B'   # primary frontend accent
AMBER2 = '#D97706'   # darker amber
TEAL   = '#14B8A6'   # second frontend accent
TEAL2  = '#0D9488'   # darker teal
GREEN  = '#16A34A'   # success
GREEN2 = '#22C55E'   # lighter success
RED    = '#EF4444'   # error
ORANGE = '#F97316'   # warning-orange
BLUE   = '#3B82F6'   # neutral blue (LSTM)
PURPLE = '#8B5CF6'   # purple (LSTM train)

plt.rcParams.update({
    'figure.facecolor':  BG,
    'axes.facecolor':    PANEL,
    'axes.edgecolor':    BORDER,
    'axes.labelcolor':   MUTED,
    'xtick.color':       MUTED,
    'ytick.color':       MUTED,
    'text.color':        INK,
    'grid.color':        HEADER,
    'grid.alpha':        1.0,
    'font.family':       'monospace',
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

fig = plt.figure(figsize=(22, 26))
fig.patch.set_facecolor(BG)

# ── Master GridSpec ───────────────────────────────────────────────────────────
outer = gridspec.GridSpec(
    5, 1,
    figure=fig,
    hspace=0.44,
    top=0.945, bottom=0.038,
    left=0.06, right=0.97,
    height_ratios=[0.85, 2.1, 2.1, 2.1, 2.05],
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def panel_title(ax, text, colour=INK):
    ax.set_title(text, color=colour, fontsize=11.5, fontweight='bold',
                 pad=11, loc='left')

def _style_axes(ax):
    ax.yaxis.grid(True, zorder=0, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_edgecolor(BORDER)
    ax.spines['bottom'].set_edgecolor(BORDER)

def _info_box(ax, lines, x=0.97, y=0.97):
    """Small annotation inset, right-aligned."""
    text = '\n'.join(lines)
    ax.text(x, y, text, transform=ax.transAxes,
            ha='right', va='top', fontsize=8, color=MUTED,
            bbox=dict(boxstyle='round,pad=0.45', facecolor=HEADER,
                      edgecolor=BORDER, linewidth=0.8))

# ═════════════════════════════════════════════════════════════════════════════
# ROW 0 — Title header  (dark bar, matches App.tsx dark header)
# ═════════════════════════════════════════════════════════════════════════════
ax_title = fig.add_subplot(outer[0])
ax_title.set_axis_off()

hdr = FancyBboxPatch((0, 0), 1, 1, boxstyle='square,pad=0',
                     facecolor=INK, edgecolor='none',
                     transform=ax_title.transAxes, clip_on=False)
ax_title.add_patch(hdr)

ax_title.text(0.5, 0.73, 'KAIROS Entropy Engine',
              fontsize=27, fontweight='bold', color='#FFFFFF',
              ha='center', va='center', transform=ax_title.transAxes)
ax_title.text(0.5, 0.28, 'ML Pipeline  ·  Model Training & Evaluation Metrics',
              fontsize=13, color=FAINT,
              ha='center', va='center', transform=ax_title.transAxes)

# Left / right metadata chips
for x, txt, col in [
    (0.017, 'Anomaly · Predictor · Classifier', AMBER),
    (0.983, 'PyTorch · scikit-learn · Python 3.10', TEAL),
]:
    ax_title.text(x, 0.73, txt, fontsize=8.5, color=col,
                  ha='left' if x < 0.5 else 'right', va='center',
                  transform=ax_title.transAxes)

ax_title.text(0.983, 0.28,
              'All values verified from live training runs',
              fontsize=7.5, color='#555555',
              ha='right', va='center', transform=ax_title.transAxes)

# ═════════════════════════════════════════════════════════════════════════════
# ROW 1 — Dataset composition + EDA highlights
# ═════════════════════════════════════════════════════════════════════════════
row1 = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer[1], wspace=0.34, width_ratios=[1.6, 1])

ax_data = fig.add_subplot(row1[0])
ax_eda  = fig.add_subplot(row1[1])

# ── Dataset bar chart ─────────────────────────────────────────────────────────
colors_data = [GREEN2, AMBER, RED, TEAL]
bars = ax_data.bar(range(4), DATA_COUNTS, color=colors_data,
                   width=0.52, edgecolor=BORDER, linewidth=0.7, zorder=3)
for bar, cnt in zip(bars, DATA_COUNTS):
    ax_data.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 65,
                 f'{cnt:,}', ha='center', va='bottom', fontsize=11,
                 color=INK, fontweight='bold')

ax_data.set_xticks(range(4))
ax_data.set_xticklabels(DATA_SPLITS, fontsize=8.5, color=MUTED)
ax_data.set_ylabel('Sample Count', fontsize=9)
ax_data.set_ylim(0, 6400)
_style_axes(ax_data)
panel_title(ax_data, 'Training Dataset Composition')
ax_data.tick_params(axis='y', labelsize=8.5)

ax_data.text(0.97, 0.97,
             f'Total labelled: {sum(DATA_COUNTS[:3]):,} samples',
             transform=ax_data.transAxes, ha='right', va='top',
             fontsize=8.5, color=MUTED,
             bbox=dict(boxstyle='round,pad=0.4', facecolor=HEADER,
                       edgecolor=BORDER, linewidth=0.8))

# ── EDA highlights panel ──────────────────────────────────────────────────────
ax_eda.set_axis_off()
ax_eda.set_facecolor(PANEL)
panel_title(ax_eda, 'EDA Highlights')

eda_rows = [
    ('Class overlap  (normal 2σ band)', f'{EDA_OVERLAP:.1f}%',   AMBER,
     'Degraded output almost identical\nto normal — state features needed'),
    ('Entropy autocorrelation (lag-1)', f'+{EDA_AUTOCORR:.4f}',  BLUE,
     '5s cache at 62ms sampling rate\nSame value repeats ~80x in a row'),
    ('Hash chi-squared statistic',      f'{EDA_CHISQ:.2f}',      TEAL,
     'Expected ~255 for uniform output\nConfirms hash bytes are random'),
    ('Enriched training rows',          '2,830',                  GREEN,
     'After rolling & delta features\nfor autoencoder window (w=60)'),
]

y0 = 0.88
row_h = 0.235
for (lbl, val, col, note) in eda_rows:
    # left accent bar
    rect = FancyBboxPatch(
        (0.0, y0 - row_h + 0.01), 0.022, row_h - 0.012,
        boxstyle='square,pad=0', facecolor=col, edgecolor='none',
        transform=ax_eda.transAxes, clip_on=False)
    ax_eda.add_patch(rect)
    ax_eda.text(0.055, y0 - 0.015, lbl,
                fontsize=8, color=MUTED, transform=ax_eda.transAxes, va='top')
    ax_eda.text(0.055, y0 - 0.075, val,
                fontsize=16, color=col, fontweight='bold',
                transform=ax_eda.transAxes, va='top')
    ax_eda.text(0.055, y0 - 0.152, note,
                fontsize=7.5, color=FAINT, transform=ax_eda.transAxes,
                va='top', linespacing=1.45)
    # thin horizontal rule
    ax_eda.plot([0.0, 1.0], [y0 - row_h + 0.005, y0 - row_h + 0.005],
                color=HEADER, linewidth=0.8, transform=ax_eda.transAxes,
                clip_on=False, solid_capstyle='butt')
    y0 -= row_h

# ═════════════════════════════════════════════════════════════════════════════
# ROW 2 — Model 1: Training curves  +  Detection-rate breakdown
# ═════════════════════════════════════════════════════════════════════════════
row2 = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer[2], wspace=0.34, width_ratios=[1.85, 1])

ax_ae  = fig.add_subplot(row2[0])
ax_det = fig.add_subplot(row2[1])

# ── Autoencoder loss curves ───────────────────────────────────────────────────
ax_ae.plot(AE_EPOCHS, AE_TRAIN_LOSS, 'o-', color=AMBER, linewidth=2.2,
           markersize=6, label='Train MSE', zorder=5)
ax_ae.plot(AE_EPOCHS, AE_VAL_LOSS,   's--', color=ORANGE, linewidth=2.2,
           markersize=6, label='Val MSE',   zorder=5)
ax_ae.axhline(AE_THRESHOLD, color=RED, linewidth=1.4, linestyle=':',
              label=f'Anomaly threshold (p95) = {AE_THRESHOLD:.4f}', zorder=4)
ax_ae.fill_between(AE_EPOCHS, AE_TRAIN_LOSS, AE_VAL_LOSS,
                   alpha=0.07, color=AMBER)

for ep, tl, vl in zip(AE_EPOCHS, AE_TRAIN_LOSS, AE_VAL_LOSS):
    ax_ae.annotate(f'{tl:.3f}', (ep, tl), textcoords='offset points',
                   xytext=(0, 9), ha='center', fontsize=7, color=AMBER2)
    ax_ae.annotate(f'{vl:.3f}', (ep, vl), textcoords='offset points',
                   xytext=(0, -13), ha='center', fontsize=7, color=ORANGE)

ax_ae.set_xlabel('Epoch', fontsize=9)
ax_ae.set_ylabel('Reconstruction MSE', fontsize=9)
_style_axes(ax_ae)
panel_title(ax_ae, 'Model 1  ·  Hybrid Anomaly Detector  —  Autoencoder Training')
ax_ae.legend(fontsize=8.5, frameon=True, facecolor=PANEL,
             edgecolor=BORDER, labelcolor=INK, loc='upper right')

_info_box(ax_ae, [
    f'Best val loss : {AE_BEST_VAL:.6f}',
    f'Input dim     : 780  (60 x 13 features)',
    f'Sub-detectors : AE reconstruction + motion check',
])

# ── Per-mode detection rate (horizontal bar chart) ────────────────────────────
y_pos    = np.arange(len(DET_MODES))
bar_cols = [GREEN if r >= 99 else RED for r in DET_RATES]

hbars = ax_det.barh(y_pos, DET_RATES, color=bar_cols,
                    height=0.48, edgecolor=BORDER, linewidth=0.6, zorder=3)

# Value labels inside / outside bars
for i, (rate, col) in enumerate(zip(DET_RATES, bar_cols)):
    x_lbl = rate - 3 if rate > 15 else rate + 2
    ha     = 'right' if rate > 15 else 'left'
    t_col  = '#FFFFFF' if rate > 50 else col
    ax_det.text(x_lbl, y_pos[i], f'{rate:.1f}%',
                va='center', ha=ha, fontsize=11,
                color=t_col, fontweight='bold', zorder=6)

# Overall average dashed line
ax_det.axvline(DET_OVERALL, color=AMBER, linewidth=2,
               linestyle='--', zorder=5)
ax_det.text(DET_OVERALL + 1.2, 3.48,
            f'Overall\n{DET_OVERALL:.1f}%',
            va='top', ha='left', fontsize=8.5,
            color=AMBER2, fontweight='bold')

# 70% target
ax_det.axvline(70, color=FAINT, linewidth=1.2, linestyle=':', zorder=4)
ax_det.text(70.8, -0.55, '70% target', va='top',
            fontsize=7.5, color=FAINT)

ax_det.set_yticks(y_pos)
ax_det.set_yticklabels(DET_MODES, fontsize=9.5, color=MUTED)
ax_det.set_xlim(0, 118)
ax_det.set_xlabel('Detection Rate (%)', fontsize=9)
ax_det.xaxis.grid(True, zorder=0, linewidth=0.8)
ax_det.yaxis.grid(False)
ax_det.set_axisbelow(True)
for spine in ('top', 'right'):
    ax_det.spines[spine].set_visible(False)
ax_det.spines['left'].set_edgecolor(BORDER)
ax_det.spines['bottom'].set_edgecolor(BORDER)
panel_title(ax_det, 'Detection Rate  by Failure Mode')

# Threshold annotation box
ax_det.text(0.97, 0.06,
            f'AE threshold: {AE_THRESHOLD:.4f}\nMotion threshold: 0.1261',
            transform=ax_det.transAxes, ha='right', va='bottom',
            fontsize=7.5, color=MUTED,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=HEADER,
                      edgecolor=BORDER, linewidth=0.8))

# ═════════════════════════════════════════════════════════════════════════════
# ROW 3 — Model 2: LSTM curves  +  Prediction-resistance comparison
# ═════════════════════════════════════════════════════════════════════════════
row3 = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer[3], wspace=0.34, width_ratios=[1.85, 1])

ax_lstm = fig.add_subplot(row3[0])
ax_mse  = fig.add_subplot(row3[1])

# ── LSTM training curves ──────────────────────────────────────────────────────
ax_lstm.plot(LSTM_EPOCHS, LSTM_TRAIN, 'o-', color=PURPLE, linewidth=2.2,
             markersize=6, label='Train MSE', zorder=5)
ax_lstm.plot(LSTM_EPOCHS, LSTM_VAL,   's--', color=BLUE,   linewidth=2.2,
             markersize=6, label='Val MSE',   zorder=5)
ax_lstm.axhline(LSTM_NAIVE, color=GREEN, linewidth=1.4, linestyle=':',
                label=f'Naive baseline = {LSTM_NAIVE:.6f}', zorder=4)
ax_lstm.axhline(LSTM_CEIL, color=AMBER, linewidth=1.2, linestyle='-.',
                label=f'Uniform ceiling  1/12 = {LSTM_CEIL:.4f}', zorder=4)
ax_lstm.fill_between(LSTM_EPOCHS, LSTM_TRAIN, LSTM_VAL,
                     alpha=0.06, color=PURPLE)

for ep, tl in zip(LSTM_EPOCHS, LSTM_TRAIN):
    ax_lstm.annotate(f'{tl:.5f}', (ep, tl), textcoords='offset points',
                     xytext=(0, 9), ha='center', fontsize=7, color=PURPLE)

ax_lstm.set_xlabel('Epoch', fontsize=9)
ax_lstm.set_ylabel('Prediction MSE  (hash / 255)', fontsize=9)
_style_axes(ax_lstm)
panel_title(ax_lstm, 'Model 2  ·  LSTM Predictor  —  Training Curves')
ax_lstm.legend(fontsize=8.5, frameon=True, facecolor=PANEL,
               edgecolor=BORDER, labelcolor=INK, loc='upper right')

_info_box(ax_lstm, [
    f'Best val MSE : {LSTM_BEST_VAL:.6f}',
    f'Naive base.  : {LSTM_NAIVE:.6f}',
    f'Delta        : +{(LSTM_BEST_VAL - LSTM_NAIVE)*1000:.3f}e-3  (noise floor)',
])

# ── MSE comparison — resistance panel ─────────────────────────────────────────
# Show LSTM val MSE vs naive baseline vs theoretical ceiling on a zoomed axis
# All three converge to ~0.0848 — visually confirms unpredictability
mse_labels = ['Theoretical\nceiling\n(1/12)', 'Naive\nbaseline\n(mean pred.)',
              'LSTM\nbest val\nMSE']
mse_vals   = [LSTM_CEIL, LSTM_NAIVE, LSTM_BEST_VAL]
mse_colors = [AMBER, TEAL, PURPLE]

x_pos = np.arange(len(mse_labels))
vbars = ax_mse.bar(x_pos, mse_vals, color=mse_colors,
                   width=0.48, edgecolor=BORDER, linewidth=0.7, zorder=3)

# Value labels at top of each bar
for i, (v, col) in enumerate(zip(mse_vals, mse_colors)):
    ax_mse.text(x_pos[i], v + 0.000035, f'{v:.6f}',
                ha='center', va='bottom', fontsize=9,
                color=col, fontweight='bold')

ax_mse.set_xticks(x_pos)
ax_mse.set_xticklabels(mse_labels, fontsize=8.5, color=MUTED, linespacing=1.4)
ax_mse.set_ylim(0.083, 0.0852)
ax_mse.set_ylabel('MSE', fontsize=9)
ax_mse.yaxis.grid(True, zorder=0, linewidth=0.8)
ax_mse.set_axisbelow(True)
for spine in ('top', 'right'):
    ax_mse.spines[spine].set_visible(False)
ax_mse.spines['left'].set_edgecolor(BORDER)
ax_mse.spines['bottom'].set_edgecolor(BORDER)
ax_mse.tick_params(axis='y', labelsize=8)
panel_title(ax_mse, 'Prediction Resistance')

# Interpretation note
ax_mse.text(0.5, 0.22,
            'LSTM indistinguishable\nfrom mean predictor\n\nHash stream confirmed\ngenuinely unpredictable',
            transform=ax_mse.transAxes, ha='center', va='center',
            fontsize=8.5, color=MUTED, linespacing=1.6,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=HEADER,
                      edgecolor=BORDER, linewidth=0.8))

# ═════════════════════════════════════════════════════════════════════════════
# ROW 4 — Model 4: Classification heatmap  +  Per-class F1 bars
# ═════════════════════════════════════════════════════════════════════════════
row4 = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer[4], wspace=0.34, width_ratios=[1.85, 1])

ax_cls = fig.add_subplot(row4[0])
ax_f1  = fig.add_subplot(row4[1])

# ── Classification heatmap ────────────────────────────────────────────────────
data_matrix = np.array([CLS_PRECISION, CLS_RECALL, CLS_F1])

# Use a warm green-to-red colormap consistent with the light theme
im = ax_cls.imshow(data_matrix, aspect='auto', cmap='RdYlGn',
                   vmin=0.0, vmax=1.0)

ax_cls.set_xticks(range(4))
ax_cls.set_xticklabels(CLS_CLASSES, fontsize=10.5, color=INK)
ax_cls.set_yticks(range(3))
ax_cls.set_yticklabels(['Precision', 'Recall', 'F1-Score'], fontsize=10.5, color=INK)
panel_title(ax_cls, 'Model 4  ·  MLP Classifier  —  Per-Class Metrics')

for i in range(3):
    for j in range(4):
        val = data_matrix[i, j]
        txt_col = INK if 0.35 < val < 0.85 else ('#FFFFFF' if val <= 0.35 else INK)
        ax_cls.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=14, color=txt_col, fontweight='bold')

# Support row below heatmap
for j, (cls, sup) in enumerate(zip(CLS_CLASSES, CLS_SUPPORT)):
    ax_cls.text(j, 3.6, f'n = {sup:,}', ha='center', va='center',
                fontsize=8, color=MUTED,
                transform=ax_cls.get_xaxis_transform())

cbar = plt.colorbar(im, ax=ax_cls, pad=0.02, shrink=0.88)
cbar.ax.yaxis.set_tick_params(labelcolor=MUTED, labelsize=8)
cbar.outline.set_edgecolor(BORDER)

_info_box(ax_cls, [
    f'5-fold CV acc : {CLS_CV_ACC:.1%}  (sd={CLS_CV_STD:.3f})',
    f'Train acc     : {CLS_TRAIN_ACC:.0%}',
    f'Features      : 8  (entropy x4 + state x4)',
])

# ── Per-class F1 score bars ───────────────────────────────────────────────────
# Display in descending F1 order
cls_order = np.argsort(CLS_F1)[::-1]
f1_sorted     = [CLS_F1[i]     for i in cls_order]
name_sorted   = [CLS_CLASSES[i] for i in cls_order]
support_sorted = [CLS_SUPPORT[i] for i in cls_order]

f1_colors = [GREEN if v >= 0.95 else (TEAL if v >= 0.85 else AMBER) for v in f1_sorted]
y4 = np.arange(len(name_sorted))

ax_f1.barh(y4, f1_sorted, color=f1_colors, height=0.5,
           edgecolor=BORDER, linewidth=0.6, zorder=3)

for i, (v, col) in enumerate(zip(f1_sorted, f1_colors)):
    x_lbl = v - 0.02 if v > 0.3 else v + 0.01
    ha     = 'right'  if v > 0.3 else 'left'
    t_col  = '#FFFFFF' if v > 0.6 else col
    ax_f1.text(x_lbl, y4[i], f'{v:.2f}',
               va='center', ha=ha, fontsize=11.5,
               color=t_col, fontweight='bold', zorder=6)
    # support count
    ax_f1.text(1.01, y4[i], f'n={support_sorted[i]:,}',
               va='center', ha='left', fontsize=7.5, color=FAINT)

ax_f1.set_yticks(y4)
ax_f1.set_yticklabels(name_sorted, fontsize=10, color=INK)
ax_f1.set_xlim(0, 1.18)
ax_f1.set_xlabel('F1-Score', fontsize=9)
ax_f1.xaxis.grid(True, zorder=0, linewidth=0.8)
ax_f1.yaxis.grid(False)
ax_f1.set_axisbelow(True)
for spine in ('top', 'right'):
    ax_f1.spines[spine].set_visible(False)
ax_f1.spines['left'].set_edgecolor(BORDER)
ax_f1.spines['bottom'].set_edgecolor(BORDER)
panel_title(ax_f1, 'Per-class F1  (train set)')

# CV / accuracy footer
ax_f1.text(0.5, -0.17,
           f'5-fold CV: {CLS_CV_ACC:.1%}  ±  {CLS_CV_STD:.3f}     '
           f'Train acc: {CLS_TRAIN_ACC:.0%}',
           transform=ax_f1.transAxes, ha='center', va='top',
           fontsize=8.5, color=MUTED)

# ═════════════════════════════════════════════════════════════════════════════
# Section labels — left gutter (rotated, warm palette)
# ═════════════════════════════════════════════════════════════════════════════
for y_pos, label, col in [
    (0.775, '[1] DATASET', TEAL2),
    (0.570, '[2] ANOMALY', AMBER2),
    (0.365, '[3] PREDICTOR', PURPLE),
    (0.150, '[4] CLASSIFIER', GREEN),
]:
    fig.text(0.007, y_pos, label, fontsize=7.5, color=col,
             fontweight='bold', rotation=90, va='center',
             transform=fig.transFigure)

# ── Footer ────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.015,
         'PyTorch 2.10  ·  scikit-learn 1.7  ·  Python 3.10  ·  '
         'Trained on live entropy data from double-pendulum & Lorenz chaos engines',
         ha='center', fontsize=8, color=FAINT)

OUT = 'ml/KAIROS_ML_Metrics.png'
plt.savefig(OUT, dpi=180, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
print(f'Saved --> {OUT}')
