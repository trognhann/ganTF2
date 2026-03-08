"""
AnimeGANv3 Training Log Analyzer
=================================
Parses gantf2.log (with timestamp prefix), generates comprehensive charts,
prints training diagnostics, and provides actionable improvement suggestions.

Usage:
    python logs/analysis_log.py                              # default: logs/gantf2.log
    python logs/analysis_log.py --log path/to/gantf2.log
"""

import re
import argparse
import os
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("ERROR: matplotlib is required. Install with: pip install matplotlib")
    exit(1)

# ─────────────────────────── Parsing ────────────────────────────────────────

# Pre-train line pattern (handles optional timestamp prefix like "76.6s 124")
RE_PRETRAIN = re.compile(
    r"Epoch:\s*(\d+),\s*Step:\s*(\d+)\s*/\s*(\d+),.*Pre_train_G_loss:\s*([\d.]+)"
)

# GAN training line pattern — handles the actual log format:
# D_loss:0.565 ~ G_loss: 28.482 || G_support_loss: 5.425203, g_s_loss: 0.810823,
# con_loss: 0.084055, rs_loss: 0.777939, sty_loss: 2.778852, s22: 0.074608,
# s33: 0.585410, s44: 2.118834, color_loss: 0.973532, tv_loss: 0.000002
# ~ D_support_loss: 0.465154 || G_main_loss: 23.056595, g_m_loss: 0.020006,
# p0_loss: 22.673161, p4_loss: 0.363428, tv_loss_m: 0.000001
# ~ D_main_loss: 0.100012
RE_GAN = re.compile(
    r"Epoch:\s*(\d+),\s*Step:\s*(\d+)\s*/\s*(\d+),.*"
    r"D_loss:([\d.]+)\s*~\s*G_loss:\s*([\d.]+)\s*\|\|"
    r".*G_support_loss:\s*([\d.]+).*"
    r"g_s_loss:\s*([\d.]+).*"
    r"con_loss:\s*([\d.]+).*"
    r"rs_loss:\s*([\d.]+).*"
    r"sty_loss:\s*([\d.]+).*"
    r"s22:\s*([\d.]+).*s33:\s*([\d.]+).*s44:\s*([\d.]+).*"
    r"color_loss:\s*([\d.]+).*"
    r"tv_loss:\s*([\d.]+).*"
    r"D_support_loss:\s*([\d.]+).*"
    r"G_main_loss:\s*([\d.]+).*"
    r"g_m_loss:\s*([\d.]+).*"
    r"p0_loss:\s*([\d.]+).*"
    r"p4_loss:\s*([\d.]+).*"
    r"D_main_loss:\s*([\d.]+)"
)


def parse_log(filepath):
    """Parse log file and return pre-train and GAN data dicts."""
    pretrain = {"epoch": [], "step": [], "g_loss": [], "global_step": []}
    gan = {
        "epoch": [], "step": [], "global_step": [],
        "D_loss": [], "G_loss": [],
        "G_support_loss": [], "g_s_loss": [], "con_loss": [], "rs_loss": [],
        "sty_loss": [], "s22": [], "s33": [], "s44": [],
        "color_loss": [], "tv_loss": [], "D_support_loss": [],
        "G_main_loss": [], "g_m_loss": [], "p0_loss": [], "p4_loss": [],
        "D_main_loss": [],
    }

    pretrain_gs = 0
    gan_gs = 0

    with open(filepath, 'r') as f:
        for line in f:
            stripped = line.strip()

            m = RE_PRETRAIN.search(stripped)
            if m:
                pretrain["epoch"].append(int(m.group(1)))
                pretrain["step"].append(int(m.group(2)))
                pretrain["g_loss"].append(float(m.group(4)))
                pretrain["global_step"].append(pretrain_gs)
                pretrain_gs += 1
                continue

            m = RE_GAN.search(stripped)
            if m:
                gan["epoch"].append(int(m.group(1)))
                gan["step"].append(int(m.group(2)))
                gan["global_step"].append(gan_gs)
                gan["D_loss"].append(float(m.group(4)))
                gan["G_loss"].append(float(m.group(5)))
                gan["G_support_loss"].append(float(m.group(6)))
                gan["g_s_loss"].append(float(m.group(7)))
                gan["con_loss"].append(float(m.group(8)))
                gan["rs_loss"].append(float(m.group(9)))
                gan["sty_loss"].append(float(m.group(10)))
                gan["s22"].append(float(m.group(11)))
                gan["s33"].append(float(m.group(12)))
                gan["s44"].append(float(m.group(13)))
                gan["color_loss"].append(float(m.group(14)))
                gan["tv_loss"].append(float(m.group(15)))
                gan["D_support_loss"].append(float(m.group(16)))
                gan["G_main_loss"].append(float(m.group(17)))
                gan["g_m_loss"].append(float(m.group(18)))
                gan["p0_loss"].append(float(m.group(19)))
                gan["p4_loss"].append(float(m.group(20)))
                gan["D_main_loss"].append(float(m.group(21)))
                gan_gs += 1

    # Convert to numpy
    for k in pretrain:
        pretrain[k] = np.array(pretrain[k])
    for k in gan:
        gan[k] = np.array(gan[k])

    return pretrain, gan


# ─────────────────────────── Smoothing ──────────────────────────────────────

def smooth(y, window=50):
    """Simple moving average smoothing."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='valid')


def ema(y, alpha=0.05):
    """Exponential moving average for better trend analysis."""
    result = np.zeros_like(y)
    result[0] = y[0]
    for i in range(1, len(y)):
        result[i] = alpha * y[i] + (1 - alpha) * result[i - 1]
    return result


# ─────────────────────────── Style Helpers ──────────────────────────────────

COLORS = {
    "blue": "#2196F3", "red": "#F44336", "green": "#4CAF50",
    "orange": "#FF9800", "purple": "#9C27B0", "teal": "#009688",
    "pink": "#E91E63", "amber": "#FFC107", "cyan": "#00BCD4",
    "lime": "#CDDC39", "indigo": "#3F51B5", "brown": "#795548",
    "deep_orange": "#FF5722", "light_green": "#8BC34A",
    "yellow": "#FFEB3B", "grey": "#9E9E9E",
}


def style_ax(ax, title="", xlabel="Step", ylabel="Loss"):
    """Apply dark-theme styling to an axis."""
    ax.set_facecolor('#16213e')
    ax.set_title(title, color='white', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, color='white', fontsize=11)
    ax.set_ylabel(ylabel, color='white', fontsize=11)
    ax.tick_params(colors='white', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#444')
    ax.grid(True, alpha=0.15, color='white')


def add_legend(ax, loc='best', fontsize=9):
    ax.legend(fontsize=fontsize, facecolor='#16213e',
              edgecolor='#444', labelcolor='white', loc=loc)


def save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


# ─────────────────────────── Charting ───────────────────────────────────────

def plot_pretrain(pretrain, save_dir):
    """Plot pre-training G loss curve."""
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')
    style_ax(ax, 'Phase 1: Pre-training Generator (VGG Content Loss)',
             'Global Step', 'Pre-train G Loss')

    gs = pretrain["global_step"]
    loss = pretrain["g_loss"]
    sm = smooth(loss, 30)
    em = ema(loss, 0.05)

    ax.plot(gs, loss, alpha=0.2, color=COLORS["blue"], linewidth=0.5)
    ax.plot(gs[:len(sm)] + 15, sm, color=COLORS["cyan"], linewidth=2,
            label=f'SMA-30 (final={sm[-1]:.4f})')
    ax.plot(gs, em, color=COLORS["amber"], linewidth=1.5, alpha=0.8,
            label=f'EMA (final={em[-1]:.4f})')

    # Epoch boundaries
    epochs = np.unique(pretrain["epoch"])
    for ep in epochs[1:]:
        idx = np.where(pretrain["epoch"] == ep)[0][0]
        ax.axvline(gs[idx], color='white', alpha=0.15,
                   linestyle='--', linewidth=0.8)
        ax.text(gs[idx], ax.get_ylim()[1] * 0.95, f'E{ep}',
                color='white', alpha=0.4, fontsize=8, ha='center')

    add_legend(ax)
    path = os.path.join(save_dir, 'pretrain_loss.png')
    save_fig(fig, path)


def plot_gan_overview(gan, save_dir):
    """Plot D_loss and G_loss overview with trend analysis."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.patch.set_facecolor('#1a1a2e')

    gs = gan["global_step"]
    w = 50

    # D_loss
    ax = axes[0]
    style_ax(ax, 'Phase 2: GAN Training – Discriminator Loss (D_loss)',
             '', 'D_loss')
    d = gan["D_loss"]
    sd = smooth(d, w)
    ed = ema(d, 0.02)
    ax.plot(gs, d, alpha=0.15, color=COLORS["red"], linewidth=0.5)
    ax.plot(gs[:len(sd)] + w // 2, sd, color=COLORS["red"], linewidth=2,
            label=f'D_loss SMA-{w} (final={sd[-1]:.4f})')
    ax.plot(gs, ed, color=COLORS["orange"], linewidth=1.5, alpha=0.7,
            label=f'D_loss EMA (final={ed[-1]:.4f})')
    ax.axhline(0.1, color=COLORS["amber"], linestyle=':', alpha=0.4,
               label='Healthy min (0.1)')
    ax.axhline(0.5, color=COLORS["lime"], linestyle=':', alpha=0.4,
               label='Healthy max (0.5)')
    add_legend(ax)

    # G_loss
    ax = axes[1]
    style_ax(ax, 'Phase 2: GAN Training – Generator Loss (G_loss)',
             'Global Step', 'G_loss')
    g = gan["G_loss"]
    sg = smooth(g, w)
    eg = ema(g, 0.02)
    ax.plot(gs, g, alpha=0.15, color=COLORS["blue"], linewidth=0.5)
    ax.plot(gs[:len(sg)] + w // 2, sg, color=COLORS["blue"], linewidth=2,
            label=f'G_loss SMA-{w} (final={sg[-1]:.4f})')
    ax.plot(gs, eg, color=COLORS["cyan"], linewidth=1.5, alpha=0.7,
            label=f'G_loss EMA (final={eg[-1]:.4f})')
    add_legend(ax)

    # Epoch boundaries
    epochs = np.unique(gan["epoch"])
    for ep in epochs[1:]:
        idx = np.where(gan["epoch"] == ep)[0][0]
        for a in axes:
            a.axvline(gs[idx], color='white', alpha=0.12,
                      linestyle='--', linewidth=0.7)

    fig.tight_layout()
    path = os.path.join(save_dir, 'gan_overview.png')
    save_fig(fig, path)


def plot_d_loss_breakdown(gan, save_dir):
    """Plot D_support_loss and D_main_loss."""
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')
    style_ax(ax, 'Discriminator Loss Breakdown (D_support vs D_main)',
             'Global Step', 'Loss')

    gs = gan["global_step"]
    w = 50

    for name, color in [("D_support_loss", COLORS["orange"]),
                        ("D_main_loss", COLORS["purple"])]:
        y = gan[name]
        sy = smooth(y, w)
        ax.plot(gs, y, alpha=0.12, color=color, linewidth=0.5)
        ax.plot(gs[:len(sy)] + w // 2, sy, color=color, linewidth=2,
                label=f'{name} (final={sy[-1]:.4f})')

    # Plot ratio
    ds = smooth(gan["D_support_loss"], w)
    dm = smooth(gan["D_main_loss"], w)
    min_len = min(len(ds), len(dm))
    ratio = ds[:min_len] / (dm[:min_len] + 1e-8)
    ax2 = ax.twinx()
    ax2.plot(gs[:min_len] + w // 2, ratio, color=COLORS["amber"],
             linewidth=1.5, alpha=0.6, linestyle='--',
             label=f'Ratio D_sup/D_main (final={ratio[-1]:.1f}x)')
    ax2.set_ylabel('Ratio', color=COLORS["amber"], fontsize=11)
    ax2.tick_params(axis='y', colors=COLORS["amber"])

    add_legend(ax, loc='upper right')
    add_legend(ax2, loc='upper left')
    path = os.path.join(save_dir, 'discriminator_breakdown.png')
    save_fig(fig, path)


def plot_g_loss_breakdown(gan, save_dir):
    """Plot G_support_loss vs G_main_loss with g_s_loss."""
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#1a1a2e')
    style_ax(ax, 'Generator Loss Breakdown: Support vs Main vs Adversarial',
             'Global Step', 'Loss')

    gs = gan["global_step"]
    w = 50

    for name, color, lbl in [
        ("G_support_loss", COLORS["green"], "G_support (content/style/color)"),
        ("G_main_loss", COLORS["teal"], "G_main (reconstruction)"),
        ("g_s_loss", COLORS["pink"], "g_s_loss (adversarial support)"),
        ("g_m_loss", COLORS["amber"], "g_m_loss (adversarial main)"),
    ]:
        y = gan[name]
        sy = smooth(y, w)
        ax.plot(gs, y, alpha=0.1, color=color, linewidth=0.5)
        ax.plot(gs[:len(sy)] + w // 2, sy, color=color, linewidth=2,
                label=f'{lbl} (final={sy[-1]:.4f})')

    add_legend(ax, fontsize=8)
    path = os.path.join(save_dir, 'generator_breakdown.png')
    save_fig(fig, path)


def plot_content_style_color(gan, save_dir):
    """Plot content, style, color, and reconstruction losses in a 2×2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor('#1a1a2e')

    gs = gan["global_step"]
    w = 50

    panels = [
        (axes[0, 0], "Content Loss (con_loss)", "con_loss", COLORS["blue"]),
        (axes[0, 1], "Style Loss (sty_loss)", "sty_loss", COLORS["purple"]),
        (axes[1, 0], "Color Loss", "color_loss", COLORS["orange"]),
        (axes[1, 1], "Reconstruction Loss (rs_loss)", "rs_loss", COLORS["green"]),
    ]

    for ax, title, key, color in panels:
        style_ax(ax, title, 'Step', 'Loss')
        y = gan[key]
        sy = smooth(y, w)
        ey = ema(y, 0.03)
        ax.plot(gs, y, alpha=0.15, color=color, linewidth=0.5)
        ax.plot(gs[:len(sy)] + w // 2, sy, color=color, linewidth=2,
                label=f'{key} (final={sy[-1]:.4f})')
        ax.plot(gs, ey, color='white', linewidth=1, alpha=0.4,
                label=f'EMA (final={ey[-1]:.4f})')
        add_legend(ax)

    fig.suptitle('Generator Sub-Losses: Content / Style / Color / Reconstruction',
                 color='white', fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    path = os.path.join(save_dir, 'content_style_color.png')
    save_fig(fig, path)


def plot_style_layers(gan, save_dir):
    """Plot individual style layer losses: s22, s33, s44."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.patch.set_facecolor('#1a1a2e')

    gs = gan["global_step"]
    w = 50

    # Absolute values
    ax = axes[0]
    style_ax(ax, 'Style Loss per VGG Layer (Absolute)', '', 'Loss')
    for name, color in [("s22", COLORS["cyan"]),
                        ("s33", COLORS["amber"]),
                        ("s44", COLORS["pink"])]:
        y = gan[name]
        sy = smooth(y, w)
        ax.plot(gs, y, alpha=0.12, color=color, linewidth=0.5)
        ax.plot(gs[:len(sy)] + w // 2, sy, color=color, linewidth=2,
                label=f'{name} (final={sy[-1]:.4f})')
    add_legend(ax)

    # Proportions
    ax = axes[1]
    style_ax(ax, 'Style Loss Proportions (s22/s33/s44 as % of sty_loss)',
             'Global Step', 'Proportion (%)')
    s22_sm = smooth(gan["s22"], w)
    s33_sm = smooth(gan["s33"], w)
    s44_sm = smooth(gan["s44"], w)
    min_len = min(len(s22_sm), len(s33_sm), len(s44_sm))
    total = s22_sm[:min_len] + s33_sm[:min_len] + s44_sm[:min_len]
    x = gs[:min_len] + w // 2
    ax.fill_between(x, 0, (s22_sm[:min_len] / total) * 100,
                    alpha=0.4, color=COLORS["cyan"], label='s22 %')
    ax.fill_between(x, (s22_sm[:min_len] / total) * 100,
                    ((s22_sm[:min_len] + s33_sm[:min_len]) / total) * 100,
                    alpha=0.4, color=COLORS["amber"], label='s33 %')
    ax.fill_between(x, ((s22_sm[:min_len] + s33_sm[:min_len]) / total) * 100,
                    100, alpha=0.4, color=COLORS["pink"], label='s44 %')
    add_legend(ax)

    fig.tight_layout()
    path = os.path.join(save_dir, 'style_layers.png')
    save_fig(fig, path)


def plot_main_losses(gan, save_dir):
    """Plot p0_loss, p4_loss, g_m_loss."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.patch.set_facecolor('#1a1a2e')

    gs = gan["global_step"]
    w = 50

    # p0_loss (dominant)
    ax = axes[0]
    style_ax(ax, 'Main Branch: p0_loss (Dominant Component)', '', 'Loss')
    y = gan["p0_loss"]
    sy = smooth(y, w)
    ax.plot(gs, y, alpha=0.15, color=COLORS["red"], linewidth=0.5)
    ax.plot(gs[:len(sy)] + w // 2, sy, color=COLORS["red"], linewidth=2,
            label=f'p0_loss (final={sy[-1]:.4f})')
    ax.axhline(2.0, color=COLORS["amber"], linestyle=':', alpha=0.5,
               label='Target threshold (2.0)')
    add_legend(ax)

    # p4_loss + g_m_loss
    ax = axes[1]
    style_ax(ax, 'Main Branch: p4_loss & g_m_loss', 'Global Step', 'Loss')
    for name, color in [("p4_loss", COLORS["lime"]),
                        ("g_m_loss", COLORS["indigo"])]:
        y = gan[name]
        sy = smooth(y, w)
        ax.plot(gs, y, alpha=0.15, color=color, linewidth=0.5)
        ax.plot(gs[:len(sy)] + w // 2, sy, color=color, linewidth=2,
                label=f'{name} (final={sy[-1]:.4f})')
    add_legend(ax)

    fig.tight_layout()
    path = os.path.join(save_dir, 'main_branch_losses.png')
    save_fig(fig, path)


def plot_dg_ratio(gan, save_dir):
    """Plot D_loss / G_loss ratio – key indicator of GAN health."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.patch.set_facecolor('#1a1a2e')

    gs = gan["global_step"]
    w = 50

    # D/G ratio
    ax = axes[0]
    style_ax(ax, 'D/G Loss Ratio (GAN Balance Indicator)',
             '', 'D_loss / G_loss')
    ratio = gan["D_loss"] / (gan["G_loss"] + 1e-8)
    sr = smooth(ratio, w)
    ax.plot(gs, ratio, alpha=0.15, color=COLORS["amber"], linewidth=0.5)
    ax.plot(gs[:len(sr)] + w // 2, sr, color=COLORS["amber"], linewidth=2,
            label=f'D/G ratio (final={sr[-1]:.4f})')
    ax.axhline(0.05, color=COLORS["green"], linestyle='--', alpha=0.5,
               label='Healthy range (0.01–0.05)')
    ax.axhline(0.01, color=COLORS["green"], linestyle='--', alpha=0.5)
    ax.fill_between([gs[0], gs[-1]], 0.01, 0.05,
                    color=COLORS["green"], alpha=0.06)
    add_legend(ax)

    # D_support / D_main ratio
    ax = axes[1]
    style_ax(ax, 'D_support / D_main Ratio (Discriminator Balance)',
             'Global Step', 'Ratio')
    ds = gan["D_support_loss"]
    dm = gan["D_main_loss"]
    d_ratio = ds / (dm + 1e-8)
    sdr = smooth(d_ratio, w)
    ax.plot(gs, d_ratio, alpha=0.15, color=COLORS["purple"], linewidth=0.5)
    ax.plot(gs[:len(sdr)] + w // 2, sdr, color=COLORS["purple"], linewidth=2,
            label=f'D_sup/D_main (final={sdr[-1]:.2f}x)')
    ax.axhline(1.0, color='white', linestyle=':', alpha=0.3,
               label='Equal balance (1.0x)')
    add_legend(ax)

    fig.tight_layout()
    path = os.path.join(save_dir, 'dg_ratio.png')
    save_fig(fig, path)


def plot_epoch_summary(pretrain, gan, save_dir):
    """Bar chart of mean loss per epoch across all phases."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#1a1a2e')

    for ax in axes:
        style_ax(ax)

    # Pre-train per epoch
    ax = axes[0]
    epochs = np.unique(pretrain["epoch"])
    means = [pretrain["g_loss"][pretrain["epoch"] == ep].mean()
             for ep in epochs]
    bars = ax.bar(epochs, means, color=COLORS["cyan"],
                  alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.set_title('Pre-train G Loss (per epoch)',
                 color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch', color='white')
    ax.set_ylabel('Mean Loss', color='white')
    for i, v in enumerate(means):
        ax.text(epochs[i], v + 0.003, f'{v:.3f}', ha='center',
                va='bottom', color='white', fontsize=7)

    # GAN G_loss per epoch
    ax = axes[1]
    gan_epochs = np.unique(gan["epoch"])
    g_means = [gan["G_loss"][gan["epoch"] == ep].mean() for ep in gan_epochs]
    ax.bar(gan_epochs, g_means,
           color=COLORS["blue"], alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.set_title('G_loss (per epoch)', color='white',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch', color='white')
    ax.set_ylabel('Mean Loss', color='white')
    for i, v in enumerate(g_means):
        ax.text(gan_epochs[i], v + 0.1, f'{v:.1f}',
                ha='center', va='bottom', color='white', fontsize=6,
                rotation=45)

    # GAN D_loss per epoch
    ax = axes[2]
    d_means = [gan["D_loss"][gan["epoch"] == ep].mean() for ep in gan_epochs]
    ax.bar(gan_epochs, d_means,
           color=COLORS["red"], alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.set_title('D_loss (per epoch)', color='white',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch', color='white')
    ax.set_ylabel('Mean Loss', color='white')
    for i, v in enumerate(d_means):
        ax.text(gan_epochs[i], v + 0.003,
                f'{v:.3f}', ha='center', va='bottom', color='white',
                fontsize=6, rotation=45)

    fig.suptitle('Per-Epoch Loss Summary', color='white',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(save_dir, 'epoch_summary.png')
    save_fig(fig, path)


def plot_loss_composition(gan, save_dir):
    """Stacked area chart showing how G_loss decomposes into components."""
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor('#1a1a2e')
    style_ax(ax, 'G_loss Composition (Stacked Area)',
             'Global Step', 'Loss')

    gs = gan["global_step"]
    w = 80

    components = [
        ("con_loss", COLORS["blue"]),
        ("rs_loss", COLORS["green"]),
        ("sty_loss", COLORS["purple"]),
        ("color_loss", COLORS["orange"]),
        ("g_s_loss", COLORS["pink"]),
        ("p0_loss", COLORS["red"]),
        ("p4_loss", COLORS["lime"]),
        ("g_m_loss", COLORS["indigo"]),
    ]

    smoothed = {}
    min_len = len(gs)
    for name, _ in components:
        s = smooth(gan[name], w)
        smoothed[name] = s
        min_len = min(min_len, len(s))

    x = gs[:min_len] + w // 2
    bottom = np.zeros(min_len)
    for name, color in components:
        vals = smoothed[name][:min_len]
        ax.fill_between(x, bottom, bottom + vals, alpha=0.6,
                        color=color, label=f'{name} ({vals[-1]:.3f})')
        bottom += vals

    add_legend(ax, loc='upper right', fontsize=8)
    path = os.path.join(save_dir, 'loss_composition.png')
    save_fig(fig, path)


def plot_convergence_speed(gan, save_dir):
    """Plot rate of change of key losses to identify convergence/divergence."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.patch.set_facecolor('#1a1a2e')

    gs = gan["global_step"]
    w = 100

    # G_loss rate of change (per-epoch mean)
    ax = axes[0]
    style_ax(ax, 'Training Convergence: Per-Epoch Mean Losses',
             '', 'Mean Loss')
    gan_epochs = np.unique(gan["epoch"])
    g_means = [gan["G_loss"][gan["epoch"] == ep].mean() for ep in gan_epochs]
    d_means = [gan["D_loss"][gan["epoch"] == ep].mean() for ep in gan_epochs]
    ax.plot(gan_epochs, g_means, 'o-', color=COLORS["blue"], linewidth=2,
            markersize=4, label='G_loss (per epoch)')
    ax.plot(gan_epochs, d_means, 's-', color=COLORS["red"], linewidth=2,
            markersize=4, label='D_loss (per epoch)')
    add_legend(ax)

    # Loss improvement rate
    ax = axes[1]
    style_ax(ax, 'Epoch-over-Epoch Improvement Rate',
             'Epoch', 'Change (%)')
    if len(g_means) > 1:
        g_change = [(g_means[i] - g_means[i - 1]) / g_means[i - 1] * 100
                    for i in range(1, len(g_means))]
        ax.bar(gan_epochs[1:] - 0.15, g_change, width=0.3,
               color=COLORS["blue"], alpha=0.7, label='G_loss change %')
        d_change = [(d_means[i] - d_means[i - 1]) / d_means[i - 1] * 100
                    for i in range(1, len(d_means))]
        ax.bar(gan_epochs[1:] + 0.15, d_change, width=0.3,
               color=COLORS["red"], alpha=0.7, label='D_loss change %')
        ax.axhline(0, color='white', linestyle='-', alpha=0.3)
    add_legend(ax)

    fig.tight_layout()
    path = os.path.join(save_dir, 'convergence_speed.png')
    save_fig(fig, path)


# ─────────────────────────── Diagnostics ────────────────────────────────────

def print_diagnostics(pretrain, gan):
    """Print comprehensive training diagnostics with improvement suggestions."""
    print("\n" + "=" * 80)
    print("  📊  ANIMEGAN v3 TRAINING DIAGNOSTICS")
    print("=" * 80)

    # ── Pre-training Summary ──
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  PHASE 1: PRE-TRAINING (Generator Content Learning)    │")
    print("└─────────────────────────────────────────────────────────┘")
    n_pretrain_epochs = len(np.unique(pretrain["epoch"]))
    n_pretrain_steps = len(pretrain["g_loss"])
    print(f"  Epochs: {n_pretrain_epochs} (Epoch 0 → {n_pretrain_epochs - 1})")
    print(f"  Total steps: {n_pretrain_steps}")
    print(
        f"  G_loss: {pretrain['g_loss'][0]:.4f} → {pretrain['g_loss'][-1]:.4f}")
    print(
        f"  Reduction: {(1 - pretrain['g_loss'][-1] / pretrain['g_loss'][0]) * 100:.1f}%")

    # Per-epoch pre-train
    for ep in np.unique(pretrain["epoch"]):
        mask = pretrain["epoch"] == ep
        vals = pretrain["g_loss"][mask]
        print(f"    Epoch {ep}: mean={vals.mean():.4f}, min={vals.min():.4f}, "
              f"max={vals.max():.4f}, std={vals.std():.4f}")

    # ── GAN Training Summary ──
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  PHASE 2: GAN ADVERSARIAL TRAINING                     │")
    print("└─────────────────────────────────────────────────────────┘")
    n_gan_epochs = len(np.unique(gan["epoch"]))
    n_gan_steps = len(gan["G_loss"])
    print(
        f"  Epochs: {n_gan_epochs} (Epoch {gan['epoch'][0]} → {gan['epoch'][-1]})")
    print(f"  Total steps: {n_gan_steps}")
    print(f"\n  D_loss: {gan['D_loss'][0]:.4f} → {gan['D_loss'][-1]:.4f}")
    print(f"  G_loss: {gan['G_loss'][0]:.2f} → {gan['G_loss'][-1]:.2f}")

    # Key sub-losses final
    w = 50
    keys_final = {}
    for k in ["D_loss", "G_loss", "G_support_loss", "G_main_loss",
              "D_support_loss", "D_main_loss", "con_loss", "sty_loss",
              "color_loss", "rs_loss", "s22", "s33", "s44",
              "g_s_loss", "g_m_loss", "p0_loss", "p4_loss", "tv_loss"]:
        keys_final[k] = smooth(gan[k], w)[-1]

    # Per-epoch GAN
    print("\n  Per-Epoch Summary:")
    print(
        f"  {'Epoch':>5} │ {'D_loss':>8} │ {'G_loss':>8} │ {'con':>7} │ {'sty':>7} │ {'color':>7} │ {'p0':>8}")
    print(
        f"  {'─' * 5}─┼─{'─' * 8}─┼─{'─' * 8}─┼─{'─' * 7}─┼─{'─' * 7}─┼─{'─' * 7}─┼─{'─' * 8}")
    for ep in np.unique(gan["epoch"]):
        mask = gan["epoch"] == ep
        print(f"  {ep:>5} │ {gan['D_loss'][mask].mean():>8.4f} │ "
              f"{gan['G_loss'][mask].mean():>8.2f} │ "
              f"{gan['con_loss'][mask].mean():>7.4f} │ "
              f"{gan['sty_loss'][mask].mean():>7.3f} │ "
              f"{gan['color_loss'][mask].mean():>7.3f} │ "
              f"{gan['p0_loss'][mask].mean():>8.3f}")

    # ── Health Checks ──
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  🔍 HEALTH CHECKS & ISSUE DETECTION                    │")
    print("└─────────────────────────────────────────────────────────┘")

    issues = []
    warnings = []
    improvements = []

    # 1. Check for NaN/Inf
    nan_found = False
    for key in gan:
        if key in ("epoch", "step", "global_step"):
            continue
        if np.any(np.isnan(gan[key])) or np.any(np.isinf(gan[key])):
            nan_found = True
            issues.append(f"❌ NaN/Inf detected in {key}")
    if not nan_found:
        print("  ✅ No NaN/Inf values detected in any loss")

    # 2. D_loss analysis
    if keys_final["D_loss"] < 0.1:
        warnings.append(
            f"⚠️  D_loss very low ({keys_final['D_loss']:.4f}) — D may be overpowering G.\n"
            f"     D learns too fast, providing weak gradients to G.")
        improvements.append(
            "🔧 Reduce D learning rate (try 0.00005 instead of 0.0001)\n"
            "     or increase D update ratio to train D less frequently")
    elif keys_final["D_loss"] > 0.4:
        print(f"  ✅ D_loss ({keys_final['D_loss']:.4f}) healthy — D is not dominating")
    else:
        print(f"  ✅ D_loss ({keys_final['D_loss']:.4f}) in reasonable range")

    # 3. G_loss convergence
    gl_first_third = gan["G_loss"][:len(gan["G_loss"]) // 3].mean()
    gl_last_third = gan["G_loss"][2 * len(gan["G_loss"]) // 3:].mean()
    if gl_last_third < gl_first_third:
        print(
            f"  ✅ G_loss decreasing ({gl_first_third:.2f} → {gl_last_third:.2f})")
    else:
        warnings.append(
            f"⚠️  G_loss NOT decreasing ({gl_first_third:.2f} → {gl_last_third:.2f})")

    # 4. G_loss plateau detection
    last_5_epochs = np.unique(gan["epoch"])[-5:]
    epoch_g_means = [gan["G_loss"][gan["epoch"] == ep].mean()
                     for ep in last_5_epochs]
    if len(epoch_g_means) >= 3:
        g_std = np.std(epoch_g_means)
        g_range = max(epoch_g_means) - min(epoch_g_means)
        if g_range < 0.3:
            warnings.append(
                f"⚠️  G_loss has plateaued in last 5 epochs (range={g_range:.3f})\n"
                f"     Training may have stalled.")
            improvements.append(
                "🔧 Consider reducing learning rate (cosine annealing or step decay)\n"
                "     Current lr=0.0001 → try 0.00005 or use warm restart")

    # 5. D/G ratio
    ratio = gan["D_loss"] / (gan["G_loss"] + 1e-8)
    last_ratio = smooth(ratio, w)[-1]
    if last_ratio < 0.01:
        warnings.append(
            f"⚠️  D/G ratio very low ({last_ratio:.4f}) — D is collapsing.\n"
            f"     Discriminator provides poor learning signal.")
    elif last_ratio > 0.1:
        warnings.append(
            f"⚠️  D/G ratio high ({last_ratio:.4f}) — D dominates.")
    else:
        print(f"  ✅ D/G ratio ({last_ratio:.4f}) within reasonable range")

    # 6. Style loss dominance
    if keys_final["sty_loss"] > 3 * keys_final["con_loss"]:
        warnings.append(
            f"⚠️  Style loss ({keys_final['sty_loss']:.3f}) >> Content loss ({keys_final['con_loss']:.3f})\n"
            f"     Style is {keys_final['sty_loss'] / keys_final['con_loss']:.0f}x stronger than content.\n"
            f"     This causes loss of content structure in generated images.")
        improvements.append(
            "🔧 INCREASE con_loss weight (e.g., con_weight: 1.5→2.5 or 3.0)\n"
            "     or DECREASE sty_loss weight to balance content vs style")
    else:
        print(
            f"  ✅ Style/Content balance: sty={keys_final['sty_loss']:.3f}, con={keys_final['con_loss']:.3f}")

    # 7. s44 dominance analysis
    s22_f = keys_final["s22"]
    s33_f = keys_final["s33"]
    s44_f = keys_final["s44"]
    s_total = s22_f + s33_f + s44_f
    print(
        f"  📌 Style layers: s22={s22_f:.4f} ({s22_f / s_total * 100:.0f}%), "
        f"s33={s33_f:.4f} ({s33_f / s_total * 100:.0f}%), "
        f"s44={s44_f:.4f} ({s44_f / s_total * 100:.0f}%)")
    if s44_f > 2.5 * (s22_f + s33_f):
        warnings.append(
            f"⚠️  s44 dominates style loss ({s44_f / s_total * 100:.0f}% of total)\n"
            f"     Deep VGG features drive style → high-level patterns over fine textures.\n"
            f"     This may produce blurry/overly-abstract anime textures.")
        improvements.append(
            "🔧 Increase weight of s22/s33 relative to s44 in style loss\n"
            "     This will encourage finer texture details from shallower VGG layers")

    # 8. p0_loss analysis (major component of G_main_loss)
    p0_f = keys_final["p0_loss"]
    p4_f = keys_final["p4_loss"]
    gm_f = keys_final["G_main_loss"]
    print(f"  📌 G_main decomposition: p0={p0_f:.3f} ({p0_f / gm_f * 100:.0f}%), "
          f"p4={p4_f:.3f} ({p4_f / gm_f * 100:.0f}%), "
          f"g_m={keys_final['g_m_loss']:.4f}")
    if p0_f > 2.0:
        warnings.append(
            f"⚠️  p0_loss still high ({p0_f:.3f}) — main branch reconstruction"
            f" is underperforming.\n"
            f"     This is the largest single component of total loss.")
        improvements.append(
            "🔧 p0_loss is the patch-level adversarial loss for the main branch\n"
            "     Consider: increase training epochs, or fine-tune with lower lr")

    # 9. Color loss
    color_f = keys_final["color_loss"]
    if color_f > 0.6:
        warnings.append(
            f"⚠️  Color loss is elevated ({color_f:.3f}) — color fidelity is poor.\n"
            f"     Generated images may have incorrect color distribution.")
        improvements.append(
            "🔧 Increase color_loss weight to enforce better color preservation\n"
            "     Or add color histogram matching as preprocessing")

    # 10. Pre-training convergence
    pt_final = smooth(pretrain["g_loss"], 30)[-1]
    if pt_final < 0.20:
        print(f"  ✅ Pre-training converged well (final={pt_final:.4f})")
    elif pt_final < 0.30:
        print(f"  ✅ Pre-training converged reasonably (final={pt_final:.4f})")
    else:
        warnings.append(
            f"⚠️  Pre-training loss is still high ({pt_final:.4f}).\n"
            f"     Consider more pre-train epochs (currently {n_pretrain_epochs}).")
        improvements.append(
            f"🔧 Increase init_G_epoch from {n_pretrain_epochs} to 8-10 epochs\n"
            f"     Better pre-training leads to more stable GAN training")

    # 11. D_support vs D_main imbalance
    ds_f = keys_final["D_support_loss"]
    dm_f = keys_final["D_main_loss"]
    ds_dm_ratio = ds_f / (dm_f + 1e-8)
    if ds_dm_ratio > 5.0:
        warnings.append(
            f"⚠️  D_support ({ds_f:.4f}) >> D_main ({dm_f:.4f}), ratio={ds_dm_ratio:.1f}x\n"
            f"     Support discriminator is weaker than main discriminator.")
    elif ds_dm_ratio < 0.5:
        warnings.append(
            f"⚠️  D_main ({dm_f:.4f}) >> D_support ({ds_f:.4f})\n"
            f"     Main discriminator may be too strong relative to support.")
    else:
        print(f"  ✅ D_support/D_main balance: {ds_dm_ratio:.1f}x (reasonable)")

    # 12. Overall G_loss magnitude
    if keys_final["G_loss"] > 5.0:
        warnings.append(
            f"⚠️  G_loss is still high ({keys_final['G_loss']:.2f}) after {n_gan_epochs} epochs.\n"
            f"     For good visual quality, G_loss should typically converge to 3-5.")
        improvements.append(
            "🔧 Need more training epochs (40-60 total GAN epochs recommended)\n"
            "     Also consider: larger dataset, data augmentation, or architecture changes")

    # Print all issues, warnings, improvements
    for issue in issues:
        print(f"\n  {issue}")
    for warn in warnings:
        print(f"\n  {warn}")

    # ── Improvement Suggestions ──
    if improvements:
        print("\n┌─────────────────────────────────────────────────────────┐")
        print("│  💡 ACTIONABLE IMPROVEMENT SUGGESTIONS                  │")
        print("└─────────────────────────────────────────────────────────┘")
        for i, imp in enumerate(improvements, 1):
            print(f"\n  {i}. {imp}")

    # ── Final Summary ──
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  📋 TRAINING SUMMARY                                    │")
    print("└─────────────────────────────────────────────────────────┘")
    total_epochs = n_pretrain_epochs + n_gan_epochs
    print(
        f"  Total epochs: {total_epochs} ({n_pretrain_epochs} pre-train + {n_gan_epochs} GAN)")
    print(f"  Total steps:  {n_pretrain_steps + n_gan_steps}")
    print(
        f"\n  Pre-train G_loss:  {pretrain['g_loss'][0]:.4f} → {pt_final:.4f}")
    last_d = smooth(gan["D_loss"], w)
    last_g = smooth(gan["G_loss"], w)
    print(f"  GAN D_loss:        {gan['D_loss'][0]:.4f} → {last_d[-1]:.4f}")
    print(f"  GAN G_loss:        {gan['G_loss'][0]:.2f} → {last_g[-1]:.2f}")

    n_issues = len(issues)
    n_warn = len(warnings)
    if n_issues == 0 and n_warn <= 1:
        print("\n  🎉 Overall: Training looks HEALTHY!")
    elif n_issues > 0:
        print(
            f"\n  🚨 Overall: {n_issues} critical issue(s) found. See above.")
    else:
        print(
            f"\n  ⚡ Overall: Training progressing but has {n_warn} warning(s).")

    # Priority improvements
    print("\n  🎯 TOP PRIORITY IMPROVEMENTS:")
    print("     1. Train for MORE epochs (current: 35 GAN epochs → target: 60-80)")
    print("     2. Increase content loss weight to preserve structure better")
    print("     3. Balance style layers (reduce s44 dominance)")
    print("     4. Consider reducing learning rate for fine-tuning phase")
    print("     5. Use larger/more diverse training dataset if possible")
    print("=" * 80)


# ─────────────────────────── Main ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AnimeGANv3 Training Log Analyzer")
    parser.add_argument("--log", type=str, default=None,
                        help="Path to log file (default: logs/gantf2.log)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for charts (default: logs/charts/)")
    args = parser.parse_args()

    # Resolve paths relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = args.log or os.path.join(script_dir, "gantf2.log")
    save_dir = args.output or os.path.join(script_dir, "charts")

    if not os.path.exists(log_path):
        print(f"❌ Log file not found: {log_path}")
        return

    os.makedirs(save_dir, exist_ok=True)

    print(f"\n📁 Log file:   {log_path}")
    print(f"📂 Charts dir: {save_dir}\n")

    # Parse
    print("📖 Parsing log file...")
    pretrain, gan = parse_log(log_path)
    print(f"   Pre-train: {len(pretrain['g_loss'])} steps "
          f"({len(np.unique(pretrain['epoch']))} epochs)")
    print(f"   GAN train: {len(gan['G_loss'])} steps "
          f"({len(np.unique(gan['epoch']))} epochs)")

    if len(pretrain['g_loss']) == 0 and len(gan['G_loss']) == 0:
        print("\n❌ No training data found! Check log format.")
        return

    # Generate charts
    print("\n🎨 Generating charts...")
    if len(pretrain['g_loss']) > 0:
        plot_pretrain(pretrain, save_dir)
    if len(gan['G_loss']) > 0:
        plot_gan_overview(gan, save_dir)
        plot_d_loss_breakdown(gan, save_dir)
        plot_g_loss_breakdown(gan, save_dir)
        plot_content_style_color(gan, save_dir)
        plot_style_layers(gan, save_dir)
        plot_main_losses(gan, save_dir)
        plot_dg_ratio(gan, save_dir)
        plot_loss_composition(gan, save_dir)
        plot_convergence_speed(gan, save_dir)
    if len(pretrain['g_loss']) > 0 and len(gan['G_loss']) > 0:
        plot_epoch_summary(pretrain, gan, save_dir)

    # Diagnostics
    print_diagnostics(pretrain, gan)


if __name__ == "__main__":
    main()
