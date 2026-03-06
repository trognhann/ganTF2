"""
AnimeGANv3 Training Log Analyzer
=================================
Parses train.log, generates comprehensive charts, and prints training diagnostics.

Usage:
    python logs/analysis_log.py                       # default: logs/train.log
    python logs/analysis_log.py --log path/to/train.log
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

# Pre-train line pattern
RE_PRETRAIN = re.compile(
    r"Epoch:\s*(\d+),\s*Step:\s*(\d+)\s*/\s*(\d+),.*Pre_train_G_loss:\s*([\d.]+)"
)

# GAN training line pattern
RE_GAN = re.compile(
    r"Epoch:\s*(\d+),\s*Step:\s*(\d+)\s*/\s*(\d+),.*"
    r"D_loss:([\d.]+)\s*~\s*G_loss:\s*([\d.]+)\s*\|\|"
    r".*G_support_loss:\s*([\d.]+).*"
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
        "G_support_loss": [], "con_loss": [], "rs_loss": [],
        "sty_loss": [], "s22": [], "s33": [], "s44": [],
        "color_loss": [], "tv_loss": [], "D_support_loss": [],
        "G_main_loss": [], "g_m_loss": [], "p0_loss": [], "p4_loss": [],
        "D_main_loss": [],
    }

    pretrain_gs = 0
    gan_gs = 0

    with open(filepath, 'r') as f:
        for line in f:
            m = RE_PRETRAIN.search(line.strip())
            if m:
                pretrain["epoch"].append(int(m.group(1)))
                pretrain["step"].append(int(m.group(2)))
                pretrain["g_loss"].append(float(m.group(4)))
                pretrain["global_step"].append(pretrain_gs)
                pretrain_gs += 1
                continue

            m = RE_GAN.search(line.strip())
            if m:
                gan["epoch"].append(int(m.group(1)))
                gan["step"].append(int(m.group(2)))
                gan["global_step"].append(gan_gs)
                gan["D_loss"].append(float(m.group(4)))
                gan["G_loss"].append(float(m.group(5)))
                gan["G_support_loss"].append(float(m.group(6)))
                gan["con_loss"].append(float(m.group(7)))
                gan["rs_loss"].append(float(m.group(8)))
                gan["sty_loss"].append(float(m.group(9)))
                gan["s22"].append(float(m.group(10)))
                gan["s33"].append(float(m.group(11)))
                gan["s44"].append(float(m.group(12)))
                gan["color_loss"].append(float(m.group(13)))
                gan["tv_loss"].append(float(m.group(14)))
                gan["D_support_loss"].append(float(m.group(15)))
                gan["G_main_loss"].append(float(m.group(16)))
                gan["g_m_loss"].append(float(m.group(17)))
                gan["p0_loss"].append(float(m.group(18)))
                gan["p4_loss"].append(float(m.group(19)))
                gan["D_main_loss"].append(float(m.group(20)))
                gan_gs += 1

    # Convert to numpy
    for k in pretrain:
        pretrain[k] = np.array(pretrain[k])
    for k in gan:
        gan[k] = np.array(gan[k])

    return pretrain, gan


# ─────────────────────────── Smoothing ──────────────────────────────────────

def smooth(y, window=20):
    """Simple moving average smoothing."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='valid')


# ─────────────────────────── Charting ───────────────────────────────────────

COLORS = {
    "blue": "#2196F3", "red": "#F44336", "green": "#4CAF50",
    "orange": "#FF9800", "purple": "#9C27B0", "teal": "#009688",
    "pink": "#E91E63", "amber": "#FFC107", "cyan": "#00BCD4",
    "lime": "#CDDC39", "indigo": "#3F51B5", "brown": "#795548",
}


def plot_pretrain(pretrain, save_dir):
    """Plot pre-training G loss curve."""
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    gs = pretrain["global_step"]
    loss = pretrain["g_loss"]
    sm = smooth(loss, 30)

    ax.plot(gs, loss, alpha=0.25, color=COLORS["blue"], linewidth=0.6)
    ax.plot(gs[:len(sm)] + 15, sm, color=COLORS["cyan"], linewidth=2,
            label=f'Smoothed (final={sm[-1]:.4f})')

    # Epoch boundaries
    epochs = np.unique(pretrain["epoch"])
    for ep in epochs[1:]:
        idx = np.where(pretrain["epoch"] == ep)[0][0]
        ax.axvline(gs[idx], color='white', alpha=0.15,
                   linestyle='--', linewidth=0.8)

    ax.set_xlabel('Global Step', color='white', fontsize=12)
    ax.set_ylabel('Pre-train G Loss (VGG Content)', color='white', fontsize=12)
    ax.set_title('Phase 1: Pre-training Generator',
                 color='white', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, facecolor='#16213e',
              edgecolor='#444', labelcolor='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444')
    ax.grid(True, alpha=0.15)

    path = os.path.join(save_dir, 'pretrain_loss.png')
    fig.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


def plot_gan_overview(gan, save_dir):
    """Plot D_loss and G_loss overview."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor('#1a1a2e')

    gs = gan["global_step"]
    w = 30

    for ax in axes:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#444')
        ax.grid(True, alpha=0.15)

    # D_loss
    ax = axes[0]
    d = gan["D_loss"]
    sd = smooth(d, w)
    ax.plot(gs, d, alpha=0.2, color=COLORS["red"], linewidth=0.5)
    ax.plot(gs[:len(sd)] + w // 2, sd, color=COLORS["red"], linewidth=2,
            label=f'D_loss (final={sd[-1]:.4f})')
    ax.set_ylabel('D_loss', color='white', fontsize=12)
    ax.set_title('Phase 2: GAN Training – Discriminator & Generator Losses',
                 color='white', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, facecolor='#16213e',
              edgecolor='#444', labelcolor='white')

    # G_loss
    ax = axes[1]
    g = gan["G_loss"]
    sg = smooth(g, w)
    ax.plot(gs, g, alpha=0.2, color=COLORS["blue"], linewidth=0.5)
    ax.plot(gs[:len(sg)] + w // 2, sg, color=COLORS["blue"], linewidth=2,
            label=f'G_loss (final={sg[-1]:.4f})')
    ax.set_xlabel('Global Step', color='white', fontsize=12)
    ax.set_ylabel('G_loss', color='white', fontsize=12)
    ax.legend(fontsize=10, facecolor='#16213e',
              edgecolor='#444', labelcolor='white')

    # Epoch boundaries
    epochs = np.unique(gan["epoch"])
    for ep in epochs[1:]:
        idx = np.where(gan["epoch"] == ep)[0][0]
        for a in axes:
            a.axvline(gs[idx], color='white', alpha=0.15,
                      linestyle='--', linewidth=0.8)

    fig.tight_layout()
    path = os.path.join(save_dir, 'gan_overview.png')
    fig.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


def plot_d_loss_breakdown(gan, save_dir):
    """Plot D_support_loss and D_main_loss."""
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    gs = gan["global_step"]
    w = 30

    for name, color in [("D_support_loss", COLORS["orange"]),
                        ("D_main_loss", COLORS["purple"])]:
        y = gan[name]
        sy = smooth(y, w)
        ax.plot(gs, y, alpha=0.15, color=color, linewidth=0.5)
        ax.plot(gs[:len(sy)] + w // 2, sy, color=color, linewidth=2,
                label=f'{name} (final={sy[-1]:.4f})')

    ax.set_xlabel('Global Step', color='white', fontsize=12)
    ax.set_ylabel('Loss', color='white', fontsize=12)
    ax.set_title('Discriminator Loss Breakdown',
                 color='white', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, facecolor='#16213e',
              edgecolor='#444', labelcolor='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444')
    ax.grid(True, alpha=0.15)

    path = os.path.join(save_dir, 'discriminator_breakdown.png')
    fig.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


def plot_g_loss_breakdown(gan, save_dir):
    """Plot G_support_loss and G_main_loss."""
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    gs = gan["global_step"]
    w = 30

    for name, color in [("G_support_loss", COLORS["green"]),
                        ("G_main_loss", COLORS["teal"])]:
        y = gan[name]
        sy = smooth(y, w)
        ax.plot(gs, y, alpha=0.15, color=color, linewidth=0.5)
        ax.plot(gs[:len(sy)] + w // 2, sy, color=color, linewidth=2,
                label=f'{name} (final={sy[-1]:.4f})')

    ax.set_xlabel('Global Step', color='white', fontsize=12)
    ax.set_ylabel('Loss', color='white', fontsize=12)
    ax.set_title('Generator Loss Breakdown: Support vs Main',
                 color='white', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, facecolor='#16213e',
              edgecolor='#444', labelcolor='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444')
    ax.grid(True, alpha=0.15)

    path = os.path.join(save_dir, 'generator_breakdown.png')
    fig.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


def plot_content_style_color(gan, save_dir):
    """Plot content, style, color, and reconstruction losses."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor('#1a1a2e')

    gs = gan["global_step"]
    w = 30

    panels = [
        (axes[0, 0], "Content Loss (con_loss)", "con_loss", COLORS["blue"]),
        (axes[0, 1], "Style Loss (sty_loss)", "sty_loss", COLORS["purple"]),
        (axes[1, 0], "Color Loss", "color_loss", COLORS["orange"]),
        (axes[1, 1], "Reconstruction Loss (rs_loss)", "rs_loss", COLORS["green"]),
    ]

    for ax, title, key, color in panels:
        ax.set_facecolor('#16213e')
        y = gan[key]
        sy = smooth(y, w)
        ax.plot(gs, y, alpha=0.2, color=color, linewidth=0.5)
        ax.plot(gs[:len(sy)] + w // 2, sy, color=color, linewidth=2,
                label=f'{key} (final={sy[-1]:.4f})')
        ax.set_title(title, color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel('Step', color='white', fontsize=10)
        ax.set_ylabel('Loss', color='white', fontsize=10)
        ax.legend(fontsize=9, facecolor='#16213e',
                  edgecolor='#444', labelcolor='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#444')
        ax.grid(True, alpha=0.15)

    fig.suptitle('Generator Sub-Losses: Content / Style / Color / Reconstruction',
                 color='white', fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    path = os.path.join(save_dir, 'content_style_color.png')
    fig.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


def plot_style_layers(gan, save_dir):
    """Plot individual style layer losses: s22, s33, s44."""
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    gs = gan["global_step"]
    w = 30

    for name, color in [("s22", COLORS["cyan"]),
                        ("s33", COLORS["amber"]),
                        ("s44", COLORS["pink"])]:
        y = gan[name]
        sy = smooth(y, w)
        ax.plot(gs, y, alpha=0.15, color=color, linewidth=0.5)
        ax.plot(gs[:len(sy)] + w // 2, sy, color=color, linewidth=2,
                label=f'{name} (final={sy[-1]:.4f})')

    ax.set_xlabel('Global Step', color='white', fontsize=12)
    ax.set_ylabel('Loss', color='white', fontsize=12)
    ax.set_title('Style Loss per VGG Layer (s22, s33, s44)',
                 color='white', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, facecolor='#16213e',
              edgecolor='#444', labelcolor='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444')
    ax.grid(True, alpha=0.15)

    path = os.path.join(save_dir, 'style_layers.png')
    fig.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


def plot_main_losses(gan, save_dir):
    """Plot p0_loss, p4_loss, g_m_loss."""
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    gs = gan["global_step"]
    w = 30

    for name, color in [("p0_loss", COLORS["red"]),
                        ("p4_loss", COLORS["lime"]),
                        ("g_m_loss", COLORS["indigo"])]:
        y = gan[name]
        sy = smooth(y, w)
        ax.plot(gs, y, alpha=0.15, color=color, linewidth=0.5)
        ax.plot(gs[:len(sy)] + w // 2, sy, color=color, linewidth=2,
                label=f'{name} (final={sy[-1]:.4f})')

    ax.set_xlabel('Global Step', color='white', fontsize=12)
    ax.set_ylabel('Loss', color='white', fontsize=12)
    ax.set_title('Main Branch Losses (p0_loss, p4_loss, g_m_loss)',
                 color='white', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, facecolor='#16213e',
              edgecolor='#444', labelcolor='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444')
    ax.grid(True, alpha=0.15)

    path = os.path.join(save_dir, 'main_branch_losses.png')
    fig.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


def plot_dg_ratio(gan, save_dir):
    """Plot D_loss / G_loss ratio – key indicator of GAN health."""
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    gs = gan["global_step"]
    ratio = gan["D_loss"] / (gan["G_loss"] + 1e-8)
    sr = smooth(ratio, 30)

    ax.plot(gs, ratio, alpha=0.2, color=COLORS["amber"], linewidth=0.5)
    ax.plot(gs[:len(sr)] + 15, sr, color=COLORS["amber"], linewidth=2,
            label=f'D/G ratio (final={sr[-1]:.4f})')
    ax.axhline(0.05, color=COLORS["red"], linestyle='--',
               alpha=0.5, label='Healthy range bounds')
    ax.axhline(0.01, color=COLORS["red"], linestyle='--', alpha=0.5)
    ax.fill_between([gs[0], gs[-1]], 0.01, 0.05,
                    color=COLORS["green"], alpha=0.05)

    ax.set_xlabel('Global Step', color='white', fontsize=12)
    ax.set_ylabel('D_loss / G_loss', color='white', fontsize=12)
    ax.set_title('D/G Loss Ratio (GAN Balance Indicator)',
                 color='white', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, facecolor='#16213e',
              edgecolor='#444', labelcolor='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444')
    ax.grid(True, alpha=0.15)

    path = os.path.join(save_dir, 'dg_ratio.png')
    fig.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


def plot_epoch_summary(pretrain, gan, save_dir):
    """Bar chart of mean loss per epoch across all phases."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#1a1a2e')

    for ax in axes:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#444')

    # Pre-train per epoch
    ax = axes[0]
    epochs = np.unique(pretrain["epoch"])
    means = [pretrain["g_loss"][pretrain["epoch"] == ep].mean()
             for ep in epochs]
    ax.bar(epochs, means, color=COLORS["cyan"],
           alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.set_title('Pre-train G Loss (per epoch)',
                 color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch', color='white')
    ax.set_ylabel('Mean Loss', color='white')
    for i, v in enumerate(means):
        ax.text(epochs[i], v + 0.005, f'{v:.3f}', ha='center',
                va='bottom', color='white', fontsize=8)

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
        ax.text(gan_epochs[i], v + 0.1, f'{v:.2f}',
                ha='center', va='bottom', color='white', fontsize=8)

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
        ax.text(gan_epochs[i], v + 0.005,
                f'{v:.3f}', ha='center', va='bottom', color='white', fontsize=8)

    fig.suptitle('Per-Epoch Loss Summary', color='white',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(save_dir, 'epoch_summary.png')
    fig.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


# ─────────────────────────── Diagnostics ────────────────────────────────────

def print_diagnostics(pretrain, gan):
    """Print comprehensive training diagnostics."""
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
    print(f"  G_loss: {gan['G_loss'][0]:.4f} → {gan['G_loss'][-1]:.4f}")

    # Per-epoch GAN
    print("\n  Per-Epoch Summary:")
    print(f"  {'Epoch':>5} │ {'D_loss':>8} │ {'G_loss':>8} │ {'D_sup':>8} │ {'D_main':>8} │ {'G_sup':>8} │ {'G_main':>8}")
    print(f"  {'─' * 5}─┼─{'─' * 8}─┼─{'─' * 8}─┼─{'─' * 8}─┼─{'─' * 8}─┼─{'─' * 8}─┼─{'─' * 8}")
    for ep in np.unique(gan["epoch"]):
        mask = gan["epoch"] == ep
        print(f"  {ep:>5} │ {gan['D_loss'][mask].mean():>8.4f} │ "
              f"{gan['G_loss'][mask].mean():>8.3f} │ "
              f"{gan['D_support_loss'][mask].mean():>8.4f} │ "
              f"{gan['D_main_loss'][mask].mean():>8.4f} │ "
              f"{gan['G_support_loss'][mask].mean():>8.3f} │ "
              f"{gan['G_main_loss'][mask].mean():>8.3f}")

    # ── Health Checks ──
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  🔍 HEALTH CHECKS & ISSUE DETECTION                    │")
    print("└─────────────────────────────────────────────────────────┘")

    issues = []
    warnings = []

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

    # 2. Check D_loss collapse (too low = D is too strong)
    last_d = smooth(gan["D_loss"], 30)
    if last_d[-1] < 0.1:
        warnings.append(
            f"⚠️  D_loss is very low ({last_d[-1]:.4f}) — Discriminator may be overpowering Generator.\n"
            f"     This is common in early training. If it persists, consider:\n"
            f"     • Reducing Discriminator learning rate\n"
            f"     • Increasing d_update_ratio (train D less frequently)")
    else:
        print(f"  ✅ D_loss ({last_d[-1]:.4f}) looks healthy")

    # 3. Check G_loss trend
    gl_first_half = gan["G_loss"][:len(gan["G_loss"]) // 2].mean()
    gl_second_half = gan["G_loss"][len(gan["G_loss"]) // 2:].mean()
    if gl_second_half < gl_first_half:
        print(
            f"  ✅ G_loss is decreasing ({gl_first_half:.2f} → {gl_second_half:.2f})")
    else:
        warnings.append(
            f"⚠️  G_loss is NOT decreasing ({gl_first_half:.2f} → {gl_second_half:.2f})")

    # 4. Check D/G ratio
    ratio = gan["D_loss"] / (gan["G_loss"] + 1e-8)
    last_ratio = smooth(ratio, 30)[-1]
    if last_ratio < 0.01:
        warnings.append(
            f"⚠️  D/G ratio very low ({last_ratio:.4f}) — D may be collapsing.\n"
            f"     The Discriminator is providing poor gradients to the Generator.")
    elif last_ratio > 0.1:
        warnings.append(
            f"⚠️  D/G ratio high ({last_ratio:.4f}) — D is dominating training.")
    else:
        print(f"  ✅ D/G ratio ({last_ratio:.4f}) within reasonable range")

    # 5. Check D_support vs D_main balance
    ds_final = smooth(gan["D_support_loss"], 30)[-1]
    dm_final = smooth(gan["D_main_loss"], 30)[-1]
    print(f"  📌 D_support ({ds_final:.4f}) vs D_main ({dm_final:.4f}): "
          f"ratio = {ds_final / (dm_final + 1e-8):.1f}x")

    # 6. Check style loss dominance
    sty_final = smooth(gan["sty_loss"], 30)[-1]
    con_final = smooth(gan["con_loss"], 30)[-1]
    if sty_final > 5 * con_final:
        warnings.append(
            f"⚠️  Style loss ({sty_final:.3f}) >> Content loss ({con_final:.3f}).\n"
            f"     Style may be overpowering content preservation.")
    else:
        print(
            f"  ✅ Style/Content balance: sty={sty_final:.3f}, con={con_final:.3f}")

    # 7. Check s44 dominance in style
    s22_f = smooth(gan["s22"], 30)[-1]
    s33_f = smooth(gan["s33"], 30)[-1]
    s44_f = smooth(gan["s44"], 30)[-1]
    print(
        f"  📌 Style layers: s22={s22_f:.4f}, s33={s33_f:.4f}, s44={s44_f:.4f}")
    if s44_f > 3 * (s22_f + s33_f):
        warnings.append(
            f"⚠️  s44 dominates style loss — deep VGG features are driving style.\n"
            f"     This means the Generator focuses on high-level patterns over textures.")

    # 8. Pre-training convergence
    pt_final = smooth(pretrain["g_loss"], 30)[-1]
    if pt_final < 0.25:
        print(f"  ✅ Pre-training converged well (final={pt_final:.4f})")
    elif pt_final < 0.4:
        print(f"  ✅ Pre-training converged reasonably (final={pt_final:.4f})")
    else:
        warnings.append(
            f"⚠️  Pre-training loss is still high ({pt_final:.4f}). "
            f"Consider more pre-train epochs.")

    # Print all issues and warnings
    for issue in issues:
        print(f"\n  {issue}")
    for warn in warnings:
        print(f"\n  {warn}")

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
    print(f"  GAN D_loss:        {gan['D_loss'][0]:.4f} → {last_d[-1]:.4f}")
    print(
        f"  GAN G_loss:        {gan['G_loss'][0]:.2f} → {smooth(gan['G_loss'], 30)[-1]:.2f}")

    if len(issues) == 0 and len(warnings) <= 1:
        print("\n  🎉 Overall: Training looks HEALTHY! Losses are decreasing as expected.")
    elif len(issues) > 0:
        print(
            f"\n  🚨 Overall: {len(issues)} critical issue(s) found. See above.")
    else:
        print(
            f"\n  ⚡ Overall: Training is progressing but has {len(warnings)} warning(s).")

    print("\n  💡 Recommendations:")
    print("     • Continue training for more GAN epochs to further reduce losses")
    print("     • Monitor visual quality of generated images alongside loss curves")
    print("     • If D_loss approaches 0, reduce D learning rate or increase d_update_ratio")
    print("     • Consider saving checkpoints every few epochs for comparison")
    print("=" * 80)


# ─────────────────────────── Main ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AnimeGANv3 Training Log Analyzer")
    parser.add_argument("--log", type=str, default=None,
                        help="Path to train.log (default: logs/train.log)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for charts (default: logs/charts/)")
    args = parser.parse_args()

    # Resolve paths relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = args.log or os.path.join(script_dir, "train.log")
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
    print(f"   Pre-train: {len(pretrain['g_loss'])} steps")
    print(f"   GAN train: {len(gan['G_loss'])} steps")

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
    if len(pretrain['g_loss']) > 0 and len(gan['G_loss']) > 0:
        plot_epoch_summary(pretrain, gan, save_dir)

    # Diagnostics
    print_diagnostics(pretrain, gan)


if __name__ == "__main__":
    main()
