import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from kuhn.evaluator import compute_ev, compute_exploitability
from kuhn.solver import CFRPlus, VanillaCFR


def run_benchmark(total_iterations: int = 20_000) -> dict[str, list[float]]:
    """Run Vanilla CFR and CFR+ side-by-side and log exact metrics at checkpoints."""
    vanilla = VanillaCFR()
    cfr_plus = CFRPlus()

    checkpoints = sorted(
        set(np.unique(np.geomspace(1, total_iterations, num=160).astype(int))) | {1_000}
    )
    checkpoint_set = set(checkpoints)

    history = {
        "iterations": [],
        "vanilla_ev": [],
        "cfr_plus_ev": [],
        "vanilla_expl": [],
        "cfr_plus_expl": [],
        "vanilla_qb": [],
        "cfr_plus_qb": [],
        "cfr_plus_jp": [],
    }

    for i in range(1, total_iterations + 1):
        vanilla.step()
        cfr_plus.step()

        if i in checkpoint_set:
            v_prof = vanilla.get_average_strategy_profile()
            p_prof = cfr_plus.get_average_strategy_profile()

            history["iterations"].append(i)
            history["vanilla_ev"].append(compute_ev(v_prof))
            history["cfr_plus_ev"].append(compute_ev(p_prof))
            history["vanilla_expl"].append(compute_exploitability(v_prof))
            history["cfr_plus_expl"].append(compute_exploitability(p_prof))
            history["vanilla_qb"].append(v_prof["Qb"][1])
            history["cfr_plus_qb"].append(p_prof["Qb"][1])
            history["cfr_plus_jp"].append(p_prof["Jp"][1])

    return history


def _format_iters(x, _pos):
    """Format log-scale iteration ticks as 1, 10, 100, 1k, 10k."""
    if x >= 1000:
        return f"{int(x / 1000)}k"
    return f"{int(x)}"


def _style_axis(ax):
    """Apply clean, publication-grade styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors="#334155", labelsize=9.5)
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, color="#e2e8f0")
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(_format_iters))


def plot_benchmark_dashboard(
    history: dict[str, list[float]],
    output_path: str = "kuhn_cfr_vs_cfr_plus.png",
):
    """Render and save a 3-panel Vanilla CFR vs. CFR+ benchmark dashboard."""
    iters = np.array(history["iterations"], dtype=float)
    v_ev = np.array(history["vanilla_ev"], dtype=float)
    p_ev = np.array(history["cfr_plus_ev"], dtype=float)
    v_expl = np.array(history["vanilla_expl"], dtype=float)
    p_expl = np.array(history["cfr_plus_expl"], dtype=float)
    v_qb = np.array(history["vanilla_qb"], dtype=float) * 100
    p_qb = np.array(history["cfr_plus_qb"], dtype=float) * 100
    p_jp = np.array(history["cfr_plus_jp"], dtype=float) * 100

    NAVY = "#1d4ed8"
    CRIMSON = "#dc2626"
    EMERALD = "#059669"
    AMBER = "#d97706"
    PURPLE = "#7c3aed"
    SLATE = "#64748b"

    # Compute exact speedup at 1,000 iterations for callout & terminal summary
    idx_1k = int(np.where(iters == 1_000)[0][0])
    speedup_1k = v_expl[idx_1k] / p_expl[idx_1k]
    speedup_final = v_expl[-1] / p_expl[-1]

    fig, axes = plt.subplots(
        1, 3, figsize=(15.5, 5.2), dpi=125, constrained_layout=True
    )
    fig.patch.set_facecolor("white")

    fig.suptitle(
        "Kuhn Poker GTO Solver — Vanilla CFR vs. CFR+ Convergence Benchmark",
        fontsize=15,
        fontweight="bold",
        color="#0f172a",
    )

    # --- Panel 1: Player 1 EV Convergence ---
    ax1 = axes[0]
    nash_ev = -1.0 / 18.0
    ax1.plot(iters, v_ev, color=SLATE, alpha=0.55, linewidth=1.6, label="Vanilla CFR EV")
    ax1.plot(iters, p_ev, color=NAVY, linewidth=2.3, label="CFR+ EV")
    ax1.axhline(
        nash_ev,
        color=CRIMSON,
        linestyle="--",
        linewidth=1.5,
        label=f"Nash Target (-1/18 = {nash_ev:.4f})",
    )
    ax1.set_xscale("log")
    _style_axis(ax1)
    ax1.set_title(
        "1. Expected Value Convergence\nCFR+ locks onto -1/18 with minimal oscillation",
        fontsize=10.5,
        fontweight="bold",
        color="#0f172a",
        pad=10,
    )
    ax1.set_xlabel("Training Iterations (log scale)", fontsize=10, color="#334155")
    ax1.set_ylabel("Player 1 EV (chips / hand)", fontsize=10, color="#334155")
    ax1.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=8.5, loc="upper right")

    ax1.annotate(
        f"CFR+ Final: {p_ev[-1]:.5f}",
        xy=(iters[-1], p_ev[-1]),
        xytext=(-105, 24),
        textcoords="offset points",
        fontsize=8.5,
        fontweight="bold",
        color=NAVY,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#eff6ff", edgecolor=NAVY, lw=0.8),
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.0),
    )

    # --- Panel 2: Exact Exploitability (Log-Log Hero Chart) ---
    ax2 = axes[1]
    ax2.loglog(iters, v_expl, color=SLATE, alpha=0.75, linewidth=1.8, label="Vanilla CFR")
    ax2.loglog(iters, p_expl, color=EMERALD, linewidth=2.4, label="CFR+ (Regret Floor + Linear Avg)")

    anchor_idx = np.searchsorted(iters, 100)
    ref_line = v_expl[anchor_idx] * np.sqrt(iters[anchor_idx] / iters)
    ax2.loglog(
        iters,
        ref_line,
        color=SLATE,
        linestyle=":",
        linewidth=1.4,
        label=r"Theoretical $O(1/\sqrt{T})$ Slope",
    )
    _style_axis(ax2)
    ax2.set_title(
        "2. Exact Exploitability (Log-Log)\nCFR+ crushes Vanilla CFR error by orders of magnitude",
        fontsize=10.5,
        fontweight="bold",
        color="#0f172a",
        pad=10,
    )
    ax2.set_xlabel("Training Iterations (log scale)", fontsize=10, color="#334155")
    ax2.set_ylabel("Exploitability (chips / hand)", fontsize=10, color="#334155")
    ax2.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=8.3, loc="lower left")

    ax2.annotate(
        f"{speedup_1k:.0f}x lower error\nat 1k iters",
        xy=(1_000, p_expl[idx_1k]),
        xytext=(-10, 35),
        textcoords="offset points",
        fontsize=8.5,
        fontweight="bold",
        color=EMERALD,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecfdf5", edgecolor=EMERALD, lw=0.8),
        arrowprops=dict(arrowstyle="->", color=EMERALD, lw=1.0),
    )

    # --- Panel 3: Strategy Convergence ---
    ax3 = axes[2]
    ax3.plot(
        iters,
        v_qb,
        color=SLATE,
        alpha=0.5,
        linewidth=1.5,
        linestyle="-",
        label="Vanilla CFR: P2 Queen Call %",
    )
    ax3.plot(iters, p_qb, color=AMBER, linewidth=2.3, label="CFR+: P2 Queen Call %")
    ax3.plot(iters, p_jp, color=PURPLE, linewidth=2.3, label="CFR+: P2 Jack Bluff %")
    ax3.axhline(
        100.0 / 3.0,
        color=CRIMSON,
        linestyle="--",
        linewidth=1.5,
        label="Nash Target (33.3%)",
    )
    ax3.set_xscale("log")
    ax3.set_ylim(-4, 104)
    _style_axis(ax3)
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax3.set_title(
        "3. Equilibrium Strategy Convergence\nCFR+ stabilizes at exact 1/3 bluff & call rates",
        fontsize=10.5,
        fontweight="bold",
        color="#0f172a",
        pad=10,
    )
    ax3.set_xlabel("Training Iterations (log scale)", fontsize=10, color="#334155")
    ax3.set_ylabel("Action Frequency", fontsize=10, color="#334155")
    ax3.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=8.2, loc="upper right")

    ax3.annotate(
        f"CFR+: {p_qb[-1]:.2f}%",
        xy=(iters[-1], p_qb[-1]),
        xytext=(-88, -32),
        textcoords="offset points",
        fontsize=8.5,
        fontweight="bold",
        color=AMBER,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fffbeb", edgecolor=AMBER, lw=0.8),
        arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.0),
    )

    plt.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"\nSaved benchmark dashboard to {output_path}")
    print(f"At  1,000 iters — Vanilla: {v_expl[idx_1k]:.6f} | CFR+: {p_expl[idx_1k]:.6f} ({speedup_1k:.1f}x lower)")
    print(f"At 20,000 iters — Vanilla: {v_expl[-1]:.6f} | CFR+: {p_expl[-1]:.6f} ({speedup_final:.1f}x lower)")
    plt.show()


if __name__ == "__main__":
    history = run_benchmark(total_iterations=20_000)
    plot_benchmark_dashboard(history)