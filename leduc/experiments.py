from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from leduc.evaluator import StrategyProfile, compute_best_response_ev, compute_ev
from leduc.game import get_current_player, get_legal_actions
from leduc.inspect_strategy import load_profile


def _shift_action_prob(probs: np.ndarray, action_idx: int, delta: float) -> np.ndarray:
    """
    Shift probs[action_idx] by `delta` (clamped to [0, 1]) and scale
    the remaining actions proportionally so the distribution still sums to 1.
    """
    new_probs = probs.copy()
    new_p = float(np.clip(new_probs[action_idx] + delta, 0.0, 1.0))
    remaining_target = 1.0 - new_p

    other_mask = np.ones(len(probs), dtype=bool)
    other_mask[action_idx] = False
    other_sum = new_probs[other_mask].sum()

    if other_sum > 0:
        new_probs[other_mask] *= remaining_target / other_sum
    else:
        new_probs[other_mask] = remaining_target / (len(probs) - 1)

    new_probs[action_idx] = new_p
    return new_probs


def inject_p2_leak(
    gto_profile: StrategyProfile, leak_type: str, strength: float
) -> StrategyProfile:
    """Return a copy of `gto_profile` where only Player 2's infosets exhibit `leak_type`."""
    leaking_profile: StrategyProfile = {}

    for key, probs in gto_profile.items():
        parts = key.split("|")
        history = parts[-1]
        my_card = parts[0]
        board = parts[1] if len(parts) == 3 else None

        # Keep Player 1 strictly at GTO
        if get_current_player(history) != 1:
            leaking_profile[key] = probs.copy()
            continue

        actions = get_legal_actions(history)
        new_probs = probs.copy()
        is_paired = (board is not None) and (my_card == board)

        if leak_type == "over_fold" and "f" in actions and not is_paired:
            new_probs = _shift_action_prob(new_probs, actions.index("f"), +strength)

        elif leak_type == "calling_station" and "f" in actions and "c" in actions:
            new_probs = _shift_action_prob(new_probs, actions.index("c"), +strength)

        elif leak_type == "no_bluff" and "r" in actions:
            is_unpaired_jack = (my_card == "J") and (board != "J")
            if is_unpaired_jack:
                r_idx = actions.index("r")
                new_probs = _shift_action_prob(new_probs, r_idx, -strength * new_probs[r_idx])

        leaking_profile[key] = new_probs

    return leaking_profile


def evaluate_leak(
    gto_profile: StrategyProfile, leak_type: str, strength: float
) -> dict[str, float]:
    """Compare Player 1's GTO EV vs. Maximally Exploitative EV against a leaking Player 2."""
    leaking_profile = inject_p2_leak(gto_profile, leak_type, strength)
    gto_ev = compute_ev(leaking_profile)
    expl_ev = compute_best_response_ev(0, leaking_profile)
    return {
        "gto_ev": gto_ev,
        "expl_ev": expl_ev,
        "left_on_table": expl_ev - gto_ev,
    }


def run_experiment_suite() -> None:
    gto_profile = load_profile()
    baseline_ev = compute_ev(gto_profile)

    scenarios = [
        ("GTO Baseline (0% leak)", "over_fold", 0.00),
        ("Over-Folder Unpaired (+15%)", "over_fold", 0.15),
        ("Over-Folder Unpaired (+30%)", "over_fold", 0.30),
        ("Calling Station (+15% call)", "calling_station", 0.15),
        ("Calling Station (+30% call)", "calling_station", 0.30),
        ("No-Bluff Nit (-50% J bluffs)", "no_bluff", 0.50),
        ("No-Bluff Nit (-100% J bluffs)", "no_bluff", 1.00),
    ]

    print(f"=== Leduc Exploitative 'What-If' Experiments (P1 Baseline EV: {baseline_ev:+.5f}) ===\n")
    header = f"{'Opponent Profile':<32} | {'GTO EV':>9} | {'Expl EV':>9} | {'Left on Table':>14}"
    print(header)
    print("-" * len(header))

    for label, leak_type, strength in scenarios:
        res = evaluate_leak(gto_profile, leak_type, strength)
        print(
            f"{label:<32} | {res['gto_ev']:>+9.5f} | {res['expl_ev']:>+9.5f} | {res['left_on_table']:>+14.5f}"
        )


def plot_exploit_curves(output_path: Path = Path("leduc_exploit_curves.png")) -> None:
    """Generate a 3-panel chart comparing Rigid GTO EV vs. Maximally Exploitative EV."""
    gto_profile = load_profile()
    baseline_ev = compute_ev(gto_profile)

    panels = [
        ("Over-Folder (Unpaired Hands)", "over_fold", np.linspace(0.0, 0.40, 9), "Extra Fold %"),
        ("Calling Station", "calling_station", np.linspace(0.0, 0.40, 9), "Extra Call %"),
        ("No-Bluff Nit (Unpaired J)", "no_bluff", np.linspace(0.0, 1.00, 11), "Bluff Reduction %"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    fig.suptitle(
        "Leduc Hold'em: Cost of Rigid GTO Against Leaking Opponents",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    for ax, (title, leak_type, grid, xlabel) in zip(axes, panels):
        gto_evs, expl_evs = [], []
        for s in grid:
            res = evaluate_leak(gto_profile, leak_type, float(s))
            gto_evs.append(res["gto_ev"])
            expl_evs.append(res["expl_ev"])

        x_pct = grid * 100
        ax.axhline(baseline_ev, color="gray", linestyle="--", linewidth=1, label="GTO Baseline")
        ax.plot(x_pct, gto_evs, marker="o", linewidth=2, color="#2563eb", label="Rigid GTO EV")
        ax.plot(x_pct, expl_evs, marker="s", linewidth=2, color="#dc2626", label="Exploitative EV")
        ax.fill_between(
            x_pct, gto_evs, expl_evs, color="#f87171", alpha=0.18, label="Value Left on Table"
        )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Player 1 EV (chips / hand)")
    axes[0].legend(loc="upper left", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved Phase 4 chart to {output_path}")


if __name__ == "__main__":
    run_experiment_suite()
    plot_exploit_curves()