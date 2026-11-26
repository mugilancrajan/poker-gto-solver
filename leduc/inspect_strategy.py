import json
from pathlib import Path

import numpy as np
from leduc.evaluator import compute_ev, compute_exploitability
from leduc.game import get_legal_actions
from leduc.solver import LeducCFRPlus

PROFILE_PATH = Path("leduc_gto_profile.json")
ACTION_LABELS = {"f": "Fold", "c": "Check/Call", "r": "Bet/Raise"}


def train_and_save(iterations: int = 3_000, path: Path = PROFILE_PATH) -> dict[str, np.ndarray]:
    """Train LeducCFRPlus, save the 288-infoset GTO profile to JSON, and return it."""
    print(f"Training LeducCFRPlus for {iterations:,} iterations...")
    solver = LeducCFRPlus()
    solver.train(iterations=iterations)
    profile = solver.get_average_strategy_profile()

    serializable = {key: probs.tolist() for key, probs in profile.items()}
    path.write_text(json.dumps(serializable, indent=2))
    print(f"Saved {len(profile)} infosets to {path}\n")
    return profile


def load_profile(path: Path = PROFILE_PATH) -> dict[str, np.ndarray]:
    """Load a cached Leduc GTO strategy profile from JSON."""
    raw = json.loads(path.read_text())
    return {key: np.array(probs, dtype=np.float64) for key, probs in raw.items()}


def _format_strategy(infoset_key: str, profile: dict[str, np.ndarray]) -> str:
    """Format an infoset's mixed strategy as readable percentages."""
    history = infoset_key.split("|")[-1]
    actions = get_legal_actions(history)
    probs = profile[infoset_key]
    parts = [f"{ACTION_LABELS[a]}: {p * 100:5.1f}%" for a, p in zip(actions, probs)]
    return " | ".join(parts)


def print_strategy_report(profile: dict[str, np.ndarray]) -> None:
    """Print exact metrics and key emergent GTO spots."""
    ev = compute_ev(profile)
    expl = compute_exploitability(profile)

    print("=== Leduc Hold'em GTO Baseline ===")
    print(f"Infosets:       {len(profile)}")
    print(f"Exact P1 EV:    {ev:.5f}  (Nash target: -0.08561)")
    print(f"Exploitability: {expl:.5f}  (chips/hand)\n")

    sections = [
        (
            "1. Round 1 — P1 Opening Action (First to Act)",
            ["J|", "Q|", "K|"],
        ),
        (
            "2. Round 1 — P2 Facing a P1 Bet ('r')",
            ["J|r", "Q|r", "K|r"],
        ),
        (
            "3. Round 2 — P1 First to Act After Check-Check on a KING Board ('cc/')",
            ["J|K|cc/", "Q|K|cc/", "K|K|cc/"],
        ),
        (
            "4. Round 2 — P1 First to Act After Check-Check on a JACK Board ('cc/')",
            ["J|J|cc/", "Q|J|cc/", "K|J|cc/"],
        ),
    ]

    for title, keys in sections:
        print(title)
        print("-" * len(title))
        for key in keys:
            print(f"  {key:<10} ->  {_format_strategy(key, profile)}")
        print()


if __name__ == "__main__":
    if PROFILE_PATH.exists():
        print(f"Loaded cached profile from {PROFILE_PATH}\n")
        profile = load_profile()
    else:
        profile = train_and_save(iterations=3_000)

    print_strategy_report(profile)