import random
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from leduc.evaluator import compute_ev
from leduc.experiments import evaluate_leak
from leduc.game import apply_action, get_current_player, get_legal_actions
from leduc.inspect_strategy import load_profile

st.set_page_config(page_title="Leduc GTO Solver", page_icon="♠️", layout="wide")

# --- CUSTOM CSS FOR CARDS & ANIMATIONS ---
st.markdown(
    """
    <style>
    @keyframes popIn {
        0% { transform: scale(0.85) translateY(8px); opacity: 0; }
        100% { transform: scale(1) translateY(0); opacity: 1; }
    }
    .poker-table {
        background: radial-gradient(circle at center, #14532d 0%, #052e16 100%);
        border: 4px solid #78350f;
        border-radius: 20px;
        padding: 24px 20px;
        margin: 10px 0 18px 0;
        box-shadow: inset 0 0 30px rgba(0,0,0,0.45);
        color: #f8fafc;
    }
    .card-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 6px;
    }
    .card-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #cbd5e1;
        font-weight: 600;
    }
    .poker-card {
        width: 82px;
        height: 114px;
        border-radius: 10px;
        background: #ffffff;
        color: #0f172a;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        padding: 8px 10px;
        font-weight: 800;
        font-size: 1.4rem;
        box-shadow: 0 6px 14px rgba(0,0,0,0.35);
        animation: popIn 0.28s ease-out;
        border: 1px solid #e2e8f0;
    }
    .poker-card.red { color: #dc2626; }
    .poker-card.hidden {
        background: repeating-linear-gradient(
            45deg, #1e3a8a, #1e3a8a 8px, #1d4ed8 8px, #1d4ed8 16px
        );
        border: 2px solid #93c5fd;
        justify-content: center;
        align-items: center;
        color: rgba(255,255,255,0.85);
        font-size: 1.6rem;
    }
    .poker-card.empty {
        background: rgba(255,255,255,0.08);
        border: 2px dashed rgba(255,255,255,0.28);
        box-shadow: none;
        justify-content: center;
        align-items: center;
        color: rgba(255,255,255,0.45);
        font-size: 0.9rem;
    }
    .pot-badge {
        background: rgba(0,0,0,0.4);
        border: 1px solid #fbbf24;
        color: #fde68a;
        padding: 8px 18px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.05rem;
        text-align: center;
        margin: 18px auto 0 auto;
        width: fit-content;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("♠️ Leduc Hold'em GTO Solver & Exploitative Lab")
st.caption(
    "288-Infoset CFR+ Nash Equilibrium Engine  •  Exact P1 EV: **-0.08561**  •  Exploitability: **0.00003 chips/hand**"
)

ACTION_LABELS = {"f": "Fold", "c": "Check / Call", "r": "Bet / Raise"}
ACTION_COLORS = {"f": "#ef4444", "c": "#3b82f6", "r": "#10b981"}
RANK_VAL = {"J": 0, "Q": 1, "K": 2}
SUIT_MAP = {"J": ("♣", False), "Q": ("♥", True), "K": ("♠", False)}

LEAK_CONFIG = {
    "Over-Folder (Unpaired Hands)": (
        "over_fold",
        40,
        15,
        "Extra Fold %",
        "When an opponent folds weak hands too often, rigid GTO only gains modestly—while an exploitative strategy prints money by relentlessly pressuring them.",
    ),
    "Calling Station": (
        "calling_station",
        40,
        15,
        "Extra Call %",
        "Against a player who calls too wide, an exploitative strategy cuts out bluffs and hammers strong hands for maximum value.",
    ),
    "No-Bluff Nit (Unpaired J)": (
        "no_bluff",
        100,
        50,
        "Bluff Reduction %",
        "Notice the flat blue line: rigid GTO captures almost zero extra value against a player who never bluffs, because equilibrium play is designed to make them indifferent.",
    ),
}


@st.cache_data
def get_gto_profile():
    return load_profile()


@st.cache_data
def get_leak_curve(leak_type: str, max_pct: int):
    profile = get_gto_profile()
    grid = np.linspace(0.0, max_pct / 100.0, 9)
    gto_evs, expl_evs = [], []
    for s in grid:
        res = evaluate_leak(profile, leak_type, float(s))
        gto_evs.append(res["gto_ev"])
        expl_evs.append(res["expl_ev"])
    return grid * 100, gto_evs, expl_evs


def format_history_human(history: str) -> str:
    """Translate compact history like 'cr/r' into a readable betting sequence."""
    if history == "":
        return "Start of Hand (P1 to act)"

    rounds = history.split("/")
    parts = []
    for r_idx, r_hist in enumerate(rounds):
        prefix = f"R{r_idx + 1}"
        if not r_hist:
            parts.append(f"{prefix}: Start")
            continue
        facing_bet = False
        moves = []
        for ch in r_hist:
            if ch == "c":
                moves.append("Call" if facing_bet else "Check")
                facing_bet = False
            elif ch == "r":
                moves.append("Raise" if facing_bet else "Bet")
                facing_bet = True
            elif ch == "f":
                moves.append("Fold")
        parts.append(f"{prefix}: " + " → ".join(moves))
    return "  |  ".join(parts)


def render_card_html(label: str, card: str | None, hidden: bool = False) -> str:
    if hidden:
        inner = '<div class="poker-card hidden">♠</div>'
    elif card is None:
        inner = '<div class="poker-card empty">Round 2</div>'
    else:
        suit, is_red = SUIT_MAP[card]
        cls = "poker-card red" if is_red else "poker-card"
        inner = (
            f'<div class="{cls}">'
            f'<div>{card}</div>'
            f'<div style="text-align:center; font-size:1.8rem;">{suit}</div>'
            f'<div style="text-align:right;">{card}</div>'
            f"</div>"
        )
    return f'<div class="card-wrap"><div class="card-label">{label}</div>{inner}</div>'


def _pot_and_p1_payoff(cards: tuple[str, str, str], history: str) -> tuple[int, int]:
    p1_inv, p2_inv = 1, 1
    folder = None

    for round_idx, round_hist in enumerate(history.split("/")):
        bet_size = 2 if round_idx == 0 else 4
        to_call = 0
        actor = 0
        for act in round_hist:
            if act == "f":
                folder = actor
                break
            elif act == "c":
                if actor == 0:
                    p1_inv += to_call
                else:
                    p2_inv += to_call
                to_call = 0
            elif act == "r":
                put_in = to_call + bet_size
                if actor == 0:
                    p1_inv += put_in
                else:
                    p2_inv += put_in
                to_call = bet_size
            actor = 1 - actor

    total_pot = p1_inv + p2_inv
    if get_legal_actions(history):
        return total_pot, 0

    if folder == 0:
        return total_pot, -p1_inv
    if folder == 1:
        return total_pot, p2_inv

    p1_card, p2_card, board = cards
    p1_strength = (1 if p1_card == board else 0, RANK_VAL[p1_card])
    p2_strength = (1 if p2_card == board else 0, RANK_VAL[p2_card])
    if p1_strength > p2_strength:
        return total_pot, p2_inv
    if p2_strength > p1_strength:
        return total_pot, -p1_inv
    return total_pot, 0


def _start_new_hand(human_seat: int, gto_profile: dict):
    deck = ["J", "J", "Q", "Q", "K", "K"]
    random.shuffle(deck)
    st.session_state.cards = (deck[0], deck[1], deck[2])
    st.session_state.human_seat = human_seat
    st.session_state.history = ""
    st.session_state.log = []
    st.session_state.bot_decisions = []
    st.session_state.hand_settled = False
    _advance_bot_turns(gto_profile, animate=False)


def _advance_bot_turns(gto_profile: dict, animate: bool = True):
    while True:
        hist = st.session_state.history
        actions = get_legal_actions(hist)
        if not actions:
            if not st.session_state.hand_settled:
                _, p1_payoff = _pot_and_p1_payoff(st.session_state.cards, hist)
                human_payoff = p1_payoff if st.session_state.human_seat == 0 else -p1_payoff
                st.session_state.bankroll += human_payoff
                st.session_state.hands_played += 1
                st.session_state.last_payoff = human_payoff
                st.session_state.hand_settled = True
            break

        actor = get_current_player(hist)
        if actor == st.session_state.human_seat:
            break

        if animate:
            with st.spinner("GTO Bot is sampling its mixed strategy..."):
                time.sleep(0.35)

        bot_card = st.session_state.cards[actor]
        board = st.session_state.cards[2]
        key = f"{bot_card}|{board}|{hist}" if "/" in hist else f"{bot_card}|{hist}"
        probs = gto_profile[key]
        chosen = str(np.random.choice(actions, p=probs))

        prob_str = " | ".join(f"{ACTION_LABELS[a]}: {p*100:.0f}%" for a, p in zip(actions, probs))
        st.session_state.bot_decisions.append((format_history_human(hist), ACTION_LABELS[chosen], prob_str))
        st.session_state.log.append(f"🤖 **Bot** → {ACTION_LABELS[chosen]}")
        st.session_state.history = apply_action(hist, chosen)


gto_profile = get_gto_profile()
baseline_ev = compute_ev(gto_profile)

tab1, tab2, tab3 = st.tabs([
    "📊 Exploitative 'What-If' Lab",
    "🧠 288-Infoset Strategy Explorer",
    "🃏 Play vs. GTO Bot",
])

# --- TAB 1: EXPLOITATIVE WHAT-IF LAB ---
with tab1:
    st.markdown(
        "Test how much value **rigid GTO** leaves on the table when facing a predictable human leak, "
        "compared to switching to a **Maximally Exploitative** counter-strategy."
    )
    col_ctrl, col_viz = st.columns([1, 2.1], gap="large")

    with col_ctrl:
        leak_label = st.selectbox("1. Choose Opponent Leak", list(LEAK_CONFIG.keys()))
        leak_type, max_pct, default_pct, xlabel, insight = LEAK_CONFIG[leak_label]
        strength_pct = st.slider(f"2. Leak Severity ({xlabel})", 0, max_pct, default_pct, step=5)

        res = evaluate_leak(gto_profile, leak_type, strength_pct / 100.0)

        st.divider()
        c_m1, c_m2 = st.columns(2)
        c_m1.metric("Rigid GTO EV", f"{res['gto_ev']:+.4f}", f"{res['gto_ev'] - baseline_ev:+.4f} vs base")
        c_m2.metric("Exploitative EV", f"{res['expl_ev']:+.4f}")
        st.metric(
            "Value Left on Table",
            f"{res['left_on_table']:+.4f} chips / hand",
            help="Exploitative EV minus Rigid GTO EV.",
        )
        st.info(f"**Takeaway:** {insight}")

    with col_viz:
        x_pct, gto_evs, expl_evs = get_leak_curve(leak_type, max_pct)
        fig = go.Figure()
        fig.add_hline(
            y=baseline_ev,
            line_dash="dash",
            line_color="gray",
            annotation_text="GTO Baseline (-0.0856)",
            annotation_position="bottom right",
        )
        fig.add_trace(
            go.Scatter(
                x=x_pct,
                y=gto_evs,
                mode="lines+markers",
                name="Rigid GTO EV",
                line=dict(color="#2563eb", width=3),
                hovertemplate="%{x:.0f}% leak<br>Rigid GTO: %{y:+.4f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_pct,
                y=expl_evs,
                mode="lines+markers",
                name="Exploitative EV",
                line=dict(color="#dc2626", width=3),
                fill="tonexty",
                fillcolor="rgba(248, 113, 113, 0.18)",
                hovertemplate="%{x:.0f}% leak<br>Exploitative: %{y:+.4f}<extra></extra>",
            )
        )
        fig.add_vline(
            x=strength_pct,
            line_dash="dot",
            line_color="#111827",
            annotation_text=f"Selected: {strength_pct}%",
            annotation_position="top right",
        )
        fig.update_layout(
            title=f"{leak_label}: Rigid GTO vs. Exploitative EV",
            xaxis_title=xlabel,
            yaxis_title="Player 1 EV (chips / hand)",
            hovermode="x unified",
            height=460,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: 288-INFOSET STRATEGY EXPLORER ---
with tab2:
    st.markdown(
        "Inspect the exact **Nash Equilibrium mixed strategy** for Jack, Queen, and King at any decision point."
    )
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        round_choice = st.radio("Betting Round", ["Round 1", "Round 2"], horizontal=True)
    with c2:
        board_card = st.selectbox("Public Board Card", ["J", "Q", "K"], disabled=(round_choice == "Round 1"))

    if round_choice == "Round 1":
        histories = sorted({k.split("|")[-1] for k in gto_profile if len(k.split("|")) == 2})
    else:
        histories = sorted({k.split("|")[-1] for k in gto_profile if len(k.split("|")) == 3})

    with c3:
        history = st.selectbox("Game State / Betting Sequence", histories, format_func=format_history_human)

    actions = get_legal_actions(history)
    hand_labels = []
    prob_matrix = {a: [] for a in actions}
    table_rows = []

    for card in ["J", "Q", "K"]:
        key = f"{card}|{history}" if round_choice == "Round 1" else f"{card}|{board_card}|{history}"
        probs = gto_profile[key]
        label = f"{card} (Pair!)" if (round_choice == "Round 2" and card == board_card) else card
        hand_labels.append(label)
        row = {"Hand": label}
        for a, p in zip(actions, probs):
            prob_matrix[a].append(p * 100)
            row[ACTION_LABELS[a]] = f"{p * 100:.1f}%"
        table_rows.append(row)

    fig_bar = go.Figure()
    for a in actions:
        fig_bar.add_trace(
            go.Bar(
                y=hand_labels,
                x=prob_matrix[a],
                name=ACTION_LABELS[a],
                orientation="h",
                marker_color=ACTION_COLORS[a],
                text=[f"{v:.0f}%" if v >= 8 else "" for v in prob_matrix[a]],
                textposition="inside",
                hovertemplate="%{y} — " + ACTION_LABELS[a] + ": %{x:.1f}%<extra></extra>",
            )
        )
    fig_bar.update_layout(
        barmode="stack",
        xaxis=dict(title="Action Frequency (%)", range=[0, 100]),
        yaxis=dict(autorange="reversed"),
        height=280,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander("View Exact Probability Table"):
        st.dataframe(pd.DataFrame(table_rows).set_index("Hand"), use_container_width=True)

# --- TAB 3: PLAY VS. GTO BOT ---
with tab3:
    if "bankroll" not in st.session_state:
        st.session_state.bankroll = 0
        st.session_state.hands_played = 0
        _start_new_hand(human_seat=0, gto_profile=gto_profile)

    top_l, top_m, top_r = st.columns([2, 1, 1])
    with top_l:
        seat_label = st.radio("Your Seat", ["Player 1 (First to Act)", "Player 2 (In Position)"], horizontal=True)
        desired_seat = 0 if "Player 1" in seat_label else 1
    with top_m:
        st.metric("Session Net", f"{st.session_state.bankroll:+d} chips", f"{st.session_state.hands_played} hands")
    with top_r:
        if st.button("🔄 Deal New Hand", type="primary", use_container_width=True):
            _start_new_hand(desired_seat, gto_profile)
            st.rerun()

    hist = st.session_state.history
    cards = st.session_state.cards
    human_seat = st.session_state.human_seat
    bot_seat = 1 - human_seat
    pot, _ = _pot_and_p1_payoff(cards, hist)
    is_r2 = "/" in hist
    legal = get_legal_actions(hist)

    # Visual Poker Table
    table_html = f"""
    <div class="poker-table">
        <div style="display:flex; justify-content:space-around; align-items:center; flex-wrap:wrap; gap:20px;">
            {render_card_html("Your Hand", cards[human_seat])}
            {render_card_html("Board Card", cards[2] if is_r2 else None)}
            {render_card_html("GTO Bot", cards[bot_seat], hidden=bool(legal))}
        </div>
        <div class="pot-badge">🪙 Current Pot: {pot} chips  •  {format_history_human(hist)}</div>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    if legal:
        btn_cols = st.columns(len(legal))
        for col, act in zip(btn_cols, legal):
            icon = {"f": "🏳️", "c": "✅", "r": "🔥"}[act]
            if col.button(f"{icon} {ACTION_LABELS[act]}", use_container_width=True, key=f"act_{act}"):
                st.session_state.log.append(f"🧑 **You** → {ACTION_LABELS[act]}")
                st.session_state.history = apply_action(hist, act)
                _advance_bot_turns(gto_profile, animate=True)
                st.rerun()
    else:
        payoff = st.session_state.last_payoff
        if payoff > 0:
            st.success(f"🎉 You won **+{payoff} chips**!")
        elif payoff < 0:
            st.error(f"🤖 GTO Bot won **+{abs(payoff)} chips**.")
        else:
            st.info("🤝 Split pot (**0 chips**).")

        if st.session_state.bot_decisions:
            with st.expander("🔍 Reveal Bot's GTO Mixed Strategy This Hand", expanded=True):
                for spot, chosen, prob_str in st.session_state.bot_decisions:
                    st.markdown(f"- **{spot}** → Bot chose **{chosen}** (`{prob_str}`)")

    if st.session_state.log:
        st.caption("Action Log: " + "  →  ".join(st.session_state.log))