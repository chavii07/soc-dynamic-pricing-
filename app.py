"""
app.py (Week 7 -- Streamlit Dashboard & Results Packaging)
-------------------------------------------------------------
Interactive dashboard summarising the whole project: theoretical
benchmarks, baseline tournament, Q-learning mid-project gate, and the
Week 6 self-play / collusion analysis.

Run with:
    streamlit run dashboard/app.py
"""
import json, os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
from PIL import Image

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

st.set_page_config(page_title="Algorithmic Collusion in Dynamic Pricing", layout="wide")
st.title("Competitive Dynamic Pricing: Nash Equilibria, Multi-Agent RL & Collusion Detection")
st.caption("8-week project dashboard -- Bertrand duopoly, rule-based baselines, Q-learning, DQN/PPO self-play")


def load_json(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


tab1, tab2, tab3, tab4 = st.tabs([
    "Theory & Baselines", "Q-Learning (Week 4)", "Deep RL Self-Play (Week 5-6)", "Raw Reports"
])

with tab1:
    st.subheader("Theoretical benchmarks")
    from market import BertrandMarket
    m = BertrandMarket()
    col1, col2, col3 = st.columns(3)
    col1.metric("Nash price", f"{m.nash_price():.2f}")
    col2.metric("Monopoly price", f"{m.monopoly_price():.2f}")
    col3.metric("Price of Anarchy", f"{m.price_of_anarchy():.3f}")

    st.subheader("Week 3 -- Rule-based baseline tournament")
    tournament = load_json("week3_tournament.json")
    if tournament:
        rows = []
        for matchup, res in tournament.items():
            rows.append({"Matchup": matchup, **res})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Run `python -m train.run_baseline_tournament` first.")

with tab2:
    st.subheader("Week 4 -- Mid-project review gate")
    report = load_json("week4_report.json")
    if report:
        col1, col2, col3 = st.columns(3)
        col1.metric("Q-learning vs Random", f"{report['mean_profit_vs_random']:.0f}")
        col2.metric("Random vs Random (baseline)", f"{report['random_baseline_vs_random']:.0f}")
        col3.metric("Gate passed?", "YES" if report["gate_passed"] else "NO")
        st.json(report)
    else:
        st.info("Run `python -m train.train_qlearning` first.")

with tab3:
    st.subheader("Week 6 -- Multi-seed equilibrium & collusion analysis")
    analysis = load_json("week6_analysis_report.json")
    if analysis:
        st.json(analysis)
        for fname, caption in [
            ("price_trajectory.png", "Price trajectory: self-play agents vs. Nash/Monopoly benchmarks"),
            ("profit_boxplot.png", "Profit distribution by algorithm"),
        ]:
            fpath = os.path.join(RESULTS_DIR, fname)
            if os.path.exists(fpath):
                st.image(Image.open(fpath), caption=caption, use_container_width=True)
    else:
        st.info("Run `python -m train.train_deep_rl --algo dqn` and `--algo ppo`, "
                "then `python -m analysis.run_full_analysis`.")

with tab4:
    st.subheader("All raw result files")
    if os.path.exists(RESULTS_DIR):
        for fname in sorted(os.listdir(RESULTS_DIR)):
            st.write(f"- `{fname}`")
    else:
        st.info("No results yet -- run the training/analysis scripts first.")
