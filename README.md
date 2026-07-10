# Competitive Dynamic Pricing: Nash Equilibria, Multi-Agent RL & Collusion Detection

An 8-week project studying whether independently learning pricing algorithms
converge to competitive (Nash), collusive (monopoly), or supra-competitive
outcomes in a Bertrand duopoly — the question at the heart of Calvano,
Calzolari, Denicolò & Pastorello (2020), *"Artificial Intelligence,
Algorithmic Pricing, and Collusion,"* QJE.

## TL;DR result

Trained via self-play on the same environment, **DQN converges to a
supra-monopoly price (48.3, above the 46.7 monopoly price)** — a textbook
case of tacit algorithmic collusion — while **PPO converges close to the
competitive Nash price (38.3 vs. 37.5 theoretical)**. This mirrors the
Calvano et al. finding that off-policy, value-based RL is more prone to
supporting tacit collusion than on-policy policy-gradient methods.

![price trajectory](results/price_trajectory.png)

---

## Project structure

```
project/
├── README.md
├── requirements.txt
├── src/
│   ├── market.py              # Bertrand duopoly economics (demand, profit, Nash/monopoly solvers)
│   ├── env.py                 # Custom Gymnasium environment (Week 2)
│   ├── agents/
│   │   ├── rule_based.py      # Random, Always-Nash, Always-Collude, Tit-for-Tat (Week 3)
│   │   └── q_learning.py      # Tabular Q-learning from scratch (Week 4)
│   ├── train/
│   │   ├── run_baseline_tournament.py   # Week 3
│   │   ├── train_qlearning.py           # Week 4 (mid-project review gate)
│   │   └── train_deep_rl.py             # Week 5 (DQN & PPO self-play via SB3)
│   └── analysis/
│       ├── collusion.py               # Week 6, deliverable 1
│       ├── price_of_anarchy.py        # Week 6, deliverable 2
│       └── run_full_analysis.py       # Week 6 runner (stats + plots)
├── dashboard/
│   └── app.py                 # Streamlit results dashboard (Week 7)
└── results/                   # All generated JSON reports, models, and plots
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running the full pipeline (weeks 3 → 7)

Run from inside `src/` so the relative imports resolve:

```bash
cd src

# Week 3 — rule-based baseline tournament (1000+ rounds, all seeds logged)
python -m train.run_baseline_tournament

# Week 4 — tabular Q-learning + MANDATORY mid-project review gate
python -m train.train_qlearning

# Week 5 — deep RL self-play (repeat for both algorithms)
python -m train.train_deep_rl --algo dqn --timesteps 40000 --freeze_every 4000
python -m train.train_deep_rl --algo ppo --timesteps 40000 --freeze_every 4000

# Week 6 — multi-seed equilibrium & collusion analysis (produces plots + report)
python -m analysis.run_full_analysis

# Week 7 — interactive dashboard (run from the project root, not src/)
cd .. && streamlit run dashboard/app.py
```

All outputs (trained models, JSON reports, PNG plots, TensorBoard logs) are
written to `results/`.

---

## Week-by-week notes and design decisions

### Week 1 — Game Theory Foundations
No code this week by design. `market.py`'s `nash_price()` and
`monopoly_price()` implement, in code, exactly the pen-and-paper derivation
this week's resources emphasize: the symmetric best-response fixed point for
Nash, and joint-profit maximization for the collusive benchmark.

### Week 2 — Market Environment Design (`src/env.py`)
- 25 discretized price levels (well above the "5 levels is too coarse"
  pitfall).
- Demand parameters (`a`, `b`, `c`) and marginal cost are fully configurable
  via the injected `BertrandMarket` dataclass — nothing is hard-coded.
- Observations are normalized to `[0, 1]` before being returned.
- **Design note:** default market parameters were tuned (`a=100, b=2, c=0.8,
  mc=10`, price range `[10, 50]`) so that the Nash price (37.5) and monopoly
  price (46.7) both sit safely *inside* the price grid — an earlier
  parameterization pushed both equilibria to the upper price bound, which
  would have silently corrupted every downstream result (the exact failure
  mode this week's resource sheet warns about).

### Week 3 — Rule-Based Baseline Agents (`src/agents/rule_based.py`, `train/run_baseline_tournament.py`)
Four agents in round-robin, 1,000 rounds per matchup, all random seeds fixed
and logged. Sanity-checks the environment: `AlwaysCollude` vs `AlwaysCollude`
earns the highest joint profit; `AlwaysNash` undercuts `AlwaysCollude` and
earns more than it in a mixed matchup — exactly as Bertrand theory predicts.

### Week 4 — Q-Learning Agent (`src/agents/q_learning.py`, `train/train_qlearning.py`)
Tabular Q-learning implemented from scratch (no libraries) with an
explicit Bellman update. **Mid-project review gate:** the agent must beat a
Random baseline before proceeding.

> **Real bug caught during development:** training the agent only against a
> static `AlwaysNash` opponent left most of the (own-price, rival-price)
> state space completely unvisited, so those Q-table entries stayed at zero
> — the agent *looked* trained but its policy was undefined almost
> everywhere. Evaluated against a Random opponent it actually **lost** to a
> purely random agent (43,837 vs. 49,353 mean profit) — the gate failed
> exactly as the resource sheet predicts a buggy setup should. The fix was
> to train against the same (Random) opponent distribution used at
> evaluation time, which fixed state coverage and passed the gate (61,335 vs.
> 49,353). This is documented here rather than silently patched because it's
> the kind of failure the mid-project review is specifically designed to
> catch.

### Week 5 — Deep RL: DQN & PPO + Self-Play (`train/train_deep_rl.py`)
- Stable-Baselines3 `DQN` and `PPO`, environment wrapped in `DummyVecEnv`.
- TensorBoard logging enabled from the first training step
  (`results/tb_logs/`).
- **Self-play non-stationarity fix:** rather than having both agents update
  simultaneously (unstable), the opponent is a *frozen* snapshot of the
  learner's own weights, refreshed every `freeze_every` steps
  (`SelfPlayRefreshCallback`).

### Week 6 — Equilibrium Analysis & Collusion Detection (`src/analysis/`)
Two required deliverables, both implemented as specified on the resource
sheet:
1. **`collusion.py`** — the heuristic from the sheet
   (`mean(prices[-window:]) > nash_p * 1.05`), plus a corrected
   *symmetric* version requiring **both** firms to be persistently
   supra-Nash (the sheet's own pitfall: a single firm's lucky high-price
   streak is not collusion).
2. **`price_of_anarchy.py`** — `PoA = welfare_competitive / welfare_nash`
   (welfare, not profit — the sheet's second pitfall), evaluated across 5
   seeds with a bootstrap 95% CI.

**Findings** (see `results/week6_analysis_report.json` and
`results/price_trajectory.png` / `profit_boxplot.png`):

| | Nash price | Monopoly price | DQN self-play | PPO self-play |
|---|---|---|---|---|
| Mean converged price | 37.5 | 46.7 | **48.3** | 38.3 |
| Supra-Nash fraction (Calvano threshold) | — | — | 100% | 0% |
| Symmetric collusion flagged | — | — | **Yes, all 5 seeds** | No |

DQN's converged policies are deterministic given the trained weights, so the
5-seed evaluation is a genuine robustness check (a different training seed
would be needed to test whether this collusive outcome itself is seed-robust
— see "Limitations" below) rather than a source of episode-to-episode
variance; the reported CI is therefore tight around a single fixed point.

### Week 7 — Streamlit Dashboard & Results Packaging (`dashboard/app.py`)
Four tabs: theoretical benchmarks + Week 3 tournament, Week 4 gate results,
Week 5/6 self-play analysis with plots, and a raw-file browser. Run with
`streamlit run dashboard/app.py` from the project root.

### Week 8 — Final Report
Use `results/week6_analysis_report.json`, the two PNGs, and the table above
as the empirical core of the report. Suggested structure, following how
Calvano et al. frame their own paper (per the Week 3/6 resource notes):
introduction & theory (Week 1) → environment & baselines (Weeks 2–3) →
methods (Weeks 4–5) → results (Week 6, use the table + plots above) →
policy implications, citing the CMA (2021) and OECD (2017) algorithmic
collusion reports → limitations (below).

---

## Limitations / next steps

- **Single training seed per algorithm.** The 5-seed evaluation in Week 6
  tests robustness of the *converged policy* across episodes, not
  robustness of the *training run* itself. A stronger claim ("DQN reliably
  colludes") would require re-running `train_deep_rl.py --algo dqn` with
  5+ different `--seed` values and checking whether collusion emerges in
  most of them — exactly as Calvano et al. do in their own robustness
  section (Sections 4–5, flagged as required Week 6 reading).
- **Two-firm, symmetric-cost duopoly only.** The linear differentiated-demand
  setup was chosen for its closed-form Nash/monopoly solutions; extending to
  3+ firms or asymmetric costs would require the numerical (scipy) solvers
  already used for the monopoly price.
- **Self-play freeze interval was not tuned.** `freeze_every=4000` was a
  reasonable default, not the result of a hyperparameter sweep; the
  Week 5 "Hyperparameter Tuning for RL" resource would be the natural next
  step if training instability is observed at longer horizons.
