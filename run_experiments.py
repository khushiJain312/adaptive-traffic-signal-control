#!/usr/bin/env python3
"""Paired evaluation: fixed-time program vs. the repo's Q-learning controller.

Place this file in the ROOT of the SUMO-RL-Dashboard repository (next to
sumo_runner.py and rl_agent.py) and run, for example:

    export SUMO_HOME=/path/to/sumo/share/sumo
    python run_experiments.py --train-episodes 20 --eval-episodes 10

It writes into ./results/ :
    episodes.csv               one row per evaluated episode
    training.csv               one row per training episode
    results_table.tex          tabular body for the paper (Table II)
    results_learning_curve.pdf learning curve for the paper (Fig. 4)
Copy results_table.tex and results_learning_curve.pdf next to main.tex.

Metrics are computed from SUMO over ALL twelve incoming lanes, not from the six
detectors the agent observes, so the comparison does not reuse the reward.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import statistics
from pathlib import Path

import sumo_runner as sr
try:
    from config import SUMO_CONFIG_PATH
except ImportError:
    SUMO_CONFIG_PATH = Path(__file__).resolve().parent / "sumo" / "RL.sumocfg"
from rl_agent import QLearningAgent

INCOMING_LANES = [
    f"{edge}_{i}"
    for edge in ("-E2Node2_3_WB", "-E2Node2_5_NB", "Node1_2_EB", "Node2_7_SB")
    for i in range(3)
]
DT = 0.1
# two-sided 97.5% Student-t quantiles for df = 1..30 (paired 95% CI)
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]
METRICS = [
    ("mean_halting", "Queue", "{:.2f}"),
    ("mean_wait", "Wait (s)", "{:.1f}"),
    ("mean_travel", "Travel (s)", "{:.1f}"),
    ("throughput", "Throughput", "{:.0f}"),
    ("emerg_travel", "Emerg. (s)", "{:.1f}"),
]


def sumo_binary() -> str:
    found = shutil.which("sumo")
    if found:
        return found
    home = os.environ.get("SUMO_HOME", "")
    for cand in (Path(home) / "bin" / "sumo", Path(home).parent.parent / "bin" / "sumo"):
        if cand.exists():
            return str(cand)
    raise FileNotFoundError("Could not find the 'sumo' binary; set SUMO_HOME or add it to PATH.")


def run_episode(traci, mode: str, agent: QLearningAgent | None, seed: int,
                steps: int, learn: bool) -> dict:
    traci.start([
        sumo_binary(), "-c", str(SUMO_CONFIG_PATH),
        "--step-length", str(DT), "--seed", str(seed),
        "--no-step-log", "true", "--no-warnings", "true",
        "--lateral-resolution", "0", "--waiting-time-memory", "100000",
    ])
    depart, wait_last, vtype = {}, {}, {}
    trips = []  # (travel_time, waiting_time, is_emergency)
    halting_sum = 0.0
    cum_reward = 0.0
    if agent is not None:
        agent.last_switch_step = -agent.min_green_steps
    try:
        for step in range(steps):
            if mode == "rl":
                old = sr.get_state(traci)
                action = agent.choose_action(old)
                sr.apply_action(traci, agent, action, step)
            traci.simulationStep()
            if mode == "rl":
                new = sr.get_state(traci)
                reward = agent.reward(new)
                cum_reward += reward
                if learn:
                    agent.update(old, action, reward, new)
            now = traci.simulation.getTime()
            for vid in traci.simulation.getDepartedIDList():
                depart[vid] = now
                vtype[vid] = traci.vehicle.getTypeID(vid)
            for vid in traci.vehicle.getIDList():
                wait_last[vid] = traci.vehicle.getAccumulatedWaitingTime(vid)
            for vid in traci.simulation.getArrivedIDList():
                trips.append((now - depart.pop(vid, now), wait_last.pop(vid, 0.0),
                              vtype.pop(vid, "car") == "emergency"))
            halting_sum += sum(traci.lane.getLastStepHaltingNumber(l) for l in INCOMING_LANES)
    finally:
        traci.close()

    emerg = [t for t, _, e in trips if e]
    return {
        "mean_halting": halting_sum / steps,
        "mean_wait": statistics.fmean(w for _, w, _ in trips) if trips else math.nan,
        "mean_travel": statistics.fmean(t for t, _, _ in trips) if trips else math.nan,
        "throughput": len(trips),
        "emerg_travel": statistics.fmean(emerg) if emerg else math.nan,
        "cum_reward": cum_reward,
    }


def mean_std(values):
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return math.nan, math.nan
    return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def paired_ci(rl, fixed):
    diffs = [a - b for a, b in zip(rl, fixed) if not (math.isnan(a) or math.isnan(b))]
    n = len(diffs)
    if n < 2:
        return math.nan, math.nan
    m, s = statistics.fmean(diffs), statistics.stdev(diffs)
    t = T975[n - 2] if n - 1 <= len(T975) else 1.96
    half = t * s / math.sqrt(n)
    return m - half, m + half


def fmt(mean, std, f):
    return "--" if math.isnan(mean) else (f + r"$\pm$" + f).format(mean, std)


def write_table(path: Path, rows_fixed, rows_rl):
    """Writes a tabular in the IEEE template's \\hline style (no extra packages needed)."""
    lines = [r"\begin{tabular}{|l|" + "c|" * len(METRICS) + "}", r"\hline",
             r"\textbf{Controller} & " + " & ".join(r"\textbf{%s}" % h for _, h, _ in METRICS) + r" \\ \hline"]
    fx, rl, delta, ci = [], [], [], []
    for key, _, f in METRICS:
        a = [r[key] for r in rows_fixed]
        b = [r[key] for r in rows_rl]
        ma, sa = mean_std(a)
        mb, sb = mean_std(b)
        fx.append(fmt(ma, sa, f))
        rl.append(fmt(mb, sb, f))
        delta.append("--" if math.isnan(ma) or ma == 0 or math.isnan(mb)
                     else f"{100.0 * (mb - ma) / ma:+.1f}")
        lo, hi = paired_ci(b, a)
        ci.append("--" if math.isnan(lo) else f"[{lo:.2f}, {hi:.2f}]")
    lines += ["Fixed-time & " + " & ".join(fx) + r" \\ \hline",
              "Q-learning & " + " & ".join(rl) + r" \\ \hline",
              r"$\Delta$ (\%) & " + " & ".join(delta) + r" \\ \hline",
              r"95\% CI of diff. & " + " & ".join(ci) + r" \\ \hline",
              r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def write_curve(path: Path, train_rows, fixed_mean):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.plot([r["episode"] for r in train_rows], [r["mean_halting"] for r in train_rows],
            marker="o", ms=3, lw=1, label="Q-learning (training)")
    if not math.isnan(fixed_mean):
        ax.axhline(fixed_mean, ls="--", lw=1, color="gray", label="Fixed-time (eval. mean)")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean halting vehicles")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=10000, help="steps per episode (0.1 s each)")
    p.add_argument("--train-episodes", type=int, default=20)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--train-seed", type=int, default=1000)
    p.add_argument("--eval-seed", type=int, default=5000)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--min-green", type=int, default=100)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sr.prepare_sumo_tools()
    import traci  # noqa: E402  (needs SUMO_HOME/tools on sys.path)

    agent = QLearningAgent(alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon,
                           min_green_steps=args.min_green)

    train_rows = []
    for ep in range(args.train_episodes):
        res = run_episode(traci, "rl", agent, args.train_seed + ep, args.steps, learn=True)
        train_rows.append({"episode": ep + 1, "seed": args.train_seed + ep,
                           "q_states": len(agent.q_table), **res})
        print(f"[train {ep + 1}/{args.train_episodes}] halting={res['mean_halting']:.3f} "
              f"states={len(agent.q_table)}", flush=True)

    agent.epsilon = 0.0  # greedy evaluation, no updates
    fixed_rows, rl_rows = [], []
    for i in range(args.eval_episodes):
        seed = args.eval_seed + i
        fixed_rows.append({"controller": "fixed", "seed": seed,
                           **run_episode(traci, "fixed", None, seed, args.steps, learn=False)})
        rl_rows.append({"controller": "rl", "seed": seed,
                        **run_episode(traci, "rl", agent, seed, args.steps, learn=False)})
        print(f"[eval {i + 1}/{args.eval_episodes}] fixed={fixed_rows[-1]['mean_halting']:.3f} "
              f"rl={rl_rows[-1]['mean_halting']:.3f}", flush=True)

    write_csv(out / "training.csv", train_rows)
    write_csv(out / "episodes.csv", fixed_rows + rl_rows)
    write_table(out / "results_table.tex", fixed_rows, rl_rows)
    fixed_mean, _ = mean_std([r["mean_halting"] for r in fixed_rows])
    write_curve(out / "results_learning_curve.pdf", train_rows, fixed_mean)
    print(f"Done. Copy {out}/results_table.tex and {out}/results_learning_curve.pdf next to main.tex.")


if __name__ == "__main__":
    main()
