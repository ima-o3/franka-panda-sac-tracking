"""Combine matched feedback-only and feedforward runs, without rerunning physics."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT/"results/reference")
    parser.add_argument("--feedforward", type=Path, default=ROOT/"results/ik_feedforward")
    parser.add_argument("--output", type=Path, default=ROOT/"results/comparison")
    args = parser.parse_args()
    manifests = [json.loads((p/"manifest.json").read_text()) for p in (args.reference, args.feedforward)]
    for key in ("duration", "episodes", "seed", "frame_skip", "obs_std", "action_std", "conditions", "trajectories"):
        if manifests[0]["arguments"][key] != manifests[1]["arguments"][key]:
            raise ValueError(f"Cannot combine unmatched {key}")
    for key in ("assets_sha256", "trajectory_config", "control_dt_s", "source_sha256"):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError(f"Cannot combine unmatched {key}")
    if manifests[0]["arguments"]["ik_feedforward"] or not manifests[1]["arguments"]["ik_feedforward"]:
        raise ValueError("Expected feedback-only reference and feedforward supplement")
    args.output.mkdir(parents=True, exist_ok=False)
    tables = []
    for run in (args.reference, args.feedforward):
        with (run/"summary.csv").open(newline="") as file:
            tables.append(list(csv.DictReader(file)))
    index = {(r["trajectory"],r["condition"],r["controller"]):r for r in tables[0]}
    ff = {(r["trajectory"],r["condition"]):r for r in tables[1] if r["controller"]=="ik"}
    lines = ["| Trajectory | Condition | IK feedback | IK + feedforward | Clean SAC | SAC candidate* |",
             "|---|---|---:|---:|---:|---:|"]
    keys = list(ff)
    for kind, condition in keys:
        rs = [index[(kind,condition,"ik")], ff[(kind,condition)],
              index[(kind,condition,"clean_sac")], index[(kind,condition,"candidate_sac")]]
        values = [f"{100*float(r['mean_error_m']):.2f} ± {100*float(r['mean_error_m_sd']):.2f}" for r in rs]
        lines.append(f"| {kind} | {condition} | " + " | ".join(values) + " |")
    lines += ["", "Mean Cartesian error in cm ± sample SD across episode means. *Training provenance unverified.",
        "Assembly SAC is zero-shot; IK feedforward receives analytic reference velocity."]
    (args.output/"table.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    labels = ["IK feedback", "IK + feedforward", "Clean SAC", "SAC candidate (unverified)"]
    colors = ["#0072B2", "#56B4E9", "#D55E00", "#009E73"]
    seed = manifests[0]["arguments"]["seed"]
    fig, axes = plt.subplots(2,2,figsize=(11,7),layout="constrained")
    for j,kind in enumerate(("circle", "assembly")):
        traces = [np.load(run/"traces"/f"{kind}_clean_{c}_{seed}.npz") for run,c in
            ((args.reference,"ik"),(args.feedforward,"ik"),(args.reference,"clean_sac"),(args.reference,"candidate_sac"))]
        axes[0,j].plot(traces[0]["target"][:,0],traces[0]["target"][:,1],"k--",label="Target")
        for tr,label,color in zip(traces,labels,colors):
            axes[0,j].plot(tr["actual"][:,0],tr["actual"][:,1],color=color,label=label,linewidth=1.2)
            axes[1,j].plot(tr["time"],100*tr["error"],color=color,label=label,linewidth=1)
        axes[0,j].set(title=kind.title()+" / clean",xlabel="x [m]",ylabel="y [m]")
        axes[0,j].set_aspect("equal", adjustable="datalim")
        axes[1,j].set(xlabel="Time [s]",ylabel="Tracking error [cm]")
        for ax in axes[:,j]:
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.savefig(args.output/"clean_comparison.png",dpi=200)
    fig.savefig(args.output/"clean_comparison.pdf")
    plt.close(fig)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
