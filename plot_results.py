"""Publication-sized plots and descriptive tables generated only from measured data."""
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {"ik": "IK", "clean_sac": "Clean SAC", "candidate_sac": "SAC candidate (unverified)",
          "disturbed_sac": "Disturbance-trained SAC"}
COLORS = {"ik": "#0072B2", "clean_sac": "#D55E00", "candidate_sac": "#009E73", "disturbed_sac": "#CC79A7"}


def make_report(output, rows, examples):
    from evaluate_controllers import save_csv
    plots = output/"plots"
    plots.mkdir()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 200, "axes.grid": True, "grid.alpha": .2})
    groups = defaultdict(list)
    for row in rows:
        groups[(row["trajectory"], row["condition"], row["controller"])].append(row)
    summaries = []
    for (kind, condition, controller), values in groups.items():
        summary = dict(trajectory=kind, condition=condition, controller=controller, episodes=len(values))
        for key in values[0]:
            if key in ("episode", "seed") or not isinstance(values[0][key], (float, int)):
                continue
            a = np.array([r[key] for r in values])
            summary[key] = a.mean()
            summary[key+"_sd"] = a.std(ddof=1) if len(a)>1 else 0.
        summaries.append(summary)
    # IK reward intentionally blank: make the heterogeneous CSV schema explicit.
    fields = list(dict.fromkeys(k for s in summaries for k in s))
    save_csv(output/"summary.csv", [{k: s.get(k, "") for k in fields} for s in summaries])
    def save(fig, name):
        fig.savefig(plots/f"{name}.png", bbox_inches="tight")
        fig.savefig(plots/f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
    kinds = list(dict.fromkeys(r["trajectory"] for r in rows))
    controllers = list(dict.fromkeys(r["controller"] for r in rows))
    conditions = list(dict.fromkeys(r["condition"] for r in rows))
    for kind in kinds:
        for condition in conditions:
            available = [c for c in controllers if (kind,condition,c) in examples]
            if not available:
                continue
            fig = plt.figure(figsize=(7.2, 5))
            ax = fig.add_subplot(111, projection="3d")
            first = examples[(kind,condition,available[0])]
            ax.plot(*first["target"].T, "k--", label="Target", linewidth=2)
            for c in available:
                tr = examples[(kind,condition,c)]
                ax.plot(*tr["actual"].T, label=LABELS[c], color=COLORS[c], linewidth=1.2)
            ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title=f"{kind.title()} / {condition} (first seed)")
            ax.legend(fontsize=8)
            save(fig, f"{kind}_{condition}_trajectory")
            fig, ax = plt.subplots(figsize=(7.2,3.5), layout="constrained")
            for c in available:
                tr = examples[(kind,condition,c)]
                ax.plot(tr["time"], tr["error"]*100, label=LABELS[c], color=COLORS[c], linewidth=1)
            ax.axhline(3, color="gray", linestyle=":", label="3 cm")
            ax.set(xlabel="Time [s]", ylabel="Tracking error [cm]", title=f"{kind.title()} / {condition} (first seed)")
            ax.legend(fontsize=8)
            save(fig, f"{kind}_{condition}_error")
        for metric, label, filename in (("mean_error_m", "Mean tracking error [m]", "performance"),
                ("mean_command_second_difference_rad", "Mean command second difference [rad/step²]", "smoothness")):
            fig, ax = plt.subplots(figsize=(9,4), layout="constrained")
            x = np.arange(len(conditions))
            width = .8/len(controllers)
            for j,c in enumerate(controllers):
                selected = [next(s for s in summaries if (s["trajectory"],s["condition"],s["controller"]) == (kind,cond,c)) for cond in conditions]
                ax.bar(x+(j-(len(controllers)-1)/2)*width, [s[metric] for s in selected], width,
                       yerr=[s[metric+"_sd"] for s in selected], capsize=2, color=COLORS[c], label=LABELS[c])
            ax.set_xticks(x, [c.replace("_", "\n") for c in conditions], fontsize=9)
            ax.set(ylabel=label, title=f"{kind.title()}: episode means ± sample SD")
            ax.legend(fontsize=8)
            save(fig, f"{kind}_{filename}")
    lines = ["# Measured evaluation report", "", "Descriptive episode means ± sample SD; paired seeds, not independent training runs.",
        "Assembly SAC results are zero-shot generalisation. Candidate training provenance is unverified.", "",
        "| Trajectory | Condition | Controller | Mean error (cm) ± SD | RMSE (cm) | Within 3 cm (%) | Command Δ (rad/step) |",
        "|---|---|---|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['trajectory']} | {s['condition']} | {LABELS[s['controller']]} | "
            f"{100*s['mean_error_m']:.2f} ± {100*s['mean_error_m_sd']:.2f} | {100*s['rmse_m']:.2f} | "
            f"{s['within_3cm_pct']:.1f} | {s['mean_command_change_rad']:.5f} |")
    lines += ["", "## Data-driven observations", ""]
    for kind in kinds:
        for condition in conditions:
            subset = [s for s in summaries if s["trajectory"]==kind and s["condition"]==condition]
            best = min(subset, key=lambda s: s["mean_error_m"])
            lines.append(f"- {kind}/{condition}: {LABELS[best['controller']]} has the lowest mean error "
                         f"({100*best['mean_error_m']:.2f} cm) among evaluated controllers.")
    lines += ["", "These rankings are descriptive, not significance tests. For unreachable references, inspect joint/command",
        "limit occupancy, speed, action clipping and command second differences in summary.csv alongside distance.",
        "A small command difference alone can also indicate a stuck controller, not useful tracking."]
    (output/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
