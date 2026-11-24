"""Three descriptive plots using the training partition only."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from churn.data import clean_features

TEAL = "#16877c"
CORAL = "#df775c"


def save_eda(features, target, directory):
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "#fafaf7"})
    frame = clean_features(features).assign(Churn=target)
    counts = target.value_counts().reindex([0, 1])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(["Stayed", "Churned"], counts, color=[TEAL, CORAL], width=0.55)
    ax.bar_label(bars, labels=[f"{n:,} ({n/len(target):.1%})" for n in counts], padding=5)
    ax.set(title="Most training customers stayed", ylabel="Customers", ylim=(0, max(counts)*1.18))
    fig.tight_layout()
    fig.savefig(directory / "target_balance.png", dpi=160)
    plt.close(fig)

    rates = frame.groupby("Contract").Churn.mean().reindex(["Month-to-month", "One year", "Two year"])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(rates.index, rates*100, color=[CORAL, TEAL, TEAL], width=0.55)
    ax.bar_label(bars, fmt="%.1f%%", padding=5)
    ax.set(title="Churn rate differs by contract", ylabel="Churn rate (%)", ylim=(0, max(rates)*120))
    fig.tight_layout()
    fig.savefig(directory / "churn_by_contract.png", dpi=160)
    plt.close(fig)

    frame["TenureBand"] = pd.cut(frame.tenure, [-1, 12, 24, 48, 72],
                                 labels=["0–12", "13–24", "25–48", "49–72"])
    tenure = frame.groupby("TenureBand", observed=True).Churn.mean()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(tenure.index.astype(str), tenure*100, color=TEAL, width=0.55)
    ax.bar_label(bars, fmt="%.1f%%", padding=5)
    ax.set(title="Newer customers churn more often", xlabel="Tenure (months)",
           ylabel="Churn rate (%)", ylim=(0, max(tenure)*120))
    fig.tight_layout()
    fig.savefig(directory / "churn_by_tenure.png", dpi=160)
    plt.close(fig)
    return {"partition": "training only", "rows": len(frame),
            "contract_churn_rates": rates.to_dict(),
            "tenure_churn_rates": {str(k): v for k, v in tenure.items()}}
