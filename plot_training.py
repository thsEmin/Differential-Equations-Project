"""Plot the measured validation curves used to select saved models."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent / "results"
metrics = json.loads((root / "metrics.json").read_text())
fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
for name, result in metrics["models"].items():
    steps, errors = zip(*result["validation_history"])
    line, = ax.plot(steps, errors, label=name.replace("_", " "))
    selected = result["best_iteration"]
    ax.scatter([selected], [errors[steps.index(selected)]], color=line.get_color(), zorder=3)
ax.axvline(5000, color="gray", linestyle="--", label="Previous 5,000-step budget")
ax.set(xlabel="Training updates (32 maps per batch)", ylabel="Validation temperature RMSE",
       title="Training length selected using 200 validation configurations")
ax.legend()
fig.savefig(root / "training_selection.png", dpi=160)
