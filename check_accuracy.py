"""Evaluate saved networks across unseen cases and times, and save a comparison."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from heat import AMBIENT, DT, sample_cases, simulate
from model import Network

output = Path("results")
models = {name: Network.load(output / f"{name}.npz")
          for name in ("physics_informed",)}
report = {}
for name, model in models.items():
    squared_error = 0.0
    squared_rise = 0.0
    count = 0
    max_error = 0.0
    for case in sample_cases(40, 2026):
        frames = simulate(*case)
        for step in (0, 10, 25, 50, 75, 100):
            reference = frames[step]
            error = model.predict(*case, step * DT) - reference
            squared_error += float(np.sum(error**2))
            squared_rise += float(np.sum((reference - AMBIENT)**2))
            max_error = max(max_error, float(np.abs(error).max()))
            count += error.size
    report[name] = {"rmse": (squared_error / count)**0.5,
                    "relative_temperature_rise_error": (squared_error / squared_rise)**0.5,
                    "maximum_absolute_error": max_error}
(output / "accuracy_check.json").write_text(json.dumps(report, indent=2))
case = (0.5, 0.5, 1.0)
reference = simulate(*case, 50)[-1]
maps = [reference] + [model.predict(*case, 50 * DT) for model in models.values()]
fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
for ax, data, title in zip(axes, maps, ("Heat equation", "Equation-informed model")):
    image = ax.imshow(data, origin="lower", extent=(0, 1, 0, 1), vmin=AMBIENT,
                      vmax=max(float(m.max()) for m in maps), cmap="inferno")
    ax.set_title(title)
fig.colorbar(image, ax=axes, label="Temperature (illustrative degrees)", shrink=0.8)
fig.savefig(output / "accuracy_comparison.png", dpi=140)
print(json.dumps(report, indent=2))
