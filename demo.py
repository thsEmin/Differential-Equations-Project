"""Interactive heat maps for the solver and the physics-informed network."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

from heat import DT, STEPS, simulate
from model import Network


def main():
    path = Path("results") / "physics_informed.npz"
    if not path.exists():
        raise SystemExit("Train first: python run.py")
    model = Network.load(path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.subplots_adjust(bottom=0.28)
    labels = ("Heat equation solver", "Equation-informed network")
    images = []
    for ax, label in zip(axes, labels):
        image = ax.imshow(np.zeros((24, 24)), origin="lower", extent=(0, 1, 0, 1), cmap="inferno")
        ax.set_title(label)
        images.append(image)
    sliders = []
    for i, (name, initial, low, high) in enumerate((
        ("Heater X", 0.5, 0.18, 0.82),
        ("Heater Y", 0.5, 0.18, 0.82),
        ("Strength", 1.0, 0.6, 1.4),
        ("Time", STEPS * DT, 0.0, STEPS * DT),
    )):
        ax = fig.add_axes((0.15, 0.19 - i * 0.045, 0.72, 0.025))
        sliders.append(Slider(ax, name, low, high, valinit=initial))

    def refresh(_=None):
        x, y, strength, time = (slider.val for slider in sliders)
        step = min(STEPS, round(time / DT))
        actual = simulate(x, y, strength, step)[-1]
        informed = model.predict(x, y, strength, step * DT)
        maps = (actual, informed)
        low = min(float(m.min()) for m in maps)
        high = max(float(m.max()) for m in maps)
        for image, array in zip(images, maps):
            image.set_data(array)
            image.set_clim(low, high)
        fig.canvas.draw_idle()

    for slider in sliders:
        slider.on_changed(refresh)
    refresh()
    plt.show()


if __name__ == "__main__":
    main()
