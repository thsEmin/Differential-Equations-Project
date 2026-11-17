"""A small, dimensionless 2D heat-equation simulator."""

import numpy as np

N = 24
ALPHA = 0.05
DT = 0.003
STEPS = 100
AMBIENT = 20.0
X = np.linspace(0.0, 1.0, N)
XX, YY = np.meshgrid(X, X)
DX = X[1] - X[0]


def source(x_heater, y_heater, strength):
    """Heater power per unit time; strength is dimensionless."""
    return strength * 20.0 * np.exp(-((XX - x_heater) ** 2 + (YY - y_heater) ** 2) / (2 * 0.08**2))


def laplacian(field):
    out = np.zeros_like(field)
    out[1:-1, 1:-1] = (
        field[2:, 1:-1] + field[:-2, 1:-1]
        + field[1:-1, 2:] + field[1:-1, :-2]
        - 4 * field[1:-1, 1:-1]
    ) / DX**2
    return out


def simulate(x_heater, y_heater, strength, steps=STEPS):
    """Explicit finite differences; boundary remains at ambient temperature."""
    rise = np.zeros((N, N), dtype=np.float64)
    power = source(x_heater, y_heater, strength)
    frames = [rise.copy()]
    for _ in range(steps):
        rise[1:-1, 1:-1] += DT * (
            ALPHA * laplacian(rise)[1:-1, 1:-1]
            + power[1:-1, 1:-1]
        )
        frames.append(rise.copy())
    return np.asarray(frames) + AMBIENT


def sample_cases(count, seed):
    rng = np.random.default_rng(seed)
    return np.column_stack((
        rng.uniform(0.18, 0.82, count),
        rng.uniform(0.18, 0.82, count),
        rng.uniform(0.6, 1.4, count),
    ))
