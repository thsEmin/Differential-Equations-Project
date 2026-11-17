"""Accuracy by simulation time, including exact initial and edge checks."""
import numpy as np
from heat import AMBIENT, N
from model import make_inputs


def evaluate_times(model, cases, frames):
    report = {}
    for step in (0, 10, 25, 50, 75, 100):
        rise, _ = model.forward(make_inputs(cases, np.full(len(cases), step)))
        reference = frames[:, step] - AMBIENT
        error = rise - reference
        norm = float(np.linalg.norm(reference))
        edge = np.concatenate((rise[:, 0, :].ravel(), rise[:, -1, :].ravel(),
                               rise[:, :, 0].ravel(), rise[:, :, -1].ravel()))
        hot = reference > 0.5 * reference.max(axis=(1, 2), keepdims=True)
        report[str(step)] = {
            "rmse": float(np.sqrt(np.mean(error**2))),
            "relative_rise_error": float(np.linalg.norm(error) / norm) if norm else None,
            "max_error": float(np.max(np.abs(error))),
            "hot_region_rmse": float(np.sqrt(np.mean(error[hot]**2))) if hot.any() else None,
            "edge_max_error": float(np.abs(edge).max()),
        }
        if model.hard_constraints:
            assert np.all(edge == 0)
            if step == 0:
                assert np.all(rise == 0)
    return report
