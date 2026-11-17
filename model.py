"""A two-layer NumPy neural network with hand-written gradients."""

import numpy as np

from heat import ALPHA, AMBIENT, DT, N, source


class Network:
    def __init__(self, seed=0, width=64):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 0.2, (20, width))
        self.b1 = np.zeros(width)
        self.w2 = rng.normal(0, 0.06, (width, N * N))
        self.b2 = np.zeros(N * N)
        self.rng = rng
        self.iteration = 0
        self.moments = {}
        self.hard_constraints = True

    def forward(self, inputs):
        hidden = np.tanh(self.features(inputs) @ self.w1 + self.b1)
        output = hidden @ self.w2 + self.b2
        output = output.reshape(-1, N, N)
        if self.hard_constraints:
            output = output * self.envelope(inputs)
        return output, hidden

    @staticmethod
    def envelope(inputs):
        mask = np.ones((N, N))
        mask[[0, -1], :] = 0
        mask[:, [0, -1]] = 0
        return inputs[:, 3, None, None] * mask

    @staticmethod
    def features(inputs):
        # Smooth spatial features help represent a localized moving heater.
        angles = inputs[:, :2, None] * np.arange(1, 5) * np.pi
        return np.concatenate((inputs * 2 - 1, np.sin(angles).reshape(-1, 8),
                               np.cos(angles).reshape(-1, 8)), axis=1)

    def backward(self, inputs, hidden, grad_output):
        if self.hard_constraints:
            grad_output = grad_output * self.envelope(inputs)
        grad = grad_output.reshape(-1, N * N)
        dh = (grad @ self.w2.T) * (1 - hidden**2)
        return (self.features(inputs).T @ dh, dh.sum(axis=0), hidden.T @ grad, grad.sum(axis=0))

    def update(self, gradients, rate):
        self.iteration += 1
        for name, gradient in zip(("w1", "b1", "w2", "b2"), gradients):
            first, second = self.moments.get(name, (np.zeros_like(gradient), np.zeros_like(gradient)))
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient**2
            self.moments[name] = first, second
            corrected = first / (1 - 0.9**self.iteration)
            variance = second / (1 - 0.999**self.iteration)
            setattr(self, name, getattr(self, name) - rate * corrected / (np.sqrt(variance) + 1e-8))

    def predict(self, x_heater, y_heater, strength, time):
        inp = np.array([[x_heater, y_heater, strength, time / (DT * 100)]])
        rise, _ = self.forward(inp)
        return rise[0] + AMBIENT

    def save(self, path):
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2,
                 hard_constraints=self.hard_constraints)

    @classmethod
    def load(cls, path):
        model = cls()
        with np.load(path) as data:
            model.hard_constraints = bool(data["hard_constraints"]) if "hard_constraints" in data else False
            for name in ("w1", "b1", "w2", "b2"):
                setattr(model, name, data[name])
        return model


def physics_loss_and_gradient(before, after, cases, step_gap):
    """Residual of one finite-difference heat-equation step over a time gap."""
    batch = len(before)
    h = 1 / (N - 1)
    power = np.asarray([source(*case) for case in cases])
    lap = (
        before[:, 2:, 1:-1] + before[:, :-2, 1:-1]
        + before[:, 1:-1, 2:] + before[:, 1:-1, :-2]
        - 4 * before[:, 1:-1, 1:-1]
    ) / h**2
    residual = after[:, 1:-1, 1:-1] - before[:, 1:-1, 1:-1]
    residual -= step_gap * DT * (ALPHA * lap + power[:, 1:-1, 1:-1])
    scale = 2 / (batch * (N - 2) ** 2)
    grad_after = np.zeros_like(after)
    grad_before = np.zeros_like(before)
    g = scale * residual
    grad_after[:, 1:-1, 1:-1] = g
    factor = step_gap * DT * ALPHA / h**2
    grad_before[:, 1:-1, 1:-1] += g * (-1 + 4 * factor)
    grad_before[:, 2:, 1:-1] -= factor * g
    grad_before[:, :-2, 1:-1] -= factor * g
    grad_before[:, 1:-1, 2:] -= factor * g
    grad_before[:, 1:-1, :-2] -= factor * g
    # Normalize the rate residual by the fixed unit-strength source peak.
    # This removes the artificial DT-squared suppression of the old loss.
    normalization = (step_gap * DT * 20.0)**2
    return float(np.mean(residual**2) / normalization), grad_before / normalization, grad_after / normalization


def make_inputs(cases, steps):
    return np.column_stack((cases, np.asarray(steps) / 100))


def train(model, cases, frames, training_steps=30000, physics_weight=0.01, rate=0.001,
          validation=None, batch_size=32, validation_interval=500, patience=8):
    """Supervised snapshots plus equation and boundary penalties."""
    if not np.isfinite(physics_weight) or physics_weight <= 0:
        raise ValueError("Physics weight must be positive and finite")
    count = len(cases)
    history = []
    best_error = float("inf")
    best_parameters = None
    model.validation_history = []
    total_maps = count * frames.shape[1]
    order = model.rng.permutation(total_maps)
    cursor = 0
    seen = 0
    stale = 0
    significant_best = float("inf")
    model.stop_reason = "training_step_limit"
    for update in range(training_steps):
        if cursor == total_maps:
            order = model.rng.permutation(total_maps)
            cursor = 0
        selected = order[cursor:cursor + batch_size]
        cursor += len(selected)
        seen += len(selected)
        ids, steps = np.divmod(selected, frames.shape[1])
        batch_cases = cases[ids]
        inputs = make_inputs(batch_cases, steps)
        target = frames[ids, steps] - AMBIENT
        prediction, hidden = model.forward(inputs)
        difference = prediction - target
        loss = float(np.mean(difference**2))
        gradient = 2 * difference / difference.size
        # All plate edges are held at ambient temperature.
        boundary = np.zeros_like(prediction)
        boundary[:, 0, :] = prediction[:, 0, :]
        boundary[:, -1, :] = prediction[:, -1, :]
        boundary[:, :, 0] = prediction[:, :, 0]
        boundary[:, :, -1] = prediction[:, :, -1]
        gradient += 0.2 * 2 * boundary / boundary.size
        gradients = list(model.backward(inputs, hidden, gradient))
        if physics_weight:
            # Use one stable simulator step, including a backward pair at the endpoint.
            physics_steps = np.minimum(steps, 99)
            physics_inputs = make_inputs(batch_cases, physics_steps)
            before, before_hidden = model.forward(physics_inputs)
            future_inputs = make_inputs(batch_cases, physics_steps + 1)
            future, future_hidden = model.forward(future_inputs)
            phys_loss, grad_before, grad_after = physics_loss_and_gradient(
                before, future, batch_cases, 1
            )
            first = model.backward(physics_inputs, before_hidden, physics_weight * grad_before)
            second = model.backward(future_inputs, future_hidden, physics_weight * grad_after)
            gradients = [a + b + c for a, b, c in zip(gradients, first, second)]
            loss += physics_weight * phys_loss
        model.update(gradients, rate)
        if update % 25 == 0 or update == training_steps - 1:
            history.append((update + 1, loss))
        if validation is not None and ((update + 1) % validation_interval == 0 or update == training_steps - 1):
            validation_cases, validation_frames = validation
            errors = []
            for step in (0, 10, 25, 50, 75, 100):
                values, _ = model.forward(make_inputs(validation_cases, np.full(len(validation_cases), step)))
                errors.append(np.mean((values - (validation_frames[:, step] - AMBIENT))**2))
            error = float(np.sqrt(np.mean(errors)))
            model.validation_history.append((update + 1, error))
            print(f"Step {update + 1}, epochs {seen / total_maps:.2f}, validation RMSE {error:.5f}", flush=True)
            if error < best_error:
                best_error = error
                model.best_iteration = update + 1
                model.best_epochs = seen / total_maps
                best_parameters = {name: getattr(model, name).copy() for name in ("w1", "b1", "w2", "b2")}
            if error < significant_best * 0.995:
                significant_best = error
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                model.stop_reason = "validation_plateau"
                break
    model.steps_completed = update + 1
    model.epochs_completed = seen / total_maps
    if best_parameters is not None:
        for name, value in best_parameters.items():
            setattr(model, name, value)
    return history
