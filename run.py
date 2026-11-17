"""Train the physics-informed model and report accuracy and timing on unseen heater settings."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from heat import DT, N, STEPS, sample_cases, simulate
from model import Network, train
from evaluation import evaluate_times
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cases", type=int, default=2000)
    parser.add_argument("--validation-cases", type=int, default=200)
    parser.add_argument("--test-cases", type=int, default=200)
    parser.add_argument("--training-steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--regenerate-data", action="store_true", help="Rebuild saved datasets even if valid")
    parser.add_argument("--prepare-data-only", action="store_true", help="Prepare datasets without training")
    args = parser.parse_args()
    if min(args.train_cases, args.validation_cases, args.test_cases, args.training_steps, args.batch_size) < 1:
        parser.error("Case counts, training steps and batch size must be positive")
    output = Path("results")
    output.mkdir(exist_ok=True)
    train_cases = sample_cases(args.train_cases, 11)
    validation_cases = sample_cases(args.validation_cases, 123)
    test_cases = sample_cases(args.test_cases, 20261117)
    all_cases = np.concatenate((train_cases, validation_cases, test_cases))
    assert len(np.unique(all_cases, axis=0)) == len(all_cases), "Duplicate configurations across splits"
    start = perf_counter()
    train_frames, train_status = load_dataset(output, "train", train_cases, args.regenerate_data)
    validation_frames, validation_status = load_dataset(output, "validation", validation_cases, args.regenerate_data)
    test_frames, test_status = load_dataset(output, "test", test_cases, args.regenerate_data)
    split_path = output / "dataset_splits.npz"
    expected = {"train": train_cases, "validation": validation_cases, "test": test_cases}
    same = False
    if split_path.exists():
        try:
            with np.load(split_path, allow_pickle=False) as saved:
                same = all(key in saved and np.array_equal(saved[key], value) for key, value in expected.items())
        except (ValueError, OSError):
            pass
    if not same:
        np.savez(split_path, **expected)
    if args.prepare_data_only:
        return
    data_seconds = perf_counter() - start
    summary = {"grid": N, "train_cases": args.train_cases,
               "validation_cases": args.validation_cases,
               "split_seeds": {"train": 11, "validation": 123, "test": 20261117},
               "test_cases": args.test_cases, "training_step_limit": args.training_steps,
               "batch_size": args.batch_size,
               "data_preparation_seconds": data_seconds,
               "dataset_status": {"train": train_status, "validation": validation_status, "test": test_status}, "models": {}}
    label = "physics_informed"
    physics_weight = 0.01
    print(f"Training with equation weight {physics_weight}", flush=True)
    model = Network(seed=7)
    start = perf_counter()
    history = train(model, train_cases, train_frames, args.training_steps, physics_weight,
                    validation=(validation_cases, validation_frames), batch_size=args.batch_size)
    training_seconds = perf_counter() - start
    model.save(output / "physics_informed.npz")
    summary["physics_weight"] = physics_weight
    # Test is evaluated only after all validation decisions are frozen.
    errors = []
    inference_times = []
    solver_times = []
    for case in test_cases:
        start = perf_counter()
        reference = simulate(*case)[-1]
        solver_times.append(perf_counter() - start)
        start = perf_counter()
        predicted = model.predict(*case, 100 * DT)
        inference_times.append(perf_counter() - start)
        errors.append(float(np.sqrt(np.mean((predicted - reference) ** 2))))
    summary["models"][label] = {
        "training_seconds": training_seconds,
        "by_time": evaluate_times(model, test_cases, test_frames),
        "mean_rmse_degrees": float(np.mean(errors)),
        "mean_prediction_seconds": float(np.mean(inference_times)),
        "mean_solver_seconds": float(np.mean(solver_times)),
        "history": history,
        "best_iteration": model.best_iteration,
        "best_epochs": model.best_epochs,
        "steps_completed": model.steps_completed,
        "epochs_completed": model.epochs_completed,
        "stop_reason": model.stop_reason,
        "validation_history": model.validation_history,
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2))
    for label, metrics in summary["models"].items():
        print(f"{label}: RMSE={metrics['mean_rmse_degrees']:.3f}, "
              f"prediction={metrics['mean_prediction_seconds'] * 1000:.3f} ms, "
              f"solver={metrics['mean_solver_seconds'] * 1000:.3f} ms, "
              f"training={metrics['training_seconds']:.2f} s")
    print("Saved models and metrics in results/. Run: python demo.py")


if __name__ == "__main__":
    main()
