# Heat spreading across a metal plate

This project compares a heat-equation simulator with one physics-informed neural network written in NumPy. The network learns from example temperature maps and receives a penalty when its predicted changes violate the heat equation.

The plate is a 24 × 24 grid. Its edges stay at 20 degrees. A Gaussian heater can move within the plate and change strength. Time and distance are normalized units; the temperature values are illustrative, not calibrated measurements of a particular metal.

The equation is `temperature change = alpha × spatial Laplacian + heater power`. `heat.py` approximates it by updating the interior grid cells at each small time step. The chosen step size satisfies the explicit two-dimensional diffusion stability limit `DT <= DX² / (4 × ALPHA)`.

## Run

Install Python 3.10+ and then, from this folder:

```powershell
python -m pip install -r requirements.txt
python run.py
python demo.py
```

If `python` points to the Windows Store placeholder, use your installed Python executable instead. For a quicker trial: `python run.py --train-cases 20 --test-cases 5 --training-steps 30`.

`run.py` saves the network and `metrics.json` in `results/`. The test heater settings use a separate random seed and are never used for training. RMSE is measured on the final full temperature map. Timing compares one full solver run with one full network map prediction, on the same machine. Training and example-generation time are listed separately. NumPy operations and Python loops have different overheads, so these timings characterize this implementation rather than all possible solvers or hardware.

## Files

- `heat.py`: heat equation, heater, grid update, and random heater settings.
- `model.py`: neural network, hand-written backpropagation, training, and heat-equation residual penalty.
- `run.py`: generate examples, train the physics-informed model, test on unseen heater settings, save results.
- `demo.py`: interactive side-by-side temperature maps and error map.
- `export_onnx.py`: export the trained model and verify it with ONNX Runtime.
- `requirements.txt`: NumPy, Matplotlib, PyTorch, ONNX, and ONNX Runtime dependencies.

The equation penalty compares network predictions one stable solver step apart. It uses the same spatial finite-difference rule as the simulator. The network uses sine/cosine position features to represent a moving hot spot, and hand-written Adam optimization to make small gradients useful.

## Dataset splits

Defaults are 2,000 training configurations, 200 validation configurations, and 200 test configurations. Each has 101 maps of 24 × 24 temperatures. These are disjoint heater configurations, not randomly split frames from the same simulation. Seeds are recorded in the metrics, and exact duplicate configurations are checked.

The validation set (sometimes called a cross-validation set) is a fixed holdout, not k-fold cross-validation. Every 500 training updates, validation RMSE at six times selects the best checkpoint. Validation examples do not contribute gradients. The separate test set evaluates the selected checkpoint. Use `--training-steps` (default maximum 30,000) and `--batch-size` (default 32). Each epoch now shuffles and visits every training map once. A partial final batch completes each epoch. Training stops after eight validation checks without a cumulative 0.5% improvement, and restores the checkpoint with the lowest validation RMSE. Metrics record the selected step, equivalent epochs, steps completed, and whether training stopped at a plateau or the limit.

`results/dataset_splits.npz` stores the heater settings. `train_frames.npy`, `validation_frames.npy`, and `test_frames.npy` in `results/` store all temperature maps as float32 arrays using about 558 MB total. They are memory-mapped to limit working memory. Training reuses compatible saved datasets by default. Missing, corrupted, or incompatible splits are regenerated automatically. Use `--train-cases`, `--validation-cases`, and `--test-cases` to change the sizes.

Run `python check_accuracy.py` to evaluate the saved model on 40 additional unseen configurations at six times, including the initial state, and generate `results/accuracy_comparison.png`. This check reports relative error against temperature rise above ambient, which avoids hiding errors behind the constant 20-degree background. Initial and edge temperatures are enforced exactly by multiplying predicted temperature rise by normalized time and a mask that is zero on the edges. The neural approximations are not guaranteed to preserve physical bounds. The saved `metrics_before_improvement.json` contains the original run for comparison.



## Constrained model and equation weight

`python run.py` trains once with a fixed equation weight of **0.01**, selected by our earlier validation experiment. The weight-search loop and plural command-line option have been removed. The heat residual is divided by the time interval and a fixed reference heater rate of 20 before squaring. This avoids suppressing its magnitude simply because the time step is small. Old weights are therefore not directly comparable.

The lowest validation-RMSE checkpoint from this single training run is selected. Only then is the fresh 200-configuration test split (seed 20261117) scored. `metrics.json` records the fixed equation weight, training time, and errors at steps 0, 10, 25, 50, 75, 100. Each time includes RMSE, relative rise error, largest error, hot-region RMSE, and edge error. Relative error is undefined at time zero because the true temperature rise is zero.

`evaluation.py` implements these checks. Versioned checkpoint metadata preserves the interpretation of old weights; the archived pre-change model remains readable. Constraints guarantee the initial state and edges, not every physical property of interior predictions. Test cases become a reused benchmark on subsequent runs; avoid tuning choices based on those results.


## Reusing datasets

- `python run.py`: validate and reuse saved datasets, then train.
- `python run.py --regenerate-data`: explicitly regenerate datasets, then train.
- `python run.py --prepare-data-only`: prepare/check datasets without training or replacing models.
- `python run.py --regenerate-data --prepare-data-only`: regenerate datasets only.

`datasets.py` checks simulation-code fingerprints, heater configurations, shapes, dtypes, and file checksums. Each split has its own JSON manifest. Changing counts or simulation code rebuilds affected splits. A one-time full comparison against the simulator registers existing legacy datasets without rewriting their arrays. Fixed seeds mean explicit regeneration reproduces the same examples when settings are unchanged. Dataset loading/verification time is recorded as `data_preparation_seconds`, separate from model training. Test data reused across runs remains a benchmark, not a fresh untouched test each time.

## Exporting to ONNX

Run `python export_onnx.py` to create `results/physics_informed.onnx`. It accepts batches of `[heater_x, heater_y, strength, time]` and returns `24 × 24` temperature maps. Time uses this project's model units, with training covering 0–0.3. The script checks the ONNX file and compares ONNX Runtime output against NumPy predictions before reporting success.

## Project website

The interactive website is in `web/`. Start a local server from the project root with `python -m http.server 8000`, then open `http://localhost:8000/web/`. It uses the trained model weights in `web/model.json`, runs the heat solver and model in the browser, and shows live timing, full-map RMSE, and hot-region RMSE. Retrain the model and refresh the website weights with `python -c "import json,numpy as np; d=np.load('results/physics_informed.npz'); m={k:d[k].tolist() for k in ('w1','b1','w2','b2')}; m['hard_constraints']=bool(d['hard_constraints']); open('web/model.json','w').write(json.dumps(m,separators=(',',':')))"`.
