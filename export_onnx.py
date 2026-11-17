"""Export the trained temperature predictor as a portable ONNX model."""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from heat import AMBIENT, DT, N
from model import Network


class PortableHeatModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.register_buffer("w1", torch.from_numpy(model.w1.astype(np.float32)))
        self.register_buffer("b1", torch.from_numpy(model.b1.astype(np.float32)))
        self.register_buffer("w2", torch.from_numpy(model.w2.astype(np.float32)))
        self.register_buffer("b2", torch.from_numpy(model.b2.astype(np.float32)))
        mask = np.ones((1, N, N), dtype=np.float32)
        mask[:, 0, :] = mask[:, -1, :] = 0
        mask[:, :, 0] = mask[:, :, -1] = 0
        self.register_buffer("mask", torch.from_numpy(mask))

    def forward(self, inputs):
        # Inputs: [heater_x, heater_y, strength, time_seconds], each a batch row.
        normalized_time = inputs[:, 3:4] / (DT * 100.0)
        normalized_inputs = torch.cat((inputs[:, :3], normalized_time), dim=1)
        xy = normalized_inputs[:, :2].unsqueeze(-1)
        frequencies = torch.arange(1, 5, dtype=inputs.dtype, device=inputs.device) * np.pi
        angles = xy * frequencies
        features = torch.cat((normalized_inputs * 2.0 - 1.0,
                              torch.sin(angles).reshape(-1, 8),
                              torch.cos(angles).reshape(-1, 8)), dim=1)
        hidden = torch.tanh(features @ self.w1 + self.b1)
        rise = (hidden @ self.w2 + self.b2).reshape(-1, N, N)
        rise = rise * normalized_time.reshape(-1, 1, 1) * self.mask
        return rise + AMBIENT


def main():
    root = Path(__file__).resolve().parent
    model_path = root / "results" / "physics_informed.npz"
    output_path = root / "results" / "physics_informed.onnx"
    numpy_model = Network.load(model_path)
    torch_model = PortableHeatModel(numpy_model).eval()
    example = torch.tensor([[0.5, 0.5, 1.0, 0.15]], dtype=torch.float32)
    torch.onnx.export(
        torch_model, example, str(output_path), input_names=["heater_time"],
        output_names=["temperature"], dynamic_axes={"heater_time": {0: "batch"},
                                                     "temperature": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    exported = onnx.load(str(output_path))
    onnx.checker.check_model(exported)
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    cases = np.array([[0.5, 0.5, 1.0, 0.15], [0.28, 0.71, 1.3, 0.075],
                      [0.8, 0.2, 0.7, 0.3]], dtype=np.float32)
    actual = session.run(["temperature"], {"heater_time": cases})[0]
    expected = np.stack([
        numpy_model.predict(x, y, strength, time)
        for x, y, strength, time in cases
    ]).astype(np.float32)
    max_error = float(np.max(np.abs(actual - expected)))
    if not np.allclose(actual, expected, rtol=2e-5, atol=2e-5):
        raise SystemExit(f"ONNX parity check failed: max absolute difference {max_error}")
    print(f"Exported {output_path}")
    print(f"Input: heater_time [batch, 4] = x, y, strength, time in model units")
    print(f"Output: temperature [batch, {N}, {N}] in illustrative temperature units")
    print(f"ONNX Runtime parity check passed; max absolute difference {max_error:.2g}")


if __name__ == "__main__":
    main()
