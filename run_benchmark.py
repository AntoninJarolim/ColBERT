import torch
import torch.nn as nn
import time


if __name__ == "__main__":
    class TransformerFFN(nn.Module):
        def __init__(self, d_model=128, d_ff=768, dropout=0.1, skip_norm=False):
            super().__init__()
            self.skip_norm = skip_norm

            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            )
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x):
            if self.skip_norm:
                return x + self.ff(x)
            return self.norm(x + self.ff(x))


    # run benchmark of TransformerFFN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    model = TransformerFFN(skip_norm=True).to(device).eval()

    def forward_pass(x):
        # x = x.to(device)
        _ = model(x)
        torch.cuda.synchronize() if device.type == "cuda" else None

    x = torch.randn(180 * 100, 128).to(device)
    forward_pass(x)

    times = []
    with torch.no_grad():
        for _ in range(5000):
            x = torch.randn(180 * 100, 128).to(device)
            start = time.perf_counter()
            forward_pass(x)
            end = time.perf_counter()
            times.append(end - start)

    def print_times(times):
        times = torch.tensor(times, dtype=torch.float64)
        # times = times * 1000
        # print with exponential notation
        # print("times:", times)
        print(f"Mean: {times.mean().item():.6e}")
        std = times.std(unbiased=False).item()
        print(f"Std Dev: {std:.6e}")
        print()

    print("TransformerFFN with skip norm")
    print_times(times)

    nr_v = 180 * 100
    m = torch.randn(nr_v).to(device)
    x = torch.randn(nr_v, 128).to(device)
    _ = x * m[:, None]

    times = []
    with torch.no_grad():
        for _ in range(5000):
            m = torch.randn(nr_v).to(device)
            x = torch.randn(nr_v, 128).to(device)

            start = time.perf_counter()
            _ = x * m[:, None]
            torch.cuda.synchronize() if device.type == "cuda" else None
            end = time.perf_counter()
            times.append(end - start)

    print("Elementwise multiplication")
    print_times(times)


    """
mean: 7.733533e-04
Std Dev: 1.394955e-04

Elementwise multiplication
Mean: 7.771490e-05
Std Dev: 1.998073e-05
"""