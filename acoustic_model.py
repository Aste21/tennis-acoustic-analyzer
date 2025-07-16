import torch
import torch.nn as nn
from pathlib import Path

MAX_T = 16

class FourClassNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 24, (5, 5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, (5, 5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 48, (5, 5), padding=2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * (75 // 4) * (MAX_T // 4), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


def load_acoustic_model(path: Path) -> nn.Module:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and not hasattr(state, "eval"):
        model = FourClassNet()
        model.load_state_dict(state)
    else:
        model = state
    model.eval()
    return model
