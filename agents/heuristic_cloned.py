# agents/heuristic_cloned.py
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

class PolicyNet(nn.Module):
    def __init__(self, in_dim=4, hid=64, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class HeuristicClonedAgent:
    """
    Loads a behavioral-cloning model and exposes select(state) to return action {0,1,2}.
    """
    def __init__(self, model_path="models/heuristic_cloned.pth", device=None):
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = PolicyNet(in_dim=4).to(self.device)
        if self.model_path.exists():
            try:
                self.net.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.net.eval()
                self._loaded = True
            except Exception:
                self._loaded = False
        else:
            self._loaded = False

    def select(self, state):
        """
        state: numpy-like vector of length 4 (matching app state: prev_correct, time_norm, q_frac, last_action)
        """
        try:
            x = torch.tensor(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.net(x)
                a = int(logits.argmax(dim=1).item())
            return a
        except Exception:
            # fallback random
            return int(np.random.choice([0,1,2]))
