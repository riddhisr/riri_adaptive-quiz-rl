# train_behavioral_cloning.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from sklearn.model_selection import train_test_split

CSV = Path("data/interactions.csv")
MODEL_OUT = Path("models/heuristic_cloned.pth")
BATCH = 64
EPOCHS = 25
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

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
    def forward(self, x): return self.net(x)

def build_state(row):
    prev_correct = float(row.get("correct", 0))
    time_taken = float(row.get("time_taken", 0))
    time_norm = min(time_taken / 20.0, 1.0)
    q_no = float(row.get("q_no", 0))
    max_q = 5.0
    q_frac = q_no / max(1.0, max_q)
    diff_map = {"easy": 0, "medium": 1, "hard": 2}
    last_action = diff_map.get(str(row.get("difficulty", "")).lower(), 1)
    return [prev_correct, time_norm, q_frac, float(last_action)]

def main():
    if not CSV.exists():
        print("No interactions.csv at", CSV); return
    df = pd.read_csv(CSV)
    df = df.dropna(subset=["difficulty"])
    diff_map = {"easy": 0, "medium": 1, "hard": 2}
    df["action"] = df["difficulty"].astype(str).str.lower().map(diff_map).fillna(1).astype(int)
    X = np.vstack([build_state(r) for _, r in df.iterrows()])
    y = df["action"].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    train_loader = DataLoader(SimpleDataset(X_train,y_train), batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(SimpleDataset(X_val,y_val), batch_size=BATCH, shuffle=False)

    model = PolicyNet(in_dim=X.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS+1):
        model.train()
        tot_loss = 0.0
        for xb, yb in train_loader:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item()*xb.size(0)
        tot_loss /= len(train_loader.dataset)
        # val
        model.eval()
        correct = 0; tot=0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb,yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb).argmax(dim=1)
                correct += (pred == yb).sum().item(); tot += xb.size(0)
        acc = correct / tot if tot else 0.0
        print(f"Epoch {epoch}/{EPOCHS} loss={tot_loss:.4f} val_acc={acc:.4f}")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_OUT)
    print("Saved behavioral clone to", MODEL_OUT)

if __name__ == "__main__":
    main()
