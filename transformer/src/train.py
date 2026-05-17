import torch
import torch.nn as nn
import yaml

from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from model import Transformer

_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "self_imessages.txt"
_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "weights" / "model.pt"
_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_data(txt_path=_DATA_PATH, block_size=128):
    lines = [line.strip() for line in open(txt_path, encoding="utf-8") if line.strip()]
    text = "\n".join(lines)
    stoi = {c: i + 1 for i, c in enumerate(sorted(set(text)))}
    ids = [stoi[c] for c in text]

    xs, ys = [], []
    for i in range(len(ids) - block_size):
        chunk = ids[i : i + block_size + 1]
        xs.append(chunk[:-1])
        ys.append(chunk[1:])

    return torch.tensor(xs), torch.tensor(ys), stoi, len(stoi) + 1


def main():
    with open(_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    model_cfg = config["model"]
    train_cfg = config["train"]

    d_model = model_cfg["d_model"]
    h = model_cfg["h"]
    nx = model_cfg["nx"]
    d_k = model_cfg["d_k"]
    d_v = model_cfg["d_v"]
    maxTokens = model_cfg["max_tokens"]
    batch_size = train_cfg["batch_size"]
    epochs = train_cfg["epochs"]
    lr = train_cfg["lr"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device}")

    xs, ys, stoi, vocabSize = load_data(block_size=maxTokens)
    loader = DataLoader(TensorDataset(xs, ys), batch_size=batch_size, shuffle=True)

    model = Transformer(d_model, d_k, d_v, h, maxTokens, nx, vocabSize).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits.view(-1, vocabSize), y_batch.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"epoch {epoch + 1}  loss {total_loss / len(loader):.4f}")

    _WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "stoi": stoi,
            "config": {
                "d_model": d_model,
                "h": h,
                "nx": nx,
                "maxTokens": maxTokens,
                "vocabSize": vocabSize,
            },
        },
        _WEIGHTS_PATH,
    )
    print(f"saved to {_WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
