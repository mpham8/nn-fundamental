import torch
from pathlib import Path

from model import Transformer

_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "weights" / "model.pt"
_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "self_imessages.txt"


def load_vocab(txt_path):
    lines = [line.strip() for line in open(txt_path, encoding="utf-8") if line.strip()]
    stoi = {c: i + 1 for i, c in enumerate(sorted(set("".join(lines))))}
    itos = {i: c for c, i in stoi.items()}
    itos[0] = "<pad>"
    return stoi, itos


def generate(model, stoi, itos, prompt="", num_tokens=50, block_size=128, temperature=0.8, device="cpu"):
    model.eval()
    ids = [stoi.get(c, 0) for c in prompt][-block_size:]
    with torch.no_grad():
        for _ in range(num_tokens):
            x = torch.tensor([ids], dtype=torch.long, device=device)
            probs = torch.softmax(model(x)[0, -1] / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            if next_id == 0: break
            ids.append(next_id)
    return prompt + "".join(itos[i] for i in ids[len(prompt):])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(_WEIGHTS_PATH, map_location=device)
    
    cfg, stoi, state = ckpt["config"], ckpt["stoi"], ckpt["model"]
    d_model, h, nx, maxTokens, vocabSize = cfg["d_model"], cfg["h"], cfg["nx"], cfg["maxTokens"], cfg["vocabSize"]
    
    itos = {i: c for c, i in stoi.items()}
    itos[0] = "<pad>"
    model = Transformer(d_model, d_model // h, d_model // h, h, maxTokens, nx, vocabSize).to(device)
    model.load_state_dict(state)
    while True:
        p = input("start phrase ('exit()' to quit): ")
        if p.strip().lower() in {"exit()"}: break
        print("entire phrase:", generate(model, stoi, itos, prompt=p, block_size=maxTokens, device=device))


if __name__ == "__main__":
    main()
