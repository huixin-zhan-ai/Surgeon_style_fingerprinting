import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import random

# Configuration
NUM_SAMPLES = 1000
NUM_SURGEONS = 8
GESTURE_VOCAB_SIZE = 16  # 15 gestures + 1 MASK
MASK_TOKEN_ID = 15
SEQ_LEN = 5
HIDDEN_DIM = 512
VISION_DIM = 1000
LANG_DIM = 768
NUM_TIMESTEPS = 10
EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Forward corruption: transition matrices
q_matrix = np.zeros((NUM_TIMESTEPS, GESTURE_VOCAB_SIZE, GESTURE_VOCAB_SIZE))
for t in range(NUM_TIMESTEPS):
    alpha = 1 - (t + 1) / NUM_TIMESTEPS
    for i in range(GESTURE_VOCAB_SIZE):
        q_matrix[t, i] = (1 - alpha) / (GESTURE_VOCAB_SIZE - 1)
        q_matrix[t, i, i] = alpha
q_matrix = torch.tensor(q_matrix, dtype=torch.float32)

def corrupt(x0, t):
    B, T = x0.shape
    probs = q_matrix[t][:, x0.view(-1)]  # (V, B*T)
    probs = probs.permute(1, 0).view(B, T, GESTURE_VOCAB_SIZE)
    xt = torch.multinomial(probs.view(-1, GESTURE_VOCAB_SIZE), 1).view(B, T)
    return xt

# Build GRS mapping: Surgeon ID → average GRS score
def load_grs_mapping(meta_path="/content/drive/My Drive/surgeon_profiling_fingerprint/data/Suturing/meta_file_Suturing.txt"):
    grs_map = {}
    with open(meta_path, "r") as f:
        for line in f:
            if line.startswith("Suturing_"):
                parts = line.strip().split()
                if len(parts) >= 9:
                    surgeon_char = parts[0].split("_")[1][0]  # e.g., 'B'
                    scores = list(map(float, parts[3:9]))
                    avg_grs = sum(scores) / len(scores)
                    if surgeon_char not in grs_map:
                        grs_map[surgeon_char] = []
                    grs_map[surgeon_char].append(avg_grs)

    # Average across trials for each surgeon ID (B to I → index 0 to 7)
    final_map = {}
    for sid, scores in grs_map.items():
        idx = ord(sid) - ord("B")
        final_map[idx] = sum(scores) / len(scores)
    return final_map

GRS_MAP = load_grs_mapping()

# class GestureSeqDataset(Dataset):
#     def __init__(self, num_samples):
#         self.vision_feat = torch.randn(num_samples, VISION_DIM)
#         self.lang_feat = torch.randn(num_samples, LANG_DIM)
#         self.surgeon_id = torch.randint(0, NUM_SURGEONS, (num_samples,))
#         self.gesture_seq = torch.randint(0, GESTURE_VOCAB_SIZE - 1, (num_samples, SEQ_LEN))
#         self.timesteps = torch.randint(0, NUM_TIMESTEPS, (num_samples,))

#     def __len__(self):
#         return len(self.surgeon_id)

#     def __getitem__(self, idx):
#         x0 = self.gesture_seq[idx]
#         t = self.timesteps[idx]
#         xt = corrupt(x0.unsqueeze(0), t).squeeze(0)
#         return {
#             "vision_feat": self.vision_feat[idx],
#             "lang_feat": self.lang_feat[idx],
#             "surgeon_id": self.surgeon_id[idx],
#             "x0": x0,
#             "xt": xt,
#             "t": t
#         }

class RealJIGSAWSDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.vision_feat = torch.tensor(data["vision_feat"], dtype=torch.float32)
        self.lang_feat = torch.tensor(data["lang_feat"], dtype=torch.float32)
        self.gesture_seq = torch.tensor(data["gesture_seq"], dtype=torch.long)
        self.surgeon_id = torch.tensor(data["surgeon_id"], dtype=torch.long)
        self.timesteps = torch.randint(0, NUM_TIMESTEPS, (len(self.surgeon_id),))

    def __len__(self):
        return len(self.surgeon_id)

    def __getitem__(self, idx):
        x0 = self.gesture_seq[idx]
        t = self.timesteps[idx]
        xt = corrupt(x0.unsqueeze(0), t).squeeze(0)
        return {
            "vision_feat": self.vision_feat[idx],
            "lang_feat": self.lang_feat[idx],
            "surgeon_id": self.surgeon_id[idx],
            "x0": x0,
            "xt": xt,
            "t": t
        }


class TimeEmbedding(nn.Module):
    def __init__(self, num_timesteps, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(num_timesteps, hidden_dim)

    def forward(self, t):  # (B,)
        return self.embed(t)

# class SurgeonEmbedding(nn.Module):
#     def __init__(self, num_surgeons, hidden_dim):
#         super().__init__()
#         self.embedding = nn.Embedding(num_surgeons, hidden_dim)

#     def forward(self, surgeon_id):
#         return self.embedding(surgeon_id)
from sentence_transformers import SentenceTransformer
class SurgeonEmbedding(nn.Module):
    def __init__(self, model_name="all-MiniLM-L6-v2", hidden_dim=512):
        super().__init__()
        self.llm = SentenceTransformer(model_name)
        self.proj = nn.Linear(self.llm.get_sentence_embedding_dimension(), hidden_dim)

    def forward(self, surgeon_id):
        if isinstance(surgeon_id, torch.Tensor):
            ids = surgeon_id.tolist()
        else:
            ids = list(surgeon_id)

        text = []
        for i in ids:
            grs = GRS_MAP.get(i, 3.0)  # fallback to average GRS if missing
            text.append(f"Surgeon ID: {i}, average skill score: {grs:.2f}")

        with torch.no_grad():
            emb = self.llm.encode(text, convert_to_tensor=True)
        return self.proj(emb)


class VLAEncoder(nn.Module):
    def __init__(self, vision_dim, lang_dim, hidden_dim):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.lang_proj = nn.Linear(lang_dim, hidden_dim)
        self.fuse = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, vision_feat, lang_feat):
        v = self.vision_proj(vision_feat)
        l = self.lang_proj(lang_feat)
        return self.fuse(torch.cat([v, l], dim=-1))

class DiffusionGestureModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )

    def forward(self, xt, cond):
        x = self.token_embed(xt)
        cond = cond.unsqueeze(1).expand_as(x)
        return self.net(x + cond)

class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vla = VLAEncoder(VISION_DIM, LANG_DIM, HIDDEN_DIM)
        #self.surgeon_embed = SurgeonEmbedding(NUM_SURGEONS, HIDDEN_DIM)
        self.surgeon_embed = SurgeonEmbedding()  # no need for num_surgeons
        self.time_embed = TimeEmbedding(NUM_TIMESTEPS, HIDDEN_DIM)
        self.denoise = DiffusionGestureModel(GESTURE_VOCAB_SIZE, HIDDEN_DIM)

    def forward(self, xt, vision_feat, lang_feat, surgeon_id, t):
        cond = self.vla(vision_feat, lang_feat) + self.surgeon_embed(surgeon_id) + self.time_embed(t)
        return self.denoise(xt, cond)

# def train():
#     dataset = GestureSeqDataset(NUM_SAMPLES)
#     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
#     model = FullModel()
#     optimizer = optim.Adam(model.parameters(), lr=LR)
#     criterion = nn.CrossEntropyLoss()

#     model.train()
#     for epoch in range(EPOCHS):
#         total_loss = 0.0
#         for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
#             vision_feat = batch["vision_feat"]
#             lang_feat = batch["lang_feat"]
#             surgeon_id = batch["surgeon_id"]
#             x0 = batch["x0"]
#             xt = batch["xt"]
#             t = batch["t"]

#             logits = model(xt, vision_feat, lang_feat, surgeon_id, t)
#             loss = criterion(logits.view(-1, GESTURE_VOCAB_SIZE), x0.view(-1))

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")
#     return model

def train():
    from sklearn.metrics import classification_report
    #dataset = GestureSeqDataset(NUM_SAMPLES)
    dataset = RealJIGSAWSDataset("/content/drive/My Drive/surgeon_profiling_fingerprint/data/jigsaws_full_processed.npz")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = FullModel()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        total_acc = 0.0
        total_top5 = 0.0
        all_preds = []
        all_labels = []

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            vision_feat = batch["vision_feat"]
            lang_feat = batch["lang_feat"]
            surgeon_id = batch["surgeon_id"]
            x0 = batch["x0"]
            xt = batch["xt"]
            t = batch["t"]

            logits = model(xt, vision_feat, lang_feat, surgeon_id, t)
            loss = criterion(logits.view(-1, GESTURE_VOCAB_SIZE), x0.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)  # Top-1
                correct = (preds == x0).float().mean().item()
                total_acc += correct

                top5_preds = torch.topk(logits, k=5, dim=-1).indices
                match_top5 = top5_preds.eq(x0.unsqueeze(-1)).any(dim=-1).float().mean().item()
                total_top5 += match_top5

                all_preds.extend(preds.view(-1).cpu().numpy())
                all_labels.extend(x0.view(-1).cpu().numpy())

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)
        avg_top5 = total_top5 / len(dataloader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f} | Top-1 Acc: {avg_acc:.4f} | Top-5 Acc: {avg_top5:.4f}")

        # Optional: print classification report
        print(classification_report(all_labels, all_preds, labels=list(range(GESTURE_VOCAB_SIZE))))

    return model

if __name__ == "__main__":
    model = train()
    torch.save(model.state_dict(), "diffusion_model.pt")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

def sample_gesture_sequence(model, vision_feat, lang_feat, surgeon_id, seq_len, T, device):
    model.eval()
    with torch.no_grad():
        xt = torch.randint(0, GESTURE_VOCAB_SIZE, (1, seq_len)).to(device)
        for t in reversed(range(T)):
            t_tensor = torch.tensor([t], dtype=torch.long).to(device)
            logits = model(xt, vision_feat.to(device), lang_feat.to(device), surgeon_id.to(device), t_tensor)
            probs = torch.softmax(logits, dim=-1)
            xt = torch.multinomial(probs.view(-1, GESTURE_VOCAB_SIZE), 1).view(1, seq_len)
    return xt

# def extract_surgeon_embeddings(model):
#     return model.surgeon_embed.embedding.weight.detach().cpu().numpy()
def extract_surgeon_embeddings(model, num_surgeons=8):
    ids = [f"Surgeon ID: {i}" for i in range(num_surgeons)]
    with torch.no_grad():
        emb = model.surgeon_embed.llm.encode(ids, convert_to_tensor=True)
        return model.surgeon_embed.proj(emb).cpu().numpy()


def plot_gesture_distribution(model, num_surgeons, seq_len, T, device):
    gesture_counts = np.zeros((num_surgeons, GESTURE_VOCAB_SIZE))

    for sid in range(num_surgeons):
        vision_feat = torch.randn(1, VISION_DIM)
        lang_feat = torch.randn(1, LANG_DIM)
        surgeon_id = torch.tensor([sid], dtype=torch.long)

        xt = sample_gesture_sequence(
            model, vision_feat, lang_feat, surgeon_id,
            seq_len=seq_len, T=T, device=device
        )
        for tok in xt[0].cpu().numpy():
            gesture_counts[sid, tok] += 1

    plt.figure(figsize=(12, 6))
    sns.heatmap(gesture_counts, annot=True, fmt=".0f", cmap="Blues")
    plt.xlabel("Gesture Token")
    plt.ylabel("Surgeon ID")
    plt.title("Predicted Gesture Distribution per Surgeon")
    plt.tight_layout()
    plt.savefig("gesture_distribution_heatmap.png")
    plt.show()

# 🔽 AFTER training, run this:
model = FullModel()
model.load_state_dict(torch.load("diffusion_model.pt", map_location=torch.device("cpu")))
model.eval()

# Now safe to run:
plot_gesture_distribution(model, NUM_SURGEONS, SEQ_LEN, NUM_TIMESTEPS, device=torch.device("cpu"))