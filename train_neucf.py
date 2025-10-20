# train_neucf.py
import csv
import random
import json
import argparse
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from neucf_model import NeuCF

class InteractionDataset(Dataset):
    def __init__(self, user_item_pairs, n_items, neg_samples=4):
        # user_item_pairs: list of (user_idx, item_idx)
        self.pos = user_item_pairs
        self.n_items = n_items
        self.neg_samples = neg_samples

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, idx):
        u, i = self.pos[idx]
        samples = [(u, i, 1.0)]
        for _ in range(self.neg_samples):
            j = random.randrange(self.n_items)
            while j == i:
                j = random.randrange(self.n_items)
            samples.append((u, j, 0.0))
        users = torch.tensor([s[0] for s in samples], dtype=torch.long)
        items = torch.tensor([s[1] for s in samples], dtype=torch.long)
        labels = torch.tensor([s[2] for s in samples], dtype=torch.float)
        return users, items, labels

def load_interactions(path):
    user2idx = {}
    item2idx = {}
    interactions = []
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            user, item = row[0].strip(), row[1].strip()
            if user == "" or item == "": continue
            if user not in user2idx:
                user2idx[user] = len(user2idx)
            if item not in item2idx:
                item2idx[item] = len(item2idx)
            interactions.append((user2idx[user], item2idx[item], user, item))
    return interactions, user2idx, item2idx

def build_pairs(interactions):
    # interactions: list of (u_idx, i_idx, user_str, item_str)
    return [(u,i) for u,i,_,_ in interactions]

def train(args):
    interactions, user2idx, item2idx = load_interactions(args.data)
    print(f"Loaded {len(interactions)} interactions; {len(user2idx)} users; {len(item2idx)} items")
    pairs = build_pairs(interactions)
    n_users = len(user2idx)
    n_items = len(item2idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuCF(n_users, n_items, emb_size=args.emb_size, mlp_layers=[args.mlp1, args.mlp2, args.mlp3], dropout=args.dropout)
    model.to(device)

    dataset = InteractionDataset(pairs, n_items, neg_samples=args.neg)
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for users, items, labels in pbar:
            users = users.view(-1).to(device)
            items = items.view(-1).to(device)
            labels = labels.view(-1).to(device)
            preds = model(user_idx=users, item_idx=items)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / ( (pbar.n+1) ))
        print(f"Epoch {epoch} avg loss {total_loss/len(dataloader):.4f}")

    # Save model + maps + item embeddings
    torch.save(model.state_dict(), args.out_model)
    print("Saved model:", args.out_model)
    with open(args.out_user_map, "w") as f:
        json.dump(user2idx, f)
    with open(args.out_item_map, "w") as f:
        json.dump(item2idx, f)
    print("Saved maps")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="interactions.csv", help="CSV with user,item")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--bs", type=int, default=2048)
    parser.add_argument("--emb_size", type=int, default=64)
    parser.add_argument("--neg", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mlp1", type=int, default=128)
    parser.add_argument("--mlp2", type=int, default=64)
    parser.add_argument("--mlp3", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--out_model", default="neucf.pt")
    parser.add_argument("--out_user_map", default="user2idx.json")
    parser.add_argument("--out_item_map", default="item2idx.json")
    args = parser.parse_args()
    train(args)
