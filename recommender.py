# recommender.py
import json
import time
import requests
import numpy as np
import torch
from neucf_model import NeuCF

CF_PROBLEMSET = "https://codeforces.com/api/problemset.problems"
CF_USER_STATUS = "https://codeforces.com/api/user.status"

# cache
_cache = {"problems": None, "ts": 0, "ttl": 60*60*6}

def load_problemset():
    now = time.time()
    if _cache["problems"] and now - _cache["ts"] < _cache["ttl"]:
        return _cache["problems"]
    r = requests.get(CF_PROBLEMSET, timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "OK":
        raise RuntimeError("problemset fetch failed")
    problems = data["result"]["problems"]
    # build mapping from key -> problem
    pm = {}
    for p in problems:
        key = f"{p.get('contestId')}-{p.get('index')}"
        pm[key] = p
    _cache["problems"] = pm
    _cache["ts"] = now
    return pm

def fetch_user_solved_keys(handle):
    r = requests.get(CF_USER_STATUS, params={"handle": handle}, timeout=12)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "OK":
        return []
    solved = set()
    for sub in data.get("result", []):
        if sub.get("verdict") == "OK":
            prob = sub.get("problem", {})
            key = f"{prob.get('contestId')}-{prob.get('index')}"
            solved.add(key)
    return list(solved)

class NeuCFRecommender:
    def __init__(self, model_path="neucf.pt", user_map="user2idx.json", item_map="item2idx.json", device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        with open(user_map, "r") as f: self.user2idx = json.load(f)
        with open(item_map, "r") as f: self.item2idx = json.load(f)
        self.idx2item = {int(v):k for k,v in self.item2idx.items()}
        n_users = len(self.user2idx)
        n_items = len(self.item2idx)
        self.device = device
        self.model = NeuCF(n_users, n_items)
        state = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.to(device)
        self.model.eval()
        # item embeddings matrix (GMF part)
        with torch.no_grad():
            self.item_emb_matrix = self.model.item_gmf.weight.detach().cpu().numpy()  # shape (n_items, emb)
        # problemset mapping
        self.problems = load_problemset()

    def _item_key_to_idx(self, item_key):
        # item_key like "1705-A" or "1705-A" as stored in item2idx mapping
        return int(self.item2idx[item_key]) if item_key in self.item2idx else None

    def recommend_for_handle(self, handle, k=10, restrict_rating=None):
        solved_keys = fetch_user_solved_keys(handle)
        solved_set = set(solved_keys)
        # known user?
        if handle in self.user2idx:
            user_idx = torch.tensor([self.user2idx[handle]], dtype=torch.long).to(self.device)
            # score all items (use batch)
            item_indices = []
            item_keys = []
            for idx_str, key in self.idx2item.items():
                item_keys.append(key)
                item_indices.append(int(idx_str))
            items_tensor = torch.tensor(item_indices, dtype=torch.long).to(self.device)
            # replicate user idx
            users_tensor = user_idx.repeat(items_tensor.size(0))
            with torch.no_grad():
                scores = self.model(user_idx=users_tensor, item_idx=items_tensor).cpu().numpy()
        else:
            # cold user: build pseudo-user embedding by averaging item embeddings of solved items if possible.
            solved_item_idxs = []
            for kkey in solved_keys:
                if kkey in self.item2idx:
                    solved_item_idxs.append(int(self.item2idx[kkey]))
            if len(solved_item_idxs) == 0:
                return {"error": "User is unknown to model and has no intersection with training items (cold start)."}
            # average embeddings
            user_emb = np.mean(self.item_emb_matrix[solved_item_idxs], axis=0)  # (emb,)
            # compute scores using GMF* + MLP path manually
            # We'll feed user_emb repeated and item_embs to model.forward using precomputed embeddings
            items_list = []
            keys_list = []
            for idx_str, key in self.idx2item.items():
                items_list.append(int(idx_str))
                keys_list.append(key)
            item_embs = torch.tensor(self.item_emb_matrix[items_list], dtype=torch.float).to(self.device)
            user_emb_t = torch.tensor(user_emb, dtype=torch.float).unsqueeze(0).to(self.device)  # (1, emb)
            user_embs_rep = user_emb_t.repeat(item_embs.size(0), 1)
            with torch.no_grad():
                scores = self.model(user_emb=user_embs_rep, item_emb=item_embs).cpu().numpy()
        # build mapping idx -> key and filter out solved items and restrict by rating if requested
        results = []
        # items_list and keys_list are aligned; build them if not present
        items_list = list(map(int, self.idx2item.keys()))
        keys_list = [self.idx2item[idx] for idx in items_list]
        for idx_pos, score in enumerate(scores):
            key = keys_list[idx_pos]
            if key in solved_set:
                continue
            prob = self.problems.get(key)
            if not prob:
                continue
            if restrict_rating is not None:
                rt = prob.get("rating")
                if not rt or not (restrict_rating[0] <= rt <= restrict_rating[1]):
                    continue
            results.append((float(score), key, prob))
        results.sort(reverse=True, key=lambda x: x[0])
        out = []
        for score, key, prob in results[:k]:
            out.append({
                "key": key,
                "contestId": prob.get("contestId"),
                "index": prob.get("index"),
                "name": prob.get("name"),
                "rating": prob.get("rating"),
                "tags": prob.get("tags", [])[:6],
                "score": score
            })
        return {"recommended": out}
