# prepare_problem_features.py
import csv
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MultiLabelBinarizer
import os

PROBLEMS_CSV = "problems.csv"
OUT_FEATURES = "problem_features.npy"
OUT_MAP = "problem_id_to_idx.json"
OUT_TAGS = "tag_classes.json"
OUT_SCALER = "rating_scaler.json"

def load_problems(csv_path=PROBLEMS_CSV):
    problems = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            pid = row['problem_id']
            tags = [t.strip() for t in (row.get('tags') or "").split(',') if t.strip()]
            rating = None
            try:
                rating = float(row['rating']) if row.get('rating') else None
            except:
                rating = None
            problems.append({"problem_id": pid, "tags": tags, "rating": rating})
    return problems

def build_features(problems):
    # tags
    all_tags = [p['tags'] for p in problems]
    mlb = MultiLabelBinarizer(sparse_output=False)
    tag_matrix = mlb.fit_transform(all_tags)  # shape (N, n_tags)

    # ratings -> fill missing with median then scale 0..1
    ratings = np.array([ (p['rating'] if p['rating'] is not None else np.nan) for p in problems ], dtype=float)
    # fill missing with median of available ratings (or 1500 fallback)
    if np.isnan(ratings).all():
        ratings = np.full_like(ratings, 1500.0)
    else:
        med = np.nanmedian(ratings)
        ratings = np.where(np.isnan(ratings), med, ratings)

    scaler = MinMaxScaler()
    ratings_scaled = scaler.fit_transform(ratings.reshape(-1,1))  # shape (N,1)

    # final feature matrix
    features = np.hstack([tag_matrix, ratings_scaled])  # (N, n_tags+1)

    # build problem_id->row index mapping
    pid2idx = {p['problem_id']: idx for idx, p in enumerate(problems)}

    return features.astype(np.float32), pid2idx, mlb.classes_.tolist(), {"min":float(scaler.data_min_[0]), "max":float(scaler.data_max_[0])}

if __name__ == "__main__":
    problems = load_problems(PROBLEMS_CSV)
    features, pid2idx, tag_classes, scaler_info = build_features(problems)
    np.save(OUT_FEATURES, features)
    with open(OUT_MAP, "w", encoding="utf-8") as f:
        json.dump(pid2idx, f, ensure_ascii=False, indent=2)
    with open(OUT_TAGS, "w", encoding="utf-8") as f:
        json.dump(tag_classes, f, ensure_ascii=False, indent=2)
    with open(OUT_SCALER, "w", encoding="utf-8") as f:
        json.dump(scaler_info, f, ensure_ascii=False, indent=2)

    print("Saved features:", OUT_FEATURES)
    print("Saved pid2idx:", OUT_MAP)
    print("num problems:", features.shape[0], "feature dim:", features.shape[1])
