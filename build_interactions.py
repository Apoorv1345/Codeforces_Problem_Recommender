#!/usr/bin/env python3
"""
Robust build_interactions.py

- Usage:
    python build_interactions.py --handles handles.txt --out interactions.csv

Features:
- caches each handle's API response in ./cache/<handle>.json
- resumes: skips handles already present in interactions.csv
- retries with exponential backoff on transient failures
- polite delay (default 0.25s between requests)
- prints progress and error summary
"""
import requests
import csv
import time
import argparse
import os
import json
from urllib.parse import quote_plus

CF_USER_STATUS = "https://codeforces.com/api/user.status?handle={}"
CACHE_DIR = "cache"

def ensure_cache_dir():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)

def cache_path_for(handle):
    # safe filename
    safe = quote_plus(handle)
    return os.path.join(CACHE_DIR, f"{safe}.json")

def load_cached(handle):
    path = cache_path_for(handle)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def save_cache(handle, data):
    path = cache_path_for(handle)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def fetch_solved(handle, session, max_retries=3, backoff_base=0.8, timeout=12):
    # try cached first
    cached = load_cached(handle)
    if cached is not None:
        return cached

    url = CF_USER_STATUS.format(handle)
    attempt = 0
    while attempt < max_retries:
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                # save to cache even on non-OK, to avoid repeated failing calls
                save_cache(handle, data)
                return data
            else:
                # non-200, treat as transient for a few retries
                attempt += 1
                wait = backoff_base ** attempt
                print(f"Warning: {handle} returned HTTP {r.status_code}. retry {attempt}/{max_retries} after {wait:.2f}s")
                time.sleep(wait)
        except requests.exceptions.RequestException as e:
            attempt += 1
            wait = backoff_base ** attempt
            print(f"Network error for {handle}: {e}. retry {attempt}/{max_retries} after {wait:.2f}s")
            time.sleep(wait)
    # final try: return whatever (None) and save empty fail record to avoid re-hitting repeatedly
    save_cache(handle, {"status":"ERROR", "error": f"failed after {max_retries} attempts"})
    return {"status":"ERROR", "error": "failed after retries"}

def parse_solved_from_api_result(api_json):
    """Return list of problem keys like '1705-A' if api_json indicates OK."""
    out = []
    if not api_json:
        return out
    if api_json.get("status") != "OK":
        return out
    for sub in api_json.get("result", []):
        if sub.get("verdict") == "OK":
            prob = sub.get("problem", {})
            contest = prob.get("contestId")
            index = prob.get("index")
            if contest is None or index is None:
                continue
            out.append(f"{contest}-{index}")
    return out

def read_handles(handles_file):
    with open(handles_file, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines

def read_already_done(out_csv):
    done_handles = set()
    if not os.path.exists(out_csv):
        return done_handles
    try:
        with open(out_csv, "r", newline='', encoding="utf-8") as f:
            reader = csv.reader(f)
            # file rows are [handle, problemkey]
            for row in reader:
                if not row: continue
                handle = row[0].strip()
                if handle:
                    done_handles.add(handle)
    except Exception:
        pass
    return done_handles

def build(handles_file, out_csv, delay=0.25, verbose_every=10):
    ensure_cache_dir()
    handles = read_handles(handles_file)
    if not handles:
        print("No handles found in", handles_file)
        return

    done_handles = read_already_done(out_csv)
    total = len(handles)
    print(f"Total handles: {total}. Already processed handles (from {out_csv}): {len(done_handles)}")

    session = requests.Session()
    session.headers.update({"User-Agent": "CF-Interaction-Collector/1.0 (+https://example.com)"})

    # open CSV in append mode - this makes script resumable
    with open(out_csv, "a", newline='', encoding="utf-8") as of:
        writer = csv.writer(of)
        processed = 0
        errors = []
        for idx, h in enumerate(handles):
            if h in done_handles:
                if (idx+1) % verbose_every == 0:
                    print(f"Skipping {idx+1}/{total} {h} (already done)")
                continue
            # fetch (with caching & retries)
            t = ""
            for ch in range(len(h)) :
                if h[ch] != '\"':
                    t = t + h[ch]
            
            h = t
            api_json = fetch_solved(t, session)
            solved = parse_solved_from_api_result(api_json)
            if solved:
                for item in solved:
                    writer.writerow([h, item])
                # flush to disk so progress persists
                of.flush()
                os.fsync(of.fileno())
            else:
                # no solved found (could mean private/empty/invalid)
                # still write nothing but record error for reporting
                errors.append((h, api_json.get("status", "NO_DATA")))
            processed += 1
            if (idx+1) % verbose_every == 0:
                print(f"Processed {idx+1}/{total} handles; kept so far: {processed}; errors: {len(errors)}")
            time.sleep(delay)

    print("DONE. Processed (new) handles:", processed)
    if errors:
        print("Errors for some handles (sample up to 10):")
        for e in errors[:10]:
            print(" ", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--handles", default="handles.txt")
    parser.add_argument("--out", default="interactions.csv")
    parser.add_argument("--delay", type=float, default=0.25, help="seconds between requests (be polite)")
    parser.add_argument("--verbose-every", type=int, default=10)
    args = parser.parse_args()
    build(args.handles, args.out, delay=args.delay, verbose_every=args.verbose_every)
