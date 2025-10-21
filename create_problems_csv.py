# create_problems_csv_stdlib.py
import requests
import csv

CF_PROBLEMSET = "https://codeforces.com/api/problemset.problems"

def fetch_problemset():
    r = requests.get(CF_PROBLEMSET, timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "OK":
        raise RuntimeError("Failed to fetch problemset")
    return data["result"]["problems"]

def build_problems_csv(out_file="problems.csv"):
    problems = fetch_problemset()
    with open(out_file, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["problem_id","contestId","index","name","rating","tags"])
        writer.writeheader()
        for p in problems:
            contestId = p.get("contestId")
            index = p.get("index")
            problem_id = f"{contestId}-{index}"
            name = p.get("name")
            rating = p.get("rating")  # can be None
            tags = ",".join(p.get("tags", []))
            writer.writerow({
                "problem_id": problem_id,
                "contestId": contestId,
                "index": index,
                "name": name,
                "rating": rating,
                "tags": tags
            })
    print(f"Saved {len(problems)} problems to {out_file}")

if __name__ == "__main__":
    build_problems_csv()
