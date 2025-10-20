from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from recommender import NeuCFRecommender

app = FastAPI(title="CF Recommender (FastAPI)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate recommender once at startup
reco = NeuCFRecommender(
    model_path="neucf.pt",
    user_map="user2idx.json",
    item_map="item2idx.json"
)

@app.get("/recommend")
def recommend(handle: str = Query(..., min_length=1), k: int = Query(8, ge=1, le=50)):
    try:
        result = reco.recommend_for_handle(handle, k=k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if result is None or "error" in result:
        raise HTTPException(status_code=400, detail=result.get("error", "Unknown error"))
    return result

@app.get("/health")
def health():
    return {"status": "ok"}
