from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI(title="AI Image Understanding & Content Matching Engine", version="1.0.0")

# 1. Structured Vision Output Schema (Pydantic)[cite: 15, 22]
class ImageMetadata(BaseModel):
    subject: str = Field(..., description="The main subject of the image")
    category: str = Field(..., description="The category of the subject, e.g., animal")
    attributes: List[str] = Field(..., description="Visual attributes and traits")
    caption: str = Field(..., description="Descriptive caption of the image")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")

class PostMatchQuery(BaseModel):
    post_id: str
    post_text: str
    target_category: str
    expected_subject: str

# In-memory storage mock for corpus images and metadata
IMAGE_DB = {
    "img_fox_01": {
        "id": "img_fox_01",
        "metadata": {
            "subject": "red fox",
            "category": "animal",
            "attributes": ["orange fur", "wild", "forest"],
            "caption": "A red fox standing in a forest",
            "confidence": 0.94
        },
        "embedding": [0.12, 0.85, 0.43]
    },
    "img_wolf_01": {
        "id": "img_wolf_01",
        "metadata": {
            "subject": "gray wolf",
            "category": "animal",
            "attributes": ["gray fur", "wild", "snow"],
            "caption": "A gray wolf in the snow",
            "confidence": 0.92
        },
        "embedding": [0.10, 0.20, 0.89]
    }
}

COST_LOG = []

# 2. Mismatch Guard & Safety Layer[cite: 13, 15, 16, 22]
def evaluate_mismatch_guard(post: PostMatchQuery, candidate_metadata: dict, similarity_score: float) -> dict:
    """
    Safety layer combining tag validation, confidence scores, and similarity thresholds.
    Refuses incorrect recommendations with structured explanations[cite: 13, 15, 22].
    """
    # Check confidence threshold
    if candidate_metadata.get("confidence", 0.0) < 0.70:
        return {
            "accepted": False,
            "reason": f"Low confidence score ({candidate_metadata.get('confidence')}): flagged for review."
        }
    
    # Check category or subject mismatch (e.g., wolf on a red fox post)[cite: 13, 15]
    if candidate_metadata.get("category") != post.target_category:
        return {
            "accepted": False,
            "reason": f"Category mismatch: expected {post.target_category}, detected {candidate_metadata.get('category')}."
        }
    
    if candidate_metadata.get("subject") != post.expected_subject:
        return {
            "accepted": False,
            "reason": f"Subject mismatch: expected '{post.expected_subject}', found '{candidate_metadata.get('subject')}'."
        }
    
    # Check semantic similarity threshold[cite: 22]
    if similarity_score < 0.75:
        return {
            "accepted": False,
            "reason": f"Similarity score {similarity_score:.2f} is below the threshold (0.75)."
        }
        
    return {"accepted": True, "reason": "Match approved by safety guard."}

# 3. Cost Tracking Utility[cite: 16, 20, 22]
def track_ai_cost(call_type: str, model: str, cost: float):
    """Records what every AI call costs, attributed per call[cite: 22]."""
    COST_LOG.append({
        "call_type": call_type,
        "model": model,
        "cost": cost
    })

# 4. Endpoints
@app.post("/posts/match-image")
async def match_image_for_post(query: PostMatchQuery):
    """
    Matches a blog post to an image using semantic similarity and the mismatch guard[cite: 13, 15, 16].
    """
    best_match = None
    best_score = 0.0
    guard_result = {"accepted": False, "reason": "No confident match found."}

    # Track token/embedding API costs[cite: 16, 20, 22]
    track_ai_cost(call_type="embedding", model="gemini-embedding-001", cost=0.0001)

    # Evaluate candidates
    for img_id, img_data in IMAGE_DB.items():
        meta = img_data["metadata"]
        
        # Simulate cosine similarity computation[cite: 22]
        similarity = 0.94 if meta["subject"] == query.expected_subject else 0.45
        
        # Run through the mismatch guard safety layer[cite: 13, 15, 22]
        guard_check = evaluate_mismatch_guard(query, meta, similarity)
        
        if guard_check["accepted"] and similarity > best_score:
            best_score = similarity
            best_match = {
                "image_id": img_id,
                "metadata": meta,
                "similarity_score": similarity
            }
            guard_result = guard_check

    if not best_match:
        return {
            "status": "no_match",
            "message": "No confident match found",
            "reason": guard_result["reason"] # Safe rejection explanation[cite: 13, 16]
        }

    return {
        "status": "success",
        "match": best_match,
        "guard_explanation": guard_result["reason"]
    }

@app.get("/admin/costs")
async def get_cost_report():
    """Returns attributed AI execution costs[cite: 16, 20, 22]."""
    total_cost = sum(item["cost"] for item in COST_LOG)
    return {
        "total_cost": total_cost,
        "calls_logged": len(COST_LOG),
        "details": COST_LOG
    }