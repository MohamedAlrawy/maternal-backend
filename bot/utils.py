import os
import numpy as np
import openai
from typing import List, Tuple

OPENAI_API_KEY = "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"
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable not set")
openai.api_key = OPENAI_API_KEY

# Choose embedding model
EMBEDDING_MODEL = "text-embedding-3-small"  # or "text-embedding-3-large"
CHAT_MODEL = "gpt-4o-mini"  # example; choose one you have access to

def embed_text(text: str) -> List[float]:
    # call OpenAI embeddings
    resp = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return resp["data"][0]["embedding"]

def cosine_similarity(a: List[float], b: List[float]) -> float:
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return -1.0
    num = np.dot(a, b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return -1.0
    return float(num / denom)

def find_best_match(embedding: List[float], candidates: List[Tuple[int, List[float]]]) -> Tuple[int, float]:
    """
    candidates: list of tuples (object_id, embedding_list)
    returns (best_id, best_score)
    """
    best_id = None
    best_score = -1.0
    for oid, emb in candidates:
        score = cosine_similarity(embedding, emb)
        if score > best_score:
            best_score = score
            best_id = oid
    return best_id, best_score

def ask_openai_chat(system_prompt: str, messages: list, max_tokens: int = 512) -> str:
    """
    messages: list of dicts: [{"role": "user"/"assistant"/"system", "content": "..."}]
    """
    print(11)
    resp = openai.ChatCompletion.create(
        model=CHAT_MODEL,
        messages=[{"role":"system","content":system_prompt}] + messages,
        max_tokens=max_tokens,
        temperature=0.2,
    )
    print(22)
    print(resp)
    return resp["choices"][0]["message"]["content"].strip()
