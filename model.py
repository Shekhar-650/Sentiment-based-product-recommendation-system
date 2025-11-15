# model.py
import os
import pickle
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

# ---- Config: filenames (adjust if your files are named differently) ----
VECT_FILE = "tfidf_vectorizer.pkl"
SENT_MODEL_FILE = "sentiment_model.pkl"
USER_ITEM_PIVOT = "user_item_pivot.pkl"
ITEM_SIM_FILE = "item_sim.pkl"
USER_SIM_FILE = "user_neighbors.pkl"

# ---- Load artifacts ----
def safe_load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    return pickle.load(open(path, "rb"))

print("Loading artifacts... (this may take a second)")
tfidf = safe_load(VECT_FILE)
sent_model = safe_load(SENT_MODEL_FILE)
user_item = safe_load(USER_ITEM_PIVOT)   # pandas DataFrame pivot (users x items)
item_sim = safe_load(ITEM_SIM_FILE)      # DataFrame (items x items) or numpy
user_sim = safe_load(USER_SIM_FILE)      # DataFrame (users x users) or numpy
print("Artifacts loaded.")

# If item_sim / user_sim are numpy arrays, convert to DataFrame with proper index/columns
if isinstance(item_sim, (np.ndarray,)):
    item_sim = pd.DataFrame(item_sim, index=user_item.columns, columns=user_item.columns)
if isinstance(user_sim, (np.ndarray,)):
    user_sim = pd.DataFrame(user_sim, index=user_item.index, columns=user_item.index)

# Precompute simple popularity (global mean rating) for fallback
item_popularity = user_item.mean(axis=0).fillna(0).sort_values(ascending=False)

# Helper: get top-K candidate items (item-based). returns list of item names
def recommend_items_item_based(user_id, top_k=20):
    if user_id not in user_item.index:
        # fallback: top popular items
        return list(item_popularity.index[:top_k])
    user_ratings = user_item.loc[user_id].dropna()
    scores = {}
    for item, r in user_ratings.items():
        if item not in item_sim.columns:
            continue
        sims = item_sim[item]
        # accumulate weighted score
        for other_item, sim in sims.items():
            if other_item in user_ratings.index:
                continue
            scores.setdefault(other_item, 0.0)
            scores[other_item] += sim * r
    if not scores:
        return list(item_popularity.index[:top_k])
    recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [x[0] for x in recs[:top_k]]

# Helper: compute percent positive sentiment for a product using sentiment model and tfidf
# We will pull reviews from user_item metadata if available. Since pivot doesn't include text,
# we rely on the original dataset saved in a file 'reviews_df.pkl' if present; otherwise, we
# attempt to return popularity as proxy.
REVIEWS_DF_FILE = "reviews_df.pkl"  # optional, not required. If present, expected to include 'name' and 'clean_text' columns.
reviews_df = None
if os.path.exists(REVIEWS_DF_FILE):
    reviews_df = pickle.load(open(REVIEWS_DF_FILE, "rb"))

def product_positive_pct(product_name):
    # If we have raw reviews DF with clean_text, use it.
    if reviews_df is not None and 'clean_text' in reviews_df.columns:
        texts = reviews_df.loc[reviews_df['name'] == product_name, 'clean_text'].dropna().values
        if len(texts) == 0:
            return 0.0
        Xv = tfidf.transform(texts)
        if hasattr(sent_model, "predict_proba"):
            return sent_model.predict_proba(Xv)[:, 1].mean()
        else:
            preds = sent_model.predict(Xv)
            return np.mean(preds)
    # Else fallback: use global rating (normalized)
    if product_name in item_popularity.index:
        # map mean rating (1-5) -> [0,1] roughly
        r = item_popularity.loc[product_name]
        return min(max((r - 1) / 4.0, 0.0), 1.0)
    return 0.0

# Final public function: returns list of top 5 tuples (product, sentiment_score, reason)
def recommend_for_user(username, base="item", top_k=20, top_n=5):
    """
    username: str (must match index of pivot)
    base: 'item' or 'user' (we use item by default)
    Returns: list of dicts: [{'product': name, 'score': float, 'reason': str}, ...]
    """
    # candidate generation
    if base == "item":
        try:
            candidates = recommend_items_item_based(username, top_k=top_k)
        except Exception:
            candidates = list(item_popularity.index[:top_k])
    else:
        # fallback to popularity if user-based not implemented
        candidates = list(item_popularity.index[:top_k])

    # compute sentiment pct for each candidate
    scored = []
    for p in candidates:
        pos_pct = product_positive_pct(p)
        scored.append((p, pos_pct))
    # sort by pos_pct descending and return top_n
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]

    # prepare nice reason text
    results = []
    for product, pct in scored_sorted:
        reason = f"{pct*100:.0f}% positive reviews (estimated)"
        results.append({"product": product, "sentiment_score": float(pct), "reason": reason})
    return results

# small test function (safe)
def available_users(n=10):
    return list(user_item.index[:n])

if __name__ == "__main__":
    print("Model module loaded. Sample users:", available_users(5))
