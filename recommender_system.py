"""
RetailRocket E-Commerce Product Recommendation System
======================================================
Architecture: Item-to-Item Collaborative Filtering vs Popularity Baseline
Author: Likith Yadav
Repository: product_recommendation_using_ML

This module implements:
1. Ingestion of implicit user interactions (view, addtocart, transaction)
2. Intent-weighted interaction matrix construction
3. Item-Item Cosine Similarity computation
4. Top-K personalized candidate recommendation ranking
5. Held-out offline validation (Hit Rate@10, Precision@10, Recall@10)
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
TOP_ITEMS = 300
MAX_USERS = 3000
TOP_K = 10
RANDOM_STATE = 42

EVENT_WEIGHTS = {
    "view": 1.0,
    "addtocart": 2.0,
    "transaction": 3.0
}


def load_and_preprocess_data(events_path="events.csv"):
    """
    Loads raw interaction events and applies intent weighting.
    """
    if not os.path.exists(events_path):
        raise FileNotFoundError(f"Events dataset not found: {events_path}")

    print(f"Loading interactions from: {events_path}")
    df = pd.read_csv(events_path, usecols=["visitorid", "itemid", "event"])
    df = df.dropna(subset=["visitorid", "itemid", "event"])
    
    # Map event types to intent weights
    df = df[df["event"].isin(EVENT_WEIGHTS.keys())].copy()
    df["weight"] = df["event"].map(EVENT_WEIGHTS)
    
    print(f"Total usable interaction logs: {len(df):,}")
    print(f"Unique visitors: {df['visitorid'].nunique():,}, Unique items: {df['itemid'].nunique():,}")
    
    # Aggregate duplicate interactions per user-item pair
    user_item_interactions = df.groupby(["visitorid", "itemid"], as_index=False)["weight"].sum()
    
    # Filter to active core subset for robust evaluation
    popular_items = (
        user_item_interactions.groupby("itemid")["weight"]
        .sum()
        .nlargest(TOP_ITEMS)
        .index
    )
    filtered = user_item_interactions[user_item_interactions["itemid"].isin(popular_items)]
    
    active_users = (
        filtered.groupby("visitorid")["itemid"]
        .count()
        .nlargest(MAX_USERS)
        .index
    )
    filtered = filtered[filtered["visitorid"].isin(active_users)].copy()
    
    print(f"Filtered evaluation subset: {filtered['visitorid'].nunique()} users, {filtered['itemid'].nunique()} items.")
    return filtered


def build_user_item_matrix(interactions_df):
    """
    Constructs user-item matrix where values represent interaction affinity.
    """
    matrix = interactions_df.pivot(index="visitorid", columns="itemid", values="weight").fillna(0.0)
    return matrix


def compute_item_similarity(user_item_matrix):
    """
    Calculates pairwise cosine similarity between item interaction vectors.
    """
    # Columns are items, transpose to get (items x users)
    item_vectors = user_item_matrix.T.values
    sim_matrix = cosine_similarity(item_vectors)
    sim_df = pd.DataFrame(sim_matrix, index=user_item_matrix.columns, columns=user_item_matrix.columns)
    return sim_df


def evaluate_recommenders(user_item_matrix, sim_df, top_k=TOP_K):
    """
    Performs leave-one-out (held-out) offline evaluation:
    For each eligible user with >= 2 interactions, holds out 1 item and checks
    if the models include it in the Top-K recommendations.
    """
    eligible_users = user_item_matrix[(user_item_matrix > 0).sum(axis=1) >= 2].index
    print(f"Eligible evaluation users (>=2 interactions): {len(eligible_users)}")

    popularity_scores = user_item_matrix.sum(axis=0)
    pop_top_k = popularity_scores.nlargest(top_k).index.tolist()

    pop_hits, sim_hits = 0, 0
    total_eval = 0

    for user in eligible_users:
        user_row = user_item_matrix.loc[user]
        interacted_items = user_row[user_row > 0].index.tolist()

        # Hold out the last interacted item
        held_out_item = interacted_items[-1]
        train_items = interacted_items[:-1]

        # 1. Popularity Evaluation
        if held_out_item in pop_top_k:
            pop_hits += 1

        # 2. Item-Item Collaborative Filtering
        sub_sim = sim_df.loc[train_items].sum(axis=0)
        # Mask already interacted train items
        sub_sim = sub_sim.drop(train_items, errors="ignore")
        cf_recommendations = sub_sim.nlargest(top_k).index.tolist()

        if held_out_item in cf_recommendations:
            sim_hits += 1

        total_eval += 1

    pop_hit_rate = pop_hits / total_eval
    sim_hit_rate = sim_hits / total_eval

    # Precision@K = Hits / (total_eval * K), Recall@K = Hits / total_eval (for 1 held-out item)
    pop_precision = pop_hit_rate / top_k
    sim_precision = sim_hit_rate / top_k

    metrics = [
        {
            "Model": "Popularity-Based",
            "Hit Rate": pop_hit_rate,
            "Precision@10": pop_precision,
            "Recall@10": pop_hit_rate
        },
        {
            "Model": "Item Similarity",
            "Hit Rate": sim_hit_rate,
            "Precision@10": sim_precision,
            "Recall@10": sim_hit_rate
        }
    ]
    return pd.DataFrame(metrics)


def get_recommendations_for_user(user_id, user_item_matrix, sim_df, top_k=TOP_K):
    """
    Generates Top-K personalized recommendations for a specific visitorid.
    """
    if user_id not in user_item_matrix.index:
        # Fallback to popularity
        return user_item_matrix.sum(axis=0).nlargest(top_k).index.tolist()

    user_row = user_item_matrix.loc[user_id]
    interacted = user_row[user_row > 0].index.tolist()

    if not interacted:
        return user_item_matrix.sum(axis=0).nlargest(top_k).index.tolist()

    scores = sim_df.loc[interacted].sum(axis=0).drop(interacted, errors="ignore")
    return scores.nlargest(top_k).index.tolist()


if __name__ == "__main__":
    print("=" * 60)
    print("E-COMMERCE PRODUCT RECOMMENDATION SYSTEM - INFERENCE PIPELINE")
    print("=" * 60)

    # Check if pre-saved model exists for fast inference
    model_path = "case1_recommender.joblib"
    if os.path.exists(model_path):
        print(f"Loading trained model bundle from {model_path}...")
        bundle = joblib.load(model_path)
        u_matrix = bundle["user_item_matrix"]
        i_sim = bundle["item_similarity"]
        sample_user = u_matrix.index[0]
        recs = get_recommendations_for_user(sample_user, u_matrix, i_sim)
        print(f"Top 10 recommended items for visitor {sample_user}:")
        for rank, item in enumerate(recs, 1):
            print(f"  Rank #{rank}: Product ID {item}")
    else:
        print("Pre-saved model not found. Run training script with events.csv dataset.")
