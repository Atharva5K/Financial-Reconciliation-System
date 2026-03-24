import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def find_ml_matches(bank_unmatched, check_unmatched, lag_probs):
    """
    Uses Sentence Transformers to find semantic matches for non-unique transactions,
    incorporating the learned date lag distribution.
    """
    print("Loading Sentence Transformer model (this takes a few seconds)...")
    # 'all-MiniLM-L6-v2' is incredibly fast, lightweight, and perfect for standard text
    model = SentenceTransformer('all-MiniLM-L6-v2') 

    # 1. Clean descriptions (handle NaNs just in case)
    bank_descriptions = bank_unmatched['description'].fillna("").tolist()
    check_descriptions = check_unmatched['description'].fillna("").tolist()

    # 2. Generate Vector Embeddings
    print("Generating vector embeddings...")
    bank_embeddings = model.encode(bank_descriptions)
    check_embeddings = model.encode(check_descriptions)

    # 3. Calculate exactly how similar every bank text is to every check text
    text_similarity_matrix = cosine_similarity(bank_embeddings, check_embeddings)

    proposed_matches = []

    # Weights for our Confidence Score Formula
    ALPHA = 0.6  # Text similarity weight
    BETA = 0.3   # Date lag probability weight
    GAMMA = 0.1  # Amount similarity weight

    # 4. Evaluate all possible pairs
    for i, bank_row in enumerate(bank_unmatched.itertuples()):
        best_match = None
        highest_confidence = 0.0

        for j, check_row in enumerate(check_unmatched.itertuples()):
            # A. Textual Similarity (S_text)
            s_text = text_similarity_matrix[i][j]

            # B. Temporal Penalty (P_date)
            # Calculate lag for this specific pair
            date_diff = (pd.to_datetime(bank_row.date) - pd.to_datetime(check_row.date)).days
            # Look up the probability we learned in Phase 1 (default to 0 if we've never seen this lag)
            p_date = lag_probs.get(date_diff, 0.0) 
            # Normalize p_date so the most common lag = 1.0 (Optional, but helps balancing)
            max_prob = max(lag_probs.values()) if lag_probs else 1.0
            normalized_p_date = p_date / max_prob 

            # C. Amount Similarity (S_amount)
            # We penalize based on how many dollars/cents they differ by
            amount_diff = abs(bank_row.amount - check_row.amount)
            # Simple decay: 0 diff = 1.0 score. 5 diff = 0.5 score.
            s_amount = max(0, 1.0 - (amount_diff / 10.0)) 

            # Calculate Final Confidence Score
            confidence = (ALPHA * s_text) + (BETA * normalized_p_date) + (GAMMA * s_amount)

            if confidence > highest_confidence:
                highest_confidence = confidence
                best_match = {
                    'transaction_id_bank': bank_row.transaction_id,
                    'transaction_id_check': check_row.transaction_id,
                    'bank_desc': bank_row.description,
                    'check_desc': check_row.description,
                    'bank_amount': bank_row.amount,
                    'check_amount': check_row.amount,
                    'text_score': s_text,
                    'confidence_score': confidence
                }

        # If the highest confidence passes a basic threshold, we accept it as a match
        if highest_confidence >= 0.5: 
            proposed_matches.append(best_match)

    return pd.DataFrame(proposed_matches)