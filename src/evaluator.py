def evaluate_performance(proposed_matches_df, ground_truth_df, total_expected_matches):
    """
    Calculates Precision, Recall, and F1 Score for the proposed matches.
    
    Args:
        proposed_matches_df: DataFrame containing the system's predicted matches.
                             Expected columns: ['transaction_id_bank', 'transaction_id_check']
        ground_truth_df: DataFrame containing the actual correct matches (The Answer Key).
                         Expected columns: ['transaction_id_bank', 'transaction_id_check']
        total_expected_matches: Integer representing how many matches the system SHOULD have found.
    """
    print("\n--- Running Evaluation Metrics ---")
    
    # 1. Convert DataFrames to sets of tuples for lightning-fast comparison
    # e.g., {('bank_id_1', 'check_id_1'), ('bank_id_2', 'check_id_2')}
    proposed_set = set(zip(proposed_matches_df['transaction_id_bank'], 
                           proposed_matches_df['transaction_id_check']))
    
    ground_truth_set = set(zip(ground_truth_df['transaction_id_bank'], 
                               ground_truth_df['transaction_id_check']))

    # 2. Calculate the core metrics
    # True Positives: Matches the system made that are actually correct
    correctly_matched = len(proposed_set.intersection(ground_truth_set))
    
    # All System Matches: Every match the system confidently proposed
    all_system_matches = len(proposed_set)
    
    # 3. Calculate Precision, Recall, and F1
    # Handle division by zero just in case the model found 0 matches
    precision = correctly_matched / all_system_matches if all_system_matches > 0 else 0.0
    recall = correctly_matched / total_expected_matches if total_expected_matches > 0 else 0.0
    
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    # 4. Print the final report
    print(f"Total Expected Matches : {total_expected_matches}")
    print(f"Total System Matches   : {all_system_matches}")
    print(f"Correct System Matches : {correctly_matched}")
    print("-" * 32)
    print(f"Precision : {precision:.2%}")
    print(f"Recall    : {recall:.2%}")
    print(f"F1 Score  : {f1_score:.2%}")
    print("-" * 32)

    return precision, recall, f1_score