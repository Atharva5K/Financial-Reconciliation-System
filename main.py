import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from src.preprocessor import phase_1_unique_amount_matching, calculate_date_lag_distribution
from src.matcher import find_ml_matches
from src.evaluator import evaluate_performance

def create_ground_truth_key(bank_filepath, check_filepath):
    """
    Creates the 'Answer Key' for Phase 3 evaluation.
    Exploits the synthetic data pattern where Bank IDs (e.g., 'B1234') 
    match Check IDs (e.g., 'R1234') based on their numeric substring.
    """
    df_bank = pd.read_csv(bank_filepath)
    df_check = pd.read_csv(check_filepath)
    
    # Extract the numeric part (everything after the first character)
    df_bank['numeric_id'] = df_bank['transaction_id'].str[1:]
    df_check['numeric_id'] = df_check['transaction_id'].str[1:]
    
    # Merge them on this matching numeric ID to create the perfect answer key
    ground_truth = pd.merge(
        df_bank[['transaction_id', 'numeric_id']], 
        df_check[['transaction_id', 'numeric_id']], 
        on='numeric_id', 
        suffixes=('_bank', '_check')
    )
    
    return ground_truth, len(df_bank)

def export_reconciliation_report(phase_1_df, phase_2_df, bank_unmatched, check_unmatched, filepath='final_reconciliation_report.csv'):
    """
    Combines deterministic matches, ML matches, and completely unmatched anomalies 
    into a single, highly readable audit report.
    """
    # 1. Format Phase 1 Matches
    phase_1_df['match_type'] = 'Phase 1: Deterministic (Unique Amount)'
    
    # 2. Format Phase 2 Matches
    if not phase_2_df.empty:
        phase_2_df['match_type'] = 'Phase 2: ML Semantic Match'
        phase_2_df = phase_2_df.rename(columns={
            'bank_desc': 'description_bank',
            'check_desc': 'description_check',
            'bank_amount': 'amount_bank',    
            'check_amount': 'amount_check'
        })
        final_report = pd.concat([phase_1_df, phase_2_df], ignore_index=True)
    else:
        final_report = phase_1_df

    # 3. Identify and Format Truly Unmatched Transactions
    matched_bank_ids = phase_2_df['transaction_id_bank'].tolist() if not phase_2_df.empty else []
    matched_check_ids = phase_2_df['transaction_id_check'].tolist() if not phase_2_df.empty else []

    # Filter out the ones the ML model successfully matched
    leftover_bank = bank_unmatched[~bank_unmatched['transaction_id'].isin(matched_bank_ids)].copy()
    leftover_check = check_unmatched[~check_unmatched['transaction_id'].isin(matched_check_ids)].copy()

    frames_to_concat = [final_report]

    if not leftover_bank.empty:
        leftover_bank = leftover_bank.rename(columns={
            'transaction_id': 'transaction_id_bank',
            'date': 'date_bank',
            'amount': 'amount_bank',
            'description': 'description_bank'
        })
        leftover_bank['match_type'] = 'UNMATCHED ANOMALY (Bank Statement)'
        frames_to_concat.append(leftover_bank)

    if not leftover_check.empty:
        leftover_check = leftover_check.rename(columns={
            'transaction_id': 'transaction_id_check',
            'date': 'date_check',
            'amount': 'amount_check',
            'description': 'description_check'
        })
        leftover_check['match_type'] = 'UNMATCHED ANOMALY (Check Register)'
        frames_to_concat.append(leftover_check)

    # Combine everything into the ultimate audit report
    final_report = pd.concat(frames_to_concat, ignore_index=True)

    # 4. Reorder columns for side-by-side readability
    columns_to_display = [
        'match_type', 'confidence_score', 
        'date_bank', 'date_check', 
        'amount', 'amount_bank', 'amount_check', 
        'description_bank', 'description_check',
        'transaction_id_bank', 'transaction_id_check'
    ]
    
    existing_columns = [col for col in columns_to_display if col in final_report.columns]
    
    # 5. Export
    try:
        final_report[existing_columns].to_csv(filepath, index=False)
        print(f" -> Full audit report exported to: {filepath}")
    except PermissionError:
        print(f"\n[ERROR] Could not save '{filepath}' because it is open in another program.")
        
    return final_report


def main():
    print("="*60)
    print("  AI Financial Reconciliation Pipeline Started  ")
    print("="*60)

    bank_file = 'data/bank_statements.csv'
    check_file = 'data/check_register.csv'

    # --- PHASE 1 ---
    print("\n[Phase 1] Executing Unique Amount Matching...")
    matched_df, bank_unmatched, check_unmatched = phase_1_unique_amount_matching(bank_file, check_file)
    print(f" -> Found {len(matched_df)} perfect deterministic matches.")
    
    # --- PHASE 1.5 ---
    print("\n[Phase 1.5] Learning Temporal Patterns...")
    lag_probs, _ = calculate_date_lag_distribution(matched_df)
    
    # --- PHASE 2 ---
    print("\n[Phase 2] Executing ML Semantic Matching on remaining rows...")
    if not bank_unmatched.empty:
        ml_matches_df = find_ml_matches(bank_unmatched, check_unmatched, lag_probs)
        
        ml_matches_df = ml_matches_df.merge(
            bank_unmatched[['transaction_id', 'date']].rename(columns={'transaction_id': 'transaction_id_bank', 'date': 'date_bank'}),
            on='transaction_id_bank', how='left'
        )
        ml_matches_df = ml_matches_df.merge(
            check_unmatched[['transaction_id', 'date']].rename(columns={'transaction_id': 'transaction_id_check', 'date': 'date_check'}),
            on='transaction_id_check', how='left'
        )
        print(f" -> AI Model confidently proposed {len(ml_matches_df)} additional matches.")
    else:
        ml_matches_df = pd.DataFrame()
    
    # --- PHASE 3: EVALUATION & EXPORT ---
    print("\n[Phase 3] Generating Evaluation Metrics and Final Report...")
    
    phase_1_predictions = matched_df[['transaction_id_bank', 'transaction_id_check']].copy()
    if not ml_matches_df.empty:
        phase_2_predictions = ml_matches_df[['transaction_id_bank', 'transaction_id_check']].copy()
    else:
        phase_2_predictions = pd.DataFrame(columns=['transaction_id_bank', 'transaction_id_check'])
        
    all_proposed_matches = pd.concat([phase_1_predictions, phase_2_predictions], ignore_index=True)
    
    ground_truth_df, total_expected = create_ground_truth_key(bank_file, check_file)

    evaluate_performance(all_proposed_matches, ground_truth_df, total_expected)

    export_reconciliation_report(matched_df, ml_matches_df, bank_unmatched, check_unmatched)

    print("\n" + "="*60)
    print("  Pipeline Execution Complete!  ")
    print("="*60)

if __name__ == "__main__":
    main()
