import pandas as pd
import warnings

# Suppress minor warnings from SentenceTransformers
warnings.filterwarnings('ignore')

from src.preprocessor import phase_1_unique_amount_matching, calculate_date_lag_distribution
from src.matcher import find_ml_matches

def export_reconciliation_report(phase_1_df, phase_2_df, filepath='final_reconciliation_report.csv'):
    """
    Combines the deterministic and ML matches into a single, highly readable report
    for Human Quality Control (QC).
    """
    # 1. Tag where the match came from
    phase_1_df['match_type'] = 'Phase 1: Deterministic (Unique Amount)'
    
    if not phase_2_df.empty:
        phase_2_df['match_type'] = 'Phase 2: ML Semantic Match'
        
        # Align Phase 2 columns with Phase 1 columns
        phase_2_df = phase_2_df.rename(columns={
            'bank_desc': 'description_bank',
            'check_desc': 'description_check',
            'bank_amount': 'amount_bank',    
            'check_amount': 'amount_check'
        })
        
        # Combine both sets of matches
        final_report = pd.concat([phase_1_df, phase_2_df], ignore_index=True)
    else:
        final_report = phase_1_df

    # 2. Reorder columns so it's easy to read side-by-side
    columns_to_display = [
        'match_type',
        'confidence_score', 
        'date_bank', 'date_check', 
        'amount', 'amount_bank', 'amount_check', 
        'description_bank', 'description_check',
        'transaction_id_bank', 'transaction_id_check'
    ]
    
    # 3. Safely select columns that actually exist in the merged dataframe
    existing_columns = [col for col in columns_to_display if col in final_report.columns]
    
    # 4. Export to CSV
    final_report[existing_columns].to_csv(filepath, index=False)
    print(f"\n[Success] Full reconciliation report exported to: {filepath}")
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
    print(f" -> Successfully found {len(matched_df)} perfect deterministic matches.")
    
    # --- PHASE 1.5 ---
    print("\n[Phase 1.5] Learning Temporal Patterns...")
    lag_probs, _ = calculate_date_lag_distribution(matched_df)
    
    # --- PHASE 2 ---
    print("\n[Phase 2] Executing ML Semantic Matching on remaining rows...")
    if not bank_unmatched.empty:
        ml_matches_df = find_ml_matches(bank_unmatched, check_unmatched, lag_probs)
        
        # --- THE FIX: Fetch the missing dates for the ML matches ---
        # Look up the bank dates based on the matched IDs
        ml_matches_df = ml_matches_df.merge(
            bank_unmatched[['transaction_id', 'date']].rename(columns={'transaction_id': 'transaction_id_bank', 'date': 'date_bank'}),
            on='transaction_id_bank', how='left'
        )
        # Look up the check register dates based on the matched IDs
        ml_matches_df = ml_matches_df.merge(
            check_unmatched[['transaction_id', 'date']].rename(columns={'transaction_id': 'transaction_id_check', 'date': 'date_check'}),
            on='transaction_id_check', how='left'
        )
        
        print(f" -> AI Model confidently proposed {len(ml_matches_df)} additional matches.")
    else:
        ml_matches_df = pd.DataFrame()
    
    # --- PHASE 3: Human QC Export ---
    print("\n[Phase 3] Generating Human QC Report...")
    export_reconciliation_report(matched_df, ml_matches_df)

    print("\n" + "="*60)
    print("  Pipeline Execution Complete!  ")
    print("="*60)

if __name__ == "__main__":
    main()