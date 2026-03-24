import pandas as pd

def phase_1_unique_amount_matching(bank_filepath, check_filepath):
    """
    Phase 1: Identifies 1-to-1 transaction matches based strictly on unique monetary amounts.
    This creates our deterministic "ground truth" dataset to train the ML model.
    """
    # 1. Load the synthetic datasets
    df_bank = pd.read_csv(bank_filepath)
    df_check = pd.read_csv(check_filepath)

    # 2. Count the frequency of every amount in both datasets
    bank_amount_counts = df_bank['amount'].value_counts()
    check_amount_counts = df_check['amount'].value_counts()

    # 3. Filter for amounts that appear exactly ONCE in their respective datasets
    unique_bank_amounts = bank_amount_counts[bank_amount_counts == 1].index
    unique_check_amounts = check_amount_counts[check_amount_counts == 1].index

    # 4. Find the intersection (amounts that are uniquely 1:1 across BOTH datasets)
    truly_unique_amounts = set(unique_bank_amounts).intersection(set(unique_check_amounts))

    # 5. Extract these perfect matches from the original dataframes
    bank_matches = df_bank[df_bank['amount'].isin(truly_unique_amounts)]
    check_matches = df_check[df_check['amount'].isin(truly_unique_amounts)]

    # 6. Merge them together to form our "Ground Truth" training data
    # Suffixes ensure we keep the transaction_id from both original files
    ground_truth_matches = pd.merge(
        bank_matches,
        check_matches,
        on='amount',
        suffixes=('_bank', '_check')
    )

    # 7. Isolate the leftover transactions for Phase 2 (The Machine Learning Phase)
    bank_remaining = df_bank[~df_bank['amount'].isin(truly_unique_amounts)].copy()
    check_remaining = df_check[~df_check['amount'].isin(truly_unique_amounts)].copy()

    return ground_truth_matches, bank_remaining, check_remaining


def calculate_date_lag_distribution(matched_df):
    """
    Phase 1.5: Calculates the probabilistic distribution of the date difference (lag)
    between the check register and the bank statement using the ground truth matches.
    """
    # 1. Ensure the date columns are actual datetime objects
    matched_df['date_bank'] = pd.to_datetime(matched_df['date_bank'])
    matched_df['date_check'] = pd.to_datetime(matched_df['date_check'])

    # 2. Calculate the lag in days (Bank Date - Check Register Date)
    matched_df['lag_days'] = (matched_df['date_bank'] - matched_df['date_check']).dt.days

    # 3. Calculate the probability distribution (percentage of occurrence)
    lag_distribution = matched_df['lag_days'].value_counts(normalize=True).to_dict()
    
    # Sort the dictionary by the lag days for a cleaner return object
    lag_distribution = dict(sorted(lag_distribution.items()))

    return lag_distribution, matched_df

# --- Optional Execution Block for Testing ---
if __name__ == "__main__":
    # Test the script independently by running `python src/preprocessor.py`
    # Ensure paths are correct relative to where you run the script
    matched, b_rem, c_rem = phase_1_unique_amount_matching('../data/bank_statements.csv', '../data/check_register.csv')
    print(f"Matched rows: {len(matched)}")
    
    if len(matched) > 0:
        lag_probs, _ = calculate_date_lag_distribution(matched)
        print("Calculated Lag Probabilities:", lag_probs)