import streamlit as st
import pandas as pd
import os
import warnings

# Suppress minor warnings for a cleaner terminal
warnings.filterwarnings('ignore')

# Import your existing pipeline functions
from src.preprocessor import phase_1_unique_amount_matching, calculate_date_lag_distribution
from src.matcher import find_ml_matches

# --- Page Config ---
st.set_page_config(page_title="AI Reconciliation Copilot", layout="wide")
st.title("🏦 AI Financial Reconciliation Copilot")
st.markdown("Upload your bank statement and check register to automatically match transactions using semantic AI.")

# --- Session State to store results ---
if 'final_report' not in st.session_state:
    st.session_state.final_report = None
if 'check_raw' not in st.session_state:
    st.session_state.check_raw = None

# --- Sidebar: File Uploaders ---
with st.sidebar:
    st.header("1. Upload Data")
    bank_file = st.file_uploader("Upload Bank Statement (CSV)", type=['csv'])
    check_file = st.file_uploader("Upload Check Register (CSV)", type=['csv'])
    
    run_button = st.button("Run AI Reconciliation", type="primary", use_container_width=True)

# --- Main Logic: Running the Pipeline ---
if run_button:
    if bank_file and check_file:
        with st.spinner("Initializing AI Pipeline..."):
            # Save uploaded files temporarily so your existing functions can read them
            with open("temp_bank.csv", "wb") as f:
                f.write(bank_file.getbuffer())
            with open("temp_check.csv", "wb") as f:
                f.write(check_file.getbuffer())

            # 1. Store raw check data for the dropdown later
            st.session_state.check_raw = pd.read_csv("temp_check.csv")

            # 2. Run Phase 1
            st.write("✅ Extracting deterministic matches...")
            matched_df, bank_unm, check_unm = phase_1_unique_amount_matching("temp_bank.csv", "temp_check.csv")
            
            # 3. Run Phase 1.5
            st.write("✅ Learning temporal clearing patterns...")
            lag_probs, _ = calculate_date_lag_distribution(matched_df)
            
            # 4. Run Phase 2
            st.write("🧠 Running Semantic Machine Learning on edge cases (This takes a moment)...")
            if not bank_unm.empty:
                ml_matches_df = find_ml_matches(bank_unm, check_unm, lag_probs)
                
                # Fetch dates for ML matches
                ml_matches_df = ml_matches_df.merge(
                    bank_unm[['transaction_id', 'date']].rename(columns={'transaction_id': 'transaction_id_bank', 'date': 'date_bank'}),
                    on='transaction_id_bank', how='left'
                )
                ml_matches_df = ml_matches_df.merge(
                    check_unm[['transaction_id', 'date']].rename(columns={'transaction_id': 'transaction_id_check', 'date': 'date_check'}),
                    on='transaction_id_check', how='left'
                )
            else:
                ml_matches_df = pd.DataFrame()

            # 5. Combine Results & Standardize Columns
            matched_df['match_type'] = 'Phase 1: Deterministic'
            matched_df['confidence_score'] = 1.0  # Deterministic matches are 100% confident
            
            # Ensure Phase 1 has the split amount columns for the UI
            matched_df['amount_bank'] = matched_df['amount']
            matched_df['amount_check'] = matched_df['amount']
            
            if not ml_matches_df.empty:
                ml_matches_df['match_type'] = 'Phase 2: ML Semantic'
                ml_matches_df = ml_matches_df.rename(columns={
                    'bank_desc': 'description_bank', 'check_desc': 'description_check',
                    'bank_amount': 'amount_bank', 'check_amount': 'amount_check'
                })
                final_combined = pd.concat([matched_df, ml_matches_df], ignore_index=True)
            else:
                final_combined = matched_df
            
            # Save to session state so it persists during interaction
            st.session_state.final_report = final_combined
            
            # Cleanup temp files
            os.remove("temp_bank.csv")
            os.remove("temp_check.csv")
            
            st.success(f"Pipeline Complete! Successfully matched {len(final_combined)} transactions.")
    else:
        st.error("Please upload both CSV files to begin.")

st.divider()

# --- Interactive Viewer ---
if st.session_state.final_report is not None:
    st.header("2. Investigate Matches")
    
    # Create a user-friendly dropdown list: "ID | Description ($Amount)"
    check_raw = st.session_state.check_raw
    check_options = check_raw.apply(
        lambda row: f"{row['transaction_id']} | {row['description']} (${row['amount']:.2f})", axis=1
    ).tolist()
    
    # Dropdown selector
    selected_check_string = st.selectbox("Select a Check Register Transaction to inspect:", check_options)
    
    # Extract the ID from the selected string (everything before the first ' | ')
    selected_check_id = selected_check_string.split(" | ")[0]
    
    # Find the match in our final report
    match_row = st.session_state.final_report[st.session_state.final_report['transaction_id_check'] == selected_check_id]
    
    if not match_row.empty:
        match_data = match_row.iloc[0]
        
        st.subheader("AI Match Results")
        
        # Top Row: Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Match Confidence", f"{match_data['confidence_score'] * 100:.1f}%")
        col2.metric("Algorithm Used", match_data['match_type'])
        col3.metric("Bank Amount", f"${match_data['amount_bank']:.2f}")
        
        st.write("") # Spacer
        
        # Bottom Row: Side-by-Side Comparison
        col_check, col_bank = st.columns(2)
        
        with col_check:
            st.info("🧾 **Check Register (Query)**")
            st.write(f"**ID:** {match_data['transaction_id_check']}")
            st.write(f"**Date:** {match_data['date_check']}")
            st.write(f"**Description:** {match_data['description_check']}")
            
        with col_bank:
            st.success("🏦 **Bank Statement (Match)**")
            st.write(f"**ID:** {match_data['transaction_id_bank']}")
            st.write(f"**Date:** {match_data['date_bank']}")
            st.write(f"**Description:** {match_data['description_bank']}")
            
    else:
        st.warning("⚠️ The AI pipeline marked this transaction as an UNMATCHED ANOMALY. No confident match was found in the bank statement.")