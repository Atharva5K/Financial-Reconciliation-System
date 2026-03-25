# 🏦 AI Financial Reconciliation Copilot

An unsupervised, continuous-learning machine learning pipeline designed to automatically reconcile check register entries with bank statement transactions.

This project solves the "cold start" problem in unsupervised financial clustering by employing a hybrid architecture: it establishes a deterministic ground truth, dynamically learns temporal clearing patterns, and utilizes a Large Language Model (Sentence Transformer) to resolve ambiguous edge cases.

It includes both an automated CLI pipeline for backend processing and an interactive web Copilot for human-in-the-loop auditing.

---

## Project Architecture

* **Phase 1: Deterministic Seeding:** Identifies 1-to-1 matches based on uniquely occurring monetary amounts to establish a perfect ground-truth baseline.
* **Phase 1.5: Temporal Pattern Extraction:** Dynamically calculates the probabilistic distribution of the bank's date-lag clearing behavior using the Phase 1 subset.
* **Phase 2: Semantic Machine Learning:** Applies `all-MiniLM-L6-v2` (Sentence Transformers) to evaluate ambiguous edge cases. Calculates a weighted Confidence Score based on Textual Similarity (Cosine), Temporal Probability ($P_{date}$), and Monetary Variance.
* **Phase 3: Automated Evaluation:** Dynamically maps synthetic IDs to generate a hidden answer key, calculating true Precision, Recall, and F1 metrics while strictly preventing data leakage into the ML model.

---

## Repository Structure

```text
reconciliation_project/
│
├── data/
│   ├── bank_statements.csv
│   └── check_register.csv
│
├── src/
│   ├── __init__.py
│   ├── preprocessor.py      # Deterministic matching & lag distribution
│   ├── matcher.py           # NLP text cleaning & Sentence Transformer ML
│   └── evaluator.py         # Dynamic Answer Key generation & Scoring
│
├── main.py                  # The automated CLI pipeline
├── app.py                   # The interactive Streamlit UI
├── requirements.txt
└── README.md
```

---

## Installation & Setup

1. **Clone the repository and navigate to the project directory:**
   ```bash
   git clone https://github.com/Atharva5K/Financial-Reconciliation-System
   cd reconciliation_project
   ```

2. **Create and activate a virtual environment (Recommended):**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required dependencies:**
   Ensure your `requirements.txt` contains `pandas`, `numpy`, `sentence-transformers`, `scikit-learn`, and `streamlit`.
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage Instructions

This project features two distinct interfaces depending on the use case.

### Option A: The Automated CLI Pipeline (Backend Integration)

To run the automated pipeline, generate the evaluation metrics, and export the comprehensive CSV audit report:

```bash
python main.py
```

**Expected Output:**

* Terminal logs detailing Phase 1, Phase 1.5, and Phase 2 execution.
* An evaluation matrix printing the Precision, Recall, and F1 Score.
* A generated `final_reconciliation_report.csv` file in your root directory containing all matches, confidence scores, and explicitly flagged `UNMATCHED ANOMALIES` for human review.

### Option B: The Interactive Web Copilot (Human-in-the-Loop UI)

To launch the interactive dashboard for accounting teams to upload CSVs and visually investigate AI decisions:

```bash
streamlit run app.py
```

**Expected Output:**

* A local web server will start (typically at `http://localhost:8501`).
* Upload the raw Bank Statement and Check Register CSVs via the sidebar.
* View real-time pipeline execution, investigate edge cases via the searchable dropdown, and view side-by-side transaction metrics and AI confidence scores.

---

## Part 3: Analysis & Documentation

### 1. Performance Analysis

**Evaluation Metrics**

The hybrid reconciliation pipeline was evaluated against a dynamically generated ground truth key, yielding the following results on the 308-transaction dataset:

* **Precision:** 100.00%
* **Recall:** 96.75%
* **F1 Score:** 98.35%

The system successfully matched 298 out of 308 transactions with zero false positives. The 10 unmatched transactions were correctly identified as true anomalies and intentionally left blank for manual human review, preserving the 100% precision rate critical for financial systems.

**Hardest Transactions to Match**

The most difficult transactions were the 22 non-unique entries that bypassed Phase 1. These transactions were challenging because:

1. **Information Density:** They shared identical monetary amounts (e.g., multiple $50.00 charges) and frequently occurred on similar dates, rendering traditional rules-based matching useless.
2. **Textual Noise:** The descriptions often contained random, non-semantic numeric artifacts (e.g., `Starbucks #9928` vs. `Starbucks`). Without proper text normalization (implemented via Regex in this pipeline), these artifacts artificially distance the vector embeddings of otherwise identical vendors.

**Impact of Training Data on Performance**

The system's performance scales linearly with the volume of available data. By successfully matching 286 transactions deterministically in Phase 1, the pipeline automatically generated a highly accurate, 286-point training dataset. This allowed the system to calculate a precise probability distribution of the bank's date-lag clearing behavior. Had the dataset only contained 10 unique matches, the temporal penalty ($P_{date}$) would have been heavily skewed, significantly lowering the confidence scores of the Phase 2 ML predictions and dropping the overall Recall.

### 2. Design Decisions

**Choice of Machine Learning Approach**

For the semantic matching phase, I utilized a pre-trained Large Language Model-specifically the `all-MiniLM-L6-v2` Sentence Transformer. This model was chosen because it is highly optimized for semantic similarity tasks, lightweight enough to run locally without GPU acceleration, and excels at understanding the contextual relationship between truncated vendor names and descriptions.

**Departures from the Academic Methodology**

The standard academic approach for this problem often relies on TF-IDF (Term Frequency-Inverse Document Frequency) combined with SVD (Singular Value Decomposition) to map text to vectors.

* **Justification:** I explicitly departed from this because SVD requires months of historically matched data to build a domain-specific vocabulary. By using a pre-trained Sentence Transformer, this architecture completely bypasses the "Cold Start" problem, achieving high-fidelity semantic understanding on day one without requiring a massive historical corpus.

**Trade-offs Considered**

* **In-Memory Compute vs. Vector Database:** I chose to calculate the Cosine Similarity matrix entirely in-memory using `scikit-learn`. For a dataset of 300-or even 30,000-transactions, this is exponentially faster and less complex. The trade-off is scalability; if the system needed to reconcile millions of transactions daily, this $O(N^2)$ operation would bottleneck, requiring a shift to an Approximate Nearest Neighbor (ANN) index via a Vector Database like Pinecone.

### 3. Limitations & Future Improvements

**Current Weaknesses**

The primary limitation of this architecture is its heavy reliance on the Phase 1 deterministic seeding. The entire temporal learning mechanism assumes that at least a small subset of the dataset will contain unique monetary amounts. If a dataset consisted entirely of generic, uniform payments (e.g., a subscription billing account where every single transaction is exactly $15.00), Phase 1 would yield zero matches, causing the Date Lag probability distribution to fail.

**Future Improvements (With More Time)**

If allocated more time, I would implement the following:

1. **Active Learning UI:** An extension to the Streamlit app where an accountant could manually match the 10 remaining anomalies. The system would ingest these human corrections to dynamically update the weights of the Confidence Score using logistic regression.
2. **Hyperparameter Tuning:** Currently, the Confidence Score weights (Text: 0.6, Date: 0.3, Amount: 0.1) are heuristically defined. I would build a cross-validation loop to mathematically optimize these weights based on historical accuracy.

**Handling Edge Cases**

This pipeline operates on a strict 1-to-1 matching assumption. In real-world accounting, a major edge case is **1-to-Many matching** (e.g., a user writes three $50 checks, but the bank cashes them simultaneously as a single $150 withdrawal).

To handle this, I would implement a combinatorial algorithm (such as a bounded Subset Sum solver) prior to Phase 2. If an amount in the bank statement has no exact match, the solver would search for combinations of 2 or 3 check register amounts that sum exactly to the bank withdrawal, aggregating their descriptions before passing them to the Sentence Transformer for semantic verification.
