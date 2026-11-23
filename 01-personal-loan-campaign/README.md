# Personal Loan Campaign Prediction — AllLife Bank (Decision Trees)

**Stack:** Python, pandas, numpy, scikit-learn, matplotlib, seaborn

## Goal
Identify the bank's liability (deposit) customers who are most likely to convert to personal-loan customers, so
that AllLife Bank can target its marketing campaign and grow its loan portfolio.

## Approach
- End-to-end ML pipeline on a 5,000-customer dataset: EDA, preprocessing with anomaly correction and a
  documented rationale for outlier treatment, and a stratified train–test split
- Three Decision Tree models compared: a baseline tree, a **pre-pruned** tree (GridSearchCV, 5-fold CV) and a
  **post-pruned** tree (cost-complexity pruning)
- **Recall** used as the primary metric. With only 9.6% of customers accepting a loan, missing a likely
  converter costs more than contacting one who declines

## Results
- Final **post-pruned** model: **90.3% Recall / 92.9% Precision** on unseen holdout data, identifying
  **130 of 144** actual loan-acceptors
- It outperforms the baseline tree (88.9% recall) and the pre-pruned tree (77.1% recall)
- **Income and Education** dominate feature importance (combined > 79%)
- Customers with a **CD account convert at 6× the base rate** (46% vs 7%), a ready-made segment for the campaign

## Files
- [`Personal_Loan_Campaign_Prediction.ipynb`](Personal_Loan_Campaign_Prediction.ipynb) — full notebook with outputs
- [`Personal_Loan_Campaign_Full_Code.html`](Personal_Loan_Campaign_Full_Code.html) — rendered full-code export

*The dataset is course material distributed by Great Learning / UT Austin McCombs and is not redistributed here.*
