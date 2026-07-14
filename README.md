# AI/ML Project Portfolio — Anoop Krishnan

Applied machine learning and data analytics projects completed in the **Post-Graduate Program in Artificial Intelligence & Machine Learning: Business Applications (PGP-AIML)** at **The University of Texas at Austin — McCombs School of Business** (2025–2026).

**Verified academic e-portfolio:** [mygreatlearning.com/eportfolio/anoop-krishnan](https://www.mygreatlearning.com/eportfolio/anoop-krishnan)
**LinkedIn:** [linkedin.com/in/anoop-krishnan-executive](https://www.linkedin.com/in/anoop-krishnan-executive)

---

## 01 — Personal Loan Campaign Prediction (AllLife Bank)

**Goal:** Identify liability customers most likely to convert to personal-loan customers, so the bank can target its campaign and grow its loan portfolio.

**Approach:** End-to-end ML pipeline on a 5,000-customer dataset — EDA, data preprocessing (anomaly correction, outlier rationale), stratified train-test split, baseline Decision Tree, pre-pruned model (GridSearchCV, 5-fold CV), and post-pruned model (cost-complexity pruning), compared on Recall as the primary metric due to class imbalance (9.6% positive rate).

**Results:**
- Final post-pruned model: **90.3% Recall / 92.9% Precision** on unseen holdout data (130 of 144 actual loan-acceptors correctly identified), outperforming the baseline (88.9% Recall) and the pre-pruned model (77.1% Recall)
- Income and Education dominate feature importance (combined >79%)
- CD-account holders convert at **6x the base rate** (46% vs. 7%) — enabling precision customer segmentation

**Files:** [`Personal_Loan_Campaign_Prediction.ipynb`](01-personal-loan-campaign/Personal_Loan_Campaign_Prediction.ipynb) · [Full-code HTML export](01-personal-loan-campaign/Personal_Loan_Campaign_Full_Code.html)

**Stack:** Python, pandas, numpy, scikit-learn, matplotlib, seaborn

---

## 02 — FoodHub Delivery Operations & Revenue Analytics

**Goal:** Analyze order data for a NYC food aggregator to surface demand, revenue, and delivery-operations insights that inform pricing, promotion, and staffing decisions.

**Approach:** Comprehensive exploratory data analysis (univariate and bivariate) on 1,898 delivery orders — demand patterns, revenue concentration, delivery-time baselines, and customer-feedback gaps.

**Results:**
- Weekend orders account for **71% of total volume**; orders over $20 (29% of volume) generate **60% of platform revenue** ($3,688 of $6,166) — direct input into pricing-tier strategy
- **10.54% of orders exceed 60 minutes** end-to-end; weekday deliveries average 6 minutes slower than weekends — a measurable baseline for delivery-SLA improvement and staffing optimization
- Top 5 restaurants account for 33% of all orders; 38.8% of orders are unrated — a structural gap in the customer feedback loop

**Files:** [Full-code HTML export](02-foodhub-analytics/FoodHub_Analysis_Full_Code.html) · [Presentation (PDF)](02-foodhub-analytics/FoodHub_Presentation.pdf) · [Presentation (PPTX)](02-foodhub-analytics/FoodHub_Presentation.pptx)

**Stack:** Python, pandas, numpy, matplotlib, seaborn

---

## About

16 years in financial-services technology and data (2009–2025), including 11 years at Northern Trust Corporation — Data Governance and Data Domain Lead on the $2B FundMaster book-of-record modernization across a $1.2T+ global custody platform. Now combining enterprise-scale data governance experience with hands-on machine learning.

📫 adkanoop@gmail.com
