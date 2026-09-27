# AI/ML Project Portfolio — Anoop Krishnan

Applied machine learning, generative AI and data analytics projects completed in the **Post-Graduate Program in
Artificial Intelligence & Machine Learning: Business Applications (PGP-AIML)** at **The University of Texas at
Austin — McCombs School of Business** (2025–2026).

**Verified academic e-portfolio:** [mygreatlearning.com/eportfolio/anoop-krishnan](https://www.mygreatlearning.com/eportfolio/anoop-krishnan)
**LinkedIn:** [linkedin.com/in/anoop-krishnan-executive](https://www.linkedin.com/in/anoop-krishnan-executive)

**Start here:** [05 — Medical Assistant](05-medical-assistant-rag) (retrieval-augmented generation with an
open-weights LLM), [04 — ReneWind](04-renewind-neural-networks) (neural networks) and
[03 — EasyVisa](03-easyvisa-ensemble) (ensemble methods) are the most technically involved.

---

## 01 — Personal Loan Campaign Prediction (AllLife Bank)

**Goal:** Identify liability customers most likely to convert to personal-loan customers, so the bank can target its
campaign and grow its loan portfolio.

**Approach:** End-to-end ML pipeline on a 5,000-customer dataset — EDA, preprocessing (anomaly correction, outlier
rationale), stratified train-test split, baseline Decision Tree, pre-pruned model (GridSearchCV, 5-fold CV), and
post-pruned model (cost-complexity pruning), compared on Recall as the primary metric given 9.6% class imbalance.

**Results:**
- Final post-pruned model: **90.3% Recall / 92.9% Precision** on unseen holdout data (130 of 144 actual
  loan-acceptors identified), outperforming the baseline (88.9%) and the pre-pruned model (77.1%)
- Income and Education dominate feature importance (combined >79%)
- CD-account holders convert at **6x the base rate** (46% vs. 7%)

**Stack:** Python, pandas, numpy, scikit-learn, matplotlib, seaborn

---

## 02 — FoodHub Delivery Operations & Revenue Analytics

**Goal:** Analyse order data for a NYC food aggregator to surface demand, revenue and delivery-operations insights
informing pricing, promotion and staffing decisions.

**Approach:** Univariate and bivariate exploratory data analysis on 1,898 delivery orders.

**Results:**
- Weekend orders account for **71% of volume**; orders over $20 (29% of volume) generate **60% of revenue**
  ($3,688 of $6,166) — direct input into pricing-tier strategy
- **10.54% of orders exceed 60 minutes** end-to-end; weekday deliveries average 6 minutes slower than weekends
- Top 5 restaurants account for 33% of orders; 38.8% unrated — a structural gap in the feedback loop

**Stack:** Python, pandas, numpy, matplotlib, seaborn

---

## 03 — EasyVisa: Visa Approval Prediction (Ensemble Methods)

**Goal:** Predict whether a US work-visa application will be certified or denied, to triage case review.

**Approach:** Six classifiers (Decision Tree, Bagging, Random Forest, AdaBoost, Gradient Boosting, XGBoost) each
fitted on original, **SMOTE-oversampled** and **undersampled** training data, with validation and test sets held at
their natural class ratio. Final selection on validation F1 only.

**Results:**
- Final tuned **AdaBoost**: test **F1 0.820, Recall 0.872, Precision 0.774** — train/validation/test F1 within
  **0.003**, indicating essentially no overfitting
- Beats the naive baseline (66.8% accuracy), catching **618 of 1,269 denials** it would miss
- **Resampling did not help** — SMOTE moved F1 by 0.002, undersampling cost 5.8 points. Documented as a finding
- Limitation stated openly: recall on the *Denied* class is 0.487, so the model is a **triage aid, not an
  autonomous decision-maker**

**Details:** [03-easyvisa-ensemble](03-easyvisa-ensemble) · **Stack:** Python, scikit-learn, XGBoost

---

## 04 — ReneWind: Predictive Maintenance for Wind Turbine Generators (Neural Networks)

**Goal:** Predict generator failure from sensor data to replace reactive repair with scheduled inspection, where a
failure-driven replacement costs roughly **20x** an inspection.

**Approach:** **Eight neural network architectures** on 20,000 observations with 40 sensor features and a 5.55%
failure rate — isolating depth, SGD vs Adam, dropout and class weighting one lever at a time. Recall as the
selection metric, realised cost as the economic check, decision-threshold sweep on validation only.

**Results:**
- Final model (3 hidden layers, Adam, dropout 0.3, class weights): test **Recall 0.8794** — 248 of 282 failures
  caught — at **Precision 0.9018**, with 27 false alarms across 4,718 healthy units
- **Cost reduced from 28,200 to 10,975 units — a 61.1% saving**
- Validation-to-test recall drop of only 0.017
- Techniques did not simply add up: class weighting gave +7.2 recall points under SGD but +1.8 under Adam, and
  needed dropout to stay stable. Depth bought nothing — a 4,737-parameter model matched a 15,617-parameter one
- Threshold tuning returned **zero saving**, because the 20:1 cost asymmetry and the 17:1 class weighting cancel out

**Details:** [04-renewind-neural-networks](04-renewind-neural-networks) · **Stack:** Python, TensorFlow/Keras, scikit-learn

---

## 05 — Medical Assistant: Clinical Knowledge Assistant with RAG (Generative AI)

**Goal:** Give clinicians fast, verifiable answers from the 4,114-page *Merck Manual* instead of an LLM's memory —
complete, consistent between users, and traceable to the page that justifies them.

**Approach:** Five clinical questions answered three ways so each step could be measured against the last: a raw
**Mistral-7B-Instruct** baseline (4-bit GGUF on a T4), **five prompt-engineering / decoding combinations**, then a
full **RAG pipeline** — watermark cleaning, 17,047 overlapping chunks, MiniLM embeddings, persistent ChromaDB index,
answer-only-from-context prompt — with **six chunking / retriever / LLM configurations** compared and the retrieved
page numbers tabulated for every run. An LLM-as-a-judge scored groundedness and relevance.

**Results:**
- Final configuration (`k = 4`, 512 tokens, strict structured prompt) scored **5/5 groundedness and 5/5 relevance**
  on all five questions, every answer built from cited manual pages
- The raw LLM truncated four of five answers and never reached the treatment part of one question; prompting fixed
  completeness but not knowledge — the "constrained" prompt still recommended a treatment trauma guidance does not
  support, with no way to verify it
- **Retrieval was the limiting factor, not the model:** `k = 3` pulled neighbouring-chapter noise into the sepsis
  answer and missed the brain-injury chapter entirely; a finer 500-character index recovered both. Documented as the
  key finding, with hybrid retrieval and a re-ranker recommended next
- Self-evaluation shown to be lenient — a context-based judge cannot detect a retrieval miss — so an independent
  judge, retrieval-recall checks and clinician review are recommended before any clinical use

**Details:** [05-medical-assistant-rag](05-medical-assistant-rag) · **Stack:** Python, llama-cpp-python, LangChain, ChromaDB, sentence-transformers

---

## About

16 years in financial-services technology and data (2009–2025), including 11 years at Northern Trust Corporation —
Data Governance and Data Domain Lead on the $2B FundMaster book-of-record modernization, and delivery lead on
settlement, reference-data and conversational-AI programmes across a global custody platform. Now combining
enterprise-scale data governance experience with hands-on machine learning.

Currently completing the Data Engineering Academy professional programme (2026–2027).

📫 adkanoop@gmail.com

## Note on datasets
Source datasets are the property of Great Learning / UT Austin McCombs and are not redistributed here. Notebooks
include all outputs, and the HTML exports render the complete analysis end to end.
