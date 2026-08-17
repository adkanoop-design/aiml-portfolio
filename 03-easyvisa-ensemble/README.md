# EasyVisa — Visa Approval Prediction (Ensemble Methods)

**Stack:** Python, pandas, numpy, scikit-learn, XGBoost, matplotlib, seaborn

## Goal
Predict whether a US work-visa application will be certified or denied, so that a case-review
team can triage incoming applications and concentrate human scrutiny where it changes outcomes.

## Approach
- EDA across applicant, employer and wage attributes on 25,480 applications (66.8% certified / 33.2% denied)
- Six classifiers built and compared: Decision Tree, Bagging, Random Forest, AdaBoost, Gradient Boosting, XGBoost
- Each fitted three ways — original, **SMOTE-oversampled**, and **randomly undersampled** — with validation and
  test sets deliberately held at their natural class ratio throughout
- Hyperparameter tuning on the leading candidates; final selection made on validation F1 only, with the test set
  untouched until the final evaluation

## Results
Final model: **tuned AdaBoost**, chosen on the highest validation F1 (0.821) and the smallest train–validation gap (0.002).

| Dataset | Accuracy | Recall | Precision | F1 |
|---|---|---|---|---|
| Training | 0.749 | 0.877 | 0.776 | 0.823 |
| Validation | 0.744 | 0.880 | 0.769 | 0.821 |
| **Test (unseen)** | **0.744** | **0.872** | **0.774** | **0.820** |

- Train, validation and test F1 sit within **0.003** of each other — essentially no overfitting, and a credible
  estimate of production behaviour because the test set influenced no selection or tuning decision
- Beats the naive "certify everything" baseline (66.8% accuracy, zero analytical value), correctly flagging
  **618 of 1,269 denials** the naive rule misses entirely
- **Resampling did not help.** SMOTE moved F1 by 0.002 (noise) and undersampling cost 5.8 F1 points by discarding
  5,990 genuine records. At only 2:1 imbalance there was enough minority data already — establishing that a
  technique *doesn't* work is a real finding, and it kept needless complexity out of the pipeline
- Learned feature importance independently confirmed the EDA: job experience (0.287), Master's (0.200),
  Doctorate (0.157)

## Stated limitation
Performance is asymmetric: F1 0.820 on *Certified* against 0.558 on *Denied*, catching under half of all denials.
The model is therefore recommended as a **triage aid, not an autonomous decision-maker**, with full human review
retained on every predicted denial.

## Files
- [`EasyVisa_Visa_Approval_Prediction.ipynb`](EasyVisa_Visa_Approval_Prediction.ipynb) — full notebook
- [`EasyVisa_Full_Code.html`](EasyVisa_Full_Code.html) — rendered full-code export
