# SuperKart — Sales Forecasting & Model Deployment (Flask API + Streamlit, Docker)

**Stack:** Python, pandas, scikit-learn, XGBoost, joblib, Flask, gunicorn, Streamlit, Docker, GitHub Codespaces, Google Colab

**Deployment repository:** [adkanoop-design/superkart-sales-forecast](https://github.com/adkanoop-design/superkart-sales-forecast)

![Streamlit UI returning a live forecast](streamlit_ui.jpg)

## Goal
SuperKart, a chain of supermarkets and food marts across Tier 1–3 cities, needs to forecast the quarterly revenue
of each product in each outlet to plan inventory, procurement and regional sales strategy. A notebook model isn't
enough: the forecast has to be callable from SuperKart's own systems and usable by non-technical planners. So the
project ends in a deployed service, not just a score.

## Approach
- **Data:** 8,763 product–store records from 4 stores. EDA showed that every store is a fixed combination of type,
  size, city tier and age, so `Store_Id` was dropped in favour of the descriptive attributes. That way the model
  can score a store profile it hasn't seen before
- **Feature engineering, designed to match the API contract:** product family from the `Product_Id` prefix
  (FD / DR / NC), store age from the establishment year, and 16 product types grouped into Perishables / Non
  Perishables. The `reg` / `Regular` labelling inconsistency was fixed
- **Models:** Random Forest and XGBoost, each inside a scikit-learn pipeline with one-hot encoding
  (`handle_unknown="ignore"`, so the API can't fail on a new category). Both were tuned with `GridSearchCV`
  (5-fold CV, RMSE scoring), and the test set was held out until the final comparison
- **Serialization:** the whole pipeline (encoder + model) is saved with `joblib`. It was reloaded and checked to
  reproduce the original predictions exactly, and compressed from 35 MB to 11 MB
- **Deployment:** a Flask API (`/v1/predict` for one record, `/v1/predictbatch` for a CSV) served by gunicorn,
  and a Streamlit UI with single and batch forecasts and CSV download. Each runs in its own Docker container,
  connected over a Docker network in a GitHub Codespace. The endpoints were tested in-process with Flask's test
  client before deployment, then called live from Colab

## Results

| Model | Test RMSE | Test R² | Test MAPE | CV RMSE |
|---|---|---|---|---|
| Random Forest (default) | 290.9 | 0.927 | 3.83% | n/a |
| XGBoost (default) | 300.1 | 0.922 | 4.58% | n/a |
| **Random Forest (tuned), final** | **283.2** | **0.930** | **3.67%** | **279.8** |
| XGBoost (tuned) | 284.8 | 0.930 | 3.85% | 282.2 |

- The tuned Random Forest won on every accuracy measure, including cross-validated RMSE, so the ranking isn't
  an accident of one test split. XGBoost had the smaller train–test gap, and the notebook notes it as an equally
  sound production choice given the 1.6-point RMSE margin
- Tuning roughly halved the default models' overfitting (RMSE gap from 183 to 86 for Random Forest)
- **Price is the main lever:** MRP alone carries 42% of the model's feature importance, and weight a further 10%
- **The store matters more than the product:** every product type sells about 4,950 per item in the Tier 1
  Departmental Store and about 1,760 in the Tier 3 Food Mart, a gap of nearly 3× across all 16 categories
- **Shelf space has no link to sales** (r ≈ 0.00), which points to a cheap merchandising experiment
- The live API returned the same predictions as the local test, and unseen categories in the batch file
  (e.g. `Supermarket Type3`) were scored without errors

## Engineering notes
- Colab moved to Python 3.13 during the project, and the template's pinned `numpy 2.0.2` / `pandas 2.2.2` have no
  builds for it. The pins were moved to the nearest compatible releases, and the results were re-verified as
  identical before the final run
- All markdown figures are generated from, or checked against, the notebook's own outputs, so the narrative
  can't contradict the code

## Business recommendations
Plan quarterly procurement store by store from batch forecasts; use the Departmental Store's premium,
higher-MRP assortment as the template for new Tier 1 and Tier 2 formats; raise revenue per product at the
highest-volume Supermarket Type2 instead of adding range; reposition the Tier 3 Food Mart around value packs;
run controlled shelf-space tests on the top forecast categories; retrain quarterly; and add seasonality and
promotion features once dated transactions are available.

## Files
- [`SuperKart_Sales_Forecast_Deployment.ipynb`](SuperKart_Sales_Forecast_Deployment.ipynb) — full notebook with outputs (85 code cells, executed sequentially in Colab)
- [`SuperKart_Full_Code.html`](SuperKart_Full_Code.html) — rendered full-code export
- [`streamlit_ui.jpg`](streamlit_ui.jpg) — the deployed Streamlit UI returning a forecast

*The SuperKart dataset is course material distributed by Great Learning / UT Austin McCombs and is not
redistributed here. The GitHub token used to push the deployment files was read from Colab Secrets and does not
appear in the notebook.*
