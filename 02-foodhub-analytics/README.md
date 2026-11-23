# FoodHub — Delivery Operations & Revenue Analytics (Exploratory Data Analysis)

**Stack:** Python, pandas, numpy, matplotlib, seaborn

## Goal
Analyse order data for a New York City food-delivery aggregator to surface demand, revenue and
delivery-operations insights that inform pricing, promotion and staffing decisions.

## Approach
Univariate and bivariate exploratory data analysis on **1,898 delivery orders**, covering cuisine and restaurant
demand, order cost, customer ratings, food-preparation and delivery times, and weekday versus weekend patterns.
The findings were also delivered as a business presentation.

## Results
- **Weekend orders account for 71% of volume**, so staffing and promotions should follow the weekend peak
- Orders over $20 are 29% of volume but generate **60% of revenue** ($3,688 of $6,166), a direct input
  into pricing-tier strategy
- **10.54% of orders take more than 60 minutes** end to end, and weekday deliveries average 6 minutes slower
  than weekend deliveries
- The **top 5 restaurants account for 33% of orders**, and **38.8% of orders are unrated**, a structural gap
  in the feedback loop

## Files
- [`FoodHub_Analysis.ipynb`](FoodHub_Analysis.ipynb) — full notebook with outputs
- [`FoodHub_Analysis_Full_Code.html`](FoodHub_Analysis_Full_Code.html) — rendered full-code export
- [`FoodHub_Presentation.pdf`](FoodHub_Presentation.pdf) / [`FoodHub_Presentation.pptx`](FoodHub_Presentation.pptx) — business presentation

*The dataset is course material distributed by Great Learning / UT Austin McCombs and is not redistributed here.*
