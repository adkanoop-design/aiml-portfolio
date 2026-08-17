# ReneWind — Predictive Maintenance for Wind Turbine Generators (Neural Networks)

**Stack:** Python, TensorFlow/Keras, pandas, numpy, scikit-learn, matplotlib, seaborn

## Goal
Predict generator failure from sensor data so that ReneWind can replace reactive repair with scheduled inspection.
The cost structure is asymmetric — a failure-driven replacement costs roughly **20x** an inspection — so the
business problem is a cost problem, not an accuracy problem.

## Approach
- 20,000 training observations, 40 anonymised sensor features, **5.55% failure rate** (1,110 of 20,000)
- Class imbalance ruled out accuracy as a metric and made stratified splitting essential; **recall** selected as the
  primary metric, with realised cost as the economic check
- **Eight neural network architectures** built and compared, isolating one lever at a time: network depth,
  SGD vs Adam, dropout regularisation, and class weighting
- Decision-threshold sweep performed on the validation set only, then applied to the test set

## Results
Final model: **Model 7** — three hidden layers (128/64/32), Adam, dropout 0.3, class weights, 15,617 parameters.

On 5,000 held-out test generators:

| Metric | Value |
|---|---|
| Recall | **0.8794** — 248 of 282 real failures caught |
| Precision | 0.9018 — 27 false alarms across 4,718 healthy units |
| Cost | **28,200 → 10,975 units, a 61.1% saving** |

- Validation recall (0.8964) transferred to test with a drop of only 0.017 — a dependable estimate, not an
  optimistic one
- **Techniques did not simply add up.** Class weighting gave +7.2 recall points under SGD but only +1.8 under Adam,
  and Adam-plus-class-weights without regularisation overfitted badly (train–validation gap 0.1148). Dropout
  repaired it, cutting the gap to 0.0372. The finding is the *combination*, not the depth
- **Depth bought nothing.** Model 7 (15,617 parameters) and Model 6 (4,737) caught the identical 199 of 222
  validation failures. The constraint is data, not model capacity
- **Threshold tuning returned zero saving — and that is the interesting result.** The 20:1 cost asymmetry pushes the
  optimal cut-off down while the 17:1 class weighting already applied in training pushes it up; the two effects
  very nearly cancel and the optimum lands on the 0.5 default. The cost curve is also flat within 100 units across
  0.35–0.75, so ReneWind can move the dial to suit inspection capacity without damaging cost performance

## Note on method
Recall was the right metric for **selecting between models**; realised cost was the right criterion for **setting the
operating point** of the chosen model. Those are distinct questions and were answered separately. Model selection was
close — Model 6 would have won on a parsimony or generalisation-gap tie-break, and that is recorded in the notebook
rather than presented as decisive.

## Files
- [`ReneWind_Predictive_Maintenance.ipynb`](ReneWind_Predictive_Maintenance.ipynb) — full notebook with outputs
- [`ReneWind_Rerunnable_Clean.ipynb`](ReneWind_Rerunnable_Clean.ipynb) — clean re-runnable version
- [`ReneWind_Full_Code.html`](ReneWind_Full_Code.html) — rendered full-code export
