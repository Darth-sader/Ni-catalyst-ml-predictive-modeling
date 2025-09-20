Ni Catalyst Predictive Modeling

Machine learning analysis for predicting H₂ and CO selectivity in Ni-based catalysts for dry reforming of methane (DRM).


Overview

This project applies machine learning to experimental catalyst data in order to:
Predict H₂ and CO selectivity from CH₄ and CO₂ conversion data.
Compare the performance of five Ni-based catalysts supported on ZrO₂.
Rank catalysts by experimental performance while validating predictions with multiple ML models.


Features

1. Multi-Model Analysis- 6 ML algorithms implemented:
Linear Regression
Random Forest
Gradient Boosting
Support Vector Regression (SVR)
Multi-Layer Perceptron (MLP)
K-Neighbors

2. Catalyst Comparison
Ni-Ba, Ni-Ca, Ni-Mn, Ni-Mg, Ni-K (all supported on ZrO₂).

3. Comprehensive Evaluation
Metrics: R², MAE, RMSE with 5-fold cross-validation.

4. Visualization Tools
Heatmaps for model performance metrics.
Experimental vs predicted scatter plots.
Catalyst ranking bar plots with detailed metrics.

5. Experimental Validation
ML predictions benchmarked against actual catalyst performance data.


Key Findings

Best Catalyst:
Ni-Ba/ZrO₂ demonstrates the highest overall performance across CH₄ conversion, CO₂ conversion, and H₂ selectivity.

Model Insights:
Non-linear models (Random Forest, Gradient Boosting, MLP) generally outperform linear regression.

Predictive Relationships:
Strong R² values confirm reliable conversion → selectivity predictions.
