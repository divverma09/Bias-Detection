# Bias Detection, Mitigation, and Explainability in Machine Learning Models

This repository contains a Jupyter Notebook (`Bias Detection, Mitigation, and Explainability in Machine Learning Models.ipynb`) demonstrating methodologies to detect and mitigate bias in Machine Learning models, alongside model explainability using SHAP. It approaches fairness from both demographic parity and equalized odds perspectives, investigating the trade-offs between model accuracy and fairness.

## Project Overview

Algorithmic bias can manifest in models by treating certain privileged groups more favorably than unprivileged groups based on sensitive attributes (e.g., race, sex, age). This project evaluates fairness metrics on baseline models and explores two different mitigation strategies: post-processing (Threshold Optimizer) and in-processing (Adversarial Debiasing). It then leverages SHAP to explain how different features drive model predictions.

## Datasets

The notebook evaluates bias using two popular tabular datasets widely analyzed for algorithmic fairness:

1. **Adult Census Income Dataset (UCI)**: Used for predicting whether an individual earns over $50K a year based on demographic and employment features. (Sensitive attribute: `sex`)
2. **COMPAS Recidivism Dataset (ProPublica)**: Used for predicting two-year recidivism rates among criminal defendants. (Sensitive attributes: `race`, `sex`)

## Libraries and Tools

The implementation utilizes the following primary Python libraries:

- **Data Manipulation & Visualization:** `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn`, `tensorflow`
- **Fairness & Bias Mitigation:**
  - `fairlearn` (Threshold Optimizer, Fairness Metrics)
  - `aif360` / AI Fairness 360 (BinaryLabelDataset, Adversarial Debiasing, ClassificationMetric)
- **Explainability:** `shap`

## Methodology

### 1. Baseline Model Training

Baseline predictive models are trained without any explicit fairness constraints to establish underlying performance and bias metrics.

- **Models:** Logistic Regression and Random Forest Classifier.
- **Evaluation:** Standard classification metrics alongside custom fairness metrics:
  - Demographic Parity Difference (DP)
  - Disparate Impact (DI)
  - Equalized Odds Difference (EO)

### 2. Bias Mitigation Techniques

To reduce observed biases in predictions, two main techniques are explored:

- **Post-processing Constraint Optimization:** Using `fairlearn`'s `ThresholdOptimizer` to adjust decision thresholds conditionally based on sensitive features in order to enforce Demographic Parity or Equalized Odds.
- **In-processing with Adversarial Networks:** Using `aif360`'s `AdversarialDebiasing` built on TensorFlow. This attempts to minimize predictive loss while simultaneously maximizing an adversary's inability to predict the sensitive attribute from the predicted labels.

### 3. Model Explainability

- **SHAP (SHapley Additive exPlanations):** Applying `shap.TreeExplainer` and `shap.summary_plot` to interpret what features have the most influence on individual predictions and understand global feature importance across both the Adult Income and COMPAS datasets.

## How to Run

1. Clone this repository to your local machine.
2. Install the necessary dependencies (listed in the top cell of the notebook):
   ```bash
   pip install shap numpy numba pandas scikit-learn matplotlib seaborn fairlearn aif360 tensorflow
   ```
3. Run Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook
   ```
4. Open the `Bias Detection, Mitigation, and Explainability in Machine Learning Models.ipynb` notebook and run the cells interactively. Ensure that your Python environment meets the required dependencies.

## Output and Results

The notebook provides:

- Calculations of disparate impact, demographic parity, and equalized odds for unmitigated predictive models.
- Side-by-side fairness comparisons demonstrating how `ThresholdOptimizer` and `AdversarialDebiasing` affect demographic parity and equalized odds metrics.
- Epoch training logs showing convergence of the primary classifier loss opposing the adversarial loss.
- SHAP summary plots detailing the directional impact of features like `age`, `education-num`, `capital-gain`, and `sex` or `race` on predictions.