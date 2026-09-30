# Risk-Aware Routing for a Portfolio of Large Language Models

Code and selected experimental outputs for the MSc dissertation *Risk-Aware Routing for a Portfolio of Large Language Models: Modelling Predictive and Decision Uncertainty for Cost-Effective LLM Selection*.

The project looks at how to route each query to one model out of a portfolio of LLMs, trading off predictive performance against inference cost. Alongside a plain utility-only router, the repository implements a risk-aware policy that penalizes predictive uncertainty, estimated from an aligned bootstrap ensemble.


## Repository Structure

```text
.
├── config/
├── figures/
├── results/
├── src/
├── .gitignore
├── README.md
└── requirements.txt
```

## Source Code

Main scripts in `src/`:

```text
collect_outputs.py
create_splits.py
data_quality.py
check_missing.py
load_dataset.py
evaluation.py
utility.py
feature_engineering.py

train_utility_router.py
train_risk_aware_router.py
validate_risk_signal.py
evaluate_test_router.py

additional_baselines.py
category_baseline_train.py

train_risk_aware_router_no_category.py
evaluate_test_router_no_category.py

statistical_test_router.py
analyse_risk_mechanism.py
category_analysis.py
prediction_ranking_errors.py

build_final_comparison.py
plot.py
```

## Environment

Python 3.11 is recommended. Install dependencies with:

```bash
pip install -r requirements.txt
```

Main dependencies: NumPy, pandas, SciPy, scikit-learn, XGBoost, sentence-transformers, matplotlib, datasets, openai, python-dotenv.

## API Configuration

Model output collection goes through OpenRouter. Create a `.env` file in the repository root:

```env
OPENROUTER_API_KEY=your_key_here
```
> Running `collect_outputs.py` may incur API costs.

## Experimental Pipeline

Dataset loading is handled by `src/load_dataset.py` and is used by the data collection scripts.

A typical experimental workflow is:

**1. Collect model outputs**
```bash
python src/collect_outputs.py
```

**2. Check for missing outputs**
```bash
python src/check_missing.py
```

**3. Evaluate raw outputs**
```bash
python src/evaluation.py
```

**4. Create dataset splits**
```bash
python src/create_splits.py
```

**5. Inspect data quality**
```bash
python src/data_quality.py
```

**6. Construct utility targets**
```bash
python src/utility.py
```

**7. Generate query features**
```bash
python src/feature_engineering.py
```

**8. Train the utility-only router**
```bash
python src/train_utility_router.py
```

**9. Train the risk-aware router**
```bash
python src/train_risk_aware_router.py
```

**10. Validate the risk signal**
```bash
python src/validate_risk_signal.py
```

**11. Evaluate the final router on the held-out test set**
```bash
python src/evaluate_test_router.py
```

**12. Run additional baselines**
```bash
python src/additional_baselines.py
python src/category_baseline_train.py
```

**13. Run the no-category ablation**
```bash
python src/train_risk_aware_router_no_category.py
python src/evaluate_test_router_no_category.py
```

**14. Run final analyses**
```bash
python src/statistical_test_router.py
python src/analyse_risk_mechanism.py
python src/category_analysis.py
python src/prediction_ranking_errors.py
```

**15. Build final comparison tables and figures**
```bash
python src/build_final_comparison.py
python src/plot.py
```
## Dataset and Experimental Setup

3,194 unique queries and 12,776 model-query observations, with 4 candidate models per query. The split is 2,233 training / 475 validation / 486 test queries.

Features total 406 dimensions: 384 MiniLM embedding features, 11 structural features, 2 dataset indicators, and 9 category indicators.

The risk-aware ensemble has 10 aligned bootstrap members — within each member, the same bootstrap sample of training queries is used across all candidate models, so the model-specific utility predictors stay aligned with each other. Risk is the population standard deviation across ensemble predictions.

## Utility and Risk-Aware Routing

Utility for cost trade-off \(\alpha\) is \(U = y - \alpha c_{\mathrm{norm}}\), evaluated at

```text
0.1, 0.3, 0.5, 0.7, 1.0
```

The risk-aware rule picks the model maximizing \(\mu_m - \lambda R_m\), with \(\lambda\) swept over

```text
0, 0.25, 0.5, 1, 2, 3, 4
```

\(\lambda\) is chosen on validation performance only; the held-out test set is touched only after that selection is frozen.

## Baselines

**Fixed-model baselines** route every query to the same model.

**Best fixed utility** picks, for each \(\alpha\), whichever model has the highest mean validation utility.

**Category-only routing** learns a preferred model per category from the training split, then freezes that mapping for validation and test.

**Utility-only routing** selects \(\operatorname*{argmax}_m \mu_m\) — equivalent to the risk-aware router at \(\lambda = 0\).

**Risk-aware routing** selects \(\operatorname*{argmax}_m (\mu_m - \lambda R_m)\).

**No-category ablation** retrains the risk-aware router with all category indicator features removed.

## Results

Selected lightweight outputs live under `results/`, including:

```text
results/final_test_results.csv
results/final_test_decisions.csv
results/final_statistical_tests.csv

results/final_policy_comparison.csv
results/final_accuracy_cost_pareto.csv
results/final_accuracy_cost_pareto_unique.csv

results/final_risk_mechanism_analysis.csv
results/final_risk_mechanism_summary.csv

results/final_category_analysis.csv
results/final_category_transitions.csv

results/risk_aware_best_lambdas_aligned.json
results/risk_aware_router_metadata_aligned.json
results/risk_aware_validation_summary_aligned.csv
results/risk_signal_validation_aligned.csv

results/additional_fixed_model_baselines.csv
results/additional_best_fixed_utility_selection.csv
results/additional_best_fixed_utility_test.csv

results/category_router_train_selected_mapping.csv
results/category_router_train_selected_validation.csv
results/category_router_train_selected_test.csv

results/ablation_no_category/best_lambdas.json
results/ablation_no_category/metadata.json
results/ablation_no_category/validation_summary.csv
results/ablation_no_category/test_results.csv
results/ablation_no_category/test_decisions.csv
```

Prediction-ranking analysis outputs live under `results/prediction_error_analysis/`.

## Reproducibility

Preprocessing and model-selection steps are kept split-specific to limit leakage: query splits are fixed before any router is trained, and the cost imputation, winsorization and normalization parameters are all fit on the training data alone. Likewise, the category-only mapping is learned only from the training split, and the risk-aversion parameter \(\lambda\) is chosen from validation performance — the test set is reserved for the very last, single evaluation.

Large intermediate files (raw model outputs, evaluated raw outputs, query embedding matrices, detailed ensemble-member predictions, trained XGBoost checkpoints, and large query-level intermediate analyses) are excluded from GitHub, since they can all be regenerated from the pipeline.

## Figures

Final figures are in `figures/`, including the accuracy-cost trade-off plots used in the analysis.

## Notes

Only the lightweight outputs needed to inspect the final results are included here — raw API responses and trained checkpoints are left out to keep the repository a reasonable size. Exact API outputs can vary with external model-provider behavior and model availability at the time a script was run.
