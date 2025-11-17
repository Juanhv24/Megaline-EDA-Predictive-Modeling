# Megaline EDA & Predictive Modeling

This project analyzes Megaline's 2018 prepaid customers to learn how subscribers use calls, texts, and data across the Surf and Ultimate plans, quantify revenue drivers, and build a model that predicts a customer's plan. The full workflow lives in `Notebooks/Megaline.ipynb`.

## Data
- Source files in `Datasets/`: `megaline_plans.csv`, `megaline_users.csv`, `megaline_calls.csv`, `megaline_messages.csv`, `megaline_internet.csv`, and `users_behavior.csv` (for modeling).
- Cleaning checks found no missing values needing imputation and no duplicates; date columns were converted to datetimes for time-based grouping.
- Feature engineering: added month names to usage tables; converted megabytes to gigabytes; rounded call minutes up to match billing rules; built per-user monthly aggregates and revenue using plan allowances/overage fees.

## Exploratory findings (Surf vs Ultimate)
- Usage: Ultimate subscribers consume more per person across calls, texts, and data; Surf shows higher totals because it has more users. Clear seasonality—volumes rise toward year end.
- Revenue: Per-user revenue is higher and steadier on Ultimate; Surf brings in higher total revenue due to its larger base and overage charges.
- Hypotheses: Shapiro–Wilk rejected normality, so Mann–Whitney U tests were used. Ultimate per-user revenue is significantly higher than Surf (one-tailed, α=0.01); NY–NJ revenue differs significantly from other regions (two-tailed).
- Overall takeaway: Surf is the current volume engine, but growing Ultimate subscriptions should lift stable per-user revenue.

## Predictive model
- Goal: classify customers into Surf vs Ultimate.
- Models tested: Decision Tree, Logistic Regression, and Random Forest with hyperparameter tuning.
- Best model: `RandomForestClassifier` (54 trees) reached ≈0.795 accuracy on the test set, clearing the 0.75 target and outperforming both random (~0.50) and majority-class (~0.66) baselines.
- A sanity check confirmed the model learns meaningful patterns beyond class imbalance.

## How to use
- Open `Notebooks/Megaline.ipynb` to review the analysis, visualizations, and model training steps end to end.
- Datasets are already staged in `Datasets/`; adjust paths in the notebook if you relocate the files.
