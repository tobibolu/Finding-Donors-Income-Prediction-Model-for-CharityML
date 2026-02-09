# Donor Propensity Modeling: Income Prediction for Nonprofit Targeting

A machine learning pipeline that predicts whether individuals earn above $50K annually using US Census data, enabling nonprofits to optimize donor outreach by targeting high-income segments.

## Key Results

| Model | Test Accuracy | Test F-beta (0.5) | ROC AUC |
|-------|--------------|-------------------|---------|
| Logistic Regression | 84.7% | 0.710 | 0.90 |
| Random Forest | 84.2% | 0.688 | 0.91 |
| Gradient Boosting | **86.2%** | **0.735** | **0.92** |
| XGBoost | 86.0% | 0.731 | 0.92 |

**Top predictive features:** capital-gain (14.4%), marital status (11.4%), age (10.1%), education level (9.6%)

## What Makes This Project Different

- **Fairness analysis**: Evaluates model bias across race and sex using disparate impact ratios, equalized odds, and demographic parity
- **Modern models**: Compares classic ML (Logistic Regression, Random Forest) with gradient boosting (XGBoost, LightGBM, GradientBoosting)
- **Feature engineering**: Derived features (capital_net, age bins, work hour categories) beyond raw census attributes
- **SQL analytics**: All exploratory analysis available as SQL queries with CTEs, window functions, and CASE expressions
- **Production pipeline**: Modular Python code with proper train/test split, cross-validation, and sklearn Pipeline compatibility

## Project Structure

```
├── app/                          # (Placeholder for future Streamlit dashboard)
├── data/
│   └── census.csv                # US Census income dataset (45,222 records)
├── notebooks/
│   └── donors.ipynb              # Analysis notebook with narrative
├── sql/
│   └── queries.sql               # Analytical SQL queries
├── src/
│   ├── data_loader.py            # Data loading, preprocessing, feature engineering
│   ├── database.py               # SQLite integration and SQL analytics
│   ├── models.py                 # Model training, comparison, optimization
│   ├── fairness.py               # Bias detection and fairness metrics
│   └── visualizations.py         # Plotting utilities
├── tests/
│   ├── test_data_loader.py       # Data validation tests
│   └── test_models.py            # Model and fairness tests
├── requirements.txt
├── Makefile
└── .github/workflows/ci.yml
```

## Quick Start

```bash
pip install -r requirements.txt
make test
```

## Fairness Analysis

The model is evaluated for algorithmic fairness across protected attributes:

- **Disparate Impact Ratio**: Measures whether positive prediction rates satisfy the 4/5ths rule across demographic groups
- **Equalized Odds**: Compares true positive and false positive rates across groups
- **Demographic Parity**: Checks if prediction rates are consistent across groups

This analysis is critical for responsible ML deployment in contexts affecting resource allocation.

## SQL Analytics

The project includes SQLite-based analytics demonstrating:
- Income rates by education and occupation (`GROUP BY` + `HAVING`)
- Demographic profiles using CTEs
- Capital gains decile analysis with `NTILE` window functions
- Age group distributions with `CASE` expressions
- Cross-tabulations with multiple aggregates

See [`sql/queries.sql`](sql/queries.sql) for the full query set.

## Dataset

- **45,222** records from the US Census Bureau
- **14** features: age, workclass, education, marital status, occupation, relationship, race, sex, capital gain/loss, hours per week, native country
- **Target**: Binary income classification (>$50K vs <=$50K)
- **Class balance**: 75.2% low income, 24.8% high income

## Tech Stack

Python 3.11 | pandas | scikit-learn | XGBoost | LightGBM | NumPy | Matplotlib | Seaborn | SQLite | pytest
