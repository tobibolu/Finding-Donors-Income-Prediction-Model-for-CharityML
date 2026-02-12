# Donor Propensity Modeling: Income Prediction for Nonprofit Targeting

A machine learning pipeline that predicts whether individuals earn above $50K annually using US Census data, enabling nonprofits to optimize donor outreach by targeting high-income segments.

## Key Results

| Model | Test Accuracy | Test F-beta (0.5) |
|-------|--------------|-------------------|
| **LightGBM** | **86.8%** | **0.750** |
| XGBoost | 86.6% | 0.747 |
| Gradient Boosting | 86.1% | 0.741 |
| Logistic Regression | 84.5% | 0.698 |
| Random Forest | 84.3% | 0.690 |
| Decision Tree | 81.9% | 0.635 |

**After hyperparameter tuning (Gradient Boosting):** 86.9% accuracy, 0.754 F-beta, 0.925 ROC AUC

## What Makes This Project Different

- **Fairness analysis**: Evaluates model bias across race and sex using disparate impact ratios, equalized odds, and demographic parity
- **Modern models**: Compares classic ML (Logistic Regression, Random Forest) with gradient boosting (XGBoost, LightGBM, GradientBoosting)
- **Feature engineering**: Derived features (capital_net, age bins, work hour categories) beyond raw census attributes
- **SQL analytics**: All exploratory analysis available as SQL queries with CTEs, window functions, and CASE expressions
- **Production pipeline**: Modular Python code with proper train/test split, cross-validation, and reproducible preprocessing

## Project Structure

```
├── app/
│   └── streamlit_app.py          # Interactive donor scoring dashboard
├── data/
│   ├── census.csv                # US Census income dataset (45,222 records)
│   └── model_features.txt        # One-hot-encoded feature names
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
make run-app    # Launch interactive dashboard
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

Python 3.11 | pandas | scikit-learn | XGBoost | LightGBM | NumPy | Matplotlib | Seaborn | Streamlit | SQLite | pytest

## Live Demo

**[Launch Interactive Dashboard](https://finding-donors-income-prediction-model-for-charityml-knkpwdvgv.streamlit.app/)**

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
