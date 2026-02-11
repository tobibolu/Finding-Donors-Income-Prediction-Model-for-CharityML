# Executive Summary

## Problem
Charity outreach is expensive when done broadly. This project predicts high-income prospects to prioritize likely high-value donor segments.

## Approach
- Built a reproducible scikit-learn pipeline (preprocessing + Random Forest).
- Added schema validation, train/test split, cross-validation, and threshold optimization.
- Produced artifacts for repeatable training and inference.

## Suggested KPIs
- Precision@Top-Decile for outreach list quality.
- Donation conversion rate uplift vs untargeted baseline.
- Campaign ROI and model score drift over time.

## Recommendation
Run a 4–8 week pilot campaign with treatment/control groups, monitor conversion uplift and fairness indicators, then iterate threshold policy.
