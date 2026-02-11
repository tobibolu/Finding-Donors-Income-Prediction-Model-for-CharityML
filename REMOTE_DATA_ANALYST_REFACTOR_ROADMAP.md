# Remote Data Analyst / AI Portfolio Refactor Roadmap (US Remote Role Ready)

## Scope analyzed
This repository currently contains:
- `donors.ipynb` (end-to-end analysis notebook)
- `census.csv` (dataset)
- `model_features.txt` (saved feature list)

The recommendations below are tailored to modern (2026-level) remote data analyst / applied AI standards expected by US employers, with practical guidance for a candidate working from Nigeria.

---

## 1) Current-state assessment (what is good and what should improve)

### Strengths already present
- Clear business framing and objective (predict likely high-income donor prospects).
- Baseline, multiple models, and model comparison are included.
- Some feature importance interpretation and deployment artifact export exist.

### Gaps against current remote analyst standards
1. **Notebook-first workflow with limited modularity**
   - Most logic lives in notebook cells; this is hard to test, review, and maintain remotely.
2. **Data quality checks are basic**
   - Missing-value check is present, but no schema validation, drift checks, duplicate checks, outlier policy, or unit tests for data assumptions.
3. **Modeling/evaluation is not production-rigorous yet**
   - Single split evaluation is used; no cross-validation, confidence intervals for headline metrics, calibration analysis, fairness checks, or threshold optimization tied to business cost.
4. **Experiment tracking is absent**
   - No experiment registry (MLflow/W&B), model versioning, reproducible runs, or artifact lineage.
5. **MLOps & deployment maturity is limited**
   - Model is exported via joblib, but no prediction service contract, inference pipeline object, model card, monitoring plan, or CI checks.
6. **Remote-collaboration readiness is not visible**
   - No issue templates, PR templates, coding standards, task board conventions, async documentation, or timezone-aware communication routines.
7. **Portfolio-to-hiring signal could be stronger**
   - Repo does not yet show analyst deliverables expected in US roles: executive summary, KPI dashboard spec, SQL analytics examples, stakeholder memo, experiment readout.

---

## 2) Target architecture (what to refactor into)

Refactor from a single notebook into this structure:

```text
.
├─ README.md
├─ pyproject.toml / requirements.txt
├─ src/
│  ├─ config.py
│  ├─ data/
│  │  ├─ load.py
│  │  ├─ validate.py
│  │  └─ transform.py
│  ├─ features/
│  │  └─ build.py
│  ├─ models/
│  │  ├─ train.py
│  │  ├─ evaluate.py
│  │  └─ predict.py
│  ├─ monitoring/
│  │  └─ drift.py
│  └─ utils/
├─ notebooks/
│  ├─ 01_eda.ipynb
│  ├─ 02_modeling.ipynb
│  └─ 03_business_recommendations.ipynb
├─ tests/
│  ├─ test_data_validation.py
│  ├─ test_features.py
│  └─ test_inference_schema.py
├─ configs/
│  ├─ base.yaml
│  ├─ dev.yaml
│  └─ prod.yaml
├─ reports/
│  ├─ executive_summary.md
│  ├─ model_card.md
│  └─ fairness_report.md
└─ .github/workflows/ci.yml
```

### Why this matters for hiring
US remote teams evaluate not only model accuracy but also:
- reproducibility,
- communication clarity,
- code quality,
- business interpretation,
- ability to work asynchronously.

This structure demonstrates all five.

---

## 3) Detailed refactor plan (priority order)

## Phase 1 — Reproducibility & engineering hygiene (Week 1)

1. **Pin environment and tooling**
   - Add `requirements.txt` or `pyproject.toml` with version pins.
   - Add `Makefile` commands: `make setup`, `make lint`, `make test`, `make train`.
2. **Code quality baseline**
   - Add `ruff`, `black`, `isort`, `mypy` (or pyright), and `pre-commit`.
3. **Notebook decomposition**
   - Move reusable logic from notebook into `src/` modules.
   - Keep notebooks for narrative + plots only.
4. **Configuration-driven runs**
   - Replace hard-coded params with YAML config (`train_test_split`, model params, random seeds).
5. **Deterministic runs**
   - Set and document random seeds for Python, NumPy, and model libraries.

**Definition of done**
- Fresh clone + `make train` runs end-to-end successfully.
- CI passes lint + tests on push/PR.

---

## Phase 2 — Data quality and analytics standards (Week 2)

1. **Data contract and validation**
   - Add schema checks (column names, dtypes, ranges, allowed categories) using Pandera/Great Expectations.
2. **Data quality report artifact**
   - Auto-generate report: null rates, duplicates, cardinality, skewness, outlier flags, class balance.
3. **Feature transformations in pipeline**
   - Use `ColumnTransformer` + `Pipeline` to avoid leakage and ensure inference consistency.
4. **Business-centric exploratory analytics**
   - Add segmented lift analysis by key cohorts (age bands, education, geography if available).
   - Add cost-aware sensitivity analysis for donor targeting.

**Definition of done**
- `tests/test_data_validation.py` enforces schema.
- Data profiling report is generated in `reports/` each run.

---

## Phase 3 — Modeling rigor expected in modern analytics teams (Week 3)

1. **Cross-validation + robust model selection**
   - Replace single train/test comparisons with stratified CV.
2. **Metric suite aligned to business**
   - Include: ROC-AUC, PR-AUC, F-beta, precision@k, recall@k, lift@decile.
3. **Threshold optimization**
   - Optimize classification threshold against business utility (e.g., outreach cost vs expected donation value).
4. **Calibration checks**
   - Add reliability curve and Brier score (important for ranking donors by propensity).
5. **Fairness diagnostics**
   - Evaluate disparate impact / opportunity by sensitive groups where legally and ethically appropriate.
6. **Confidence intervals for headline outcomes**
   - Bootstrap CIs for key metrics to communicate uncertainty to stakeholders.

**Definition of done**
- `reports/model_card.md` includes metrics, assumptions, caveats, and threshold rationale.

---

## Phase 4 — Deployment readiness and monitoring (Week 4)

1. **Package inference pipeline artifact**
   - Save a single pipeline object (`preprocess + model`) not just bare model.
2. **Prediction contract**
   - Add clear input/output schema docs and validation at inference.
3. **Basic service layer (optional but high-impact for portfolio)**
   - Minimal FastAPI endpoint `/predict` with pydantic schema.
4. **Monitoring starter kit**
   - Add drift checks (PSI/KS), prediction distribution monitoring, and threshold alarms.
5. **Rollback/versioning strategy**
   - Store model versions with metadata: training data snapshot hash, metrics, timestamp.

**Definition of done**
- You can score new records consistently and detect drift post-deployment.

---

## 4) Analyst deliverables to add (critical for US remote job positioning)

Create these assets in repo:

1. **Executive one-pager (`reports/executive_summary.md`)**
   - Problem, approach, KPI impact, recommendation, risks, next decisions.
2. **Stakeholder memo**
   - Explain trade-offs in plain English (precision vs recall vs outreach cost).
3. **KPI dashboard spec (Looker/Power BI/Tableau)**
   - Donation conversion rate, campaign ROI, model lift by decile, drift trends.
4. **SQL analytics examples**
   - Include 8–12 realistic SQL queries for funnel analysis, cohort performance, and campaign uplift.
5. **Experiment readout template**
   - Hypothesis, experiment design, success metric, confounders, decision outcome.

These artifacts often matter more than small model-accuracy gains in analyst hiring.

---

## 5) Recommended technology stack (practical and recruiter-friendly)

- **Core**: Python, pandas, scikit-learn, numpy, scipy
- **Validation**: Pandera or Great Expectations
- **Experiment tracking**: MLflow
- **Visualization**: seaborn + plotly
- **Serving**: FastAPI (lightweight)
- **Testing**: pytest
- **Automation**: GitHub Actions
- **Container**: Docker
- **Docs**: Markdown + mkdocs (optional)

Keep stack simple and interpretable; avoid overengineering unless role specifically demands it.

---

## 6) Remote-work standards checklist (what hiring managers look for)

## Async communication quality
- Every PR has: context, approach, trade-offs, screenshots/charts, and risk notes.
- Every analysis has: assumptions, caveats, and clear “so what”.

## Collaboration reliability
- Use issue templates and estimation labels (`S/M/L`).
- Maintain a short weekly changelog and decision log.

## Operational discipline
- CI required for merge.
- Versioned data/model artifacts.
- Reproducible command path from raw data to report.

## Security/professionalism
- `.env.example` and no secrets in repo.
- Respect privacy and fairness constraints in demographic modeling.

---

## 7) Nigeria-based remote candidate strategy (US market fit)

1. **Timezone strategy**
   - Offer a fixed overlap window (e.g., 3–5 hours with ET/PT) and state it clearly in README/profile.
2. **Portfolio packaging**
   - Each project should include: business brief, architecture diagram, reproducibility instructions, and decision memo.
3. **Evidence of communication**
   - Add concise Loom/video walkthrough links (5–7 min) explaining business impact.
4. **Role-aligned variants**
   - Maintain two CV/repo narratives:
     - Data Analyst track (SQL + BI + experimentation)
     - Applied AI Analyst track (ML + model monitoring + inference)
5. **Reliability signals**
   - Fast PR turnaround, clear written updates, and consistent documentation are decisive in remote hiring.

---

## 8) 30-60-90 day execution plan (job-search + portfolio)

## First 30 days
- Refactor project structure, add tests/CI, publish executive summary and model card.
- Build one BI dashboard mock with campaign metrics.

## Days 31–60
- Add MLflow tracking + FastAPI scoring endpoint + drift notebook.
- Publish 2 case-study posts (problem, method, business impact).

## Days 61–90
- Create second project (different domain: fintech/churn/fraud) using same professional template.
- Start targeted applications + recruiter outreach with tailored project links.

---

## 9) Interview readiness mapping (skills to demonstrate)

- **SQL**: joins, window functions, cohort analysis, funnel analytics, performance tuning basics.
- **Analytics**: experiment design, metric trade-offs, bias/confounding, storytelling.
- **ML**: leakage prevention, CV, thresholding, calibration, explainability.
- **Production mindset**: monitoring, reproducibility, CI/CD basics, stakeholder communication.

For each portfolio project, prepare a 2-minute and a 10-minute walkthrough.

---

## 10) Success criteria (how to know the refactor worked)

You are “US remote-ready” when this repo can show:
1. One-command reproducibility (`make train && make report`).
2. CI checks with tests passing.
3. Clear business KPI narrative and decision recommendation.
4. Model card + fairness considerations + monitoring plan.
5. Professional documentation that another analyst can pick up asynchronously.

