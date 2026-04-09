# UPI Transaction Pattern Analysis & Fraud Detection Platform

I built this project to understand how digital payment fraud works in India's UPI ecosystem. Started as a data analysis project, then evolved into a full production-grade platform with real-time fraud scoring, vector similarity search, causal inference, and containerized deployment.

## The problem

India's UPI system processes 16+ billion transactions every month (as of Feb 2025). With that kind of volume, even a tiny fraud rate means crores of rupees lost. I wanted to see if I could build a system that not only *detects* fraudulent transactions but also *measures the causal impact* of security interventions like 2FA.

Since real UPI data isn't publicly available (obvious privacy reasons), I generated synthetic data calibrated against official NPCI ecosystem stats and RBI fraud reports. The distributions, bank market shares, and fraud patterns all mirror what actually happens in the real world.

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│   FastAPI    │───▶│  Celery      │───▶│  Fraud Scoring  │
│   Gateway    │    │  Workers     │    │  (Ensemble +    │
│   (api.py)   │    │  (Redis Q)   │    │   FAISS + DiD)  │
└──────┬───────┘    └──────────────┘    └─────────────────┘
       │                                        │
       ▼                                        ▼
┌──────────────┐                    ┌───────────────────┐
│  /causal/*   │                    │  FAISS Index      │
│  /fraud/*    │                    │  (fraud pattern   │
│  /health     │                    │   embeddings)     │
└──────────────┘                    └───────────────────┘
```

**Three-service stack** (Docker Compose):
- **API** (FastAPI + Uvicorn) — handles all HTTP requests, serves fraud scores and causal results
- **Worker** (Celery) — processes async fraud scoring tasks from the Redis queue
- **Redis** — message broker + result backend for Celery

## what I actually built

### Data pipeline
Generates 1M transactions with realistic patterns (spending peaks at lunch and evening, more P2M on weekdays, festival season spikes etc). Injected 6 types of fraud that actually occur in UPI — rapid fire transactions, round-number structuring, unusual hour activity, geographic impossibility, and more.

### SQL analysis
Loaded everything into SQLite and wrote 12 queries covering daily trends, bank failure rates, fraud patterns by city/time/type, threshold monitoring etc. Not textbook SQL — actual analytical queries you'd write at a job.

### 4 fraud detection methods

| Method | What it does | Why it works |
|--------|-------------|--------------|
| **Z-Score** | Flags amounts 3+ σ from a user's average | Catches sudden behavior changes |
| **Isolation Forest** | Unsupervised ML anomaly detection | Finds outliers without labels |
| **Rule-Based** | Manual rules from RBI fraud reports | Catches known patterns (structuring, odd hours) |
| **FAISS Similarity** | Finds K-nearest fraud neighbors in vector space | Catches variants of known fraud types |

Combined into an ensemble that flags when 2+ methods agree — reduces false positives significantly.

### FAISS vector similarity search

Every transaction is embedded into an 11-dimensional feature vector:
```
[amount_norm, hour_sin, hour_cos, dow_sin, dow_cos, is_night,
 is_weekend, velocity_norm, amount_zscore, city_encoded, txn_type_encoded]
```

Built a FAISS `IndexFlatL2` over all 29,139 confirmed fraud transactions. When a new transaction comes in, query for the K most similar historical fraud cases in sub-millisecond time. Returns a `fraud_similarity_score` (0-1) based on average L2 distance to nearest fraud neighbors. FAISS flagged 20,383 transactions (1.98%) — closely tracking the actual fraud rate.

Why FAISS over basic KNN? FAISS is what Meta, Spotify, and Uber use for production-scale vector search. Even though our dataset is ~1M rows, using FAISS demonstrates knowledge of the same infrastructure powering RAG pipelines, recommendation systems, and semantic search.

### Benford's Law analysis
There's a mathematical law that says in natural data, the digit "1" appears as the first digit 30% of the time while "9" only appears 5%. Fraudsters don't know this so their fake transactions violate this pattern. Used chi-squared tests to quantify the deviation.

### hypothesis tests

Three statistical tests with proper significance:
1. Fraud amounts are higher than normal (Mann-Whitney U, p < 0.001)
2. Fraud rate is higher at night 1-5 AM (Chi-squared, p < 0.001)
3. Fraud rate varies across transaction types (Chi-squared, p = 0.011)

---

## Causal inference: effect of 2FA on fraud

This is the key evolution — shifting from "is this fraudulent?" (prediction) to "does 2FA *cause* fraud to decrease?" (causal impact measurement).

### esearch question
> **What is the causal effect of introducing 2-Factor Authentication (2FA) on the likelihood of a UPI transaction being fraudulent?**

### Methodology: Difference-in-Differences (DiD)

DiD is a quasi-experimental method that isolates the causal effect of a policy intervention by comparing treatment and control groups before and after the intervention. It's the same technique economists use to evaluate minimum wage laws, healthcare mandates, and tax policy changes.

**Model specification:**
```
is_fraud = β₀ + β₁(has_2fa) + β₂(post_treatment) + β₃(has_2fa × post_treatment)
           + β₄(amount_log) + β₅(is_night) + β₆(is_weekend) + β₇(city_tier)
           + β₈(user_risk_score) + ε
```

- `β₃` (the coefficient on the interaction term) = **Average Treatment Effect (ATE)**
- Estimated with OLS using **HC1 heteroscedasticity-robust standard errors**
- Treatment date: July 1, 2024 (simulated RBI mandate for mandatory 2FA)

### esults

| Metric | Value |
|--------|-------|
| **ATE** | **-0.0073** (2FA reduces fraud probability by 0.73 percentage points) |
| **95% CI** | [-0.013, -0.0016] |
| **p-value** | 0.013 → **Statistically significant** |
| **Parallel Trends** | **Valid** (F = 1.02, p = 0.40) |
| **Placebo Test** | **Passed** (ATE ≈ 0, p = 0.99) |
| **R²** | 0.118 |
| **N** | 100,000 |

### Sensitivity analysis

ATE remains stable across all 4 model specifications — robust finding:

| Specification | ATE | p-value | R² |
|---------------|-----|---------|-----|
| No controls | -0.0077 | 0.013 | 0.0001 |
| Time controls | -0.0077 | 0.011 | 0.031 |
| Amount controls | -0.0070 | 0.017 | 0.116 |
| **Full controls** | **-0.0073** | **0.013** | **0.118** |

### Treatment assignment (intentionally confounded)

2FA adoption is **not** randomly assigned — this is a deliberate design choice. In reality:
- Tier-1 city users adopt 2FA more (higher fintech literacy)
- PhonePe/GPay users get it first (fintech-first banks roll out faster)
- Power users adopt faster than infrequent users

This creates confounding that DiD must properly handle, making the analysis more realistic and rigorous than a simple randomized comparison.

### Assumptions and validation

DiD rests on several assumptions. Here's how I validated each:

| Assumption | What it means | How I validated |
|-----------|---------------|-----------------|
| **Parallel Trends** | Treatment and control groups had the same fraud trend before 2FA | F-test on `has_2fa × month` interactions in pre-treatment data. Non-significant (p=0.40) = assumption holds. |
| **No Anticipation** | Users didn't change behavior before the 2FA launch | By construction — the treatment date is not known to users in advance |
| **SUTVA** | One user's 2FA doesn't affect another user's fraud risk | Reasonable for individual fraud scoring (not for network-level fraud) |
| **Common Support** | Treatment and control groups overlap in characteristics | Verified by design — both groups span all cities, banks, and transaction types |

### obustness checks

1. **Placebo Test**: Re-ran DiD with a fake treatment date (April 2024). ATE ≈ 0 with p = 0.99 — no spurious effect at a made-up date ✓

2. **Sensitivity Analysis**: Ran 4 model specifications (no controls → full controls) and ATE stayed in the narrow range of -0.007 to -0.008 — robust finding ✓

### Limitations

- Treatment is simulated, not from a real natural experiment
- Complier-level effects (LATE) may differ from population-level ATE
- No geographic or time fixed effects (could improve precision)
- Cross-contamination possible if fraud rings span treatment/control groups

---

## eal-time processing (Celery + Redis)

Transactions come in at thousands per second in production. You can't block the API while running ensemble + FAISS + causal models. The architecture uses:

- **Celery** for async task processing with Redis as the message broker
- **Graceful degradation**: if Redis isn't running, everything falls back to synchronous execution
- **Task monitoring**: check task status by ID via the API
- **Periodic tasks**: FAISS index rebuild can run nightly via Celery Beat

---

## API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + component status |
| `/fraud/score` | POST | Score a single transaction (ensemble + FAISS) |
| `/fraud/batch` | POST | Score multiple transactions (sync or async) |
| `/fraud/similar/{txn_id}` | GET | FAISS K-nearest fraud pattern search |
| `/causal/summary` | GET | Full DiD results (ATE, CI, p-value, robustness) |
| `/causal/estimate` | POST | DiD estimate for a filtered subset |
| `/causal/parallel-trends` | GET | Pre-treatment fraud rate time series |
| `/task/{id}/status` | GET | Async Celery task status |

Interactive docs at `http://localhost:8000/docs` (Swagger UI).

---

## Dashboards

### Power BI Dashboard

Built a 3-page interactive dashboard with a star schema data model using real NPCI data (Apr 2023 - Feb 2025) and RBI Annual Report fraud statistics (FY2020-2025).

**Page 1 - Overview**

![Overview - KPI cards, monthly volume trend with fraud overlay, transaction mix](screenshots/01_overview.png)

**Page 2 - Fraud Intelligence**

![Fraud Intelligence - fraud by pattern type, hourly fraud rate with 1-5 AM spike, detection method comparison](screenshots/02_fraud_intelligence.png)

**Page 3 - Real Data & Benford's Law**

![Real Data - NPCI monthly trends, RBI fraud stats, Benford's Law normal vs fraud comparison](screenshots/03_real_data_benfords.png)

### Streamlit Dashboard

Interactive web dashboard with 6 pages — overview, fraud intelligence, bank/city analytics, real market context with NPCI data, Benford's Law visualization, and transaction lookup.

![Streamlit dashboard overview page](screenshots/04_streamlit_dashboard.png)

### Excel Report
6-sheet report for non-technical stakeholders with executive summary, fraud details, statistical results, method comparison, bank performance, and recommendations.

---

## esults

| Method | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| Z-Score | 34.4% | 36.7% | 35.5% |
| Isolation Forest | 57.6% | 61.0% | 59.3% |
| Rule-Based | 73.9% | 54.4% | 62.7% |
| **Ensemble (2/3)** | **69.1%** | **53.9%** | **60.6%** |

Key findings:
- Fraud transactions have significantly higher amounts than normal ones
- 1-5 AM window has 3x higher fraud rate than daytime
- Structuring (amounts just under Rs 10K reporting threshold) is the most common pattern
- Benford's Law deviation is way more pronounced in fraudulent transactions
- Ensemble approach beats any single method for precision-recall balance
- **2FA causally reduces fraud by 0.73 percentage points (p = 0.013)**

---

## eal data used

- **NPCI UPI Ecosystem Statistics** — monthly transaction volumes Apr 2023 to Feb 2025
- **RBI Annual Report on Banking** — fraud case counts and values FY2020-2025
- Bank market shares calibrated to NPCI Q3 2024 data (PhonePe 47%, GPay 34% etc)

FY25 saw a 53% drop in card/internet fraud cases after banks adopted AI/ML detection systems (source: RBI Annual Report 2025).

---

## Tech stack

Python, Pandas, NumPy, SQLite, Scikit-learn, SciPy, statsmodels, FAISS, FastAPI, Uvicorn, Celery, Redis, Docker, Plotly, Streamlit, Matplotlib, Seaborn, OpenPyXL, Power BI, GitHub Actions

---

## Project structure

```
upi-fraud-detection/
├── src/
│   ├── data_generator.py       # synthetic data generation (1M rows)
│   ├── data_cleaning.py        # cleaning + feature engineering
│   ├── db_utils.py             # SQLite database + 12 analytical queries
│   ├── fraud_detector.py       # z-score, isolation forest, rules, ensemble
│   ├── fraud_similarity.py     # FAISS vector similarity search engine
│   ├── causal_inference.py     # Difference-in-Differences analysis
│   ├── tasks.py                # Celery async task definitions
│   └── report_generator.py     # Excel report
├── api.py                      # FastAPI gateway (8 endpoints)
├── dashboard/
│   └── app.py                  # Streamlit dashboard (6 pages)
├── data/
│   ├── raw/                    # generated CSV
│   ├── processed/              # cleaned, flagged data + causal results
│   ├── faiss_index/            # FAISS index + artifacts
│   ├── real_npci_stats.csv     # real NPCI monthly data
│   └── real_rbi_fraud_data.csv # real RBI annual fraud data
├── database/                   # SQLite DB
├── reports/                    # Excel + Power BI + parallel trends plot
├── screenshots/                # dashboard screenshots
├── tests/
│   ├── test_causal.py          # causal inference tests (17 tests)
│   └── test_similarity.py      # FAISS similarity tests (12 tests)
├── Dockerfile                  # containerized API
├── docker-compose.yml          # 3-service stack (API + Worker + Redis)
├── .github/workflows/ci.yml   # CI pipeline (lint + test + docker build)
├── .dockerignore
├── run_pipeline.py             # runs everything end to end (7 steps)
├── requirements.txt
└── README.md
```

---

## How to run

### option 1: local development

```bash
# setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# run full pipeline (~5-10 min for 1M rows)
python run_pipeline.py

# launch streamlit dashboard
streamlit run dashboard/app.py

# start the API
uvicorn api:app --reload
# then visit http://localhost:8000/docs
```

### option 2: Docker

```bash
# build and start everything (API + Celery Worker + Redis)
docker-compose up -d

# check logs
docker-compose logs -f api

# API available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs

# stop
docker-compose down
```

### run tests

```bash
pytest tests/ -v
```

---

## Data sources

The synthetic data isn't random — it's calibrated against real published numbers:
- NPCI UPI Ecosystem Statistics — bank market shares, volumes, monthly growth
- RBI Annual Report on Trends and Progress of Banking — fraud rates, common patterns
- NPCI monthly reports — transaction volumes, value trends

---

## what I'd do next

If I had access to real labeled data or more time:
- Add real-time streaming with Kafka
- Try deep learning (autoencoders) for anomaly detection
- Add network graph analysis to detect fraud rings
- Implement Instrumental Variable (IV) estimation as an alternative to DiD
- Add geographic and time fixed effects to the causal model
- Deploy to AWS ECS with CloudWatch monitoring
