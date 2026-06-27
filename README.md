# Practice_Welon_Agency
# WELON AGENCY — Client Base Analysis & Campaign Performance Optimization

A complete, reproducible **data analytics pipeline** for a digital marketing agency — from data engineering and SQL, through machine learning, to an interactive web application.

> Industrial practice project · Astana IT University · Mathematics & Computational Sciences
> Author: **Aitakhmetova Sabina** (MCS-2401)

---

## Overview

This project builds an end-to-end analytics system for **WELON AGENCY**, a digital marketing company in Astana that manages campaigns for four real clients (Romantic, Biosfera, Jetour Auto, Kinetik). Because direct access to client data was restricted, a **realistic synthetic dataset** of 45 campaigns was generated from the agency's actual portfolio and industry benchmarks, then analysed with Python, SQL, and machine learning.

The goal: turn raw campaign data into **actionable business insight** — identifying at-risk clients, the best-performing campaign types, and the optimal way to allocate budget.

---

## Architecture

![Pipeline architecture](images/architecture_pipeline.png)

The pipeline runs in five layers — **data generation → SQLite storage → analysis → machine learning → interactive deployment** — and reproduces every result from a single notebook.

![Analytical process](images/analysis_flowchart.png)

---

## What's inside

| # | Workstream | Method |
|---|------------|--------|
| 1 | **SQL analysis** | Client & campaign-type performance (SQLite) |
| 2 | **RFM segmentation** | Rank-based Recency / Frequency / Monetary scoring |
| 3 | **Correlation analysis** | Pearson correlation across 9 metrics |
| 4 | **A/B testing** | Independent-samples t-tests (scipy.stats) |
| 5 | **Seasonal analysis** | Monthly demand patterns per client |
| 6 | **K-Means clustering** | 3 performance tiers (Elbow method) |
| 7 | **Churn prediction** | Logistic Regression on RFM scores |
| 8 | **Conversion forecasting** | Linear Regression (train/test split) |
| 9 | **Budget optimization** | Linear programming (`scipy.optimize.linprog`) |
| 10 | **Streamlit web app** | 5-page interactive dashboard |

---

## Database schema

![Database schema](images/database_schema.png)

A SQLite database with two tables in a one-to-many relationship: each **client** owns many **campaigns**.

---

## Key results

- **Biosfera flagged as At Risk** — a high-value client dormant for 243 days; highest churn probability (**46%**) and the priority for re-engagement.
- **Jetour** leads both spend (3.84M KZT) and average ROI (**523%**).
- Conversion forecast model reaches **R² = 0.797** on a held-out test set.
- Budget optimizer allocates 10M KZT to the highest-ROI clients for a projected **52.5M KZT** return.
- A/B tests were **not statistically significant** (small samples) — reported honestly as a need for more data.

---

## Tech stack

`Python` · `pandas` · `numpy` · `scikit-learn` · `scipy` · `matplotlib` · `seaborn` · `Plotly` · `SQLite` · `Streamlit` · `Jupyter`

---

## Repository structure

```
.
├── notebooks/
│   └── PracticeAgency.ipynb     # Full analysis pipeline (run top-to-bottom)
├── app/
│   └── app.py                   # Streamlit interactive dashboard
├── images/                      # Architecture, schema & flowchart diagrams
├── requirements.txt
└── README.md
```

---

## How to run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the analysis notebook**
```bash
jupyter notebook notebooks/PracticeAgency.ipynb
```
Run all cells top to bottom — the dataset and every result regenerate from a fixed random seed.

**3. Launch the interactive dashboard**
```bash
python -m streamlit run app/app.py
```
Then open `http://localhost:8501` in your browser.

---

## Future work

- Migrate from SQLite to **PostgreSQL** for larger datasets
- Integrate real ad-platform APIs (**Meta Ads, Google Ads**) for automatic data collection
- Add an **XGBoost** model to push forecasting accuracy beyond R² 0.80
- Deploy the Streamlit app to the **cloud** (Streamlit Cloud / Heroku)

---

## Author

**Aitakhmetova Sabina** — Mathematics & Computational Sciences, Astana IT University
Industrial practice at WELON AGENCY (TOO), Astana, Kazakhstan · 2026
