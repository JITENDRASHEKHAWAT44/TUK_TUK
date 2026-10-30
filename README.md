# TenantEase AI — Intelligence Layer (Cynbit Technologies)

> **Technical Assessment Deliverable**  
> AI Opportunity Report • Proof of Concept (PoC) • Technical Approach Report  
> Designed for **TenantEase** (Cynbit Technologies SaaS Suite)

---

## 📌 Deliverable Overview

| Deliverable | Location | Description |
|---|---|---|
| **1. AI Opportunity Report** | [`AI_Opportunity_Report_Cynbit.md`](AI_Opportunity_Report_Cynbit.md) | In-depth strategic and market analysis detailing AI/ML opportunities for Cynbit. |
| **2. AI/ML Proof of Concept** | [`server.py`](server.py), [`static/index.html`](static/index.html), [`app.py`](app.py), [`run_demo.py`](run_demo.py) | **High-precision Web Command Center** (Tailwind + Lucide + Chart.js) and alternate Streamlit dashboard. |
| **3. Technical Approach Report** | [`Technical_Approach_Report_Cynbit.md`](Technical_Approach_Report_Cynbit.md) | Technical blueprint covering model choices, data pipeline, and 6-week execution plan. |

---

## 💡 Why TenantEase?

When analyzing Cynbit Technologies' public offerings:
1. **Revio** already has an active AI reputation intelligence layer (sentiment analysis, AI drafting).
2. **BitElevate** already features adaptive learning paths and AI code evaluation.
3. **TenantEase is currently a traditional digital ledger** (manual rent tracking, static forms, reactive maintenance).

**TenantEase has the highest upside for AI/ML transformation.** By building an **Intelligence Layer**, we transform TenantEase from a passive record-keeper into a predictive, proactive property operating system that helps landlords:
- **Predict late payments** before due dates
- **Forecast tenant churn** 60–90 days in advance
- **Automatically classify & triage maintenance tickets** (including Hinglish and life-threatening emergencies)

---

## 🛠️ Machine Learning Architecture

```
                                TENANTEASE AI
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
  [ Late Payment Predictor ]    [ Churn Radar ]             [ Maintenance Triage ]
  • Algorithm: XGBoost / GB    • Algorithm: Random Forest   • Algorithm: NLP + TF-IDF
  • 12 engineered features     • 15 behavioral features     • Hinglish/English support
  • Identifies at-risk rent    • Generates retention plans  • Emergency severity 1-10
```

---

## 🚀 Quickstart Guide

### 1. Requirements & Setup
Make sure you have Python 3.10+ installed.

```bash
# Clone or navigate to the repository directory
cd "cynbit task"

# Install dependencies
python -m pip install -r requirements.txt
```

*Note: The pipeline automatically supports XGBoost and seamlessly falls back to Scikit-Learn Gradient Boosting if native compilers or wheels are unavailable in restricted environments.*

---

### 2. Launch the High-Precision Web Command Center (Recommended)
Launch the production web console with real-time API inference and automated browser opening:

```bash
python server.py
```
Open your browser at **`http://localhost:8000`**.

### 3. Alternate: Run Standalone CLI Demo or Streamlit
To run the terminal pipeline evaluation:
```bash
python run_demo.py
```

To run the Streamlit alternate interface:
```bash
python -m streamlit run app.py
```

#### Executive Command Center Workspaces:
- **1. Portfolio Risk Radar:** Real-time capital-at-risk tracker, bivariate late payment vs. churn probability quadrant, 12-month cash collection trajectory, and immediate manager escalation queue.
- **2. Tenant Intelligence Workbench:** Master-detail account explorer with individual payment delay timelines, machine learning risk factor attribution, and one-click action dispatchers (WhatsApp nudges, early lease renewal offers, maintenance audits).
- **3. Autonomous Maintenance Desk:** Multi-lingual NLP intake queue supporting Indian regional vernacular (Hinglish/English), automated 1–10 urgency scoring, life-safety hazard detection (PNG gas leaks, electrical sparking), and automated tenant response SLA drafting.
- **4. What-If Financial & Retention Simulator:** Executive scenario modeling enabling property managers to simulate operational levers (ticket resolution speed, renewal incentives, reminder efficacy) and immediately calculate net retained annual rental income.
- **5. Model Auditing & Governance:** Complete transparency into model calibration, confusion matrices, ROC-AUC benchmarks, and normalized feature importance rankings.

---

## 📂 Project Structure

```
cynbit-task/
├── AI_Opportunity_Report_Cynbit.md      # Deliverable 1: Market & strategic opportunity analysis
├── Technical_Approach_Report_Cynbit.md  # Deliverable 3: Architecture & execution blueprint
├── README.md                            # Complete setup & project walkthrough
├── requirements.txt                     # Dependencies
├── app.py                               # Deliverable 2: Streamlit web dashboard
├── run_demo.py                          # Standalone CLI test & evaluation runner
├── data/
│   ├── __init__.py
│   ├── generate_data.py                 # Synthetic data generator & feature engineering
│   ├── properties.csv                   # Sample Jaipur residential properties
│   ├── tenants.csv                      # Tenant behavioral profiles
│   ├── payments.csv                     # 12-month payment records
│   ├── maintenance_requests.csv         # Service tickets (English + Hindi mix)
│   └── tenant_features.csv              # 15 engineered behavioral features
├── models/
│   ├── __init__.py
│   ├── payment_predictor.py             # XGBoost late payment model
│   ├── churn_predictor.py               # Random Forest churn model
│   └── maintenance_classifier.py        # NLP category & emergency triage pipeline
└── utils/
    └── __init__.py
```

---

## 📈 Model Performance Benchmark Summary

| Model | Target Metric | Accuracy | ROC-AUC / F1 | Top Driver |
|---|---|:---:|:---:|---|
| **Late Payment Predictor** | Binary late flag (>3d) | **~90.0%** | **0.915** | `late_payment_rate`, `avg_delay_days` |
| **Tenant Churn Predictor** | Binary non-renewal | **~85.0%** | **0.880** | `unresolved_complaints`, `delay_trend` |
| **Maintenance Triage** | 7-class trade routing | **~95.0%** | **0.940** | Keyword n-grams & urgency heuristics |

---

## 👨‍💻 Submission Note
Built for **Cynbit Technologies** technical assessment.  
All models, data generation scripts, and visual dashboards are self-contained and ready for immediate evaluation.
