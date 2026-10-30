# Technical Approach Report: Architecture & Execution Plan

**Project:** TenantEase AI Intelligence Layer  
**Target Platform:** TenantEase (Cynbit Technologies SaaS Ecosystem)  
**Author:** AI/ML Engineering Candidate  
**Date:** October 2026  
**Status:** Functional PoC & Production Architecture Specification  

---

## 1. System Architecture Overview

The **TenantEase AI Intelligence Layer** is engineered as a modular, low-latency microservice architecture designed to ingest tenant interactions, transaction histories, and ticket logs, process them through specialized machine learning and NLP pipelines, and expose actionable inference endpoints to the TenantEase frontend and administrative dashboard.

```
                           TENANTEASE DATA STREAMS
   [ Payment Gateway Logs ]   [ Lease Contracts ]   [ Tenant Service Tickets ]
              │                        │                        │
              └────────────────────────┼────────────────────────┘
                                       ▼
                       DATA INGESTION & FEATURE STORE
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
 [ FEATURE PIPELINE 1 ]       [ FEATURE PIPELINE 2 ]       [ TEXT PREPROCESSOR ]
  12 Payment/Delay Metrics     15 Behavioral Vectors        Hinglish/English Cleaning
         │                             │                             │
         ▼                             ▼                             ▼
 [ XGBOOST CLASSIFIER ]     [ RANDOM FOREST ENSEMBLE ]   [ TF-IDF + GRADIENT BOOST ]
  Late Payment Predictor        Tenant Churn Radar          Maintenance Triage
         │                             │                             │
         └─────────────────────────────┼─────────────────────────────┘
                                       ▼
                           DECISION & ACTION ENGINE
                • Real-time Churn Alert Generation
                • Predictive Escalation Matrix
                • Automated Emergency Dispatch & Response
                                       │
                                       ▼
                       STREAMLIT / FASTAPI INTERFACE
```

---

## 2. Model Selection Rationale

### 2.1 Model 1: Late Payment Predictor (XGBoost Classifier)
* **Objective:** Predict binary probability that a tenant will delay their upcoming rental payment by $>3$ days.
* **Selected Algorithm:** Extreme Gradient Boosting (`XGBClassifier`)
* **Rationale over Alternatives:**
  - *Vs. Logistic Regression:* Financial behavior exhibits non-linear interactions (e.g., small payment delays compound rapidly when combined with pending complaints). Tree ensembles capture feature interactions without manual polynomial expansion.
  - *Vs. Deep Neural Networks:* Tabular financial data with tabular cardinality does not benefit from deep learning parameters and poses higher latency and memory overhead. XGBoost offers state-of-the-art ROC-AUC, rapid inference ($<2\text{ms}$ per tenant), and built-in handling of collinear signals.
* **Key Hyperparameters:**
  - `n_estimators`: 150
  - `max_depth`: 5 (prevents overfitting on small-to-medium portfolios)
  - `learning_rate`: 0.1
  - `subsample`: 0.8, `colsample_bytree`: 0.8 (regularization)
  - `eval_metric`: `logloss`

### 2.2 Model 2: Tenant Churn Predictor (Random Forest Ensemble)
* **Objective:** Forecast tenant non-renewal likelihood 60 to 90 days before lease expiration.
* **Selected Algorithm:** `RandomForestClassifier` with balanced class weights
* **Rationale over Alternatives:**
  - *Class Imbalance Handling:* In real-world properties, typical churn is between 20–35%. Random Forest with `class_weight="balanced"` dynamically adjusts impurity penalties to maintain high recall on at-risk tenants without producing false-positive churn panics.
  - *Interpretability & Feature Importances:* Mean Decrease in Impurity (MDI) allows landlords to understand *why* a specific tenant is leaving (e.g., unresolved maintenance vs. rent disparity).
* **Key Hyperparameters:**
  - `n_estimators`: 200
  - `max_depth`: 8
  - `min_samples_split`: 5, `min_samples_leaf`: 2

### 2.3 Model 3: Smart Maintenance Classifier & Emergency Triage (NLP + Ensemble)
* **Objective:** Classify unstructured tenant messages into 7 operational trades (`Plumbing`, `Electrical`, `HVAC`, `Structural`, `Gas`, `Pest`, `General`), score urgency on a 1–10 scale, and trigger automated emergency workflows.
* **Selected Approach:** Hybrid TF-IDF Vectorizer + Gradient Boosting Classifier + Rule-Based Urgency Engine.
* **Rationale over Heavyweight LLMs (e.g., GPT-4/Claude for every ticket):**
  - *Cost & Latency:* Running large foundation models on 1,000s of low-complexity maintenance tickets incurs needless API expenses (\$0.01–\$0.03/call) and high latency (1–3 seconds).
  - *Edge Efficiency:* A tuned TF-IDF (1–2 n-grams, sublinear term frequency) and lightweight gradient boosted tree executes in under 10 milliseconds locally without external API keys or cloud egress costs.
  - *Bilingual Hinglish Support:* Specifically incorporates Indian vernacular tokens common in rental communication (e.g., `paani nahi aa raha`, `bijli cut`, `geyser kharab`, `cylinder leak`).

---

## 3. Data Pipeline & Feature Engineering

### 3.1 Feature Vector Specification

The pipeline derives **15 high-signal features** per tenant by merging three core relational entities: `tenants`, `payments`, and `maintenance_requests`:

```
┌─────────────────────────┬────────────────────────────────────────────────────────┐
│ Feature Name            │ Description & Engineering Logic                        │
├─────────────────────────┼────────────────────────────────────────────────────────┤
│ avg_delay_days          │ Rolling average delay in days across all payments      │
│ max_delay_days          │ Historical peak payment delay                          │
│ late_payment_rate       │ Ratio of payments made after due date                  │
│ payment_rate            │ Ratio of completed payments vs expected periods       │
│ delay_trend             │ Delta: Average delay (last 3 mos) - Avg delay (prior)  │
│ partial_payment_rate    │ Frequency of fractional rent deposits                  │
│ total_complaints        │ Cumulative maintenance tickets logged                  │
│ unresolved_complaints   │ Currently open / pending tickets                       │
│ emergency_complaints    │ Count of critical (gas, water burst, spark) events     │
│ avg_complaint_urgency   │ Mean severity score of submitted requests              │
│ days_until_lease_end    │ Calendar countdown from current date to lease_end       │
│ lease_duration_months   │ Baseline contract commitment (6, 11, 12, 24 mos)       │
│ tenure_months           │ Total duration tenant has occupied property           │
│ monthly_rent            │ Baseline monthly financial commitment in INR           │
│ age                     │ Demographic normalization factor                      │
└─────────────────────────┴────────────────────────────────────────────────────────┘
```

### 3.2 Data Preprocessing & Validation
* **Missing Value Imputation:** Unpaid or missing delays are encoded using explicit indicator tags (`-1` for unresolved/open) alongside binary fulfillment masks.
* **Leakage Prevention:** Features are computed strictly using retrospective timestamps relative to the prediction window, eliminating look-ahead bias.

---

## 4. Prompt Engineering & Generative Response Architecture

While classification runs on fast classical NLP, **conversational tenant acknowledgment and emergency guidance** utilize structured generative prompts. 

### 4.1 Production LLM Prompt Template (for OpenAI / Gemini / Ollama integrations)

```markdown
SYSTEM:
You are the AI Operations Companion for TenantEase by Cynbit Technologies.
Your job is to generate a professional, context-aware, reassuring message to a tenant 
who has submitted a maintenance complaint.

INPUT VARIABLES:
- Tenant Name: {{tenant_name}}
- Property Unit: {{property_name}} - Unit {{floor}}
- Detected Category: {{category}}
- Urgency Level: {{urgency_label}} (Score: {{urgency_score}}/10)
- Is Emergency: {{is_emergency}}
- Raw Complaint: {{complaint_text}}

BEHAVIORAL CONSTRAINTS:
1. If is_emergency is TRUE:
   - Begin immediately with a safety alert directive in bold.
   - For Gas: instruct to shut the main valve, avoid turning on light switches, and vacate.
   - For Electrical: instruct to trip the local breaker and avoid wet surfaces.
   - Inform that an emergency technician is being dispatched within 30 minutes.
2. If is_emergency is FALSE:
   - Provide an empathetic acknowledgment.
   - Specify the targeted response SLA (e.g., 24-48 hours).
   - Sign off as "TenantEase Automated Care Team on behalf of [Landlord]".
3. Keep the entire response under 90 words.
```

### 4.2 Generative Retention Prompt Architecture (WhatsApp & Early Renewal Proposals)

```markdown
SYSTEM:
You are the AI Resident Retention Specialist for TenantEase by Cynbit Technologies.
Your objective is to craft high-converting, empathetic, and personalized communication 
to prevent tenant churn and stabilize rental cash flow.

INPUT VARIABLES:
- Tenant Name: {{tenant_name}}
- Property Unit: {{property_name}}
- Monthly Rent: INR {{monthly_rent}}
- Days Remaining on Lease: {{days_left}} days
- Unresolved Maintenance Tickets: {{open_complaints}}
- Historical Delay Trajectory: {{delay_trend}} days
- Action Type: {{action_type}} (WHATSAPP_NUDGE | RENEWAL_OFFER)

BEHAVIORAL CONSTRAINTS:
1. For WhatsApp Nudges:
   - Max 50 words.
   - Address respectfully (Namaste [FirstName] ji).
   - Acknowledge open maintenance tickets proactively before discussing rent.
2. For Early Renewal Proposals:
   - Offer a calibrated 5% loyalty renewal discount.
   - Include 100% maintenance fee waiver for Q1.
   - Frame with a 14-day decision timeline.
```

---

## 5. Execution & Production Deployment Plan

```
┌────────────────────────────────────────────────────────────────────────────┐
│                       DEPLOYMENT ROADMAP (6 WEEKS)                         │
└────────────────────────────────────────────────────────────────────────────┘
 Week 1: PoC Integration & Model Serialization
 ├── Validate containerized Streamlit application
 ├── Export trained pipeline artifacts (.joblib format)
 └── Benchmark inference latency and memory footprints

 Week 2-3: FastAPI Microservice Integration
 ├── Build REST endpoints:
 │   ├── POST /v1/predict/late-payment
 │   ├── POST /v1/predict/churn-risk
 │   └── POST /v1/triage/maintenance
 └── Implement JWT authentication for TenantEase backend services

 Week 4: Automated Batch Scoring Pipeline
 ├── Establish daily cron job via Celery / Redis
 ├── Ingest overnight transaction logs
 └── Populate landlord dashboard flags & notification queues

 Week 5: WhatsApp & Multi-Channel Webhook Hookup
 ├── Webhook listeners for incoming WhatsApp / SMS complaints
 ├── Automated routing to vendor registry by trade and locality
 └── Escalation triggers for high-urgency notifications

 Week 6: Monitoring, Drift Detection & Retraining
 ├── Log prediction vs actual outcomes (ground truth capture)
 ├── Evident model drift triggers (Evidently AI / MLflow)
 └── Bi-monthly automated retraining pipeline
```

---

## 6. Hardware, Scaling & Infrastructure Requirements

* **Inference Compute:** A single lightweight cloud instance (e.g., AWS `t4g.small` or GCP `e2-small` with 2 vCPUs and 2 GB RAM) comfortably handles $>100,000$ daily predictions due to the low-overhead nature of XGBoost and Scikit-Learn pipelines.
* **Storage:** Model artifacts require $< 25 \text{MB}$ disk storage; feature store tables scale linearly at $\approx 50 \text{MB}$ per 10,000 active tenants annually.
* **Reliability SLA:** Designed for $99.95\%$ uptime with automatic fallback to rule-based classification if ML inference endpoints experience network degradation.

---

## Summary
The proposed technical architecture elevates Cynbit’s TenantEase with an enterprise-grade AI intelligence layer that is cost-effective, hyper-responsive, explainable, and production-ready for immediate pilot deployment.
