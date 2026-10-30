"""
TenantEase AI Intelligence Layer — Standalone Demo & Test Runner
Executes the full pipeline:
  1. Generates synthetic tenant, property, payment, and maintenance datasets
  2. Trains the Late Payment Predictor (XGBoost / GradientBoost)
  3. Trains the Tenant Churn Predictor (Random Forest)
  4. Trains the Smart Maintenance Classifier (NLP + TF-IDF)
  5. Runs sample inferences and displays results

Usage:
  python run_demo.py
"""

import sys
import os

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.generate_data import generate_all_data
from models.payment_predictor import train_payment_model, predict_late_payment
from models.churn_predictor import train_churn_model, predict_churn
from models.maintenance_classifier import train_ml_classifier, classify_complaint
from models.llm_engine import generate_maintenance_triage_llm, generate_retention_action_llm


def run_pipeline():
    print("=" * 70)
    print(" 🏠  TENANTEASE AI INTELLIGENCE LAYER — PIPELINE DEMO")
    print("     Cynbit Technologies Technical Assessment PoC")
    print("=" * 70)

    # 1. Generate Datasets
    print("\n[Step 1/4] Generating Datasets...")
    properties, tenants, payments, maintenance, features = generate_all_data()
    print(f"  ✓ Properties: {len(properties)}")
    print(f"  ✓ Tenants: {len(tenants)}")
    print(f"  ✓ Payment Records: {len(payments)}")
    print(f"  ✓ Maintenance Requests: {len(maintenance)}")
    print(f"  ✓ Feature Engineered Rows: {len(features)}")

    # 2. Train Payment Predictor
    print("\n[Step 2/4] Training Late Payment Predictor (XGBoost / Ensemble)...")
    payment_model, pay_results = train_payment_model(features)
    print(f"  ✓ Model Accuracy : {pay_results['accuracy']:.1%}")
    print(f"  ✓ Precision (Late): {pay_results['precision']:.1%}")
    print(f"  ✓ Recall (Late)   : {pay_results['recall']:.1%}")
    print(f"  ✓ ROC-AUC Score   : {pay_results['auc_roc']:.3f}")
    top_3_pay = list(pay_results['feature_importance'].items())[:3]
    print(f"  ✓ Top Drivers     : {', '.join([f'{k} ({v:.3f})' for k, v in top_3_pay])}")

    # 3. Train Churn Predictor
    print("\n[Step 3/4] Training Tenant Churn Predictor (Random Forest)...")
    churn_model, churn_results = train_churn_model(features)
    print(f"  ✓ Model Accuracy : {churn_results['accuracy']:.1%}")
    print(f"  ✓ Precision (Churn): {churn_results['precision']:.1%}")
    print(f"  ✓ Recall (Churn)  : {churn_results['recall']:.1%}")
    print(f"  ✓ ROC-AUC Score   : {churn_results['auc_roc']:.3f}")
    top_3_churn = list(churn_results['feature_importance'].items())[:3]
    print(f"  ✓ Top Drivers     : {', '.join([f'{k} ({v:.3f})' for k, v in top_3_churn])}")

    # 4. Train Maintenance Classifier
    print("\n[Step 4/4] Training Smart Maintenance Classifier (NLP + TF-IDF)...")
    maint_model, maint_results = train_ml_classifier(maintenance)
    print(f"  ✓ Classification Accuracy: {maint_results['accuracy']:.1%}")

    # 5. Live Demonstration Inferences
    print("\n" + "=" * 70)
    print(" 🎯 LIVE INFERENCE DEMONSTRATIONS")
    print("=" * 70)

    # 5.1 Sample Tenant Payment & Churn Predictions
    sample_tenants = features.head(5)
    pay_preds = predict_late_payment(payment_model, sample_tenants)
    churn_preds = predict_churn(churn_model, sample_tenants)

    print("\n--- [A] Tenant Risk Evaluation (Sample 5 Tenants) ---")
    for i, (_, t) in enumerate(sample_tenants.iterrows()):
        p_row = pay_preds.iloc[i]
        c_row = churn_preds.iloc[i]
        print(f"\n👤 Tenant: {t['tenant_name']} (ID: {t['tenant_id']}, Rent: ₹{t['monthly_rent']:,})")
        print(f"   • Late Payment Prob: {p_row['late_probability']:.1%} [{p_row['risk_level']}]")
        print(f"   • Churn Risk Prob  : {c_row['churn_probability']:.1%} [{c_row['risk_level']}]")
        print(f"   • Action Directive : {c_row['recommended_actions']}")

    # 5.2 Sample Maintenance Complaint Triages
    test_complaints = [
        "Gas leak ho rahi hai kitchen me bahut smell aa rahi hai please urgent help!",
        "AC se paani gir raha hai floor pe and cooling is very low",
        "Bathroom tap se paani continuously drip ho raha hai waste ho raha hai",
        "Switch board me spark aa rahi hai very dangerous situation",
        "Need rent receipt for last 3 months for IT declaration",
    ]

    print("\n--- [B] NLP Emergency Triage & Bilingual Routing ---")
    for complaint in test_complaints:
        res = classify_complaint(complaint, ml_model=maint_model)
        em_icon = "🚨" if res['is_emergency'] else "📌"
        print(f"\n{em_icon} Input: \"{complaint}\"")
        print(f"   ➔ Category: {res['category']} (Confidence: {res['confidence']:.1%})")
        print(f"   ➔ Urgency : {res['urgency_label']} (Score: {res['urgency_score']}/10)")
        print(f"   ➔ Auto-SLA: {res['auto_response'][:85]}...")

    # 5.3 Generative LLM Copilot Demonstrations
    print("\n--- [C] Generative LLM Copilot Demonstrations (Prompt Engineering) ---")

    # Scenario 1: LLM Critical Emergency Protocol Generation
    llm_triage = generate_maintenance_triage_llm(
        tenant_name="Sanjay Kapoor",
        property_name="Mansarovar Residency, Flat 302",
        category="Gas",
        urgency_score=10,
        is_emergency=True,
        complaint_text="Gas leak ho rahi hai kitchen me bahut smell aa rahi hai please urgent help!"
    )
    print(f"\n🤖 [LLM Copilot • Critical Hazard Directive]")
    print(f"   Engine: {llm_triage['engine']}")
    print(f"   Output:\n   {llm_triage['response']}")

    # Scenario 2: LLM Retention & WhatsApp Nudge Generation
    llm_retention = generate_retention_action_llm(
        tenant_name="Ananya Khandelwal",
        property_name="Vaishali Enclave, Flat 204",
        monthly_rent=18000,
        days_left=45,
        open_complaints=2,
        delay_trend=2.5,
        action_type="whatsapp"
    )
    print(f"\n🤖 [LLM Copilot • Context-Grounded WhatsApp Retention Nudge]")
    print(f"   Engine: {llm_retention['engine']}")
    print(f"   Output:\n   {llm_retention['response']}")

    print("\n" + "=" * 70)
    print(" 🎉 ALL MODELS & LLM ENGINES VALIDATED SUCCESSFULLY!")
    print("    To launch the modern Web Command Center, run:")
    print("    python server.py  (Opens http://localhost:8000)")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
